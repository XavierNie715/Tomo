import argparse
import os
import h5py
import numpy as np
import scipy.optimize

from utilities import funcs


def forward_phi(_step, _forward_model):
    """
    Forward the model by _steps
    :param _step: Number of steps to forward the model

    :param _forward_model: The forward model to be used
    :return: list of frames after each step of forwarding the model
    """

    def forward_phi_matrix(_phi_init):
        """

        :param _phi_init: Initial state of the model
        :return:
        """
        _phi_frame = [_phi_init]
        _phi_temp = _phi_init
        for i_step in range(_step):
            _phi_temp = circle_process.recover_data(_phi_temp)
            _phi_temp = _forward_model(_phi_temp)
            _phi_temp = circle_process.data_in_circle(_phi_temp)
            _phi_frame.append(_phi_temp)
        return _phi_frame

    return forward_phi_matrix


def weight_dot_forward(_steps, _weight_list, _forward_model, _circle_process=None):
    """
    Calculate matrix of W @ \phi^n
    :param _steps: Number of steps to forward the model
    :param _weight_list: List of projection weight matrices
    :param _forward_model: The forward model to be used
    :param _circle_process: Function to mask out the circle region (optional)
    :return: matrix of W @ \phi^n, shape = ((steps + 1) * num_theta * num_theta, n**2)
    """
    _resolution = 256
    if _circle_process is not None:
        # mask out the circle region
        for _i, weight_temp in enumerate(_weight_list):
            _weight_list[_i] = _circle_process.data_in_circle(weight_temp)
    else:
        pass

    forward_matrix_temp = scipy.sparse.identity(_forward_model.shape[0], dtype=np.float64)
    _weight_forward_matrix = []  # list of W @ \phi^n
    for _i in range(_steps + 1):  # list of forward matrix
        _weight_forward_matrix.append(_weight_list[_i] @ forward_matrix_temp)
        forward_matrix_temp = _forward_model @ forward_matrix_temp  # \phi^n
    _weight_forward_matrix = scipy.sparse.vstack(_weight_forward_matrix).reshape(-1, _forward_model.shape[0])
    return _weight_forward_matrix.tocsr()


def opt_func(_weight_list, _proj, _cal_forward_model, _steps, _lambda_re, _circle_process=None):
    """
    The objective function of the reconstruction problem
    ** The regularization term is the 2nd order derivative of phi,
    calculated by correlate with Laplacian_kernel = np.array([[1, 1, 1],
                                                              [1, -8, 1],
                                                              [1, 1, 1]])
    :param _weight_list: list of projection weight matrix, shape of each element = (num_theta * num_proj, n**2),
    :param _proj: projection data, shape = (_steps+1 * num_theta * num_proj ),
    :param _cal_forward_model: func to calculate phi frames with input initial phi, return (steps+1, resolution**2)
    :param _lambda_re: regularization parameter,
    :param _circle_process: mask for region out of the circle
    :return: lambda function, input shape = (n**2,), return shape = (1,): residual.
    """

    def cal_opt_func(_phi_opt):
        phi_frame = _cal_forward_model(_phi_opt)
        proj_opt = []
        for _step in range(_steps + 1):
            proj_opt.append(_weight_list[_step] @ phi_frame[_step])
        proj_opt = np.vstack(proj_opt).ravel()  # shape = (steps+1 * num_theta * num_proj)

        obs_term = np.linalg.norm(proj_opt - _proj) ** 2
        TV_term = _lambda_re * np.linalg.norm(funcs.laplacian_times_x(_phi_opt, _circle_process)[~_circle_process]) ** 2

        return obs_term + TV_term

    return cal_opt_func


def opt_jacob(_weight_list, _proj, _forward_model, _weight_forward_matrix, _laplacian_matrix, _lambda_re):
    """
    The analytical Jacobian of the objective function
    :param _weight_list: list of projection weight matrix, shape = (num_theta * num_theta * _steps, n**2),
    :param _proj: projection data, shape = (num_theta * num_theta * _steps,),
    :param _forward_model: forward model matrix, shape = (n**2, n**2)
    :param _weight_forward_matrix:
    :param _laplacian_matrix: Laplacian matrix for regularization
    :param _lambda_re: regularization parameter
    :param _circle_process: mask out the circle region
    :return: lambda x, shape = (n**2,).
    """

    return lambda x: (2 * _weight_forward_matrix.T @ (_weight_forward_matrix @ x - _proj)
                      + 2 * _lambda_re * _laplacian_matrix.T @ _laplacian_matrix @ x)


def get_args():
    """
    Parse command line arguments
    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Reconstruction with Tikhonov regularization')
    parser.add_argument('--lambda_rec', '-l', type=float, help='Coefficient of Tikhonov regularization')
    parser.add_argument('--init_path', '-i', required=False, type=str, default=None,
                        help='path of x0 for initialization')
    parser.add_argument('--save_path', '-s', type=str, default=False, help='save path')
    parser.add_argument('--num_theta', '-t', type=str, default='16', help='num of theta')
    parser.add_argument('--num_proj', '-p', type=str, default='16', help='num of angle')
    return parser.parse_args()


if __name__ == '__main__':
    """
    Main function
    """
    args = get_args()
    np.random.seed(42)

    num_theta = int(args.num_theta)
    num_proj = int(args.num_proj)

    RESOLUTION = 256
    DOMAIN_LENGTH = 2 * np.pi
    STEP = 10
    DELTA_t = 0.01
    DEFUSSION_COEFFICENT = 1 / 3000

    data_root = os.path.join('/public/home/lcc-dx07/Tomo/data_cache/ScalarData-20231116',
                             f'{num_theta}_angle_{num_proj}_proj')
    u = scipy.io.loadmat('/public/home/lcc-dx07/Tomo/ScalarData-20231116/velocity/U_256x256.mat')['U']
    v = scipy.io.loadmat('/public/home/lcc-dx07/Tomo/ScalarData-20231116/velocity/V_256x256.mat')['V']
    u_pixel = u / DOMAIN_LENGTH * RESOLUTION
    v_pixel = v / DOMAIN_LENGTH * RESOLUTION

    # load phi and proj
    phi_256 = []
    proj_high_reso = []
    phi_proj_path = os.path.join(data_root, 'proj_and_phi')
    h5_files = [file for file in os.listdir(phi_proj_path) if (file.endswith(".h5"))]
    h5_files.sort(key=lambda x: int(x.split('Frame')[-1].split('_')[0]), reverse=False)
    for _, h5_file in enumerate(h5_files):
        with h5py.File(os.path.join(phi_proj_path, h5_file), 'a') as saved_h5:
            phi, proj = [saved_h5[key][:].astype(dtype=np.float64) for key in
                         saved_h5.keys()]  # order: phi, proj, weight
            phi_256.append(phi)
            proj_high_reso.append(proj)

    # phi_256 = np.vstack(phi_256)  # shape = ((steps + 1), n**2)
    # weight_256 = np.vstack(weight_256)  # shape = ((steps + 1) * num_theta * num_theta, n**2)

    # load sparse weight
    weight_256 = []
    weight_path = os.path.join(data_root, 'weight_sparse')
    weight_files = [file for file in os.listdir(weight_path) if (file.endswith(".npz"))]
    weight_files.sort(key=lambda x: int(x.split('Frame')[-1].split('_')[0]), reverse=False)
    for _, npz_file in enumerate(weight_files):
        weight = scipy.sparse.load_npz(os.path.join(weight_path, npz_file))
        weight_256.append(weight)
    # proj = np.vstack(proj_high_reso)  # shape = ((steps + 1) * num_theta * num_theta,)

    # rk4_forward_mat = funcs.rk4_forward_mat(u.ravel(), v.ravel(), Pe=3000)
    cal_dphi_dt = funcs.dphi_dt(DEFUSSION_COEFFICENT, u_pixel, v_pixel, vectorize=False)
    forward_model = funcs.forward_upwind(func_dphi_dt=cal_dphi_dt, delta_t=DELTA_t)
    cal_forward_phi = forward_phi(STEP, forward_model)

    # mask out the circle region
    circle_process = funcs.Mask_Circle(RESOLUTION, radius=120)
    laplacian_matrix = funcs.laplacian_matrix_2nd(RESOLUTION)[:, ~circle_process()][~circle_process(), :]
    for i, weight_temp in enumerate(weight_256):
        weight_256[i] = circle_process.data_in_circle(weight_temp)
    for i, phi_temp in enumerate(phi_256):
        phi_256[i] = circle_process.data_in_circle(phi_temp)

    lambda_rec = args.lambda_rec
    if args.init_path:
        x_int = np.load(os.path.join(args.init_path, 'result_0.npy'))
        x_int = circle_process.data_in_circle(x_int)
    else:
        x_int = np.random.rand(laplacian_matrix.shape[0])

    # L-BFGS-B
    obj_fun = opt_func(weight_256, np.vstack(proj_high_reso).ravel(), cal_forward_phi, lambda_rec, circle_process())
    # jac_fun = opt_jacob(weight_256, np.vstack(proj_high_reso).ravel(), rk4_forward_mat,
    #                     weight_forward_matrix, laplacian_matrix, lambda_rec)
    x_rec = scipy.optimize.minimize(fun=obj_fun,
                                    x0=x_int, method='L-BFGS-B',
                                    jac=None,
                                    bounds=scipy.optimize.Bounds(0, 1),
                                    options={'disp': True}, )

    print(x_rec)
    phi_full = circle_process.recover_data(x_rec.x)
    np.save(os.path.join(args.save_path, 'result.npy'), phi_full)

    print(f'(||W @ phi_rec - alpha||_2) = {np.linalg.norm(weight_256[0] @ x_rec.x - proj_high_reso[0])}')
    print(f'(||W @ phi_256 - alpha||_2) = '
          f'{np.linalg.norm(weight_256[0] @ circle_process.data_in_circle(phi_256[0]) - proj_high_reso[0])}')
    print(f'(||L @ phi_rec||_2) = '
          f'{lambda_rec * np.linalg.norm(funcs.laplacian_matrix_2nd(RESOLUTION) @ phi_full)}')
    print(f'(||L @ phi_256||_2) = '
          f'{np.linalg.norm(funcs.laplacian_matrix_2nd(RESOLUTION) @ phi_256[0])}')

    phi_frame_rec = [phi_full]
    for i_step in range(STEP):
        phi_temp = forward_model(phi_full)
        phi_frame_rec.append(phi_temp)

    # plot and save txt
    init_frame_index = int(4900)
    with open(os.path.join(args.save_path, 'result_phi.txt'), 'w') as f:
        for i, phi in enumerate(phi_frame_rec, 0):
            funcs.plot_opt_result(phi_frame_rec[i], phi_256[i], args.save_path, f'result_phi_{init_frame_index}.png')
            f.write(f'\nFrame {init_frame_index}\tW @ phi_rec - alpha\tW @ phi_gt - alpha\tL @ phi_rec\tL @ phi_gt\n')
            f.write(f'2-norm'
                    f'\t{np.linalg.norm(weight_256[i] @ circle_process.data_in_circle(phi_frame_rec[i]) - proj_high_reso[i]):.4e}'
                    f'\t{np.linalg.norm(weight_256[i] @ circle_process.data_in_circle(phi_256[i]) - proj_high_reso[i]):.4e}'
                    f'\t{np.linalg.norm(funcs.laplacian_matrix_2nd(RESOLUTION) @ phi_frame_rec[i]):.4e}'
                    f'\t{np.linalg.norm(funcs.laplacian_matrix_2nd(RESOLUTION) @ phi_256[i]):.4e}\n')
            f.write('\n')
            init_frame_index += 10
