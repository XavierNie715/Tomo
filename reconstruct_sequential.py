import argparse
import os
import h5py
import numpy as np
import scipy.optimize
import scipy.sparse
from scipy import ndimage

from utilities import funcs


def forward_phi(_step, _phi_init, _forward_model):
    """
    Forward the model by _steps
    :param _step: Number of steps to forward the model
    :param _phi_init: Initial state of the model (vectorized)
    :param _forward_model: The forward model to be used
    :return: list of frames after each step of forwarding the model
    """
    resolution = int(np.sqrt(_phi_init.shape[0]))

    _phi_frame = [_phi_init]
    for i_step in range(_step):
        phi_temp = ndimage.zoom(_phi_init.reshape(resolution, resolution),
                                zoom=2,
                                mode='wrap').ravel()  # 3-order spline interpolate
        phi_temp = _forward_model @ phi_temp
        phi_temp = ndimage.zoom(phi_temp.reshape(resolution * 2, resolution * 2),
                                zoom=1 / 2,
                                mode='wrap').ravel()  # down-sampling
        _phi_frame.append(phi_temp)
    return _phi_frame


def weight_dot_forward(_steps, _weight, _forward_model, _circle_process=None):
    """
    Calculate matrix of W @ M
    :param _steps: Number of steps to forward the model
    :param _weight: Projection weight matrix
    :param _forward_model: The forward model to be used
    :param _circle_process: Function to mask out the circle region (optional)
    :return: matrix of W @ M, shape = ((STEP + 1) * num_theta * num_theta, n**2)
    """

    if _circle_process is not None:
        # mask out the circle region
        _weight = _circle_process.data_in_circle(weight)
    else:
        pass

    forward_matrix_temp = scipy.sparse.identity(_forward_model.shape[0], dtype=np.float64)
    _weight_forward_matrix = []  # list of W @ M
    for _i in range(_steps + 1):  # list of forward matrix
        _weight_forward_matrix.append(_weight @ forward_matrix_temp)
        forward_matrix_temp = _forward_model @ forward_matrix_temp  # \phi^n

    frame_index = np.linspace(100, _steps, 10)
    selected_weight_forward_matrix = [_weight_forward_matrix[i] for i in frame_index]
    _weight_forward_matrix = scipy.sparse.vstack(selected_weight_forward_matrix).reshape(-1, _forward_model.shape[0])
    return _weight_forward_matrix.tocsr()


def opt_func(_proj, _weight_forward_matrix, _lambda_re, _circle_process=None):
    """
    The objective function of the reconstruction problem
    ** The regularization term is the 2nd order derivative of phi,
    calculated by correlate with Laplacian_kernel = np.array([[1, 1, 1],
                                                              [1, -8, 1],
                                                              [1, 1, 1]])
    :param _proj: projection data, shape = (num_theta * num_theta * _steps,),
    :param _weight_forward_matrix: shape = ((steps + 1) * num_theta * num_theta, n**2)
    :param _lambda_re: regularization parameter,
    :param _circle_process: mask for region out of the circle
    :return: lambda function, input shape = (n**2,), return shape = (1,): residual.
    """
    return lambda x: (np.linalg.norm(_weight_forward_matrix @ x - _proj) ** 2
                      + _lambda_re * np.linalg.norm(funcs.laplacian_times_x(x, _circle_process)) ** 2)


def opt_jac(_proj, _weight_forward_matrix, _laplacian_matrix, _lambda_re):
    """
    The analytical Jacobian of the objective function
    :param _proj: projection data, shape = (num_theta * num_theta * _steps,),
    :param _weight_forward_matrix:
    :param _laplacian_matrix: Laplacian matrix for regularization
    :param _lambda_re: regularization parameter
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
    DIFFUSION_COEFFICIENT = 1 / 3000
    ROOT = '/Users/xiangyu.nie/PycharmProjects/Tomo'
    # ROOT = '/public/home/lcc-dx07/Tomo'

    data_root = os.path.join(ROOT + '/data_cache/ScalarData-20231116',
                             f'{num_theta}_angle_{num_proj}_proj')
    u = scipy.io.loadmat(ROOT + '/data_cache/ScalarData-20231116/velocity/U_256x256.mat')['U']
    v = scipy.io.loadmat(ROOT + '/data_cache/ScalarData-20231116/velocity/V_256x256.mat')['V']
    u_pixel = u / DOMAIN_LENGTH * RESOLUTION
    v_pixel = v / DOMAIN_LENGTH * RESOLUTION

    # load phi and proj
    phi_256 = []
    proj_high_reso = []
    phi_proj_path = os.path.join(data_root, 'proj_and_phi')
    h5_files = [file for file in os.listdir(phi_proj_path) if (file.endswith(".h5"))]
    h5_files.sort(key=lambda x: int(x.split('Frame')[-1].split('_')[0]), reverse=False)
    for _, h5_file in enumerate(h5_files):
        with h5py.File(os.path.join(phi_proj_path, h5_file), 'r') as saved_h5:
            phi, proj = [saved_h5[key][:].astype(dtype=np.float64) for key in
                         saved_h5.keys()]  # order: phi, proj, weight
            phi_256.append(phi)
            proj_high_reso.append(proj)
    for index, phi_temp in enumerate(phi_256):
        phi_256[index] = ndimage.zoom(phi_temp, 2, mode='wrap')

    # phi_256 = np.vstack(phi_256)  # shape = ((steps + 1), n**2)
    # weight_256 = np.vstack(weight_256)  # shape = ((steps + 1) * num_theta * num_theta, n**2)

    # load sparse weight
    weight_path = os.path.join(ROOT + '/data_cache', 'weight')
    weight_file = os.path.join(weight_path, f'weight_sparse_{num_theta}_{num_proj}_r{512}.npz')
    weight = scipy.sparse.load_npz(weight_file)

    # proj = np.vstack(proj_high_reso)  # shape = ((steps + 1) * num_theta * num_theta,)

    if RESOLUTION == 256:
        u_pixel_512 = ndimage.zoom(u_pixel, 2, mode='wrap')  # 3-order spline interpolate to 512 resolution
        v_pixel_512 = ndimage.zoom(v_pixel, 2, mode='wrap')
        forward_model = funcs.rk4_forward_mat(u_pixel_512.ravel(), v_pixel_512.ravel(), DIFFUSION_COEFFICIENT,
                                              delta_t=0.0001)
    else:
        forward_model = funcs.rk4_forward_mat(u_pixel.ravel(), v_pixel.ravel(),
                                              DIFFUSION_COEFFICIENT,
                                              delta_t=0.0001)

    # mask out the circle region
    circle_process = funcs.Mask_Circle(resolution=512, radius=120 / 256 * 512)
    forward_model_circle = forward_model[:, ~circle_process()][~circle_process(), :]
    weight_forward_matrix = weight_dot_forward(STEP * 100, weight, forward_model_circle, circle_process)
    laplacian_matrix = funcs.laplacian_matrix_2nd(512)[:, ~circle_process()][~circle_process(), :]
    # for i, weight_temp in enumerate(weight_256):
    #     weight_256[i] = circle_process.data_in_circle(weight_temp)
    # for i, phi_temp in enumerate(phi_frame):
    #     phi_frame[i] = circle_process.data_in_circle(phi_temp)

    lambda_rec = args.lambda_rec
    if args.init_path:
        x_int = np.load(os.path.join(args.init_path, 'result_0.npy'))
        x_int = circle_process.data_in_circle(x_int)
    else:
        x_int = np.random.rand(weight_forward_matrix.shape[1])

    # L-BFGS-B
    obj_fun = opt_func(np.vstack(proj_high_reso).ravel(), weight_forward_matrix, lambda_rec, circle_process())
    jac_fun = opt_jac(np.vstack(proj_high_reso).ravel(), weight_forward_matrix, laplacian_matrix, lambda_rec)
    x_rec = scipy.optimize.minimize(fun=obj_fun,
                                    x0=x_int, method='L-BFGS-B',
                                    jac=jac_fun,
                                    bounds=scipy.optimize.Bounds(0, 1),
                                    options={'disp': True}, )

    print(x_rec)
    phi_full = circle_process.recover_data(x_rec.x)
    np.save(os.path.join(args.save_path, 'result.npy'), phi_full)

    print(f'(||W @ phi_rec - alpha||_2) = {np.linalg.norm(weight @ x_rec.x - proj_high_reso[0])}')
    print(f'(||W @ phi_256 - alpha||_2) = '
          f'{np.linalg.norm(weight @ circle_process.data_in_circle(phi_256[0]) - proj_high_reso[0])}')
    print(f'(||L @ phi_rec||_2) = '
          f'{lambda_rec * np.linalg.norm(funcs.laplacian_matrix_2nd(RESOLUTION) @ phi_full)}')
    print(f'(||L @ phi_256||_2) = '
          f'{np.linalg.norm(funcs.laplacian_matrix_2nd(RESOLUTION) @ phi_256[0])}')

    phi_frame_rec = forward_phi(STEP * 100, phi_full, forward_model)
    frame_index = np.linspace(100, STEP * 100, 10)
    phi_frame_rec = [phi_frame_rec[i] for i in frame_index]

    # plot and save txt
    init_frame_index = int(4900)
    with open(os.path.join(args.save_path, 'result_phi.txt'), 'w') as f:
        for i, phi in enumerate(phi_frame_rec, 0):
            funcs.plot_opt_result(phi_frame_rec[i], phi_256[i], args.save_path,
                                  f'result_phi_{init_frame_index}.png', resolution=512)
            f.write(f'\nFrame {init_frame_index}\tW @ phi_rec - alpha\tW @ phi_gt - alpha\tL @ phi_rec\tL @ phi_gt\n')
            f.write(f'2-norm'
                    f'\t{np.linalg.norm(weight @ circle_process.data_in_circle(phi_frame_rec[i]) - proj_high_reso[i]):.4e}'
                    f'\t{np.linalg.norm(weight @ circle_process.data_in_circle(phi_256[i]) - proj_high_reso[i]):.4e}'
                    f'\t{np.linalg.norm(funcs.laplacian_matrix_2nd(RESOLUTION) @ phi_frame_rec[i]):.4e}'
                    f'\t{np.linalg.norm(funcs.laplacian_matrix_2nd(RESOLUTION) @ phi_256[i]):.4e}\n')
            f.write('\n')
            init_frame_index += 10
