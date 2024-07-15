import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy
from scipy import ndimage
from utilities import funcs


def down_sample(_resolution, mode='wrap'):
    def zoom(_data):
        return ndimage.zoom(_data, _resolution / 512, mode=mode)

    return zoom


def partial_phi_partial_t(frame_i, delta_t, root_path, _circle_process=None):
    """
    Calculate the partial derivative of phi with respect to t (central difference).
    :param frame_i: The frame.
    :param delta_t: The time step.
    :param root_path: The root path of the data.
    :return: The partial derivative of phi with respect to t.
    """

    time = frame_i['file'].split('_')[1].split('.')[-1]
    time_minus_1 = int(frame_i['file'].split('_')[1].split('.')[-1]) - 1
    time_plus_1 = int(frame_i['file'].split('_')[1].split('.')[-1]) + 1

    phi_i_minus_1_path = frame_i['file'].replace(time, str(time_minus_1))
    phi_i_plus_1_path = frame_i['file'].replace(time, str(time_plus_1))

    phi_i_minus_1 = scipy.io.loadmat(os.path.join(root_path, phi_i_minus_1_path))['Phi']
    phi_i_plus_1 = scipy.io.loadmat(os.path.join(root_path, phi_i_plus_1_path))['Phi']

    if _circle_process:
        return circle_process.data_in_circle((phi_i_plus_1 - phi_i_minus_1) / (2 * delta_t))

    else:
        # return (phi_i_plus_1[256:768, 256:768] - frame_i['phi'][256:768, 256:768]) / (delta_t)
        # return (phi_i_plus_1[256:768, 256:768] - phi_i_minus_1[256:768, 256:768]) / (2 * delta_t)
        return down_size((phi_i_plus_1[256:768, 256:768] - phi_i_minus_1[256:768, 256:768]) / (2 * delta_t))
        # return (phi_i_plus_1[320:704, 320:704] - phi_i_minus_1[320:704, 320:704]) / (2 * delta_t)
        # return (phi_i_plus_1 - phi_i_minus_1) / (2 * delta_t)


def interpolate_piv_data(data, new_resolution, method='linear', _circle_process=None):
    original_resolution = data.shape[0]
    x_grid, y_grid = np.meshgrid(np.arange(original_resolution), np.arange(original_resolution))
    x_new_grid, y_new_grid = np.meshgrid(np.arange(new_resolution), np.arange(new_resolution))
    new_data = scipy.interpolate.griddata(np.column_stack((x_grid.ravel(), y_grid.ravel())),
                                          data.ravel(),
                                          (x_new_grid, y_new_grid), method=method)
    if _circle_process:
        return circle_process.data_in_circle(new_data.ravel())
    else:
        return new_data.ravel()


def add_noise(clean_data, intensity=0.05):
    """

    :param clean_data:
    :param intensity: 噪声大小百分比
    :return:
    """
    return clean_data + np.random.normal(0, np.abs(intensity * clean_data))


def opt_func(_A, _b, _lambda_re, _A_tik):
    return lambda x: (np.linalg.norm(_A @ x - _b) ** 2 + _lambda_re * np.linalg.norm(_A_tik @ x) ** 2)


def opt_jac(_A, _b, _lambda_re, _A_tik):
    return lambda x: (2 * _A.T @ (_A @ x - _b) + 2 * _lambda_re * _A_tik.T @ _A_tik @ x)


if __name__ == '__main__':
    DELTA_T = 1e-4
    RESOLUTION = 512
    DOMAIN_LENGTH = np.pi  # unit: [cm]
    DIFFUSION_COEFFICIENT = 0.15  # unit: [cm^2 / s]
    DELTA_x = DOMAIN_LENGTH / RESOLUTION

    # kappa = 1e-3  # regularization coefficient

    # noise_intensity = 0.05
    noise_intensity = None
    down_size = down_sample(RESOLUTION, mode='mirror')

    ROOT = '/Users/xiangyu.nie/Document/PhD/Hele-Shaw Example'

    Dx = (funcs.differential_matrix_x(1024, difference_form='order-2') / DELTA_x)  # [cm^-1]
    Dy = (funcs.differential_matrix_y(1024, difference_form='order-2') / DELTA_x)
    L = (funcs.laplacian_matrix_2nd(1024, difference_form='order-2') / DELTA_x ** 2)
    Dx_reso = (funcs.differential_matrix_x(RESOLUTION, difference_form='order-2') / DELTA_x)
    Dy_reso = (funcs.differential_matrix_y(RESOLUTION, difference_form='order-2') / DELTA_x)
    L_reso = (funcs.laplacian_matrix_2nd(RESOLUTION, difference_form='order-2') / DELTA_x ** 2)
    diffusion_term = lambda phi: down_size((DIFFUSION_COEFFICIENT * L @ phi).reshape(1024, 1024)
                                           [256:768, 256:768]).ravel()
    hstack_gradient_phi = lambda phi: scipy.sparse.hstack(
        ((scipy.sparse.diags(down_size((Dx @ phi).reshape(1024, 1024)[256:768, 256:768]).ravel())),
         scipy.sparse.diags(down_size((Dy @ phi).reshape(1024, 1024)[256:768, 256:768]).ravel())))

    phi = []
    partial_phi_partial_t_list = []
    A_func = []
    b_func = []
    frame_start = 1250
    frame_num = 1
    for frame_index in range(frame_num):
        frame_path = f'Frame_0.{frame_start + frame_index}_S.mat'
        frame_temp = {'file': frame_path,
                      'phi': scipy.io.loadmat(os.path.join(ROOT, frame_path))['Phi']}
        # 'phi': scipy.io.loadmat(os.path.join(ROOT, frame_path))['Phi'][384:512, 384:512]}
        phi_temp = (add_noise(frame_temp['phi'].ravel(), noise_intensity) if noise_intensity
                    else frame_temp['phi'].ravel())
        partial_phi_partial_t_temp = partial_phi_partial_t(frame_temp, DELTA_T, ROOT).ravel()
        phi.append(phi_temp)
        partial_phi_partial_t_list.append(partial_phi_partial_t_temp)
        A_func.append(hstack_gradient_phi(phi_temp))
        b_func.append(diffusion_term(phi_temp) - partial_phi_partial_t_temp)

    # construct Ax = bw
    A_func = scipy.sparse.vstack(A_func)
    A_div = scipy.sparse.hstack((Dx_reso, Dy_reso))
    A_curl = scipy.sparse.hstack((-Dy_reso, Dx_reso))
    A_smooth_u = scipy.sparse.hstack((kappa * L_reso, scipy.sparse.csr_array((RESOLUTION ** 2, RESOLUTION ** 2))))
    A_smooth_v = scipy.sparse.hstack((scipy.sparse.csr_array((RESOLUTION ** 2, RESOLUTION ** 2)), kappa * L_reso))
    A_smooth = scipy.sparse.vstack((A_smooth_u, A_smooth_v))
    A = scipy.sparse.vstack((A_func, A_div, A_curl, A_smooth))

    b_func = np.concatenate(b_func, axis=0)
    b_div = np.zeros(RESOLUTION ** 2)
    b_curl = np.zeros(RESOLUTION ** 2)d
    b_smooth = np.zeros(2 * RESOLUTION ** 2)
    b = np.concatenate((b_func, b_div, b_curl, b_smooth), axis=0)

    uv = scipy.io.loadmat('/Users/xiangyu.nie/PycharmProjects/Tomo/VField.mat')
    u = uv['U'][256:768, 256:768].ravel()
    v = uv['V'][256:768, 256:768].ravel()
    uv = np.concatenate((u, v), axis=0)

    # lsqr_result = scipy.sparse.linalg.lsqr(A, b, show=True)

    # scipy.io.savemat('./data_cache/for_matlab/decomposition_1e-3_alldiff.mat', {'A': A, 'b': b})

    uv_rec = scipy.io.loadmat()['uv']
    u_rec = uv_rec[:RESOLUTION ** 2]
    v_rec = uv_rec[RESOLUTION ** 2:]

    plt.imshow(u_rec.reshape(RESOLUTION, RESOLUTION), cmap=cm.jet)
    plt.colorbar()
    plt.clim(-15, 0)
    plt.show()

    plt.imshow(v_rec.reshape(RESOLUTION, RESOLUTION), cmap=cm.jet)
    plt.colorbar()
    plt.clim(0, 20)
    plt.show()

    # np.save('lsqr_result.npy', lsqr_result[0])

    # x_int = lsqr_result[0]
    # obj_fun = opt_func(A, b, kappa, A_regularization)
    # jac_fun = opt_jac(A, b, kappa, A_regularization)
    # x_rec = scipy.optimize.minimize(fun=obj_fun,
    #                                 x0=x_int, method='L-BFGS-B',
    #                                 jac=jac_fun,
    #                                 options={'disp': True}, )
    # print(x_rec)
    # np.save('result_bfgs.npy', x_rec.x)

    # u_full = circle_process.recover_data(result[0][:101765])
    # plt.imshow(u_full.reshape(RESOLUTION, RESOLUTION), cmap=cm.jet)
    # plt.colorbar()
    # plt.show()

    # scipy.io.savemat('decomposition.mat', {'A_1': A_1, 'A_2': A_2,
    #                                        'b_1': b_1, 'b_2': b_2,})

    # uutt = uv_ls[:262144]
    # uvtt = uv_ls[262144:]
    #
    # plt.imshow(result[0][:262144].reshape(512, 512), cmap=cm.jet)
    # plt.colorbar()
    # plt.show()
