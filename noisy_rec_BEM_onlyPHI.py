import os
import numpy as np
import scipy
import torch
from scipy import ndimage
from utilities import funcs
import time
from autograd_minimize import minimize


def down_sample(_resolution, mode='mirror'):
    def zoom(_data):
        return ndimage.zoom(_data, _resolution / 512, mode=mode)

    return zoom


def partial_phi_partial_t(frame_i, delta_t, root_path, _add_noise=None, _circle_process=None):
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

    p_phi_p_t = (phi_i_plus_1[256:768, 256:768] - frame_i['phi'][256:768, 256:768]) / delta_t

    # if _circle_process:
    #     return circle_process.data_in_circle((phi_i_plus_1 - phi_i_minus_1) / (2 * delta_t))

    if _add_noise:
        p_phi_p_t = add_noise(p_phi_p_t, noise_intensity)

    return down_size(ndimage.median_filter(p_phi_p_t, 5, mode='mirror'))

    # return (phi_i_plus_1[256:768, 256:768] - phi_i_minus_1[256:768, 256:768]) / (2 * delta_t)
    # return down_size((phi_i_plus_1[256:768, 256:768] - phi_i_minus_1[256:768, 256:768]) / (2 * delta_t))
    # return (phi_i_plus_1[320:704, 320:704] - phi_i_minus_1[320:704, 320:704]) / (2 * delta_t)
    # return (phi_i_plus_1 - phi_i_minus_1) / (2 * delta_t)


def add_noise(clean_data, intensity=0.05):
    """

    :param clean_data:
    :param intensity: 噪声大小百分比
    :return:
    """
    return clean_data + np.random.normal(0, np.abs(intensity * clean_data))


def opt_func_torch(diffusion_coefficient, delta_t, delta_x, diff_form, _psi, _kappa):
    """

    :param diffusion_coefficient:
    :param delta_t:
    :param delta_x:
    :param diff_form:
    :param _psi: (num, n**2)
    :param _kappa:
    :return:
    """
    _psi = torch.tensor(_psi)

    def return_opt_fun(_phi_0):
        _phi_1 = rk4_torch @ _phi_0

        # _phi = np.concatenate((_phi_0, _phi_1), axis=0).reshape(2, -1)
        _phi = torch.cat((_phi_0, _phi_1)).reshape(2, -1)

        obs_term = (torch.linalg.norm(_phi - _psi, axis=1) ** 2).mean()
        phi_smooth = 0.5 * _kappa * torch.linalg.norm(L_reso_torch @ _phi_0) ** 2

        return obs_term + phi_smooth

    return return_opt_fun


def opt_func(diffusion_coefficient, delta_t, delta_x, diff_form, _psi, _kappa):
    """

    :param diffusion_coefficient:
    :param delta_t:
    :param delta_x:
    :param diff_form:
    :param _psi: (num, n**2)
    :param _kappa:
    :return:
    """

    def return_opt_fun(_phi_0):
        _phi_1 = rk4 @ _phi_0

        _phi = np.concatenate((_phi_0, _phi_1), axis=0).reshape(2, -1)
        # _phi = torch.cat((_phi_0, _phi_1)).reshape(2, -1)

        obs_term = (np.linalg.norm(_phi - _psi, axis=1) ** 2).mean()
        phi_smooth = 0.5 * _kappa * np.linalg.norm(L_reso @ _phi_0) ** 2

        return obs_term + phi_smooth

    return return_opt_fun


def opt_jac(diffusion_coefficient, delta_t, delta_x, diff_form, _psi, _kappa):
    def return_jac_fun(_phi_0):
        _phi_1 = rk4 @ _phi_0
        p_L_p_phi = (_phi_0 - _psi[0]) + rk4.T @ (_phi_1 - _psi[1]) + _kappa * L_reso.T @ L_reso @ _phi_0
        return p_L_p_phi

    return return_jac_fun


def opt_jac_torch(diffusion_coefficient, delta_t, delta_x, diff_form, _psi, _kappa):
    def return_jac_fun(_phi_0):
        x = torch.tensor(_phi_0, requires_grad=True)
        y = BFGS_opt_func_torch(x)
        y.backward()
        return x.grad.numpy()

    return return_jac_fun


if __name__ == '__main__':
    DELTA_T = 1e-4
    RESOLUTION = 512
    # RESOLUTION = 8
    DOMAIN_LENGTH = np.pi  # unit: [cm]
    DIFFUSION_COEFFICIENT = 0.15  # unit: [cm^2 / s]
    DELTA_x = DOMAIN_LENGTH / RESOLUTION

    kappa = 1e-7  # regularization coefficient

    noise_intensity = 0.03
    # noise_intensity = None
    down_size = down_sample(RESOLUTION, mode='mirror')

    ROOT = '/Users/xiangyu.nie/Document/PhD/Hele-Shaw Example new'

    Dx_reso = (funcs.differential_matrix_x(RESOLUTION, difference_form='order-2') / DELTA_x)
    Dy_reso = (funcs.differential_matrix_y(RESOLUTION, difference_form='order-2') / DELTA_x)
    L_reso = (funcs.laplacian_matrix_2nd(RESOLUTION, difference_form='order-2') / DELTA_x ** 2)
    # L_reso_torch = torch.tensor(L_reso)
    # L_reso_torch = torch.sparse_coo_tensor(np.array(L_reso.nonzero()), L_reso.data, L_reso.shape)
    L_reso_torch = funcs.scipy_sparse2_torch_sparse(L_reso)

    # cached matrix multiply
    phi = []
    frame_start = 1250
    frame_num = 2
    for frame_index in range(frame_num):
        frame_path = f'Frame_0.{frame_start + frame_index}_S.mat'
        frame_temp = {'file': frame_path,
                      'phi': scipy.io.loadmat(os.path.join(ROOT, frame_path))['Phi']}
        # 'phi': scipy.io.loadmat(os.path.join(ROOT, frame_path))['Phi'][384:512, 384:512]}
        phi_temp = (add_noise(frame_temp['phi'].ravel(), noise_intensity) if noise_intensity
                    else (frame_temp['phi'].ravel()))
        phi.append(down_size(phi_temp.reshape(1024, 1024)[256:768, 256:768]).ravel())
    psi = np.array(phi)

    BFGS_opt_func = opt_func(DIFFUSION_COEFFICIENT, DELTA_T, DELTA_x, 'order-2', psi, kappa)
    BFGS_opt_func_torch = opt_func_torch(DIFFUSION_COEFFICIENT, DELTA_T, DELTA_x, 'order-2', psi, kappa)
    BFGS_opt_jac = opt_jac(DIFFUSION_COEFFICIENT, DELTA_T, DELTA_x, 'order-2', psi, kappa)
    BFGS_opt_jac_torch = opt_jac_torch(DIFFUSION_COEFFICIENT, DELTA_T, DELTA_x, 'order-2', psi, kappa)

    # phi_int = ndimage.median_filter(psi[0], 5, mode='mirror')
    phi_int = psi[0]
    # phi_int = np.random.random((64))
    x_int = torch.tensor(phi_int, requires_grad=True)

    # for debugging
    uv = scipy.io.loadmat('/Users/xiangyu.nie/PycharmProjects/Tomo/VField.mat')
    u_gt = (uv['U'][256:768, 256:768]).ravel()
    v_gt = (uv['V'][256:768, 256:768]).ravel()
    rk4 = funcs.rk4_forward_mat(u_gt, v_gt, DIFFUSION_COEFFICIENT, DELTA_T, DELTA_x, 'order-2')
    rk4_torch = funcs.scipy_sparse2_torch_sparse(rk4)

    phi1 = rk4 @ phi_int
    grad_u = funcs.rk4_velocity_gradient(u_gt, v_gt, phi_int, 'u', DIFFUSION_COEFFICIENT, DELTA_T, DELTA_x, 'order-2')

    # grad_torch = BFGS_opt_jac_torch(x_int)
    # grad = BFGS_opt_jac(phi_int)
    # t = scipy.optimize.check_grad(BFGS_opt_func, BFGS_opt_jac, phi_int, direction='all') / 64
    # torch.autograd.gradcheck(BFGS_opt_func_torch, x_int, eps=1e-8) / RESOLUTION

    star_time = time.time()
    # x_rec = scipy.optimize.minimize(fun=BFGS_opt_func,
    #                                 x0=phi_int, method='L-BFGS-B',
    #                                 jac=BFGS_opt_jac,
    #                                 options={'disp': True}, )

    x_rec = minimize(BFGS_opt_func_torch, phi_int, method='L-BFGS-B',
                     backend='torch', precision="float64",
                     options={'disp': True})

    # # x_rec = minimize_parallel(fun=BFGS_opt_func,
    # #                           x0=x_int, jac=BFGS_opt_jac,
    # #                           parallel={})
    print(f'opt cost {time.gmtime(time.time() - star_time)}')
    # print(x_rec)
    # np.save('result_BEM_bfgs_onlyPHI.npy', x_rec.x)

    # uv = scipy.io.loadmat('/Users/xiangyu.nie/PycharmProjects/Tomo/VField.mat')
    # u = uv['U'][256:768, 256:768].ravel()
    # v = uv['V'][256:768, 256:768].ravel()
    # uv = np.concatenate((u, v), axis=0)

    # plt.imshow(u_rec.reshape(RESOLUTION, RESOLUTION), cmap=cm.jet)
    # plt.colorbar()
    # plt.clim(-15, 0)
    # plt.show()
    #
    # plt.imshow(v_rec.reshape(RESOLUTION, RESOLUTION), cmap=cm.jet)
    # plt.colorbar()
    # plt.clim(0, 20)
    # plt.show()

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
