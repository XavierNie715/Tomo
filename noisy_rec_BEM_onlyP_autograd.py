import os

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy
import torch
from scipy import ndimage
from utilities import funcs
import BEM
import time

from autograd_minimize import minimize


# from optimparallel import minimize_parallel
def forward_phi(_step, _phi_init, _forward_model):
    """
    Forward the model by _steps
    :param _step: Number of steps to forward the model
    :param _phi_init: Initial state of the model (vectorized)
    :param _forward_model: The forward model to be used
    :return: list of frames after each step of forwarding the model
    """
    # resolution = int(np.sqrt(_phi_init.shape[0]))

    _phi_frame = [_phi_init]
    _phi_temp = _phi_init
    for i_step in range(_step):
        # _phi_temp = ndimage.zoom(_phi_init.reshape(resolution, resolution),
        #                         zoom=2,
        #                         mode='wrap').ravel()  # 3-order spline interpolate
        # _phi_temp = _forward_model @ _phi_temp
        _phi_temp = funcs.runge_kutta_4th(_phi_temp, DELTA_T, _forward_model)
        # _phi_temp = ndimage.zoom(_phi_temp.reshape(resolution * 2, resolution * 2),
        #                         zoom=1 / 2,
        #                         mode='wrap').ravel()  # down-sampling
        _phi_frame.append(_phi_temp)
    return _phi_frame


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


# def opt_func(diffusion_coefficient, delta_t, delta_x, diff_form, _psi):
#     """
#
#     :param diffusion_coefficient:
#     :param delta_t:
#     :param delta_x:
#     :param diff_form:
#     :param _psi: (num, n**2)
#     :param _kappa:
#     :return:
#     """
#
#     def return_opt_fun(_p_0):
#         _p_0 = _p_0.reshape((-1, 2), order='F')
#         _complex_potential = _p_0[:, 0] + 1j * _p_0[:, 1]
#
#         _u = (Dx_cauchy @ _complex_potential).ravel().real
#         _v = (Dy_cauchy @ _complex_potential).ravel().real
#
#         rk4_forward = funcs.rk4_forward_mat(_u, _v, diffusion_coefficient, delta_t, delta_x, diff_form)
#         _phi_1 = rk4_forward @ phi_0
#
#         obs_term = 0.5 * np.linalg.norm(_phi_1 - _psi[1]) ** 2
#         # uv_smooth = 0.5 * _kappa * np.linalg.norm(L_reso @ _u + L_reso @ _v) ** 2
#         uv_smooth = 0.5 * kappa_uv * (np.linalg.norm(L_reso @ _u) ** 2 + np.linalg.norm(L_reso @ _v) ** 2)
#
#         return obs_term + uv_smooth
#
#     return return_opt_fun


def opt_func(diffusion_coefficient, delta_t, delta_x, diff_form, _psi):
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

    def return_opt_fun(_p_0):
        _complex_potential = torch.view_as_complex(funcs.reshape_torch_fortran(_p_0, (-1, 2)).contiguous())
        _complex_potential_all = (cauchy_integral_matrix @ _complex_potential)

        _u = ((funcs.Finite_difference_operator_torch(_complex_potential_all, DELTA_x, RESOLUTION,
                                                      weights_1st=torch.tensor((-1, 0, 1), dtype=torch.complex128))
               .diff_1d_x()).ravel().real)
        _v = ((funcs.Finite_difference_operator_torch(_complex_potential_all, DELTA_x, RESOLUTION,
                                                      weights_1st=torch.tensor((-1, 0, 1), dtype=torch.complex128))
               .diff_1d_y()).ravel().real)

        du = funcs.Finite_difference_operator_torch(_u, DELTA_x, RESOLUTION)
        dv = funcs.Finite_difference_operator_torch(_v, DELTA_x, RESOLUTION)

        _dphi_dt = funcs.dphi_dt(_u, _v, DIFFUSION_COEFFICIENT, DELTA_x)
        _phi = torch.stack(forward_phi(FRAMES - 1, phi_0, _dphi_dt))

        obs_term = 0.5 * (torch.linalg.norm(_phi - _psi, axis=1) ** 2).mean()

        # uv_smooth = 0.5 * _kappa * np.linalg.norm(L_reso @ _u + L_reso @ _v) ** 2
        uv_smooth = 0.5 * kappa_uv * (torch.linalg.norm(du.laplacian_input()) ** 2
                                      + torch.linalg.norm(dv.laplacian_input()) ** 2)

        return obs_term + uv_smooth

    return return_opt_fun


# def opt_jac(diffusion_coefficient, delta_t, delta_x, diff_form, _psi):
#     def return_jac_fun(rec):
#         _complex_potential = rec.reshape((-1, 2), order='F')
#         _complex_potential = _complex_potential[:, 0] + 1j * _complex_potential[:, 1]
#
#         _u = (Dx_cauchy @ _complex_potential).ravel().real
#         _v = (Dy_cauchy @ _complex_potential).ravel().real
#
#         rk4_forward = funcs.rk4_forward_mat(_u, _v, diffusion_coefficient, delta_t, delta_x, diff_form)
#         _phi_1 = rk4_forward @ phi_0
#
#         grad_u = funcs.rk4_velocity_gradient(_u, _v, phi_0, 'u',
#                                              DIFFUSION_COEFFICIENT, DELTA_T, DELTA_x,
#                                              'order-2')  # p_phi_p_u
#         grad_v = funcs.rk4_velocity_gradient(_u, _v, phi_0, 'v',
#                                              DIFFUSION_COEFFICIENT, DELTA_T, DELTA_x,
#                                              'order-2')
#         p_L_p_potential = (np.conj(Dx_cauchy).T @ (grad_u.T @ (_phi_1 - _psi[1]) + kappa_uv * L_reso.T @ L_reso @ _u)
#                            + np.conj(Dy_cauchy).T @ (grad_v.T @ (_phi_1 - _psi[1]) + kappa_uv * L_reso.T @ L_reso @ _v))
#         p_L_p_potential = np.concatenate((p_L_p_potential.real, p_L_p_potential.imag))
#         return p_L_p_potential
#
#     return return_jac_fun


def opt_jac(diffusion_coefficient, delta_t, delta_x, diff_form, _psi):
    def return_jac_fun(rec):
        _complex_potential = rec.reshape((-1, 2), order='F')
        _complex_potential = _complex_potential[:, 0] + 1j * _complex_potential[:, 1]

        _u = (Dx_cauchy @ _complex_potential).ravel().real
        _v = (Dy_cauchy @ _complex_potential).ravel().real

        rk4_forward = funcs.rk4_forward_mat(_u, _v, diffusion_coefficient, delta_t, delta_x, diff_form)
        _phi_1 = rk4_forward @ phi_0
        _phi_2 = rk4_forward @ phi_0

        grad_u = funcs.rk4_velocity_gradient(_u, _v, _phi_1, 'u',
                                             DIFFUSION_COEFFICIENT, DELTA_T, DELTA_x,
                                             'order-2')  # p_phi_p_u
        grad_v = funcs.rk4_velocity_gradient(_u, _v, _phi_1, 'v',
                                             DIFFUSION_COEFFICIENT, DELTA_T, DELTA_x,
                                             'order-2')
        p_L_p_potential = (np.conj(Dx_cauchy).T @ (grad_u.T @ (_phi_2 - _psi[-1]) + kappa_uv * L_reso.T @ L_reso @ _u)
                           + np.conj(Dy_cauchy).T @ (
                                   grad_v.T @ (_phi_2 - _psi[-1]) + kappa_uv * L_reso.T @ L_reso @ _v))
        p_L_p_potential = np.concatenate((p_L_p_potential.real, p_L_p_potential.imag))
        return p_L_p_potential

    return return_jac_fun


def opt_jac_torch(diffusion_coefficient, delta_t, delta_x, diff_form, _psi):
    def return_jac_fun(_p_0):
        x = torch.tensor(_p_0, requires_grad=True)
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
    NUM_ksi = 256
    # NUM_ksi = 16
    FRAMES = 40
    kappa_uv = 1e-6

    # noise_intensity = 0.03
    noise_intensity = None
    np.random.seed(42)
    down_size = down_sample(RESOLUTION, mode='mirror')

    # ROOT = "C:\Ph.D_xyN\Hele-Shaw Example"
    ROOT = "//mnt/c/Ph.D_xyN/Hele-Shaw Example"

    Dx_reso = (funcs.differential_matrix_x(RESOLUTION, difference_form='order-2') / DELTA_x)
    # Dx_reso_torch = funcs.scipy_sparse2_torch_sparse(Dx_reso)
    Dy_reso = (funcs.differential_matrix_y(RESOLUTION, difference_form='order-2') / DELTA_x)
    # Dy_reso_torch = funcs.scipy_sparse2_torch_sparse(Dy_reso)
    L_reso = (funcs.laplacian_matrix_2nd(RESOLUTION, difference_form='order-2') / DELTA_x ** 2)

    # cached matrix multiply
    phi = []
    frame_start = 1250
    frame_num = FRAMES
    for frame_index in range(frame_num):
        frame_path = f'Frame_0.{frame_start + frame_index}_S.mat'
        frame_temp = {'file': frame_path,
                      'phi': scipy.io.loadmat(os.path.join(ROOT, frame_path))['Phi']}
        # 'phi': scipy.io.loadmat(os.path.join(ROOT, frame_path))['Phi'][384:512, 384:512]}
        phi_temp = (add_noise(frame_temp['phi'].ravel(), noise_intensity) if noise_intensity
                    else (frame_temp['phi'].ravel()))
        phi.append(down_size(phi_temp.reshape(1024, 1024)[256:768, 256:768]).ravel())
    psi = np.array(phi)
    phi_0 = torch.tensor(psi[0])

    boundary_ksi_coord = BEM.potential_point_circle(num_ksi=NUM_ksi, radius=RESOLUTION,
                                                    center=(int(RESOLUTION / 2), int(RESOLUTION / 2)))
    cauchy_integral_matrix = torch.tensor(BEM.integral_matrix_from_boundary(boundary_ksi_coord,
                                                                            RESOLUTION, DOMAIN_LENGTH))

    # cached matrix multiply
    # Dx_cauchy = Dx_reso @ cauchy_integral_matrix
    # Dy_cauchy = Dy_reso @ cauchy_integral_matrix
    # Dx_cauchy_torch = torch.tensor(Dx_cauchy)
    # Dy_cauchy_torch = torch.tensor(Dy_cauchy)

    BFGS_opt_func = opt_func(DIFFUSION_COEFFICIENT, DELTA_T, DELTA_x, 'order-2', psi)
    # BFGS_opt_jac = opt_jac(DIFFUSION_COEFFICIENT, DELTA_T, DELTA_x, 'order-2', psi)
    # BFGS_opt_func_torch = opt_func_torch(DIFFUSION_COEFFICIENT, DELTA_T, DELTA_x, 'order-2')
    # BFGS_opt_jac_torch = opt_jac_torch(DIFFUSION_COEFFICIENT, DELTA_T, DELTA_x, 'order-2')

    p_int = np.zeros(NUM_ksi * 2)
    # p_gt = np.load('BEM_gt_256p_512.npy')
    # p_int = np.random.random((NUM_ksi * 2))
    # x_int = torch.tensor(p_int, requires_grad=True)

    # ttt = BFGS_opt_func_torch(torch.rand(NUM_ksi * 2))

    # for debugging
    # uv = scipy.io.loadmat('/Users/xiangyu.nie/PycharmProjects/Tomo/VField.mat')
    # u_gt = (uv['U'][256:768, 256:768]).ravel()
    # v_gt = (uv['V'][256:768, 256:768]).ravel()

    # grad_torch = BFGS_opt_jac_torch(x_int)
    # grad = BFGS_opt_jac(phi_int)
    # t = scipy.optimize.check_grad(BFGS_opt_func, BFGS_opt_jac, phi_int, direction='all') / 64
    # torch.autograd.gradcheck(BFGS_opt_func_torch, x_int, eps=1e-8) / RESOLUTION

    star_time = time.time()
    x_iter = []
    # x_rec = scipy.optimize.minimize(fun=BFGS_opt_func,
    #                                 x0=p_int, method='L-BFGS-B',
    #                                 jac=BFGS_opt_jac,
    #                                 options={'disp': True}, )
    try:
        x_rec = minimize(BFGS_opt_func, p_int, method='L-BFGS-B',
                         backend='torch', precision="float64",
                         options={'disp': True, 'maxiter': 2000},
                         callback=lambda intermediate_result: x_iter.append(intermediate_result))
    except:
        cp_rec = x_iter[-1].x
    print(f'opt cost {time.gmtime(time.time() - star_time)}')

    # evaluate
    cp_rec = x_rec.x.reshape((-1, 2), order='F')
    cp_rec = cp_rec[:, 0] + 1j * cp_rec[:, 1]
    u_rec = (Dx_reso @ cauchy_integral_matrix.numpy() @ cp_rec).ravel().real
    v_rec = (Dy_reso @ cauchy_integral_matrix.numpy() @ cp_rec).ravel().real
    # uv = scipy.io.loadmat("C://Ph.D_xyN//tomo//VField.mat")
    # u_gt = (uv['U'][256:768, 256:768]).ravel()
    # v_gt = (uv['V'][256:768, 256:768]).ravel()
    # rk4 = funcs.rk4_forward_mat(u_rec, v_rec, DIFFUSION_COEFFICIENT, DELTA_T, DELTA_x, 'order-2')
    # rk4_gt = funcs.rk4_forward_mat(u_gt, v_gt, DIFFUSION_COEFFICIENT, DELTA_T, DELTA_x, 'order-2')
    # phi_0_gt = down_size(scipy.io.loadmat("C:\Ph.D_xyN\Hele-Shaw Example new\Frame_0.1250_S.mat")['Phi']
    #                      .reshape(1024, 1024)[256:768, 256:768]).ravel()
    # phi_1_gt = down_size(scipy.io.loadmat("C:\Ph.D_xyN\Hele-Shaw Example new\Frame_0.1251_S.mat")['Phi']
    #                      .reshape(1024, 1024)[256:768, 256:768]).ravel()
    # phi_1_rec = rk4 @ phi_0_gt
    # eval_obs_term = 0.5 * np.linalg.norm(phi_1_rec - psi[1]) ** 2
    # gt_obs_term = 0.5 * np.linalg.norm(rk4_gt @ phi_0_gt - psi[1]) ** 2
    # eval_smooth_term = 0.5 * kappa_uv * (np.linalg.norm(L_reso @ u_rec) ** 2 + np.linalg.norm(L_reso @ v_rec) ** 2)
    # gt_smooth_term = 0.5 * kappa_uv * (np.linalg.norm(L_reso @ u_gt) ** 2 + np.linalg.norm(L_reso @ v_gt) ** 2)
    # error_avg = ((np.linalg.norm(u_rec - u_gt) + np.linalg.norm(v_rec - v_gt))
    #              / (np.linalg.norm(u_gt) + np.linalg.norm(v_gt)))
    #
    # print(f'kappa = {kappa_uv}')
    # print(f'eval_obs_term: {eval_obs_term:.3e}')
    # print(f'gt_obs_term: {gt_obs_term:.3e}')
    # print(f'eval_smooth_term: {eval_smooth_term:.3e}')
    # print(f'gt_smooth_term: {gt_smooth_term:.3e}')
    # print(f'error_avg: {error_avg:.3e}')

    plt.figure()
    plt.imshow(u_rec.reshape(RESOLUTION, RESOLUTION), cmap=cm.jet)
    plt.colorbar()
    plt.clim(-15, 0)
    plt.show()
