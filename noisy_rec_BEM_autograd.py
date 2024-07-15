import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy
from scipy import ndimage
from utilities import funcs
from utilities import funcs_torch
import BEM
import time
import torch
from autograd_minimize import minimize


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
        _phi_temp = funcs.runge_kutta_4th(_phi_temp, DELTA_T, _forward_model)
        _phi_frame.append(_phi_temp)
    return _phi_frame


def down_sample(_resolution, mode='mirror'):
    def zoom(_data):
        return ndimage.zoom(_data, _resolution / 512, mode=mode)

    return zoom


def add_noise(clean_data, intensity=0.05):
    """

    :param clean_data:
    :param intensity: 噪声大小百分比
    :return:
    """
    return clean_data + np.random.normal(0, np.abs(intensity * clean_data))


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

    def return_opt_fun(rec):
        _phi_0 = rec[:RESOLUTION ** 2]
        _p_0 = rec[RESOLUTION ** 2:]
        _complex_potential = torch.view_as_complex(funcs_torch.reshape_torch_fortran(_p_0, (-1, 2)).contiguous())
        _complex_potential_all = (cauchy_integral_matrix @ _complex_potential)

        _u = ((funcs_torch.Finite_difference_operator_torch(_complex_potential_all, DELTA_x, RESOLUTION,
                                                      weights_1st=torch.tensor((-1, 0, 1), dtype=torch.complex128))
               .diff_1d_x()).ravel().real)
        _v = ((funcs_torch.Finite_difference_operator_torch(_complex_potential_all, DELTA_x, RESOLUTION,
                                                      weights_1st=torch.tensor((-1, 0, 1), dtype=torch.complex128))
               .diff_1d_y()).ravel().real)
        du = funcs_torch.Finite_difference_operator_torch(_u, DELTA_x, RESOLUTION)
        dv = funcs_torch.Finite_difference_operator_torch(_v, DELTA_x, RESOLUTION)

        dphi_dt = funcs.dphi_dt(_u, _v, DIFFUSION_COEFFICIENT, DELTA_x)
        _phi = torch.stack(forward_phi(FRAMES - 1, _phi_0, dphi_dt))

        obs_term = 0.5 * (torch.linalg.norm(_phi - _psi, axis=1) ** 2).mean()
        phi_smooth = (0.5 * kappa_phi *
                      torch.linalg.norm(funcs_torch.Finite_difference_operator_torch(_phi_0,
                                                                               DELTA_x,
                                                                               RESOLUTION).laplacian_input()) ** 2)
        uv_smooth = 0.5 * kappa_uv * (torch.linalg.norm(du.laplacian_input()) ** 2
                                      + torch.linalg.norm(dv.laplacian_input()) ** 2)

        return obs_term + phi_smooth + uv_smooth

    return return_opt_fun


def opt_jac(diffusion_coefficient, delta_t, delta_x, diff_form, _psi):
    def return_jac_fun(rec):
        _phi_0 = rec[:RESOLUTION ** 2]
        _complex_potential = rec[RESOLUTION ** 2:].reshape((-1, 2), order='F')
        _complex_potential = _complex_potential[:, 0] + 1j * _complex_potential[:, 1]

        _u = (Dx_cauchy @ _complex_potential).ravel().real
        _v = (Dy_cauchy @ _complex_potential).ravel().real

        rk4_forward = funcs.forward_model(_u, _v, diffusion_coefficient, delta_t, delta_x, diff_form)
        _phi_1 = rk4_forward @ _phi_0

        p_L_p_phi = (_phi_0 - _psi[0]) + rk4_forward.T @ (_phi_1 - _psi[1]) + kappa_phi * L_reso.T @ L_reso @ _phi_0
        grad_u = funcs.rk4_velocity_gradient(_u, _v, _phi_0, 'u',
                                             DIFFUSION_COEFFICIENT, DELTA_T, DELTA_x,
                                             'order-2')  # p_phi_p_u
        grad_v = funcs.rk4_velocity_gradient(_u, _v, _phi_0, 'v',
                                             DIFFUSION_COEFFICIENT, DELTA_T, DELTA_x,
                                             'order-2')
        p_L_p_potential = (np.conj(Dx_cauchy).T @ (grad_u.T @ (_phi_1 - _psi[1]) + kappa_uv * L_reso.T @ L_reso @ _u)
                           + np.conj(Dy_cauchy).T @ (grad_v.T @ (_phi_1 - _psi[1]) + kappa_uv * L_reso.T @ L_reso @ _v))
        p_L_p_potential = np.concatenate((p_L_p_potential.real, p_L_p_potential.imag))
        return np.concatenate((p_L_p_phi, p_L_p_potential), axis=None)

    return return_jac_fun


if __name__ == '__main__':
    DELTA_T = 1e-4
    RESOLUTION = 512
    DOMAIN_LENGTH = np.pi  # unit: [cm]
    DIFFUSION_COEFFICIENT = 0.15  # unit: [cm^2 / s]
    DELTA_x = DOMAIN_LENGTH / RESOLUTION
    NUM_ksi = 256
    FRAMES = 25  # total frames, including frame #0
    kappa_phi = 1e-7  # regularization coefficient
    kappa_uv = 0
    # obs_decay = torch.tensor([])

    # noise_intensity = 0.03
    np.random.seed(42)
    noise_intensity = None
    down_size = down_sample(RESOLUTION, mode='mirror')

    ROOT = "/Users/xiangyu.nie/Document/PhD/Hele-Shaw Example"
    SAVE_ROOT = "./results"
    SAVE_NAME = f'rec_all_uv_{kappa_uv}_phi_{kappa_phi}_ksi_{NUM_ksi}_noise_{noise_intensity}_frame_{FRAMES}'

    Dx_reso = (funcs.differential_matrix_x(RESOLUTION, difference_form='order-2') / DELTA_x)
    Dy_reso = (funcs.differential_matrix_y(RESOLUTION, difference_form='order-2') / DELTA_x)
    L_reso = (funcs.laplacian_matrix_2nd(RESOLUTION, difference_form='order-2') / DELTA_x ** 2)

    boundary_ksi_coord = BEM.potential_point_circle(num_ksi=NUM_ksi, radius=RESOLUTION,
                                                    center=(int(RESOLUTION / 2), int(RESOLUTION / 2)))
    cauchy_integral_matrix = torch.tensor(BEM.integral_matrix_from_boundary(boundary_ksi_coord,
                                                                            RESOLUTION, DOMAIN_LENGTH))

    # cached matrix multiply
    # Dx_cauchy = Dx_reso @ cauchy_integral_matrix
    # Dy_cauchy = Dy_reso @ cauchy_integral_matrix

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

    BFGS_opt_func = opt_func(DIFFUSION_COEFFICIENT, DELTA_T, DELTA_x, 'order-2', psi)
    # BFGS_opt_jac = opt_jac(DIFFUSION_COEFFICIENT, DELTA_T, DELTA_x, 'order-2', psi)

    phi_int = psi[0]
    # phi_int = ndimage.median_filter(psi[0], 5, mode='mirror')
    # uv_int = scipy.io.loadmat('./results/matlab_QR/ans_1e-3_alldiff_noisy_median5_256.mat')['ans'].ravel()
    uv_int = np.zeros(NUM_ksi * 2)
    x_int = np.concatenate((phi_int, uv_int), axis=0)

    star_time = time.time()
    x_iter = []
    try:
        x_rec = minimize(BFGS_opt_func, x_int, method='L-BFGS-B',
                         backend='torch', precision="float64",
                         options={'disp': True, 'maxiter': 2000},
                         callback=lambda intermediate_result: x_iter.append(intermediate_result))
    except:
        cp_rec = x_iter[-1].x
    print(f'opt cost {time.gmtime(time.time() - star_time)}')
    print(x_rec)
    os.makedirs(os.path.join(SAVE_ROOT, SAVE_NAME), exist_ok=True)
    np.save(os.path.join(SAVE_ROOT, SAVE_NAME, 'result_BEM_bfgs.npy'), x_rec.x)

    # eval
    phi_0_rec = x_rec.x[:RESOLUTION ** 2]
    cp_rec = x_rec.x[RESOLUTION ** 2:].reshape((-1, 2), order='F')
    cp_rec = cp_rec[:, 0] + 1j * cp_rec[:, 1]
    u_rec = (Dx_reso @ cauchy_integral_matrix.numpy() @ cp_rec).ravel().real
    # v_rec = (Dy_reso @ cauchy_integral_matrix @ cp_rec).ravel().real
    # uv = scipy.io.loadmat("C://Ph.D_xyN//tomo//VField.mat")
    # u_gt = (uv['U'][256:768, 256:768]).ravel()
    # v_gt = (uv['V'][256:768, 256:768]).ravel()
    # rk4_rec = funcs.forward_model(u_rec, v_rec, DIFFUSION_COEFFICIENT, DELTA_T, DELTA_x, 'order-2')
    # rk4_gt = funcs.forward_model(u_gt, v_gt, DIFFUSION_COEFFICIENT, DELTA_T, DELTA_x, 'order-2')
    # phi_0_gt = down_size(scipy.io.loadmat("C:\Ph.D_xyN\Hele-Shaw Example new\Frame_0.1250_S.mat")['Phi']
    #                      .reshape(1024, 1024)[256:768, 256:768]).ravel()
    # phi_1_gt = down_size(scipy.io.loadmat("C:\Ph.D_xyN\Hele-Shaw Example new\Frame_0.1251_S.mat")['Phi']
    #                      .reshape(1024, 1024)[256:768, 256:768]).ravel()
    # phi_rec = np.concatenate((phi_0_rec, rk4_rec @ phi_0_rec), axis=0).reshape(2, -1)
    # phi_gt_rk4 = np.concatenate((phi_0_gt, rk4_gt @ phi_0_gt), axis=0).reshape(2, -1)
    #
    # eval_obs_term = (np.linalg.norm(phi_rec - psi, axis=1) ** 2).mean()
    # gt_obs_term = (np.linalg.norm(phi_gt_rk4 - psi, axis=1) ** 2).mean()
    # eval_smooth_phi_term = 0.5 * kappa_phi * np.linalg.norm(L_reso @ phi_0_rec) ** 2
    # gt_smooth_phi_term = 0.5 * kappa_phi * np.linalg.norm(L_reso @ phi_0_gt) ** 2
    # eval_smooth_uv_term = 0.5 * kappa_uv * (np.linalg.norm(L_reso @ u_rec) ** 2 + np.linalg.norm(L_reso @ v_rec) ** 2)
    # gt_smooth_uv_term = 0.5 * kappa_uv * (np.linalg.norm(L_reso @ u_gt) ** 2 + np.linalg.norm(L_reso @ v_gt) ** 2)
    # rel_error_phi = np.linalg.norm(phi_0_gt - phi_0_rec) / np.linalg.norm(phi_0_gt)
    # rel_error_uv = ((np.linalg.norm(u_rec - u_gt) + np.linalg.norm(v_rec - v_gt))
    #                 / (np.linalg.norm(u_gt) + np.linalg.norm(v_gt)))
    #
    # print(f'kappa_phi = {kappa_phi}')
    # print(f'kappa_uv = {kappa_uv}')
    # print(f'eval_obs_term: {eval_obs_term:.3e}')
    # print(f'gt_obs_term: {gt_obs_term:.3e}')
    # print(f'eval_smooth_phi_term: {eval_smooth_phi_term:.3e}')
    # print(f'gt_smooth_phi_term: {gt_smooth_phi_term:.3e}')
    # print(f'eval_smooth_uv_term: {eval_smooth_uv_term:.3e}')
    # print(f'gt_smooth_uv_term: {gt_smooth_uv_term:.3e}')
    # print(f'rel_error_phi: {rel_error_phi:.3e}')
    # print(f'rel_error_uv: {rel_error_uv:.3e}')

    plt.figure()
    plt.imshow(phi_0_rec.reshape(RESOLUTION, RESOLUTION), cmap=cm.jet)
    plt.colorbar()
    plt.savefig(os.path.join(SAVE_ROOT, SAVE_NAME, 'rec_phi_0.jpg'), dpi=300)

    plt.figure()
    plt.imshow((Dx_reso @ cauchy_integral_matrix.numpy() @ cp_rec).ravel().real.reshape(RESOLUTION,
                                                                                        RESOLUTION), cmap=cm.jet)
    plt.colorbar()
    plt.savefig(os.path.join(SAVE_ROOT, SAVE_NAME, 'rec_u.jpg'), dpi=300)

    # plt.figure()
    # plt.imshow((Dy_reso @ cauchy_integral_matrix @ cp_rec).ravel().real.reshape(RESOLUTION, RESOLUTION), cmap=cm.jet)
    # plt.colorbar()
    # plt.savefig(os.path.join(SAVE_ROOT, SAVE_NAME, 'rec_v.jpg'), dpi=300)
