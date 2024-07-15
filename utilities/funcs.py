from functools import cache
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.sparse
import scipy.ndimage
import os


class Mask_Circle:
    """
    This class is used to create a circular mask for a given resolution and radius.
    """

    def __init__(self, resolution=256, radius=120):
        """
        Initialize the Mask_Circle class.

        :param resolution: The resolution of the mask. Default is 256.
        :param radius: The radius of the circle in the mask. Default is 120.
        """
        self.resolution = resolution
        self.radius = radius
        x = y = np.arange(0, self.resolution)
        cx = cy = self.resolution // 2
        self.mask = (((x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 > self.radius ** 2)
                     .reshape(-1, 1).squeeze())  # mask_out_circle

    def __call__(self):
        """
        Return the mask when the class instance is called.
        """
        return self.mask

    def __len__(self):
        return np.count_nonzero(~self.mask)

    def data_in_circle(self, data):
        """
        Return the data within the circle of the mask.

        :param data: The input data. The last dimension should be vectorized.
        :return: The data within the circle of the mask.
        """
        # mask_out_circle = self.mask == True
        data_in_circle = data[..., ~self.mask]
        return data_in_circle

    def recover_data(self, data_in_circle):
        """
        Recover the data with 0 outside the circle of the mask.

        :param data_in_circle: The data within the circle of the mask.
        :return: The recovered data.
        """
        data_full = np.zeros_like(self.mask, dtype=np.float64)
        data_full[~self.mask] = data_in_circle
        return data_full


@cache  # only execute once for same input n
def laplacian_matrix_2nd(n, difference_form='central'):
    """
    Calculate the 2nd order Laplacian matrix for central difference, periodic boundary
    Laplacian_kernel = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]])
    :param n: squared dimension of the vectorized x (along column)
    order-2 : 边界上分别采用向前，向后差分
    :return:n^2 * n^2 matrix
    """

    if difference_form == 'central':
        L_up = scipy.sparse.diags([[1], [1]], offsets=[n ** 2 - n, n], shape=(n ** 2, n ** 2), dtype=np.float32)
        L_down = L_up.T
        diag_block = scipy.sparse.diags([[1], [1], [-4], [1], [1]],
                                        offsets=[-(n - 1), -1, 0, 1, n - 1], shape=(n, n),
                                        dtype=np.float32)
        L = scipy.sparse.block_diag([diag_block] * n) + L_down + L_up

    elif difference_form == 'order-2':
        block_left_top_row_1 = scipy.sparse.hstack([scipy.sparse.coo_array([[2, -2, 1]]),
                                                    scipy.sparse.coo_array((1, n - 3))])
        block_left_top_row_n = scipy.sparse.hstack([scipy.sparse.coo_array((1, n - 3)),
                                                    scipy.sparse.coo_array([[1, -2, 2]])])
        block_left_top = scipy.sparse.vstack((block_left_top_row_1,
                                              scipy.sparse.diags([1, -1, 1], offsets=[0, 1, 2], shape=(n - 2, n)),
                                              block_left_top_row_n))
        block_top = scipy.sparse.hstack((block_left_top,
                                         scipy.sparse.diags([-2], shape=(n, n)),
                                         scipy.sparse.diags([1], shape=(n, n)),
                                         scipy.sparse.coo_array((n, n ** 2 - 3 * n))))

        block_bottom = scipy.sparse.hstack((scipy.sparse.coo_array((n, n ** 2 - 3 * n)),
                                            scipy.sparse.diags([1], shape=(n, n)),
                                            scipy.sparse.diags([-2], shape=(n, n)),
                                            block_left_top))

        middle_block_1 = scipy.sparse.diags([1], shape=(n, n))
        middle_block_2 = scipy.sparse.vstack((scipy.sparse.hstack([scipy.sparse.coo_array([[-1, -2, 1]]),
                                                                   scipy.sparse.coo_array((1, n - 3))]),
                                              scipy.sparse.diags([1, -4, 1], offsets=[0, 1, 2], shape=(n - 2, n)),
                                              scipy.sparse.hstack([scipy.sparse.coo_array((1, n - 3)),
                                                                   scipy.sparse.coo_array([[1, -2, -1]])])))
        middle_block_3 = scipy.sparse.diags([1], shape=(n, n))

        arr_middle_block = np.empty((n - 2, n), object)
        arr_middle_block[np.arange(n - 2), np.arange(n - 2)] = middle_block_1
        arr_middle_block[np.arange(n - 2), np.arange(n - 2) + 1] = middle_block_2
        arr_middle_block[np.arange(n - 2), np.arange(n - 2) + 2] = middle_block_3
        middle_block = scipy.sparse.bmat(arr_middle_block)

        L = scipy.sparse.vstack((block_top, middle_block, block_bottom))

    elif difference_form == 'order-6':
        top_row = scipy.sparse.coo_array((n, n ** 2))
        bottom_row = scipy.sparse.coo_array((n, n ** 2))

        middle_block_1 = scipy.sparse.vstack((scipy.sparse.coo_array((1, n)),
                                              scipy.sparse.diags([[1], [4], [1]],
                                                                 offsets=np.arange(0, 3), shape=(n, n)),
                                              scipy.sparse.coo_array((1, n))))
        middle_block_2 = scipy.sparse.vstack((scipy.sparse.coo_array((1, n)),
                                              scipy.sparse.diags([[4], [-20], [4]],
                                                                 offsets=np.arange(0, 3), shape=(n, n)),
                                              scipy.sparse.coo_array((1, n))))
        middle_block_3 = scipy.sparse.vstack((scipy.sparse.coo_array((1, n)),
                                              scipy.sparse.diags([[1], [4], [1]],
                                                                 offsets=np.arange(0, 3), shape=(n, n)),
                                              scipy.sparse.coo_array((1, n))))

        arr_middle_block = np.empty((n ** 2 - 2, n ** 2), object)
        arr_middle_block[np.arange(n - 2), np.arange(n - 2)] = middle_block_1
        arr_middle_block[np.arange(n - 2), np.arange(n - 2) + 1] = middle_block_2
        arr_middle_block[np.arange(n - 2), np.arange(n - 2) + 2] = middle_block_3

        middle_block = scipy.sparse.bmat(arr_middle_block)
        L = scipy.sparse.vstack([top_row, middle_block, bottom_row])

    return L.tocsr()


@cache  # only execute once for same input n
def differential_matrix_x(n, difference_form='central', circle_process=None):
    """
    Calculate the difference matrix Dx along x direction

    :param n: squared dimension of the vectorized x (along column)
    :param difference_form: which difference form to choose, default is upwind and periodic boundary
    :return: n^2 * n^2 matrix

    'central': periodic boundary
    'other': 边界是单向差分
    """
    if difference_form == 'central':
        diag_block = scipy.sparse.diags([[0.5], [-0.5], [0.5], [-0.5]],
                                        offsets=[-(n - 1), -1, 1, n - 1], shape=(n, n),
                                        dtype=np.float32)

    elif difference_form == 'upwind':
        diag_block = scipy.sparse.diags([[-1], [1], [-1]], offsets=[-1, 0, n - 1], shape=(n, n), dtype=np.float32)

    elif difference_form == 'order-2':
        row_1 = scipy.sparse.hstack([scipy.sparse.coo_array([[-1.5, 2, -0.5]]), scipy.sparse.coo_array((1, n - 3))])
        diag_block = scipy.sparse.diags([[-0.5], [0.5]],
                                        offsets=[0, 2], shape=(n - 2, n),
                                        dtype=np.float32)
        row_n = scipy.sparse.hstack([scipy.sparse.coo_array((1, n - 3)), scipy.sparse.coo_array([[0.5, -2, 1.5]])])
        diag_block = scipy.sparse.vstack([row_1, diag_block, row_n])

    elif difference_form == 'order-2_0_bc':
        diag_block = scipy.sparse.diags([[-0.5], [0.5]],
                                        offsets=[-1, 1], shape=(n, n),
                                        dtype=np.float32)

    elif difference_form == 'order-6':
        block_border_up = scipy.sparse.diags([[- 49 / 20], [6], [-15 / 2], [20 / 3], [-15 / 4], [6 / 5], [-1 / 6]],
                                             offsets=np.arange(0, 7), shape=(3, n))
        diag_block = scipy.sparse.diags([[-1 / 60], [3 / 20], [-3 / 4], [0], [3 / 4], [-3 / 20], [1 / 60]],
                                        offsets=np.arange(0, 7), shape=(n - 6, n))
        block_border_down = scipy.sparse.diags([[-1 / 6], [-6 / 5], [15 / 4], [-20 / 3], [15 / 2], [-6], [49 / 20]],
                                               offsets=np.arange(n - 2 - 7, n - 2), shape=(3, n))
        diag_block = scipy.sparse.vstack([block_border_up, diag_block, block_border_down])

    D = scipy.sparse.block_diag([diag_block] * n)

    if circle_process:
        return (D.tocsr())[:, ~circle_process()][~circle_process(), :]
    return D.tocsr()


@cache  # only execute once for same input n
def differential_matrix_y(n, difference_form='central', circle_process=None):
    """
    Calculate the difference matrix Dy along y direction

    :param n: squared dimension of the vectorized x (along column)
    :param difference_form: which difference form to choose, default is upwind and periodic boundary
    :return: n^2 * n^2 matrix


    """
    if difference_form == 'central':
        D_up = scipy.sparse.diags([[0.5], [-0.5]], offsets=[n, n ** 2 - n], shape=(n ** 2, n ** 2), dtype=np.float32)
        D_down = -D_up.T
        D = D_up + D_down

    elif difference_form == 'upwind':
        D = scipy.sparse.diags([[-1], [1], [-1]], offsets=[-n, 0, n ** 2 - n], shape=(n ** 2, n ** 2), dtype=np.float32)

    elif difference_form == 'order-2':
        block_1 = scipy.sparse.diags([-1.5, 2, -0.5], offsets=[0, n, 2 * n], shape=(n, n ** 2))
        block_n = scipy.sparse.diags([0.5, -2, 1.5], offsets=[n * (n - 3), n * (n - 2), n * (n - 1)], shape=(n, n ** 2))
        block = scipy.sparse.diags([-0.5, 0.5], offsets=[0, 2 * n], shape=(n * (n - 2), n ** 2))
        D = scipy.sparse.vstack((block_1, block, block_n))

    elif difference_form == 'order-2_0_bc':
        D = scipy.sparse.diags([[-0.5], [0.5]], offsets=[-n, n], shape=(n ** 2, n ** 2), dtype=np.float32)

    elif difference_form == 'order-6':
        block_border_up = scipy.sparse.diags([[- 49 / 20], [6], [-15 / 2], [20 / 3], [-15 / 4], [6 / 5], [-1 / 6]],
                                             offsets=np.arange(0, 7) * n, shape=(3 * n, n ** 2))
        diag_block = scipy.sparse.diags([[-1 / 60], [3 / 20], [-3 / 4], [0], [3 / 4], [-3 / 20], [1 / 60]],
                                        offsets=np.arange(0, 7) * n, shape=(n ** 2 - 6 * n, n ** 2))
        block_border_down = scipy.sparse.diags([[1 / 6], [-6 / 5], [15 / 4], [-20 / 3], [15 / 2], [-6], [49 / 20]],
                                               offsets=np.arange(n - 2 - 7, n - 2) * n, shape=(3 * n, n ** 2))
        D = scipy.sparse.vstack([block_border_up, diag_block, block_border_down])

    # elif difference_form == 'order-6':
    #     Laplacian = 1 / 6 * np.array([[1, 4, 1],
    #                                   [4, -20, 4],
    #                                   [1, 4, 1]])

    if circle_process:
        return (D.tocsr())[:, ~circle_process()][~circle_process(), :]
    return D.tocsr()


def laplacian_times_x(_input, _mask_out_circle=None):
    """
    Calculate the 2nd order derivative of x
    Laplacian_kernel = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]])
    :param _input: vectorized x (along column), shape = (n**2,)
    :return: vectorized 2nd order derivative of x
    """

    Laplacian = np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]])

    Laplacian_6 = np.array([[0, 0, 0, 1 / 90, 0, 0, 0],
                            [0, 0, 0, -3 / 20, 0, 0, 0],
                            [0, 0, 0, 3 / 2, 0, 0, 0],
                            [1 / 90, -3 / 20, 3 / 2, -49 / 9, 3 / 2, -3 / 20, 1 / 90],
                            [0, 0, 0, 3 / 2, 0, 0, 0],
                            [0, 0, 0, -3 / 20, 0, 0, 0],
                            [0, 0, 0, 1 / 90, 0, 0, 0]])

    Laplacian = 1 / 6 * np.array([[1, 4, 1],
                                  [4, -20, 4],
                                  [1, 4, 1]])

    if _mask_out_circle is not None:
        # 补0
        x_full = np.zeros_like(_mask_out_circle, dtype=np.float64)
        x_full[~_mask_out_circle] = _input
        return (scipy.ndimage.correlate(x_full.reshape(int(np.sqrt(x_full.shape)), -1),
                                        Laplacian, mode='wrap').ravel())[~_mask_out_circle]

    else:
        x_full = _input
        return scipy.ndimage.correlate(x_full.reshape(int(np.sqrt(x_full.shape)), -1), Laplacian, mode='wrap').ravel()


def threshold_mask(data, threshold_value: float = 1e-3):
    mask = np.ones(data.shape)
    mask[data < threshold_value] = 0
    return mask


def out_circle_mask(resolution, radius):
    x = y = np.arange(0, resolution)
    cx = cy = resolution // 2
    mask = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 > radius ** 2
    return mask.reshape(-1, 1).squeeze()


def plot_opt_result(phi_rec, phi_gt, save_path=None, file_name=None, resolution=256):
    fig, ax = plt.subplots(nrows=3, ncols=1, constrained_layout=True, dpi=300, figsize=(4, 10), )
    ax = ax.flatten()

    sub0 = ax[0].imshow(phi_gt.reshape(resolution, resolution), cmap=cm.jet)
    ax[0].set_title(r'$\xi$ (DNS)', fontsize=8)
    sub0.set_clim(0, 1)

    sub1 = ax[1].imshow(phi_rec.reshape(resolution, resolution), cmap=cm.jet)
    ax[1].set_title(r'$\xi$ (Rec.)', fontsize=8)
    sub1.set_clim(0, 1)

    error = phi_rec.reshape(resolution, resolution) - phi_gt.reshape(resolution, resolution)
    sub2 = ax[2].imshow(error.reshape(resolution, resolution), cmap=cm.RdBu_r)
    ax[2].set_title(f'Error (Rec. - DNS)\n'
                    f'rel. norm error ={((np.linalg.norm(phi_rec.reshape(resolution, resolution)) - np.linalg.norm(phi_gt.reshape(resolution, resolution))) / np.linalg.norm(phi_gt + 1e-8)):.4e}',
                    fontsize=8)
    sub2.set_clim(-0.3, 0.3)

    cb1 = fig.colorbar(sub1, ax=ax[:1], )
    cb2 = fig.colorbar(sub2, ax=ax[-1], )

    ax[0].set_aspect('auto')
    ax[1].set_aspect('auto')
    ax[2].set_aspect('auto')

    if save_path and file_name:
        plt.savefig(os.path.join(save_path, file_name))
        plt.close()
    else:
        plt.show()


def dphi_dt(u, v, D, delta_x, difference_form='order-2', **kwargs):
    """
    calculate dphi/dt for convection-diffusion equation
    :param D: diffusion coefficient
    :param u: x_velocity (unit:pixel), shape: n,n
    :param v: y_velocity (unit:pixel), shape: n,n
    :param vectorize: return vectorized scalar field or not

    :return: function 'return_dphi_dt'
    """

    def return_dphi_dt(t, phi):
        """
        :param t:
        :param phi: vectorized scalar field
        :return: scalar field at next time step, shape: n * n
        """

        resolution = int(np.sqrt(phi.shape[0]))

        L = laplacian_matrix_2nd(resolution, difference_form=difference_form) / delta_x ** 2
        Dx = differential_matrix_x(resolution, difference_form=difference_form) / delta_x
        Dy = differential_matrix_y(resolution, difference_form=difference_form) / delta_x

        diffusion_term = D * L @ phi
        convection_term = u * (Dx @ phi) + v * (Dy @ phi)
        return (diffusion_term - convection_term).ravel()

    return return_dphi_dt


def runge_kutta_4th(phi_n, delta_t, f_phi_n):
    # Heun
    # k_1 = phi_n + 1 / 3 * delta_t * f_phi_n(0, phi_n)
    # k_2 = phi_n + 2 / 3 * delta_t * f_phi_n(0, k_1)
    # return 1 / 4 * (phi_n + 3 * k_1) + 3 / 4 * delta_t * f_phi_n(0, k_2)

    # TVD
    # k_1 = phi_n + delta_t * f_phi_n(0, phi_n)
    # k_2 = 3 / 4 * phi_n + 2 / 4 * k_1 + 1 / 4 * delta_t * f_phi_n(0, k_1)
    # return 1 / 3 * phi_n + 2 / 3 * k_2 + 2 / 3 * delta_t * f_phi_n(0, k_2)

    # 4-order
    k_1 = phi_n + 1 / 2 * delta_t * f_phi_n(0, phi_n)
    k_2 = phi_n + 1 / 2 * delta_t * f_phi_n(0, k_1)
    k_3 = phi_n + delta_t * f_phi_n(0, k_2)
    return 1 / 3 * (-phi_n + k_1 + 2 * k_2 + k_3) + 1 / 6 * delta_t * f_phi_n(0, k_3)

    # # 4 - order
    # k_1 = phi_n
    # k_2 = f_phi_n(0, phi_n + delta_t / 2 * k_1)
    # k_3 = f_phi_n(0, phi_n + delta_t / 2 * k_2)
    # k_4 = f_phi_n(0, phi_n + delta_t * k_3)
    # return phi_n + delta_t / 6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)


def rk4_forward_mat(u, v, D=1 / 3000, delta_t=1e-2, delta_x=1, difference_form='order-2', inverse_direction=False):
    """
    :param u: vectorized u
    :param v: vectorized v
    :param D: diffusion coefficient, = Peclet number ^ -1
    :param delta_t: time step
    :param delta_x:
    :param difference_form:
    :param inverse_direction: whether inverse integral direction
    :return: forward matrix, shape = (n ** 2, n ** 2)
    """

    resolution = int(np.sqrt(u.shape[0]))

    u = scipy.sparse.diags_array(u)
    v = scipy.sparse.diags_array(v)

    L = laplacian_matrix_2nd(resolution, difference_form=difference_form) / delta_x ** 2
    Dx = differential_matrix_x(resolution, difference_form=difference_form) / delta_x
    Dy = differential_matrix_y(resolution, difference_form=difference_form) / delta_x
    A = (D * L - u @ Dx - v @ Dy)

    return (scipy.sparse.eye(resolution * resolution)
            + delta_t * A
            + 1 / 2 * delta_t ** 2 * (A @ A)
            + 1 / 6 * delta_t ** 3 * (A @ A @ A)
            + 1 / 24 * delta_t ** 4 * (A @ A @ A @ A))


def forward_upwind(func_dphi_dt, delta_t):
    return lambda phi: phi + delta_t * func_dphi_dt(0, phi)
