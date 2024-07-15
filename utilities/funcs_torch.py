from functools import cache
from utilities import funcs
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.sparse
import scipy.ndimage
import os


def threshold_mask(data, threshold_value: float = 1e-3):
    mask = np.ones(data.shape)
    mask[data < threshold_value] = 0
    return mask


def out_circle_mask(resolution, radius):
    x = y = np.arange(0, resolution)
    cx = cy = resolution // 2
    mask = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 > radius ** 2
    return mask.reshape(-1, 1).squeeze()


def dphi_dt(u, v, D, delta_x, difference_form='order-2', backend='torch', **kwargs):
    """
    calculate dphi/dt for convection-diffusion equation
    :param D: diffusion coefficient
    :param u: x_velocity (unit:pixel), shape: n,n
    :param v: y_velocity (unit:pixel), shape: n,n
    :param vectorize: return vectorized scalar field or not

    :return: function 'return_dphi_dt'
    """

    def return_dphi_dt_torch(t, phi):
        resolution = int(np.sqrt(phi.shape[0]))
        diff_phi = Finite_difference_operator_torch(phi, delta_x, resolution)
        diffusion_term = D * diff_phi.laplacian_input().ravel()
        convection_term = u * diff_phi.diff_1d_x().ravel() + v * diff_phi.diff_1d_y().ravel()
        return (diffusion_term - convection_term).ravel()

    return return_dphi_dt_torch


def scipy_sparse2_torch_sparse(coo_array):
    """
    Convert scipy sparse matrix to torch sparse matrix
    :param coo_array: scipy sparse matrix
    """
    coo_sparse_scipy = coo_array.tocoo()
    return torch.sparse_coo_tensor(np.array(coo_sparse_scipy.nonzero()), coo_sparse_scipy.data, coo_sparse_scipy.shape)


def reshape_torch_fortran(x, shape):
    """
    Reshape torch tensor with fortran order
    :param x:
    :param shape:
    :return:
    """
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def rk4_forward_mat_torch(u, v, D=1 / 3000, delta_t=1e-2, delta_x=1, difference_form='central',
                          **kwargs):
    """
    :param u: vectorized u
    :param v: vectorized v
    :param D: diffusion coefficient, = Peclet number ^ -1
    :param delta_t: time step
    :return: forward matrix, shape = (n ** 2, n ** 2)
    """

    resolution = int(np.sqrt(u.shape[0]))

    u_diag = torch.sparse.spdiags(u, offsets=torch.tensor([0]), shape=(resolution ** 2, resolution ** 2))
    v_diag = torch.sparse.spdiags(v, offsets=torch.tensor([0]), shape=(resolution ** 2, resolution ** 2))

    if 'L' not in kwargs:
        L = funcs.laplacian_matrix_2nd(resolution, difference_form=difference_form) / delta_x ** 2
        L = scipy_sparse2_torch_sparse(L)
    else:
        L = kwargs['L']

    if 'Dx' not in kwargs:
        Dx = funcs.differential_matrix_x(resolution, difference_form=difference_form) / delta_x
        Dx = scipy_sparse2_torch_sparse(Dx)
    else:
        Dx = kwargs['Dx']

    if 'Dy' not in kwargs:
        Dy = funcs.differential_matrix_y(resolution, difference_form=difference_form) / delta_x
        Dy = scipy_sparse2_torch_sparse(Dy)
    else:
        Dy = kwargs['Dy']

    A = (D * L - u_diag @ Dx - v_diag @ Dy)

    return (torch.sparse.spdiags(torch.tensor([1] * resolution ** 2),
                                 offsets=torch.tensor([0]), shape=(resolution ** 2, resolution ** 2))
            + delta_t * A
            + 1 / 2 * delta_t ** 2 * (A @ A)
            + 1 / 6 * delta_t ** 3 * (A @ A @ A)
            + 1 / 24 * delta_t ** 4 * (A @ A @ A @ A))


# class Finite_difference_operator_torch:
#     '''
#     B.C.: d2z/dx2 = 0 => b.c. value in dz/dx
#     '''
#
#     def __init__(self, x, delta_x, reso=512, weights_1st=(-1, 0, 1), weights_2nd=(1, -2, 1)):
#         self.reso = reso
#         self.z = (x if torch.is_tensor(x) else torch.tensor(x)).view(self.reso, self.reso)
#         self.weights_1st = torch.tensor(weights_1st, dtype=torch.float64).view(1, 1, 1, 3) / (2 * delta_x)
#         self.weights_2nd = torch.tensor(weights_2nd, dtype=torch.float64).view(1, 1, 1, 3) / delta_x ** 2
#
#         # central difference
#         dx2 = torch.nn.Conv2d(1, 1, (1, 3), bias=False, padding=(0, 1))
#         dx2.weight = torch.nn.Parameter(self.weights_2nd, requires_grad=False)
#         d2zdx2 = dx2(self.z.view(1, 1, self.reso, self.reso)).view(self.reso, self.reso)
#
#         dy2 = torch.nn.Conv2d(1, 1, (3, 1), bias=False, padding=(1, 0))
#         dy2.weight = torch.nn.Parameter(self.weights_2nd.view(1, 1, 3, 1), requires_grad=False)
#         d2zdy2 = dy2(self.z.view(1, 1, self.reso, self.reso)).view(self.reso, self.reso)
#
#         # boundary: d2zdn2 = 0
#         laplacian_z = d2zdx2 + d2zdy2
#         laplacian_z[0, :] = 0
#         laplacian_z[-1, :] = 0
#         laplacian_z[:, 0] = 0
#         laplacian_z[:, -1] = 0
#         self.laplacian_input = laplacian_z
#
#         # calculate outer boundary with b.c.
#         z = torch.nn.functional.pad(self.z, (1, 1, 1, 1), mode='constant')
#         z[1:-1, 0] = 2 * self.z[:, 0] - self.z[:, 1]
#         z[1:-1, -1] = 2 * self.z[:, -1] - self.z[:, -2]
#         z[0, 1:-1] = 2 * self.z[0, :] - self.z[1, :]
#         z[-1, 1:-1] = 2 * self.z[-1, :] - self.z[-2, :]
#
#         self.z_padded = z.view(1, 1, self.reso + 2, self.reso + 2)
#
#     @cache
#     def diff_1d_x(self):
#         dx = torch.nn.Conv2d(1, 1, (1, 3), bias=False)
#         dx.weight = torch.nn.Parameter(self.weights_1st, requires_grad=False)
#         dzdx = dx(self.z_padded)
#         return dzdx[..., 1:-1, :].view(self.reso, self.reso)
#
#     @cache
#     def diff_1d_y(self):
#         dy = torch.nn.Conv2d(1, 1, (3, 1), bias=False)
#         dy.weight = torch.nn.Parameter(self.weights_1st.view(1, 1, 3, 1), requires_grad=False)
#         dzdy = dy(self.z_padded)
#         return dzdy[..., :, 1:-1].view(self.reso, self.reso)
#

class Finite_difference_operator_torch:
    '''
    B.C.: d2z/dx2 = 0 => b.c. value in dz/dx
    '''

    def __init__(self, x, delta_x, reso=512,
                 weights_1st=torch.tensor((-1, 0, 1), dtype=torch.float64),
                 weights_2nd=torch.tensor((1, -2, 1), dtype=torch.float64)):
        self.reso = reso
        self.z = (x if torch.is_tensor(x) else torch.tensor(x)).view(self.reso, self.reso)
        self.weights_1st = weights_1st.view(1, 1, 1, 3) / (2 * delta_x)
        self.weights_2nd = weights_2nd.view(1, 1, 1, 3) / delta_x ** 2
        self.delta_x = delta_x

    @cache
    def laplacian_input(self):
        # central difference
        dx2 = torch.nn.Conv2d(1, 1, (1, 3), bias=False, padding=(0, 1))
        dx2.weight = torch.nn.Parameter(self.weights_2nd, requires_grad=False)
        d2zdx2 = dx2(self.z.view(1, 1, self.reso, self.reso)).view(self.reso, self.reso)

        dy2 = torch.nn.Conv2d(1, 1, (3, 1), bias=False, padding=(1, 0))
        dy2.weight = torch.nn.Parameter(self.weights_2nd.view(1, 1, 3, 1), requires_grad=False)
        d2zdy2 = dy2(self.z.view(1, 1, self.reso, self.reso)).view(self.reso, self.reso)

        # boundary: forward / backward difference
        laplacian_z = d2zdx2 + d2zdy2
        laplacian_z[1:-1, 0] = ((self.z[1:-1, 0] - 2 * self.z[1:-1, 1] + self.z[1:-1, 2]) / self.delta_x ** 2
                                + d2zdy2[1:-1, 0])
        laplacian_z[1:-1, -1] = ((self.z[1:-1, -1] - 2 * self.z[1:-1, -2] + self.z[1:-1, -3]) / self.delta_x ** 2
                                 + d2zdy2[1:-1, -1])
        laplacian_z[0, 1:-1] = ((self.z[0, 1:-1] - 2 * self.z[1, 1:-1] + self.z[2, 1:-1]) / self.delta_x ** 2
                                + d2zdx2[0, 1:-1])
        laplacian_z[-1, 1:-1] = ((self.z[-1, 1:-1] - 2 * self.z[-2, 1:-1] + self.z[-3, 1:-1]) / self.delta_x ** 2
                                 + d2zdx2[-1, 1:-1])

        laplacian_z[0, 0] = (2 * self.z[0, 0] - 2 * self.z[0, 1] + self.z[0, 2]
                             - 2 * self.z[1, 0] + self.z[2, 0]) / self.delta_x ** 2
        laplacian_z[0, -1] = (2 * self.z[0, -1] - 2 * self.z[0, -2] + self.z[0, -3]
                              - 2 * self.z[1, -1] + self.z[2, -1]) / self.delta_x ** 2
        laplacian_z[-1, 0] = (2 * self.z[-1, 0] - 2 * self.z[-1, 1] + self.z[-1, 2]
                              - 2 * self.z[-2, 0] + self.z[-3, 0]) / self.delta_x ** 2
        laplacian_z[-1, -1] = (2 * self.z[-1, -1] - 2 * self.z[-1, -2] + self.z[-1, -3]
                               - 2 * self.z[-2, -1] + self.z[-3, -1]) / self.delta_x ** 2
        return laplacian_z

    def diff_1d_x(self):
        dx = torch.nn.Conv2d(1, 1, (1, 3), bias=False, padding=(0, 1))
        dx.weight = torch.nn.Parameter(self.weights_1st, requires_grad=False)
        dzdx = dx(self.z.view(1, 1, self.reso, self.reso)).view(self.reso, self.reso)

        # boundary: forward / backward three-point difference
        dzdx[:, 0] = (-3 * self.z[:, 0] + 4 * self.z[:, 1] - self.z[:, 2]) / (2 * self.delta_x)
        dzdx[:, -1] = (3 * self.z[:, -1] - 4 * self.z[:, -2] + self.z[:, -3]) / (2 * self.delta_x)
        return dzdx

    def diff_1d_y(self):
        dy = torch.nn.Conv2d(1, 1, (3, 1), bias=False, padding=(1, 0))
        dy.weight = torch.nn.Parameter(self.weights_1st.view(1, 1, 3, 1), requires_grad=False)
        dzdy = dy(self.z.view(1, 1, self.reso, self.reso)).view(self.reso, self.reso)

        # boundary: forward / backward three-point difference
        dzdy[0, :] = (-3 * self.z[0, :] + 4 * self.z[1, :] - self.z[2, :]) / (2 * self.delta_x)
        dzdy[-1, :] = (3 * self.z[-1, :] - 4 * self.z[-2, :] + self.z[-3, :]) / (2 * self.delta_x)
        return dzdy
