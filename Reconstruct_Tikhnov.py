import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import scipy.sparse
import scipy.optimize

import debug_funs
import funs


def opt_fun(_weight, _lambda):
    """
    The objective function of the reconstruction problem
    ** The regularization term is the 2nd order derivative of phi,
    calculated by correlate with Laplacian_kernel = np.array([[1, 1, 1],
                                                              [1, -8, 1],
                                                              [1, 1, 1]])
    :param _weight: projection weight matrix, shape = (num_theta * num_theta, n**2),
    :param _lambda: regularization parameter,
    :return: lambda function, input shape = (n**2,), return shape = (1,): residual.
    """
    return lambda x: np.linalg.norm(_weight @ x - proj) ** 2 + _lambda * np.linalg.norm(funs.Laplacian_times_x(x)) ** 2


def opt_jac(_input, _weight, _lambda, ):
    """
    The analytical Jacobian of the objective function
    :param _input: vectorized phi (along column), shape = (n**2,)
    :param _weight: projection weight matrix, shape = (num_theta * num_theta, n**2)
    :param _lambda: regularization parameter
    :return: lambda function, input shape= (n**2, 1), return shape = (n**2,)
    """
    _laplacian = funs.laplacian_matrix_2nd(int(np.sqrt(_input.shape)))  # the 2nd order Laplacian matrix
    return lambda x: 2 * _weight.T @ (_weight @ x - proj) + 2 * _lambda * _laplacian.T @ (_laplacian @ x)


np.random.seed(42)

data = np.load('./data_cache/TOMO_data-20230908/DATA256_Re3000_Sc1/Scalar/run01_10000_S.npy',
               allow_pickle=True).item()
proj, weight, phi = data.values()
resolution = int(np.sqrt(phi.shape[0]))
lambda_re = 0.1
x0 = np.random.rand(phi.shape[0])

# Trust Region Reflective algorithm with lsmr for sparse weight (based on Golub-Kahan bidiagonalization process)
x_rec = scipy.optimize.lsq_linear(
    np.vstack((weight,
               lambda_re * funs.laplacian_matrix_2nd(resolution))),
    np.concatenate((proj, np.zeros(phi.shape[0]))),
    bounds=(0, 1),
    verbose=2, )

# x_rec = scipy.optimize.minimize(fun=opt_fun(weight, lambda_re), x0=
