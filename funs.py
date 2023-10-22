from functools import cache
import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.ndimage


@cache  # only execute once for same input n
def laplacian_matrix_2nd(n):
    """
    Calculate the 2nd order Laplacian matrix
    Laplacian_kernel = np.array([[1, 1, 1],
                                 [1, -8, 1],
                                 [1, 1, 1]])
    :param n: squared dimension of the vectorized x (along column)
    :return:n^2 * n^2 matrix
    """

    eye = scipy.sparse.eye(n, dtype=np.int8)
    eye_k1 = scipy.sparse.eye(n, k=1, dtype=np.int8)

    D = scipy.sparse.diags(-4 * np.ones(n), 0, dtype=np.int8) + eye_k1 + eye_k1.T

    left = scipy.sparse.kron(D, eye) + scipy.sparse.kron(eye, D)

    right = scipy.sparse.kron(D, (eye_k1 + eye_k1.T))
    right -= scipy.sparse.kron(eye, ((-4 * eye_k1) + (-4 * eye_k1.T)))

    L = left + right
    return L.toarray()


def Laplacian_times_x(_input):
    """
    Calculate the 2nd order derivative of x
    Laplacian_kernel = np.array([[1, 1, 1],
                                 [1, -8, 1],
                                 [1, 1, 1]])
    :param _input: vectorized x (along column), shape = (n**2,)
    :return: vectorized 2nd order derivative of x
    """
    Laplacian = np.array([[1, 1, 1],
                          [1, -8, 1],
                          [1, 1, 1]])
    return scipy.ndimage.correlate(_input.reshape(int(np.sqrt(_input.shape)), -1), Laplacian, mode='constant').ravel()
