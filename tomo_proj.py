import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy import sparse

import debug_funs


def divide_angle(num_theta):
    angle_interval = 360 / num_theta
    angles = np.arange(angle_interval / 2, angle_interval / 2 + 360, angle_interval) % 360
    return angles


def tomo_project(im, num_theta, num_proj, ts):
    """
    To calculate line integration along certain angles
    unit of angles: Degree;
    ts: threshold, the value of data which is less than threshold will not be calculated;
    projection direction: When angle == 0, the line passes through vertical (up-down) direction at the central location.

    return:
    proj_mat: projection matrix, shape = (num_theta, num_proj)
    weight: weight matrix, shape = (num_theta, num_proj, resolution, resolution)

    Example usage:
    Replace this with your image array and appropriate parameters
    im = ...
    num_theta = ...
    num_proj = ...
    ts = ...
    proj_mat, weight = tomo_project(im, num_theta, num_proj, ts)
    """

    im[im < ts] = 0

    domain_length = 2 * np.pi

    resolution = im.shape[0]
    angles = divide_angle(num_theta)  # generate angles
    x = np.arange(resolution + 1)  # X-range
    y = np.arange(resolution + 1)  # Y-range

    proj_mat = np.zeros((num_theta, num_proj))
    weight = np.zeros((num_theta, num_proj, resolution, resolution))

    lxmb = lambda x, mb: mb[0] * x + mb[1]  # Line equation: y = m*x+b

    for theta in range(num_theta):
        theta_rad = np.deg2rad(angles[theta])  # rotate counterclockwise

        d_bound = -(resolution * np.tan(theta_rad))  # maybe low bound of d
        interval_len = resolution + np.abs(d_bound)
        bound = np.sort([d_bound, 0, d_bound + resolution, resolution])
        dd = np.arange(bound[0], bound[-1], interval_len / (num_proj + 1))[1::]  # position of each bin

        for proj in range(num_proj):
            m = np.tan(theta_rad)  # Slope (or slope array)
            b = dd[proj]  # Intercept (or intercept array)
            mb = np.array([m, b])  # Array of [slope intercept] values

            hix = lambda y, mb: np.vstack([(y - mb[1]) / mb[0], y])  # 落在y横轴上的x值
            vix = lambda x, mb: np.vstack([x, lxmb(x, mb)])  # 落在x竖轴上的y值

            vrt = vix(y, mb).T  # [X Y] Array of vertical intercepts
            if not np.abs(mb[0]) < 1e-15:
                hrz = hix(x, mb).T  # [X Y] Array of horizontal intercepts
            else:
                hrz = np.array([[-1, -1]])  # line is horizontal, take any points out of scope
                vrt[:, 1] = mb[1]  # For numerical precision

            hvix = np.vstack([hrz, vrt])  # Concatenate ‘hrz’ and ‘vrt’ arrays
            exbdx = np.where((hvix[:, 0] < 0) | (hvix[:, 0] > resolution))[0]
            hvix = np.delete(hvix, exbdx, axis=0)
            exbdy = np.where((hvix[:, 1] < 0) | (hvix[:, 1] > resolution))[0]
            hvix = np.delete(hvix, exbdy, axis=0)
            srtd = np.unique(hvix, axis=0)  # Remove repeats and sort ascending by ‘x’

            for w in range(srtd.shape[0] - 1):
                dx = srtd[w, 0] - srtd[w + 1, 0]
                dy = srtd[w, 1] - srtd[w + 1, 1]
                weight_temp = np.sqrt(dx ** 2 + dy ** 2) * (domain_length / resolution)
                x_grid = int(min(np.floor(srtd[w, 0]), np.floor(srtd[w + 1, 0])))
                y_grid = int(min(np.floor(srtd[w, 1]), np.floor(srtd[w + 1, 1])))
                # proj_mat[theta, proj] += weight_temp * im[y_grid, x_grid]  # coordinate is inverse of matrix index
                weight[theta, proj, y_grid, x_grid] = weight_temp

            # Test line and weight
            # fig, ax = plt.subplots()
            # ax.plot(np.tile(x, (2, 1)), [0, resolution], 'k')  # Vertical gridlines
            # ax.plot([0, resolution], np.tile(y, (2, 1)), 'k')  # Horizontal gridlines
            # ax.plot(srtd[:, 0], srtd[:, 1], 'darkred')  # line
            # plt.imshow(np.flip(weight[theta][proj], axis=0), interpolation='None',
            #            extent=(0, resolution, 0, resolution), cmap='Blues', clim=(0, 1))
            # plt.colorbar()
            # for t in range(num_theta):
            #     for p in range(num_proj):
            #         # Scatter plot for intercept points
            #         ax.scatter(srtd[:, 0], srtd[:, 1], color='red', marker='x', s=6)
            #
            # ax.set_aspect('equal')
            # plt.savefig(f'theta{theta}_proj{proj}.png')
            # # plt.show()
            # plt.close()

            # Plot image and line
            # fig, ax = plt.subplots()
            # ax.plot(srtd[:, 0], srtd[:, 1], 'darkred')  # line
            # plt.imshow(im, interpolation='None',
            #            extent=(0, resolution, 0, resolution), cmap='jet', clim=(0, 1))
            # plt.colorbar()
            # ax.set_aspect('equal')
            # plt.savefig(f'im_theta{theta}_proj{proj}.png')
            # # plt.show()
            # plt.close()

            proj_mat = weight.reshape(num_theta, num_proj, -1) @ im.reshape(-1, 1)

    return proj_mat.squeeze(), weight


def _weights(x, dx=1, orig=0):
    x = np.ravel(x)
    floor_x = np.floor((x - orig) / dx).astype(np.int64)
    alpha = (x - orig - floor_x * dx) / dx
    return np.hstack((floor_x, floor_x + 1)), np.hstack((1 - alpha, alpha))


def _generate_center_coordinates(l_x):
    X, Y = np.mgrid[:l_x, :l_x].astype(np.float64)
    center = l_x / 2.0
    X += 0.5 - center
    Y += 0.5 - center
    return X, Y


def build_projection_operator(l_x, n_dir):
    """Compute the tomography design matrix.

    Parameters
    ----------

    l_x : int
        linear size of image array

    n_dir : int
        number of angles at which projections are acquired.

    Returns
    -------
    p : sparse matrix of shape (n_dir l_x, l_x**2)
    """
    X, Y = _generate_center_coordinates(l_x)
    angles = np.linspace(0, np.pi, n_dir, endpoint=False)
    data_inds, weights, camera_inds = [], [], []
    data_unravel_indices = np.arange(l_x ** 2)
    data_unravel_indices = np.hstack((data_unravel_indices, data_unravel_indices))
    for i, angle in enumerate(angles):
        Xrot = np.cos(angle) * X - np.sin(angle) * Y
        inds, w = _weights(Xrot, dx=1, orig=X.min())
        mask = np.logical_and(inds >= 0, inds < l_x)
        weights += list(w[mask])
        camera_inds += list(inds[mask] + i * l_x)
        data_inds += list(data_unravel_indices[mask])
    proj_operator = sparse.coo_matrix((weights, (camera_inds, data_inds)))
    return proj_operator


if __name__ == '__main__':
    # im = np.ones([8, 8])
    # for i in range(1, 9):
    #     im[i - 1] *= i
    #
    # [proj8, w8] = tomo_project(im, 3, 3, 0.001)

    mat_file_path_256 = '/Users/xiangyu.nie/Document/PhD/TOMO_data-20230908/DATA256_Re3000_Sc1/Scalar/run01_10000_S.mat'
    mat_file_path_512 = '/Users/xiangyu.nie/Document/PhD/TOMO_data-20230908/DATA512_Re3000_Sc1/Scalar/run01_10000_S.mat'
    mat_file_path_1024 = '/Users/xiangyu.nie/Document/PhD/TOMO_data-20230908/DATA1024_Re3000_Sc1/Scalar/run01_10000_S.mat'

    phi256 = scipy.io.loadmat(mat_file_path_256)['Phi']
    phi512 = scipy.io.loadmat(mat_file_path_512)['Phi']
    phi1024 = scipy.io.loadmat(mat_file_path_1024)['Phi']

    num_theta = 16
    num_proj = 16
    ts = 0.001

    [proj256, w256] = tomo_project(phi256, num_theta, num_proj, 0.001)
    [proj512, w512] = tomo_project(phi512, num_theta, num_proj, 0.001)
    [proj1024, w1024] = tomo_project(phi1024, num_theta, num_proj, 0.001)

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [2, 1, 1]})

    # Subplot 1
    axs[0].plot(proj1024.T.ravel(), 'b', linewidth=1, label='1024')  # proj first and then theta
    axs[0].plot(proj512.T.ravel(), 'g', linewidth=1, label='512')
    axs[0].plot(proj256.T.ravel(), 'r', linewidth=1, label='256')
    axs[0].set_xlim([0, 256])
    axs[0].set_title('Projection')
    axs[0].legend()
    axs[0].grid(True)

    # Subplot 2
    axs[1].plot(proj256.T.ravel() - proj512.T.ravel(), 'r', linewidth=1, label='256 - 512')
    axs[1].plot(proj256.T.ravel() - proj1024.T.ravel(), 'b', linewidth=1, label='256 - 1024')
    axs[1].set_xlim([0, 256])
    axs[1].set_ylim([-0.15, 0.15])
    axs[1].set_title('Error')
    axs[1].legend()
    axs[1].grid(True)

    # Subplot 3
    axs[2].plot((proj256.T.ravel() - proj1024.T.ravel()) / (proj1024.T.ravel() + 1e-3), 'r', linewidth=1,
                label='256 - 1024')
    axs[2].set_xlim([0, 256])
    axs[2].set_ylim([-0.15, 0.15])
    axs[2].set_title('Relative Error')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()
