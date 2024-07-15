from functools import cache
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io
from scipy import ndimage

import debug_funs


class TomoProjection:
    def __init__(self, _im, _num_theta, _num_proj, _ts=1e-3, _radius=120):
        self.im = np.where(_im < _ts, 0, _im)
        self.num_theta = _num_theta
        self.num_proj = _num_proj
        self.ts = _ts

        self.domain_length = 2 * np.pi

        self.resolution = self.im.shape[0]
        self.radius = _radius / 256 * self.resolution
        self.cx = self.cy = self.resolution // 2  # center point

        self.angles = self.divide_angle(self.num_theta)  # generate angles

        self.index_transform = self.transform_index((self.num_theta * self.num_proj, self.resolution * self.resolution),
                                                    (self.num_theta, self.num_proj, self.resolution, self.resolution))
        self.x = np.arange(self.resolution + 1)  # X-range
        self.y = np.arange(self.resolution + 1)  # Y-range
        lxmb = lambda x, mb: mb[0] * x + mb[1]  # Line equation: y = m*x+b

        self.index_transform = self.transform_index((self.resolution * self.resolution),
                                                    (self.resolution, self.resolution))
        # self.weights, self.intersection_points = zip(*(Parallel(n_jobs=4)(
        #     delayed(self.calculate_weight)(theta_index, proj_index, lxmb, )
        #     for theta_index in range(self.num_theta) for proj_index in range(self.num_proj))))
        weights_temp, coord, self.intersection_points_temp, line_spacing = zip(
            *[self.calculate_weight(theta_index, proj_index, lxmb)
              for theta_index in range(self.num_theta)
              for proj_index in range(self.num_proj)])
        self.line_spacing = line_spacing[0]
        self.weight = sum([scipy.sparse.coo_matrix((weights_temp[i], ([i] * len(coord[i]), coord[i])),
                                                   shape=(self.num_theta * self.num_proj,
                                                          self.resolution * self.resolution))
                           for i in range(self.num_theta * self.num_proj)]).tocsr()
        self.proj_mat = (self.weight @ self.im.reshape(-1, 1)).squeeze()

    @staticmethod
    def divide_angle(_num_theta):
        angle_interval = 360 / _num_theta
        angles = np.arange(angle_interval / 2, angle_interval / 2 + 360, angle_interval) % 360
        return angles

    @staticmethod
    def transform_index(_shape_1d, _shape_2d):
        """
        Transform 2d index to 1d index
        :param _shape_1d: (resolution * resolution)
        :param _shape_2d: (resolution, resolution)

        :return: (x_index, y_index) -> (x_index * y_index)
        """

        def return_index(_index_2d):
            _index_1d = np.unravel_index(np.ravel_multi_index(_index_2d, _shape_2d), _shape_1d)
            return _index_1d

        return return_index

    def calculate_weight(self, theta_index, proj_index, lxmb):
        weight = []
        coord = []  # corresponding coordinates of weight
        length_per_pixel = self.domain_length / self.resolution
        theta_rad = np.deg2rad(self.angles[theta_index])
        m = np.tan(theta_rad)
        if self.radius:
            d_center = -m * self.cx + self.cy
            interval_len = np.abs(2 * (self.radius / np.cos(theta_rad)))
            d_center_low = d_center - 1 / 2 * interval_len
            d_center_up = d_center + 1 / 2 * interval_len
            k = interval_len / (2 * self.num_proj)
            dd = np.linspace(k + d_center_low, d_center_up - k, self.num_proj)
        else:
            d_bound = -(self.resolution * np.tan(theta_rad))
            interval_len = self.resolution + np.abs(d_bound)
            bound = np.sort([d_bound, 0, d_bound + self.resolution, self.resolution])
            dd = np.arange(bound[0], bound[-1], interval_len / (self.num_proj + 1))[1::]

        b = dd[proj_index]
        mb = np.array([m, b])
        hix = lambda y, mb: np.vstack([(y - mb[1]) / mb[0], y])
        vix = lambda x, mb: np.vstack([x, lxmb(x, mb)])
        vrt = vix(self.y, mb).T
        if not np.abs(mb[0]) < 1e-15:
            hrz = hix(self.x, mb).T
        else:
            hrz = np.array([[-1, -1]])
            vrt[:, 1] = mb[1]
        hvix = np.vstack([hrz, vrt])
        exbdx = np.where((hvix[:, 0] < 0) | (hvix[:, 0] > self.resolution))[0]
        hvix = np.delete(hvix, exbdx, axis=0)
        exbdy = np.where((hvix[:, 1] < 0) | (hvix[:, 1] > self.resolution))[0]
        hvix = np.delete(hvix, exbdy, axis=0)
        srtd = np.unique(hvix, axis=0)

        # vertical line on border
        if np.abs(mb[0]) > 1e15:
            if False not in [np.isclose(srtd[i, 0], round(srtd[i, 0]), atol=1e-15) for i in
                             range(srtd.shape[0] - 1)]:
                weight_temp = (1.0 * length_per_pixel) / 2
                try:
                    x_grid_left = round(srtd[0, 0] - 1)
                    x_grid_right = round(srtd[0, 0])
                    for y_grid in range(self.resolution):
                        weight.append(weight_temp)
                        coord.append(*(self.index_transform((y_grid, x_grid_left))))  # coordinate: inverse array index
                        weight.append(weight_temp)
                        coord.append(*(self.index_transform((y_grid, x_grid_right))))
                except:
                    pass
        # horizontal line on border
        elif np.abs(mb[0]) < 1e-15:
            if False not in [np.isclose(srtd[i, 1], round(srtd[i, 1]), atol=1e-15) for i in
                             range(srtd.shape[0] - 1)]:
                weight_temp = (1.0 * length_per_pixel) / 2
                try:
                    y_grid_up = round(srtd[0, 1])
                    y_grid_down = y_grid_up - 1
                    for x_grid in range(self.resolution):
                        weight.append(weight_temp)
                        coord.append(*(self.index_transform((y_grid_up, x_grid))))
                        weight.append(weight_temp)
                        coord.append(*(self.index_transform((y_grid_down, x_grid))))
                except:
                    pass
        else:
            for w in range(srtd.shape[0] - 1):
                dx = srtd[w, 0] - srtd[w + 1, 0]
                dy = srtd[w, 1] - srtd[w + 1, 1]
                weight_temp = np.sqrt(dx ** 2 + dy ** 2) * length_per_pixel
                x_grid = int(min(np.floor(srtd[w, 0]), np.floor(srtd[w + 1, 0])))
                y_grid = int(min(np.floor(srtd[w, 1]), np.floor(srtd[w + 1, 1])))
                # proj_mat[theta, proj] += weight_temp * im[y_grid, x_grid]  # coordinate is inverse of matrix index
                weight.append(weight_temp)
                coord.append(*(self.index_transform((y_grid, x_grid))))
        line_spacing = (dd[1] - dd[0]) * np.cos(theta_rad)
        return weight, coord, srtd, line_spacing

    # todo
    # def test_line_weight(self, ):
    #     fig, ax = plt.subplots()
    #     ax.plot(np.tile(self.x, (2, 1)), [0, self.resolution], 'k')  # Vertical gridlines
    #     ax.plot([0, self.resolution], np.tile(self.y, (2, 1)), 'k')  # Horizontal gridlines
    #     ax.plot(self.intersection_points_temp[:, 0], self.intersection_points_temp[:, 1], 'darkred')  # line
    #     # plt.imshow(np.flip(weight[theta][proj], axis=0), interpolation='None',
    #     #            extent=(0, resolution, 0, resolution), cmap='Blues', clim=(0, 1))
    #     plt.imshow(np.flip(self.weight[theta][proj] / (domain_length / resolution), axis=0), interpolation='None',
    #                extent=(0, self.resolution, 0, self.resolution), cmap='hot', clim=(0, 1.5))
    #     plt.colorbar()
    #     for t in range(self.num_theta):
    #         for p in range(self.num_proj):
    #             # Scatter plot for intercept points
    #             ax.scatter(self.srtd[:, 0], self.srtd[:, 1], color='red', marker='x', s=6)
    #
    #     ax.set_aspect('equal')
    #     plt.savefig(f'theta{theta}_proj{proj}.png')
    #     # plt.show()
    #     plt.close()

    def plot_line_on_im(self):
        # # Plot image and line
        fig, ax = plt.subplots()
        for plot_index in range(self.num_theta * self.num_proj):
            ax.plot(self.intersection_points_temp[plot_index][:, 0],
                    self.intersection_points_temp[plot_index][:, 1],
                    'darkred', linewidth=0.5)  # line
        plt.imshow(self.im, interpolation='None',
                   extent=(0, self.resolution, 0, self.resolution), cmap='jet', clim=(0, 1))
        plt.colorbar()
        ax.set_aspect('equal')
        plt.savefig(f'./figs/in&line/im_theta{self.num_theta}_proj{self.num_proj}.png', dpi=300)
        plt.show()
        plt.close()

        fig, ax = plt.subplots()
        for plot_index in range(self.num_theta * self.num_proj):
            ax.plot(self.intersection_points_temp[plot_index][:, 0],
                    self.intersection_points_temp[plot_index][:, 1],
                    'darkred', linewidth=0.5)  # line
        plt.imshow(np.zeros_like(self.im), interpolation='None',
                   extent=(0, self.resolution, 0, self.resolution), cmap='Greys')
        ax.set_aspect('equal')
        plt.savefig(f'./figs/in&line/im_theta{self.num_theta}_proj{self.num_proj}_0.png', dpi=300)
        plt.show()
        plt.close()


if __name__ == '__main__':
    mat_file_path_256 = '/Users/xiangyu.nie/Document/PhD/ScalarData-20231116/Frame4900_256x256.mat'
    mat_file_path_4096 = '/Users/xiangyu.nie/Document/PhD/ScalarData-20231116/Frame4900_4096x4096.mat'

    phi256 = scipy.io.loadmat(mat_file_path_256)['Phi']
    phi4096 = scipy.io.loadmat(mat_file_path_4096)['Phi']
    phi512_down = ndimage.zoom(phi4096, 1/8, mode='wrap')
    phi512_up = ndimage.zoom(phi256, 2, mode='wrap')

    num_theta = 16
    num_proj = 16
    ts = 0.001

    tomo256 = TomoProjection(phi256, num_theta, num_proj)
    tomo4096 = TomoProjection(phi4096, num_theta, num_proj)
    tomo512_down = TomoProjection(phi512_down, num_theta, num_proj)
    tomo512_up = TomoProjection(phi512_up, num_theta, num_proj)

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [2, 1, 1]})

    # Subplot 1
    axs[0].plot(tomo4096.proj_mat.ravel(), 'b', linewidth=1, label='4096')  # proj first and then theta
    # axs[0].plot(proj512.ravel(), 'g', linewidth=1, label='512')
    axs[0].plot(tomo256.proj_mat.ravel(), 'r', linewidth=1, label='256')
    axs[0].set_xlim([0, num_theta * num_proj])
    axs[0].set_title('Projection')
    axs[0].legend()
    axs[0].grid(True)

    # Subplot 2
    # axs[1].plot(proj256.ravel() - proj512.ravel(), 'r', linewidth=1, label='256 - 512')
    axs[1].plot(tomo256.proj_mat.ravel() - tomo4096.proj_mat.ravel(), 'b', linewidth=1, label='256 - 4096')
    # axs[1].plot(proj512.ravel() - proj1024.ravel(), 'g', linewidth=1, label='512 - 1024')
    axs[1].set_xlim([0, num_theta * num_proj])
    axs[1].set_ylim([-0.05, 0.05])
    axs[1].set_title('Error')
    axs[1].legend()
    axs[1].grid(True)

    # Subplot 3
    axs[2].plot((np.linalg.norm(tomo256.proj_mat.ravel()) - np.linalg.norm(tomo4096.proj_mat.ravel()))
                / np.linalg.norm(tomo4096.proj_mat.ravel() + 1e-9), 'r',
                linewidth=1,
                label='256 - 4096')
    axs[2].plot((np.linalg.norm(tomo512_down.proj_mat.ravel()) - np.linalg.norm(tomo4096.proj_mat.ravel()))
                / np.linalg.norm(tomo4096.proj_mat.ravel() + 1e-9), 'b',
                linewidth=1,
                label='512down - 4096')
    axs[2].plot((np.linalg.norm(tomo512_up.proj_mat.ravel()) - np.linalg.norm(tomo4096.proj_mat.ravel()))
                / np.linalg.norm(tomo4096.proj_mat.ravel() + 1e-9), 'g',
                linewidth=1,
                label='512up - 4096')
    axs[2].set_xlim([0, num_theta * num_proj])
    axs[2].set_ylim([-0.05, 0.05])
    axs[2].set_title('Relative Error (2-norm)')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()
