import numpy as np
import matplotlib.pyplot as plt
import scipy.io


def product_error(proj, w, phi, num_theta, num_proj, show=True):
    """
    to calculate the error between W @ Phi and Proj
    :param show: if True, NO figs plotted, just show
    :return: time_error, relative_time_error
    example: _, _ = product_error(proj256, w256, phi256, num_theta, num_proj, show=True)
    """
    time_error = (w.reshape(num_theta, num_proj, -1) @ phi.reshape(-1, 1) - proj).squeeze()
    rel_time_error = time_error / (proj.squeeze() + 1e-8)

    fig1, axs = plt.subplots(1, 2, figsize=(12, 8), )
    sub1 = axs[0].imshow(time_error, cmap='RdBu_r')
    axs[0].set_xlabel('proj')
    axs[0].set_ylabel('theta')
    axs[0].set_title('Error of (W @ Phi) - proj')

    sub2 = axs[1].imshow(rel_time_error, cmap='RdBu_r', clim=[-1, 1])
    axs[1].set_xlabel('proj')
    axs[1].set_ylabel('theta')
    axs[1].set_title('Relative error of (W @ Phi) - proj')

    fig1.suptitle(f'Resolution = {phi.shape[0]}')

    cb1 = fig1.colorbar(sub1, ax=[axs[0]], location="bottom")
    cb2 = fig1.colorbar(sub2, ax=[axs[1]], location="bottom")

    if show:
        plt.show()
    else:
        plt.savefig(f'product_error_{phi.shape[0]}.png')
    plt.close()

    return time_error, rel_time_error
