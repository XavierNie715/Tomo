from tomo_proj import *
import numpy as np
import scipy
import scipy.io
import os

"""
Save .npy file for 256*256 data

Return:
data_256.npy, including dict {'proj': proj, 'weight': weight, 'phi': phi}
proj: shape = (len(mat_profiles), num_theta * num_theta)
weight: shape = (len(mat_profiles), num_theta * num_theta, 256 * 256)
phi: shape = (len(mat_profiles), 256 * 256)
"""


def process_file(path, _num_theta, _num_proj, _ts):
    _phi_temp = scipy.io.loadmat(path)['Phi']
    _proj_temp, _w_temp = tomo_project(_phi_temp, _num_theta, _num_proj, _ts)
    return _proj_temp.ravel(), _w_temp.reshape(_num_theta * _num_proj, -1), _phi_temp.ravel()


if __name__ == '__main__':
    num_theta = 16
    num_proj = 16
    ts = 0.001
    resolution = 256

    mat_file_path = '/Users/xiangyu.nie/Document/PhD/TOMO_data-20230908/DATA256_Re3000_Sc1/Scalar/'
    out_path = './data_cache/TOMO_data-20230908/DATA256_Re3000_Sc1/Scalar/'
    os.makedirs(out_path, exist_ok=True)

    mat_files = [file for file in os.listdir(mat_file_path) if file.endswith(".mat")]
    mat_files.sort(key=lambda x: int(x.split('_')[1]), reverse=True)

    for i, mat_file in enumerate(mat_files):
        file_path = os.path.join(mat_file_path, mat_file)
        proj_temp, w_temp, phi_temp = process_file(file_path, num_theta, num_proj, ts)
        results = {'proj': proj_temp, 'weight': w_temp, 'phi': phi_temp}
        np.save(os.path.join(out_path, mat_file.split('.')[0] + '.npy'), results)

        print(f'Finish {i} out of {len(mat_files)}')
