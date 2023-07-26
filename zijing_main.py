import torch
import numpy as np
from zijing_fid_compute import fid_score



def compute_fid(x_data, x_samples, use_cpu=False):

    assert type(x_data) == np.ndarray
    assert type(x_samples) == np.ndarray

    # RGB
    assert x_data.shape[3] == 3
    assert x_samples.shape[3] == 3

    # NHWC
    assert x_data.shape[1] == x_data.shape[2]
    assert x_samples.shape[1] == x_samples.shape[2]

    # [0,255]
    assert np.min(x_data) > 0.-1e-4
    assert np.max(x_data) < 255.+1e-4
    assert np.mean(x_data) > 10.

    # [0,255]
    assert np.min(x_samples) > 0.-1e-4
    assert np.max(x_samples) < 255.+1e-4
    assert np.mean(x_samples) > 1.

    if use_cpu:
        def create_session():
            import tensorflow.compat.v1 as tf
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.0
            config.gpu_options.visible_device_list = ''
            return tf.Session(config=config)
    else:
        def create_session():
            gpu_idx = 0
            import tensorflow.compat.v1 as tf
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.2
            config.gpu_options.visible_device_list = str(gpu_idx)
            return tf.Session(config=config)

    path = '/tmp'
    fid = fid_score(create_session, x_data, x_samples, path, cpu_only=use_cpu)

    return fid

def compute_fid_nchw(x_data, x_samples):

    to_nhwc = lambda x: np.transpose(x, (0, 2, 3, 1))

    x_data_nhwc = to_nhwc(255 * x_data)
    x_samples_nhwc = to_nhwc(255 * x_samples)

    fid = compute_fid(x_data_nhwc, x_samples_nhwc)

    return fid

def get_fid(file_path):
    data = np.load('/vol/bitbucket/fms119/score_sde_pytorch/'
                           'samples/cirfar10_true_10000.npz')
    data_samples = data['images'].transpose(0,3,1,2)
    data_samples = np.interp(data_samples, (data_samples.min(), data_samples.max()), (0, 1))

    data = np.load(file_path)
    gen_samples = data['images']
    gen_samples = np.interp(gen_samples, (gen_samples.min(), gen_samples.max()), (0, 1))

    fid = compute_fid_nchw(data_samples, gen_samples)
    return fid

if __name__ == '__main__':
    fid = get_fid()
    print(fid)
