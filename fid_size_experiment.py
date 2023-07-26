import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import numpy as np

sizes = [500,  2000,  4000,  6000,  8000, 10001, 15000,  1000, 12500, 20000, 50000]

from zijing_main import *

def get_fid(n):
    data = np.load('/vol/bitbucket/fms119/score_sde_pytorch/'
                           'samples/cirfar10_true_10000.npz')
    data_samples = data['images'].transpose(0,3,1,2)
    data_samples = np.interp(data_samples, (data_samples.min(), data_samples.max()), (0, 1))

    file_path = '/vol/bitbucket/fms119/score_sde_pytorch/samples/cifar10_true_trials/cirfar10_true_50000.npz'
    data = np.load(file_path)
    gen_samples = data['images'][:n].transpose(0,3,1,2)
    gen_samples = np.interp(gen_samples, (gen_samples.min(), gen_samples.max()), (0, 1))    
    fid = compute_fid_nchw(data_samples, gen_samples)
    return fid

fid_trails = np.zeros(len(sizes))

for i, n in enumerate(sizes):
    fid = get_fid(n)    
    fid_trails[i] = fid
    # You should always be saving progess during experiments like this.
    np.savez('/vol/bitbucket/fms119/score_sde_pytorch/samples/cifar10_true_trials/50k_train_vs_n_train.npz', fid_trails=fid_trails)