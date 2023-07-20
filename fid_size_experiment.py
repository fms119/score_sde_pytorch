import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import numpy as np

sizes = [500,  2000,  4000,  6000,  8000, 10001, 15000,  1000, 12500, 20000]

from zijing_main import *

def get_fid(file_path):
    data_samples = np.ones((10000,3,1,1))

    data = np.load(file_path)

    gen_samples = data['images'].transpose(0,3,1,2)
    gen_samples = np.interp(gen_samples, (gen_samples.min(), gen_samples.max()), (0, 1))

    fid = compute_fid_nchw(data_samples, gen_samples)
    return fid

fid_trails = np.zeros((len(sizes),5))

for i, n in enumerate(sizes):
    for j, trial in enumerate(['a', 'b', 'c', 'd', 'e']):
        print(n)
        print(trial)
        print(f'{i} out of {len(sizes)}')
        
        file_path = f'/vol/bitbucket/fms119/score_sde_pytorch/samples/cifar10_true_trials/cirfar10_true_{n}{trial}.npz'
        fid = get_fid(file_path)
        
        print(fid)
        
        fid_trails[i, j] = fid
        np.savez('/vol/bitbucket/fms119/score_sde_pytorch/samples/cifar10_true_trials/trial_fids.npz', fid_trails=fid_trails)
