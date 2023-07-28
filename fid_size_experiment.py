import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import numpy as np
import gc
from zijing_main import compute_fid_nchw

sizes = np.sort(
    [500,  2000,  4000,  6000,  8000, 10001, 15000,  1000, 12500, 20000,] 
    + list(range(30000, 60000, 10000)))

def get_fid(n, base_size=None):
    '''
    base_size: str: '10k' or '50k' or None
    '''
    #Data not relevant as I have saved statistics
    data_samples = 0.5*np.ones((100, 3, 32, 32))
    file_path = '/vol/bitbucket/fms119/score_sde_pytorch/samples/all_samples_50000.npz'
    data = np.load(file_path)
    gen_samples = data['images'][:n]
    gc.collect()
    gen_samples = np.interp(gen_samples, (gen_samples.min(), gen_samples.max()), (0, 1))    
    fid = compute_fid_nchw(data_samples, gen_samples, base_size=base_size)
    return fid

data = np.load('/vol/bitbucket/fms119/score_sde_pytorch/samples/cifar10_true_trials/50k_train_vs_n_train.npz')
fid_trails = data['fid_trails']

for j ,base_size in enumerate(['10k', '50k']):
    for i, n in enumerate(sizes):
        J = j + 2
        fid = get_fid(n, base_size)    
        fid_trails[J, i] = fid
        # You should always be saving progess during experiments like this.
        np.savez('/vol/bitbucket/fms119/score_sde_pytorch/samples/cifar10_true_trials/fid_size_experiment.npz', fid_trails=fid_trails)
