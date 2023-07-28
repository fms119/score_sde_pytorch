import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import numpy as np
import gc
from zijing_main import compute_fid_nchw

sizes = np.sort(
    [500,  2000,  4000,  6000,  8000, 10001, 15000,  1000, 12500, 20000,] 
    + list(range(30000, 60000, 10000)))

def get_fid(n, base_size=None, source='choose'):
    '''
    base_size: str: '10k' or '50k' or None
    '''
    #Data not relevant as I have saved statistics
    data_samples = 0.5*np.ones((100, 3, 32, 32))

    if source=='generated':
        file_path = ('/vol/bitbucket/fms119/score_sde_pytorch/samples/'
                     'all_samples_50000.npz')
        data = np.load(file_path)
        gen_samples = data['images'][:n]
    elif source=='training':
        file_path = ('/vol/bitbucket/fms119/score_sde_pytorch/samples/'
                     'cifar10_true_trials/cirfar10_true_50000.npz')
        data = np.load(file_path)
        gen_samples = data['images'][:n].transpose(0,3,1,2)

    gc.collect()
    gen_samples = np.interp(gen_samples, (gen_samples.min(), gen_samples.max()),
                            (0, 1))    
    fid = compute_fid_nchw(data_samples, gen_samples, base_size=base_size)
    return fid

fid_trails = np.zeros((2, len(sizes)))

base_size = '2k'

for j ,source in enumerate(['training', 'generated']):
    for i, n in enumerate(sizes):
        fid = get_fid(n, base_size=base_size, source=source)    
        fid_trails[j, i] = fid
        # You should always be saving progess during experiments like this.
        np.savez('/vol/bitbucket/fms119/score_sde_pytorch/samples/cifar10_true_trials/fid_size_experiment_extra_2k.npz', 
                 fid_trails=fid_trails)
