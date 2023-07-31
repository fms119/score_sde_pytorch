import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import numpy as np
import gc
from zijing_main import compute_fid_nchw

base_size = '50k'
source = 'test'
image_path = '/vol/bitbucket/fms119/score_sde_pytorch/samples/all_samples_4000.npz'
snr = '0.2'

def get_fid(i=0, n=0, base_size='50k', source='choose', file_path=''):
    '''
    base_size: str: '2k', '10k' or '50k' or None
    '''
    #Data not relevant as I have saved statistics
    data_samples = 0.5*np.ones((100, 3, 32, 32))

    if source=='generated':
        file_path = ('/vol/bitbucket/fms119/score_sde_pytorch/samples/'
                     'all_samples_50000.npz')
        data = np.load(file_path)
        gen_samples = data['images'][(i*n):((i+1)*n)]
    elif source=='training':
        file_path = ('/vol/bitbucket/fms119/score_sde_pytorch/samples/'
                     'cifar10_true_trials/cirfar10_true_50000.npz')
        data = np.load(file_path)
        gen_samples = data['images'][:n].transpose(0,3,1,2)
    elif source=='test':
        data = np.load(file_path)
        gen_samples = data['images']

    gc.collect()
    gen_samples = np.interp(gen_samples, (gen_samples.min(), gen_samples.max()),
                            (0, 1))    
    fid = compute_fid_nchw(data_samples, gen_samples, base_size=base_size)
    return fid

if __name__=='__main__':
    fid = get_fid(source=source, file_path=image_path)    
    print(fid)
    # You should always be saving progess during experiments like this.
    save_path = ('/vol/bitbucket/fms119/score_sde_pytorch/samples/'
                 f'fid_results/snr_{snr}.npz')
    np.savez(save_path, fid=fid)