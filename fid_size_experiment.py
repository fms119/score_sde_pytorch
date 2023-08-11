import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import numpy as np
import gc
from zijing_main import compute_fid_nchw
import argparse


def get_fid(i=0, n=0, base_size='50k', source='test', file_path=''):
    '''
    base_size: str: '2k', '10k' or '50k' or None
    '''
    # Data not relevant as I have saved statistics
    data_samples = 0.5*np.ones((100, 3, 32, 32))

    if source == 'generated':
        file_path = ('/vol/bitbucket/fms119/score_sde_pytorch/samples/'
                     'all_samples_50000.npz')
        data = np.load(file_path)
        gen_samples = data['images'][(i*n):((i+1)*n)]
    elif source == 'training':
        file_path = ('/vol/bitbucket/fms119/score_sde_pytorch/samples/'
                     'cifar10_true_trials/cirfar10_true_50000.npz')
        data = np.load(file_path)
        gen_samples = data['images'][:n].transpose(0, 3, 1, 2)
    elif source == 'test':
        data = np.load(file_path)
        gen_samples = data['images']

    gc.collect()
    gen_samples = np.interp(gen_samples, (gen_samples.min(), gen_samples.max()),
                            (0, 1))
    fid = compute_fid_nchw(data_samples, gen_samples, base_size=base_size)
    return fid


if __name__ == '__main__':

    #I think this argparsing is unnecessary as I am unlikely to enter a file path 
    # as an argument
    parser = argparse.ArgumentParser(
        description='Configure batch sizes and gpu name.'
    )
    parser.add_argument('-s', '--source', type=str, default='test',
                        help='Are we computing FID for new params?')
    parser.add_argument('-i', '--image_path', type=str,
                        default='/vol/bitbucket/fms119/score_sde_pytorch/samples/synthetic_all_samples_20000.npz',
                        help='Where are the images to be evaluated?')
    parser.add_argument('-save', '--save_path', type=str,
                        default=('/vol/bitbucket/fms119/score_sde_pytorch/assets/stats/fids/00.npz'),
                        help='Where is the FID to be saved?')
    args = parser.parse_args()

    source = args.source
    image_path = args.image_path
    save_path = args.save_path

    fid = get_fid(source=source, file_path=image_path)
    print(f'{fid} for {image_path}')
    # You should always be saving progess during experiments like this.
    np.savez(save_path, fid=fid)
