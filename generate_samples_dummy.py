import numpy as np
import time
import argparse

parser = argparse.ArgumentParser(description='Configure batch sizes and gpu name.')

parser.add_argument('-b', '--batch_size', type=int, default=120, 
                    help='Number of images to be generated per machine')
parser.add_argument('-g', '--gpu', type=str, default='texel04', 
                    help='which GPU in list has this come from')
args = parser.parse_args()

delay  = np.random.randint(0,args.batch_size)

time.sleep(args.batch_size)

numpy_x = (10*np.random.random((10, 3, 32, 32)))

np.savez(f'/vol/bitbucket/fms119/score_sde_pytorch/samples/dummy_generations/'
         f'{args.gpu}_samples.npz', x=numpy_x)

