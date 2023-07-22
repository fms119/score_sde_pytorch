import numpy as np
import time
import argparse

parser = argparse.ArgumentParser(description='Configure batch sizes and gpu name.')

parser.add_argument('-b', '--batch_size', type=int, default=4, 
                    help='Number of images to be generated per machine')
parser.add_argument('-g', '--gpu', type=str, default='ray04', 
                    help='which GPU in list has this come from')
args = parser.parse_args()

delay  = np.random.randint(0,args.batch_size)

time.wait(delay)

numpy_x = np.zeros((2,3))

np.savez(f'/vol/bitbucket/fms119/score_sde_pytorch/samples/'
         f'{args.gpu}_samples.npz', x=numpy_x)

