import argparse

from generate_samples_script import validate_images

import matplotlib.pyplot as plt
# import io
# import csv
import numpy as np
import pandas as pd
# import matplotlib
# import importlib
# import os
# import functools
# import itertools
from losses import get_optimizer
from models.ema import ExponentialMovingAverage

import numpy as np
import io
import likelihood
import controllable_generation
from utils import restore_checkpoint


import models
from models import utils as mutils
from models import ncsnv2
from models import ncsnpp
from models import ddpm as ddpm_model
from models import layerspp
from models import layers
from models import normalization
import sampling
from likelihood import get_likelihood_fn
from sde_lib import VESDE, VPSDE, subVPSDE
from sampling import (ReverseDiffusionPredictor, 
                      LangevinCorrector, 
                      EulerMaruyamaPredictor, 
                      AncestralSamplingPredictor, 
                      NoneCorrector, 
                      NonePredictor,
                      AnnealedLangevinDynamics)
import datasets
import argparse

parser = argparse.ArgumentParser(description='Configure batch sizes and gpu name.')
parser.add_argument('-b', '--batch_size', type=int, default=4, 
                    help='Number of images to be generated per machine')
parser.add_argument('-g', '--gpu', type=str, default='ray04', 
                    help='which GPU in list has this come from')
args = parser.parse_args()


# @title Load the score-based model
sde = 'VESDE' #@param ['VESDE', 'VPSDE', 'subVPSDE'] {"type": "string"}
if sde.lower() == 'vesde':
  from configs.ve import cifar10_ncsnpp_continuous as configs
  ckpt_filename = "/vol/bitbucket/fms119/score_sde_pytorch/exp/ve/cifar10_ncsnpp_continuous/checkpoint_24.pth"
  # THE BELOW CHECKPOINTS DO NOT WORK
  # ckpt_filename = "/vol/bitbucket/fms119/score_sde_pytorch/exp/ve/cifar10_ncsnpp/checkpoint_16.pth"
  # ckpt_filename = "/vol/bitbucket/fms119/score_sde_pytorch/exp/ve/cifar10_ncsnpp_deep_continuous/checkpoint_12.pth"
  config = configs.get_config()  
  sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
  sampling_eps = 1e-5
elif sde.lower() == 'vpsde':
  from configs.vp import cifar10_ddpmpp_continuous as configs  
  ckpt_filename = "/vol/bitbucket/fms119/score_sde_pytorch/exp/vp/cifar10_ddpmpp_continuous/checkpoint_8.pth"
  config = configs.get_config()
  sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
  sampling_eps = 1e-3
elif sde.lower() == 'subvpsde':
  from configs.subvp import cifar10_ddpmpp_continuous as configsls
  ckpt_filename = "/vol/bitbucket/fms119/score_sde_pytorch/exp/subvp/cifar10_ddpmpp_continuous/checkpoint_15.pth"
  config = configs.get_config()
  sde = subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
  sampling_eps = 1e-3


batch_size =   args.batch_size # 128#@param {"type":"integer"}

config.training.batch_size = batch_size
config.eval.batch_size = batch_size

random_seed = 0 #@param {"type": "integer"}
# Stops creation of identical imaged from different machines 
random_seed = np.random.randint(0, 99999) 

sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
score_model = mutils.create_model(config)

optimizer = get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.parameters(),
                               decay=config.model.ema_rate)
state = dict(step=0, optimizer=optimizer,
             model=score_model, ema=ema)

state = restore_checkpoint(ckpt_filename, state, config.device)
ema.copy_to(score_model.parameters())



#@title PC sampling
img_size = config.data.image_size
channels = config.data.num_channels
shape = (batch_size, channels, img_size, img_size)
predictor = ReverseDiffusionPredictor #@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
corrector = LangevinCorrector #@param ["LangevinCorrector", "AnnealedLangevinDynamics", "None"] {"type": "raw"}
snr = 0.16 #@param {"type": "number"}
n_steps =  1#@param {"type": "integer"}
probability_flow = False #@param {"type": "boolean"}
sampling_fn = sampling.get_pc_sampler(sde, shape, predictor, corrector,
                                      inverse_scaler, snr, n_steps=n_steps,
                                      probability_flow=probability_flow,
                                      continuous=config.training.continuous,
                                      eps=sampling_eps, device=config.device)

x, n = sampling_fn(score_model)
numpy_x = x.cpu().numpy()
    
while not validate_images({'x':numpy_x}):
  
  print('caught failed before saving files')
  print(f'std: {numpy_x.reshape(-1).std()}')

  x, n = sampling_fn(score_model)
  numpy_x = x.cpu().numpy()

np.savez(f'/vol/bitbucket/fms119/score_sde_pytorch/samples/'
         f'{args.gpu}_samples.npz', x=numpy_x)
