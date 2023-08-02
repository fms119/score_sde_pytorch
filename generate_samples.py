import argparse
import gc

from generate_samples_functions import validate_images

import numpy as np
from losses import get_optimizer
from models.ema import ExponentialMovingAverage

from utils import restore_checkpoint

print('Importing')

from models import utils as mutils
from models import ncsnpp

import sampling
from sde_lib import VESDE
from sampling import (ReverseDiffusionPredictor, 
      				  LangevinCorrector, 
					  EulerMaruyamaPredictor, 
					  AncestralSamplingPredictor, 
					  NoneCorrector, 
					  NonePredictor,
					  AnnealedLangevinDynamics)
import datasets

parser = argparse.ArgumentParser(
	description='Configure batch sizes and gpu name.'
	)
parser.add_argument('-b', '--batch_size', type=int, default=2, 
					help='Number of images to be generated per machine')
parser.add_argument('-g', '--gpu', type=str, default='texel05', 
					help='which GPU in list has this come from')
parser.add_argument('-c', '--cov', type=str, default=0.7, 
					help='channel covariance')
args = parser.parse_args()


# @title Load the score-based model
sde = 'VESDE' #@param ['VESDE', 'VPSDE', 'subVPSDE'] {"type": "string"}
if sde.lower() == 'vesde':
	from configs.ve import cifar10_ncsnpp_continuous as configs
	ckpt_filename = "/vol/bitbucket/fms119/score_sde_pytorch/exp/ve/cifar10_ncsnpp_continuous/checkpoint_24.pth"
	# THE BELOW CHECKPOINTS DO NOT WORK
	# ckpt_filename = "/vol/bitbucket/fms119/score_sde_pytorch/exp/ve/cifar10_ncsnpp/checkpoint_16.pth"
	# ckpt_filename = "/vol/bitbucket/fms119/score_sde_pytorch/exp/ve/cifar10_ncsnpp_deep_continuous/checkpoint_12.pth"
	# 
	# I think these do not work because you need to change
	# from configs.ve import cifar10_ncsnpp_continuous as configs      to
	# from configs.ve import cifar10_ncsnpp as configs         or
	# from configs.ve import cifar10_ncsnpp_deep_continuous as configs         
	# 
	config = configs.get_config()  
	sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
	sampling_eps = 1e-5

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
# Explain reverse diffusion predictor in report.
predictor = ReverseDiffusionPredictor #@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
corrector = LangevinCorrector #@param ["LangevinCorrector", "AnnealedLangevinDynamics", "None"] {"type": "raw"}

corrector.cov = args.cov

# snr = 0.16 #@param {"type": "number"}
snr = 0.16 #@param {"type": "number"}
n_steps =  1#@param {"type": "integer"}
probability_flow = False #@param {"type": "boolean"}

gc.collect()

sampling_fn = sampling.get_pc_sampler(sde, shape, predictor, corrector,
  									  inverse_scaler, snr, n_steps=n_steps,
  									  probability_flow=probability_flow,
  									  continuous=config.training.continuous,
  									  eps=sampling_eps, device=config.device,
)

x, n = sampling_fn(score_model, args.gpu)

numpy_x = x.cpu().numpy()
		
# while not validate_images({'x':numpy_x}):
	
# 	print('caught failed before saving files')
# 	print(f'std: {numpy_x.reshape(-1).std()}')

# 	x, n = sampling_fn(score_model)
# 	numpy_x = x.cpu().numpy()

np.savez(f'/vol/bitbucket/fms119/score_sde_pytorch/samples/'
		 f'{args.gpu}_samples.npz', x=numpy_x)
