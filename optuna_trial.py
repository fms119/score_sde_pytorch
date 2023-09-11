import optuna
import subprocess
import time
from fid_size_experiment import get_fid
import argparse
import numpy as np

desired_samples = 5000

parser = argparse.ArgumentParser(
    description='Give hyperparameters for optuna trial'
)
parser.add_argument('-s', '--split', type=int, default=0,
                    help='split gpu_names into s')
args = parser.parse_args()

file_path = ('/vol/bitbucket/fms119/score_sde_pytorch/samples/'
             f'all_samples_{desired_samples}_{args.split}.npz')


def objective(trial):
    params = [trial.suggest_float('k_cha', 0.4, 0.9, step=0.1),
              trial.suggest_float('k_spa', 0.3, 0.6, step=0.1),
              trial.suggest_int('d_cha', 5, 2500, step=5),
              trial.suggest_int('d_spa', 5, 2500, step=5),
              trial.suggest_float('snr', 0.32, 0.38, step=0.02),
              ]
    # Give parameters as args to 'generate_samples_script.py'
    command = ('python generate_samples_script.py '
               f'--params {params[0]} {params[1]} {params[2]} {params[3]} {params[4]} '
               f'--split {args.split}')
    process = subprocess.Popen(command, 
                            #    stdout=subprocess.DEVNULL, 
                               shell=True)

    # Wait until the process has finished
    while process.poll() is None:
        time.sleep(10)

    while np.load(file_path)['images'].shape[0]!=desired_samples:
        print('File not uploaded yet.')
        time.sleep(50)

    fid = get_fid(source='test', file_path=file_path)

    return fid

if __name__ == '__main__':
    storage_path = "sqlite:///optuna_studies/50N_5000.db"
    study = optuna.create_study(study_name='50N_5000', direction='minimize',
                                pruner=optuna.pruners.NopPruner(),
                                storage=storage_path,
                                load_if_exists=True
                                )
    # study.enqueue_trial({'k_cha': 0.8, 'k_spa': 0.3, 'd_cha': 15, 'd_spa': 35, 'snr': 0.36})  
    # study.enqueue_trial({'k_cha': 0.7, 'k_spa': 0.3, 'd_cha': 11, 'd_spa': 11, 'snr': 0.36})  
    # study.enqueue_trial({'k_cha': 0.7, 'k_spa': 0.3, 'd_cha': 11, 'd_spa': 31, 'snr': 0.36})  
    # study.enqueue_trial({'k_cha': 0.8, 'k_spa': 0.3, 'd_cha': 20, 'd_spa': 15, 'snr': 0.36})   
    # study.enqueue_trial({'k_cha': 0.8, 'k_spa': 0.3, 'd_cha': 15, 'd_spa': 5, 'snr': 0.36})   
    # study.enqueue_trial({'k_cha': 0.8, 'k_spa': 0.3, 'd_cha': 15, 'd_spa': 25, 'snr': 0.36})   
    study.enqueue_trial({'k_cha': 0.8, 'k_spa': 0.6, 'd_cha': 1000, 'd_spa': 2000, 'snr': 0.36})   

    study.optimize(objective, n_trials=30,)
