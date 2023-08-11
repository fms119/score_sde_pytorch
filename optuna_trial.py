import optuna
import subprocess
import time
from fid_size_experiment import get_fid
import argparse
import numpy as np

parser = argparse.ArgumentParser(
    description='Give hyperparameters for optuna trial'
)
parser.add_argument('-s', '--split', type=int, default=0,
                    help='split gpu_names into s')
args = parser.parse_args()

file_path = ('/vol/bitbucket/fms119/score_sde_pytorch/samples/'
             f'all_samples_3000_{args.split}.npz')


def objective(trial):
    params = [trial.suggest_float('k_cha', 0.1, 0.9, step=0.1),
              trial.suggest_float('k_spa', 0.1, 0.6, step=0.1),
              trial.suggest_int('d_cha', 101, int(5e3)+1, step=100),
              trial.suggest_int('d_spa', 101, int(5e3)+1, step=100),
              trial.suggest_float('snr', 0.16, 0.26, step=0.01),
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

    while np.load(file_path)['images'].shape[0]!=3000:
        print('File not uploaded yet.')
        time.sleep(50)

    fid = get_fid(source='test', file_path=file_path)

    return fid


if __name__ == '__main__':
    storage_path = "sqlite:///optuna_studies/100N_1step_.db"
    study = optuna.create_study(study_name='100N_1step_', direction='minimize',
                                pruner=optuna.pruners.NopPruner(),
                                storage=storage_path,
                                load_if_exists=True
                                )
    study.optimize(objective, n_trials=100,)
