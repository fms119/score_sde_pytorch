import optuna
import subprocess
import time
import numpy as np
from fid_size_experiment import get_fid

desired_samples = 3000

file_path = ('/vol/bitbucket/fms119/ncsn_vol/samples/'
             f'all_samples_{desired_samples}.npz')
study_str = 'CelebA_anneal'

def objective(trial):
    params = [trial.suggest_float('k_cha', 0.1, 0.9, step=0.05),
              trial.suggest_float('k_spa', 0.1, 0.6, step=0.05),
              trial.suggest_int('d_cha', 1, 81, step=2),
              trial.suggest_int('d_spa', 1, 51, step=2),
              trial.suggest_float('step_size', 0.000025, 0.00005, step=0.000005),
              ]
    # Give parameters as args to 'generate_samples_script.py'
    command = ('python /homes/fms119/Projects/doc_msc_project/ncsn/'
               'generate_samples_script_CELEBA.py '
               f'--params {params[0]} {params[1]} {params[2]} {params[3]} {params[4]} '
    )
    process = subprocess.Popen(command, 
                            #    stdout=subprocess.DEVNULL, 
                               shell=True)

    # Wait until the process has finished
    while process.poll() is None:
        time.sleep(10)

    while np.load(file_path)['images'].shape[0]!=desired_samples:
        print('File not uploaded yet.')
        time.sleep(50)

    fid = get_fid(source='test', file_path=file_path, base_size='celeba_50k')

    return fid

if __name__ == '__main__':
    storage_path = "sqlite:///optuna_studies/"+study_str+".db"
    study = optuna.create_study(study_name=study_str, direction='minimize',
                                pruner=optuna.pruners.NopPruner(),
                                storage=storage_path,
                                load_if_exists=True
                                )

    # study.enqueue_trial({'k_cha': 0.4, 'k_spa': 0.4, 'd_cha': 45, 'd_spa': 6, 'step_size':0.00006})  
    # study.enqueue_trial({'k_cha': 0.5, 'k_spa': 0.4, 'd_cha': 15, 'd_spa': 15, 'step_size':0.00006})
    
    study.optimize(objective, n_trials=50,)
