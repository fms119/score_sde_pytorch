import optuna
import subprocess
import time
from fid_size_experiment import get_fid

file_path = '/vol/bitbucket/fms119/score_sde_pytorch/samples/all_samples_3000.npz'

def objective(trial):
    
    params = [trial.suggest_float('k_cha', 0, 0.65),
             trial.suggest_float('k_spa', 0, 0.65),
             trial.suggest_int('d_cha', 1, int(2e5), step=1000),
             trial.suggest_int('d_spa', 1, int(2e5), step=1000), 
             ]
    # Give parameters as args to 'generate_samples_script.py'
    command = f'python generate_samples_script.py --params {params[0]} {params[1]} {params[2]} {params[3]}'
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, shell=True)

    # Wait until the process has finished
    while process.poll() is None:
        time.sleep(50)
    
    fid = get_fid(source='test', file_path=file_path)

    print(f'Trial {trial.number} achieved FID:{fid} with params: {params}')
    
    return fid

# Define a callback function to print the best trial so far
def print_best_trial(study, trial):
    best_trial = study.best_trial
    print(f'Best trial so far: score {best_trial.value}, params {best_trial.params}')

if __name__=='__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50, callbacks=[print_best_trial])

    # Print the result
    trial = study.best_trial
    print(f'Best trial after all trials: score {trial.value}, params {trial.params}')
