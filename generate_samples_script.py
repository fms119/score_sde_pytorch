import subprocess
import numpy as np
import time
from datetime import datetime

def good_images(loaded_data, key='x'):
    '''Check if the data generation has failed on this GPU'''
    images = loaded_data['x']
    if images.reshape(-1).std()<0.2:
        return False
    else:
        return True

start_time = datetime.now()

# list of GPU IDs and corresponding names
gpu_ids = ['19', '20', '21', '28', '29', '23', '02', '16', '08']
gpu_names = ['gpu'+n for n in gpu_ids]

# path to your python script
python_script = "/homes/fms119/Projects/doc_msc_project/score_sde_pytorch/generate_samples.py"

# path to the conda activation script
conda_sh = "/vol/bitbucket/fms119/miniconda3/etc/profile.d/conda.sh"
# the name of the environment to activate
env_name = "score_sde_env"

# list to hold the subprocesses
processes = []

# loop over your machines
for i, gpu_name in enumerate(gpu_names):
    print(f"Running job on {gpu_name} with GPU {gpu_ids[i]}")
    # ssh into the machine and run the command
    command = (
        f'ssh {gpu_name} '
        '"export CUDA_HOME=/vol/cuda/12.0.0 && '
        'export LD_LIBRARY_PATH=/vol/cuda/12.0.0/targets/x86_64-linux/lib:$LD_LIBRARY_PATH && '
        f'source {conda_sh} && '
        f'conda activate {env_name} && '
        f'python {python_script} --gpu {gpu_ids[i]}"'
    )
    process = subprocess.Popen(command, shell=True)
    processes.append(process)

# monitor processes
while processes:
    for i, process in enumerate(processes):
        if process.poll() is not None:  # the process has ended
            elapsed_time = datetime.now() - start_time
            print(f"Job on GPU {gpu_names[i]} finished after {elapsed_time}")
            del processes[i]
            del gpu_names[i]
            break  # break the for loop and start from the first process again
    time.sleep(10)  # wait a 10 seconds before checking the processes again




data = np.load('/vol/bitbucket/fms119/score_sde_pytorch/samples/' + gpu_ids[0] + '_samples.npz')


all_images = np.zeros((1,3,32,32))

for number in gpu_ids:
    data = np.load('/vol/bitbucket/fms119/score_sde_pytorch/samples/' + number + '_samples.npz')
    if good_images(data):
        images = data['x']
        all_images = np.concatenate((all_images, images), 0)
    else:
        print(f'gpu{number} has failed.')

all_images = all_images[1:, :, :, :]

print(f'The final number of good images is {all_images.shape[0]}')

np.savez(f'/vol/bitbucket/fms119/score_sde_pytorch/samples/'
         f'all_samples_{all_images.shape[0]}.npz', x=all_images)
