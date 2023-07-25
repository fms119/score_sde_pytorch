import subprocess
import numpy as np
import time
from datetime import datetime
import os
import atexit

from generate_samples_functions import * 

'''Generate samples from a diffusion model, this process runs '''

print(datetime.now())

desired_samples = 5001

batch_size = 128

# list of GPU IDs and corresponding names
GTX_TITAN_X = [f'0{i}' for i in range(1,10)] + ['10', '11', '12', '13']
RTX_2080_ti = ['18', '19', '20', '21', '22', '28', '29', '30',]
GTX_1080 = ['15', '14', '16', '17', '23', '24']

gpu_ids = GTX_TITAN_X + RTX_2080_ti + GTX_1080

ray_machines = ([f'ray0{i}' for i in range(1, 4)]
                + [f'ray0{i}' for i in range(6, 10)]
                + [f'ray{i}' for i in range(10, 27)])

gpu_names = ['gpu'+n for n in gpu_ids] + ray_machines

previously_saved_files = [
    '/vol/bitbucket/fms119/score_sde_pytorch/samples/' 
    + gpu_name + '_samples.npz' for gpu_name in gpu_names
    ]

# path to your python script
python_script = ('/homes/fms119/Projects/doc_msc_project/'
                 'score_sde_pytorch/generate_samples.py')

# path to the conda activation script
conda_sh = "/vol/bitbucket/fms119/miniconda3/etc/profile.d/conda.sh"
# the name of the environment to activate
env_name = "score_sde_env"

remove_old_files(previously_saved_files)

start_time = datetime.now()

# list to hold the subprocesses
processes = []

#register the process so if this script fails generations are not left running
# define a wrapper function that takes no arguments
def cleanup_wrapper():
    global processes
    cleanup(processes)

# register the wrapper function to be run at exit
atexit.register(cleanup_wrapper)

print(f'The length of gpu_names is {len(gpu_names)}')

# start processes on all machines
for i in range(len(gpu_names)):
    start_process(i, gpu_names, conda_sh, env_name, python_script,
                  processes, batch_size=batch_size)

all_images = np.zeros((1,3,32,32))

# monitor processes
limit = False
while processes:
    for i, process in enumerate(processes):
        if process.poll() is not None:  # the process has ended
            elapsed_time = datetime.now() - start_time
            print(f"Job on GPU {gpu_names[i]} finished after {elapsed_time}")
                        
            file_path = ('/vol/bitbucket/fms119/score_sde_pytorch/samples/' 
                        + gpu_names[i] + '_samples.npz')

            # Try to load the data file for up to 5 seconds
            delay_time = 5
            for t in range(delay_time):
                try:
                    data = np.load(file_path)  # Try to load the data file
                    print(f'Found file after {t} seconds')
                    break  # If the file was loaded successfully, break out of the loop
                except:
                    pass  # If the file could not be loaded, ignore the exception and try again
                time.sleep(1)  # Wait for a second before trying again

            try:
                data = np.load(file_path)
            except FileNotFoundError:
                print(f"File not found for {gpu_names[i]}. DROPPING GPU FROM LIST.")
                del processes[i]
                del gpu_names[i] 
                break

            # If the images are good, save them
            if validate_images(data):
                images = data['x']
                all_images = np.concatenate((all_images, images), 0)
                print(f'{gpu_names[i]} has obtained good images.')
                print(f'Collected {all_images.shape[0] - 1} of  {desired_samples} images.')
                print(f'Maximum: {all_images.max()}')
                print(f'Minimum: {all_images.min()}')
                
                # If we have enough samples, break
                if all_images.shape[0]>=desired_samples:
                    limit = True
                    break
            else:
                print(f'{gpu_names[i]} has failed.')
            
            print(f'[{gpu_names[i]}] The initial PID is {processes[i].pid}')
            # Restart the process on the same GPU regardless of outcome
            processes[i] = start_process(i, gpu_names, conda_sh, 
                                         env_name, python_script, 
                                         processes, return_process=True, 
                                         batch_size=batch_size)
            print(f'[{gpu_names[i]}] The next PID is {processes[i].pid}')
            
            try:
                # remove the file
                os.remove(file_path)
                print(f"File {file_path} has been removed successfully")
            except FileNotFoundError:
                print(f"File {file_path} not found")

    if limit:
        break
    time.sleep(10)  # wait a 10 seconds before checking the processes again

all_images = all_images[1:desired_samples+1, :, :, :]

print(f'The final number of good images is {all_images.shape[0]}')

np.savez(f'/vol/bitbucket/fms119/score_sde_pytorch/samples/'
        f'all_samples_{all_images.shape[0]}.npz', images=all_images)

print(f'The length of processes if {len(processes)}')

cleanup(processes)

time.sleep(20)

for gpu_name in gpu_names:
    kill_processes(gpu_name)