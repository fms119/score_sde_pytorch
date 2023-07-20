import subprocess
import numpy as np
import time
from datetime import datetime
import os
import atexit

'''Generate samples from a diffusion model, this process runs '''

desired_samples = 1000

batch_size = 64

# list of GPU IDs and corresponding names
GTX_TITAN_X = [f'0{i}' for i in range(1,10)]
RTX_2080_ti = ['18', '19', '20', '21', '22', '28', '29', '30',]
GTX_1080 = ['15', '14', '16', '17', '23', '24']

gpu_ids = GTX_TITAN_X + RTX_2080_ti + GTX_1080

ray_machines = ([f'ray0{i}' for i in range(1, 4)]
                + [f'ray0{i}' for i in range(6, 10)]
                + [f'ray{i}' for i in range(10, 27)])

gpu_names = ['gpu'+n for n in gpu_ids] + ray_machines

previously_saved_files = ['/vol/bitbucket/fms119/score_sde_pytorch/samples/' 
                          + gpu_name + '_samples.npz' for gpu_name in gpu_names]

# path to your python script
python_script = ('/homes/fms119/Projects/doc_msc_project/'
                 'score_sde_pytorch/generate_samples.py')

# path to the conda activation script
conda_sh = "/vol/bitbucket/fms119/miniconda3/etc/profile.d/conda.sh"
# the name of the environment to activate
env_name = "score_sde_env"


def remove_old_files(files):
    '''Remove files from previous run, this is necessary because if the script
    checks if the files exist when a process has finished and if they do not it
    assumes there has been an error so removes the gpu from the list of 
    available machines.'''
    for file in files:
        try:
            os.remove(file)
            # print(f"File {file} has been removed successfully")
        except FileNotFoundError:
            print(f"File {file} not found when doing initial deletion")
        except Exception as e:
            print(f"Error occurred while trying to remove {file}: {str(e)}")

def cleanup():
    '''Kill the processes created by the script so stop wasteful GPU use.'''
    for process in processes:
        if process.poll() is None:  # If the process hasn't ended
            process.terminate()     # Terminate the process

def validate_images(loaded_data, key='x'):
    '''Check if the data generation has failed on this GPU'''
    images = loaded_data[key]
    if images.reshape(-1).std()<0.2:
        return False
    elif np.isnan(images.reshape(-1).std()):
        return False
    else:
        return True

def scale_batch_size(gpu_name):
    '''Function to double the batch size when running on the fastest available
      GPUs'''
    fast_gpus = ['gpu18', 'gpu19', 'gpu20', 'gpu21', 'gpu22', 'gpu28', 'gpu29', 
                 'gpu30']
    if gpu_name in fast_gpus:
        return 2
    else:
        return 1

def start_process(i, return_process=False, batch_size=64):
    '''Starts a process on a specific GPU. Initially it creates a list of 
    processes but once a process has been run it starts another one and returns
    it'''
    gpu_name = gpu_names[i]
    command = (
        f'ssh {gpu_name} '
        '"export CUDA_HOME=/vol/cuda/12.0.0 && '
        'export LD_LIBRARY_PATH=/vol/cuda/12.0.0/targets/x86_64-linux/lib:$LD_LIBRARY_PATH && '
        # Added this line to set log level
        'export TF_CPP_MIN_LOG_LEVEL=3 && '  
        # 'targets/x86_64-linux/lib:$LD_LIBRARY_PATH && '
        f'source {conda_sh} && '
        f'conda activate {env_name} && '
        # Echo the gpu_name before  running the script
        f'echo Running on {gpu_name} && '  
        # Append the GPU name to the output
        f'python {python_script} '
        f'--batch_size {scale_batch_size(gpu_name) * batch_size} --gpu {gpu_name}'
        f' 2>&1 | sed \'s/^/[{gpu_name}] /\'"'  
    )
    process = subprocess.Popen(command, shell=True)
    if return_process:
        return process
    else:
        processes.append(process)

if __name__ == '__main__':    
    remove_old_files(previously_saved_files)

    #register the process so if this script fails generations are not left running
    atexit.register(cleanup)

    start_time = datetime.now()

    # list to hold the subprocesses
    processes = []

    print(f'The length of gpu_names is {len(gpu_names)}')

    # start processes on all machines
    for i in range(len(gpu_names)):
        print('starting processes')
        start_process(i, batch_size=batch_size)

    all_images = np.zeros((1,3,32,32))

    # monitor processes
    limit = False
    while processes:
        for i, process in enumerate(processes):
            if process.poll() is not None:  # the process has ended
                elapsed_time = datetime.now() - start_time
                print(f"Job on GPU {gpu_names[i]} finished after {elapsed_time}")
                time.sleep(5)
                
                file_path = ('/vol/bitbucket/fms119/score_sde_pytorch/samples/' 
                            + gpu_names[i] + '_samples.npz')
                
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

                # Restart the process on the same GPU regardless of outcome
                processes[i] = start_process(i, return_process=True, batch_size=batch_size)
                
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

    cleanup()

