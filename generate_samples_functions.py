import subprocess
import numpy as np
from datetime import datetime
import os
import time

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

def cleanup(processes):
    '''Kill the processes created by the script so stop wasteful GPU use.'''
    n = len(processes)
    for i, process in enumerate(processes):
        if process.poll() is None:  # If the process hasn't ended
            process.terminate()     # Try to terminate the process
            time.sleep(1)           # Wait for a moment to let the process terminate
            if process.poll() is None:
                print('Used kill')
                process.kill()      # If it's still running, kill it
                time.sleep(1)       # Wait for a moment to let the process be killed
            if process.poll() is None:
                print(f'Failed to kill process {i} of {n}')
            else:
                print(f'Successfully killed process {i} of {n}')
        else:
            print(f'Process {i} of {n} had already finished')

        
def validate_images(loaded_data, key='x'):
    '''Check if the data generation has failed on this GPU'''
    images = loaded_data[key]
    if images.reshape(-1).std()<0.17:
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

def start_process(i, gpu_names, conda_sh, env_name, python_script, processes,
                   return_process=False, batch_size=64):
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
        f'source {conda_sh} && '
        f'conda activate {env_name} && '
        # Echo the gpu_name before running the script
        f'echo Running on {gpu_name} && '  
        # Append the GPU name to the output
        # 'nice -n 1' 
        f'nice -n 1 python {python_script} '
        f'--batch_size {scale_batch_size(gpu_name) * batch_size} --gpu {gpu_name}'
        f' 2>&1 | sed \'s/^/[{gpu_name}] /\'"'  
    )
    process = subprocess.Popen(command, shell=True)
    
    if return_process:
        return process
    else:
        processes.append(process)


def kill_processes(gpu_name):
    command = [
        'ssh', gpu_name,
        'ps -eo pid,user,etimes | awk \'$2 == "fms119" && $3 > 6 {print $1}\' | xargs -r kill -15'
    ]
    # Clever way to search through processes with similar names to the typed string
    # ps aux | grep -i fms119/Projects | awk '{print $2}' | xargs kill -15

    subprocess.Popen(command, shell=False)

