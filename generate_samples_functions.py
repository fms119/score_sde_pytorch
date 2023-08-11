import subprocess
import numpy as np
from datetime import datetime, timedelta
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
                # print('Used kill')
                process.kill()      # If it's still running, kill it
                time.sleep(1)       # Wait for a moment to let the process be killed
            # if process.poll() is None:
                # print(f'Failed to kill process {i} of {n}')
        #     else:
        #         print(f'Successfully killed process {i} of {n}')
        # else:
        #     print(f'Process {i} of {n} had already finished')

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
                   return_process=False, batch_size=64, cov=0, params=[0, 0, 1, 1, 0.16]):
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
        # f'echo Running on {gpu_name} && '  
        # Append the GPU name to the output
        # 'nice -n 1' 
        f'nice -n 1 python {python_script} '
        f'--batch_size {scale_batch_size(gpu_name) * batch_size} --gpu {gpu_name} --cov {cov} '
        f'--params {params[0]} {params[1]} {params[2]} {params[3]} {params[4]}'  
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
        'find ~/.cache/torch_extensions -type f -name "*lock*" -delete;'
        'ps -eo pid,user,etimes | awk \'$2 == "fms119" && $3 > 6 {print $1}\' | xargs -r kill -15'
    ]
    subprocess.Popen(command, shell=False)
    # Clever way to search through processes with similar names to the typed string
    # ps aux | grep -i fms119/Projects | awk '{print $2}' | xargs kill -15

def estimate_end(no_good_images, desired_samples, start_time):
    time_taken = (datetime.now() - start_time).total_seconds()  # in seconds
    secs_per_image = time_taken / no_good_images
    projected_time = secs_per_image * (desired_samples-no_good_images)
    end_t = timedelta(seconds=projected_time) + datetime.now()
    print(f'The process should finish at: {end_t.time().strftime("%H:%M:%S")}')

def merge_intermediate_samples(gpu_names, desired_samples):
    '''Join all the individually created samples so FID can be evaluated at 
    times during generation.'''
    for i in range(1,11):
        dir_path = f'/vol/bitbucket/fms119/score_sde_pytorch/samples/intermediate_images/{i}'
        files = [name+'.npz' for name in gpu_names]
        all_images = np.ones((1,3,32,32))
        for file in files:
            full_path = os.path.join(dir_path,file)
            data = np.load(full_path)
            images = data['images']
            all_images = np.concatenate((all_images, images))
        if all_images.shape[0]>=1000:
            np.savez(os.path.join(dir_path, f'all_samples_{desired_samples}.npz'), 
                     images=all_images[-desired_samples:])
        else:
            print(f'Incorrect size for {i}')

def compute_intermediate_fid():
    '''
    Compute the intermediate FIDs. The image files are created by merge_intermediate_samples().
    '''
    source = 'test'
    python_script = '/homes/fms119/Projects/doc_msc_project/score_sde_pytorch/fid_size_experiment.py'
    # path to the conda activation script
    conda_sh = "/vol/bitbucket/fms119/miniconda3/etc/profile.d/conda.sh"
    # the name of the environment to activate
    env_name = "score_sde_env"

    oaks = [f'oak{i:02}' for i in range(2, 36)]
    oaks.remove('oak02')
    oaks.remove('oak03')
    oaks.remove('oak05')
    oaks.remove('oak15')
    oaks.remove('oak27')
    oaks.remove('oak32')
    print(oaks)
    for i in range(1, 11):
        for j in range(2):
            gpu_name = oaks[i-1+10*j]
            # Maybe just run on 2 different machines??
            image_path = f'/vol/bitbucket/fms119/score_sde_pytorch/samples/intermediate_images/{i}/all_samples_1000.npz'
            save_path = f'/vol/bitbucket/fms119/score_sde_pytorch/assets/stats/fids/{i}'

            command = (
                f'ssh {gpu_name} '
                '"export CUDA_HOME=/vol/cuda/12.0.0 && '
                'export LD_LIBRARY_PATH=/vol/cuda/12.0.0/targets/x86_64-linux/lib:$LD_LIBRARY_PATH && '
                # Added this line to set log level
                'export TF_CPP_MIN_LOG_LEVEL=3 && '  
                f'source {conda_sh} && '
                f'conda activate {env_name} && '
                # Echo the gpu_name before running the script
                # f'echo Running on {gpu_name} && '  
                # Append the GPU name to the output
                f'python {python_script} '
                f'--source {source} --image_path {image_path} --save_path {save_path}'
                f' 2>&1 | sed \'s/^/[{gpu_name}] /\'"'  
            )
            
            subprocess.Popen(command, shell=True)