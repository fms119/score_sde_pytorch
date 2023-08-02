import subprocess

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
            f'echo Running on {gpu_name} && '  
            # Append the GPU name to the output
            f'python {python_script} '
            f'--source {source} --image_path {image_path} --save_path {save_path}'
            f' 2>&1 | sed \'s/^/[{gpu_name}] /\'"'  
        )
        subprocess.Popen(command, shell=True)