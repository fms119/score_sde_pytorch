import subprocess

GTX_TITAN_X = [f'0{i}' for i in range(1,10)] + ['10', '11', '12', '13']
RTX_2080_ti = ['18', '19', '20', '21', '22', '28', '29', '30']
GTX_1080 = ['15', '14', '16', '17', '23', '24']

gpu_ids = GTX_TITAN_X + RTX_2080_ti + GTX_1080

ray_machines = ([f'ray0{i}' for i in range(1, 4)] + 
                [f'ray0{i}' for i in range(6, 10)] + 
                [f'ray{i}' for i in range(10, 27)])

gpu_names = ['gpu'+n for n in gpu_ids] + ray_machines

# Iterate over all GPU names
for gpu_name in gpu_names:
    # Format the command with the current GPU name
    print(gpu_name)
    command_template = (
        f'ssh {gpu_name} '
        '"kill $(ps aux | awk \'/fms119/ && /python/ && $10>=/"20:00"/ {printf \"%s \", $2}\')"'
    )
    # Execute the command
    subprocess.run(command_template, shell=True)
