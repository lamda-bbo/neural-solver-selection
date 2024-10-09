config_name = ['config_TSP.yml', 'config_CVRP.yml']
loss = ['rank']
# loss = ['CE', 'rank']
seed = ['54321', '4321', '2024', '216', '924']

shell = []
i = 0
gpus = [0, 1]
for c in  config_name:
    for l in loss:
        for s in seed:
            i += 1
            if 'TSP' in c:
                shell.append(f"python run.py --gpu_id {gpus[0]} --load {c}_{l}_{s} --test_file TSPtest --exp_name {c}_{l}_TSPtest &\n")
                shell.append(f"python run.py --gpu_id {gpus[1]} --load {c}_{l}_{s} --test_file TSPLIB --exp_name {c}_{l}_TSPLIB &\n")
            else:
                shell.append(f"python run.py --gpu_id {gpus[0]} --load {c}_{l}_{s} --test_file CVRPtest --exp_name {c}_{l}_CVRPtest &\n")
                shell.append(f"python run.py --gpu_id {gpus[1]} --load {c}_{l}_{s} --test_file CVRPLIB --exp_name {c}_{l}_CVRPLIB &\n")
            shell.append(f"wait;\n")
                

with open(f"../run_test.sh", "w") as f:
    for line in shell:
        f.writelines(line)