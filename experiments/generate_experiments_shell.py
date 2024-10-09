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
            shell.append(f"python run.py --config_name {c} --loss {l} --seed {s} --gpu_id {gpus[i % len(gpus)]} &\n")
            i += 1
            if i % len(gpus) == 0:
                shell.append(f"wait;\n")
                

with open(f"../run_experiment.sh", "w") as f:
    for line in shell:
        f.writelines(line)