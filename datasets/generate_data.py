import yaml
import random
import json
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from data_utils import *

def seed_everything(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    with open('data_config.yml', 'r', encoding='utf-8') as config_file:
        config = yaml.load(config_file.read(), Loader=yaml.FullLoader)
    distribution = config['distribution'].split(',')
    capacity_type = config['capacity'].split(',')
    seed_everything(config['seed'])
    problems = []
    for d in distribution:
        for c in capacity_type:
            for i in tqdm(range(int(config['dataset_size'] / (len(distribution) * len(capacity_type))))):
                if config['problem_size_lower'] == config['problem_size_upper']:
                    problem_size = config['problem_size_lower']
                else:
                    problem_size = np.random.randint(config['problem_size_lower'], config['problem_size_upper'])
                if config['problem_type'] == 'TSP':
                    xy = generate_tsp_data(1, problem_size, config, d)
                elif config['problem_type'] == 'CVRP':
                    xy = generate_vrp_data(1, problem_size, config, d, c)
                problems.append(xy)
                
    save_dataset(problems, config['save_path'] + '/dataset.pkl')
    with open(config['save_path'] + '/config.json', 'w') as f:
        json.dump(config, f)