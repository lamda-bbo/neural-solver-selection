import numpy as np
import torch
import random
import pickle
import os
import argparse

from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


def generate_tsp_data(dataset_size, problem_size, config, distribution):
    problems = []
    if distribution == 'Gaussian':
        problems.extend(generate_tsp_data_gaussian(dataset_size, problem_size, config['Gaussian_params']))
    if distribution == 'Explosion':
        problems.extend(generate_tsp_data_explosion(dataset_size, problem_size, config['Explosion_params']))
    if distribution == 'Rotation':
        problems.extend(generate_tsp_data_Rotation(dataset_size, problem_size))
    problems = torch.tensor(problems)

    return problems
    

def generate_tsp_data_gaussian(dataset_size, problem_size, params):
    '''
    Generate tsp datasets form gaussian mixture distributions
    params: dict with four keys ['var_lower', 'var_upper', 'num_modes_lower', 'num_modes_upper']
    '''
    if params['num_modes_lower'] == params['num_modes_upper']:
        num_modes = params['num_modes_lower']
    else:
        num_modes = np.random.randint(params['num_modes_lower'], params['num_modes_upper'])
    if num_modes == 0:
        # uniform distribution
        problems = torch.rand(size=(dataset_size, problem_size, 2)).numpy().tolist()
        # problems.shape: (batch, problem, 2)
    else:
        problems = []
        for i in range(dataset_size):
            mix_proportion = np.random.rand(num_modes)
            nums = np.random.multinomial(problem_size, mix_proportion / np.sum(mix_proportion))
            xy = []
            for num in nums:
                if params['no_cov']:
                    var = np.random.uniform(params['var_lower'], params['var_upper'])
                    cov = [[var, 0], [0, var]]
                else:
                    var_x = np.random.uniform(params['var_lower'], params['var_upper'])
                    var_y = np.random.uniform(params['var_lower'], params['var_upper'])
                    cov_xy = np.random.uniform(-np.sqrt(var_x * var_y), np.sqrt(var_x * var_y))
                    cov = [[var_x, cov_xy], [cov_xy, var_y]]
                center = np.random.uniform(0, 100, size=(1, 2))
                nxy = np.random.multivariate_normal(mean=center.squeeze(), cov=cov, size=(num,))
                xy.extend(nxy)
            xy = np.array(xy)
            xy = MinMaxScaler().fit_transform(xy).tolist()
            problems.append(xy)
    
    return problems

def generate_vrp_data(dataset_size, problem_size, config, distribution, capacity_type):
    node_xy = generate_tsp_data(dataset_size, problem_size, config, distribution)
    depot_xy = torch.rand(size=(dataset_size, 1, 2))

    if capacity_type == 'scale':
        demand = (torch.FloatTensor(dataset_size, problem_size).uniform_(0, 9).int() + 1).float()
        capacities = torch.ceil(torch.tensor(30 + problem_size / 5)).repeat(dataset_size)
    if capacity_type == 'triangular':
        choice = random.choice(['a', 'b', 'c', 'd'])
        if choice == 'a':
            demand = (torch.FloatTensor(dataset_size, problem_size).uniform_(0, 9).int() + 1).float()
        elif choice == 'b':
            demand = (torch.FloatTensor(dataset_size, problem_size).uniform_(4, 9).int() + 1).float()
        elif choice == 'c':
            demand = (torch.FloatTensor(dataset_size, problem_size).uniform_(0, 99).int() + 1).float()
        elif choice == 'd':
            demand = (torch.FloatTensor(dataset_size, problem_size).uniform_(49, 99).int() + 1).float()
        # Following the set-X of VRPLib ("New benchmark instances for the capacitated vehicle routing problem") to generate capacity 
        route_length = torch.tensor(np.random.triangular(3, 6, 25, size=dataset_size))
        capacities = torch.ceil(route_length * demand.sum(1) / problem_size)
    if capacity_type == 'random_triangular':
        choice = random.choice(['a', 'b', 'c', 'd'])
        if choice == 'a':
            demand = (torch.FloatTensor(dataset_size, problem_size).uniform_(0, 9).int() + 1).float()
        elif choice == 'b':
            demand = (torch.FloatTensor(dataset_size, problem_size).uniform_(4, 9).int() + 1).float()
        elif choice == 'c':
            demand = (torch.FloatTensor(dataset_size, problem_size).uniform_(0, 99).int() + 1).float()
        elif choice == 'd':
            demand = (torch.FloatTensor(dataset_size, problem_size).uniform_(49, 99).int() + 1).float()
        
        high = np.random.randint(20, int(problem_size / 2))
        mid = np.random.randint(5, high)
        low = np.random.randint(3, mid)
        route_length = torch.tensor(np.random.triangular(low, mid, high, size=dataset_size))
        capacities = torch.ceil(route_length * demand.sum(1) / problem_size)
    data = {
        'loc': node_xy,
        # Uniform 1 - 9, scaled by capacities
        'demand': (demand / capacities[:, None]).float(),
        'depot': depot_xy
    }
    return data

def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename

def save_dataset(dataset, filename):

    filedir = os.path.split(filename)[0]

    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)