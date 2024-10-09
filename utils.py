import random
import torch
import os
import csv
import pickle
import numpy as np

def seed_everything(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def augment_xy_by_8_fold(problems):
    # problems.shape: (batch, problem, 2)
    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (batch * 8, problem, 2)

    return aug_problems
    
def process_instance_CVRP(instance):
    depot = instance['depot']
    loc = instance['loc']
    demand = instance['demand']

    xy = torch.cat((depot, loc), dim=1)
    demand = torch.cat((torch.zeros(1, 1), demand), dim=1)
    batch_x = torch.cat((xy, demand[:, :, None]), dim=2)

    return batch_x

def prepare_dataset(problem_type, name=None):
    train_instance_set = []
    train_label_set = []
    test_instance_set = []
    test_label_set = []

    with open(f'datasets/{problem_type}train/dataset.pkl', 'rb') as f:
        train_instance_set = pickle.load(f)

    if "LIB" in name:
        test_file = f'datasets/{problem_type}LIB/dataset.pkl'
    elif "test" in name:
        test_file = f'datasets/{problem_type}test/dataset.pkl'
    else:
        test_file = f'datasets/{problem_type}val/dataset.pkl'
    with open(test_file, 'rb') as f:
        test_instance_set = pickle.load(f)
        
    with open(f'datasets/{problem_type}train/raw_label.pkl', 'rb') as f:
        train_raw_labels = pickle.load(f)
    
    if "LIB" in name:
        test_label_file = f'datasets/{problem_type}LIB/raw_label.pkl'
    elif "test" in name:
        test_label_file = f'datasets/{problem_type}test/raw_label.pkl'
    else:
        test_label_file = f'datasets/{problem_type}val/raw_label.pkl'

    with open(test_label_file, 'rb') as f:
        test_raw_labels = pickle.load(f)

    for i in range(len(train_instance_set)):
        key = str(i)
        train_label_set.append([train_raw_labels[key]['ind'], train_raw_labels[key]['cost'], train_raw_labels[key]['time'], train_raw_labels[key]['gap']])

    for i in range(len(test_instance_set)):
        key = str(i)
        test_label_set.append([test_raw_labels[key]['ind'], test_raw_labels[key]['cost'], test_raw_labels[key]['time'], test_raw_labels[key]['gap']])

    if problem_type == 'CVRP':
        train_instance_set = [process_instance_CVRP(ins) for ins in train_instance_set]
        test_instance_set = [process_instance_CVRP(ins) for ins in test_instance_set]

    return train_instance_set, train_label_set, test_instance_set, test_label_set

def representative(dataset, labels, ratio=0.01):
    representative_data = []
    representative_label = []
    for k in range(len(labels[0][1])):
        # compute gaps
        data_per_solver = []
        label_per_solver = []
        gaps = []
        idx_data = []
        for i in range(len(dataset)):
            if labels[i][0] == k:
                cost = torch.tensor(np.array(labels[i][1]))
                cost, _ = cost.topk(2, largest=False)
                gaps.append(cost[0] / cost[1])
                idx_data.append(i)
        
        # select top instances
        num = int(len(gaps) * ratio)
        gaps = torch.tensor(np.array(gaps))
        top_gaps, idx_sel = gaps.topk(num, largest=False)
        idx = []
        for j in idx_sel:
            idx.append(idx_data[j])
        for j in idx:
            data_per_solver.append(dataset[j])
            label_per_solver.append(labels[j])
        representative_data.append(data_per_solver)
        representative_label.append(label_per_solver)

    return [representative_data, representative_label]

class csv_logger():
    def __init__(self, log_dir, log_name=None):
        if log_name is None:
            file_name = log_dir + '/log.csv'
        else:
            file_name = log_dir + f'/{log_name}.csv'
        print(file_name)
        if os.path.exists(file_name):
            f = open(file_name, 'a')
            self.file_logger = csv.DictWriter(f, fieldnames=['acc', 'top_1', 'top_2', 'top_3', 'top_4', 'cover_80', 'top-p', 'time_top_1', 'time_top_2', 'time_top_3', 'time_top_4', 'time_cover_80', 'time_top-p'])

        else:
            f = open(file_name, 'w')
            self.file_logger = csv.DictWriter(f, fieldnames=['acc', 'top_1', 'top_2', 'top_3', 'top_4', 'cover_80', 'top-p', 'time_top_1', 'time_top_2', 'time_top_3', 'time_top_4', 'time_cover_80', 'time_top-p'])            
            self.file_logger.writeheader()
            
    def write(self, log_dict):
        self.file_logger.writerow(log_dict)