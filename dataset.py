import torch
import time
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import *
from utils import augment_xy_by_8_fold


class SelectionDataset(Dataset):
    def __init__(self, dataset, labels, manual_feature=False, data_aug=False):
        super(SelectionDataset, self).__init__()
        self.manual_feature = manual_feature
        self.dataset = dataset
        self.labels = labels
        if manual_feature == True:
            start_time = time.time()
            self.features = torch.cat([manual_features(data[0])[None, :] for data in dataset], dim=0)
            print(time.time() - start_time)
        else:
            self.features = None

        if data_aug == True:
            dataset = []
            labels = []
            features = []
            for k, ins in enumerate(self.dataset):
                aug_coords = augment_xy_by_8_fold(ins[:, :, :2])
                if ins.shape[-1] == 3: # CVRP
                    aug_ins = torch.cat((
                        aug_coords, ins[:, :, 2:].repeat(8, 1, 1)
                    ), dim=2)
                else:
                    aug_ins = aug_coords
                dataset.extend([aug_ins[0], aug_ins[1], aug_ins[2], aug_ins[3], aug_ins[4], aug_ins[5], aug_ins[6], aug_ins[7]])

                if manual_feature == True:
                    features.extend([self.features[k]] * 8)
                labels.extend([self.labels[k]] * 8)
            self.dataset= dataset
            self.labels = labels
            self.features = features
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.manual_feature == True:
            return [self.dataset[index], self.labels[index], self.features[index], index]
        else:
            return [self.dataset[index], self.labels[index], index] 


def manual_features(nodes):
    '''
    Adaption from "Understanding TSP Difficulty by Learning from Evolved Instances"
    '''
    nodes = nodes.squeeze(0)
    problem_type = 'TSP'
    if nodes.shape[-1] == 3:
        problem_type = 'CVRP'
        demands = nodes[:, 2]
        nodes = nodes[:, :2]

    dist_mat = (nodes[:, None, :] - nodes[None, :, :]).norm(p=2, dim=-1) # (problem, problem)
    std = torch.std(dist_mat.reshape(1, -1).squeeze(0), dim=-1, keepdim=True)   # 1-dim
    centroid = nodes.mean(dim=0, keepdim=True)  # 2-dim
    radius = (nodes - centroid).norm(dim=1).mean(dim=0, keepdim=True)   # 1-dim
    count_distinct = (torch.bincount(torch.round(dist_mat * 100).int().reshape(1, -1).squeeze(0)) == 0).sum()   # 1-dim
    nNN, _ = torch.min(dist_mat + 1e3 * torch.eye(nodes.shape[0], device=dist_mat.device), dim=-1)    
    std_nNN = torch.std(nNN)    # 1-dim
    results = HDBSCAN().fit(nodes)  
    cluster_ratio = np.max(results.labels_) / nodes.shape[0]    # 1-dim
    outlier_ratio = (results.labels_ == -1).sum() / nodes.shape[0]   # 1-dim
    radius_cluster = []

    for i in range(np.max(results.labels_)):
        centroid_cluster = nodes[results.labels_ == i].mean(dim=0, keepdim=True)
        radius_cluster.append(np.mean(np.linalg.norm(nodes[results.labels_ == i] - centroid_cluster)))

    radius_cluster = np.mean(radius_cluster)      # 1-dim
    if np.max(results.labels_) == -1:
        radius_cluster = 0.
        cluster_ratio = 0.

    if problem_type == 'TSP':
        # 9-dim manual features
        features = torch.zeros(9)
    else:
        features = torch.zeros(11)
    features[0] = std
    features[1:3] = centroid
    features[3] = radius
    features[4] = count_distinct
    features[5] = std_nNN
    features[6] = cluster_ratio
    features[7] = outlier_ratio
    features[8] = torch.tensor(radius_cluster)
    if problem_type == 'CVRP':
        features[9] = demands.mean()
        features[10] = demands.std()

    return features

def collate_fn(batch):
    batch_x = [data[0].squeeze(0) for data in batch]
    batch_y = torch.tensor([data[1][0] for data in batch])
    batch_cost = torch.cat([torch.tensor(data[1][1])[None, :] for data in batch], dim=0)
    batch_time = torch.cat([torch.tensor(data[1][2])[None, :] for data in batch], dim=0)
    if len(batch[0]) == 4:  # with manual features
        batch_feature = torch.cat([data[2][None, :] for data in batch], dim=0)
    batch_gap = torch.cat([torch.tensor(data[1][3])[None, :] for data in batch], dim=0)

    index = [data[-1] for data in batch]
    # padding
    lengths = np.array([data.shape[0] for data in batch_x])
    max_length = np.max(lengths)
    batch_x = [F.pad(batch_x[i], (0, 0, 0, max_length - lengths[i]))[None, :, :] for i in range(len(batch_x))]
    ninf_mask = torch.zeros(len(batch_x), max_length)
    for i in range(len(batch_x)):
        ninf_mask[i, lengths[i]: max_length] = float('-inf')
    lengths = torch.tensor(lengths, dtype=torch.float32)
    batch_x = torch.cat(batch_x, dim=0)
    
    if len(batch[0]) == 4:  # with manual features
        return [batch_x, batch_y, batch_cost, lengths, ninf_mask, batch_gap, batch_time, batch_feature, index]
    else:
        return [batch_x, batch_y, batch_cost, lengths, ninf_mask, batch_gap, batch_time, index]