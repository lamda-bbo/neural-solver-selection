import torch
import pickle
import argparse
import json
import os
import yaml
import wandb
import random
import datetime
import torchmetrics
import csv
import numpy as np

from tqdm import tqdm
from utils import *
from torch.utils.data import Dataset
from dataset import SelectionDataset
from model import Selection_model, Naive_classifier
from trainer import trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load config')
    parser.add_argument('--config_name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--loss', type=str, default=None)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--test_file', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--gpu_id', type=int, default=None)
    args = parser.parse_args()
    
    if args.load is not None:
        with open(f"train_logs/{args.load}/config.json", 'r', encoding='utf-8') as config_file:
            config = yaml.load(config_file.read(), Loader=yaml.FullLoader)
            config['load_path'] = args.load
    else:
        with open(args.config_name, 'r', encoding='utf-8') as config_file:
            config = yaml.load(config_file.read(), Loader=yaml.FullLoader)
    
    # params
    name = config['name']
    seed = args.seed if args.seed is not None else config['seed']
    config['train_params']['loss'] = args.loss if args.loss is not None else config['train_params']['loss']
    seed_everything(seed)
    logger_name = config['logger']
    load_path = config['load_path']
    config['model_params']['problem_type'] = config['problem_type']
    config['model_params']['output_dim'] = config['train_params']['num_classes'] 

    # Initialize logger
    ts = datetime.datetime.utcnow() + datetime.timedelta(hours=+8)
    ts_name = f'-ts{ts.month}-{ts.day}-{ts.hour}-{ts.minute}-{ts.second}'
    log_config = config.copy()
    param_config = log_config['train_params'].copy()
    log_config.pop('train_params')
    model_params_config = log_config['model_params'].copy()
    log_config.pop('model_params')
    log_config.update(param_config)
    log_config.update(model_params_config)
    logger = {}
    if(logger_name == 'wandb'):
        logger['wandb'] = wandb.init(project="selection",
                         name=name + ts_name,
                         config=log_config)
    else:
        logger['wandb'] = None
    
    if args.test_file is None:
        if load_path is not None:
            log_dir = f'train_logs/{load_path}'
        else:
            log_dir = f'train_logs/{args.config_name}_{args.loss}_{args.seed}'
            os.mkdir(log_dir)

        logger['file'] = csv_logger(log_dir)
        
        if not os.path.exists(log_dir + '/config.json'):
            with open(log_dir + '/config.json', 'w') as f:
                json.dump(config, f)
    else:
        config['train_params']['num_epochs'] = 0    # only test
        log_dir = 'results'
        logger['file'] = csv_logger(log_dir, args.exp_name)
    print(config)

    # Prepare datasets
    name = args.test_file if args.test_file is not None else config['name']
    train_set, train_label, test_set, test_label = prepare_dataset(config['problem_type'], name=name)

    train_dataset = SelectionDataset(train_set, train_label, manual_feature=config['train_params']['manual_feature'], data_aug=config['train_params']['data_aug'])
    test_dataset = SelectionDataset(test_set, test_label, manual_feature=config['train_params']['manual_feature'])
    config['model_params']['ns_feature'] = config['train_params']['ns_feature']
    if config['train_params']['ns_feature']:
        representative_set = representative(train_set, train_label)
        model = Selection_model(**config['model_params'])
        encoder_representative = model.encoder
    else:
        representative_set = None
        encoder_representative = None
        
    # Initialize models
    if config['train_params']['manual_feature']:
        model = Naive_classifier(**config['model_params'])
    else:
        model = Selection_model(**config['model_params'])

    # Initialize trainer
    cuda_device_num = config['cuda_device_num'] if args.gpu_id is None else args.gpu_id
    trainer = trainer(model=model,
        logger=logger, 
        cuda_device_num=cuda_device_num,
        encoder_representative=encoder_representative,
        train_params=config['train_params'])
    
    # Training
    trainer.run(train_dataset, test_dataset, representative_set, log_dir, load_path)