import numpy as np
import re
import pickle
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Load config')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--compute_gaps', type=bool, default=False)
    args = parser.parse_args()

    # TSP
    # methods = ['bq', 'ELG', 'LEHD', 'T2T', 'T2T500', 'DIFUSCO', 'DIFUSCO500']

    # CVRP
    methods = ['bq', 'ELG', 'LEHD', 'Omni', 'MVMoE']

    labels = {}
    file_name = f'{args.dataset}/results/result_LEHD.txt'
    with open(file_name, 'r') as f:
        data = f.readlines()
        for line in data:
            line = line.strip().split(',')
            labels[line[0]] = {'cost': [], 'time':[], 'ind': [], 'gap': []}

    if args.compute_gaps:
        opts = {}
        opt_file_name = f'{args.dataset}/results/result_opt.txt'
        with open(opt_file_name, 'r') as f:
            data = f.readlines()
            for line in data:
                line = line.strip().split(',')
                opts[line[0]] = float(line[1])

    all_gaps = []
    for method in methods:
        file_name = f'{args.dataset}/results/result_{method}.txt'
        gaps = []
        with open(file_name, 'r') as f:
            data = f.readlines()
            for line in data:
                line = line.strip().split(',')
                line[0] = f'{line[0]}'
                labels[line[0]]['cost'].append(float(line[1]))
                labels[line[0]]['time'].append(float(line[2]))
                if args.compute_gaps:
                    if opts[line[0]] == 0:
                        opts[line[0]] = float(line[1])
                    labels[line[0]]['gap'].append(100 * (float(line[1]) - opts[line[0]]) / opts[line[0]])
                    gaps.append(100 * (float(line[1]) - opts[line[0]]) / opts[line[0]])
                    
                else:
                    labels[line[0]]['gap'].append(0)
                
        print(np.mean(gaps))
        all_gaps.append(gaps)
    all_gaps = np.array(all_gaps)
    print(np.mean(np.min(all_gaps, axis=0)))
        
    inds = []
    for k, v in labels.items():
        labels[k]['ind'] = np.argmin(np.array(labels[k]['cost']))
        inds.append(np.argmin(np.array(labels[k]['cost'])))

    data_file = f'{args.dataset}/raw_label.pkl'
    with open(data_file, 'wb') as f:
        pickle.dump(labels, f)

    wins = []
    for i in range(len(methods)):
        wins.append(np.sum(np.array(inds) == i))
    print(wins)