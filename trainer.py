import torch
import csv
import time
import torchmetrics
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils import *
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from dataset import SelectionDataset, collate_fn
from loss import RankingLoss


class trainer():
    def __init__(self, model, encoder_representative, logger, cuda_device_num, train_params):
        super().__init__()
        self.train_params = train_params
        # cuda
        if cuda_device_num == -1:
            self.device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        else:
            torch.cuda.set_device(cuda_device_num)
            self.device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        self.model = model.to(self.device)
        self.optimizer = None
        self.criterion = None
        self.encoder_p = None
        self.logger = logger
        if encoder_representative is not None:
            self.m = 0.99
            self.encoder_p = encoder_representative.to(self.device)
            for param, param_p in zip(
                self.model.encoder.parameters(), self.encoder_p.parameters()
            ):
                param_p.data.copy_(param.data)  # initialize
                param_p.requires_grad = False  # not update by gradient

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train_params['learning_rate'], weight_decay=float(self.train_params['weight_decay']))            
    
    @torch.no_grad()
    def _momentum_update_representative_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param, param_p in zip(
            self.model.encoder.parameters(), self.encoder_p.parameters()
        ):
            param_p.data = param_p.data * self.m + param.data * (1.0 - self.m)

    def run(self, train_dataset, test_dataset, representative_set, log_dir, load_path=None):
        # Criterion
        if self.train_params['loss'] == 'rank':
            num_ns = len(train_dataset[0][1][1])
            self.criterion = RankingLoss(num_ns, num_ns)
        elif self.train_params['loss'] == 'CE':
            self.criterion = torch.nn.NLLLoss()

        # Dataloader
        train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=self.train_params['train_batch_size'], shuffle=True, generator=torch.Generator(device=self.device))
        test_dataloader = DataLoader(test_dataset, self.train_params['test_batch_size'], collate_fn=collate_fn, generator=torch.Generator(device=self.device))
        # Resume
        if load_path is not None:
            checkpoint_dict = torch.load('train_logs/' + load_path + '/checkpoint_epoch_best.pt', map_location='cpu')
            self.model.load_state_dict(checkpoint_dict['model_state_dict'])
            if self.encoder_p is not None:
                self.encoder_p.load_state_dict(checkpoint_dict['encoder_p_state_dict'])
            self.optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
            print("Checkpoint is loaded from {}".format('train_logs/' + load_path + '/checkpoint_epoch_best.pt'))

        # Train
        results = self.test(0, test_dataloader, representative_set)
        best_top1 = None
        for epoch in range(self.train_params['num_epochs'] - self.train_params['start_epochs']):
            print("Training {}/{} epoch: ".format(self.train_params['start_epochs'] + epoch, self.train_params['num_epochs']))
            grad_norm, loss_mean = self.train_one_epoch(epoch, train_dataloader, representative_set)
            results = self.test(epoch, test_dataloader, representative_set)
            # logging
            self.logger['file'].write(results)
            if self.logger['wandb'] is not None:
                self.logger['wandb'].log(results, step=epoch)

            # save checkpoint
            top_1 = results['top_1']
            if epoch == 0:
                best_top1 = top_1
            if epoch >= 1 and epoch % self.train_params['save_interval'] == 0:
                if top_1 < best_top1:
                    best_top1 = top_1
                    checkpoint_dict = {
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()
                        }
                    if self.encoder_p is not None:
                        checkpoint_dict['encoder_p_state_dict'] = self.encoder_p.state_dict()
                    torch.save(checkpoint_dict, log_dir + '/checkpoint_epoch_best.pt')

        # test logging
        self.logger['file'].write(results)

    def train_one_epoch(self, epoch, train_dataloader, representative_set=None):
        # log info
        gradient_norm = 0.
        loss_mean = 0.
        # compute representative feature
        representative_feature = self.compute_representative(representative_set) if representative_set is not None else None
        # training
        self.model.train()
        for batch in tqdm(train_dataloader):
            x = batch[0].to(self.device)
            y = batch[1].to(self.device)
            cost = batch[2].to(self.device)
            scales = batch[3].to(self.device)
            mask = batch[4].to(self.device)
            if self.train_params['manual_feature']:
                manual_feature = batch[7].to(self.device)
            else:
                manual_feature = None
            # Forward
            if representative_set is not None:
                self.model.update_tokens(representative_feature)
            y_pred = self.model(x, scales, manual_feature, mask)

            # Backward
            self.optimizer.zero_grad()
            if self.train_params['loss'] == 'CE':
                l = self.criterion(F.log_softmax(y_pred, 1), y)
            if self.train_params['loss'] == 'rank':
                l = self.criterion(y_pred, cost)
                
            l.retain_grad()
            l.backward()

            self.optimizer.step()
            # update representative feature
            if representative_set is not None:
                self._momentum_update_representative_encoder()
                representative_feature = self.compute_representative(representative_set)

            for p in self.model.parameters():
                if p.grad is not None:
                    gradient_norm += p.grad.detach().norm(2).item()

            loss_mean += l.mean().item()

        # logging
        print("Loss mean: {:.4f}".format(loss_mean))
        
        return gradient_norm, loss_mean
    
    def test(self, epoch, test_dataloader, representative_set=None):
        # Metrics
        test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.train_params['num_classes'])
        test_recall = torchmetrics.Recall(task="multiclass", average='macro', num_classes=self.train_params['num_classes'])
        test_precision = torchmetrics.Precision(task="multiclass", average='macro', num_classes=self.train_params['num_classes'])
        gap_mat = []
        score_mat = []
        time_mat = []
        num_instances = 0.
        # Compute representative feature
        representative_feature = self.compute_representative(representative_set) if representative_set is not None else None

        # Test
        self.model.eval()
        start = time.time()
        for batch in test_dataloader:
            x = batch[0].to(self.device)
            y = batch[1].to(self.device)
            cost = batch[2].to(self.device)
            scales = batch[3].to(self.device)
            mask = batch[4].to(self.device)
            ind = batch[-1]
            gap = batch[5].to(self.device)
            time_cost = batch[6].to(self.device)
            if self.train_params['manual_feature']:
                manual_feature = batch[7].to(self.device)
            else:
                manual_feature = None
            # Forward
            if representative_set is not None:
                self.model.update_tokens(representative_feature)

            y_pred = self.model(x, scales, manual_feature, mask)
            
            # Eval
            test_acc(y_pred, y)
            test_recall(y_pred, y)
            test_precision(y_pred, y)

            score_mat.append(F.softmax(y_pred, 1).detach())
            time_mat.append(time_cost)
            gap_mat.append(gap)
            num_instances += x.shape[0]

        select_time = (time.time() - start) / num_instances
        print("Time consumption of inference per instance (in parallel): {:.4f}s".format(select_time))
        
        acc = test_acc.compute()
        recall = test_recall.compute()
        precision = test_precision.compute()
        results = {'acc': acc.item()}
        # Logging
        print("Accuracy: {:.4f}%     Recall: {:.4f}%    Precision: {:.4f}%".format(100 * acc, 100 * recall, 100 * precision))
        gap_mat = torch.cat(gap_mat, dim=0)
        time_mat = torch.cat(time_mat, dim=0)
        score_mat = torch.cat(score_mat, dim=0)
        single_best_gap, best_ind = torch.min(gap_mat.mean(dim=0), dim=0)
        single_best_time = torch.gather(time_mat.mean(dim=0), 0, best_ind)
        print("Single best: {:.4f}%, {:.4f}s      Oracle: {:.4f}%, {:.4f}s".format(single_best_gap, single_best_time, gap_mat.min(dim=1)[0].mean(), time_mat.sum(dim=1).mean()))
        
        # Top-k sampling
        record = False
        dataset = 'cvrplib'
        loss = 'rank'
        k_list = [1, 2, 3, 4]
        if record:
            f = open(f'plots/results/rejection_{dataset}_{loss}.csv', 'w')
            file_logger = csv.DictWriter(f, fieldnames=['id', 'gap', 'time'])
            file_logger.writeheader()
        for k in k_list:
            _, topk_ind = score_mat.topk(k, 1, largest=True)
            topk_gap = gap_mat.gather(1, topk_ind).min(dim=1)[0]
            topk_time = time_mat.gather(1, topk_ind).sum(dim=1)
            if k == 1:
                top_1_gap = topk_gap
                top_1_time = topk_time
            print("Top-{} gap mean: {:.4f}%, {:.4f}s".format(k, topk_gap.mean(), topk_time.mean() + select_time))
            results[f'top_{k}'] = topk_gap.mean().item()
            results[f'time_top_{k}'] = topk_time.mean().item() + select_time

            # Rejection
            if ((k >= 2) and (record == True)) or (k == 2):
                sort_ind = score_mat.max(dim=1)[0].sort(descending=True)[1].cpu().numpy()
                cover_rates = [0.8]
                if record:
                    cover_rates = np.arange(0, 0.9, 0.05)
                for rate in cover_rates:
                    threshold = int(num_instances * rate)
                    reject_ind = sort_ind[threshold:]
                    accept_ind = sort_ind[:threshold]
                    gap_SR = torch.cat((topk_gap[reject_ind], top_1_gap[accept_ind]), dim=0)
                    time_SR = torch.cat((topk_time[reject_ind], top_1_time[accept_ind]), dim=0)

                    cover_80 = gap_SR.mean().item()
                    time_cover_80 = time_SR.mean().item()
                    print("Rejection 20%: {:.4f}%, {:.4f}s".format(cover_80, time_cover_80))
                    if record:
                        file_logger.writerow({'id': rate, 'gap': cover_80, 'time': time_cover_80 + select_time})
                results['cover_80'] = cover_80
                results['time_cover_80'] = time_cover_80 + select_time

        # Top-p sampling
        p_values = [0.8]
        if record:
            f = open(f'plots/results/top-p_{dataset}_{loss}.csv', 'w')
            file_logger = csv.DictWriter(f, fieldnames=['id', 'gap', 'time'])
            file_logger.writeheader()
            p_values = np.arange(0.4, 0.96, 0.01)
        
        for p in p_values:
            times = []
            gaps = []
            acc_num = 0.
            for i in range(len(score_mat)):
                for j in range(1, score_mat.shape[1] + 1):
                    top_j, ind = score_mat[i].topk(j, largest=True)
                    if j == 1:
                        ind_ = ind
                    if top_j.sum() >= p:
                        times.append(time_mat[i][ind_].sum().item())
                        gaps.append(gap_mat[i][ind_].min().item())
                        if gap_mat[i].argmin() in ind_:
                            acc_num += 1
                        break
                    else:
                        ind_ = ind
            print(acc_num / num_instances)
            print("Top p {}%: {:.4f}%, {:.4f}s".format(100 * p, np.array(gaps).mean(), np.array(times).mean() + select_time))
            results[f'top-p'] = np.array(gaps).mean()
            results[f'time_top-p'] = np.array(times).mean() + select_time
            if record:
                file_logger.writerow({'id': p, 'gap': np.array(gaps).mean(), 'time': np.array(times).mean() + select_time})
    
        # Results of included models
        for i in range(gap_mat.shape[1]):
            print("model {}: {:.4f}%, {:.4f}s".format(i, gap_mat[:, i].mean(), time_mat[:, i].mean()))
        
        return results

    def compute_representative(self, representative_set):
        # compute neural solver features
        representative_feature = []
        representative_data = representative_set[0]
        representative_label = representative_set[1]
        for i in range(len(representative_data)):
            dataset = SelectionDataset(representative_data[i], representative_label[i])
            dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=len(dataset), generator=torch.Generator(device=self.device))
            for batch in dataloader:
                x = batch[0].to(self.device)
                scales = batch[3].to(self.device)
                mask = batch[4].to(self.device)
                with torch.no_grad():
                    representative_feature.append(
                        torch.cat((
                            self.encoder_p(x, mask),
                            scales[:, None]
                        ), dim=1))

        return representative_feature