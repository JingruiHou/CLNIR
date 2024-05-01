from __future__ import print_function

import os.path

import pickle

import torch
import torch.nn as nn
from .. import models
from .. utils.utils import print_message, calculate_mrr
from ..utils.metric import pair_accuracy, AverageMeter, Timer
import numpy as np


class NormalRankNN(nn.Module):

    def __init__(self, agent_config):

        super(NormalRankNN, self).__init__()
        self.log = print_message if agent_config['print_freq'] > 0 else lambda \
            *args: None  # Use a void function to replace the print
        self.config = agent_config
        self.model = self.create_model()
        self.criterion_fn = torch.nn.MarginRankingLoss(margin=1, reduction='elementwise_mean')
        if agent_config['gpuid'][0] >= 0:
            self.cuda()
            self.gpu = True
        else:
            self.gpu = False
        self.init_optimizer()
        self.reset_optimizer = False
        self.valid_out_dim = 'ALL'
        self.task_ids = 0
        self.best_epochs = []

    def init_optimizer(self):
        optimizer_arg = {'params': self.model.parameters(),
                         'lr': self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD','RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'
        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[self.config['schedule']],
                                                              gamma=0.1)

    def create_model(self):
        cfg = self.config
        if cfg.get('from_config'):
            model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']].from_config(cfg)
        else:
            model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](cfg)
        # Load pre-trained weights
        if cfg['model_weights'] is not None:
            self.log('=> Load model weights:', cfg['model_weights'])
            model_state = torch.load(cfg['model_weights'],
                                     map_location=lambda storage, loc: storage)  # Load to CPU.
            model.load_state_dict(model_state)
            self.log('=> Load Done')
        return model

    def reload_model(self):
        cfg = self.config
        self.log('reloading best model weights...')
        model_state = torch.load(cfg['model_weights'],
                                 map_location=lambda storage, loc: storage)  # Load to CPU.
        self.model.load_state_dict(model_state)

    def forward(self, q, d):
        return self.model.forward(q, d)

    def predict(self, q, d):
        self.model.eval()
        out = self.forward(q, d)
        return out.detach()

    def offline_validation(self, dataloader, save_outputs=True):
        self.model.eval()
        total_scores = []
        for i, (q, d, label) in enumerate(dataloader):
            if self.gpu:
                with torch.no_grad():
                    for idx, item in q.items():
                        q[idx] = item.cuda()
                    for idx, item in d.items():
                        d[idx] = item.cuda()
            if ((self.config['print_freq'] > 0) and (i % self.config['print_freq'] == 0)) or (i + 1) == len(
                    dataloader):
                self.log('{}/{} batches have been been processed!'.format(i, len(dataloader)))
            scores = self.predict(q, d)
            score_values = scores.cpu().detach().numpy().tolist()
            total_scores += score_values
        if save_outputs:
            file_path = self.config['output_dir'].format(
                self.config['model_name'], self.config['agent_name'], self.config['model_id'], self.task_ids)
            with open(file_path, 'wb') as f:
                pickle.dump(total_scores, f)
        self.model.train()
        return total_scores

    def calculate_performance(self, total_scores):
        with open(os.path.join(self.config['dataset_path'], self.config['unique_queries']), 'rb') as f:
            total_unique_queries = pickle.load(f)
        with open(os.path.join(self.config['dataset_path'], self.config['test_label']).format(
                self.config['topic_names'][self.task_ids-1]),
                  'rb') as f:
            labels = pickle.load(f)
        with open(os.path.join(self.config['dataset_path'], self.config['test_queries']).format(
                self.config['topic_names'][self.task_ids-1]),
                  'rb') as f:
            queries = pickle.load(f)
        rels = {}
        for i in range(len(labels)):
            topic_name = self.config['topic_names'][self.task_ids-1]
            if '.demo' in topic_name:
                topic_name = topic_name.replace('.demo', '')
            qid = total_unique_queries[topic_name].index(tuple(queries[i]))
            if qid not in rels:
                rels[qid] = []
            if labels[i] == 1:
                rels[qid].append(i)

        retrieved = {}
        for i in range(len(total_scores)):
            topic_name = self.config['topic_names'][self.task_ids - 1]
            if '.demo' in topic_name:
                topic_name = topic_name.replace('.demo', '')
            qid = total_unique_queries[topic_name].index(tuple(queries[i]))
            if qid not in retrieved:
                retrieved[qid] = {}
            retrieved[qid][i] = total_scores[i]
        mrr = calculate_mrr(rels, retrieved)
        return mrr

    def criterion(self, score_pos, score_neg, **kwargs):
        y1 = torch.from_numpy(np.ones((score_pos.size(dim=0)), dtype=np.int)).cuda()
        loss = self.criterion_fn(score_pos, score_neg, y1)
        return loss

    def update_model(self, q, pos, neg):
        score_pos = self.forward(q, pos)
        score_neg = self.forward(q, neg)
        loss = self.criterion(score_pos, score_neg)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach(), score_pos, score_neg

    def learn_batch(self, train_loader, val_loader=None):
        self.task_ids += 1
        # if self.task_ids <=1:
        #     return
        if self.reset_optimizer:  # Reset optimizer before learning each task
            self.log('Optimizer is reset!')
            self.init_optimizer()
        if train_loader is not None:
            if self.task_ids > 1:
                self.config['schedule'] = 1
            max_mrr = -1
            for epoch in range(self.config['schedule']):
                data_timer = Timer()
                batch_timer = Timer()
                batch_time = AverageMeter()
                data_time = AverageMeter()
                losses = AverageMeter()
                acc = AverageMeter()

                # Config the model and optimizer
                self.log('Epoch:{0}'.format(epoch))
                self.model.train()
                self.scheduler.step(epoch)
                for param_group in self.optimizer.param_groups:
                    self.log('LR:', param_group['lr'])

                # Learning with mini-batch
                data_timer.tic()
                batch_timer.tic()
                self.log('Itr\t\t\t\t Time\t\t\t\t  Data\t\t\t\t  Loss\t\t\t\tAcc')
                indice = [e*0.1 for e in range(10)][1:] + list(range(5))[1:]
                check_points = [int(len(train_loader) * 0.2 * e) for e in indice]
                for i, (q, pos, neg) in enumerate(train_loader):
                    data_time.update(data_timer.toc())
                    if self.gpu:
                        for idx, item in q.items():
                            q[idx] = item.cuda()
                        for idx, item in pos.items():
                            pos[idx] = item.cuda()
                        for idx, item in neg.items():
                            neg[idx] = item.cuda()

                    loss, out1, out2 = self.update_model(q, pos, neg)
                    for idx, item in q.items():
                        q[idx].detach()
                    for idx, item in pos.items():
                        pos[idx].detach()
                    for idx, item in neg.items():
                        neg[idx].detach()

                    # measure accuracy and record loss
                    acc = accumulate_pair_acc(out1, out2, acc)
                    losses.update(loss, out1.size(0))
                    batch_time.update(batch_timer.toc())  # measure elapsed time
                    data_timer.toc()

                    if self.task_ids > 1 and i in check_points:
                        self.save_model(epoch_id='bst.{}'.format(i))

                    if ((self.config['print_freq']>0) and (i % self.config['print_freq'] == 0)) or (i+1)==len(train_loader):
                        self.log('[{0}/{1}]\t'
                              '{batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                              '{data_time.val:.4f} ({data_time.avg:.4f})\t'
                              '{loss.val:.3f} ({loss.avg:.3f})\t'
                              '{acc.val:.2f} ({acc.avg:.2f})'.format(
                            i, len(train_loader), batch_time=batch_time,
                            data_time=data_time, loss=losses, acc=acc))

                self.log(' * Train Acc {acc.avg:.3f} of epoch: '.format(acc=acc))
                if val_loader != None:
                    self.log('testing on validation set')
                    epoch_predictions = self.offline_validation(val_loader, save_outputs=False)
                    mrr = self.calculate_performance(epoch_predictions)
                    self.log('task_id: {}, on epoch :{}, current mrr score: {}'.
                             format(self.task_ids, epoch, mrr))
                    if mrr >= max_mrr:
                        max_mrr = mrr
                        self.log('current max mrr score: {}'.
                                 format(max_mrr))
                        self.save_model(epoch_id='bst')
                    else:
                        self.config['model_weights'] = self.config['model_path'].\
                            format(self.config['model_name'], self.config['agent_name'], self.task_ids, 'bst')
                        self.reload_model()
                        self.config['model_weights'] = None
                        self.best_epochs.append(epoch-1)
                        self.log('reload best model on previous epoch :{}, max mrr score: {}'.
                                 format(epoch-1, max_mrr))
                        break
                    self.best_epochs.append(epoch)

            with open(self.config['best_epoch_path'], 'a') as f:
                f.write('{}\t{} : {}\n'.format(self.config['model_name'], self.config['agent_name'], self.best_epochs[-1]))
            val_loader = None
        if val_loader != None:
            self.offline_validation(val_loader, save_outputs=True)

    def learn_stream(self, data, label):
        assert False,'No implementation yet'

    def add_valid_output_dim(self, dim=0):
        # This function is kind of ad-hoc, but it is the simplest way to support incremental class learning
        self.log('Incremental class: Old valid output dimension:', self.valid_out_dim)
        if self.valid_out_dim == 'ALL':
            self.valid_out_dim = 0  # Initialize it with zero
        self.valid_out_dim += dim
        self.log('Incremental class: New Valid output dimension:', self.valid_out_dim)
        return self.valid_out_dim

    def count_parameter(self):
        return sum(p.numel() for p in self.model.parameters())

    def save_model(self, epoch_id):
        model_state = self.model.state_dict()
        if isinstance(self.model, torch.nn.DataParallel):
            # Get rid of 'module' before the name of states
            model_state = self.model.module.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()
        file_name = self.config['model_path'].format(self.config['model_name'], self.config['agent_name'], self.task_ids, epoch_id)
        self.log('=> Saving model to:' + file_name)
        torch.save(model_state, file_name)
        self.log('=> Save Done')

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self


def accumulate_pair_acc(pre1, pred2, meter):
    meter.update(pair_accuracy(pre1, pred2))
    return meter

