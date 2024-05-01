import pickle

import torch
import torch.nn as nn
from .model_utils import build_glove_embedding_dic
import numpy as np


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)


class DUET(torch.nn.Module):

    def __init__(self, config):
        super(DUET, self).__init__()
        self.config = config
        self.num_docs = 1
        self.config['POOLING_KERNEL_WIDTH_QUERY'] = self.config['MAX_QUERY_TERMS'] - self.config['TERM_WINDOW_SIZE'] + 1
        self.config['NUM_POOLING_WINDOWS_DOC'] = self.config['MAX_DOC_TERMS'] - self.config['TERM_WINDOW_SIZE'] + 1 - self.config['POOLING_KERNEL_WIDTH_DOC'] + 1
        vocab, embeddings = build_glove_embedding_dic(self.config['embed'], self.config['vocab_path'])
        self.embed = nn.Embedding(embeddings.shape[0], embeddings.shape[1], padding_idx=self.config['unk_token'])
        self.embed.weight = nn.Parameter(torch.tensor(embeddings), requires_grad=False)
        # self.embed.weight.data.copy_(torch.from_numpy(np.asarray(list(self.weights.values())))).to(
        #     self.config['device'])
        self.duet_local = nn.Sequential(nn.Conv1d(self.config['MAX_DOC_TERMS'], self.config['NUM_HIDDEN_NODES'], kernel_size=1),
                                        nn.ReLU(),
                                        Flatten(),
                                        nn.Dropout(p=self.config['DROPOUT_RATE']),
                                        nn.Linear(self.config['NUM_HIDDEN_NODES'] * self.config['MAX_QUERY_TERMS'], self.config['NUM_HIDDEN_NODES']),
                                        nn.ReLU(),
                                        nn.Dropout(p=self.config['DROPOUT_RATE']),
                                        nn.Linear(self.config['NUM_HIDDEN_NODES'], self.config['NUM_HIDDEN_NODES']),
                                        nn.ReLU(),
                                        nn.Dropout(p=self.config['DROPOUT_RATE']))
        self.duet_dist_q = nn.Sequential(nn.Conv1d(self.config['NUM_HIDDEN_NODES'], self.config['NUM_HIDDEN_NODES'], kernel_size=3),
                                         nn.ReLU(),
                                         nn.MaxPool1d(self.config['POOLING_KERNEL_WIDTH_QUERY']),
                                         Flatten(),
                                         nn.Linear(self.config['NUM_HIDDEN_NODES'], self.config['NUM_HIDDEN_NODES']),
                                         nn.ReLU()
                                         )
        self.duet_dist_d = nn.Sequential(nn.Conv1d(self.config['NUM_HIDDEN_NODES'], self.config['NUM_HIDDEN_NODES'], kernel_size=3),
                                         nn.ReLU(),
                                         nn.MaxPool1d(self.config['POOLING_KERNEL_WIDTH_DOC'], stride=1),
                                         nn.Conv1d(self.config['NUM_HIDDEN_NODES'], self.config['NUM_HIDDEN_NODES'], kernel_size=1),
                                         nn.ReLU()
                                         )
        self.duet_dist = nn.Sequential(Flatten(),
                                       nn.Dropout(p=self.config['DROPOUT_RATE']),
                                       nn.Linear(self.config['NUM_HIDDEN_NODES'] * self.config['NUM_POOLING_WINDOWS_DOC'], self.config['NUM_HIDDEN_NODES']),
                                       nn.ReLU(),
                                       nn.Dropout(p=self.config['DROPOUT_RATE']),
                                       nn.Linear(self.config['NUM_HIDDEN_NODES'], self.config['NUM_HIDDEN_NODES']),
                                       nn.ReLU(),
                                       nn.Dropout(p=self.config['DROPOUT_RATE']))
        self.duet_comb = nn.Sequential(nn.Linear(self.config['NUM_HIDDEN_NODES'], self.config['NUM_HIDDEN_NODES']),
                                       nn.ReLU(),
                                       nn.Dropout(p=self.config['DROPOUT_RATE']),
                                       nn.Linear(self.config['NUM_HIDDEN_NODES'], self.config['NUM_HIDDEN_NODES']),
                                       nn.ReLU(),
                                       nn.Dropout(p=self.config['DROPOUT_RATE']),
                                       nn.Linear(self.config['NUM_HIDDEN_NODES'], 1),
                                       nn.ReLU())
        self.scale = torch.tensor([0.8], requires_grad=False).to(self.config['device'])

    def forward(self, q, d):
        x_local, x_dist_q, x_dist_d, x_mask_q, x_mask_d = self.create_duet_features(q, d)
        h_local = self.duet_local(x_local)
        h_dist_q = self.duet_dist_q((self.embed(x_dist_q) * x_mask_q).permute(0, 2, 1))
        h_dist_d = self.duet_dist_d((self.embed(x_dist_d) * x_mask_d).permute(0, 2, 1))
        h_dist = self.duet_dist(h_dist_q.unsqueeze(-1) * h_dist_d)
        y_score = self.duet_comb((h_local + h_dist)) * self.scale
        return y_score.squeeze(dim=-1)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __allocate_minibatch(self, batch_size):
        local = np.ones((batch_size, self.config['MAX_DOC_TERMS'],
                                                self.config['MAX_QUERY_TERMS']), dtype=np.float32)
        return local

    def create_duet_features(self, query, docs):
        if 'local_feature' in docs.keys():
            x_local = docs['local_feature']
        else:
            batch_size = query['input_ids'].size(dim=0)
            local = self.__allocate_minibatch(batch_size)
            x_local = torch.from_numpy(local).to(self.config['device'])
        return x_local, query['input_ids'], docs['input_ids'], \
               query['attention_mask'].unsqueeze(dim=-1), docs['attention_mask'].unsqueeze(dim=-1)