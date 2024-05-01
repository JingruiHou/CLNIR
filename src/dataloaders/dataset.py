# encoding=utf-8
import os
import numpy as np
import torch
from tqdm import tqdm
import pickle
from transformers import AutoTokenizer
import gc


class Dataset_test:
    def __init__(self, topic_name, config):
        self.config = config
        self.unk_token = config['unk_token']
        if self.config['need_local_feature']:
            self.need_local_feature = self.config['need_local_feature']
        else:
            self.need_local_feature = False
        _files = ['topic.{}.test.queries.pkl', 'topic.{}.test.doc.pkl', 'topic.{}.test.label.pkl']
        pkls = []
        for _file in _files:
            print(f'loading ' + _file.format(topic_name))
            with open(os.path.join(config['dataset_path'], _file.format(topic_name)), 'rb') as f:
                pkls.append(pickle.load(f))
        self.labels = pkls[2]
        self.queries = pkls[0]
        self.documents = pkls[1]
        if self.need_local_feature:
            pointer_file = 'topic.{}.test.idf.num.pkl'.format(topic_name)
            local_file = 'topic.{}.test.idf.csv'.format(topic_name)
            with open(os.path.join(config['dataset_path'], pointer_file.format(topic_name)), 'rb') as f:
                self.pointers = pickle.load(f)
            self.local_feature_reader = open(os.path.join(config['dataset_path'], local_file), 'r', encoding='utf8')

    def __getitem__(self, index):
        q_ids, d_ids, labels = self.queries[index], self.documents[index], self.labels[index]
        q_data, q_mask = self.pad2longest(q_ids, self.config['query_max_len'], self.unk_token)
        d_data, d_mask = self.pad2longest(d_ids, self.config['target_max_len'], self.unk_token)
        q_tensor_data = {'input_ids': torch.LongTensor(q_data),
                         'attention_mask': torch.FloatTensor(q_mask)}
        d_tensor_data = {'input_ids': torch.LongTensor(d_data),
                           'attention_mask': torch.FloatTensor(d_mask),
                         }
        if self.need_local_feature:
            local_feature = np.zeros((self.config['target_max_len'], self.config['query_max_len']),
                                     dtype=np.float32)
            self.local_feature_reader.seek(self.pointers[index])
            local_value_line = self.local_feature_reader.readline()
            if local_value_line != '\t\t\n':
                arrs = local_value_line.strip().split('\t')
                assert len(arrs) == 3
                Xs = arrs[0].split(',')
                Ys = arrs[1].split(',')
                Vs = arrs[2].split(',')
                assert len(Xs) == len(Ys)
                assert len(Xs) == len(Vs)
                for idx in range(len(Xs)):
                    local_feature[int(Xs[idx]), int(Ys[idx])] = float(Vs[idx])
            d_tensor_data['local_feature'] = torch.from_numpy(local_feature)
        return q_tensor_data, d_tensor_data, torch.LongTensor([labels])

    @staticmethod
    def pad2longest(data_ids, max_len, unk_token):
        if isinstance(data_ids[0], list):
            s_data = np.array([s[:max_len] + [unk_token] * (max_len - len(s[:max_len])) for s in data_ids])
            s_mask = np.array([[1] * m[:max_len] + [0] * (max_len - len(m[:max_len])) for m in data_ids])
        elif isinstance(data_ids[0], int):
            s_data = np.array(data_ids[:max_len] + [unk_token] * (max_len - len(data_ids[:max_len])))
            s_mask = np.array([1] * len(data_ids[:max_len]) + [0] * (max_len - len(data_ids[:max_len])))
        else:
            raise TypeError("list type is required")
        return (s_data, s_mask)

    def __len__(self):
        return len(self.queries)


class Dataset_train:
    def __init__(self, topic_name, config):
        self.config = config
        self.unk_token = config['unk_token']
        if self.config['need_local_feature']:
            self.need_local_feature = self.config['need_local_feature']
        else:
            self.need_local_feature = False
        _files = ['topic.{}.train.queries.large.pkl', 'topic.{}.train.pos.large.pkl', 'topic.{}.train.neg.large.pkl']
        pkls = []
        for _file in _files:
            _f_path = os.path.join(config['dataset_path'], _file.format(topic_name))
            if not os.path.exists(_f_path):
                _f_path = _f_path.replace('large.', '')
            print(f'loading ' + _f_path)
            with open(_f_path, 'rb') as f:
                pkls.append(pickle.load(f))
        self.queries = pkls[0]
        self.pos_documents = pkls[1]
        self.neg_documents = pkls[2]

        if self.need_local_feature:
            local_file = 'topic.{}.train.pos.idf.num.pkl'
            with open(os.path.join(config['dataset_path'], local_file.format(topic_name)), 'rb') as f:
                self.pos_pointer = pickle.load(f)
            local_file = 'topic.{}.train.neg.idf.num.pkl'
            with open(os.path.join(config['dataset_path'], local_file.format(topic_name)), 'rb') as f:
                self.neg_pointer = pickle.load(f)
            self.pos_local_feature_reader = open(os.path.join(config['dataset_path'],
                                                              'topic.{}.train.pos.idf.csv'.format(topic_name)), 'r', encoding='utf8')
            self.neg_local_feature_reader = open(os.path.join(config['dataset_path'],
                                                              'topic.{}.train.neg.idf.csv'.format(topic_name)), 'r', encoding='utf8')

    def __getitem__(self, index):
        q_ids, pos_ids, neg_ids = self.queries[index], self.pos_documents[index], self.neg_documents[index]
        q_data, q_mask = self.pad2longest(q_ids, self.config['query_max_len'], self.unk_token)
        pos_data, pos_mask = self.pad2longest(pos_ids, self.config['target_max_len'], self.unk_token)
        neg_data, neg_mask = self.pad2longest(neg_ids, self.config['target_max_len'], self.unk_token)
        q_tensor_data = {'input_ids': torch.LongTensor(q_data),
                         'attention_mask': torch.FloatTensor(q_mask)}
        pos_tensor_data = {'input_ids': torch.LongTensor(pos_data),
                           'attention_mask': torch.FloatTensor(pos_mask)}
        neg_tensor_data = {'input_ids': torch.LongTensor(neg_data),
                           'attention_mask': torch.FloatTensor(neg_mask)}
        if self.need_local_feature:
            local_features = []
            self.pos_local_feature_reader.seek(self.pos_pointer[index])
            self.neg_local_feature_reader.seek(self.neg_pointer[index])
            for local_value_line in [self.pos_local_feature_reader.readline(),
                                     self.neg_local_feature_reader.readline()]:
                local_feature = np.zeros((self.config['target_max_len'], self.config['query_max_len']),
                                         dtype=np.float32)
                if local_value_line != '\t\t\n':
                    arrs = local_value_line.strip().split('\t')
                    if len(arrs) != 3:
                        print(local_value_line)
                    assert len(arrs) == 3
                    Xs = arrs[0].split(',')
                    Ys = arrs[1].split(',')
                    Vs = arrs[2].split(',')
                    assert len(Xs) == len(Ys)
                    assert len(Xs) == len(Vs)
                    for idx in range(len(Xs)):
                        local_feature[int(Xs[idx]), int(Ys[idx])] = float(Vs[idx])
                local_features.append(local_feature)
            pos_tensor_data['local_feature'] = torch.from_numpy(local_features[0])
            neg_tensor_data['local_feature'] = torch.from_numpy(local_features[1])
        return q_tensor_data, pos_tensor_data, neg_tensor_data

    @staticmethod
    def pad2longest(data_ids, max_len, unk_token):
        if isinstance(data_ids[0], list):
            s_data = np.array([s[:max_len] + [unk_token] * (max_len - len(s[:max_len])) for s in data_ids])
            s_mask = np.array([[1] * m[:max_len] + [0] * (max_len - len(m[:max_len])) for m in data_ids])
        elif isinstance(data_ids[0], int):
            s_data = np.array(data_ids[:max_len] + [unk_token] * (max_len - len(data_ids[:max_len])))
            s_mask = np.array([1] * len(data_ids[:max_len]) + [0] * (max_len - len(data_ids[:max_len])))
        elif isinstance(data_ids[0], str):
            s_data = data_ids
            s_mask = np.array([1] * len(data_ids.split(' ')[:max_len]) + [0] * (max_len - len(data_ids.split(' ')[:max_len])))
        else:
            raise TypeError("list type is required")
        return (s_data, s_mask)

    def __len__(self):
        return len(self.queries)


class BERT_Dataset_train:
    def __init__(self, topic_name, config):
        self.config = config
        if self.config['need_merge']:
            self.need_merge = self.config['need_merge']
        else:
            self.need_merge = False
        _files = ['topic.{}.train.bert.queries.pkl', 'topic.{}.train.bert.pos.pkl', 'topic.{}.train.bert.neg.pkl']
        pkls = []
        for _file in _files:
            _f_path = os.path.join(config['dataset_path'], _file.format(topic_name))
            print(f'loading ' + _file.format(topic_name))
            with open(_f_path, 'rb') as f:
                pkls.append(pickle.load(f))
        # lens = int(len(pkls[2])*1)*0 + 100
        lens = len(pkls[2])
        self.queries = pkls[0][:lens]
        self.pos_documents = pkls[1][:lens]
        self.neg_documents = pkls[2][:lens]

    def __getitem__(self, index):
        q_ids, pos_ids, neg_ids = self.queries[index], self.pos_documents[index], self.neg_documents[index]
        q_data, q_mask = self.bert_tokens(q_ids)
        q_tensor_data = {'input_ids': torch.LongTensor(q_data),
                         'attention_mask': torch.FloatTensor(q_mask)}
        if self.need_merge:
            q_0_len = self.list_end_zero_num(q_ids)
            pos_data, pos_mask = self.bert_tokens(q_ids[:-q_0_len] + pos_ids[1:] + q_ids[-q_0_len:])
            neg_data, neg_mask = self.bert_tokens(q_ids[:-q_0_len] + neg_ids[1:] + q_ids[-q_0_len:])
        else:
            pos_data, pos_mask = self.bert_tokens(pos_ids)
            neg_data, neg_mask = self.bert_tokens(neg_ids)

        pos_tensor_data = {'input_ids': torch.LongTensor(pos_data),
                           'attention_mask': torch.FloatTensor(pos_mask)}
        neg_tensor_data = {'input_ids': torch.LongTensor(neg_data),
                           'attention_mask': torch.FloatTensor(neg_mask)}
        return q_tensor_data, pos_tensor_data, neg_tensor_data

    @staticmethod
    def bert_tokens(data_ids, padding_data=False):
        if not padding_data:
            s_data = np.array(data_ids)
            s_mask = np.ones((len(data_ids)))
            token = 1
            for k in data_ids[::-1]:
                if k == 0:
                    s_mask[-token] = 0
                    token += 1
                else:
                    break
        else:
            pass
        return (s_data, s_mask)

    @staticmethod
    def list_end_zero_num(data_ids):
        n_0 = 0
        for k in data_ids[::-1]:
            if k == 0:
                n_0 += 1
            else:
                break
        return n_0

    def __len__(self):
        return len(self.queries)


class BERT_Dataset_test:
    def __init__(self, topic_name, config):
        self.config = config
        if self.config['need_merge']:
            self.need_merge = self.config['need_merge']
        else:
            self.need_merge = False
        _files = ['topic.{}.test.bert.queries.pkl', 'topic.{}.test.bert.doc.pkl', 'topic.{}.test.bert.label.pkl']
        pkls = []
        for _file in _files:
            print(f'loading ' + _file.format(topic_name))
            with open(os.path.join(config['dataset_path'], _file.format(topic_name)), 'rb') as f:
                pkls.append(pickle.load(f))
        #lens = int(len(pkls[2])*1)*0 + 100
        lens = len(pkls[2])
        self.queries = pkls[0][:lens]
        self.documents = pkls[1][:lens]
        self.labels = pkls[2][:lens]

    def __getitem__(self, index):
        q_ids, pos_ids, labels = self.queries[index], self.documents[index], self.labels[index]
        q_data, q_mask = self.bert_tokens(q_ids)
        q_tensor_data = {'input_ids': torch.LongTensor(q_data),
                         'attention_mask': torch.FloatTensor(q_mask)}
        if self.need_merge:
            q_0_len = self.list_end_zero_num(q_ids)
            d_data, d_mask = self.bert_tokens(q_ids[:-q_0_len] + pos_ids[1:] + q_ids[-q_0_len:])
        else:
            d_data, d_mask = self.bert_tokens(pos_ids)
        doc_tensor_data = {'input_ids': torch.LongTensor(d_data),
                           'attention_mask': torch.FloatTensor(d_mask)}
        return q_tensor_data, doc_tensor_data, torch.LongTensor([labels])

    @staticmethod
    def bert_tokens(data_ids, padding_data=False):
        if not padding_data:
            s_data = np.array(data_ids)
            s_mask = np.ones((len(data_ids)))
            token = 1
            for k in data_ids[::-1]:
                if k == 0:
                    s_mask[-token] = 0
                    token += 1
                else:
                    break
        else:
            pass
        return (s_data, s_mask)

    @staticmethod
    def list_end_zero_num(data_ids):
        n_0 = 0
        for k in data_ids[::-1]:
            if k == 0:
                n_0 += 1
            else:
                break
        return n_0

    def __len__(self):
        return len(self.queries)
