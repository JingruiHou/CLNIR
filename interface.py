# encoding=utf-8
import sys
import argparse
from src.utils.utils import loadyaml, print_message
from src import agents, dataloaders
import torch
from torch.utils.data import DataLoader
import gc
import numpy as np, random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_args(argv):
    # This function prepares the variables shared across demo.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', type=str, default='Adam', help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad"
                                                                      "|Adamax ...")
    parser.add_argument('--workers', type=int, default=0, help="#Thread for dataloader")
    parser.add_argument('--model_weights', type=str, default=None,
                        help="The path to the file for the model weights (*.pth).")
    parser.add_argument('--output_dir', type=str, default='//home/lunet/cojh6/PycharmProjects/pythonProject/clnir/output.{}.{}.{}.pkl',
                        help="The path to save result.")
    parser.add_argument('--model_path', type=str, default='/home/lunet/cojh6/PycharmProjects/pythonProject/clnir/models/model.{}.{}.{}.eps.{}.pth',
                        help="The path to save model parameters.")
    parser.add_argument('--best_epoch_path', type=str, default='/home/lunet/cojh6/PycharmProjects/pythonProject/clnir/results/best_epoch.txt',
                        help="The path to save model parameters.")
    parser.add_argument('--print_freq', type=float, default=500, help="print the log  at every x iteration")
    parser.add_argument('--gpuid', type=float, default=[0], help="gpu ID")
    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    # default configuration of neural ranker
    # 设置随机数种子
    setup_seed(42)
    ranker_config_pool = ['src/conf/ranker.knrm.yaml',
                      'src/conf/ranker.drmm.yaml',
                      'src/conf/ranker.duet.yaml',
                      'src/conf/ranker.colbert.yaml',
                      'src/conf/ranker.bertdot.yaml'
    ]

    strategy_config_pool = [
        'src/conf/strategy_None.yaml',
        'src/conf/strategy_L2.yaml',
        'src/conf/strategy_SI.yaml',
        'src/conf/strategy_MAS.yaml',
        'src/conf/strategy_EWC_online.yaml',
        'src/conf/strategy_EWC.yaml',
        'src/conf/strategy_Naive_Rehearsal.yaml',
        'src/conf/strategy_GEM.yaml',
    ]
    for ranker_config_path in ranker_config_pool:
        for strategy_config_path in strategy_config_pool:
            ranker_config = loadyaml(ranker_config_path)
            # default configuration of learning strategy
            strategy_config = loadyaml(strategy_config_path)
            agent_config = {**ranker_config, **strategy_config}
            args = get_args(sys.argv[1:])
            agent_config.update(dict(model_weights=args.model_weights, optimizer=args.optimizer,
                                     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                     unique_queries='total_unique_queries.pkl', test_label='topic.{}.test.label.pkl',
                                     test_queries='topic.{}.test.queries.pkl', output_dir=args.output_dir,
                                     model_path=args.model_path, best_epoch_path=args.best_epoch_path,
                                     print_freq=args.print_freq, gpuid=args.gpuid, weight_decay=0, workers=0))

            topics = ['IT', 'furnishing', 'food', 'health',  'tourism', 'finance']
            topics = [s+'.demo' for s in topics]
            agent_config['topic_names'] = topics
            agent = agents.__dict__[agent_config['agent_type']].__dict__[agent_config['agent_name']](agent_config)
            D_train = dataloaders.__dict__['dataset'].__dict__[agent_config['train_type']]
            D_test = dataloaders.__dict__['dataset'].__dict__[agent_config['test_type']]

            for t_id, topic in enumerate(topics):
                print_message("Begin to build topic dataset")
                train_dataset = D_train(topic,  agent_config)
                train_loader = DataLoader(train_dataset, batch_size=agent_config['batch_size'], shuffle=True, num_workers=agent_config['workers'])
                test_dataset = D_test(topic, agent_config)
                test_loader = DataLoader(test_dataset, batch_size=agent_config['batch_size'], shuffle=False,
                                         num_workers=agent_config['workers'])
                print_message("Begin to train dataset :{}_{}".format(t_id, topic))
                agent.learn_batch(train_loader, test_loader)

