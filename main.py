import torch
import logging
from torch.utils.tensorboard import SummaryWriter
import argparse
import json
import pickle
import os
from fede import FedE
from kge_trainer import KGETrainer
import numpy as np
from fusion import train_fusion


def init_dir(args):
    # state
    if not os.path.exists(args.state_dir):
        os.makedirs(args.state_dir)

    # tensorboard log
    if not os.path.exists(args.tb_log_dir):
        os.makedirs(args.tb_log_dir)

    # logging
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)


def init_logger(args):
    log_file = os.path.join(args.log_dir, args.name + '.log')

    logging.basicConfig(
        format='%(asctime)s | %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        filename=log_file,
        filemode='a+'
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/FB15k237-Fed3.pkl', type=str, help='path for loading data')
    parser.add_argument('--name', default='fb15k237_fed3_fed', type=str, help='name of current experiment')
    parser.add_argument('--state_dir', '-state_dir', default='./state', type=str, help='directory for saving model state dict')
    parser.add_argument('--log_dir', '-log_dir', default='./log', type=str, help='directory for saving log')
    parser.add_argument('--tb_log_dir', '-tb_log_dir', default='./tb_log', type=str, help='directory for saving tensorboard log')
    parser.add_argument('--setting', default='Entire', choices=['FedE',
                                                                 'Isolation',
                                                                 'Collection',
                                                                 'Model_Fusion'], help='setting for current experiment')
    parser.add_argument('--mode', default='train', choices=['train', 'test'], help='model training or testing')

    parser.add_argument('--model', default='TransE', choices=['TransE', 'RotatE', 'DistMult', 'ComplEx'], help='specific KGE method for training KGE')

    # hyper parameter for KGE training on isolation or collection
    parser.add_argument('--one_client_idx', default=0, type=int, help='the client index on Isolation or Collection setting')
    parser.add_argument('--max_epoch', default=10000, type=int, help='the max training epoch on Isolation or Collection setting')
    parser.add_argument('--log_per_epoch', default=1, type=int, help='take log per epoch on Isolation or Collection setting')
    parser.add_argument('--check_per_epoch', default=10, type=int, help='do validation per epoch on Isolation or Collection setting')
    parser.add_argument('--isolation_name_list', default=None, type=list, help='list with names for experiments on isolation training of a dataset')

    parser.add_argument('--batch_size', default=512, type=int, help='batch size for training KGE on FedE, Isolation or Collection,')
    parser.add_argument('--test_batch_size', default=16, type=int, help='batch size for training KGE on FedE, Isolation or Collection,')
    parser.add_argument('--num_neg', default=256, type=int, help='number of negative sample for training KGE on FedE, Isolation or Collection,')
    parser.add_argument('--lr', default=0.001, type=int, help='learning rate for training KGE on FedE, Isolation or Collection,')

    # hyper parameter for FedE
    parser.add_argument('--max_round', default=10000, type=int, help='the max training round on FedE')
    parser.add_argument('--local_epoch', default=3, help='number of local training epochs on FedE')
    parser.add_argument('--fraction', default=1, type=float, help='client selection fraction each round on FedE setting')
    parser.add_argument('--log_per_round', default=1, type=int, help='take log per epoch on FedE setting')
    parser.add_argument('--check_per_round', default=5, type=int, help='do validation per epoch on FedE setting')

    parser.add_argument('--early_stop_patience', default=15, type=int, help='early stop patience for training')
    parser.add_argument('--gamma', default=10.0, type=float, help='gamma in self-adversarial loss')
    parser.add_argument('--epsilon', default=2.0, type=float)
    parser.add_argument('--hidden_dim', default=128, type=float)
    parser.add_argument('--gpu', default='1', type=str)
    parser.add_argument('--num_cpu', default=10, type=int)
    parser.add_argument('--adversarial_temperature', default=1.0, type=float)

    parser.add_argument('--seed', default=12345, type=int)
    parser.add_argument('--num_client', default=None, type=int, help='no need to specifiy')

    # parameter for model fusion
    parser.add_argument('--fusion_state', nargs=2, default=['fb15k237_fed3_transe_isolation', 'fb15k237_fed3_transe_fede'], help='the name of isolation and fed experiments for model fusion')

    args = parser.parse_args()

    # ONLY for Isolation, add client index in the end of name
    if args.setting == 'Isolation':
        args.name = f'{args.name}_client_{args.one_client_idx}'

    # random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # load data and get number of clients
    all_data = pickle.load(open(args.data_path, 'rb'))
    args.num_client = len(all_data)

    # init dir, logger and log args
    init_dir(args)
    init_logger(args)
    args_str = json.dumps(vars(args))
    logging.info(args_str)

    # assign cuda device
    args.gpu = torch.device('cuda:' + args.gpu)

    # init tensorboard
    writer = SummaryWriter(os.path.join(args.tb_log_dir, args.name))
    args.writer = writer

    if args.setting == 'FedE':
        learner = FedE(args, all_data)
        if args.mode == 'train':
            learner.train()
        elif args.mode == 'test':
            learner.before_test_load()
            learner.evaluate(istest=True)
    elif args.setting == 'Isolation':
        data = all_data[args.one_client_idx]
        learner = KGETrainer(args, data)
        if args.mode == 'train':
            learner.train()
        elif args.mode == 'test':
            learner.before_test_load()
            learner.evaluate(eval_split='test')
    elif args.setting == 'Collection':
        learner = KGETrainer(args, all_data)
        if args.mode == 'train':
            learner.train()
        elif args.mode == 'test':
            learner.before_test_load()
            learner.evaluate_multi(eval_split='test')
    elif args.setting == 'Model_Fusion':
        train_fusion(args, all_data, args.num_client, args.fusion_state)

