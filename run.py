import torch
import argparse
from utils.utils import prepare_save_dir, create_logger, save_logger
from engine import Engine
from datasets.data import CIFAR10_loader_single, CIFAR100_loader_single
from datasets.cifar_dataset import CIFAR10, CIFAR100

# fix too many open file error
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

def get_args_parser():
    parser = argparse.ArgumentParser('parameters', add_help=False)
    parser.add_argument('--dataset', default='cifar10', help='dataset setting')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--name', type=str, default='debug')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=5e-2)
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size')
    return parser


def main(args):

    args.name = "_".join([args.dataset, args.name])
    args = prepare_save_dir(args, __file__)

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'cifar10':

        labeled_list = range(5)
        unlabeled_list = range(5,10)
        data_loader = CIFAR10_loader_single

    elif args.dataset == 'cifar100':

        labeled_list = range(50)
        unlabeled_list = range(50, 100)
        data_loader = CIFAR100_loader_single

    elif args.dataset == 'imgnet':

        pass

    else:

        print("not supported dataset")

    engine = Engine(args, labeled_list=labeled_list, 
                    unlabeled_list=unlabeled_list,
                    dataloader=data_loader)
    engine.train()


if __name__ == "__main__":

    parser = argparse.ArgumentParser("pararmeters", parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)