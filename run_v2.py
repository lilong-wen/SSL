import torch
import argparse
from utils.utils import prepare_save_dir, create_logger, save_logger
from engine2 import train, test
from models.resnet_v2 import ResNet, BasicBlock
from datasets.cifarloader import CIFAR10LoaderMix, CIFAR10Loader

# fix too many open file error
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

def get_args_parser():
    parser = argparse.ArgumentParser('parameters', add_help=False)

    parser.add_argument("--lr", default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", default=1e-4, help="weight decay")
    parser.add_argument("--step_size", default=170, help="step size for lr scheduler")
    parser.add_argument("--gamma", default=0.1, help="gamma for lr scheduler")
    parser.add_argument("--epoch", default=100, help="training epoch")
    parser.add_argument("--num_labeled_classes", default=5, help="labeled classes")
    parser.add_argument("--num_unlabeled_classes", default=5, help="unlabeled classes")

    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--name", type=str, default="debug")
    parser.add_argument("--train_batch_size", type=int, default=1024)
    parser.add_argument("--test_batch_size", type=int, default=1000)
    parser.add_argument("--dataset_root", type=str, default="/home/wll/data")

    parser.add_argument("--device", default="cpu", help="device for cuda or cpu")
    

    parser.add_argument('--wd', type=float, default=5e-2)
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

    return parser


def main(args):

    args.name = "_".join([args.dataset, args.name])
    args = prepare_save_dir(args, __file__)

    args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet(BasicBlock, [2, 2, 2, 2], args.num_classes)

    model = model.to(args.device)

    if args.dataset == 'cifar10':


        train_loader = CIFAR10LoaderMix(root=args.dataset_root, batch_size=args.train_batch_size,
                                        split='train', aug='once', shuffle=True,
                                        labeled_list=range(args.num_labeled_classes),
                                        unlabeled_list=range(args.num_unlabeled_classes, args.num_classes))


        labeled_seen_eval_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.test_batch_size,
                                              split='test', aug='once', shuffle=False,
                                              target_list=range(args.num_labeled_classes))

        unlabeled_unseen_eval_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.test_batch_size,
                                              split='test', aug='once', shuffle=False,
                                              target_list=range(args.num_unlabeled_classes, args.num_classes))

        test_loader = CIFAR10LoaderMix(root=args.dataset_root, batch_size=args.train_batch_size,
                                        split='test', aug='once', shuffle=True,
                                        labeled_list=range(args.num_labeled_classes),
                                        unlabeled_list=range(args.num_unlabeled_classes, args.num_classes))

        # train(model, train_loader, unlabeled_unseen_eval_loader, args)
        train(model, train_loader, test_loader, args)




if __name__ == "__main__":

    parser = argparse.ArgumentParser("pararmeters", parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)