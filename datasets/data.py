import numpy as np
import torch
from .cifar_dataset import CIFAR10, CIFAR100


def CIFAR10_loader_single(batch_size, split='train', num_workers=0,
                        shuffle=True, target_list=range(5), labeled=True, drop_ratio=0.5):


    dataset = CIFAR10(mode=split, target_list=target_list, labeled=labeled, drop_ratio=drop_ratio)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=shuffle,
                                         num_workers=num_workers)
    return loader


def CIFAR10_loader_mix(batch_size, split='train', num_workers=0,
                       shuffle=True, labeled_list=range(5), unlabeled_list=range(5, 10)):

    dataset_labeled = CIFAR10(mode=split, target_list=labeled_list)
    dataset_unlabeled = CIFAR10(mode=split, target_list=unlabeled_list, labeled=False)
    dataset_labeled.labels = np.concatenate(
        (dataset_labeled.labels, dataset_unlabeled.labels))
    dataset_labeled.imgs = np.concatenate(
        (dataset_labeled.imgs, dataset_unlabeled.imgs), 0)
    loader = torch.utils.data.DataLoader(dataset_labeled,
                                         batch_size=batch_size,
                                         shuffle=shuffle,
                                         num_workers=num_workers)

    return loader


def CIFAR100_loader_single(batch_size, split='train', num_workers=2,
                        shuffle=True, target_list=range(50), labeled=True):

    dataset = CIFAR100(mode=split, target_list=target_list, labeled=labeled)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=shuffle,
                                         num_workers=num_workers)
    return loader


def CIFAR100_loader_mix(batch_size, split='train', num_workers=2,
                       shuffle=True, labeled_list=range(50), unlabeled_list=range(50, 100)):

    dataset_labeled = CIFAR100(mode=split, target_list=labeled_list)
    dataset_unlabeled = CIFAR100(mode=split, target_list=unlabeled_list, labeled=False)
    dataset_labeled.labels = np.concatenate(
        (dataset_labeled.labels, dataset_unlabeled.labels))
    dataset_labeled.imgs = np.concatenate(
        (dataset_labeled.imgs, dataset_unlabeled.imgs), 0)
    loader = torch.utils.data.DataLoader(dataset_labeled,
                                         batch_size=batch_size,
                                         shuffle=shuffle,
                                         num_workers=num_workers)

    return loader