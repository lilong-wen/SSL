from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
import pickle

import random
import torch
import torch.utils.data as data
from .transform import TransformTwice, TransformKtimes, Transform1times, RandomTranslateWithReflect
import torchvision.transforms as transforms


class CIFAR10(data.Dataset):

    base_folder = 'cifar-10-batches-py'

    train_list = [
            'data_batch_1',
            'data_batch_2',
            'data_batch_3',
            'data_batch_4',
            'data_batch_5',
    ]

    test_list = [
            'test_batch',
    ]

    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
    }

    def __init__(self, root, split='train+test',
                 transform=None, target_transform=None,
                 target_list = range(5)):


        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform


        self.data = []
        self.targets = []

        if split == "train":
            loading_list = self.train_list
        elif split == "test":
            loading_list = self.test_list
        elif split == "train+test":
            loading_list = self.train_list.append(self.test_list)

        # now load the picked numpy arrays
        for file_name in loading_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry: # cifar10 label
                    self.targets.extend(entry['labels'])
                else: # cifar100 label
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self._load_meta()

        ind = [i for i in range(len(self.targets)) if self.targets[i] in target_list]

        self.data = self.data[ind]
        self.targets = np.array(self.targets)
        self.targets = self.targets[ind].tolist()


    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])

        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]

        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)


class CIFAR100(CIFAR10):

    base_folder = 'cifar-100-python'

    train_list = [
        'train',
    ]

    test_list = [
        'test',
    ]

    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
    }


def CIFAR10Data(root, split='train', aug=None, target_list=range(5)):
    if aug == None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif aug == 'twice':
        transform = TransformTwice(transforms.Compose([
            RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]))
    elif aug == 'once':
        transform = Transform1times(transforms.Compose([
            RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]))
    dataset = CIFAR10(root=root, split=split, transform=transform, target_list=target_list)
    return dataset


def CIFAR10Loader(root, batch_size, split='train', num_workers=2,  aug=None, shuffle=True, target_list=range(5)):
    dataset = CIFAR10Data(root, split, aug, target_list)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

def CIFAR10LoaderMix(root, batch_size, split='train',num_workers=2, aug=None, shuffle=True, labeled_list=range(5), unlabeled_list=range(5, 10)):
    dataset_labeled = CIFAR10Data(root=root, split=split, aug=aug, target_list=labeled_list)
    dataset_unlabeled = CIFAR10Data(root=root, split=split, aug=aug, target_list=unlabeled_list)
    dataset_labeled.targets = np.concatenate((dataset_labeled.targets, dataset_unlabeled.targets))
    dataset_labeled.data = np.concatenate((dataset_labeled.data, dataset_unlabeled.data),0)
    loader = data.DataLoader(dataset_labeled, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader


def CIFAR100Data(root, split='train', aug=None, target_list=range(80)):
    if aug==None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        ])
    elif aug=='twice':
        transform = TransformTwice(transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        ]))
    dataset = CIFAR100(root=root, split=split, transform=transform, target_list=target_list)
    return dataset


def CIFAR100Loader(root, batch_size, split='train', num_workers=2,  aug=None, shuffle=True, target_list=range(80)):
    dataset = CIFAR100Data(root, split, aug,target_list)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader


def CIFAR100LoaderMix(root, batch_size, split='train',num_workers=2, aug=None, shuffle=True, labeled_list=range(80), unlabeled_list=range(90, 100)):
    dataset_labeled = CIFAR100Data(root, split, aug, labeled_list)
    dataset_unlabeled = CIFAR100Data(root, split, aug, unlabeled_list)

    dataset_labeled.targets = np.concatenate((dataset_labeled.targets, dataset_unlabeled.targets))
    dataset_labeled.data = np.concatenate((dataset_labeled.data,dataset_unlabeled.data),0)
    loader = data.DataLoader(dataset_labeled, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader


if __name__ == "__main__":

    dataset_root = "/home/wll/data/"
    batch_size = 5
    num_labeled_classes = 5
    num_classes = 10

    mix_train_loader = CIFAR10LoaderMix(root=dataset_root, 
                                        batch_size=batch_size, 
                                        split='train', 
                                        aug='once', 
                                        shuffle=True, 
                                        labeled_list=range(num_labeled_classes), 
                                        unlabeled_list=range(num_labeled_classes, num_classes))

    unlabeled_eval_loader = CIFAR10Loader(root=dataset_root, 
                                          batch_size=batch_size, 
                                          split='train', 
                                          aug=None, 
                                          shuffle=False, 
                                          target_list = range(num_labeled_classes, num_classes))

    unlabeled_eval_loader_test = CIFAR10Loader(root=dataset_root, 
                                               batch_size=batch_size, 
                                               split='test', 
                                               aug=None, 
                                               shuffle=False, 
                                               target_list = range(num_labeled_classes, num_classes))

    labeled_eval_loader = CIFAR10Loader(root=dataset_root, 
                                        batch_size=batch_size, 
                                        split='test', 
                                        aug=None, 
                                        shuffle=False, 
                                        target_list = range(num_labeled_classes))

    for img, target, _ in mix_train_loader:

        print(target)
        mask_lb = target < num_labeled_classes
        print(target[mask_lb])
        break