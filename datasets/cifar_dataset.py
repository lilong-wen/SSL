import pickle
from torch.utils.data import dataset
from torchvision import transforms
import numpy as np
import os
import time

class CIFAR10(dataset.Dataset):
    def __init__(self, mode, target_list, labeled=True, data_root="/home/wll/data/cifar-10-batches-py/"):
        assert mode in ['train', 'test'], print('mode must be "train" or "test"')
        data_files = {'train': ['data_batch_1', 'data_batch_2', 'data_batch_3',
                                'data_batch_4', 'data_batch_5'], 
                      'test': ['test_batch']}
        self.imgs = None
        self.labels = []
        self.data_root = data_root
        self.labeled = labeled
        
        if mode=='test' and target_list == None:
            target_list = range(5)

        # self.class_names = self._unpickle(os.path.join(data_root, 'batches.meta'))[b'label_names]
        for f in data_files[mode]:
            data_dict = self._unpickle(os.path.join(data_root, f))
            data = data_dict[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            if self.imgs is None:
                self.imgs = data
            else:
                self.imgs = np.vstack((self.imgs, data))
            self.labels += data_dict[b'labels']
            
        if mode == 'train':

            self.trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        self.load_meta()

        ind = [i for i in range(len(self.labels)) if self.labels[i] in target_list]
        self.imgs = self.imgs[ind]

        self.labels = np.array(self.labels)
        if not self.labeled:
            self.labels[:] = -1

        self.labels = self.labels[ind].tolist()
    
    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        img = self.trans(img)
        
        return img, label
    
    def __len__(self):
        return len(self.labels)

    def load_meta(self):
        meta_file_path = os.path.join(self.data_root, "batches.meta")
        with open(meta_file_path, 'rb') as f_meta:
            data = pickle.load(f_meta, encoding='latin1')
        self.classes = data['label_names']

        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    
    def _unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict


class CIFAR100(dataset.Dataset):
    def __init__(self, mode, target_list, labeled=True, data_root="/home/wll/data/cifar-100-python/"):
        assert mode in ['train', 'test'], print('mode must be "train" or "test"')
        data_files = {'train': ['train'], 'test': ['test']}
        self.imgs = None
        self.labels = []
        self.data_root = data_root
        self.target_list = target_list
        self.labeled = labeled
        # self.class_names = self._unpickle(os.path.join(data_root, 'batches.meta'))[b'label_names]
        for f in data_files[mode]:
            data_dict = self._unpickle(os.path.join(data_root, f))
            # print(data_dict.keys())
            data = data_dict[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            if self.imgs is None:
                self.imgs = data
            else:
                self.imgs = np.vstack((self.imgs, data))
            self.labels += data_dict[b'fine_labels']
            
        if mode == 'train':

            self.trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        self.load_meta()

        ind = [i for i in range(len(self.labels)) if self.labels[i] in target_list]
        self.imgs = self.imgs[ind]
        self.labels = np.array(self.labels)

        self.labels = np.array(self.labels)
        if not self.labeled:
            self.labels[:] = -1

        self.labels = self.labels[ind].tolist()
    
    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        img = self.trans(img)
        
        return img, label
    
    def __len__(self):
        return len(self.labels)
    
    def load_meta(self):
        meta_file_path = os.path.join(self.data_root, "meta")
        with open(meta_file_path, 'rb') as f_meta:
            data = pickle.load(f_meta, encoding='latin1')
        
        print(data.keys())
        self.classes = data['fine_label_names']

        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
    def _unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
