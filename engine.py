# Modified by ll-wen
# original implement in  https://github.com/snap-stanford/stellar
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import models
import datasets
from utils.utils import cluster_acc, AverageMeter, write_txt, entropy, MarginLoss, accuracy
from sklearn import metrics
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from itertools import cycle
import pickle
import copy
from torch_geometric.data import ClusterData, ClusterLoader
from models.resnet import ResNet, BasicBlock



class Engine:

    def __init__(self, args, num_labeled_classes, num_classes, 
                 labeled_train_loader, labeled_test_loader,
                 unlabeled_train_loader, unlabeled_test_loader):

        self.args = args
        # self.dataset = datasets.CodexGraphDataset(labeled_X, labeled_y, unlabeled_X, labeled_pos, unlabeled_pos, self.args.distance_thres)
        # self.model = models.Encoder(args.input_dim, args.num_heads)

        num_unlabeled_classes = num_classes - num_labeled_classes
        self.labeled_train_loader = labeled_train_loader
        self.labeled_test_loader = labeled_test_loader
        self.unlabeled_train_loader = unlabeled_train_loader
        self.unlabeled_test_loader = unlabeled_test_loader

        self.model = ResNet(BasicBlock, [2,2,2,2], num_labeled_classes=num_labeled_classes, 
                            num_unlabeled_classes=num_unlabeled_classes)
        self.model = self.model.to(args.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)


    def train_epoch(self, m):

        """ Train for 1 epoch."""
        self.model.train()
        bce = nn.BCELoss()
        ce = MarginLoss(m=-m)
        sum_loss = 0
        bce_losses = AverageMeter('bce_loss', ':.4e')
        ce_losses = AverageMeter('ce_loss', ':.4e')
        entropy_losses = AverageMeter('entropy_loss', ':.4e')
        seen_uncerts = AverageMeter('seen_uncert', ':.4e')
        unseen_uncerts = AverageMeter('unseen_conf', ':.4e')

        # labeled_graph, unlabeled_graph = dataset.labeled_data, dataset.unlabeled_data
        # labeled_data = ClusterData(labeled_graph, num_parts=100, recursive=False)
        # labeled_loader = ClusterLoader(labeled_data, batch_size=1, shuffle=True,
        #                             num_workers=1)
        # unlabeled_data = ClusterData(unlabeled_graph, num_parts=100, recursive=False)
        # unlabeled_loader = ClusterLoader(unlabeled_data, batch_size=1, shuffle=True,
        #                             num_workers=1)

        unlabel_loader_iter = cycle(self.unlabeled_train_loader)

        for batch_idx, (labeled_x, labeled_y, _) in enumerate(self.labeled_train_loader):
            unlabeled_x = next(unlabel_loader_iter)[0]
            labeled_x, unlabeled_x = labeled_x.to(self.args.device), unlabeled_x.to(self.args.device)
            self.optimizer.zero_grad()
            # labeled_output, labeled_feat, _ = model(labeled_x)
            labeled_output, labeled_feat = self.model(labeled_x)
            # unlabeled_output, unlabeled_feat, _ = model(unlabeled_x)
            unlabeled_output, unlabeled_feat = self.model(unlabeled_x)
            labeled_len = len(labeled_output)
            batch_size = len(labeled_output) + len(unlabeled_output)
            output = torch.cat([labeled_output, unlabeled_output], dim=0)
            feat = torch.cat([labeled_feat, unlabeled_feat], dim=0)
            
            prob = F.softmax(output, dim=1)
            # Similarity labels
            feat_detach = feat.detach()
            feat_norm = feat_detach / torch.norm(feat_detach, 2, 1, keepdim=True)
            cosine_dist = torch.mm(feat_norm, feat_norm.t())

            # record uncert
            conf, _ = prob.max(1)
            conf = conf.detach()
            seen_conf = conf[:labeled_len]
            unseen_conf = conf[labeled_len:]
            seen_uncerts.update(1 - seen_conf.mean().item(), batch_size)
            unseen_uncerts.update(1 - unseen_conf.mean().item(), batch_size)

            pos_pairs = []
            # target = labeled_x.y
            target = labeled_y.to(self.args.device)
            target_np = target.cpu().numpy()
            
            for i in range(labeled_len):
                target_i = target_np[i]
                idxs = np.where(target_np == target_i)[0]
                if len(idxs) == 1:
                    pos_pairs.append(idxs[0])
                else:
                    selec_idx = np.random.choice(idxs, 1)
                    while selec_idx == i:
                        selec_idx = np.random.choice(idxs, 1)
                    pos_pairs.append(int(selec_idx))
            
            unlabel_cosine_dist = cosine_dist[labeled_len:, :]
            vals, pos_idx = torch.topk(unlabel_cosine_dist, 2, dim=1)
            pos_idx = pos_idx[:, 1].cpu().numpy().flatten().tolist()
            pos_pairs.extend(pos_idx)
            
            # Clustering and consistency losses
            pos_prob = prob[pos_pairs, :] # most similar pair prob
            pos_sim = torch.bmm(prob.view(batch_size, 1, -1), pos_prob.view(batch_size, -1, 1)).squeeze()
            ones = torch.ones_like(pos_sim)
            bce_loss = bce(pos_sim, ones)
            ce_loss = ce(output[:labeled_len], target)
            entropy_loss = entropy(torch.mean(prob, 0))
            
            loss = 1 * bce_loss + 1 * ce_loss - 0.3 * entropy_loss

            bce_losses.update(bce_loss.item(), batch_size)
            ce_losses.update(ce_loss.item(), batch_size)
            entropy_losses.update(entropy_loss.item(), batch_size)
            self.optimizer.zero_grad()
            sum_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            if batch_idx % (len(self.labeled_train_loader) // 4) == 0:
                print('Loss: {:.6f}'.format(sum_loss / (batch_idx + 1)
                ))

        write_txt(self.args, f"labeled uncert: {seen_uncerts.avg} unseen uncert: {unseen_uncerts.avg}")


    def pred(self):

        self.model.eval()
        preds = np.array([])
        confs = np.array([])
        with torch.no_grad():
            # _, unlabeled_graph = self.dataset.labeled_data, self.dataset.unlabeled_data
            # test_data = self.dataset(mode='test', target_list=None, labeled=True)
            # unlabeled_graph_cp = copy.deepcopy(unlabeled_graph)
            acc_list = []
            for batch_idx, (test_data_img, test_data_label, _) in enumerate(self.unlabeled_test_loader):
                # test_data_img = copy.deepcopy(test_data.imgs)
                # test_data_label = np.asarray(copy.deepcopy(test_data.labels))
                # test_data_img = test_data_img.permute(0, -1, 1, 2).to(torch.float)
                # unlabeled_graph_cp = unlabeled_graph_cp.to(self.args.device)
                test_data_img = test_data_img.to(self.args.device)
                # output, _, _ = self.model(unlabeled_graph_cp)
                output, _,  = self.model(test_data_img)
                prob = F.softmax(output, dim=1)
                conf, pred = prob.max(1)
                preds = np.append(preds, pred.cpu().numpy())
                confs = np.append(confs, conf.cpu().numpy())
                # print(test_data_label)
                # print(preds)
                acc = cluster_acc(test_data_label.numpy().astype(int)-5, preds.astype(int))
                # acc = accuracy(test_data_label.numpy().astype(int)-5, preds.astype(int))
                acc_list.append(acc)

        preds = preds.astype(int)
        mean_uncert = 1 - np.mean(confs)
        final_acc = sum(acc_list) / len(acc_list)
        return mean_uncert, preds, final_acc

    def train(self):
        # Set the optimizer
        # train_epoch(self, args, model, device, optimizer, m):
        for epoch in range(self.args.epochs):
            mean_uncert, _, acc = self.pred()
            print(f"test unseen class cluster acc : {acc}")
            self.train_epoch(mean_uncert)
            # _, _, acc = self.pred(self.test_loader)
            # print(acc)
