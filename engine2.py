import math
from enum import EnumMeta
import torch
from torch.functional import _return_counts
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
import models
import datasets
from utils.utils import cluster_acc, AverageMeter, write_txt, entropy, MarginLoss, accuracy
from sklearn import metrics
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from itertools import cycle
from models.resnet import ResNet, BasicBlock
from tqdm import tqdm


def train(model, train_loader, test_loader, args):

    print("training start here")

    optimizer = Adam(model.parameters(), lr=args.lr, 
                     weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, 
                                           step_size=args.step_size, 
                                           gamma=args.gamma)

    bce = nn.BCELoss()
    sum_loss = 0
    bce_losses_labeled = AverageMeter('bce_loss', ':.4e')
    bce_losses_unlabeled = AverageMeter('bce_loss', ':.4e')
    ce_losses = AverageMeter('ce_loss', ':.4e')
    entropy_losses_labeled = AverageMeter('entropy_loss', ':.4e')
    entropy_losses_unlabeled = AverageMeter('entropy_loss', ':.4e')
    seen_uncerts = AverageMeter('seen_uncert', ':.4e')
    unseen_uncerts = AverageMeter('unseen_conf', ':.4e')
    mean_uncert = 0.1

    for epoch in range(args.epoch):

        if epoch % 1 == 0:
            mean_uncert, preds, final_acc = test(model, test_loader, args)
            print(f"test unseen acc: {final_acc}")

        ce = MarginLoss(m = -mean_uncert)
        bce = nn.BCELoss()
        sum_loss = 0
        bce_losses = AverageMeter('bce_loss', ':.4e')
        ce_losses = AverageMeter('ce_loss', ':.4e')
        entropy_losses = AverageMeter('entropy_loss', ':.4e')
        seen_uncerts = AverageMeter('seen_uncert', ':.4e')
        unseen_uncerts = AverageMeter('unseen_conf', ':.4e')
        mean_uncert = 0.1

        model.train()

        exp_lr_scheduler.step()
        
        # for batch_idx, (x_mixed, label, idx_x) in enumerate(tqdm(train_loader)):
        for batch_idx, (x_mixed, label, idx_x) in enumerate(train_loader):


            x_mixed,  label = x_mixed.to(args.device),  label.cpu().numpy()

            # mask half labeled data (by setting half label to unlabled class so \
            # can be mask next)
            idx_sort = np.argsort(label)
            sorted_label = label[idx_sort]
            vals, idx_u, count = np.unique(sorted_label, return_counts=True, return_index=True)
            half_count = np.array(list(map(lambda x: math.ceil(x/2), count)))
            new_label = []
            for ii, (count_i, half_count_i) in enumerate(zip(count, half_count)):
                for i in range(half_count_i):
                    new_label.append(vals[ii])
                for i in range(count_i - half_count_i):
                    new_label.append((args.num_labeled_classes + args.num_unlabeled_classes))

            label = torch.tensor(new_label).to(args.device)


            # mask unlabeled label value
            masked_index = label < args.num_labeled_classes


            optimizer.zero_grad()

            # feat, feat_norm, out_head_1, \
            #     out_head_2 = model(x_mixed, 'feat_logit')

            feat, feat_norm, out= model(x_mixed, 'feat_logit')

            prob_x = F.softmax(out)

            feat_labeled = feat[masked_index]
            feat_unlabeled = feat[~masked_index] # seen and unseen

            labeled_out = out[masked_index]
            unlabeled_out = out[~masked_index]

            prob_labeled = F.softmax(out[masked_index], dim=1)
            prob_unlabeled = F.softmax(out[~masked_index], dim=1)

            labeled_len = sum(masked_index).item()
            unlabeled_len = sum(~masked_index).item()

            if labeled_len == 0 or unlabeled_len == 0:
                continue

            # feat_norm = feat_norm.detach()
            # cosine_dist = torch.mm(feat_norm, feat_norm.t())

            unlabeled_feat_norm = feat_norm[~masked_index].detach()
            unlabeled_cosine_dist = torch.mm(unlabeled_feat_norm, unlabeled_feat_norm.t())

            # uncertainty
            seen_conf, _ = prob_x.max(1) 
            unseen_conf, _ = prob_x.max(1) 
            seen_uncerts.update(1 - seen_conf.mean().item(), labeled_len)
            unseen_uncerts.update(1 - unseen_conf.mean().item(), unlabeled_len)

            # pair
            labeled_pos_pairs = []
            unlabeled_pos_pairs = []
            target = label[masked_index]
            target_np = target.cpu().numpy()

            for i in range(labeled_len):
                target_i = target_np[i]
                idx = np.where(target_i == target_np)[0]
                if len(idx) == 1: # only one has same label
                    # labeled_pos_pairs.append(idx[0]) 
                    labeled_pos_pairs.append(np.array([idx[0]]))
                else: # more than one
                    select_idx = np.random.choice(idx, 1)
                    while select_idx == i: # item itself should not be selected
                        select_idx = np.random.choice(idx, 1) # choice a new one
                    labeled_pos_pairs.append(select_idx)

            vals, pos_idx = torch.topk(unlabeled_cosine_dist, 2, dim=1) 
            # second largest index as pos to the current one
            pos_idx = pos_idx[:, 1].cpu().numpy().flatten().tolist() 

            unlabeled_pos_pairs.extend(pos_idx)
            
            # clustering 
            pos_prob_labeled = prob_labeled[labeled_pos_pairs, :]

            pos_prob_unlabeled = prob_unlabeled[unlabeled_pos_pairs, :]

            pos_sim_labeled = torch.bmm(prob_labeled.view(labeled_len, 1, -1), 
                                        pos_prob_labeled.view(labeled_len, -1, 1)).squeeze()

            pos_sim_unlabeled = torch.bmm(prob_unlabeled.view(unlabeled_len, 1, -1), 
                                        pos_prob_unlabeled.view(unlabeled_len, -1, 1)).squeeze()

            labeled_ones = torch.ones_like(pos_sim_labeled)
            unlabeled_ones = torch.ones_like(pos_sim_unlabeled)

            bce_loss_1 = bce(pos_sim_labeled, labeled_ones)
            bce_loss_2 = bce(pos_sim_unlabeled, unlabeled_ones)

            ce_loss = ce(labeled_out, target)
            entropy_loss_1 = entropy(torch.mean(prob_labeled, 0))
            entropy_loss_2 = entropy(torch.mean(prob_unlabeled, 0))

            loss = 1 * (bce_loss_1 + bce_loss_2) + 1 * ce_loss - 0.3 * (entropy_loss_1 + entropy_loss_2)

            bce_losses_labeled.update(bce_loss_1.item(), labeled_len)
            bce_losses_unlabeled.update(bce_loss_2.item(), unlabeled_len)
            ce_losses.update(ce_loss.item(), (labeled_len + unlabeled_len))
            entropy_losses_labeled.update(entropy_loss_1.item(), labeled_len)
            entropy_losses_unlabeled.update(entropy_loss_2.item(), unlabeled_len)
            optimizer.zero_grad()
            sum_loss += loss.item()
            loss.backward()
            optimizer.step()

            # if batch_idx % (len(train_loader) // 2) == 0:
            #     print('Loss: {:.6f}'.format(sum_loss / (batch_idx + 1)
            #     ))
            if (batch_idx + 1) == len(train_loader):
                print('Loss: {:.6f}'.format(sum_loss / (batch_idx + 1)))
        # write_txt(args, f"labeled uncert: {seen_uncerts.avg} unseen uncert: {unseen_uncerts.avg}")


def test(model, test_loader, args):

    model.eval()
    preds = np.array([])
    confs = np.array([])
    accs = np.array([])

    with torch.no_grad():

        for batch_idx, (x, label, _) in enumerate(test_loader):
            x, label = x.to(args.device), label.to(args.device)
            label_head_output= model(x)
            prob = F.softmax(label_head_output, dim=1)
            conf, pred = prob.max(1)
            acc = cluster_acc(label.cpu().numpy().astype(int), pred.cpu().numpy().astype(int))
            preds = np.append(preds, pred.cpu().numpy())
            confs = np.append(confs, conf.cpu().numpy())

            accs = np.append(accs, acc)

    preds = preds.astype(int)
    mean_uncert = 1 - np.mean(confs)
    final_acc = accs.sum() / len(accs)

    return mean_uncert, preds, final_acc