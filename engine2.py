from enum import EnumMeta
import torch
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
    bce_losses = AverageMeter('bce_loss', ':.4e')
    ce_losses = AverageMeter('ce_loss', ':.4e')
    entropy_losses = AverageMeter('entropy_loss', ':.4e')
    seen_uncerts = AverageMeter('seen_uncert', ':.4e')
    unseen_uncerts = AverageMeter('unseen_conf', ':.4e')
    mean_uncert = 0.1

    for epoch in range(args.epoch):

        if epoch % 4 == 0:
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
        
        for batch_idx, ((x_labeled, x_unlabeled), label, idx) in enumerate(tqdm(train_loader)):

            x_labeled, x_unlabeled, label = x_labeled.to(args.device), x_unlabeled.to(args.device), label.to(args.device)

            # mask unlabeled label value
            masked_label = label < args.num_labeled_classes

            feat, feat_norm, l_out_labeled_head, \
                l_out_unlabeled_head = model(x_labeled, 'feat_logit')
            feat_un, feat_un_norm, u_out_labeled_head, \
                u_out_unlabeled_head = model(x_unlabeled, 'feat_logit')

            prob_x_labeled_1 = F.softmax(l_out_labeled_head)
            prob_x_unlabeled_1 = F.softmax(u_out_labeled_head)
            prob_x_labeled_2 = F.softmax(l_out_unlabeled_head)
            prob_x_unlabeled_2 = F.softmax(u_out_unlabeled_head)

            prob_all = torch.cat([prob_x_labeled_1, prob_x_unlabeled_1], dim=0)

            labeled_feat_len = len(feat.size(0))
            all_feat_len = len(feat.size(0)) + len(feat_un.size(0))

            # Similarity labels
            feat_all = torch.cat([feat_norm, feat_un_norm]).detach()
            cosine_dist = torch.mm(feat_all, feat_all.t())

            # record uncert
            conf, _ = prob_all.max(1)
            conf = conf.detach()
            seen_conf = conf[: labeled_feat_len]
            unseen_conf = conf[labeled_feat_len:]
            seen_uncerts.update(1 - seen_conf.mean().item(), all_feat_len)
            unseen_uncerts.update(1 - unseen_conf.mean().item(), all_feat_len)
            
            pos_pairs = []
            target = label[masked_label].to(args.device)
            target_np = target.cpu().numpy()

            for i in range(labeled_feat_len):

                target_i = target_np[i]
                idx_match = np.where(target_np == target_i)[0]
                if len(idx_match) == 1:
                    pos_pairs.append(idx_match[0])
                else:
                    select_idx = np.random.choice(idx_match, 1)
                    while select_idx == i:
                        select_idx = np.random.choice(idx_match, 1)
                    pos_pairs.append(int(select_idx))

            unlabel_cosine_dist = cosine_dist[labeled_feat_len:, :]
            vals, pos_idx = torch.topk(unlabel_cosine_dist, 2, dim=1)
            pos_idx = pos_idx[:, 1].cpu().numpy().flatten().tolist()
            pos_pairs.extend(pos_idx)

            # clustering and sonsistency losses
            pos_prob = prob_all[pos_pairs, :]
            pos_sim = torch.bmm(prob_all.view(all_feat_len, 1, -1), 
                                pos_prob.view(all_feat_len, -1, 1)).squeeze()

            ones = torch.ones_like(pos_sim)
            bce_loss = bce(pos_sim, ones)
            ce_loss = ce(l_out_labeled_head, target)
            entropy_loss = entropy(torch.mean(prob_all, 0))

            loss = 1 * bce_loss + 1 * ce_loss - 0.3 * entropy_loss

            bce_losses.update(bce_loss.item(), all_feat_len)
            ce_losses.update(ce_loss.item(), all_feat_len)
            entropy_losses.update(entropy_loss.item(), all_feat_len)
            optimizer.zero_grad()
            sum_loss += loss.item()
            loss.backward()
            optimizer.step()           


            if batch_idx % (len(train_loader) // 4) == 0:
                print('Loss: {:.6f}'.format(sum_loss / (batch_idx + 1)
                ))

        # write_txt(args, f"labeled uncert: {seen_uncerts.avg} unseen uncert: {unseen_uncerts.avg}")


def test(model, test_loader, args):

    model.eval()
    preds = np.array([])
    confs = np.array([])
    accs = np.array([])

    with torch.no_grad():

        for batch_idx, (x, label, _) in enumerate(tqdm(test_loader)):
            x, label = x.to(args.device), label.to(args.device)
            label_head_output, unlabel_head_output = model(x)
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