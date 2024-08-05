import time
import pandas as pd
import torch
import argparse
import torch.optim.lr_scheduler as lr_sche
from utils import cls_data_aug, equipment, cls_model_sel, confusion
from torch.optim import Adam
import random
import numpy as np
import os


def seed_torch(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def train(model, dataset, train_type, index, seed, num_class, decay, bs, value, epoch, lr, min_loss):
    seed_torch(seed)
    model = cls_model_sel(model, num_class)
    device = equipment()
    model = model.to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=value) if decay else Adam(model.parameters(), lr=lr)
    scheduler = lr_sche.StepLR(optimizer, step_size=20, gamma=0.9)
    train_l, val_l, test_l, train_d, val_d, test_d = cls_data_aug(dataset, index, bs, device)
    con_str = f"{model.__class__.__name__}-{dataset}-{decay}-{index}-{seed}-"
    stop_index = 0
    print('Batch size: %d\nLearning rate: %s\nNumber of epoch: %d' % (bs, lr, epoch),
          file=open(f"./recording/{train_type}/{con_str}log.txt", "w"))

    for i in range(epoch):
        t_loss, v_loss, t_loss_b, v_loss_b = 0, 0, 0, 0
        t_tle, t_ple, v_tle, v_ple = [], [], [], []  # predicted, true; label; epoch
        print('\nEpoch %d/%d \n' % (i + 1, epoch) + '-' * 60, file=open(f"./recording/{train_type}/{con_str}log.txt", "a"))

        since = time.time()
        model.train()
        for step, (t_x, t_y) in enumerate(train_l):
            t_x, t_y = t_x.to(device), t_y.to(device)
            t_tle.append(t_y)
            output = model(t_x)
            loss = loss_func(output, t_y)
            lab = torch.argmax(output, 1)
            t_ple.append(lab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_loss_b += loss.item() * t_x.size(0)
        t_loss = t_loss_b / len(train_d.targets)
        t_acc, _, _, _, _, _ = confusion(con_str, num_class, t_tle, t_ple)

        model.eval()
        for step, (v_x, v_y) in enumerate(val_l):
            v_x, v_y = v_x.to(device), v_y.to(device)
            v_tle.append(v_y)
            output = model(v_x)
            loss = loss_func(output, v_y)
            lab = torch.argmax(output, 1)
            v_ple.append(lab)
            v_loss_b += loss.item() * v_x.size(0)
        v_loss = v_loss_b / len(val_d.targets)
        v_acc, v_npv, v_ppv, v_sen, v_spe, v_fos = confusion(con_str, num_class, v_tle, v_ple)

        t_c = time.time() - since
        scheduler.step()
        print('Train and validation done in %d m %d s \nTrain loss: %.3f, acc: %.3f; Val loss: %.3f, acc: %.3f, '
              'npv: %.3f, ppv: %.3f, sen: %.3f, spe: %.3f, fos: %.3f' % (t_c // 60, t_c % 60, t_loss, t_acc, v_loss,
              v_acc, v_npv, v_ppv, v_sen, v_spe, v_fos), file=open(f"./recording/{train_type}/{con_str}log.txt", "a"))

        te_loss, te_loss_b = 0, 0
        te_tle, te_ple = [], []
        since = time.time()
        model.eval()
        for step, (t_x, t_y) in enumerate(test_l):
            t_x, t_y = t_x.to(device), t_y.to(device)
            te_tle.append(t_y)
            output = model(t_x)
            loss = loss_func(output, t_y)
            lab = torch.argmax(output, 1)
            te_ple.append(lab)
            te_loss_b += loss.item() * t_x.size(0)

        t_c = time.time() - since
        te_loss = te_loss_b / len(test_d.targets)
        save = True if v_loss < min_loss else False
        te_acc, tev_npv, te_ppv, te_sen, te_spe, te_fos = confusion(con_str, num_class, te_tle, te_ple, save=save)
        print('Test done in %d m %d s \nTest loss: %.3f, acc: %.3f, npv: %.3f, ppv: %.3f, sen: %.3f, spe: %.3f, '
              'fos: %.3f' % (t_c // 60, t_c % 60, te_loss, te_acc, tev_npv, te_ppv, te_sen, te_spe, te_fos),
              file=open(f"./recording/{train_type}/{con_str}log.txt", "a"))

        if v_loss < min_loss:
            stop_index = 0
            min_loss = v_loss
            torch.save(model, f"./recording/{train_type}/{con_str}model.pkl")
            print("Model saved", file=open(f"./recording/{train_type}/{con_str}log.txt", "a"))
        else:
            stop_index += 1

        if stop_index == 8:
            print("Early stopping triggered", file=open(f"./recording/{train_type}/{con_str}log.txt", "a"))
            break


def classification():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_argument_group()
    group.add_argument('--model', help='Choose your own model', choices=['VGGNet', 'ResNeXt', 'ConvNeXt'], default='VGGNet')
    group.add_argument('--dataset', help='Select dataset', choices=['busi', 'lung', 'btmri', 'cataract'], default='busi')
    group.add_argument('--train_type', help='Select train type', default='classification')
    group.add_argument('--index', help='Index for method of run', required=True, choices=[1, 2, 3, 4, 5], type=int, metavar='INT')
    group.add_argument('--seed', type=int, default=1, help='random seed')
    group.add_argument('--num_class', help='Number of classes', default=4, type=int, metavar='INT')
    group.add_argument('--decay', help='Setting of weight decay', default=True, metavar='BOOL')
    group.add_argument('--bs', help='Batch size for training', default=128, type=int, metavar='INT')
    group.add_argument('--value', help='Decay value', default=1e-2, type=float, metavar='FLOAT')
    group.add_argument('--epoch', help='Number of epochs', default=40, type=int, metavar='INT')
    group.add_argument('--lr', help='Learning rate', default=0.002, type=float, metavar='FLOAT')
    group.add_argument('--min_loss', help='Minimum loss', default=1e4, type=float, metavar='FLOAT')
    args = parser.parse_args()
    train(**vars(args))


if __name__ == '__main__':
    classification()
