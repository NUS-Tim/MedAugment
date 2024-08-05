import time
import pandas as pd
import torch
import argparse
import numpy as np
import os
import torch.optim.lr_scheduler as lr_sche
from utils import equipment, seg_model_sel, SoftIoULoss, seg_data_aug, calculate_metrics_seg
from torch.optim import Adam
import segmentation_models_pytorch as smp
import cv2
import random


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
    model = seg_model_sel(model, num_class)
    device = equipment()
    model = model.to(device)
    loss_func = SoftIoULoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=value) if decay else Adam(model.parameters(), lr=lr)
    scheduler = lr_sche.StepLR(optimizer, step_size=20, gamma=0.9)
    train_l, val_l, test_l, train_d, val_d, test_d = seg_data_aug(dataset, index, bs, device)
    con_str = f"{model.__class__.__name__}-{dataset}-{decay}-{index}-{seed}-"
    stop_index = 0
    print('Batch size: %d\nLearning rate: %s\nNumber of epoch: %d' % (bs, lr, epoch), file=open(
          f"./recording/{train_type}/{con_str}log.txt", "w"))

    for i in range(epoch):
        t_loss, t_loss_b, v_loss, v_loss_b = 0, 0, 0, 0
        t_dice, t_dice_b, v_dice, v_dice_b, v_iou, v_iou_b, v_pa, v_pa_b = 0, 0, 0, 0, 0, 0, 0, 0
        print('\nEpoch %d/%d \n' % (i + 1, epoch) + '-' * 60, file=open(f"./recording/{train_type}/{con_str}log.txt", "a"))

        since = time.time()
        model.train()
        for step, (name, t_x, t_y) in enumerate(train_l):
            t_x, t_y = t_x.to(device), t_y.to(device)
            output = model(t_x)
            loss = loss_func(output, t_y)
            dice, _, _ = calculate_metrics_seg(output, t_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_loss_b += loss.item() * t_x.size(0)
            t_dice_b += dice.item() * t_x.size(0)

        model.eval()
        for step, (name, v_x, v_y) in enumerate(val_l):
            v_x, v_y = v_x.to(device), v_y.to(device)
            output = model(v_x)
            loss = loss_func(output, v_y)
            dice, iou, pa = calculate_metrics_seg(output, v_y)
            v_loss_b += loss.item() * v_x.size(0)
            v_dice_b += dice.item() * v_x.size(0)
            v_iou_b += iou.item() * v_x.size(0)
            v_pa_b += pa.item() * v_x.size(0)

        t_c = time.time() - since
        t_loss, t_dice = (a / len(train_d.img_name) for a in [t_loss_b, t_dice_b])
        v_loss, v_dice, v_iou, v_pa = (b / len(val_d.img_name) for b in [v_loss_b, v_dice_b, v_iou_b, v_pa_b])
        scheduler.step()
        print('Train and validation done in %d m %d s \nTrain loss: %.3f, dice: %.3f; Val loss: %.3f, dice: %.3f, iou: %.3f, pa: %.3f' % (
        t_c // 60, t_c % 60, t_loss, t_dice, v_loss, v_dice, v_iou, v_pa), file=open(f"./recording/{train_type}/{con_str}log.txt", "a"))

        te_loss, te_loss_b, te_dice, te_dice_b, te_iou, te_iou_b, te_pa, te_pa_b = 0, 0, 0, 0, 0, 0, 0, 0
        since = time.time()
        model.eval()
        for step, (name, te_x, te_y) in enumerate(test_l):
            te_x, te_y = te_x.to(device), te_y.to(device)
            output = model(te_x)
            loss = loss_func(output, te_y)
            dice, iou, pa = calculate_metrics_seg(output, te_y)
            te_loss_b += loss.item() * te_x.size(0)
            te_dice_b += dice.item() * te_x.size(0)
            te_iou_b += iou.item() * te_x.size(0)
            te_pa_b += pa.item() * te_x.size(0)

            if v_loss < min_loss:
                for i in range(output.size(0)):
                    pre_path = f"./recording/{train_type}/{con_str}pre"
                    os.makedirs(pre_path, exist_ok=True)
                    img_name, img_ext = os.path.splitext(name[i])
                    img = torch.sigmoid(output[i])
                    img = (img > 0.5).float() * 255
                    img = img.detach().squeeze().cpu().numpy()
                    img_path = os.path.join(pre_path, img_name + img_ext)
                    cv2.imwrite(img_path, img)

        t_c = time.time() - since
        te_loss, te_dice, te_iou, te_pa = (b / len(test_d.img_name) for b in [te_loss_b, te_dice_b, te_iou_b, te_pa_b])
        print('Test done in %d m %d s \nTest loss: %.3f, dice: %.3f, iou: %.3f, pa: %.3f' % (t_c // 60, t_c % 60,
              te_loss, te_dice, te_iou, te_pa), file=open(f"./recording/{train_type}/{con_str}log.txt", "a"))

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


def segmentation():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_argument_group()
    group.add_argument('--model', help='Choose your own model', choices=['UNetPP', 'FPN', 'DeepLabP'], default='UNetPP')
    group.add_argument('--dataset', help='Select dataset', choices=['lung', 'kvasir', 'cvc', 'covid'], default='covid')
    group.add_argument('--train_type', help='Select train type', default='segmentation')
    group.add_argument('--index', help='Index for method of run', required=True, choices=[1, 2, 3, 4], type=int, metavar='INT')
    group.add_argument('--seed', type=int, default=1, help='random seed')
    group.add_argument('--num_class', help='Number of classes', default=1, type=int, metavar='INT')
    group.add_argument('--decay', help='Setting of weight decay', default=True, metavar='BOOL')
    group.add_argument('--bs', help='Batch size for training', default=128, type=int, metavar='INT')
    group.add_argument('--value', help='Decay value', default=1e-2, type=float, metavar='FLOAT')
    group.add_argument('--epoch', help='Number of epochs', default=40, type=int, metavar='INT')
    group.add_argument('--lr', help='Learning rate', default=0.002, type=float, metavar='FLOAT')
    group.add_argument('--min_loss', help='Minimum loss', default=1e4, type=float, metavar='FLOAT')
    args = parser.parse_args()
    train(**vars(args))


if __name__ == '__main__':
    segmentation()
