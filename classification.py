import time
import pandas as pd
import torch
import argparse
import torch.optim.lr_scheduler as lr_sche
from utils import cls_data_aug, equipment, cls_model_sel, confusion, class_acc
from torch.optim import Adam


def train(model, dataset, train_type, index, num_class, decay, bs, log, value, epoch, lr, min_loss):

    model = cls_model_sel(model, num_class)
    device = equipment()
    model = model.to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=value) if decay else Adam(model.parameters(), lr=lr)
    scheduler = lr_sche.StepLR(optimizer, step_size=20, gamma=0.9)
    train_l, val_l, test_l, train_d, val_d, test_d = cls_data_aug(dataset, index, bs, device)
    con_str = f"{model.__class__.__name__}-{dataset}-{decay}-{index}-"
    eva_sta = []
    stop_index = 0

    print('Model training started\nBatch size: %d\nLearning rate: %s\nNumber of epoch: %d' % (bs, lr, epoch), file=open(
          f"./recording/{train_type}/{con_str}log.txt", "w")) if log else print('Model training started\nBatch size: %d\n'
          'Learning rate: %s\nNumber of epoch: %d' % (bs, lr, epoch))
    since = time.time()

    for i in range(epoch):

        t_loss, v_loss, t_loss_b, v_loss_b, train_in, val_in = 0, 0, 0, 0, 0, 0
        t_tle, t_ple, v_tle, v_ple = [], [], [], []
        print('\nEpoch %d/%d \n' % (i + 1, epoch) + '-' * 60, file=open(f"./recording/{train_type}/{con_str}log.txt",
              "a")) if log else print('\nEpoch %d/%d \n' % (i + 1, epoch) + '-' * 60)

        model.train()
        for step, (t_x, t_y) in enumerate(train_l):
            if str(device) == 'cuda': t_x, t_y = t_x.to(device), t_y.to(device)
            t_tle.append(t_y)
            output = model(t_x)
            loss = loss_func(output, t_y)
            lab = torch.argmax(output, 1)
            t_ple.append(lab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_loss_b += loss.item() * t_x.size(0)
            train_in += t_x.size(0)

        model.eval()
        for step, (v_x, v_y) in enumerate(val_l):
            if str(device) == 'cuda': v_x, v_y = v_x.to(device), v_y.to(device)
            v_tle.append(v_y)
            output = model(v_x)
            loss = loss_func(output, v_y)
            lab = torch.argmax(output, 1)
            v_ple.append(lab)
            v_loss_b += loss.item() * v_x.size(0)
            val_in += v_x.size(0)

        t_c = time.time() - since
        t_loss, v_loss = t_loss_b / len(train_d.targets), v_loss_b / len(val_d.targets)
        t_acc, v_acc = class_acc(t_tle, t_ple), class_acc(v_tle, v_ple)
        scheduler.step()  # v_loss
        eva_sta_e = [t_loss, t_acc, v_loss, v_acc]
        eva_sta.append(eva_sta_e)

        print('Train and validation done in %d m %d s \nTrain loss: %.3f, acc: %.3f; Val loss: %.3f, acc: %.3f' % (
        t_c // 60, t_c % 60, t_loss, t_acc, v_loss, v_acc), file=open(f"./recording/{train_type}/{con_str}log.txt", "a")) \
        if log else print('Train and validation done in %d m %d s \nTrain loss: %.3f, acc: %.3f; Val loss: %.3f, acc: '
                          '%.3f' % (t_c // 60, t_c % 60, t_loss, t_acc, v_loss, v_acc))

        if v_loss < min_loss:
            stop_index = 0
            min_loss = v_loss
            torch.save(model, f"./recording/{train_type}/{con_str}model.pkl")
            if log: print("Model saved", file=open(f"./recording/{train_type}/{con_str}log.txt", "a"))
            else: print("Model saved")
        else:
            stop_index += 1

        if stop_index == 8:
            if log: print("Early stopping triggered", file=open(f"./recording/{train_type}/{con_str}log.txt", "a"))
            else: print("Early stopping triggered")
            break

    df = pd.DataFrame(eva_sta)
    df.to_excel(f"./recording/{train_type}/{con_str}eva.xlsx", index=False, header=False)


def test(model, dataset, train_type, index, num_class, decay, bs):

    loss_func = torch.nn.CrossEntropyLoss()
    con_str = f"{model}-{dataset}-{decay}-{index}-"
    device = equipment()
    train_l, val_l, test_l, train_d, val_d, test_d = cls_data_aug(dataset, index, bs, device)
    model = torch.load(f"./recording/{train_type}/{con_str}model.pkl") if str(device) == 'cuda' else torch.load(
                       f"./recording/{train_type}/{con_str}model.pkl", map_location=torch.device('cpu'))
    te_loss, te_loss_b, te_in = 0, 0, 0
    te_tle, te_ple = [], []

    print('Model testing started')
    since = time.time()
    model.eval()
    for step, (t_x, t_y) in enumerate(test_l):
        if str(device) == 'cuda': t_x, t_y = t_x.to(device), t_y.to(device)
        te_tle.append(t_y)
        output = model(t_x)
        loss = loss_func(output, t_y)
        lab = torch.argmax(output, 1)
        te_ple.append(lab)
        te_loss_b += loss.item() * t_x.size(0)
        te_in += t_x.size(0)

    t_c = time.time() - since
    te_loss = te_loss_b / len(test_d.targets)
    confusion(con_str, num_class, te_tle, te_ple)
    print('Test done in %d m %d s. Loss: %.3f' % (t_c // 60, t_c % 60, te_loss))


def classification():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_argument_group()
    group.add_argument('--model', help='Choose your own model', choices=['VGGNet', 'ResNeXt', 'ConvNeXt'], default='VGGNet')
    group.add_argument('--dataset', help='Select dataset', choices=['btmri', 'busi', 'lung', 'cataract'], default='btmri')
    group.add_argument('--train_type', help='Select train type', default='classification')
    group.add_argument('--index', help='Index for number of run', required=True, choices=[1, 2, 3, 4, 5], type=int, metavar='INT')
    group.add_argument('--num_class', help='Number of classes', default=4, type=int, metavar='INT')
    group.add_argument('--decay', help='Setting of weight decay', default=True, metavar='BOOL')
    group.add_argument('--bs', help='Batch size for training', default=128, type=int, metavar='INT')
    group.add_argument('--log', help='Save log to file', default=True, metavar='BOOL')
    group.add_argument('--value', help='Decay value', default=1e-2, type=float, metavar='FLOAT')
    group.add_argument('--epoch', help='Number of epochs', default=40, type=int, metavar='INT')
    group.add_argument('--lr', help='Learning rate', default=0.002, type=float, metavar='FLOAT')
    group.add_argument('--min_loss', help='Minimum loss', default=1e4, type=float, metavar='FLOAT')
    args = parser.parse_args()
    train(**vars(args))
    test_args = {'model': args.model, 'dataset': args.dataset, 'train_type': args.train_type, 'index': args.index,
                 'num_class': args.num_class, 'decay': args.decay, 'bs': args.bs}
    test(**test_args)


if __name__ == '__main__':
    classification()
