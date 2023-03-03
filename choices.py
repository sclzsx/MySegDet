
from Nets.ENet import ENet

from Nets.UNet import UNet
from Nets.SegNet import SegNet
from Nets.LaneNet0508 import LaneNet0508

import torch
import torch.nn as nn

from torchsummary import summary

from lr_scheduler import *


def get_lr_scheduler(optimizer, max_iters, sch_name):
    if sch_name == 'warmup_poly':
        return WarmupPolyLR(optimizer, max_iters=max_iters, power=0.9, warmup_factor=float(1.0/3), warmup_iters=0, warmup_method='linear')
    else:
        return None


def get_optimizer(net, optim_name):
    if optim_name == 'adam':
        optimizer = torch.optim.Adam(net.parameters())
    elif optim_name == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(net.parameters())
    return optimizer


def get_criterion(out_channels, class_weights=None):
    if out_channels == 1:
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights).cuda()
    return criterion


def choose_net(name, out_channels):
    if name == 'unet':
        return UNet(n_classes=out_channels)
    elif name == 'segnet':
        return SegNet(label_nbr=out_channels)
    elif name == 'enet':
        return ENet(num_classes=out_channels)
    elif name == 'lanenet0508':
        return LaneNet0508(num_classes=out_channels)


if __name__ == '__main__':
    net_names = [
        # 'enet'，
        # 'lanenet0508'
    ]
    resizes = [
        # (320, 320),
        # (224, 224)
        (528, 960)
    ]

    # batch_cal_comlexity(net_names, resizes, out_channels=2, method=0)
    summary(choose_net(net_names[0], 2).cuda(), (3, resizes[0][0], resizes[0][1]))
