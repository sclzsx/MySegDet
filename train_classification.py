from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
from models2 import SegmentNet, weights_init_normal, DecisionNet
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path
from train_segment import dataset


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4, help="batch size of input")
    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
    parser.add_argument("--end_epoch", type=int, default=50, help="end_epoch")
    parser.add_argument('--resume_path', type=str, default='')
    parser.add_argument('--dataSetRoot', type=str, default='datasets/carpet_binary')
    parser.add_argument('--saveRoot', type=str, default='results/no_pretrain/cls')
    parser.add_argument("--vis_data", type=bool, default=0)
    parser.add_argument("--dilate", type=bool, default=1)
    parser.add_argument("--img_size", type=tuple, default=(512, 512))
    opt = parser.parse_args()
    return opt

def trainval(opt):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(opt.saveRoot):
        os.makedirs(opt.saveRoot)

    writer = SummaryWriter(opt.saveRoot)

    # Build nets
    net_seg = SegmentNet().to(device)
    net = DecisionNet().to(device)

    # Loss functions
    criterion = torch.nn.MSELoss().to(device)  # mean squared error (squared L2 norm)

    # Optimizers
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)

    milestones = [i * opt.end_epoch // 10 for i in range(7, 10)]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)

    seg_pt_path = opt.saveRoot[:-3] + 'cls/best.pt'
    net_seg.load_state_dict(torch.load(seg_pt_path)['net'])
    net_seg.eval()

    if os.path.exists(opt.resume_path):
        checkpoint = torch.load(opt.resume_path)
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0
        net.apply(weights_init_normal)

    train_ok_dataset = dataset(opt.dataSetRoot, 'Train_OK', opt.img_size, opt.dilate)
    train_ng_dataset = dataset(opt.dataSetRoot, 'Train_NG', opt.img_size, opt.dilate)
    test_dataset = dataset(opt.dataSetRoot, 'Test', opt.img_size, opt.dilate)

    train_ok_dataloader = DataLoader(train_ok_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
    train_ng_dataloader = DataLoader(train_ng_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)

    iter_n_train_ng = len(train_ng_dataloader)
    iter_n_train_ok = len(train_ok_dataloader)
    iter_n = (min(iter_n_train_ng, iter_n_train_ok) - 1) * 2

    min_val_loss = 10000
    for epoch in range(start_epoch + 1, opt.end_epoch + 1):
        # train
        net.train()

        iterOK = train_ok_dataloader.__iter__()
        iterNG = train_ng_dataloader.__iter__()

        avg_loss, gt_labels, pred_labels = [], [], []
        for i in range(iter_n):
            if i % 2 == 0:
                batchData = iterOK.__next__()
            else:
                batchData = iterNG.__next__()
            img = batchData["img"].to(device)
            mask = batchData["mask"].to(device)
            label = batchData['label'].to(device)

            optimizer.zero_grad()
            with torch.no_grad():
                out_seg = net_seg(img)["seg"]
            out = net(out_seg['f'], out_seg['seg'])
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()


            # if opt.cal_metric:

                print(img.shape, mask.shape, label.shape)
                show = vis_img_mask_out_tensor(img, mask, out)
                save_dir = opt.saveRoot + '/train_vis'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                plt.imshow(show)
                plt.savefig(save_dir + '/iter_' + str(i) + '.jpg')

            avg_loss.append(loss.item())
            if i % int(iter_n * 0.3) == 0:
                print('Epoch:{}, Iter:[{}/{}], loss:{}'.format(epoch, i + 1, iter_n, loss.item()))
        avg_loss = sum(avg_loss) / len(avg_loss)
        writer.add_scalar('train_loss', avg_loss, global_step=epoch)

        # val
        net.eval()
        avg_loss, gt_labels, pred_labels = [], [], []
        with torch.no_grad():
            for batchData in test_dataloader:
                img = batchData["img"].to(device)
                mask = batchData["mask"].to(device)
                label = batchData['label'].to(device)

                out = net(img)["seg"]
                loss = criterion(out, mask)
                avg_loss.append(loss.item())
        avg_loss = sum(avg_loss) / len(avg_loss)
        writer.add_scalar('val_loss', avg_loss, global_step=epoch)

        # save model parameters
        if avg_loss < min_val_loss:
            min_val_loss = avg_loss
            checkpoint = {
                "net": net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                "epoch": epoch
            }
            torch.save(checkpoint, opt.saveRoot + '/best.pt')
            print('-' * 60)
            print('Saved checkpoint as min_val_loss:', min_val_loss)
            print('-' * 60)


def tes(opt):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    opt.saveRoot = opt.saveRoot + '/seg'

    net = SegmentNet(init_weights=True).to(device)

    checkpoint = torch.load(opt.saveRoot + '/best.pt')
    net.load_state_dict(checkpoint['net'])

    test_dataset = dataset(opt.dataSetRoot, 'Test', (512, 512))

    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)

    net.eval()
    avg_loss, gt_labels, pred_labels = [], [], []
    with torch.no_grad():
        for i, batchData in enumerate(test_dataloader):
            img = batchData["img"].to(device)
            mask = batchData["mask"].to(device)
            label = batchData['label'].to(device)
            out = net(img)["seg"]
            print(img.shape, mask.shape, label.shape)
            show = vis_img_mask_out_tensor(img, mask, out)
            save_dir = opt.saveRoot + '/test_vis'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.imshow(show)
            plt.savefig(save_dir + '/iter_' + str(i) + '.jpg')


if __name__ == '__main__':
    opt = get_opt()
    trainval(opt)
    tes(opt)
