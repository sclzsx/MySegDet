from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path
import torch.nn as nn
import torchvision

class_dict = {'good': 0, 'cut': 1, 'color': 2, 'hole': 3, 'metal_contamination': 4, 'thread': 5}


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2, help="batch size of input")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--end_epoch", type=int, default=120, help="end_epoch")
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--dataSetRoot', type=str, default='datasets/carpet_multi_aug')
    parser.add_argument('--saveRoot', type=str, default='results/alexnet/seg')
    parser.add_argument("--dilate", type=bool, default=1)
    parser.add_argument("--do_train", type=bool, default=0)
    parser.add_argument("--img_size", type=tuple, default=(512, 512))
    opt = parser.parse_args()
    return opt


class segnet(nn.Module):
    def __init__(self, n_class, pretrained=True):
        super().__init__()
        self.base_model = torchvision.models.alexnet(pretrained=pretrained)
        self.features = self.base_model.features[:-1]
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, n_class, kernel_size=2, stride=2)

    def forward(self, x0):
        x = self.features(x0)
        x = nn.functional.interpolate(x, size=(x0.size(2) // 4, x0.size(3) // 4), mode='bilinear', align_corners=True)
        x = self.up1(x)
        x = self.up2(x)
        x = x + x0
        return x


class dataset(Dataset):
    def __init__(self, dataRoot, subFold, img_size, dilate):
        self.mask_paths = [i for i in Path(os.path.join(dataRoot, subFold)).rglob('*_mask.png')]
        self.img_size = img_size
        self.dilate = dilate

    def __len__(self):
        return len(self.mask_paths)

    def __getitem__(self, idx):
        mask_path = str(self.mask_paths[idx])
        img_path = mask_path.replace('_mask', '')

        class_name = self.mask_paths[idx].parent.name
        class_id = class_dict[class_name]
        label = torch.tensor(class_id, dtype=torch.long)

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)

        img = cv2.resize(img, self.img_size)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        if self.dilate:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.dilate(mask, kernel)

        img = img.astype('float32') / 255
        mask = np.where(mask > 0, class_id, 0)

        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask).long()
        # print(img.shape, mask.shape)

        return {"img": img, "mask": mask, 'label': label}


def train(opt):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(opt.saveRoot):
        os.makedirs(opt.saveRoot)

    writer = SummaryWriter(opt.saveRoot)

    # Build nets
    net = segnet(n_class=len(class_dict)).to(device)

    # Loss functions
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizers
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)

    milestones = [i * opt.end_epoch // 10 for i in range(7, 10)]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)

    if opt.resume:
        checkpoint = torch.load(opt.saveRoot + '/best.pt')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']
        print('resume from epoch', start_epoch)
    else:
        start_epoch = 0

    train_dataset = dataset(opt.dataSetRoot, 'train', opt.img_size, opt.dilate)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)

    test_dataset = dataset(opt.dataSetRoot, 'test', opt.img_size, opt.dilate)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)

    iter_n = len(train_dataloader)
    min_val_loss = 10000
    for epoch in range(start_epoch + 1, opt.end_epoch + 1):
        # train
        net.train()

        avg_loss, gt_labels, pred_labels = [], [], []
        for i, batchData in enumerate(train_dataloader):
            img = batchData["img"].to(device)
            mask = batchData["mask"].to(device)
            label = batchData['label'].to(device)
            optimizer.zero_grad()
            out = net(img)
            # print(img.shape, mask.shape, out.shape)
            loss = criterion(out, mask)
            loss.backward()
            optimizer.step()

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

                out = net(img)
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


def add_mask_to_source_multi_classes(source_np, mask_np, num_classes):
    colors = [[0, 0, 0], [0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 0, 255], [0, 255, 255], [255, 255, 0]]
    foreground_mask_bool = mask_np.astype('bool')
    foreground_mask = mask_np * foreground_mask_bool
    foreground = np.zeros(source_np.shape, dtype='uint8')
    background = source_np.copy()

    for i in range(1, num_classes + 1):
        fg_tmp = np.where(foreground_mask == i, 1, 0)
        fg_tmp_mask_bool = fg_tmp.astype('bool')

        fg_color_tmp = np.zeros(source_np.shape, dtype='uint8')
        fg_color_tmp[:, :] = colors[i]
        for c in range(3):
            fg_color_tmp[:, :, c] *= fg_tmp_mask_bool
        foreground += fg_color_tmp
    foreground = cv2.addWeighted(source_np, 0.8, foreground, 0.2, 0)

    for i in range(3):
        foreground[:, :, i] *= foreground_mask_bool
        background[:, :, i] *= ~foreground_mask_bool

    show = foreground + background
    # plt.imshow(show)
    # plt.pause(0.5)
    return show


def tes(opt):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = segnet(n_class=len(class_dict)).to(device)

    checkpoint = torch.load(opt.saveRoot + '/best.pt')
    net.load_state_dict(checkpoint['net'])

    test_dataset = dataset(opt.dataSetRoot, 'test', opt.img_size, opt.dilate)

    net.eval()
    avg_loss, gt_labels, pred_labels = [], [], []
    with torch.no_grad():
        for i, Data in enumerate(test_dataset):
            img = Data["img"].unsqueeze(0).to(device)
            mask = Data["mask"].unsqueeze(0).to(device)
            label = Data['label'].unsqueeze(0).to(device)

            out = net(img)
            out = torch.max(out.data, 1)[1]

            img_np = (np.clip(np.array(img[0, 1, :, :].cpu().squeeze()), 0, 1) * 255).astype('uint8')
            mask_np = (np.array(mask[0, :, :].cpu()) * 255).astype('uint8')
            out_np = (np.array(out[0, :, :].cpu()) * 255).astype('uint8')

            show = cv2.hconcat([img_np, mask_np, out_np])
            save_dir = opt.saveRoot + '/test_vis'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.imshow(show)
            plt.savefig(save_dir + '/iter_' + str(i) + '.jpg')


if __name__ == '__main__':
    # img = torch.randn(4, 3, 512, 512).cuda()
    # net = segnet().cuda()
    # out = net(img)
    # print(out.shape)

    opt = get_opt()
    if opt.do_train:
        train(opt)
    tes(opt)
