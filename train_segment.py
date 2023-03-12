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


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="batch size of input")
    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
    parser.add_argument("--end_epoch", type=int, default=10, help="end_epoch")
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--dataSetRoot', type=str, default='datasets/carpet_binary')
    parser.add_argument('--saveRoot', type=str, default='results/no_pretrain/seg')
    parser.add_argument("--vis_data", type=bool, default=0)
    parser.add_argument("--dilate", type=bool, default=1)
    parser.add_argument("--do_train", type=bool, default=1)
    parser.add_argument("--img_size", type=tuple, default=(512, 512))
    opt = parser.parse_args()
    return opt

class Decoder(nn.Module):
  def __init__(self, in_channels, middle_channels, out_channels):
    super(Decoder, self).__init__()
    self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    self.conv_relu = nn.Sequential(
        nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
        )
  def forward(self, x1, x2):
    x1 = self.up(x1)
    x1 = torch.cat((x1, x2), dim=1)
    x1 = self.conv_relu(x1)
    return x1

class Unet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = torchvision.models.resnet18(True)
        self.base_layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            self.base_layers[1],
            self.base_layers[2])
        self.layer2 = nn.Sequential(*self.base_layers[3:5])
        self.layer3 = self.base_layers[5]
        self.layer4 = self.base_layers[6]
        self.layer5 = self.base_layers[7]
        self.decode4 = Decoder(512, 256+256, 256)
        self.decode3 = Decoder(256, 256+128, 256)
        self.decode2 = Decoder(256, 128+64, 128)
        self.decode1 = Decoder(128, 64+64, 64)
        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
            )
        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        e1 = self.layer1(input) # 64,128,128
        print(e1.shape)
        e2 = self.layer2(e1) # 64,64,64
        e3 = self.layer3(e2) # 128,32,32
        e4 = self.layer4(e3) # 256,16,16
        f = self.layer5(e4) # 512,8,8
        d4 = self.decode4(f, e4) # 256,16,16
        d3 = self.decode3(d4, e3) # 256,32,32
        d2 = self.decode2(d3, e2) # 128,64,64
        d1 = self.decode1(d2, e1) # 64,128,128
        d0 = self.decode0(d1) # 64,256,256
        out = self.conv_last(d0) # 1,256,256
        return out
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
        if 'OK' in mask_path:
            label = torch.Tensor([0]).float()
        else:
            label = torch.Tensor([1]).float()
        # print(label.shape, label, label.dtype)

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)

        img = cv2.resize(img, self.img_size)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        if self.dilate:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.dilate(mask, kernel)

        mask = cv2.resize(mask, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_NEAREST)

        img = img.astype('float32') / 255
        mask = np.where(mask > 0, 1, 0).astype('float32')

        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return {"img": img, "mask": mask, 'label': label}


def vis_img_mask_out_tensor(img, mask, out, thresh=0.5):
    with torch.no_grad():
        img_np = (np.clip(np.array(img[0, 1, :, :].cpu().squeeze()), 0, 1) * 255).astype('uint8')
        mask_np = (np.clip(np.array(mask[0, 0, :, :].cpu().squeeze()), 0, 1) * 255).astype('uint8')
        out_np = (np.where(np.array(out[0, 0, :, :].cpu().squeeze()) > thresh, 1, 0) * 255).astype('uint8')
    mask_np = cv2.resize(mask_np, None, fx=8, fy=8, interpolation=cv2.INTER_LINEAR)
    out_np = cv2.resize(out_np, None, fx=8, fy=8, interpolation=cv2.INTER_LINEAR)
    show = cv2.hconcat([img_np, mask_np, out_np])
    return show


def train(opt):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(opt.saveRoot):
        os.makedirs(opt.saveRoot)

    writer = SummaryWriter(opt.saveRoot)

    # Build nets
    net = SegmentNet().to(device)

    # Loss functions
    # criterion = torch.nn.MSELoss().to(device)  # mean squared error (squared L2 norm)
    # criterion = torch.nn.L1Loss().to(device)
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
        net.apply(weights_init_normal)

    train_ok_dataset = dataset(opt.dataSetRoot, 'Train_OK', opt.img_size, opt.dilate)
    train_ng_dataset = dataset(opt.dataSetRoot, 'Train_NG', opt.img_size, opt.dilate)
    test_ok_dataset = dataset(opt.dataSetRoot, 'Test/OK', opt.img_size, opt.dilate)
    test_ng_dataset = dataset(opt.dataSetRoot, 'Test/NG', opt.img_size, opt.dilate)

    train_ok_dataloader = DataLoader(train_ok_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
    train_ng_dataloader = DataLoader(train_ng_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)

    iter_n_train_ng = len(train_ng_dataloader)
    iter_n_train_ok = len(train_ok_dataloader)
    iter_n = min(iter_n_train_ng, iter_n_train_ok) * 2

    min_val_loss = 10000
    for epoch in range(start_epoch + 1, opt.end_epoch + 1):
        # train
        net.train()

        iterOK = train_ok_dataloader.__iter__()
        iterNG = train_ng_dataloader.__iter__()

        avg_loss, gt_labels, pred_labels = [], [], []
        for i in range(iter_n):
            if i % 2 == 0:
                batchData = iterNG.__next__()
            else:
                batchData = iterOK.__next__()
            img = batchData["img"].to(device)
            mask = batchData["mask"].to(device)
            label = batchData['label'].to(device)

            optimizer.zero_grad()
            out = net(img)["seg"]
            loss = criterion(out, mask)
            loss.backward()
            optimizer.step()

            if opt.vis_data:
                # print(img.shape, mask.shape, label.shape)
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

    net = SegmentNet(init_weights=True).to(device)

    checkpoint = torch.load(opt.saveRoot + '/best.pt')
    net.load_state_dict(checkpoint['net'])

    test_dataset = dataset(opt.dataSetRoot, 'Test', opt.img_size, opt.dilate)

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    net.eval()
    avg_loss, gt_labels, pred_labels = [], [], []
    with torch.no_grad():
        for i, batchData in enumerate(test_dataloader):
            img = batchData["img"].to(device)
            mask = batchData["mask"].to(device)
            label = batchData['label'].to(device)
            out = net(img)["seg"]

            label = label[0, 0].cpu().item()
            if label == 0:
                continue

            print(img.shape, mask.shape, label)
            show = vis_img_mask_out_tensor(img, mask, out)
            save_dir = opt.saveRoot + '/test_vis'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.imshow(show)
            plt.savefig(save_dir + '/iter_' + str(i) + '.jpg')


if __name__ == '__main__':
    img = torch.randn(4, 3, 512, 512).cuda()
    net = Unet(2).cuda()
    out = net(img)
    print(out.shape)


    # opt = get_opt()
    # if opt.do_train:
    #     train(opt)
    # tes(opt)
