import os
import argparse
# import math
# import torch
# import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
# from tqdm import tqdm
# from einops import rearrange
# from seg_model import seg_ViT
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
from pathlib import Path
from PIL import Image
import numpy as np
# import random
import torch
import cv2
# import numpy as np
from models2 import SegmentNet
import time
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class_dict = {'color': 0, 'cut': 1, 'good': 2, 'hole': 3, 'metal_contamination': 4, 'thread': 5}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_only', type=bool, default=0)
    parser.add_argument('--train_dil', type=bool, default=0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--show_iter_rate', type=int, default=0.3)
    parser.add_argument('--base_learning_rate', type=float, default=0.0001)
    parser.add_argument('--total_epoch', type=int, default=100)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--save_dir', type=str, default='results/sx/seg')
    parser.add_argument('--data_root', type=str, default='datasets/carpet_multi')
    parser.add_argument('--resume_path', type=str, default='')
    args = parser.parse_args()
    return args


class Dataset(Dataset):
    def __init__(self, data_dir, train_dil, image_size):
        self.mask_paths = [i for i in Path(data_dir).rglob('*_mask.png')]
        self.train_dil = train_dil
        self.image_size = image_size

    def __len__(self):
        return len(self.mask_paths)

    def __getitem__(self, i):
        path = self.mask_paths[i]
        mask_path = str(path)
        img_path = mask_path.replace('_mask', '')
        img = Image.open(img_path)
        img = img.resize((self.image_size, self.image_size))
        img = np.array(img).transpose((2, 0, 1))
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        if self.train_dil:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.dilate(mask, kernel)
        mask = cv2.resize(mask, None, fx=0.125, fy=0.125)
        img = img / 255
        mask = np.where(mask > 0, 1, 0)
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        class_name = path.parent.name
        class_id = class_dict[class_name]
        label = torch.tensor(class_id, dtype=torch.long)
        return img, mask, label, class_name


def mkdir(d):
    if not os.path.exists(d): os.makedirs(d)


def seg_train_val(args):
    mkdir(args.save_dir)

    train_set = Dataset(args.data_root + '/train', args.train_dil, args.image_size)
    print('Length of train_set:', len(train_set))
    val_set = Dataset(args.data_root + '/test', 0, args.image_size)
    print('Length of val_set:', len(val_set))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    step_num = len(train_loader)
    show_iter_step = int(step_num * args.show_iter_rate)

    writer = SummaryWriter(args.save_dir)

    model = SegmentNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.base_learning_rate)

    milestones = [i * args.total_epoch // 10 for i in range(7, 10)]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)

    criterion_segment = torch.nn.MSELoss()

    start_epoch = 0
    if os.path.exists(args.resume_path):
        checkpoint = torch.load(args.save_dir + '/seg_pretrain.pt')
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']

    min_val_loss = 10000
    optimizer.zero_grad()
    for epoch in range(start_epoch, args.total_epoch):
        model.train()
        time0 = time.time()
        losses = []
        for i, (img, mask, label, name) in enumerate(train_loader):
            img = img.to(device)
            mask = mask.to(device)
            predicted_img = model(img)["seg"]
            loss = criterion_segment(predicted_img, mask)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            if i % show_iter_step == 0:
                print('Epoch:{}, Iter:[{}/{}], loss:{}'.format(epoch, i, step_num, loss.item()))
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        writer.add_scalar('train_loss', avg_loss, global_step=epoch)

        model.eval()
        losses = []
        for img, mask, label, name in val_loader:
            img = img.to(device)
            mask = mask.to(device)
            predicted_img = model(img)["seg"]
            loss = criterion_segment(predicted_img, mask)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        avg_loss = sum(losses) / len(losses)
        writer.add_scalar('val_loss', avg_loss, global_step=epoch)
        epoch_time = time.time() - time0
        print('Epoch:{}, val_loss:{}, epoch_time:{}'.format(epoch, avg_loss, epoch_time))

        if avg_loss < min_val_loss:
            min_val_loss = avg_loss
            checkpoint = {
                "model": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                "epoch": epoch
            }
            torch.save(checkpoint, args.save_dir + '/model.pt')
            print('Saved checkpoint as min_val_loss!')


def seg_predict(model, pt_path, img_dir, save_dir):
    mkdir(save_dir)
    checkpoint = torch.load(pt_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)

    paths = [i for i in Path(img_dir).rglob('*.png') if 'mask' not in i.name]
    avg_time = []
    for img_path in tqdm(paths):
        img = Image.open(img_path)
        img = np.array(img).transpose((2, 0, 1))
        img = img / 255
        img = torch.from_numpy(img).float()
        img = img.unsqueeze(0)
        with torch.no_grad():
            img = img.to(device)
            time0 = time.time()
            predicted_img = model(img)["seg"]
            avg_time.append(time.time() - time0)
        predicted_img = torch.clamp(predicted_img.cpu().squeeze(), 0, 1) * 255
        predicted_img = Image.fromarray(np.array(predicted_img).astype('uint8'))
        predicted_img.save(save_dir + '/' + img_path.name)
    avg_time = sum(avg_time) / len(paths)
    print('avg_time of segmentation:', avg_time)


def seg_test(args):
    model = SegmentNet()
    pt_path = args.save_dir + '/model.pt'
    save_dir = args.save_dir + '/vis'
    img_dir = args.data_root + '/test'
    seg_predict(model, pt_path, img_dir, save_dir)


if __name__ == '__main__':
    args = get_args()

    if not args.test_only:
        seg_train_val(args)

    seg_test(args)
