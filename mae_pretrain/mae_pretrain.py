import os
import argparse
import math
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm
from einops import rearrange
from mae_model import MAE_ViT
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
import numpy as np
import random
import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_only', type=bool, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--base_learning_rate', type=float, default=0.001)
    parser.add_argument('--total_epoch', type=int, default=3000)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--save_dir', type=str, default='results/ksdd')
    parser.add_argument('--data_root', type=str, default='../KolektorSDD_Data')
    parser.add_argument('--resume_path', type=str, default='')
    args = parser.parse_args()
    return args

class MAEDataset(Dataset):
    def __init__(self, dataset_dir, image_size, keyword):
        self.paths = [i for i in Path(dataset_dir).rglob('*.jpg') if keyword in str(i)]
        self.resize_wh = (image_size, image_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        image = Image.open(str(path)).convert('RGB')
        img_transform = transforms.Compose([transforms.Resize(self.resize_wh), transforms.ToTensor()])
        img_tensor = img_transform(image)
        # print(img_tensor.shape)
        return img_tensor

def mkdir(d):
    if not os.path.exists(d): os.makedirs(d)

def mae_train_val(args):
    mkdir(args.save_dir)

    train_set = MAEDataset(args.data_root, args.image_size, 'Train')
    print('Length of train_set:', len(train_set))
    val_set = MAEDataset(args.data_root, args.image_size, 'Test')
    print('Length of val_set:', len(val_set))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    writer = SummaryWriter(args.save_dir)
    
    model = MAE_ViT(mask_ratio=args.mask_ratio, image_size=args.image_size, patch_size=args.patch_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256)
    lr_func = lambda epoch: min((epoch + 1) / (args.total_epoch * 2 // 10 + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func, verbose=True)

    # optimizer = torch.optim.Adam(model.parameters(),lr=args.base_learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(),lr=args.base_learning_rate)
    # milestones = [i * args.total_epoch // 5 for i in range(1, 5)]
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=milestones,gamma=0.1)

    start_epoch = 0
    if os.path.exists(args.resume_path):
        checkpoint = torch.load(args.save_dir + '/mae_pretrain.pt')
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']

    min_val_loss = 10000
    optimizer.zero_grad()
    for epoch in range(start_epoch, args.total_epoch):
        model.train()
        losses = []
        for img in train_loader:
            img = img.to(device)
            predicted_img, mask = model(img)
            loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        writer.add_scalar('mae_train_loss', avg_loss, global_step=epoch)
        print(f'[{epoch}/{args.total_epoch}], average train loss is {avg_loss}.')

        model.eval()
        losses = []
        for img in val_loader:
            with torch.no_grad():
                img = img.to(device)
                predicted_img, mask = model(img)
            loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
            losses.append(loss.item())
        avg_loss = sum(losses) / len(losses)
        writer.add_scalar('mae_val_loss', avg_loss, global_step=epoch)
        if avg_loss < min_val_loss:
            min_val_loss = avg_loss
            checkpoint = {
                "model": model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'lr_scheduler':lr_scheduler.state_dict(),
                "epoch": epoch
            }
            torch.save(checkpoint, args.save_dir + '/mae_pretrain.pt')
            print(f'Got min_val_loss: {min_val_loss}. Updated checkpoint.')

def mae_predict(model, pt_path, img_dir, image_size, save_dir):
    mkdir(save_dir)
    checkpoint = torch.load(pt_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    img_transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    paths = [i for i in Path(img_dir).glob('*.jpg')]
    for path in paths:
        print(str(path))
        image = Image.open(str(path)).convert('RGB')
        img_tensor = img_transform(image).unsqueeze(0)
        with torch.no_grad():
            img_tensor = img_tensor.to(device)
            # print(img_tensor.shape)
            predicted_img, mask = model(img_tensor)
            predicted_img = predicted_img * mask + img_tensor * (1 - mask)
        predicted_img = torch.clamp(predicted_img.cpu().squeeze(0).permute(1, 2, 0), 0, 1) * 255
        predicted_img = Image.fromarray(np.array(predicted_img).astype('uint8'))
        predicted_img.save(save_dir + '/' + path.name)

def mae_test(args):
    model = MAE_ViT(mask_ratio=args.mask_ratio, image_size=args.image_size, patch_size=args.patch_size).to(device)
    pt_path = args.save_dir + '/mae_pretrain.pt'
    save_dir = args.save_dir + '/mae_vis'
    img_dir = args.data_root + '/Test_NG'
    image_size = args.image_size
    mae_predict(model, pt_path, img_dir, image_size, save_dir)

if __name__ == '__main__':
    args = get_args()
    
    if not args.test_only:
        mae_train_val(args)
    
    mae_test(args)

        