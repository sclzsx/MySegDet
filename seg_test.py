import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
# from models2 import SegmentNet, weights_init_normal
from dataset import KolektorDataset
import numpy as np
import cv2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument("--img_height", type=int, default=512, help="size of image height")
parser.add_argument("--img_width", type=int, default=512, help="size of image width")
parser.add_argument('--dataSetRoot', type=str, default='datasets/carpet_binary_aug')
parser.add_argument('--saveRoot', type=str, default='results/no_pretrain/seg')
opt = parser.parse_args()

# Build nets
net = SegmentNet(init_weights=True).to(device)

checkpoint = torch.load(opt.saveRoot + '/best.pt')
net.load_state_dict(checkpoint['net'])

# DataLoader
transforms_ = transforms.Compose([
    transforms.Resize((opt.img_height, opt.img_width), transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
])

transforms_mask = transforms.Compose([
    transforms.Resize((opt.img_height // 8, opt.img_width // 8), transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
])

valOKloader = DataLoader(KolektorDataset(opt.dataSetRoot, transforms_, transforms_mask, subFold="Test_OK"),
                         batch_size=1, shuffle=False, num_workers=0)

valNGloader = DataLoader(KolektorDataset(opt.dataSetRoot, transforms_, transforms_mask, subFold="Test_NG"),
                         batch_size=1, shuffle=False, num_workers=0)

save_dir = opt.saveRoot + '/vis'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# val
net.eval()
cnt = 0
with torch.no_grad():
    for batchData in valNGloader:
        img = batchData["img"].to(device)
        mask = batchData["mask"].to(device)
        # label = batchData['label']
        # gt_labels.append(label)
        out = net(img)["seg"]

        img = img.cpu().squeeze().permute(1, 2, 0).numpy()
        img = (np.clip(img, 0, 1) * 255).astype('uint8')
        cv2.imwrite(save_dir + '/' + str(cnt) + '.jpg', img)
        out = out.cpu().squeeze().numpy()
        out = (np.clip(out, 0, 1) * 255).astype('uint8')
        out = cv2.resize(out, img.shape[:2])
        cv2.imwrite(save_dir + '/' + str(cnt) + '_out.jpg', out)

        cnt += 1
