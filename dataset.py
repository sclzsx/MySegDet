import os
import cv2
from PIL import Image
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import enet_weighing, median_freq_balancing
import torch.nn as nn
from collections import OrderedDict, Counter
import random
from utils import add_mask_to_source_multi_classes, add_mask_to_source
from pathlib import Path

def get_class_weights(loader, out_channels, weighting):
    print('Weighting method is:{}, please wait.'.format(weighting))
    if weighting == 'enet':
        class_weights = enet_weighing(loader, out_channels)
        class_weights = torch.from_numpy(class_weights).float().cuda()
    elif weighting == 'mfb':
        class_weights = median_freq_balancing(loader, out_channels)
        class_weights = torch.from_numpy(class_weights).float().cuda()
    else:
        class_weights = None
    return class_weights


class PILToLongTensor(object):
    def __call__(self, pic):
        if not isinstance(pic, Image.Image):
            raise TypeError("pic should be PIL Image. Got {}".format(
                type(pic)))
        # handle numpy array
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.long()
        # Convert PIL image to ByteTensor
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # Reshape tensor
        nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # Convert to long and squeeze the channels
        return img.transpose(0, 1).transpose(0, 2).contiguous().long().squeeze_()


class SegDataset(Dataset):
    def __init__(self, dataset_dir, num_classes=2, appoint_size=(512, 512), dilate=0):
        self.img_paths = [i for i in Path(dataset_dir).rglob('*.jpg')]
        self.num_classes = num_classes
        self.appoint_size = appoint_size
        self.dilate = dilate

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        img_path = self.img_paths[i]
        class_name = img_path.parent.name
        if class_name == 'ng':
            class_id = 1
        else:
            class_id = 0
        
        class_tensor = torch.tensor([class_id], dtype=torch.long)

        img_path = str(img_path)
        mask_path = str(img_path).replace('.jpg', '.bmp')

        # print(img_path, mask_path)
        
        image = cv2.imread(img_path)
        mask = Image.open(mask_path).convert('L')

        mask_np = np.array(mask)
        mask_np[mask_np > 0] = 1
        # print(mask_np.shape)
        mask = Image.fromarray(mask_np)

        img_transform = transforms.Compose(
            [transforms.ToPILImage(), transforms.Resize(self.appoint_size), transforms.ToTensor()])
        img_tensor = img_transform(image)

        mask = mask.resize((self.appoint_size[1], self.appoint_size[0]), Image.NEAREST)
        if self.dilate > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.dilate, self.dilate))
            mask_np = cv2.dilate(np.array(mask), kernel)
            mask = Image.fromarray(mask_np)

        if self.num_classes == 1:
            mask_transform = transforms.Compose(transforms.ToTensor())
        else:
            mask_transform = transforms.Compose([PILToLongTensor()])

        mask_tensor = mask_transform(mask)

        check = 0
        if check:
            mask_check = np.array(mask_tensor)
            mask_dict = Counter(mask_check.flatten())
            mask_list = [j for j in range(self.num_classes)]
            for k, v in mask_dict.items():
                if k not in mask_list:
                    print(img_path, mask_path, mask_dict)
            print(mask_dict)
            print(img_tensor.shape, mask_tensor.shape, img_tensor.dtype, mask_tensor.dtype, class_tensor)
            
        return img_tensor, mask_tensor, class_tensor


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    num_classes = 2
    appoint_size = (512, 512)
    dataset_dir = 'data/KolektorSDD_split/test'

    dataset = SegDataset(dataset_dir, num_classes=num_classes, appoint_size=appoint_size, dilate=0)

    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    for i, batch_data in enumerate(loader):
        if i % 100 == 0:
            print('Check done', i)