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
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# class_dict = {'good': 0, 'cut': 1, 'color': 2, 'hole': 3, 'metal_contamination': 4, 'thread': 5}

class_dict = {'good': 0, 'cut': 1, 'color': 1, 'hole': 1, 'metal_contamination': 1, 'thread': 1}


def get_num_classes(class_dict):
    class_ids = []
    for k, v in class_dict.items():
        if v not in class_ids:
            class_ids.append(v)
    num_classes = len(class_ids)
    return num_classes


num_classes = get_num_classes(class_dict)


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--end_epoch", type=int, default=50)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--dataSetRoot', type=str, default='datasets/carpet_multi')
    parser.add_argument('--saveRoot', type=str, default='results/alexnet512')
    parser.add_argument("--dilate", type=bool, default=1)
    parser.add_argument("--augment", type=bool, default=1)
    parser.add_argument("--do_train", type=bool, default=1)
    parser.add_argument("--img_size", type=tuple, default=(512, 512))
    opt = parser.parse_args()
    return opt

class segnet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.base_model = torchvision.models.alexnet(pretrained=pretrained)
        # print(self.base_model)
        self.features = self.base_model.features[:-1]
        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Conv2d(64, num_classes, kernel_size=1, padding=0, bias=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=9216, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=num_classes, bias=True),
        )

    def forward(self, x0):
        f = self.features(x0)

        x1 = F.interpolate(f, size=(x0.size(2), x0.size(3)), mode='bilinear', align_corners=True)
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)

        x2 = self.maxpool(f)
        x2 = self.avgpool(x2)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.classifier(x2)
        return x1, x2


class dataset(Dataset):
    def __init__(self, dataRoot, subFold, img_size, dilate, augment):
        self.mask_paths = [i for i in Path(os.path.join(dataRoot, subFold)).rglob('*_mask.png')]
        self.img_size = img_size
        self.dilate = dilate
        self.augment = augment

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

        if self.augment:
            if np.random.rand() < 0.5:
                img = np.fliplr(img)
                mask = np.fliplr(mask)
            if np.random.rand() < 0.5:
                img = np.flipud(img)
                mask = np.flipud(mask)
            if np.random.rand() < 0.5:
                img = np.rot90(img, 1)
                mask = np.rot90(mask, 1)
            if np.random.rand() < 0.5:
                img = np.rot90(img, 3)
                mask = np.rot90(mask, 3)

        img = img.astype('float32') / 255
        mask = np.where(mask > 0, class_id, 0)

        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask).long()

        # print(img.shape, mask.shape, label.shape, img.dtype, mask.dtype, label.dtype)
        return {"img": img, "mask": mask, 'label': label}


def train(opt):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(opt.saveRoot):
        os.makedirs(opt.saveRoot)

    writer = SummaryWriter(opt.saveRoot)

    # Build nets
    net = segnet(num_classes=num_classes).to(device)
    # print(net)

    # Loss functions
    criterion_seg = torch.nn.CrossEntropyLoss()
    criterion_cls = torch.nn.CrossEntropyLoss()

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

    train_dataset = dataset(opt.dataSetRoot, 'train', opt.img_size, opt.dilate, opt.augment)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)

    test_dataset = dataset(opt.dataSetRoot, 'test', opt.img_size, dilate=0, augment=0)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)

    iter_n = len(train_dataloader)
    max_f1 = -1
    for epoch in range(start_epoch + 1, opt.end_epoch + 1):
        # train
        net.train()

        avg_loss, avg_loss_seg, avg_loss_cls = [], [], []
        for i, batchData in enumerate(train_dataloader):
            img = batchData["img"].to(device)
            mask = batchData["mask"].to(device)
            label = batchData['label'].to(device)
            optimizer.zero_grad()
            out = net(img)
            pred_mask = out[0]
            pred_label = out[1]
            loss_seg = criterion_seg(pred_mask, mask)
            loss_cls = criterion_cls(pred_label, label)
            # print(loss_seg.item(), loss_cls.item())
            loss = loss_seg + loss_cls
            loss.backward()
            optimizer.step()
            avg_loss.append(loss.item())
            avg_loss_seg.append(loss_seg.item())
            avg_loss_cls.append(loss_cls.item())
            if i % int(iter_n * 0.3) == 0:
                print('Epoch:{}, Iter:[{}/{}], loss:{}, loss_seg:{}, loss_cls:{}'.format(epoch, i + 1, iter_n,
                                                                                         loss.item(), loss_seg.item(),
                                                                                         loss_cls.item()))
        avg_loss = sum(avg_loss) / len(avg_loss)
        avg_loss_seg = sum(avg_loss_seg) / len(avg_loss_seg)
        avg_loss_cls = sum(avg_loss_cls) / len(avg_loss_cls)
        writer.add_scalar('train_loss', avg_loss, global_step=epoch)
        writer.add_scalar('train_loss_seg', avg_loss_seg, global_step=epoch)
        writer.add_scalar('train_loss_cls', avg_loss_cls, global_step=epoch)

        # val
        net.eval()
        labels, preds = [], []
        avg_loss, avg_loss_seg, avg_loss_cls = [], [], []
        for batchData in test_dataloader:
            with torch.no_grad():
                img = batchData["img"].to(device)
                mask = batchData["mask"].to(device)
                label = batchData['label'].to(device)
                out = net(img)
            pred_mask = out[0]
            pred_label = out[1]
            loss_seg = criterion_seg(pred_mask, mask)
            loss_cls = criterion_cls(pred_label, label)
            loss = loss_seg + loss_cls
            avg_loss.append(loss.item())
            avg_loss_seg.append(loss_seg.item())
            avg_loss_cls.append(loss_cls.item())

            label = label.cpu().tolist()
            softmax = F.softmax(pred_label.cpu(), dim=1)
            pred_label = torch.max(softmax, 1)[1].cpu().tolist()
            # print(label, pred_label)
            labels.extend(label)
            preds.extend(pred_label)

        avg_loss = sum(avg_loss) / len(avg_loss)
        avg_loss_seg = sum(avg_loss_seg) / len(avg_loss_seg)
        avg_loss_cls = sum(avg_loss_cls) / len(avg_loss_cls)
        writer.add_scalar('val_loss', avg_loss, global_step=epoch)
        writer.add_scalar('val_loss_seg', avg_loss_seg, global_step=epoch)
        writer.add_scalar('val_loss_cls', avg_loss_cls, global_step=epoch)

        if num_classes > 2:
            val_f1 = f1_score(labels, preds, average='weighted')
        else:
            val_f1 = f1_score(labels, preds, average='binary')
        writer.add_scalar('val_f1', val_f1, global_step=epoch)

        print(
            '[eval] loss:{}, loss_seg:{}, loss_cls:{}, val_f1:{}'.format(loss.item(), loss_seg.item(), loss_cls.item(),
                                                                         val_f1))

        # save model parameters
        if val_f1 > max_f1:
            max_f1 = val_f1
            checkpoint = {
                "net": net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                "epoch": epoch
            }
            torch.save(checkpoint, opt.saveRoot + '/best.pt')
            print('-' * 50)
            print('Saved checkpoint as max_f1:', max_f1)
            print('-' * 50)


def evaluate(opt):
    net = segnet(num_classes=num_classes).to(device)

    checkpoint = torch.load(opt.saveRoot + '/best.pt')
    net.load_state_dict(checkpoint['net'])

    test_dataset = dataset(opt.dataSetRoot, 'test', opt.img_size, dilate=0, augment=0)

    net.eval()

    labels, preds = [], []
    cnt = 0
    for Data in tqdm(test_dataset):
        img = Data["img"].unsqueeze(0).to(device)
        mask = Data["mask"].unsqueeze(0)
        label = Data['label'].item()
        with torch.no_grad():
            out = net(img)
        pred_mask = out[0].cpu()
        pred_label = out[1].cpu()
        pred_mask = torch.max(pred_mask.data, 1)[1].squeeze()
        softmax = F.softmax(pred_label, dim=1)
        score = torch.max(softmax, 1)[0].squeeze().item()
        pred_label = torch.max(softmax, 1)[1].squeeze().item()
        labels.append(label)
        preds.append(pred_label)
        # print(label, pred_label, score)

        img = (np.array(img.cpu().squeeze().permute(1, 2, 0)) * 255).astype('uint8')
        mask = (np.array(mask) * 255).astype('uint8')
        pred_mask = np.array(pred_mask)
        pred_mask = np.where(pred_mask > 0, 255, 0).astype('uint8')
        # print(pred_mask.shape, pred_mask.dtype)
        _, contours, _ = cv2.findContours(pred_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            max_area = -1
            for c in contours:
                area = cv2.contourArea(c)
                if area > max_area:
                    max_contour = c
                    max_area = area
                [x, y, w, h] = cv2.boundingRect(max_contour)
            cv2.rectangle(img, (x + 1, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(img, (x, max(y - 40, 0)), (x + w, y), (0, 0, 255), -1)
            cv2.putText(img, str(np.round(score, 6)), (x + 4, max(y - 10, 0)), cv2.FONT_HERSHEY_COMPLEX, 0.9,
                        (255, 255, 255), 2)
        save_dir = opt.saveRoot + '/test_vis'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cv2.imwrite(save_dir + '/iter_' + str(cnt) + '.jpg', img)

        cnt += 1

    if num_classes > 2:
        precision = precision_score(labels, preds, average='weighted')
        recall = recall_score(labels, preds, average='weighted')
        f1 = f1_score(labels, preds, average='weighted')
    else:
        precision = precision_score(labels, preds, average='binary')
        recall = recall_score(labels, preds, average='binary')
        f1 = f1_score(labels, preds, average='binary')
    print('precision:{}, recall:{}, f1:{}'.format(precision, recall, f1))


if __name__ == '__main__':
    # img = torch.randn(4, 3, 512, 512).cuda()
    # net = segnet().cuda()
    # out = net(img)
    # print(out.shape)

    opt = get_opt()
    if opt.do_train:
        train(opt)
    evaluate(opt)
