from statistics import mode
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
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from pathlib import Path
import json
from torch.cuda.amp import autocast as autocast
from mae_model.Transformers.VIT.mae import MAEVisionTransformers as MAE

if torch.cuda.is_available():
    device = 'cuda'
    torch.backends.cudnn.benchmark = True
else:
    device = 'cpu'

class_dict = {'good': 0, 'cut': 1, 'color': 1, 'hole': 1, 'metal_contamination': 1, 'thread': 1}
print(class_dict)

class_ids = []
for k, v in class_dict.items():
    if v not in class_ids:
        class_ids.append(v)
num_classes = len(class_ids)


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--end_epoch", type=int, default=100)
    parser.add_argument('--resume', type=int, default=0)

    parser.add_argument("--dilate", type=int, default=5)
    parser.add_argument("--augment", type=int, default=1)
    parser.add_argument("--do_train", type=int, default=1)
    parser.add_argument("--img_size", type=tuple, default=(224, 224))
    parser.add_argument("--backbone", type=str, default='maebase')
    parser.add_argument("--pretrained", type=int, default=1)

    parser.add_argument('--dataset', type=str, default='../datasets/carpet_multi64')
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument("--save_tag", type=str, default='v0')

    opt = parser.parse_args()
    return opt


class SegClsNet(nn.Module):
    def __init__(self, backbone, num_classes, pretrained):
        super().__init__()
        self.backbone = backbone
        if backbone == 'alexnet':
            model = torchvision.models.alexnet(pretrained=pretrained)
            self.features = model.features[:-1]
            fea_c_num = 256
        elif backbone == 'vgg16':
            model = torchvision.models.vgg16(pretrained=pretrained)
            self.features = model.features[:-1]
            fea_c_num = 512
        elif backbone == 'maebase':
            model = MAE(
                img_size=224,
                patch_size=16,
                encoder_dim=768,
                encoder_depth=12,
                encoder_heads=12,
                decoder_dim=512,
                decoder_depth=8,
                decoder_heads=16,
                mask_ratio=0.75
            ).to(device)
            if pretrained:
                ckpt_path = '../MAE/weights/vit-mae_losses_0.20102281799793242.pth'
                ckpt = torch.load(ckpt_path, map_location="cpu")['state_dict']
                model.load_state_dict(ckpt, strict=True)
            self.autoencoder = model.Encoder.autoencoder
            self.proj = model.proj
            self.mae_decoder = model.Decoder.decoder
            self.restruction = model.restruction
            self.patch_norm = model.patch_norm
            self.num_patch0 = model.num_patch[0]
            self.num_patch1 = model.num_patch[1]
            self.unconv = model.unconv
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
            )
            fea_c_num = 512
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
            )
            fea_c_num = 256

        self.decoder = nn.Sequential(
            nn.Conv2d(fea_c_num, fea_c_num // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fea_c_num // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(fea_c_num // 2, fea_c_num // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fea_c_num // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(fea_c_num // 4, 3, kernel_size=3, padding=1, bias=False),
        )

        self.attention_act = nn.Tanh()
        self.attention_conv = nn.Conv2d(3, num_classes, kernel_size=1, padding=0, bias=False)

        self.line = nn.Sequential(
            nn.Conv2d(fea_c_num, fea_c_num * 2, kernel_size=1, padding=1, bias=False),
            nn.BatchNorm2d(fea_c_num * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=fea_c_num * 2, out_features=fea_c_num, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=fea_c_num, out_features=64, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=num_classes, bias=True),
        )

    def forward(self, x0):
        if self.backbone == 'maebase':
            # norm_embeeding, sample_index, mask_index = self.autoencoder(x0)
            # proj_embeeding = self.proj(norm_embeeding).type(torch.float16)
            # decode_embeeding = self.mae_decoder(proj_embeeding, sample_index, mask_index)
            # outputs = self.restruction(decode_embeeding)
            # image_token = outputs[:, 1:, :]  # (b, num_patches, patches_vector)
            # image_norm_token = self.patch_norm(image_token)
            # n, l, dim = image_norm_token.shape
            # image_norm_token = image_norm_token.view(-1, self.num_patch0, self.num_patch1, dim).permute(0, 3, 1, 2)
            # restore_image = self.unconv(image_norm_token)
            # f = self.features(restore_image)

            f = self.features(x0)
        else:
            f = self.features(x0)
        # print(f.shape, f.dtype)

        x1 = F.interpolate(f, size=(x0.size(2), x0.size(3)), mode='bilinear', align_corners=True)
        x1 = self.decoder(x1)
        x1 = x0 + x1
        x1 = self.attention_act(x1)
        x1 = x1 * x0
        x1 = self.attention_conv(x1)

        x2 = self.line(f)
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

        if self.augment == 1:
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

        if self.dilate > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.dilate, self.dilate))
            mask = cv2.dilate(mask, kernel)

        if self.img_size is not None:
            img = cv2.resize(img, self.img_size)
            mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        img = img.astype('float32') / 255
        mask = np.where(mask > 0, class_id, 0)

        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask).long()

        # print(img.shape, mask.shape, label.shape, img.dtype, mask.dtype, label.dtype)
        return {"img": img, "mask": mask, 'label': label}


def train(opt):
    opt_dict = vars(opt)
    for k, v in opt_dict.items():
        print(k, ':', v)
    with open(opt.save_dir + '/train_opts.json', 'w') as f:
        json.dump(opt_dict, f, indent=2)

    writer = SummaryWriter(opt.save_dir)

    net = SegClsNet(opt.backbone, num_classes, opt.pretrained).to(device)

    print(net)

    criterion_seg = torch.nn.CrossEntropyLoss()
    criterion_cls = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)

    milestones = [i * opt.end_epoch // 10 for i in range(7, 10)]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)

    if opt.resume:
        checkpoint = torch.load(opt.save_dir + '/best.pt')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']
        print('resume from epoch', start_epoch)
    else:
        start_epoch = 0

    train_dataset = dataset(opt.dataset, 'train', opt.img_size, opt.dilate, opt.augment)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)

    test_dataset = dataset(opt.dataset, 'test', img_size=opt.img_size, dilate=0, augment=0)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)

    iter_n = len(train_dataloader)
    max_f1 = -1000
    min_loss = 1000
    for epoch in range(start_epoch + 1, opt.end_epoch + 1):
        net.train()

        avg_loss, avg_loss_seg, avg_loss_cls = [], [], []
        for i, batchData in enumerate(train_dataloader):
            img = batchData["img"].to(device)
            mask = batchData["mask"].to(device)
            label = batchData['label'].to(device)

            if img.shape[0] % opt.batch_size != 0:
                continue

            optimizer.zero_grad()
            out = net(img)
            pred_mask = out[0]
            pred_label = out[1]
            loss_seg = criterion_seg(pred_mask, mask) * 0.3
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
        writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], global_step=epoch)

        # val
        net.eval()
        labels, preds = [], []
        avg_loss, avg_loss_seg, avg_loss_cls = [], [], []
        for batchData in test_dataloader:
            with torch.no_grad():
                img = batchData["img"].to(device)
                mask = batchData["mask"].to(device)
                label = batchData['label'].to(device)

                if img.shape[0] % opt.batch_size != 0:
                    continue

                out = net(img)
            pred_mask = out[0]
            pred_label = out[1]
            loss_seg = criterion_seg(pred_mask, mask) * 0.3
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
            '[eval] loss:{}, loss_seg:{}, loss_cls:{}, val_f1:{}'.format(avg_loss, avg_loss_seg, avg_loss_cls,
                                                                         val_f1))

        if avg_loss < min_loss:
            min_loss = avg_loss
            checkpoint = {
                "net": net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                "epoch": epoch
            }
            torch.save(checkpoint, opt.save_dir + '/min_loss.pt')
            print('-' * 50)
            print('Saved checkpoint as min_loss:', min_loss)
            print('-' * 50)

        if val_f1 > max_f1:
            max_f1 = val_f1
            checkpoint = {
                "net": net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                "epoch": epoch
            }
            torch.save(checkpoint, opt.save_dir + '/max_f1.pt')
            print('-' * 50)
            print('Saved checkpoint as max_f1:', max_f1)
            print('-' * 50)


def evaluate(opt, test_pt_name):
    net = SegClsNet(opt.backbone, num_classes, opt.pretrained).to(device)

    checkpoint = torch.load(opt.save_dir + '/' + test_pt_name + '.pt')
    net.load_state_dict(checkpoint['net'])

    test_dataset = dataset(opt.dataset, 'test', img_size=opt.img_size, dilate=0, augment=0)

    net.eval()

    labels, preds, pred_scores = [], [], []
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
        pred_scores.append(score)
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
        save_vis_dir = opt.save_dir + '/test_vis_' + test_pt_name
        if not os.path.exists(save_vis_dir):
            os.makedirs(save_vis_dir)
        cv2.imwrite(save_vis_dir + '/iter_' + str(cnt) + '.jpg', img)

        cnt += 1

    if num_classes > 2:
        precision = precision_score(labels, preds, average='weighted')
        recall = recall_score(labels, preds, average='weighted')
        f1 = f1_score(labels, preds, average='weighted')
        metrics = {'precision': precision, 'recall': recall, 'f1_score': f1}
    else:
        acc = accuracy_score(labels, preds)
        p = precision_score(labels, preds, average='binary')
        r = recall_score(labels, preds, average='binary')
        f1 = f1_score(labels, preds, average='binary')
        conf = confusion_matrix(labels, preds).tolist()
        metrics = {'accuracy': acc, 'precision': p, 'recall': r, 'f1_score': f1, 'confusion_matric': conf}

        fpr, tpr, thresholds = roc_curve(labels, pred_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
        plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig(opt.save_dir + '/test_roc_auc.jpg')

    for item in metrics.items():
        print(item)

    with open(opt.save_dir + '/test_metrics_' + test_pt_name + '.json', 'w') as f:
        json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    # x = torch.randn(2, 3, 224, 224).cuda()
    # net = SegClsNet('maebase', 2, False).cuda()
    # out = net(x)
    # print(out[0].shape, out[1].shape)

    opt = get_opt()

    opt.save_dir = opt.save_dir + '/' + Path(opt.dataset).name + '-' + Path(opt.backbone).name + '-' + opt.save_tag
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)

    if opt.do_train == 1:
        train(opt)

    evaluate(opt, test_pt_name='min_loss')
