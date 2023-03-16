import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import sys
import argparse
import time
import PIL.Image as Image

from models import SegmentNet, weights_init_normal
from dataset import KolektorDataset

parser = argparse.ArgumentParser()

parser.add_argument("--cuda", type=bool, default=True, help="number of gpu")
parser.add_argument("--gpu_num", type=int, default=1, help="number of gpu")
parser.add_argument("--worker_num", type=int, default=0, help="number of input workers")
parser.add_argument("--batch_size", type=int, default=2, help="batch size of input")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

parser.add_argument("--begin_epoch", type=int, default=0, help="begin_epoch")
parser.add_argument("--end_epoch", type=int, default=51, help="end_epoch")

parser.add_argument("--need_test", type=bool, default=True, help="need to test")
parser.add_argument("--test_interval", type=int, default=50, help="interval of test")
parser.add_argument("--need_save", type=bool, default=True, help="need to save")
parser.add_argument("--save_interval", type=int, default=50, help="interval of save weights")

parser.add_argument("--img_height", type=int, default=512, help="size of image height")  # 1408x512 704x256
parser.add_argument("--img_width", type=int, default=512, help="size of image width")

opt = parser.parse_args()
print(opt)

dataSetRoot = "../datasets/carpet_binary73"
saveRoot = "results/carpet_binary73"

# Build nets
segment_net = SegmentNet(init_weights=True)

# Loss functions
criterion_segment = torch.nn.MSELoss()  # mean squared error (squared L2 norm)
# criterion_segment  = torch.nn.BCEWithLogitsLoss() # non convergence

# Optimizers
optimizer_seg = torch.optim.Adam(segment_net.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
# optimizer_seg = torch.optim.SGD(segment_net.parameters(),lr=opt.lr)

if opt.cuda:
    segment_net = segment_net.cuda()
    criterion_segment.cuda()

if opt.gpu_num > 1:
    segment_net = torch.nn.DataParallel(segment_net, device_ids=list(range(opt.gpu_num)))

if opt.begin_epoch != 0:
    # Load pretrained models
    segment_net.load_state_dict(torch.load(saveRoot + "/saved_models/segment_net_%d.pth" % (opt.begin_epoch)))
else:
    # Initialize weights
    segment_net.apply(weights_init_normal)

# DataLoader
transforms_ = transforms.Compose([
    transforms.Resize((opt.img_height, opt.img_width), transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transforms_mask = transforms.Compose([
    transforms.Resize((opt.img_height // 8, opt.img_width // 8), transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainOKloader = DataLoader(
    KolektorDataset(dataSetRoot, transforms_=transforms_, transforms_mask=transforms_mask,
                    subFold="Train_OK", isTrain=True),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.worker_num,
)
trainNGloader = DataLoader(
    KolektorDataset(dataSetRoot, transforms_=transforms_, transforms_mask=transforms_mask,
                    subFold="Train_NG", isTrain=True),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.worker_num,
)

testloader = DataLoader(
    KolektorDataset(dataSetRoot, transforms_=transforms_, transforms_mask=transforms_mask,
                    subFold="Test_NG", isTrain=False),
    batch_size=1,
    shuffle=False,
    num_workers=opt.worker_num,
)

# device = torch.device('cuda') 

for epoch in range(opt.begin_epoch, opt.end_epoch):
    iterOK = trainOKloader.__iter__()
    iterNG = trainNGloader.__iter__()

    lenNum = min(len(trainNGloader), len(trainOKloader))
    lenNum = 2 * (lenNum - 1)

    # train 
    segment_net.train()
    for i in range(0, lenNum):
        if i % 2 == 0:
            batchData = iterOK.__next__()
            # idx, batchData = enumerate(trainOKloader)
        else:
            batchData = iterNG.__next__()
            # idx, batchData = enumerate(trainNGloader)

        if opt.cuda:
            img = batchData["img"].cuda()
            mask = batchData["mask"].cuda()
        else:
            img = batchData["img"]
            mask = batchData["mask"]

        optimizer_seg.zero_grad()

        rst = segment_net(img)
        seg = rst["seg"]

        loss_seg = criterion_segment(seg, mask)
        loss_seg.backward()
        optimizer_seg.step()

        sys.stdout.write(
            "\r [Epoch %d/%d]  [Batch %d/%d] [loss %f]"
            % (
                epoch,
                opt.end_epoch,
                i,
                lenNum,
                loss_seg.item()
            )
        )

    # test 
    if opt.need_test and epoch % opt.test_interval == 0 and epoch >= opt.test_interval:
        segment_net.eval()
        all_time = 0
        for i, testBatch in enumerate(testloader):
            imgTest = testBatch["img"].cuda()
            t1 = time.time()
            rstTest = segment_net(imgTest)
            segTest = rstTest["seg"]
            t2 = time.time()

            save_path_str = saveRoot + "/testResultSeg/epoch_%d" % epoch
            if os.path.exists(save_path_str) == False:
                os.makedirs(save_path_str, exist_ok=True)
                # os.mkdir(save_path_str)
            save_image(imgTest.data, "%s/img_%d.jpg" % (save_path_str, i))
            save_image(segTest.data, "%s/img_%d_seg.jpg" % (save_path_str, i))

            # print("processing image NO %d, time comsuption %fs"%(i, t2 - t1))
            all_time = (t2 - t1) + all_time
            count_time = i + 1
            # print(all_time, count_time)

        avg_time = all_time / count_time
        print("\na image avg time %fs" % avg_time)
        segment_net.train()

    # save model parameters 
    if opt.need_save and epoch % opt.save_interval == 0 and epoch >= opt.save_interval:
        save_path_str = saveRoot + "/saved_models"
        if os.path.exists(save_path_str) == False:
            os.makedirs(save_path_str, exist_ok=True)
        torch.save(segment_net.state_dict(), "%s/segment_net_%d.pth" % (save_path_str, epoch))
        print("save weights ! epoch = %d" % epoch)
        pass
