import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import sys
import argparse
import time
import PIL.Image as Image
import numpy as np
import cv2

from models2 import SegmentNet, DecisionNet, weights_init_normal
from dataset import KolektorDataset


def tensor_vis_result(imgTest, segTest, labelStr, score, save_path):
    imgTest = imgTest.cpu().squeeze().permute(1, 2, 0).numpy()
    imgTest = (np.clip(imgTest, 0, 1) * 255).astype('uint8')
    imgTest = cv2.cvtColor(imgTest, cv2.COLOR_RGB2BGR)
    H, W, _ = imgTest.shape
    if labelStr == "NG":
        segTest = segTest.cpu().squeeze().numpy()
        segTest = np.where(segTest > 0, 255, 0).astype('uint8')
        segTest = cv2.resize(segTest, (W, H))
        _, contours, _ = cv2.findContours(segTest, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            max_area = -1
            for c in contours:
                area = cv2.contourArea(c)
                if area > max_area:
                    max_contour = c
                    max_area = area
                [x, y, w, h] = cv2.boundingRect(max_contour)
            cv2.rectangle(imgTest, (x + 1, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(imgTest, (x, max(y - 40, 0)), (x + w, y), (0, 0, 255), -1)
            cv2.putText(imgTest, str(np.round(score, 6)), (x + 4, max(y - 10, 0)), cv2.FONT_HERSHEY_COMPLEX, 0.9,
                        (255, 255, 255), 2)
    cv2.imwrite(save_path, imgTest)


parser = argparse.ArgumentParser()
parser.add_argument("--cuda", type=bool, default=True, help="number of gpu")
parser.add_argument("--test_seg_epoch", type=int, default=100, help="test segment epoch")
parser.add_argument("--test_dec_epoch", type=int, default=100, help="test segment epoch")
parser.add_argument("--img_height", type=int, default=512, help="size of image height")  # 1408x512 704x256
parser.add_argument("--img_width", type=int, default=512, help="size of image width")
opt = parser.parse_args()
print(opt)

dataSetRoot = "datasets/carpet_binary"
saveRoot = "results/models2/carpet_binary"

# Build nets
segment_net = SegmentNet(init_weights=True)
decision_net = DecisionNet(init_weights=True)

if opt.cuda:
    segment_net = segment_net.cuda()
    decision_net = decision_net.cuda()

if opt.test_seg_epoch != 0:
    # Load pretrained models
    segment_net.load_state_dict(torch.load(saveRoot + "/saved_models/segment_net_%d.pth" % (opt.test_seg_epoch)))

if opt.test_dec_epoch != 0:
    # Load pretrained models
    decision_net.load_state_dict(torch.load(saveRoot + "/saved_models/decision_net_%d.pth" % (opt.test_dec_epoch)))

transforms_ = transforms.Compose([
    transforms.Resize((opt.img_height, opt.img_width), transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
testloader = DataLoader(
    KolektorDataset(dataSetRoot, transforms_=transforms_, transforms_mask=None,
                    subFold="Test_NG", isTrain=False),
    batch_size=1,
    shuffle=True,
    num_workers=0,
)
testloader2 = DataLoader(
    KolektorDataset(dataSetRoot, transforms_=transforms_, transforms_mask=None,
                    subFold="Test_OK", isTrain=False),
    batch_size=1,
    shuffle=True,
    num_workers=0,
)

segment_net.eval()
decision_net.eval()

all_time = 0
count_time = 0
count = 0
count_TP = 0  # Pii
count_FP = 0  # Pij 
count_TN = 0  # Pjj
count_FN = 0  # Pji 

for i, testBatch in enumerate(testloader):
    torch.cuda.synchronize()
    iterNG = testloader.__iter__()
    batchData = iterNG.__next__()

    # Train_NG
    gt_c = Variable(torch.Tensor(np.ones((batchData["img"].size(0), 1))), requires_grad=False)
    # Train_OK
    # gt_c = Variable(torch.Tensor(np.zeros((batchData["img"].size(0), 1))), requires_grad=False)

    # print(gt_c.numpy()[0,0]) # Train_NG:1.0,Train_OK:0.0

    gt = gt_c.numpy()[0, 0]
    t1 = time.time()

    imgTest = testBatch["img"].cuda()
    with torch.no_grad():
        rstTest = segment_net(imgTest)

    fTest = rstTest["f"]
    segTest = rstTest["seg"]
    with torch.no_grad():
        cTest = decision_net(fTest, segTest)

    torch.cuda.synchronize()
    t2 = time.time()

    print("Image NO %d, Score %f" % (i, cTest.item()))
    if cTest.item() > 0.45:
        labelStr = "NG"  # 0.51
    else:
        labelStr = "OK"  # 0.45

    save_path_str = saveRoot + "/testResult"
    if os.path.exists(save_path_str) == False:
        os.makedirs(save_path_str, exist_ok=True)
    # save_image(imgTest.data, "%s/img_%d_%s.jpg"% (save_path_str, i, labelStr))
    # save_image(segTest.data, "%s/img_%d_seg_%s.jpg"% (save_path_str, i, labelStr))
    tensor_vis_result(imgTest, segTest, labelStr, cTest.item(), "%s/img_%d_%s.jpg" % (save_path_str, i, labelStr))

    count += 1
    if gt == 1 and labelStr == "NG":
        count_TP += 1
    elif gt == 1:
        count_FN += 1
    elif labelStr == "NG":
        count_FP += 1
    else:
        count_TN += 1

    # print("processing image NO %d, time comsuption %fs"%(i, t2 - t1))
    all_time = (t2 - t1) + all_time
    count_time = count_time + 1
    # print(all_time, count_time)

for i, testBatch in enumerate(testloader2):
    torch.cuda.synchronize()
    iterNG = testloader.__iter__()
    batchData = iterNG.__next__()

    # Train_NG
    # gt_c = Variable(torch.Tensor(np.ones((batchData["img"].size(0), 1))), requires_grad=False)
    # Train_OK
    gt_c = Variable(torch.Tensor(np.zeros((batchData["img"].size(0), 1))), requires_grad=False)

    # print(gt_c.numpy()[0,0]) # Train_NG:1.0,Train_OK:0.0

    gt = gt_c.numpy()[0, 0]
    t1 = time.time()

    imgTest = testBatch["img"].cuda()
    with torch.no_grad():
        rstTest = segment_net(imgTest)

    fTest = rstTest["f"]
    segTest = rstTest["seg"]
    with torch.no_grad():
        cTest = decision_net(fTest, segTest)

    torch.cuda.synchronize()
    t2 = time.time()

    print("Image NO %d, Score %f" % (i, cTest.item()))
    if cTest.item() > 0.45:
        labelStr = "NG"  # 0.51
    else:
        labelStr = "OK"  # 0.45

    save_path_str = saveRoot + "/testResult"
    if os.path.exists(save_path_str) == False:
        os.makedirs(save_path_str, exist_ok=True)
    # save_image(imgTest.data, "%s/img_%d_%s.jpg"% (save_path_str, i, labelStr))
    # save_image(segTest.data, "%s/img_%d_seg_%s.jpg"% (save_path_str, i, labelStr))
    tensor_vis_result(imgTest, segTest, labelStr, cTest.item(), "%s/img_%d_%s.jpg" % (save_path_str, i, labelStr))

    count += 1
    if gt == 1 and labelStr == "NG":
        count_TP += 1
    elif gt == 1:
        count_FN += 1
    elif labelStr == "NG":
        count_FP += 1
    else:
        count_TN += 1

    # print("processing image NO %d, time comsuption %fs"%(i, t2 - t1))
    all_time = (t2 - t1) + all_time
    count_time = count_time + 1
    # print(all_time, count_time)

avg_time = all_time / count_time
print("\na image avg time %fs" % avg_time)

accuracy = (count_TP + count_TN) / count
precision = count_TP / (count_TP + count_FP)
recall = count_TP / (count_TP + count_FN)

print("total number of samples = {}".format(count))
print("positive = {}".format(count_TP + count_FN))
print("negative = {}".format(count_FP + count_TN))
print("TP = {}".format(count_TP))
print("FP = {}".format(count_FP))
print("TN = {}".format(count_TN))
print("FN = {}".format(count_FN))
print("accuracy = {:.4f}".format((count_TP + count_TN) / count))
print("precision = {:.4f}".format(precision))
print("recall = {:.4f}".format(recall))
