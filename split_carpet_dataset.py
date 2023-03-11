from pathlib import Path

import torch
from PIL import Image
import numpy as np
import random
import os
import shutil
from torchvision import transforms
from tqdm import tqdm


def mkdir(d):
    d = str(d)
    if not os.path.exists(d):
        os.makedirs(d)


def split_binary():
    data_root = 'datasets/carpet'
    ok_dir_train = 'datasets/carpet_binary/Train_OK'
    ng_dir_train = 'datasets/carpet_binary/Train_NG'
    ok_dir_test = 'datasets/carpet_binary/Test_OK'
    ng_dir_test = 'datasets/carpet_binary/Test_NG'

    test_rate = 0.2  # 测试集所占的比例

    mkdir(ok_dir_train)
    mkdir(ng_dir_train)
    mkdir(ok_dir_test)
    mkdir(ng_dir_test)

    # 把所有缺陷样本放入Train_NG中
    for i, mask_path in enumerate(Path(data_root).rglob('*_mask.png')):
        mask_path = str(mask_path)
        img_path = str(mask_path).replace('_mask', '').replace('ground_truth', 'test')
        class_name = Path(mask_path).parent.name
        shutil.copyfile(mask_path, ng_dir_train + '/' + class_name + '_' + str(i) + '_mask.png')
        shutil.copyfile(img_path, ng_dir_train + '/' + class_name + '_' + str(i) + '.png')

    # 把所有正常样本放入Train_OK中，并生成空白mask
    for i, img_path in enumerate(Path(data_root + '/train/good').rglob('*.png')):
        img_path = str(img_path)
        shutil.copyfile(img_path, ok_dir_train + '/good_' + str(i) + '.png')
        img = Image.open(img_path)
        h, w = img.height, img.width
        mask = np.zeros((h, w), dtype='uint8')
        mask = Image.fromarray(mask)
        mask.save(ok_dir_train + '/good_' + str(i) + '_mask.png')

    # 打乱样本顺序
    ng_mask_paths = [i for i in Path(ng_dir_train).glob('*_mask.png')]
    ok_mask_paths = [i for i in Path(ok_dir_train).glob('*_mask.png')]
    random.shuffle(ng_mask_paths)
    random.shuffle(ok_mask_paths)

    # 算出测试集数量
    ng_num = len(ng_mask_paths)
    ok_num = len(ok_mask_paths)
    test_num_ok = int(ok_num * test_rate)
    test_num_ng = int(ng_num * test_rate)

    # 转移指定数目的正常样本入测试集
    for i, mask_path in enumerate(ok_mask_paths):
        if i < test_num_ok:
            shutil.move(str(mask_path), ok_dir_test)
            shutil.move(str(mask_path).replace('_mask', ''), ok_dir_test)

    # 转移指定数目的缺陷样本入测试集
    for i, mask_path in enumerate(ng_mask_paths):
        if i < test_num_ng:
            shutil.move(str(mask_path), ng_dir_test)
            shutil.move(str(mask_path).replace('_mask', ''), ng_dir_test)


def split_multi(aug=0):
    data_root = 'datasets/carpet'
    save_root = 'datasets/carpet_multi'

    test_rate = 0.2  # 测试集所占的比例

    mkdir(save_root)

    # 对所有缺陷样本重命名，全放入save_root，并统计类别名
    class_names = []
    for i, mask_path in enumerate(Path(data_root).rglob('*_mask.png')):
        mask_path = str(mask_path)
        img_path = str(mask_path).replace('_mask', '').replace('ground_truth', 'test')
        class_name = Path(mask_path).parent.name
        shutil.copyfile(mask_path, save_root + '/' + class_name + '_' + str(i) + '_mask.png')
        shutil.copyfile(img_path, save_root + '/' + class_name + '_' + str(i) + '.png')
        if class_name not in class_names:
            class_names.append(class_name)

    # 把test中正常样本放入save_root，并生成空白mask.
    for i, img_path in enumerate(Path(data_root + '/test').rglob('*.png')):
        if 'good' == img_path.parent.name:
            img_path = str(img_path)
            shutil.copyfile(img_path, save_root + '/good_' + str(i) + '.png')
            img = Image.open(img_path)
            h, w = img.height, img.width
            mask = np.zeros((h, w), dtype='uint8')
            mask = Image.fromarray(mask)
            mask.save(save_root + '/good_' + str(i) + '_mask.png')
    class_names.append('good')

    # 对不同类别分开处理，放入各自文件夹
    for class_name in class_names:
        mask_paths = [i for i in Path(save_root).glob('*_mask.png') if class_name in i.name]  # 找出该类别的文件路径
        total_n = len(mask_paths)
        test_n = int(total_n * test_rate)
        random.shuffle(mask_paths)
        for i, mask_path in enumerate(mask_paths):
            if i < test_n:
                class_dir = save_root + '/test/' + class_name
            else:
                class_dir = save_root + '/train/' + class_name
            mkdir(class_dir)
            mask_path = str(mask_path)
            img_path = mask_path.replace('_mask', '')
            shutil.move(mask_path, class_dir)
            shutil.move(img_path, class_dir)

    if aug:
        # 对训练集的缺陷样本做数据增强
        train_mask_paths = [i for i in Path(save_root + '/train').rglob('*_mask.png')]
        for mask_path in tqdm(train_mask_paths):
            if 'good' in mask_path.name:
                aug_n = 4
            else:
                aug_n = 4
            mask_path = str(mask_path)
            img_path = str(mask_path).replace('_mask', '')
            mask = Image.open(mask_path).convert('L')
            img = Image.open(img_path).convert('RGB')
            for i in range(aug_n):
                transform = transforms.Compose([
                    transforms.RandomVerticalFlip(0.5),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomRotation(2, transforms.InterpolationMode.NEAREST, expand=False)
                ])
                torch.manual_seed(i)
                img = transform(img)
                torch.manual_seed(i)
                mask = transform(mask)
                save_img_path = img_path[:-4] + '_aug' + str(i) + '.png'
                img.save(save_img_path)
                mask.save(save_img_path.replace('.png', '_mask.png'))
            # break


if __name__ == '__main__':
    # split_binary()
    split_multi()
