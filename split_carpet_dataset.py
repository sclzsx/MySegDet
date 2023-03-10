from pathlib import Path
from PIL import Image
import numpy as np
import random
import os
import shutil

def mkdir(d):
    if not os.path.exists(d):
        os.makedirs(d)

data_root = 'carpet'
ok_dir_train = 'carpet_binary/Train_OK'
ng_dir_train = 'carpet_binary/Train_NG'
ok_dir_test = 'carpet_binary/Test_OK'
ng_dir_test = 'carpet_binary/Test_NG'

test_rate = 0.2

mkdir(ok_dir_train)
mkdir(ng_dir_train)
mkdir(ok_dir_test)
mkdir(ng_dir_test)

for i, mask_path in enumerate(Path(data_root).rglob('*_mask.png')):
    mask_path = str(mask_path)
    img_path = str(mask_path).replace('_mask', '').replace('ground_truth', 'test')
    mask_parent = Path(mask_path).parent.name
    img_parent = Path(img_path).parent.name
    shutil.copyfile(mask_path, ng_dir_train + '/' + mask_parent + '_' + str(i) + '_mask.png')
    shutil.copyfile(img_path,  ng_dir_train + '/' + mask_parent + '_' + str(i) + '.png')

for i, img_path in enumerate(Path(data_root + '/train/good').rglob('*.png')):
    img_path = str(img_path)
    shutil.copyfile(img_path,  ok_dir_train + '/good_' + str(i) + '.png')

    img = Image.open(img_path)
    h, w = img.height, img.width
    mask = np.zeros((h, w), dtype='uint8')
    mask = Image.fromarray(mask)
    mask.save(ok_dir_train + '/good_' + str(i) + '_mask.png')

ng_mask_paths = [i for i in Path(ng_dir_train).glob('*_mask.png')]
ok_mask_paths = [i for i in Path(ok_dir_train).glob('*_mask.png')]

random.shuffle(ng_mask_paths)
random.shuffle(ok_mask_paths)

ng_num = len(ng_mask_paths)
ok_num = len(ok_mask_paths)

test_num_ok = int(ok_num * test_rate)
test_num_ng = int(ng_num * test_rate)

for i, mask_path in enumerate(ok_mask_paths):
    if i < test_num_ok:
        shutil.move(str(mask_path), ok_dir_test)
        shutil.move(str(mask_path).replace('_mask', ''), ok_dir_test)

for i, mask_path in enumerate(ng_mask_paths):
    if i < test_num_ng:
        shutil.move(str(mask_path), ng_dir_test)
        shutil.move(str(mask_path).replace('_mask', ''), ng_dir_test)
