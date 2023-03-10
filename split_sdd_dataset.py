from pathlib import Path
from PIL import Image
import numpy as np
import random
import os
import shutil

sdd_dir = 'KolektorSDD'

ok_dir_train = 'KolektorSDD_Data/Train_OK'
ng_dir_train = 'KolektorSDD_Data/Train_NG'
ok_dir_test = 'KolektorSDD_Data/Test_OK'
ng_dir_test = 'KolektorSDD_Data/Test_NG'

test_rate = 0.1

def mkdir(d):
    if not os.path.exists(d):
        os.makedirs(d)

mkdir(ok_dir_train)
mkdir(ng_dir_train)
mkdir(ok_dir_test)
mkdir(ng_dir_test)

ok_names, ng_names = [], []
for path in Path(sdd_dir).rglob('*.bmp'):
    parent = path.parent.name
    name = path.name.split('_label')[0]
    name = parent + '_' + name

    lab = Image.open(str(path)).convert('L')
    lab = np.asarray(lab)
    # print(np.min(lab), np.max(lab), lab.shape)

    if np.max(lab) > 0:
        ng_names.append(name)
    else:
        ok_names.append(name)

ok_num = len(ok_names)
ng_num = len(ng_names)
print('len(ok_names):{}, len(ng_names):{}'.format(ok_num, ng_num))

random.shuffle(ok_names)
random.shuffle(ng_names)

test_num_ok = int(ok_num * test_rate)
test_num_ng = int(ng_num * test_rate)

for i, ok_name in enumerate(ok_names):
    dir_name = ok_name.split('_')[0]
    img_name = ok_name.split('_')[1]
    print(ok_name, dir_name, img_name)

    if i < test_num_ok:
        shutil.copyfile(sdd_dir + '/' + dir_name + '/' + img_name + '.jpg', ok_dir_test + '/' + ok_name + '.jpg')
        shutil.copyfile(sdd_dir + '/' + dir_name + '/' + img_name + '_label.bmp', ok_dir_test + '/' + ok_name + '.bmp')
    else:
        shutil.copyfile(sdd_dir + '/' + dir_name + '/' + img_name + '.jpg', ok_dir_train + '/' + ok_name + '.jpg')
        shutil.copyfile(sdd_dir + '/' + dir_name + '/' + img_name + '_label.bmp', ok_dir_train + '/' + ok_name + '.bmp')

for i, ng_name in enumerate(ng_names):
    dir_name = ng_name.split('_')[0]
    img_name = ng_name.split('_')[1]
    print(ng_name, dir_name, img_name)

    if i < test_num_ng:
        shutil.copyfile(sdd_dir + '/' + dir_name + '/' + img_name + '.jpg', ng_dir_test + '/' + ng_name + '.jpg')
        shutil.copyfile(sdd_dir + '/' + dir_name + '/' + img_name + '_label.bmp', ng_dir_test + '/' + ng_name + '.bmp')
    else:
        shutil.copyfile(sdd_dir + '/' + dir_name + '/' + img_name + '.jpg', ng_dir_train + '/' + ng_name + '.jpg')
        shutil.copyfile(sdd_dir + '/' + dir_name + '/' + img_name + '_label.bmp', ng_dir_train + '/' + ng_name + '.bmp')