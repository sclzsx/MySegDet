import torch
import argparse
from PIL import Image 
from torchvision.transforms import transforms
from torch.cuda.amp import autocast as autocast
import numpy as np 
from model.Transformers.VIT.mae import MAEVisionTransformers as MAE
from loss.mae_loss import build_mask
from pathlib import Path
import cv2
import os


ckpt_path = 'weights/vit-mae_losses_0.20102281799793242.pth'

image_dir = '../datasets/carpet_multi64/test'

save_dir = './results/mae_base'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)]
)

model = MAE(
    img_size = 224,
    patch_size = 16,  
    encoder_dim = 768,
    encoder_depth = 12,
    encoder_heads = 12,
    decoder_dim = 512,
    decoder_depth = 8,
    decoder_heads = 16, 
    mask_ratio = 0.75
)
print(model)

ckpt = torch.load(ckpt_path, map_location="cpu")['state_dict']
model.load_state_dict(ckpt, strict=True)
model.cuda()
model.eval()

saved_names = []
for image_path in Path(image_dir).rglob('*.png'):
    if 'mask' in image_path.name:
        continue

    save_name = image_path.parent.name
    if save_name in saved_names:
        continue
    else:
        saved_names.append(save_name)
    print(save_name)

    image = Image.open(str(image_path))
    raw_image = image.resize((224, 224))
    raw_image.save(save_dir + '/' + save_name + '-input.jpg')

    raw_tensor = torch.from_numpy(np.array(raw_image))

    image = transform(raw_image)
    image_tensor = image.unsqueeze(0)
    image_tensor = image_tensor.cuda()
    with torch.no_grad():
        with autocast():
            output, mask_index = model(image_tensor)

    output_image = output.squeeze(0)
    output_image = output_image.permute(1,2,0).cpu().numpy()
    output_image = output_image * std + mean
    output_image = output_image * 255

    cv2.imwrite(save_dir + '/' + save_name + '-output.jpg', output_image[:,:,::-1])

    mask_map = build_mask(mask_index, patch_size=16, img_size=224)

    non_mask = 1 - mask_map 
    non_mask = non_mask.unsqueeze(-1)

    non_mask_image = non_mask * raw_tensor

    mask_map = mask_map * 127
    mask_map = mask_map.unsqueeze(-1)

    non_mask_image += mask_map 

    non_mask_image = non_mask_image.cpu().numpy()

    cv2.imwrite(save_dir + '/' + save_name + '-masked.jpg', non_mask_image[:,:,::-1])
    
    # break
        
        
        
        

        