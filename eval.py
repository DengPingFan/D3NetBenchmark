#pytorch 
import torch
import torchvision
from torch.utils.data import DataLoader

#general 
import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm

#mine
import utils
import my_custom_transforms as mtr
from dataloader_rgbdsod import RgbdSodDataset
from PIL import Image
from model.RgbNet import MyNet as RgbNet
from model.RgbdNet import MyNet as RgbdNet
from model.DepthNet import MyNet as DepthNet

size=(224, 224)
datasets_path='./dataset/'
test_datasets=['SSD']
pretrained_models={'RgbNet':'./eval/pretrained_models/RgbNet.pth', 'RgbdNet':'eval/pretrained_models/RgbdNet.pth' , 'DepthNet':'eval/pretrained_models/DepthNet.pth' }
result_path='./eval/result/'
os.makedirs(result_path,exist_ok=True)

for tmp in ['D3Net']:
    os.makedirs(os.path.join(result_path,tmp),exist_ok=True)
    for test_dataset in test_datasets:
        os.makedirs(os.path.join(result_path,tmp,test_dataset),exist_ok=True)

model_rgb=RgbNet().cuda()
model_rgbd=RgbdNet().cuda()
model_depth=DepthNet().cuda()

model_rgb.load_state_dict(torch.load(pretrained_models['RgbNet'])['model'])
model_rgbd.load_state_dict(torch.load(pretrained_models['RgbdNet'])['model'])
model_depth.load_state_dict(torch.load(pretrained_models['DepthNet'])['model'])

model_rgb.eval()
model_rgbd.eval()
model_depth.eval()

transform_test = torchvision.transforms.Compose([mtr.Resize(size),mtr.ToTensor(),mtr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],elems_do=['img'])])

test_loaders=[]
for test_dataset in test_datasets:
    val_set=RgbdSodDataset(datasets_path+test_dataset,transform=transform_test)
    test_loaders.append(DataLoader(val_set, batch_size=1, shuffle=False,pin_memory=True))

for index, test_loader in enumerate(test_loaders):
    dataset=test_datasets[index]
    print('Test [{}]'.format(dataset))

    for i, sample_batched in enumerate(tqdm(test_loader)):
        input, gt = model_rgb.get_input(sample_batched),model_rgb.get_gt(sample_batched)

        with torch.no_grad(): 
            output_rgb = model_rgb(input)
            output_rgbd = model_rgbd(input)
            output_depth = model_depth(input)

        result_rgb = model_rgb.get_result(output_rgb)
        result_rgbd = model_rgbd.get_result(output_rgbd)
        result_depth = model_depth.get_result(output_depth)

        id=sample_batched['meta']['id'][0]
        gt_src=np.array(Image.open(sample_batched['meta']['gt_path'][0]).convert('L'))

        result_rgb=(cv2.resize(result_rgb, gt_src.shape[::-1], interpolation=cv2.INTER_LINEAR) *255).astype(np.uint8)
        result_rgbd=(cv2.resize(result_rgbd, gt_src.shape[::-1], interpolation=cv2.INTER_LINEAR) *255).astype(np.uint8)
        result_depth=(cv2.resize(result_depth, gt_src.shape[::-1], interpolation=cv2.INTER_LINEAR) *255).astype(np.uint8)

        ddu_mae=np.mean(np.abs(result_rgbd/255.0 - result_depth/255.0))
        result_d3net=result_rgbd if ddu_mae<0.15 else result_rgb

        Image.fromarray(result_d3net).save(os.path.join(result_path,'D3Net',dataset,id+'.png'))



