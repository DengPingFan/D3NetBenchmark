import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import glob
import os
from os.path import join
class RgbdSodDataset(Dataset):
    def __init__(self, datasets , transform=None, max_num=0 , if_memory=False):
        super().__init__()
        if not isinstance(datasets,list) :  datasets=[datasets]
        self.imgs_list, self.gts_list, self.depths_list = [], [], []

        for dataset in  datasets:
            ids=sorted(glob.glob(os.path.join(dataset,'RGB','*.jpg')))
            ids=[os.path.splitext(os.path.split(id)[1])[0] for id in ids]  
            for id in ids:
                self.imgs_list.append(os.path.join(dataset,'RGB',id+'.jpg')) 
                self.gts_list.append(os.path.join(dataset,'GT',id+'.png')) 
                self.depths_list.append(os.path.join(dataset,'depth',id+'.png')) 

        if  max_num!=0 and len(self.imgs_list)> abs(max_num):
            indices= random.sample(range(len(self.imgs_list)),max_num) if max_num>0 else range(abs(max_num))
            self.imgs_list= [self.imgs_list[i] for i in indices]
            self.gts_list = [self.gts_list[i]  for i in indices]
            self.depths_list = [self.depths_list[i]  for i in indices]

        self.transform, self.if_memory = transform, if_memory

        if if_memory:
            self.samples=[]
            for index in range(len(self.imgs_list)):
                self.samples.append(self.get_sample(index))
        
    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, index):
        if self.if_memory:
            return self.transform(self.samples[index].copy()) if self.transform !=None else self.samples[index].copy()
        else:
            return self.transform(self.get_sample(index)) if self.transform !=None else self.get_sample(index)

    def get_sample(self,index):
        img = np.array(Image.open(self.imgs_list[index]).convert('RGB'))
        gt = np.array(Image.open(self.gts_list[index]).convert('L'))
        depth = np.array(Image.open(self.depths_list[index]).convert('L'))
        sample={'img':img , 'gt' : gt,'depth':depth}

        sample['meta'] = {'id': os.path.splitext(os.path.split(self.gts_list[index])[1])[0]}
        sample['meta']['source_size'] = np.array(gt.shape[::-1])
        sample['meta']['img_path'] = self.imgs_list[index]
        sample['meta']['gt_path'] = self.gts_list[index]
        sample['meta']['depth_path'] = self.depths_list[index]
        return sample

if __name__=='__main__':
    pass
      

