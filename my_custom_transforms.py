import cv2
import math
import torch
import random
import numbers
import numpy as np
from PIL import Image
########################################[ function ]########################################

def img_rotate(img, angle, center=None, if_expand=False, scale=1.0, mode=None):
    (h, w) = img.shape[:2]
    if center is None: center = (w // 2 ,h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    if mode is None: mode=cv2.INTER_LINEAR if len(img.shape)==3 else cv2.INTER_NEAREST
    if if_expand:
        h_new=int(w*math.fabs(math.sin(math.radians(angle)))+h*math.fabs(math.cos(math.radians(angle))))
        w_new=int(h*math.fabs(math.sin(math.radians(angle)))+w*math.fabs(math.cos(math.radians(angle)))) 
        M[0,2] +=(w_new-w)/2 
        M[1,2] +=(h_new-h)/2 
        h, w =h_new, w_new  
    rotated = cv2.warpAffine(img, M, (w, h),flags=mode)
    return rotated


def img_rotate_point(img, angle, center=None, if_expand=False, scale=1.0):
    (h, w) = img.shape[:2]
    if center is None: center = (w // 2 ,h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    if if_expand:
        h_new=int(w*math.fabs(math.sin(math.radians(angle)))+h*math.fabs(math.cos(math.radians(angle))))
        w_new=int(h*math.fabs(math.sin(math.radians(angle)))+w*math.fabs(math.cos(math.radians(angle)))) 
        M[0,2] +=(w_new-w)/2 
        M[1,2] +=(h_new-h)/2 
        h, w =h_new, w_new  


    pts_y, pts_x= np.where(img==1)
    pts_xy=np.concatenate( (pts_x[:,np.newaxis], pts_y[:,np.newaxis]), axis=1 )
    pts_xy_new= np.rint(np.dot( np.insert(pts_xy,2,1,axis=1), M.T)).astype(np.int64)

    img_new=np.zeros((h,w),dtype=np.uint8)
    for pt in pts_xy_new:
        img_new[pt[1], pt[0]]=1
    return img_new


def img_resize_point(img, size):
    (h, w) = img.shape
    if not isinstance(size, tuple): size=( int(w*size), int(h*size) )
    M=np.array([[size[0]/w,0,0],[0,size[1]/h,0]])

    pts_y, pts_x= np.where(img==1)
    pts_xy=np.concatenate( (pts_x[:,np.newaxis], pts_y[:,np.newaxis]), axis=1 )
    pts_xy_new= np.dot( np.insert(pts_xy,2,1,axis=1), M.T).astype(np.int64)

    img_new=np.zeros(size[::-1],dtype=np.uint8)
    for pt in pts_xy_new:
        img_new[pt[1], pt[0]]=1
    return img_new

########################################[ General ]########################################

#Template for all same operation
class Template(object):
    def __init__(self, elems_do=None, elems_undo=[]):
        self.elems_do, self.elems_undo = elems_do, (['meta']+elems_undo)
    def __call__(self, sample):
        for elem in sample.keys():
            if self.elems_do!= None  and elem not in self.elems_do :continue
            if elem in self.elems_undo:continue
            pass
        return sample


class Transform(object):
    def __init__(self, transform, if_numpy=True, elems_do=None, elems_undo=[]):
        self.transform, self.if_numpy = transform, if_numpy
        self.elems_do, self.elems_undo = elems_do, (['meta']+elems_undo)
    def __call__(self, sample):
        for elem in sample.keys():
            if self.elems_do!= None  and elem not in self.elems_do :continue
            if elem in self.elems_undo:continue
            tmp = self.transform(Image.fromarray(sample[elem]))
            sample[elem] = np.array(tmp) if self.if_numpy else tmp
        return sample
    

class ToPilImage(object):
    def __init__(self, elems_do=None, elems_undo=[]):
        self.elems_do, self.elems_undo = elems_do, (['meta']+elems_undo)
    def __call__(self, sample):
        for elem in sample.keys():
            if self.elems_do!= None  and elem not in self.elems_do :continue
            if elem in self.elems_undo:continue
            sample[elem] = Image.fromarray(sample[elem])
        return sample


class ToNumpyImage(object):
    def __init__(self, elems_do=None, elems_undo=[]):
        self.elems_do, self.elems_undo = elems_do, (['meta']+elems_undo)
    def __call__(self, sample):
        for elem in sample.keys():
            if self.elems_do!= None  and elem not in self.elems_do :continue
            if elem in self.elems_undo:continue
            sample[elem] = np.array(sample[elem])
        return sample


class ImageToOne(object):
    def __init__(self, elems_do=None, elems_undo=[]):
        self.elems_do, self.elems_undo = elems_do, (['meta']+elems_undo)
    def __call__(self, sample):
        for elem in sample.keys():
            if self.elems_do!= None  and elem not in self.elems_do :continue
            if elem in self.elems_undo:continue
            sample[elem]=np.array(sample[elem])/255.0
        return sample


class ToTensor(object):
    def __init__(self, if_div=True, elems_do=None, elems_undo=[]):
        self.if_div = if_div
        self.elems_do, self.elems_undo = elems_do, (['meta']+elems_undo)
    def __call__(self, sample):
        for elem in sample.keys():
            if self.elems_do!= None  and elem not in self.elems_do :continue
            if elem in self.elems_undo:continue
            tmp = sample[elem]
            tmp = tmp[np.newaxis,:,:] if tmp.ndim == 2 else tmp.transpose((2, 0, 1))
            tmp = torch.from_numpy(tmp).float()
            tmp = tmp.float().div(255) if self.if_div else tmp
            sample[elem] = tmp                          
        return sample


class Normalize(object):
    def __init__(self, mean, std, elems_do=None, elems_undo=[]):
        self.mean, self.std = mean, std 
        self.elems_do, self.elems_undo = elems_do, (['meta']+elems_undo)
    def __call__(self, sample):
        for elem in sample.keys():
            if self.elems_do!= None  and elem not in self.elems_do :continue
            if elem in self.elems_undo:continue
            tensor = sample[elem]
            #print(tensor.min(),tensor.max())
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
            #print(tensor.min(),tensor.max())

        return sample
    

class Show(object):
    def __init__(self, elems_show=['img','gt'], elems_do=None, elems_undo=[]):
        self.elems_show = elems_show
        self.elems_do, self.elems_undo = elems_do, (['meta']+elems_undo)
    def __call__(self, sample):
        show_list=[  sample[elem] for elem in self.elems_show ]
        return sample



class TestDebug(object):
    def __init__(self, elems_do=None, elems_undo=[]):
        self.elems_do, self.elems_undo = elems_do, (['meta']+elems_undo)
    def __call__(self, sample):
        #print(sample['depth'].min(),sample['depth'].max())
        return sample


########################################[ Basic Image Augmentation ]########################################


class RandomFlip(object):
    def __init__(self, direction=Image.FLIP_LEFT_RIGHT, p=0.5, elems_do=None, elems_undo=[]):
        self.direction, self.p = direction, p
        self.elems_do, self.elems_undo = elems_do, (['meta']+elems_undo)
    def __call__(self, sample):
        if random.random() < self.p:
            for elem in sample.keys():
                if self.elems_do!= None  and elem not in self.elems_do :continue
                if elem in self.elems_undo:continue
                sample[elem]= np.array(Image.fromarray(sample[elem]).transpose(self.direction)) 
            sample['meta']['flip']=1
        else:
            sample['meta']['flip']=0
        return sample


class RandomRotation(object):
    def __init__(self, angle_range=30, if_expand=False, mode=None, elems_point=['pos_points_mask','neg_points_mask'], elems_do=None, elems_undo=[]):
        self.angle_range = (-angle_range, angle_range) if isinstance(angle_range, numbers.Number) else angle_range
        self.if_expand, self.mode = if_expand, mode
        self.elems_point = elems_point
        self.elems_do, self.elems_undo = elems_do, (['meta']+elems_undo)

    def __call__(self, sample):
        angle = random.uniform(self.angle_range[0], self.angle_range[1])
        for elem in sample.keys():
            if self.elems_do!= None  and elem not in self.elems_do :continue
            if elem in self.elems_undo:continue
            
            if elem in self.elems_point:
                sample[elem]=img_rotate_point(sample[elem], angle, if_expand=self.if_expand)
                continue

            sample[elem]=img_rotate(sample[elem], angle, if_expand=self.if_expand, mode=self.mode)  
        return sample


class Resize(object):
    def __init__(self, size, mode=None,  elems_point=['pos_points_mask','neg_points_mask'], elems_do=None, elems_undo=[]):
        self.size, self.mode = size, mode
        self.elems_point = elems_point
        self.elems_do, self.elems_undo = elems_do, (['meta']+elems_undo)
    def __call__(self, sample):
        for elem in sample.keys():
            if self.elems_do!= None  and elem not in self.elems_do :continue
            if elem in self.elems_undo:continue

            if elem in self.elems_point:
                sample[elem]=img_resize_point(sample[elem],self.size)
                continue

            if self.mode is None: 
                mode = cv2.INTER_LINEAR if len(sample[elem].shape)==3 else cv2.INTER_NEAREST
            sample[elem] = cv2.resize(sample[elem], self.size, interpolation=mode)
            
        return sample


#扩充边界pad(上下左右)
class Expand(object):
    def __init__(self, pad=(0,0,0,0), elems_do=None, elems_undo=[]):
        if isinstance(pad, int):
            self.pad=(pad, pad, pad, pad)
        elif len(pad)==2:
            self.pad=(pad[0],pad[0],pad[1],pad[1])
        elif len(pad)==4:
            self.pad= pad
        self.elems_do, self.elems_undo = elems_do, (['meta']+elems_undo)
    def __call__(self, sample):
        for elem in sample.keys():
            if self.elems_do!= None  and elem not in self.elems_do :continue
            if elem in self.elems_undo:continue
            sample[elem]=cv2.copyMakeBorder(sample[elem],self.pad[0],self.pad[1],self.pad[2],self.pad[3],cv2.BORDER_CONSTANT)  
        return sample


class Crop(object):
    def __init__(self, x_range, y_range, elems_do=None, elems_undo=[]):
        self.x_range, self.y_range = x_range, y_range
        self.elems_do, self.elems_undo = elems_do, (['meta']+elems_undo)
    def __call__(self, sample):
        for elem in sample.keys():
            if self.elems_do!= None  and elem not in self.elems_do :continue
            if elem in self.elems_undo:continue
            sample[elem]=sample[elem][self.y_range[0]:self.y_range[1], self.x_range[0]:self.x_range[1], ...]

        sample['meta']['crop_size'] = np.array((self.x_range[1]-self.x_range[0],self.y_range[1]-self.y_range[0]))
        sample['meta']['crop_lt'] = np.array((self.x_range[0],self.y_range[0])) 
        return sample


class RandomScale(object):
    def __init__(self, scale=(0.75, 1.25), elems_do=None, elems_undo=[]):
        self.scale = scale
        self.elems_do, self.elems_undo = elems_do, (['meta']+elems_undo)
    def __call__(self, sample):
        scale_tmp = random.uniform(self.scale[0], self.scale[1])
        src_size=sample['gt'].shape[::-1]
        dst_size= ( int(src_size[0]*scale_tmp), int(src_size[1]*scale_tmp))
        Resize(size=dst_size)(sample)   
        return sample


########################################[ RGBD_SOD ]########################################
class Depth2RGB(object):
    def __init__(self):
        pass
    def __call__(self, sample):
        #print('->old:',sample['depth'].size())
        sample['depth']=sample['depth'].repeat(3,1,1)
        #print('->new:',sample['depth'].size())
        return sample

