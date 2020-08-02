#pytorch 
import torch
import torchvision
from torch.utils.data import DataLoader

#general 
import os
import cv2
import sys
import time
import math
import random
import shutil
import argparse
import numpy as np
from tqdm import tqdm

#mine
import utils
import my_custom_transforms as mtr
from dataloader_rgbdsod import RgbdSodDataset

#log_recorder
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
	    self.terminal = stream
	    self.log = open(filename, 'w')
    def write(self, message):
	    self.terminal.write(message)
	    self.log.write(message)
    def flush(self):
	    pass

def SetLogFile(file_path='log'):
    sys.stdout = Logger(file_path, sys.stdout)

parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str, default='RgbNet',choices=['RgbNet','RgbdNet','DepthNet'],help='train net')
args = parser.parse_args()

utils.set_seed(10)

p={}
p['datasets_path']='./dataset/'
p['train_datasets']=[p['datasets_path']+'NJU2K_TRAIN',p['datasets_path']+'NLPR_TRAIN']
p['val_datasets']=[p['datasets_path']+'NJU2K_TEST']

p['gpu_ids']=list(range(torch.cuda.device_count()))
p['start_epoch']=0
p['epochs']=30
p['bs']=8*len(p['gpu_ids'])  
p['lr']=1.25e-5*(p['bs']/len(p['gpu_ids']))
p['num_workers']=4*len(p['gpu_ids'])

p['optimizer']=[ 'Adam'   , {} ]
p['scheduler']=['Constant',{}]

p['if_memory']=False
p['max_num']= 0
p['size']=(224, 224)
p['train_only_epochs']=0
p['val_interval']=1
p['resume']= None
p['model']=args.net

p['note']=''
p['if_use_tensorboard']=False
p['snapshot_path']='snapshot/[{}]_[{}]'.format(time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(time.time())),p['model'])
if p['note']!='': p['snapshot_path']+='_[{}]'.format(p['note'])

p['if_debug']=0

p['if_only_val']=0 if p['resume'] is  None else 1
p['if_save_checkpoint']=False

if p['if_only_val']:
    p['snapshot_path']+='[val]'
    p['if_use_tensorboard']=False

if p['if_debug']:
    if os.path.exists('snapshot/debug'):shutil.rmtree('snapshot/debug')
    p['snapshot_path']='snapshot/debug'
    p['max_num']=32

exec('from model.{} import MyNet'.format(p['model']))

if p['if_use_tensorboard']:
    from torch.utils.tensorboard import SummaryWriter

class Trainer(object):
    def __init__(self,p):
        self.p=p
        os.makedirs(p['snapshot_path'],exist_ok=True)
        shutil.copyfile(os.path.join('model',p['model']+'.py'), os.path.join(p['snapshot_path'],p['model']+'.py'))
        SetLogFile('{}/log.txt'.format(p['snapshot_path']))  
        if p['if_use_tensorboard']:
            self.writer = SummaryWriter(p['snapshot_path'])
            
        transform_train = torchvision.transforms.Compose([  
                                                            mtr.RandomFlip(),
                                                            mtr.Resize(p['size']),
                                                            mtr.ToTensor(),
                                                            mtr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],elems_do=['img']),
                                                            
                                                         ])

        transform_val = torchvision.transforms.Compose([  
                                                    mtr.Resize(p['size']),
                                                    mtr.ToTensor(),
                                                    mtr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],elems_do=['img']),
                                                    ])

        self.train_set = RgbdSodDataset(datasets=p['train_datasets'],transform=transform_train,max_num=p['max_num'],if_memory=p['if_memory'])
        self.train_loader = DataLoader(self.train_set, batch_size=p['bs'], shuffle=True, num_workers=p['num_workers'],pin_memory=True)

        self.val_loaders=[]
        for val_dataset in p['val_datasets']:
            val_set=RgbdSodDataset(val_dataset,transform=transform_val,max_num=p['max_num'],if_memory=p['if_memory'])
            self.val_loaders.append(DataLoader(val_set, batch_size=1, shuffle=False,pin_memory=True))

        self.model=MyNet()
    
        self.model = self.model.cuda()

        self.optimizer = utils.get_optimizer(p['optimizer'][0], self.model.get_train_params(lr=p['lr']), p['optimizer'][1])
        self.scheduler = utils.get_scheduler(p['scheduler'][0], self.optimizer, p['scheduler'][1])

        self.best_metric=None

        if p['resume']!=None:
            print('Load checkpoint from [{}]'.format(p['resume']))
            checkpoint = torch.load(p['resume'])
            self.p['start_epoch']=checkpoint['current_epoch']+1
            self.best_metric=checkpoint['best_metric']
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])

    def main(self):
        print('Start time : ',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        print('---[   NOTE: {}   ]---'.format(self.p['note']))
        print('-'*79,'\ninfos : ' , self.p, '\n'+'-'*79)

        if self.p['if_only_val']:
            result_save_path=os.path.join(p['snapshot_path'],'result')
            os.makedirs(result_save_path,exist_ok=True)
            self.validation(self.p['start_epoch']-1,result_save_path)
            exit()

        for epoch in range(self.p['start_epoch'],self.p['epochs']):
            lr_str = ['{:.7f}'.format(i) for i in self.scheduler.get_lr()]
            print('-'*79+'\n'+'Epoch [{:03d}]=>    |-lr:{}-|  \n'.format(epoch, lr_str))
            #training
            if p['train_only_epochs']>=0:
                self.training(epoch)
                self.scheduler.step()
            
            if epoch<p['train_only_epochs']: continue
            #validation
            if (epoch+1) % p['val_interval']==0:
                self.validation(epoch)
        
        if self.p['if_use_tensorboard']:self.writer.close()
        print('-'*79+'\nEnd time : ', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))


    def training(self, epoch): 
        print('Training :')
        loss_total = 0
        self.model.train()
        tbar = tqdm(self.train_loader)
        for i, sample_batched in enumerate(tbar):
            self.optimizer.zero_grad()
            input = self.model.get_input(sample_batched)
            gt = self.model.get_gt(sample_batched)
            output = self.model(input)
            loss = self.model.get_loss(output, gt)
            loss_total+=loss.item()
            loss.backward()
            self.optimizer.step()
            tbar.set_description('Loss: %.3f' % (loss_total / (i + 1)))
        print('Loss: %.3f' % (loss_total / (i + 1)))
        if self.p['if_use_tensorboard']:self.writer.add_scalar('Loss/train', (loss_total / (i + 1)), epoch)

 
    def validation(self, epoch, result_save_path=None):
        print('Validation :')
        self.model.eval()
        metric_all=np.zeros(2)
        for index, val_loader in enumerate(self.val_loaders):
            dataset=self.p['val_datasets'][index].split('/')[-1]
            print('Validation [{}]'.format(dataset))

            result_save_path_tmp=None
            if result_save_path is not None:
                result_save_path_tmp=os.path.join(result_save_path, dataset)
                os.makedirs(result_save_path_tmp,exist_ok=True)

            loss_total = 0
            tbar = tqdm(val_loader)
            
            mae_avg,f_score_avg=0,0
            for i, sample_batched in enumerate(tbar):
                input = self.model.get_input(sample_batched)
                gt = self.model.get_gt(sample_batched)
                with torch.no_grad(): 
                    output = self.model(input)
                loss = self.model.get_loss(output, gt)
                loss_total+=loss.item()
                tbar.set_description('Loss: {:.3f}'.format(loss_total/(i + 1)))
                result = self.model.get_result(output)

                mae,f_score=utils.get_metric(sample_batched, result,result_save_path_tmp)
                mae_avg,f_score_avg=mae_avg+mae,f_score_avg+f_score

            print('Loss: %.3f' % (loss_total / (i + 1)))
            mae_avg,f_score_avg=mae_avg/len(tbar),f_score_avg/len(tbar)
            
            metric =  np.array([mae_avg, f_score_avg.max().item()])
            print('[{}]-> mae:{:.4f} f_max:{:.4f}'.format(dataset,metric[0],metric[1]))

            metric_all+=metric

        metric_all=metric_all/len(self.val_loaders)
    
        is_best = utils.metric_better_than(metric_all, self.best_metric)
        self.best_metric = metric_all if is_best else self.best_metric

        print('Metric_Select[MAE]: {:.4f} ({:.4f})'.format(metric_all[0],self.best_metric[0]))

        pth_state={
            'current_epoch': epoch,
            'best_metric': self.best_metric,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler':self.scheduler.state_dict()
        }

        if self.p['if_save_checkpoint']:
            torch.save(pth_state, os.path.join(self.p['snapshot_path'], 'checkpoint.pth'))
        if is_best:
            torch.save(pth_state, os.path.join(self.p['snapshot_path'], 'best.pth'))
        
        if self.p['if_use_tensorboard']:
            self.writer.add_scalar('Loss/test', (loss_total / (i + 1)), epoch)
            self.writer.add_scalar('Metric/mae', metric_all[0], epoch)
            self.writer.add_scalar('Metric/f_max', metric_all[1], epoch)


if __name__ == "__main__": 
    mine =Trainer(p)
    mine.main()





















