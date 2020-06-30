import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.vgg_new import VGG_backbone

def init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(m.bias, -bound, bound) 
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

class Decoder(nn.Module):
    def __init__(self,in_channel=32,side_channel=512):
        super(Decoder, self).__init__()
        self.reduce_conv=nn.Sequential(
            #nn.Conv2d(side_channel, in_channel, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(side_channel, in_channel, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)  ###
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channel*2, in_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)  ###
        )
        init_weight(self)

    def forward(self, x, side):
        x=F.interpolate(x, size=side.size()[2:], mode='bilinear', align_corners=True)
        side=self.reduce_conv(side)
        x=torch.cat((x, side), 1)
        x = self.decoder(x)
        return x


class Single_Stream(nn.Module):
    def __init__(self,in_channel=3):
        super(Single_Stream, self).__init__()
        self.backbone =  VGG_backbone(in_channel=in_channel,pre_train_path='./model/vgg16_feat.pth')
        self.toplayer = nn.Sequential(
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(512, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True) ###
        )
        channels = [64, 128, 256, 512, 512, 32]
        # Decoders
        decoders = []
        for idx in range(5):
            decoders.append(Decoder(in_channel=32,side_channel=channels[idx]))
        self.decoders = nn.ModuleList(decoders)
        init_weight(self.toplayer)

    def forward(self, input):
        l1 = self.backbone.conv1(input)
        l2 = self.backbone.conv2(l1)
        l3 = self.backbone.conv3(l2)
        l4 = self.backbone.conv4(l3)
        l5 = self.backbone.conv5(l4)
        l6 = self.toplayer(l5)
        feats=[l1, l2, l3, l4, l5, l6]

        x=feats[5]
        for idx in [4, 3, 2, 1, 0]:
            x=self.decoders[idx](x,feats[idx])

        return x

class PredLayer(nn.Module):
    def __init__(self, in_channel=32):
        super(PredLayer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.outlayer = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        init_weight(self)
        
    def forward(self, x):
        x = self.enlayer(x)
        x = self.outlayer(x)
        return x


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        # Main-streams
        self.main_stream = Single_Stream(in_channel=3)

        # Prediction
        self.pred_layer = PredLayer()

    def forward(self, input, if_return_feat=False):
        rgb, dep=input
        dep=dep.repeat(1,3,1,1)
        feat = self.main_stream(dep)
        result = self.pred_layer(feat)

        if if_return_feat:
            return feat
        else:
            return result

    def get_train_params(self, lr):
        train_params = [{'params': self.parameters(), 'lr': lr}]
        return train_params
    
    def get_input(self, sample_batched):
        rgb,dep = sample_batched['img'].cuda(),sample_batched['depth'].cuda()
        return rgb,dep
    
    def get_gt(self, sample_batched):
        gt = sample_batched['gt'].cuda()
        return gt
    
    def get_result(self, output, index=0):
        if isinstance(output, list):
            result = output[0].data.cpu().numpy()[index,0,:,:]
        else:
            result = output.data.cpu().numpy()[index,0,:,:]

        # if isinstance(output, list):
        #     result = torch.sigmoid(output[0].data.cpu()).numpy()[index,0,:,:]
        # else:
        #     result = torch.sigmoid(output.data.cpu()).numpy()[index,0,:,:]
        return result
    
    def get_loss(self, output, gt, if_mean=True):
        criterion = nn.BCELoss().cuda()
        #criterion = nn.BCEWithLogitsLoss().cuda()
        if isinstance(output, list):
            loss=0
            for i in range(len(output)):
                loss+=criterion(output[i], gt)
            if if_mean:loss/=len(output)
        else:
            loss = criterion(output, gt)
        return loss


if __name__ == "__main__":
    pass






