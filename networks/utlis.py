import shutil

from random import  shuffle
import os
import numpy as np

import random

from skimage.io import imread,imsave
import pickle
import cv2

import torchvision


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models

from torch.autograd import Variable

#from torchsummary import summary


BatchNorm = nn.BatchNorm2d





import time
import torch.optim as optim


import matplotlib.pyplot as plt
from skimage import filters

from skimage.io import imread,imsave




def accuracy(seg, pred):    
    _, preds = torch.max(pred.data.cpu(), dim=1)
    segs=seg.data.cpu()   
    
    valid = (segs >= 0)
    acc = 1.0 * torch.sum(valid * (preds == segs)) / (torch.sum(valid) + 1e-10)
    return acc, torch.sum(valid)


class Upsample(nn.Module):
    def __init__(self, inplanes, planes):
        super(Upsample, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x, size):
        x = F.upsample_bilinear(x, size=size)
        x = self.conv1(x)
        x = self.bn(x)
        return x



class Fusion(nn.Module):
    def __init__(self, inplanes):
        super(Fusion, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size=1)
        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        
        out = self.bn(self.conv(x1)) + x2
        out = self.relu(out)

        return out


class SegNet(nn.Module):
    def __init__(self, num_classes):
        super(SegNet, self).__init__()

        self.num_classes = num_classes

        resnet = models.resnet101(pretrained=True)

        self.conv1 = resnet.conv1
        self.bn0 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.upsample1 = Upsample(2048, 1024)
        self.upsample2 = Upsample(1024, 512)
        self.upsample3 = Upsample(512, 64)
        self.upsample4 = Upsample(64, 64)
        self.upsample5 = Upsample(64, 32)

        self.fs1 = Fusion(1024)
        self.fs2 = Fusion(512)
        self.fs3 = Fusion(256)
        self.fs4 = Fusion(64)
        self.fs5 = Fusion(64)

        #self.out0 = self._classifier(2048)
        #self.out1 = self._classifier(1024)
        #self.out2 = self._classifier(512)
        #self.out_e = self._classifier(256)
        #self.out3 = self._classifier(64)
        #self.out4 = self._classifier(64)

        self.out5 = self._classifier(32)

        #self.out6 = self._reconstructor(32)





    def _classifier(self, inplanes):
        if inplanes == 32:
            return nn.Sequential(
                nn.Conv2d(inplanes, self.num_classes, 1),
                nn.Conv2d(self.num_classes, self.num_classes,
                          kernel_size=3, padding=1)
            )
        return nn.Sequential(
            nn.Conv2d(inplanes,int(inplanes/2) , 3, padding=1, bias=False),
            nn.BatchNorm2d(int(inplanes/2)),
            nn.ReLU(inplace=True),
            nn.Dropout(.1),
            nn.Conv2d(int(inplanes/2), self.num_classes, 1),
        )

    # def _reconstructor(self, inplanes):
    #     if inplanes == 32:
    #         return nn.Sequential(
    #             nn.Conv2d(inplanes, self.num_classes, 1),
    #             nn.Conv2d(self.num_classes, 3,
    #                       kernel_size=3, padding=1)
    #         )
    #     return nn.Sequential(
    #         nn.Conv2d(inplanes,int(inplanes/2) , 3, padding=1, bias=False),
    #         nn.BatchNorm2d(int(inplanes/2)),
    #         nn.ReLU(inplace=True),
    #         nn.Dropout(.1),
    #         nn.Conv2d(int(inplanes/2), 3, 1),
    #     )

    def forward(self, x):
       

        #print(x.shape)

        #x = F.Variable(torch.from_numpy(x).cuda())
        #x=x.float()

        input = x

        #print()

       

        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        conv_x = x
        x = self.maxpool(x)
        pool_x = x

        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        #out32 = self.out0(fm4)

        #print('out32 size()',out32.size())

        fsfm1 = self.fs1(fm3, self.upsample1(fm4, fm3.size()[2:]))
        #out16 = self.out1(fsfm1)

        fsfm2 = self.fs2(fm2, self.upsample2(fsfm1, fm2.size()[2:]))
        #out8 = self.out2(fsfm2)
        

        fsfm3 = self.fs4(pool_x, self.upsample3(fsfm2, pool_x.size()[2:]))
        # print(fsfm3.size())
        #out4 = self.out3(fsfm3)

        fsfm4 = self.fs5(conv_x, self.upsample4(fsfm3, conv_x.size()[2:]))

        #out2 = self.out4(fsfm4)

        fsfm5 = self.upsample5(fsfm4, input.size()[2:])
        out = self.out5(fsfm5)
        #original=self.out6(fsfm5)



        return out  #out2, out4, out8, out16, out32


class FCN(nn.Module):
    def __init__(self, num_classes):
        super(FCN, self).__init__()

        self.num_classes = num_classes

        resnet = models.resnet101(pretrained=True)

        self.conv1 = resnet.conv1
        self.bn0 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.upsample1 = Upsample(2, 2)   
        

        

        #self.out0 = self._classifier(2048)
        #self.out1 = self._classifier(1024)
        #self.out2 = self._classifier(512)
        #self.out_e = self._classifier(256)
        #self.out3 = self._classifier(64)
        #self.out4 = self._classifier(64)

        self.out0 = self._classifier(2)
        self.last_conv=nn.Conv2d(2048,num_classes, kernel_size=5, padding=2)

        #self.out6 = self._reconstructor(32)





    def _classifier(self, inplanes):
        if inplanes == 32:
            return nn.Sequential(
                nn.Conv2d(inplanes, self.num_classes, 1),
                nn.Conv2d(self.num_classes, self.num_classes,
                          kernel_size=3, padding=1)
            )
        return nn.Sequential(
            nn.Conv2d(inplanes,int(inplanes/2) , 3, padding=1, bias=False),
            nn.BatchNorm2d(int(inplanes/2)),
            nn.ReLU(inplace=True),
            nn.Dropout(.1),
            nn.Conv2d(int(inplanes/2), self.num_classes, 1),
        )

    

    def forward(self, x):
       

        

        input = x
        input_size=input.size()

        #print()

       

        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        conv_x = x
        x = self.maxpool(x)
        pool_x = x

        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)
       
       


        out32=self.last_conv(fm4)
        #print('out32',out32.size())        
        out32=self.upsample1(out32,input_size[2:] )
        out32=self.out0(out32)


        #print('input_size=',input_size)

        #print('out32',out32.size())



        
        return out32  #out2, out4, out8, out16, out32


class SegNetMax(nn.Module):
    def __init__(self, num_classes):
        super(SegNetMax, self).__init__()

        self.num_classes = num_classes

        resnet = models.resnet101(pretrained=True)

        self.conv1 = resnet.conv1
        self.bn0 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.upsample1 = Upsample(2048+1, 1024+1)
        self.upsample2 = Upsample(1024+1, 512+1)
        self.upsample3 = Upsample(512+1, 64+1)
        self.upsample4 = Upsample(64+1, 64+1)
        self.upsample5 = Upsample(64+1, 32)

        self.fs1 = Fusion(1024+1)
        self.fs2 = Fusion(512+1)
        #self.fs3 = Fusion(256)
        self.fs4 = Fusion(64+1)
        self.fs5 = Fusion(64+1)

        #self.out0 = self._classifier(2048)
        #self.out1 = self._classifier(1024)
        #self.out2 = self._classifier(512)
        #self.out_e = self._classifier(256)
        #self.out3 = self._classifier(64)
        #self.out4 = self._classifier(64)

        self.out5 = self._classifier(32)

        #self.out6 = self._reconstructor(32)





    def _classifier(self, inplanes):
        if inplanes == 32:
            return nn.Sequential(
                nn.Conv2d(inplanes, self.num_classes, 1),
                nn.Conv2d(self.num_classes, self.num_classes,
                          kernel_size=3, padding=1)
            )
        return nn.Sequential(
            nn.Conv2d(inplanes,int(inplanes/2) , 3, padding=1, bias=False),
            nn.BatchNorm2d(int(inplanes/2)),
            nn.ReLU(inplace=True),
            nn.Dropout(.1),
            nn.Conv2d(int(inplanes/2), self.num_classes, 1),
        )

    
    def forward(self, x):

        
       

        #print(x.shape)

        #x = F.Variable(torch.from_numpy(x).cuda())
        #x=x.float()

        input = x

        #print()

       

        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        conv_x = x
        x = self.maxpool(x)
        pool_x = x

        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)


        

        #out32 = self.out0(fm4)

        #print('out32 size()',out32.size())

        

        mx_=torch.max(fm4,dim=1,keepdim=True)[0]
        fm4=torch.cat([fm4,mx_],1)

        mx_=torch.max(fm3,dim=1,keepdim=True)[0]        
        fm3=torch.cat([fm3,mx_],1)
       


        fsfm1 = self.fs1(fm3,self.upsample1(fm4, fm3.size()[2:]) )   



        


        mx_=torch.max(fm2,dim=1,keepdim=True)[0]        
        fm2=torch.cat([fm2,mx_],1)



        

        fsfm2 = self.fs2(fm2, self.upsample2(fsfm1, fm2.size()[2:]))

        
        mx_=torch.max(pool_x,dim=1,keepdim=True)[0]        
        pool_x=torch.cat([pool_x,mx_],1)
        

    

                

        fsfm3 = self.fs4(pool_x,self.upsample3(fsfm2, pool_x.size()[2:])  )        


        mx_=torch.max(conv_x,dim=1,keepdim=True)[0]        
        conv_x=torch.cat([conv_x,mx_],1)

        #print()''



        

        fsfm4 = self.fs5(conv_x, self.upsample4(fsfm3, conv_x.size()[2:]))

        

        fsfm5 = self.upsample5(fsfm4, input.size()[2:])

        

        out = self.out5(fsfm5)
        #original=self.out6(fsfm5)



        return out  #out2, out4, out8, out16, out32



class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins, BatchNorm):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                BatchNorm(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)
    
    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)



class PSPSegNetMax(nn.Module):
    def __init__(self, num_classes):
        super(PSPSegNetMax, self).__init__()

        
        self.num_classes = num_classes

        resnet = models.resnet101(pretrained=True)

        self.conv1 = resnet.conv1
        self.bn0 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.upsample1 = Upsample(2048+1, 1024+1)
        self.upsample2 = Upsample(1024+1, 512+1)
        self.upsample3 = Upsample(512+1, 64+1)
        self.upsample4 = Upsample(64+1, 64+1)
        self.upsample5 = Upsample(64+1, 32)

        self.fs1 = Fusion(1024+1)
        self.fs2 = Fusion(512+1)
        #self.fs3 = Fusion(256)
        self.fs4 = Fusion(64+1)
        self.fs5 = Fusion(64+1)

        fea_dim=2048
        bins=(1, 2, 3, 6)
        self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins, BatchNorm)
        self.conv_after_ppm=nn.Conv2d(2*fea_dim,fea_dim , kernel_size=1, bias=False)


        #self.out0 = self._classifier(2048)
        #self.out1 = self._classifier(1024)
        #self.out2 = self._classifier(512)
        #self.out_e = self._classifier(256)
        #self.out3 = self._classifier(64)
        #self.out4 = self._classifier(64)

        self.out5 = self._classifier(32)

        #self.out6 = self._reconstructor(32)





    def _classifier(self, inplanes):
        if inplanes == 32:
            return nn.Sequential(
                nn.Conv2d(inplanes, self.num_classes, 1),
                nn.Conv2d(self.num_classes, self.num_classes,
                          kernel_size=3, padding=1)
            )
        return nn.Sequential(
            nn.Conv2d(inplanes,int(inplanes/2) , 3, padding=1, bias=False),
            nn.BatchNorm2d(int(inplanes/2)),
            nn.ReLU(inplace=True),
            nn.Dropout(.1),
            nn.Conv2d(int(inplanes/2), self.num_classes, 1),
        )

    
    def forward(self, x):

        
       

        #print(x.shape)

        #x = F.Variable(torch.from_numpy(x).cuda())
        #x=x.float()

        input = x

        #print()

       

        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        conv_x = x
        x = self.maxpool(x)
        pool_x = x

        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        fm4=F.upsample_bilinear(fm4, size=fm2.size()[2:] )
        fm4=self.ppm(fm4)
        fm4=self.conv_after_ppm(fm4)


        

        
        #print('fm4=',fm4.size())
        #out32 = self.out0(fm4)
        #print('out32 size()',out32.size())



        

        mx_=torch.max(fm4,dim=1,keepdim=True)[0]
        fm4=torch.cat([fm4,mx_],1)

        mx_=torch.max(fm3,dim=1,keepdim=True)[0]        
        fm3=torch.cat([fm3,mx_],1)
       


        fsfm1 = self.fs1(fm3,self.upsample1(fm4, fm3.size()[2:]) )   



        


        mx_=torch.max(fm2,dim=1,keepdim=True)[0]        
        fm2=torch.cat([fm2,mx_],1)


        



        

        fsfm2 = self.fs2(fm2, self.upsample2(fsfm1, fm2.size()[2:]))

        
        mx_=torch.max(pool_x,dim=1,keepdim=True)[0]        
        pool_x=torch.cat([pool_x,mx_],1)
        

    

                

        fsfm3 = self.fs4(pool_x,self.upsample3(fsfm2, pool_x.size()[2:])  )        


        mx_=torch.max(conv_x,dim=1,keepdim=True)[0]        
        conv_x=torch.cat([conv_x,mx_],1)

        #print()''



        

        fsfm4 = self.fs5(conv_x, self.upsample4(fsfm3, conv_x.size()[2:]))

        

        fsfm5 = self.upsample5(fsfm4, input.size()[2:])

        

        out = self.out5(fsfm5)
        #original=self.out6(fsfm5)



        return out  #out2, out4, out8, out16, out32


class PSPSegNet(nn.Module):
    def __init__(self, num_classes):
        super(PSPSegNet, self).__init__()

        
        self.num_classes = num_classes

        resnet = models.resnet101(pretrained=True)

        self.conv1 = resnet.conv1
        self.bn0 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.upsample1 = Upsample(2048, 1024)
        self.upsample2 = Upsample(1024, 512)
        self.upsample3 = Upsample(512, 64)
        self.upsample4 = Upsample(64, 64)
        self.upsample5 = Upsample(64, 32)

        self.fs1 = Fusion(1024)
        self.fs2 = Fusion(512)
        #self.fs3 = Fusion(256)
        self.fs4 = Fusion(64)
        self.fs5 = Fusion(64)

        fea_dim=2048
        bins=(1, 2, 3, 6)
        self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins, BatchNorm)
        self.conv_after_ppm=nn.Conv2d(2*fea_dim,fea_dim , kernel_size=1, bias=False)


        #self.out0 = self._classifier(2048)
        #self.out1 = self._classifier(1024)
        #self.out2 = self._classifier(512)
        #self.out_e = self._classifier(256)
        #self.out3 = self._classifier(64)
        #self.out4 = self._classifier(64)

        self.out5 = self._classifier(32)

        #self.out6 = self._reconstructor(32)





    def _classifier(self, inplanes):
        if inplanes == 32:
            return nn.Sequential(
                nn.Conv2d(inplanes, self.num_classes, 1),
                nn.Conv2d(self.num_classes, self.num_classes,
                          kernel_size=3, padding=1)
            )
        return nn.Sequential(
            nn.Conv2d(inplanes,int(inplanes/2) , 3, padding=1, bias=False),
            nn.BatchNorm2d(int(inplanes/2)),
            nn.ReLU(inplace=True),
            nn.Dropout(.1),
            nn.Conv2d(int(inplanes/2), self.num_classes, 1),
        )

    
    def forward(self, x):

        
       

        #print(x.shape)

        #x = F.Variable(torch.from_numpy(x).cuda())
        #x=x.float()

        input = x

        #print()

       

        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        conv_x = x
        x = self.maxpool(x)
        pool_x = x

        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        fm4=F.upsample_bilinear(fm4, size=fm2.size()[2:] )
        fm4=self.ppm(fm4)
        fm4=self.conv_after_ppm(fm4)



        fsfm1 = self.fs1(fm3,self.upsample1(fm4, fm3.size()[2:]) )
        fsfm2 = self.fs2(fm2, self.upsample2(fsfm1, fm2.size()[2:]))
        fsfm3 = self.fs4(pool_x,self.upsample3(fsfm2, pool_x.size()[2:])  )        
        fsfm4 = self.fs5(conv_x, self.upsample4(fsfm3, conv_x.size()[2:]))
        fsfm5 = self.upsample5(fsfm4, input.size()[2:])
        out = self.out5(fsfm5)

        return out  #out2, out4, out8, out16, out32



# if use_ppm:
#     self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins, BatchNorm)


#***********************************************************
import torch.nn as nn
import torch.nn.functional as F


#***********************************************************

def ResizeMask(old_mask,sz):  
    

    unq=np.unique(old_mask)
    
    unq=unq[1:]

    
    new_mask=np.zeros((sz[1],sz[0]),dtype=np.uint8)

    

    for i in range(unq.shape[0]):
       inds=np.where(old_mask==unq[i])
       tmp=0*old_mask
       tmp[inds]=1
       tmp=cv2.resize(tmp,sz)
       
       inds=np.where(tmp==1)
       new_mask[inds]=unq[i]
    
    return new_mask


def label2color(img,colors):    
    r_pred=0*img
    g_pred=0*img
    b_pred=0*img
    for cl in range(colors.shape[0]):
        c=colors[cl]                
        inds=np.where((img == cl))
        r_pred[inds]=c[0]
        g_pred[inds]=c[1]
        b_pred[inds]=c[2]
    
    
    x=np.concatenate((r_pred,g_pred,b_pred),axis=2)

    x=np.array(x,dtype=np.uint8)
    #print('************************************')
    return x



content=[
[0,0,0],
[75,75,75],

[205,140,240],
[110,155,250],
[105,150,35],

[90,0,50],
[100,100,200],
[225,185,45],


[150,255,255],
[95,155,150],
[90,0,250],

#******************************************

[160,95,80],
[55,55,15],
[220,0,55],
#****************************

[200,200,250],
[90,55,150],
[240,205,155],
]







