from PIL import Image
import cv2
from matplotlib import pyplot as plt
#*************************************#
from sklearn.metrics import precision_score,recall_score,confusion_matrix
from sklearn.metrics import accuracy_score,f1_score,jaccard_score
#*****************************#
import os
import math
import random
from random import  shuffle
import time
import datetime
import shutil
#***********************************#
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
#**********************************#
from torch.utils.data import DataLoader
from MyDataset import MyDataset
#********************************************#
# from networks.fcn import VGGNet,FCNs,FCN8s
# from networks.network import U_Net
# from networks.Nested_UNet import Nested_UNet
# from networks.tiramisu import FCDenseNet103
# from networks.segnet import SegNet
# from networks.utlis import SegNet,PSPSegNet
# from networks.selfcorrection import SelfCorrection
#************************************#
num_classes=2
batch_size=1
#******************* Models ******************************#
# Modelname='FCNs.model'
# Modelname='FCN8s.model'
Modelname='FCN8sResNet.model'
# Modelname='UNet.model'
# Modelname='N_UNet.model'
# Modelname='FCDenseNet103.model'
# Modelname='SegNetOrigin.model'
# Modelname='SegNetResNet.model'
# Modelname='PSPSegNet.model'
# Modelname='SelfCorrection.model'

# vgg_model = VGGNet(requires_grad=True, remove_fc=True)
# model = FCNs(pretrained_net=vgg_model, n_class=num_classes)
# model = FCN8s(pretrained_net=vgg_model, n_class=num_classes)
model = torch.hub.load('pytorch/vision:v0.6.0', 'fcn_resnet101', pretrained=False,num_classes=num_classes)
# model = U_Net(output_ch=num_classes)
# model = Nested_UNet(out_ch=num_classes)
# model = FCDenseNet103(n_classes=num_classes)
# model = SegNet(num_classes=2)
# model = PSPSegNet(num_classes=num_classes)
# model = SelfCorrection(num_classes=num_classes)
model.cuda()

model.load_state_dict(torch.load('models/'+Modelname))
model.eval()
#**************************************#
if not os.path.exists('gt/'+Modelname.split('.')[0]):
    os.mkdir('gt/'+Modelname.split('.')[0])

if not os.path.exists('out/'+Modelname.split('.')[0]):
    os.mkdir('out/'+Modelname.split('.')[0])
#*****************************************#
#***************** Data ****************************#
# datapath='Data/Original_Data'
datapath='Data/Normalized_Data'

# Train_ImageMainPath=datapath+'/train'
Test_ImageMainPath=datapath+'/test'

# img_size=(512,512) # for FCNs & SegNets
# ann_size=(512,512)

img_size=(256,256) # for DenseNet103 & UNet
ann_size=(256,256)

# img_size=(512,512) # for SelfCorrection
# ann_size=(128,128)

test_Dataset=MyDataset(Test_ImageMainPath,img_size=img_size,ann_siz=ann_size)

test_loader = DataLoader(test_Dataset,batch_size=batch_size,shuffle=True)
#****************************************************************#
#*********************** Testing ************************************#
Loss=[]

print('')
print('No. of Testing Data: ',len(test_Dataset))
print('Model: ',Modelname.split('.')[0])
print('')
print('Start Testing')
print('-----------------')

PredList=[]
AnnList = []
for f , img, ann in test_loader :

    ann=np.asarray(ann.cpu(),dtype=np.int8)
    ann=np.squeeze(ann)
    cv2.imwrite('gt/'+Modelname.split('.')[0]+'/'+f[0],ann)
    
    # out=model(img)        
    out = model(img)['out']  # for FCN8s with ResNet101

    _,out2=torch.max(out,1)
    out2=out2.cpu().data.numpy()
    out2=np.squeeze(out2)
    out2=np.array(out2,dtype=np.uint8)
    cv2.imwrite('out/'+Modelname.split('.')[0]+'/'+f[0],out2)

    AnnList.append(ann)
    PredList.append(out2)

PredList=np.array(PredList)
AnnList=np.array(AnnList)
# print(PredList.shape)
# print(AnnList.shape)

PredList=PredList.reshape((-1,))
AnnList=AnnList.reshape((-1,))
# print(PredList.shape)
# print(AnnList.shape)

Acc = accuracy_score(AnnList, PredList)
F1=f1_score(AnnList,PredList,average='macro')
av_iou = jaccard_score(AnnList,PredList,average='macro')

print('======================================')
print('precision_score=',precision_score(AnnList, PredList, average='macro')  )
print('recall_score=',recall_score(AnnList, PredList, average='macro')  )
print('av_iou=',av_iou)
print('F1_score=', F1 )
print('accuracy_score=', Acc)
print('======================================')

f=open('results/'+Modelname.split('.')[0]+'.txt',"w+")
f.write(Modelname.split('.')[0]+'\n')
f.write('==============='+'\n')
f.write('precision_score= '+str(precision_score(AnnList, PredList, average='macro'))+'\n'  )
f.write('recall_score= '+str(recall_score(AnnList, PredList, average='macro'))+'\n'  )
f.write('Av_IoU = '+str(av_iou)+'\n'  )
f.write('f1_score = '+str(F1)+'\n')
f.write('accuracy_score= '+str(Acc)+'\n')

