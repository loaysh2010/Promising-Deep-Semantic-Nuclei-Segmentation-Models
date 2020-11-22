import cv2
from PIL import Image
from matplotlib import pyplot as plt
#******************************************#
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import accuracy_score,jaccard_score
#******************************************************************#
import os
import math
import time
import datetime
import shutil
import random
from random import  shuffle
#***********************************#
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
#************************************#
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
#*****************************************************************#
num_classes=2
batch_size=2
num_epoch=10
factor=2

#******************* Model ******************************#
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

weight_decay=1e-8
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-1,momentum=0.99, weight_decay=weight_decay)
#****************************************************************#
#***************** Data ****************************#
# datapath='Data/Original_Data'
datapath='Data/Normalized_Data'

Train_ImageMainPath=datapath+'/train'
Test_ImageMainPath=datapath+'/test'

# img_size=(512,512) # for FCNs & SegNets
# ann_size=(512,512)

img_size=(256,256) # for DenseNet103 & UNet
ann_size=(256,256)

# img_size=(512,512) # for SelfCorrection
# ann_size=(128,128)

train_Dataset = MyDataset(Train_ImageMainPath,img_size=img_size,ann_siz=ann_size)
test_Dataset = MyDataset(Test_ImageMainPath,img_size=img_size,ann_siz=ann_size)

train_loader = DataLoader(train_Dataset,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_Dataset,batch_size=1,shuffle=True)
#****************************************************************#
#***************************** Prepair for Training *************************#
print('')
print('No. of Training Data: ',len(train_Dataset))
print('Model: ',Modelname.split('.')[0])
if os.path.isfile('models/'+Modelname):
    model.load_state_dict(torch.load('models/'+Modelname))
    print('exist')
print('********************************')    
#***************************************************************#
#**==================== Helper Functions ======================*#
def GetTestList(model,Test_loader):   

    model.eval()
    All_AnnList=[]
    All_PredList=[]
    All_Acc = 0
    All_F1 = 0
    for _ , img, ann in test_loader :  
        AnnList=[]
        PredList=[]
        
        ann=np.asarray(ann.data.cpu(),dtype=np.int8)
        All_AnnList.append(ann)
        AnnList.append(ann)
        
        # out=model(img)
        # _,out2=torch.max(out,1)
        out = model(img)['out'][0] # for FCN with ResNet101
        out2 = out.argmax(0)

        out2=out2.data.cpu().numpy()
        out2=np.array(out2,dtype=np.uint8)
        All_PredList.append(out2)
        PredList.append(out2)

        PredList=np.array(PredList)
        AnnList=np.array(AnnList)
        PredList=PredList.reshape((-1,))
        AnnList=AnnList.reshape((-1,))

        Acc = accuracy_score(AnnList, PredList)
        All_Acc+=Acc
        F1=f1_score(AnnList, PredList, average='macro')
        All_F1 += F1
    
    Avg_Acc = All_Acc/len(test_loader)
    Avg_F1 = All_F1/len(test_loader)
    
    print('======================================')
    print('Avg_F1_score=', Avg_F1 )
    print('Avg_accuracy_score=', Avg_Acc)

    return All_AnnList,All_PredList

def calc_Accuracy(model,test_loader,Best_IoU,Best_F1):
    
    AnnList,PredList = GetTestList(model,test_loader)

    PredList= np.array(PredList)
    AnnList= np.array(AnnList)

    PredList= PredList.reshape((-1,))
    AnnList= AnnList.reshape((-1,))

    Acc = accuracy_score(AnnList, PredList)
    F1 = f1_score(AnnList,PredList,average='macro')
    av_iou = jaccard_score(AnnList,PredList,average='macro')

    print('======================================')
    print('av_iou=',av_iou,',Best_IoU=',Best_IoU)
    print('precision_score=',precision_score(AnnList, PredList, average='macro')  )
    print('recall_score=',recall_score(AnnList, PredList, average='macro')  )
    print('F1_score=', F1 )
    print('accuracy_score=', Acc)


    if av_iou>Best_IoU:
        Best_IoU=av_iou
    #     torch.save(model.state_dict(), 'models/'+Modelname)
    #     print('model has been changed',datetime.datetime.now())
    # print('*********************************')

    if F1>Best_F1:
        Best_F1=F1
        torch.save(model.state_dict(), 'models/'+Modelname)
        print('model has been changed',datetime.datetime.now())
    print('*********************************')

    return Best_IoU,Best_F1
#***************************************************************#
print('')
Best_IoU=-1000
Best_F1=-1000
print('Before of Training')
print('-----------------')
print('')
Best_IoU,Best_F1= calc_Accuracy(model,test_loader,Best_IoU,Best_F1)
#*********************************************************************#
#*********************** Training ************************************#
Loss=[]
IoU_List=[]
F1_List=[]

print('')
print('Start Training')
print('-----------------')
for epoch in range(num_epoch): 
    total=0    
    counter=0    
    step_counter=0
    
    model.train()
    for _ , img, ann in train_loader :
        
        loss=0 
        optimizer.zero_grad()
        
        ann=ann.type(torch.int64)
        # out=model(img)
        out = model(img)['out']  # for FCN8s with ResNet101
        
        loss=(5/1000.0)*criterion(out , ann)

        loss.backward(retain_graph=True)
        optimizer.step()

        total=total+loss.data
        step_counter=step_counter+1
        
        if (step_counter) % 324==0 and step_counter >0:
            print ('Epoch:', epoch+1,'step',step_counter,'Last Batch loss %.4f:' %loss.data )
            Loss.append(1000*loss.data.cpu().numpy())
        
        # if (step_counter) % 290 ==0 and step_counter >0:
        #     print('')
        #     print('Start Validating')
        #     print('-------------')               

        #     Best_IoU,Best_F1= calc_Accuracy(model,test_loader,Best_IoU,Best_F1)
        #     model.train()
    
    print('End of epoch')
    print('')
    print('Start Validating after end of epoch: ', epoch+1)
    print('-------------')               

    Best_IoU,Best_F1= calc_Accuracy(model,test_loader,Best_IoU,Best_F1)
    Loss.append(total/len(train_loader))
    F1_List.append(Best_F1)
    IoU_List.append(Best_IoU)

plt.subplot(3,1,1)
plt.title('Loss')
plt.plot(Loss,color='red')

plt.subplot(3,1,2)
plt.title('Dice')
plt.plot(F1_List,color='orange')

plt.subplot(3,1,3)
plt.title('Jaccard')
plt.plot(IoU_List,color='blue')


plt.savefig('training_plots'+'/'+Modelname.split('.')[0]+'.png')
plt.close()