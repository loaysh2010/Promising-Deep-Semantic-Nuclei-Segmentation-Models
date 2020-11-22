import os
import cv2
from matplotlib import image
import matplotlib.pyplot as plt
#*********************************#
import torch
import numpy as np
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from torch.autograd import Variable
#***************************************#
class MyDataset(Dataset):
    def __init__(self, image_paths, img_size=(512,512), ann_siz=(512,512)):
        
        self.images_paths=image_paths
        self.images = os.listdir(image_paths+'/a')
        self.img_size=img_size
        self.ann_size=ann_siz
      
    def __getitem__(self, index):
        
        f = self.images[index] # Image name

        x = cv2.imread(self.images_paths+'/a/'+f)
        x = x / 255.0
        x = cv2.resize(x,self.img_size)
        x = np.transpose(x,(2,0,1))
        x = (Variable(torch.from_numpy(x).cuda().float()))
        
        y = cv2.imread(self.images_paths+'/b/'+f)
        y = cv2.resize(y,self.ann_size)
        y = y[:,:,0]
        y = (Variable(torch.from_numpy(y).cuda()))
        return f, x, y
 
    def __len__(self):
        return len(self.images)
    

if __name__ == "__main__":
    
    datapath='Data/Normalized_Data'
    Train_data_Path=datapath+'/train'
    # Test_data_Path=datapath+'/test'

    train_Data=MyDataset(Train_data_Path)
    # test_Data=MyDataset(Test_data_Path)
    f,img,gt = train_Data.__getitem__(0)
    plt.subplot(1,2,1)
    plt.imshow(img.cpu().permute(1,2,0))
    plt.subplot(1,2,2)
    plt.imshow(gt.cpu() * 255)
    plt.show()


# train_loader = DataLoader(train_Data,batch_size=2,shuffle=True)
# test_loader = DataLoader(test_Data,batch_size=2,shuffle=True)
# train_iter=iter(train_loader)
# # print(len(train_iter))
# print(len(train_loader))
# f, img,gt = train_iter.next()
# # print(type(img),' ',type(gt))
# # img =np.array(img)
# # print(img.shape)
# # img= np.transpose(img,(0,2,3,1))
# # gt = np.array(gt)
# # fig = plt.figure(figsize=(100,100))
# # sub1=fig.add_subplot(2,2,1)
# # sub1.imshow(img[2])
# # sub2=fig.add_subplot(2,2,2)
# # sub2.imshow(gt[2])

# # plt.show()
# for img,msk in train_loader:
#     print(type(msk))
#     msk=msk.unsqueeze(dim=1)
#     img_grid = torchvision.utils.make_grid(img)
#     msk_grid = torchvision.utils.make_grid(msk)
#     # plt.imshow(grid.numpy().transpose(1,2,0))
#     fig = plt.figure(figsize=(100,100))
#     sub1=fig.add_subplot(2,2,1)
#     sub1.imshow(img_grid.cpu().numpy().transpose(1,2,0))
#     sub2=fig.add_subplot(2,2,2)
#     sub2.imshow(msk_grid.numpy().transpose(1,2,0))

#     plt.show()