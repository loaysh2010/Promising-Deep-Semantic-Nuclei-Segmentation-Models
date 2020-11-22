import os
import numpy as np
from sklearn.metrics import confusion_matrix,accuracy_score


def IoU_for_2_matrices(target,predict,num_classes): #Intersection over Union

    cm=confusion_matrix(target,predict)
    av_iou=0
    for i in range(num_classes):
        iou=cm[i,i]/( np.sum(cm[i,:]) + np.sum(cm[:,i])  -cm[i,i]+1e-06 ) 
        print('iou',i,':',iou)
        av_iou=av_iou+iou
 
    precision= cm[1,1]/(cm[1,1]+cm[0,1])
    recall= cm[1,1]/(cm[1,1]+cm[1,0])
    F1= (2*recall*precision)/(precision+recall)
    print('precision=',precision)
    print('recall=',recall)
    print('F1=',F1)

    print('****************************')
    
    return av_iou/num_classes,cm

def IoU_for_2lists_matrices(target,predict,num_classes):
    
    print('Started Computeing IoU:')
    
    target=np.array(target)
    predict=np.array(predict)

    target=target.reshape((-1,))
    predict=predict.reshape((-1,))

    av_iou,_=IoU_for_2_matrices(target,predict,num_classes)
    return av_iou

def Acc_for_2lists_matrices(target,predict):

    print('Started Computeing Acc:')

    target=np.array(target)
    predict=np.array(predict)
   

    target=target.reshape((-1,))
    predict=predict.reshape((-1,))

    acc=accuracy_score(target,predict)    
    return acc

