#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
from natsort import natsorted

class reg_data(Dataset):
    def __init__(self,image_dir,gt_dir,transform=None):
        self.image_dir=image_dir
        self.gt_dir=gt_dir
        self.transform=transform
        self.images=natsorted(os.listdir(image_dir))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,index):
        img_path=os.path.join(self.image_dir,self.images[index])
        gt_path=os.path.join(self.gt_dir,self.images[index].replace("slim10","slim40"))
        image=np.array(Image.open(img_path))
        gt=np.array(Image.open(gt_path))
        mina=-2
        maxa=2
        image=(image-mina)/(maxa-mina)
        gt=(gt-mina)/(maxa-mina)
        gt=-1+2*gt

        if self.transform is not None:
            image= self.transform(image)
            gt=self.transform(gt)
            image=torch.cat((image,image,image),axis=0)
            gt=torch.cat((gt,gt,gt),axis=0)
            
        return image,gt
            
