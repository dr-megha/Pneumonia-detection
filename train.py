
#from fastai.vision import *
#from fastai.callbacks.hooks import *
#from fastai.utils.mem import *
#from fastai import *
import os
import numpy as np
from pathlib import Path
#import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets,transforms
from PIL import Image
import torch 
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,6,3,1)
        self.conv2=nn.Conv2d(6,16,3,1) 
        self.fc1=nn.Linear(16*254*254, 120)
        self.fc2=nn.Linear(120,60)
        self.fc3=nn.Linear(60,2)
        
    def forward(self,x):
        #pdb.set_trace()
        x=F.relu(self.conv1(x))
        x=F.max_pool2d(x,2,2)
        x=F.relu(self.conv2(x))
        x=F.max_pool2d(x,2,2)
        x=x.reshape(-1,16*254*254)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return torch.sigmoid(x)



class CNN_Dataset(torch.utils.data.Dataset):
      
    def __init__(self,features,labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, index):
        img_path = self.features[index]
        img = Image.open(img_path)
        trans=transforms.Compose([transforms.Resize(1024), 
                                  transforms.CenterCrop(1023), 
                                  transforms.Grayscale(num_output_channels=1), 
                                  transforms.ToTensor()])
        X=trans(img)
        y = torch.tensor([self.labels[index]])
        return X,y



def load_data(path):
    imgs=[]
    labels=[]
    for item in os.listdir(path):
        for items in os.listdir(path/item):
            imgs.append(path/item/items)
            labels.append(item)
    return imgs, labels        
           
    
def map_labels(labels):
    lable_map={'NORMAL':0,'PNEUMONIA':1}
    labels = [lable_map[item] for item in labels]
    return labels


    

    

if __name__=="__main__":

    path = Path("/data/analytics/naveen.bansal/Jeremy/data/chest_xray/")
    test_data_path = path/'test'
    train_data_path = path/'train'
    val_data_path = path/'val'
    
    
    train_imgs,train_labels = load_data(train_data_path)
    test_imgs,test_labels = load_data(test_data_path)
    val_imgs,val_labels = load_data(val_data_path)
    
    train_labels = map_labels(train_labels)
    test_labels = map_labels(test_labels)
    val_labels = map_labels(val_labels)
    
    
    train_dataset = CNN_Dataset(train_imgs,train_labels)
    val_dataset= CNN_Dataset(val_imgs,val_labels)
    test_dataset=CNN_Dataset(test_imgs, test_labels)
    
    
    training_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 16,shuffle=True)
    Validation_dataloader=torch.utils.data.DataLoader(val_dataset, batch_size = 4,shuffle=True)
    test_dataloader= torch.utils.data.DataLoader(test_dataset, batch_size = 5,shuffle=True)
    
    net=ConvolutionalNetwork()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(),lr = 0.001)
    
    if torch.cuda.is_available():
        net = net.cuda()
        
    epochs=50
    best_val_loss=100000
    

    for i in range(epochs):

        net.train() 
        train_losses=[]
        for batch in training_dataloader:
            x = batch[0]
            y_true  = batch[1]

            if torch.cuda.is_available():
                x = x.cuda()
                y_true  = y_true.cuda()

            optimizer.zero_grad()
            y_pred=net.forward(x)
            train_loss = criterion(y_pred,y_true.squeeze())
            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.item())


        #Validation phase
        net.eval()
        val_losses=[]

        for batch in Validation_dataloader:
            x=batch[0]
            y_true=batch[1]

            if torch.cuda.is_available():
                x = x.cuda()
                y_true  = y_true.cuda()

            y_pred=net.forward(x)
            vali_loss = criterion(y_pred,y_true.squeeze())
            val_losses.append(vali_loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses) 
        if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_name = f"checkpoints_xrays/train_loss_{train_loss}_val_loss_{val_loss}.pt"
                torch.save(net.state_dict(),model_name)

        print (f"Epoch_number: {i}, Train loss: {train_loss}, Val Loss: {val_loss}") 
        
        
        