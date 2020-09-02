
import os
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix

from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets,transforms
from PIL import Image
import torch 
import torch.nn as nn
import torch.nn.functional as F
import random
from sklearn.model_selection import train_test_split
import pdb
import joblib


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
        self.features = list(features)
        self.labels = list(labels)
    
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

    path = Path("data")
    test_data_path = path/'test'
    train_data_path = path/'train'
    val_data_path = path/'val'
    
    
    train_imgs,train_labels = load_data(train_data_path)
    test_imgs,test_labels = load_data(test_data_path)
    val_imgs,val_labels = load_data(val_data_path)
    
    train_labels = map_labels(train_labels)
    test_labels = map_labels(test_labels)
    val_labels = map_labels(val_labels)

    # the dataset is highly unbalanced
    # to balance this data, we are mixing all the train and test and then
    # redividing after shuffling
    
    train_imgs = train_imgs + test_imgs + val_imgs
    train_labels = train_labels + test_labels + val_labels
    
    # selecting equal number of pnumonia and normal images
    # number of samples 1583 as the number of normal images are 1583.
    
    data = list(zip(train_imgs,train_labels))
    random.shuffle(data)
    
    count_normal=0
    count_pnemonia=0

    train_data=[]
    for item in data:
        
        if item[1] == 0 and count_normal < 1583:
            count_normal+=1
            train_data.append(item)
            
        if item[1] == 1 and count_pnemonia < 1583:
            count_pnemonia+=1
            train_data.append(item)
            
    df = pd.DataFrame(train_data,columns=('img_path','label'))    
    
    train_imgs,test_imgs, train_labels,test_labels = train_test_split(df['img_path'],
                                                                      df['label'], 
                                                                      test_size=0.30, stratify = df['label'], 
                                                                      random_state=42)
    
    val_imgs, test_imgs,val_labels, test_labels = train_test_split(test_imgs,
                                                                      test_labels, 
                                                                      test_size=0.30, stratify = test_labels, 
                                                                      random_state=42)
    
    
    #pdb.set_trace()
    
    train_dataset = CNN_Dataset(train_imgs,train_labels)
    val_dataset= CNN_Dataset(val_imgs,val_labels)
    test_dataset=CNN_Dataset(test_imgs, test_labels)
    
    # dumping test_dataset for future model evaluation
    joblib.dump(test_dataset,'data/test_dataset.pkl')
    
    
    training_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 16,shuffle=True)
    Validation_dataloader=torch.utils.data.DataLoader(val_dataset, batch_size = 4,shuffle=True)
    test_dataloader= torch.utils.data.DataLoader(test_dataset, batch_size = 5,shuffle=True)
    
    net=ConvolutionalNetwork()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(),lr = 0.001)
    
    if torch.cuda.is_available():
        net = net.cuda()
        
    epochs=100
    best_val_loss=100000
    

    for i in range(epochs):

        net.train() 
        train_losses=[]
        for batch in training_dataloader:
            x = batch[0]
            y_true  = batch[1]
            
            #print (f"Train:{x.shape},{y_true.shape}")
            if torch.cuda.is_available():
                x = x.cuda()
                y_true  = y_true.cuda()

            optimizer.zero_grad()
            y_pred=net.forward(x)
      
            train_loss = criterion(y_pred,y_true.flatten())
          
            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.item())


        #Validation phase
        net.eval()
        val_losses=[]

        for batch in Validation_dataloader:
            x=batch[0]
            y_true=batch[1]
            #print (f"Val:{x.shape},{y_true.shape}")
            if torch.cuda.is_available():
                x = x.cuda()
                y_true  = y_true.cuda()

            y_pred=net.forward(x)
            vali_loss = criterion(y_pred,y_true.flatten())
            val_losses.append(vali_loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses) 
        if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_name = f"model/train_loss_{train_loss}_val_loss_{val_loss}.pt"
                torch.save(net.state_dict(),model_name)

        print (f"Epoch_number: {i}, Train loss: {train_loss}, Val Loss: {val_loss}") 
        
        
        