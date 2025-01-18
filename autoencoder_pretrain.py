import numpy as np
import cv2
import pandas as pd
import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

class ImageRotation(Dataset):
    def __init__(self, root, split, img_transforms):

        # read data
        csv_file = pd.read_csv(os.path.join(root, f"{split}.csv"))

        image_ids = csv_file["image_id"].to_numpy()
        self.img_transforms = img_transforms
        
        # read images
        images_path = os.path.join(root, f"{split}_images")
        self.data = []
        for img_id in image_ids:
            img = cv2.imread(os.path.join(images_path, f"{img_id}.jpg"))
            self.data.append(img)

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # get image and apply transform
        sample_img = self.data[idx]
        sample_img = self.img_transforms(sample_img)

        return sample_img
    
class Downsample(nn.Module):
    # downsample class
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()
            
        # conv layer with maxpool before
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(self.maxpool(x))

class Upsample(nn.Module):
    # upsample class
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        
        # upsample layer with factor of 2
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        # conv layer
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        

        # dropout
        self.use_dropout = dropout            
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        out = self.conv(self.up(x))
        return self.dropout(out) if self.use_dropout else out

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # init conv
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        # downsample layers
        self.down1 = Downsample(in_channels=16, out_channels=32)
        self.down2 = Downsample(in_channels=32, out_channels=32)
        self.down3 = Downsample(in_channels=32, out_channels=64)
        # upsample layers
        self.up1 = Upsample(in_channels=64, out_channels=32, dropout=True)
        self.up2 = Upsample(in_channels=32, out_channels=32, dropout=True)
        self.up3 = Upsample(in_channels=32, out_channels=16, dropout=True)
        # last conv
        self.last_conv = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        # forward pass

        # init conv
        x = self.init_conv(x)
        # downsample the input
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        # save downsampled features
        features = x
        
        # reconstruct the image with upsample layers
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        
        # last conv
        x = self.last_conv(x)
        return x, features
        
batch_size = 64
lr = 2e-4
epochs = 50
device = "cuda" if torch.cuda.is_available() else "cpu"

root = "/kaggle/input/image-sentence-pair-v2"

img_transforms = T.Compose([
    T.ToTensor(),
    T.Resize((96, 96))
])

train_dataset = ImageRotation(root, "train", img_transforms)
val_dataset = ImageRotation(root, "val", img_transforms)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)     

img = next(iter(train_dataloader))
model = Autoencoder()
y = model(img)

def train_step(model, train_loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0

    for img_inputs in tqdm(train_loader):
        img_inputs = img_inputs.to(device)
        
        optimizer.zero_grad()
        
        outputs, _ = model(img_inputs)
        loss = loss_fn(outputs, img_inputs)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_step_loss = total_loss / len(train_loader)
    return train_step_loss

def val_step(model, val_loader, optimizer, loss_fn):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for img_inputs in tqdm(val_loader):
            img_inputs = img_inputs.to(device)

            outputs, _ = model(img_inputs)
            loss = loss_fn(outputs, img_inputs)
    
            total_loss += loss.item()

    val_step_loss = total_loss / len(val_loader)

    return val_step_loss

def visualize(model, val_dataloader):
    model.eval()

    x = next(iter(val_dataloader))
    x = x.to(device)
    
    with torch.no_grad():
        y, _ = model(x)
    
    # put them on cpu
    x = x.cpu()
    y = y.detach().cpu()
    
    # denormalize
    x = x * 0.5 + 0.5
    y = y * 0.5 + 0.5
    
    # plot images
    fig = plt.figure()
    
    fig.add_subplot(1, 2, 1) 

    plt.imshow(x[0].permute(1, 2, 0)) 
    plt.axis('off') 
    plt.title("x") 

    fig.add_subplot(1, 2, 2) 

    plt.imshow(y[0].permute(1, 2, 0)) 
    plt.axis('off') 
    plt.title("y")
        
    plt.show()

def train():
    model = Autoencoder().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    best_loss = 100.0
    best_model = None

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(epochs):
        print(f"epoch: {epoch+1}")
        train_step_loss = train_step(model, train_dataloader, optimizer, loss_fn)
        print(f"train_step_loss: {train_step_loss}")
        
        val_step_loss= val_step(model, val_dataloader, optimizer, loss_fn)
        print(f"val_step_loss: {val_step_loss}")

        train_losses.append(train_step_loss)
        val_losses.append(val_step_loss)

        if val_step_loss < best_loss:
            best_loss = val_step_loss
            torch.save(model.state_dict(), "best_autoencoder.pt")
            visualize(model, val_dataloader)
            

    plt.plot(range(epochs), train_losses)
    plt.title("Train loss")
    plt.savefig("train_loss.jpg")
    plt.plot()
    
    plt.plot(range(epochs), val_losses)
    plt.title("Val loss")
    plt.savefig("val_loss.jpg")
    plt.plot()

    train()
