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
    def __init__(self, root, split, img_transforms=None):
        csv_file = pd.read_csv(os.path.join(root, f"{split}.csv"))

        image_ids = csv_file["image_id"].to_numpy()
        self.labels = np.array([-60, -30, 0, 30, 60] * (len(image_ids) // 5))
        self.labels = LabelEncoder().fit_transform(self.labels)
        self.img_transforms = img_transforms
        
        images_path = os.path.join(root, f"{split}_images")
        self.data = []
        for img_id in image_ids:
            img = cv2.imread(os.path.join(images_path, f"{img_id}.jpg"))
            self.data.append(img)

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        sample_img = self.data[idx]
        label = self.labels[idx]

        if self.img_transforms:
            sample_img = self.img_transforms(sample_img)

        sample_img = T.functional.rotate(sample_img, int(label))

        return sample_img, label
    
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super(CNNBlock, self).__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
        
        self.residual_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        

    def forward(self, x):
        residual = self.residual_conv(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        
        return out
        
class VisionModel(nn.Module):
    # Vision model inspired by ResNet18
    def __init__(self, vision_params={}):
        super(VisionModel, self).__init__()

        if len(vision_params) > 0:
            self.predict_rotation = vision_params["predict_rotation"]
        
        self.first_layer = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = nn.Sequential(
            CNNBlock(64, 64, False),
            CNNBlock(64, 64, False)
        )
        self.layer2 = nn.Sequential(
            CNNBlock(64, 128, False),
            CNNBlock(128, 128, True)
        )
        self.layer3 = nn.Sequential(
            CNNBlock(128, 256, False),
            CNNBlock(256, 256, True)
        )
        self.layer4 = nn.Sequential(
            CNNBlock(256, 512, False),
            CNNBlock(512, 512, True)
        )

        self.avg_pool = nn.AvgPool2d(7)

        self.flatten = nn.Flatten()

        if self.predict_rotation:
            self.rotation_classifier = nn.Sequential(
                nn.Linear(512, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 5)
            )

    def forward(self, x):
        # print(x.shape)
        x = self.first_layer(x)
        x = self.maxpool(x)
        # print(x.shape)

        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)

        x = self.avg_pool(x)
        # print(x.shape)
        x = self.flatten(x)

        if self.predict_rotation:
            out = self.rotation_classifier(x)
            return out
        else:
            return x
        
batch_size = 64
lr = 1e-3
epochs = 50
device = "cuda" if torch.cuda.is_available() else "cpu"

root = "/kaggle/input/image-sentence-pair-matching"

img_transforms = T.Compose([
    T.ToTensor(),
    T.Resize((224, 244))
])

train_dataset = ImageRotation(root, "train", img_transforms)
val_dataset = ImageRotation(root, "val", img_transforms)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

vision_params = {
    "predict_rotation": True
}

def train_step(model, train_loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0
    correct_preds = 0.0
    total_preds = 0

    for img_inputs, labels in tqdm(train_loader):
        img_inputs, labels = img_inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        
        outputs = model(img_inputs)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = torch.argmax(outputs, dim=1)
        correct_preds += torch.sum(predictions == labels).item()
        total_preds += labels.shape[0]

    train_step_loss = total_loss / len(train_loader)
    train_step_acc = correct_preds / total_preds

    return train_step_loss, train_step_acc

def val_step(model, val_loader, optimizer, loss_fn):
    model.eval()
    total_loss = 0.0
    correct_preds = 0.0
    total_preds = 0

    with torch.no_grad():
        for img_inputs, labels in tqdm(val_loader):
            img_inputs, labels = img_inputs.to(device), labels.to(device)

            outputs = model(img_inputs)
            loss = loss_fn(outputs, labels)
    
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            correct_preds += torch.sum(predictions == labels).item()
            total_preds += labels.shape[0]

    val_step_loss = total_loss / len(val_loader)
    val_step_acc = correct_preds / total_preds

    return val_step_loss, val_step_acc

def train():
    model = VisionModel(vision_params).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_model = None

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(epochs):
        print(f"epoch: {epoch+1}")
        train_step_loss, train_step_acc = train_step(model, train_dataloader, optimizer, loss_fn)
        print(f"train_step_loss: {train_step_loss} | train_step_acc = {train_step_acc}")
        
        val_step_loss, val_step_acc = val_step(model, val_dataloader, optimizer, loss_fn)
        print(f"val_step_loss: {val_step_loss} | val_step_acc = {val_step_acc}")

        train_losses.append(train_step_loss)
        train_accs.append(train_step_acc)
        val_losses.append(val_step_loss)
        val_accs.append(val_step_acc)

        if val_step_acc > best_acc:
            best_acc = val_step_acc
            best_model = model

        torch.save(best_model.state_dict(), "best_vision_model.pt")

    plt.plot(range(epochs), train_losses)
    plt.title("Train loss")
    plt.savefig("train_loss.jpg")
    plt.plot()
    
    plt.plot(range(epochs), train_accs)
    plt.title("Train acc")
    plt.savefig("train_acc.jpg")
    plt.plot()
    
    plt.plot(range(epochs), val_losses)
    plt.title("Val loss")
    plt.savefig("val_loss.jpg")
    plt.plot()
    
    plt.plot(range(epochs), val_accs)
    plt.title("Val acc")
    plt.savefig("val_acc.jpg")
    plt.plot()

train()