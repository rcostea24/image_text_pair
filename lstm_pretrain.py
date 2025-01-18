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
from tqdm.notebook import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import copy

STOP_WORDS_EN = set(stopwords.words('english'))
STEMMER = SnowballStemmer('romanian')
MAX_LEN = 32

WORD_TO_INDEX = {}
WORD_TO_INDEX["<mask>"] = 0 
WORD_TO_INDEX["<pad>"] = 1
WORD_TO_INDEX["<unk>"] = 2

def get_tokens(caption):
    caption = caption.lower()
    
    caption = re.sub(r'[^\w\s]', ' ', caption)
    caption = re.sub(r'\s+', ' ', caption)
    caption = word_tokenize(caption)
    
    tokens = []
    for word in caption:
        if word not in STOP_WORDS_EN:
            stemmed_word = STEMMER.stem(word)
            tokens.append(stemmed_word)
    
    return tokens

def build_vocab(tokenized_captions):
    global WORD_TO_INDEX
    vocab = Counter(word for sent in tokenized_captions for word in sent)
    start_id = list(WORD_TO_INDEX.values())[-1]
    increment = 1
    for id, (word, _) in enumerate(vocab.items()):
        if word not in WORD_TO_INDEX:
            WORD_TO_INDEX[word] = start_id + increment
            increment += 1

class CaptionsDataset(Dataset):
    def __init__(self, root, split):
        global WORD_TO_INDEX
        csv_file = pd.read_csv(os.path.join(root, f"{split}.csv"))
        self.split = split

        captions = csv_file["caption"].to_numpy()
        
        tokenized_captions = []
        for caption in captions:
            tokenized_captions.append(get_tokens(caption))

        build_vocab(tokenized_captions)
        
        encoded_captions = []
        masked_captions = []
        self.labels = []
        for tokenized_caption in tokenized_captions:
            encoded_caption = []
            for token in tokenized_caption:
                if token in WORD_TO_INDEX:
                    encoded_caption.append(WORD_TO_INDEX[token])
                else:
                    encoded_caption.append(WORD_TO_INDEX["<unk>"])

            encoded_captions.append(encoded_caption)

            mask_idx = np.random.randint(len(encoded_caption))
            masked_caption = copy.deepcopy(encoded_caption)
            masked_caption[mask_idx] = WORD_TO_INDEX["<mask>"]

            self.labels.append(encoded_caption[mask_idx])
            masked_captions.append(masked_caption)
        
        encoded_captions_padded = []
        masked_captions_padded = []
        for encoded_caption, masked_caption in zip(encoded_captions, masked_captions):
            num_pads = MAX_LEN - len(encoded_caption)
            encoded_captions_padded.append(encoded_caption[:MAX_LEN] + [WORD_TO_INDEX["<pad>"]]*num_pads)
            masked_captions_padded.append(masked_caption[:MAX_LEN] + [WORD_TO_INDEX["<pad>"]]*num_pads)

        self.data = [encoded_captions_padded, masked_captions_padded]

    def __len__(self):
        return len(self.data[0])
        
    def __getitem__(self, idx):
        sample = self.data[0][idx]
        sample = torch.tensor(sample).int()

        masked_sample = self.data[1][idx]
        masked_sample = torch.tensor(masked_sample).int()

        label = self.labels[idx]
        
        return masked_sample, label
    
class LanguageModel(nn.Module):
    def __init__(self, params):
        super(LanguageModel, self).__init__()
        if len(params) > 0:
            vocab_size = params["vocab_size"]
            embed_dim = params["embed_dim"]
            hidden_dim = params["hidden_dim"]
            num_layers = params["num_layers"]
            
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden_state, _) = self.lstm(x)
        out = self.linear(hidden_state[-1])
        return out
    
batch_size = 64
lr = 1e-3
epochs = 50
device = "cuda" if torch.cuda.is_available() else "cpu"

root = "/kaggle/input/image-sentence-pair-v2"

train_dataset = CaptionsDataset(root, "train")
val_dataset = CaptionsDataset(root, "val")
test_dataset = CaptionsDataset(root, "test")

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

params = {
    "vocab_size": len(WORD_TO_INDEX), 
    "embed_dim": 256,
    "hidden_dim": 512,
    "num_layers": 1
}

def train_step(model, train_loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0
    correct_preds = 0.0
    total_preds = 0

    for txt_inputs, labels in tqdm(train_loader):
        txt_inputs, labels = txt_inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        
        outputs = model(txt_inputs)
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

def val_step(model, val_loader, loss_fn):
    model.eval()
    total_loss = 0.0
    correct_preds = 0.0
    total_preds = 0

    with torch.no_grad():
        for txt_inputs, labels in tqdm(val_loader):
            txt_inputs, labels = txt_inputs.to(device), labels.to(device)

            outputs = model(txt_inputs)
            loss = loss_fn(outputs, labels)
    
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            correct_preds += torch.sum(predictions == labels).item()
            total_preds += labels.shape[0]

    val_step_loss = total_loss / len(val_loader)
    val_step_acc = correct_preds / total_preds

    return val_step_loss, val_step_acc

def train():
    model = LanguageModel(params).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(epochs):
        print(f"epoch: {epoch+1}")
        train_step_loss, train_step_acc = train_step(model, train_dataloader, optimizer, loss_fn)
        print(f"train_step_loss: {train_step_loss} | train_step_acc = {train_step_acc}")
        
        val_step_loss, val_step_acc = val_step(model, val_dataloader, loss_fn)
        print(f"val_step_loss: {val_step_loss} | val_step_acc = {val_step_acc}")

        train_losses.append(train_step_loss)
        train_accs.append(train_step_acc)
        val_losses.append(val_step_loss)
        val_accs.append(val_step_acc)

        if val_step_acc > best_acc:
            best_acc = val_step_acc
            torch.save(model.state_dict(), "language_model.pt")

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