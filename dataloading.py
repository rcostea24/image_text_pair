import cv2
import pandas as pd
import torch
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import re
from collections import Counter

STOP_WORDS_EN = set(stopwords.words('english'))
STEMMER = SnowballStemmer('romanian')
MAX_LEN = 32
WORD_TO_INDEX = None

# class VocabSize():
#     def __init__(self, vocab_size):
#         self.vocab_size = vocab_size

class VocabSize():
    vocab_size = None

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
    vocab = Counter(word for sent in tokenized_captions for word in sent)
    
    word_to_index = {}
    word_to_index["<pad>"] = 0 
    word_to_index["<unk>"] = 1

    for id, (word, _) in enumerate(vocab.items()):
        word_to_index[word] = id+2
    
    return word_to_index

class ImagePairDataset(Dataset):
    def __init__(self, root, split, img_transforms=None):
        global WORD_TO_INDEX
        csv_file = pd.read_csv(os.path.join(root, f"{split}.csv"))
        self.split = split

        image_ids = csv_file["image_id"].to_numpy()
        captions = csv_file["caption"].to_numpy()
        
        tokenized_captions = []
        for caption in captions:
            tokenized_captions.append(get_tokens(caption))

        if split == "train":
            WORD_TO_INDEX = build_vocab(tokenized_captions)
            VocabSize.vocab_size = len(WORD_TO_INDEX)
        
        encoded_captions = []
        for tokenized_caption in tokenized_captions:
            encoded_caption = []
            for token in tokenized_caption:
                if token in WORD_TO_INDEX:
                    encoded_caption.append(WORD_TO_INDEX[token])
                else:
                    encoded_caption.append(WORD_TO_INDEX["<unk>"])
            encoded_captions.append(encoded_caption)
        
        encoded_captions_padded = []
        for encoded_caption in encoded_captions:
            num_pads = MAX_LEN - len(encoded_caption)
            encoded_captions_padded.append(encoded_caption[:MAX_LEN] + [0]*num_pads)

        if split != "test":
            self.labels = csv_file["label"].to_numpy()
        self.img_transforms = img_transforms
        
        images_path = os.path.join(root, f"{split}_images")
        self.data = []
        for img_id, encoded_caption_padded in zip(image_ids, encoded_captions_padded):
            img = cv2.imread(os.path.join(images_path, f"{img_id}.jpg"))
            self.data.append([img, encoded_caption_padded])

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        sample_img = self.data[idx][0]
        sample_txt = self.data[idx][1]

        sample_txt = torch.tensor(sample_txt).int()
        
        label = self.labels[idx] if self.split != "test" else -1

        if self.img_transforms:
            sample_img = self.img_transforms(sample_img)

        return sample_img, sample_txt, label
    
def load_data(cfg):
    img_transforms = T.Compose([
        T.ToTensor(),
        T.Resize((224, 244))
    ])

    train_dataset = ImagePairDataset(cfg["data_root_path"], "train", img_transforms)
    val_dataset = ImagePairDataset(cfg["data_root_path"], "val", img_transforms)
    test_dataset = ImagePairDataset(cfg["data_root_path"], "test", img_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader