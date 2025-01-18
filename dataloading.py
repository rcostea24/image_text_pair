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

# constants
STOP_WORDS_EN = set(stopwords.words('english'))
STEMMER = SnowballStemmer('romanian')
MAX_LEN = 32

WORD_TO_INDEX = {}
WORD_TO_INDEX["<mask>"] = 0 
WORD_TO_INDEX["<pad>"] = 1
WORD_TO_INDEX["<unk>"] = 2

class VocabSize():
    # class used in main file to get the vocabulary size needed for language model
    vocab_size = None

def get_tokens(caption):
    # function for extracting the words from caption
    caption = caption.lower()
    
    # remove punctuation and extra spaces
    caption = re.sub(r'[^\w\s]', ' ', caption)
    caption = re.sub(r'\s+', ' ', caption)

    # get the words
    caption = word_tokenize(caption)
    
    # apply stemming on each word
    tokens = []
    for word in caption:
        if word not in STOP_WORDS_EN:
            stemmed_word = STEMMER.stem(word)
            tokens.append(stemmed_word)
    
    return tokens

def build_vocab(tokenized_captions):
    # function to build the vocabulary
    global WORD_TO_INDEX

    # counter object with each unique word and it's frequency
    vocab = Counter(word for sent in tokenized_captions for word in sent)

    # starting index
    start_id = list(WORD_TO_INDEX.values())[-1]
    increment = 1

    # iterate words and create vocab dictionary
    for id, (word, _) in enumerate(vocab.items()):
        if word not in WORD_TO_INDEX:
            WORD_TO_INDEX[word] = start_id + increment
            increment += 1
    

class ImagePairDataset(Dataset):
    def __init__(self, root, split, img_transforms=None):
        # use global constant
        global WORD_TO_INDEX

        # read data
        csv_file = pd.read_csv(os.path.join(root, f"{split}.csv"))
        self.split = split

        image_ids = csv_file["image_id"].to_numpy()
        captions = csv_file["caption"].to_numpy()
        
        # get tokens from caption
        tokenized_captions = []
        for caption in captions:
            tokenized_captions.append(get_tokens(caption))

        # build / extend the vocab
        build_vocab(tokenized_captions)
        VocabSize.vocab_size = len(WORD_TO_INDEX)
        
        # encode each token with it's specific index from vocabulary
        encoded_captions = []
        for tokenized_caption in tokenized_captions:
            encoded_caption = []
            for token in tokenized_caption:
                if token in WORD_TO_INDEX:
                    encoded_caption.append(WORD_TO_INDEX[token])
                else:
                    encoded_caption.append(WORD_TO_INDEX["<unk>"])
            encoded_captions.append(encoded_caption)
        
        # pad the tokens such that all of them have the same length
        encoded_captions_padded = []
        for encoded_caption in encoded_captions:
            num_pads = MAX_LEN - len(encoded_caption)
            encoded_captions_padded.append(encoded_caption[:MAX_LEN] + [0]*num_pads)

        # get labels if the split is different than test
        if split != "test":
            self.labels = csv_file["label"].to_numpy()
        self.img_transforms = img_transforms
        
        # read images and pair with the captions
        images_path = os.path.join(root, f"{split}_images")
        self.data = []
        for img_id, encoded_caption_padded in zip(image_ids, encoded_captions_padded):
            img = cv2.imread(os.path.join(images_path, f"{img_id}.jpg"))
            self.data.append([img, encoded_caption_padded])

    def __len__(self):
        # len method
        return len(self.data)
        
    def __getitem__(self, idx):
        # get the image and caption
        sample_img = self.data[idx][0]
        sample_txt = self.data[idx][1]

        # make the caption a tensor
        sample_txt = torch.tensor(sample_txt).int()
        
        # get the label
        label = self.labels[idx] if self.split != "test" else -1

        # apply transform to image
        if self.img_transforms:
            sample_img = self.img_transforms(sample_img)

        return sample_img, sample_txt, label
    
def load_data(cfg):
    # init image transforms
    img_transforms = T.Compose([
        T.ToTensor(),
        T.Resize(cfg["vision_params"]["img_size"])
    ])

    # create each dataset
    train_dataset = ImagePairDataset(cfg["data_root_path"], "train", img_transforms)
    val_dataset = ImagePairDataset(cfg["data_root_path"], "val", img_transforms)
    test_dataset = ImagePairDataset(cfg["data_root_path"], "test", img_transforms)

    # create each dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader