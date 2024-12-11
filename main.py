import os
from torch.utils.data import DataLoader
import torchvision.transforms as T
import json
import torch

from dataloading import ImagePairDataset
from logger import Logger
from trainer import Trainer
from dataloading import VocabSize

DATA_ROOT_PATH = "/kaggle/input/image-sentence-pair-matching"
EXPERIMENTS_ROOT = r"experiments"

def load_data(batch_size):
    img_transforms = T.Compose([
        T.ToTensor(),
        T.Resize((224, 244))
    ])

    train_dataset = ImagePairDataset(DATA_ROOT_PATH, "train", img_transforms)
    val_dataset = ImagePairDataset(DATA_ROOT_PATH, "val", img_transforms)
    test_dataset = ImagePairDataset(DATA_ROOT_PATH, "test", img_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader

if __name__ == "__main__":
    exp_cfgs = sorted(os.listdir(EXPERIMENTS_ROOT))
    print(exp_cfgs)
    for cfg_file_name in exp_cfgs:
        cfg_path = os.path.join(EXPERIMENTS_ROOT, cfg_file_name)
    
        with open(cfg_path, "r") as file:
            cfg = json.load(file)
            
        cfg["exp_id"] = cfg_path.split("_")[-1].split(".")[0]
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        cfg["data_root_path"] = DATA_ROOT_PATH

        train_dataloader, val_dataloader, test_dataloader = load_data(cfg["batch_size"])

        vision_params = cfg["vision_params"]

        language_params = cfg["language_params"]
        language_params["vocab_size"] = VocabSize.vocab_size

        classifier_params = cfg["classifier_params"]

        logger = Logger(f"logs/log_{cfg['exp_id']}.txt")

        logger.log(f"{'-'*50} Parameters {'-'*50}")
        for key, value in cfg.items():
            logger.log(f"{key}: {value}")
        logger.log(f"{'-'*50}------------{'-'*50}")

        trainer = Trainer(cfg, logger, train_dataloader, val_dataloader, test_dataloader)
        trainer.train()
        trainer.test_step()
