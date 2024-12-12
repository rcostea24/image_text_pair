import argparse
import os
import json
import torch

from logger import Logger
from trainer import Trainer
from dataloading import VocabSize, load_data

EXPERIMENTS_ROOT = r"experiments"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root_path", default="/kaggle/input/image-sentence-pair-matching")
    args = parser.parse_args()

    exp_cfgs = sorted(os.listdir(EXPERIMENTS_ROOT))
    print(exp_cfgs)
    for cfg_file_name in exp_cfgs:
        cfg_path = os.path.join(EXPERIMENTS_ROOT, cfg_file_name)
    
        with open(cfg_path, "r") as file:
            cfg = json.load(file)
            
        cfg["exp_id"] = cfg_path.split("_")[-1].split(".")[0]
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        cfg["data_root_path"] = args.data_root_path

        train_dataloader, val_dataloader, test_dataloader = load_data(cfg)

        vision_params = cfg["vision_params"]

        language_params = cfg["language_params"]
        language_params["vocab_size"] = VocabSize.vocab_size

        classifier_params = cfg["classifier_params"]

        if not os.path.exists("logs"):
            os.makedirs("logs")

        logger = Logger(f"logs/log_{cfg['exp_id']}.txt")

        logger.log(f"{'-'*50} Parameters {'-'*50}")
        for key, value in cfg.items():
            logger.log(f"{key}: {value}")
        logger.log(f"{'-'*50}------------{'-'*50}")

        trainer = Trainer(cfg, logger, train_dataloader, val_dataloader, test_dataloader)
        trainer.train()
        trainer.test_step()
