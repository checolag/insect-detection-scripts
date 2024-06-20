import argparse # Argument parsing module
import yaml  # YAML (YAML Ain't Markup Language) for configuration files
from pathlib import Path  # Path object from pathlib for working with file paths
import logging
import utils

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import gc
import json
from tqdm.notebook import tqdm
import random
from PIL import Image

import torch, detectron2
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

from detectron2.utils.logger import setup_logger
setup_logger()
print(f"Torch version: {torch.__version__}", "\n", f"CUDA availability: {torch.cuda.is_available()}")
print(f"Setup logger of detectron2: {setup_logger()}")

def main(config_path):
    # training_val_config.yaml
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    experiments_path = Path(config["experiments_path"])
    print(f"K-fold path is: {experiments_path}")

    logging.basicConfig(level=logging.INFO)
    experiment_path = experiments_path / f"{config['experiment_name']}"
    logging.info("Training...")
    yamls = list(experiment_path.glob("split*/*.yaml"))
    yamls = sorted([str(i) for i in yamls], key=utils.mixedsort)
    # print(yamls)
    
    for i in range(len(yamls)):
        logging.info(f"Training for yaml file: {yamls[i]}")
        split_path = experiment_path / f"split_{i+1}"
        print(split_path)
        print(f"Split number: {i+1}")
        cfg = utils.detectron2_setup(split_path, config)
        utils.detectron2_train(cfg, config)
        logging.info(f"END TRAINING {i+1}")
    
    logging.info("Script finished.")
    utils.stop_instance()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your script description here.")
    parser.add_argument("config", type=str, help="Path to the configuration file.")

    args = parser.parse_args()

    main(args.config)