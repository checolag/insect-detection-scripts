import os  # Operating System related functions
import shutil  # File and directory manipulation functions
import cv2  # OpenCV library for computer vision tasks
import numpy as np  # NumPy for numerical operations
from pathlib import Path  # Path object from pathlib for working with file paths
import random  # Random number generation
import json  # JSON (JavaScript Object Notation) for data interchange
import pandas as pd  # Pandas library for data manipulation and analysis
import yaml  # YAML (YAML Ain't Markup Language) for configuration files
from collections import Counter  # Counter class from collections for counting occurrences
from sklearn.model_selection import KFold  # KFold from scikit-learn for cross-validation
from tqdm.notebook import tqdm  # tqdm for creating progress bars in loops
from ultralytics import YOLO  # YOLO (You Only Look Once) from Ultralytics for object detection
import sys  # sys for system-specific parameters and functions
import boto3  # Boto3 for Amazon Web Services (AWS) SDK for Python
import gc  # Garbage Collection for memory management
import torch  # PyTorch deep learning framework
import time  # Time-related functions
import re # Regular expression module
import argparse # Argument parsing module
import utils
import logging

def main(config_path):
    
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    image_path = Path(config["image_path"])
    print(f"The image path is: {image_path}")
    experiments_path = Path(config["experiments_path"])
    print(f"K-fold path is: {experiments_path}")
    seed = 42
    
    logging.basicConfig(level=logging.INFO)
    experiment_path = experiments_path / f"{config['experiment_name']}"
    experiment_path.mkdir(parents=True, exist_ok=True)

    logging.info("Splitting the dataset...")
    utils.new_k_fold_split(config, seed, image_path, experiment_path)
    logging.info("Script finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your script description here.")
    parser.add_argument("config", type=str, help="Path to the configuration file.")

    args = parser.parse_args()

    main(args.config)
    