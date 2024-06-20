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
import pickle
import matplotlib.pyplot as plt

# detectron2 functions
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

from detectron2.evaluation import COCOEvaluator, inference_on_dataset, LVISEvaluator
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.utils.visualizer import ColorMode

import torchvision
import torchvision.transforms as transforms

from newTrainer import NewTrainer
from valLossHook import ValLossHook
from detectron2.engine import hooks
from computervisionengLoss import ValidationLoss


## augmentation part
from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T
import copy

def mixedsort(item):
    """
    Sort a string containing both numeric and non-numeric parts.

    Parameters:
    - item (str): The string to be sorted.

    Returns:
    - list: A list containing numeric and non-numeric parts of the input string.

    Example:
    strings_to_sort = ["file2.txt", "file11.txt", "file1.txt", "file20.txt", "file10.txt"]
    sorted_strings = sorted(strings_to_sort, key=mixedsort)
    Output: ['file1.txt', 'file2.txt', 'file10.txt', 'file11.txt', 'file20.txt']
    """
    
    parts = re.split(r'(\d+)', item) # Use a regular expression to extract numeric and non-numeric parts
    parts[1::2] = map(int, parts[1::2]) # Convert numeric parts to integers for proper numerical sorting
    return parts

def single_labeling(label_path, insect_id):
    """
    This function creates separate a label directory for a specific insect class.

    Args:
        label_path (str): The source directory containing label files.
        insect_id (int): The insect class ID to filter for and create separate directories.

    Example:
        single_labeling("all_labels", 0)
    """
    if isinstance(label_path, Path):
        label_path = str(label_path)
        
    destination_dir = label_path + f"_{insect_id}"

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir,exist_ok=True)
    else:
        print(f"{destination_dir} already exists")

    for filename in os.listdir(label_path): # List all text files in the source directory
        if filename.endswith('.txt') and filename not in os.listdir(destination_dir):
            source_file = os.path.join(label_path, filename) # Read the source file
            with open(source_file, 'r') as file:
                lines = [line for line in file if line.startswith(str(insect_id))] # Read and filter the lines

            destination_file = os.path.join(destination_dir, filename)
            with open(destination_file, 'w') as file:
                file.writelines(lines)
        else:
            print("The file already exists")


def image_enhancing(source_dir, flags, resize_options=None):
    """
    Enhance images in the specified directory using various techniques.

    Args::
        source_dir (str): The source directory containing images.
        flags (list): A list of enhancement flags. Supported flags include:
            - "crop" for cropping the image
            - "bright" for adjusting brightness
            - "sharp" for sharpening
            - "bright_and_sharp" for both brightness and sharpening
        resize_options (list of tuples, optional): A list of (image type, size) pairs
            where image type is one of the supported flags and size is an integer representing
            the size in pixels.

    Examples:
        To crop and adjust brightness, use: 
            image_path = Path(os.getcwd()) / f"franca_traps/"
            image_enhancing(str(image_path), ["crop", "bright"], ["crop", 1280])
        To sharpen images and apply resize to them, use: image_enhancing("my_dir", ["sharp"], ["sharp", 1280])
    """

    if not isinstance(flags, list):
        raise ValueError("flags must be a list of enhancement flags.")

    def adjust_brightness_contrast(image, alpha, beta):
        return cv2.addWeighted(image, alpha, image, 0, beta)

    def sharpen_image(image):
        kernel = np.array([[-1, -1, -1],
                        [-1, 9, -1],
                        [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)

    image_path = Path(source_dir)
    project_path = image_path.parent
    
    image_path_list = list(image_path.glob("*.jpg"))
    
    if "crop" in flags:
        sub_dir = project_path / "hsv"
        os.makedirs(sub_dir, exist_ok=True)
    for flag in flags:
        # Create subdirectories if they don't exist
        sub_dir = project_path / flag
        os.makedirs(sub_dir, exist_ok=True)
    if resize_options:
        sub_dir = project_path / f"{resize_options[0]}_{resize_options[1]}"
        os.makedirs(sub_dir, exist_ok=True)

    for i in image_path_list:
        image = cv2.imread(str(i))

        # crop
        if "crop" in flags:
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([40, 255, 255])
            mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
            result = cv2.bitwise_and(image, image, mask=mask)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.imwrite(f'{project_path}/hsv/{i.name}', result)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cropped_image = image[y:y+h, x:x+w]
            cv2.imwrite(f'{project_path}/crop/{i.name}', cropped_image)

        # brightness
        if "bright" in flags:
            enhanced_image = adjust_brightness_contrast(image, 1.1, 10)
            cv2.imwrite(f'{project_path}/bright/{i.name}', enhanced_image)

        if ["bright", "crop"] in flags:
            enhanced_image = adjust_brightness_contrast(cropped_image, 1.1, 10)
            cv2.imwrite(f'{project_path}/bright/{i.name}', enhanced_image)

        if "sharp" in flags:
            sharpened_image = sharpen_image(image)
            cv2.imwrite(f'{project_path}/sharp/{i.name}', sharpened_image)

        if ["sharp", "crop"] in flags:
            sharpened_image = sharpen_image(cropped_image)
            cv2.imwrite(f'{project_path}/sharp/{i.name}', sharpened_image)

        # brightness and sharpness
        if "bright_and_sharp" in flags:
            sharpened_image = sharpen_image(enhanced_image)
            cv2.imwrite(f'{project_path}/bright_and_sharp/{i.name}', sharpened_image)

        # resize
        if resize_options:
            if resize_options[0] == "crop":
                resized_image = cv2.resize(cropped_image, (resize_options[1], resize_options[1]))
            elif resize_options[0] == "bright":
                resized_image = cv2.resize(enhanced_image, (resize_options[1], resize_options[1]))
            elif resize_options[0] == "sharp":
                resized_image = cv2.resize(sharpened_image, (resize_options[1], resize_options[1]))
            else:
                resized_image = cv2.resize(sharpened_image, (resize_options[1], resize_options[1]))

            cv2.imwrite(f'{project_path}/{resize_options[0]}_{resize_options[1]}/{i.name}', resized_image)


def new_k_fold_split(data_param, seed, image_path, save_path):
    
    insect = data_param["insect_id"]
    augm = data_param["enhancement"]
    ksplit = data_param["ksplit"]

    test_path = save_path.parent
    
    if len(insect) < 2:
        label_files = list(image_path.rglob(f"*labels_{insect}/*.txt"))
    else:
        label_files = list(image_path.rglob("*labels/*.txt"))

    label_files.sort()
    label_files = [file for file in label_files if ".ipynb_checkpoints" not in file.parts]
          
    if insect == "0,1":
        yaml_file = test_path / f'config_template.yaml'  # your data YAML with data directories and names dictionary
    else:
        yaml_file = test_path / f'config_template_{insect}.yaml'
    
    with open(yaml_file, 'r', encoding="utf8") as y:
        classes = yaml.safe_load(y)['names']
    
    cls_idx = list(range(len(classes)))
    print(f"Insect number class are: {cls_idx}")
    
    indx_name = [l.stem for l in label_files] # uses base filename as ID (no extension)
    
    if insect == "0":
        labels_df = pd.DataFrame([], columns=classes, index=indx_name)
        for label in label_files:
            lbl_counter = Counter()
            with open(label,'r') as lf:
                lines = lf.readlines()
            for l in lines:
                lbl_counter[int(l[0])] += 1
            labels_df.loc[label.stem] = lbl_counter[0]
        labels_df = labels_df.fillna(0)
    elif insect == "1":
        labels_df = pd.DataFrame([], columns=classes, index=indx_name)
        for label in label_files:
            lbl_counter = Counter()
            with open(label,'r') as lf:
                lines = lf.readlines()
            for l in lines:
                lbl_counter[int(l[0])] += 1
            labels_df.loc[label.stem] = lbl_counter[1]
        labels_df = labels_df.fillna(0)
    elif insect == "0,1":
        labels_df = pd.DataFrame([], columns=cls_idx, index=indx_name)
        for label in label_files:
            lbl_counter = Counter()
            with open(label,'r') as lf:
                lines = lf.readlines()
            for l in lines:
                lbl_counter[int(l[0])] += 1
            labels_df.loc[label.stem] = lbl_counter
        labels_df = labels_df.fillna(0)
    elif insect == "0+1":
        labels_df = pd.DataFrame([], columns=classes, index=indx_name)
        for label in label_files:
            lbl_counter = Counter()
            with open(label,'r') as lf:
                lines = lf.readlines()
            for l in lines:
                lbl_counter[int(l[0])] += 1
            labels_df.loc[label.stem] = lbl_counter[0] + lbl_counter[1]
        labels_df = labels_df.fillna(0)
        
    kf = KFold(n_splits=ksplit, shuffle=True, random_state=seed)   # setting random_state for repeatable results
    kfolds = list(kf.split(labels_df))
    
    folds = [f'split_{n}' for n in range(1, ksplit + 1)]
    folds_df = pd.DataFrame(index=indx_name, columns=folds)
    
    for idx, (train, val) in enumerate(kfolds, start=1):
        folds_df[f'split_{idx}'].loc[labels_df.iloc[train].index] = 'train'
        folds_df[f'split_{idx}'].loc[labels_df.iloc[val].index] = 'val'
    
    if insect == "0":
        fold_lbl_distrb = pd.DataFrame(index=folds, columns=classes)
    elif insect == "1":
        fold_lbl_distrb = pd.DataFrame(index=folds, columns=classes)
    elif insect == "0,1":
        fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)
    elif insect == "0+1":
        fold_lbl_distrb = pd.DataFrame(index=folds, columns=classes)  
        
    for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
        train_totals = labels_df.iloc[train_indices].sum()
        val_totals = labels_df.iloc[val_indices].sum()
    
        # To avoid division by zero, we add a small value (1E-7) to the denominator
        ratio = val_totals / (train_totals + 1E-7)
        fold_lbl_distrb.loc[f'split_{n}'] = ratio
    
    # images
    image_files = list(image_path.rglob(f"{augm}/*.jpg"))
    image_files.sort()
    
    image_files = [file for file in image_files if ".ipynb_checkpoints" not in file.parts]
    ds_yamls = []
    
    for split in folds_df.columns:
        # Create directories
        split_dir = save_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / 'train' / 'images').mkdir(parents=True, exist_ok=True)
        (split_dir / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
        (split_dir / 'val' / 'images').mkdir(parents=True, exist_ok=True)
        (split_dir / 'val' / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Create dataset YAML files
        dataset_yaml = split_dir / f'{split}_dataset.yaml'
        ds_yamls.append(dataset_yaml)
        if not os.path.isfile(dataset_yaml):
            with open(dataset_yaml, 'w') as ds_y:
                yaml.safe_dump({
                    'path': os.path.abspath(split_dir),
                    'train': 'train',
                    'val': 'val',
                    'names': classes
                }, ds_y)
                
    print(f"Start copying")
    for image, label in zip(image_files, label_files):
        for split, k_split in folds_df.loc[image.stem].items():
            # Destination directory
            img_to_path = save_path / split / k_split / 'images'
            lbl_to_path = save_path / split / k_split / 'labels'
            # Copy image and label files to new directory (SamefileError if file already exists)
            shutil.copy(image, img_to_path / image.name)
            shutil.copy(label, lbl_to_path / label.name)
    print(f"Finished copying")
    # Optionally, you can save the records of the K-Fold split and label distribution DataFrames as CSV files for future reference.
    folds_df.to_csv(save_path / "kfold_datasplit.csv")
    fold_lbl_distrb.to_csv(save_path / "kfold_label_distribution.csv")
    print(f"END")
    
def training(test_path, config, yaml_file):

    start = time.time()
    model = YOLO(test_path.parents[1] / f"{config['model_type']}.pt")
    model.to("cuda")
    model.train(data=yaml_file,
                epochs=config["epochs"],
                batch=config["batch_size"],
                imgsz=config["image_size"],
                save=True,
                project=test_path/config["project_name"],
                name="train",
                single_cls=config["single_cls"],
                cache = config["cache"],
                workers=config["workers"],
                hsv_h=config["hsv_h"],
                hsv_s=config["hsv_s"],
                hsv_v=config["hsv_v"],
                degrees=config["degrees"],
                translate=config["translate"],
                scale=config["scale"],
                shear=config["shear"],
                perspective=config["perspective"],
                flipud=config["flipud"],
                fliplr=config["fliplr"],
                mosaic=config["mosaic"],
                mixup=config["mixup"],
                copy_paste=config["copy_paste"],
                erasing=config["erasing"],
                crop_fraction=config["crop_fraction"],
                auto_augment=config["auto_augment"],
                optimizer=config['optimizer'],
                seed=config['seed'],
                lr0=config['lr_init'],
                lrf=config['lr_end']
               )
    end = time.time()
    print(f"Finish with:{round((end - start), 2)} second, num_workers={config['workers']}")
    del yaml_file
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
def validation_save(test_path, project_name, conf_score, json_set):
    start = time.time()
    model = YOLO(test_path/project_name/f"train/weights/best.pt")  # cambiare
    results = model.val(project=test_path/project_name, conf=conf_score, name="val", save_json=json_set) # cambiare
    end = time.time()
    print(f"Finish with:{round((end - start), 2)} second")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    det_metrics = results
    box_information = det_metrics.box
    box_dict = {
        'all_ap': box_information.all_ap,
        'ap': box_information.ap,
        'ap50': box_information.ap50,
        'f1_score': box_information.f1,
        'p': box_information.p,
        'r': box_information.r
    }

    metrics_dict = {
        'ap_class_index': det_metrics.ap_class_index,
        'box': box_dict,
        'confusion_matrix': det_metrics.confusion_matrix.matrix,
        'curves': det_metrics.curves,
        'curves_results': det_metrics.curves_results,
        'names': dict(det_metrics.names),
        'results_dict': dict(det_metrics.results_dict),
        'save_dir': str(det_metrics.save_dir),
        'speed': dict(det_metrics.speed),
        'task': det_metrics.task,
        'conf_score': conf_score,
    }

    pickle_file_path = os.path.abspath(test_path/project_name/"val/validation_results.pkl") # cambiare
    with open(pickle_file_path, 'wb') as pickle_file:
        pickle.dump(metrics_dict, pickle_file) 

def detectron2_get_dicts(image_dir, label_dir):
    
    dataset_dicts = []
    for idx, file in enumerate(os.listdir(label_dir)):
        record = {}
        filename = os.path.join(image_dir, file[:-4] + ".jpg")
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        objs = []
        with open(os.path.join(label_dir, file)) as r:
            lines = [l[:-1] for l in r.readlines()]

        for _, line in enumerate(lines):
            if len(line) > 2:
                label, cx, cy, w_, h_ = line.split(' ')

                obj = {
                    "bbox": [int((float(cx) - (float(w_) / 2)) * width),
                             int((float(cy) - (float(h_) / 2)) * height),
                             int(float(w_) * width),
                             int(float(h_) * height)],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": int(label),
                }

                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def detectron2_custom_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    transform_list = [
        T.Resize((800,600)),
        T.RandomBrightness(0.8, 1.8),
        T.RandomContrast(0.6, 1.3),
        T.RandomSaturation(0.8, 1.4),
        T.RandomRotation(angle=[90, 90]),
        T.RandomLighting(0.7),
        T.RandomFlip(prob=0.4, horizontal=False, vertical=True),
    ]
    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict
    
def detectron2_setup(split_path, config):
    
    class_names = ['ST', 'OI']
    class_list_file = split_path / "class.names"
    try:
        with open(class_list_file, 'x') as writer:  # 'x' mode creates the file if it doesn't exist
            for class_name in class_names:
                writer.write(class_name + '\n')
    except FileExistsError:
        print("File class.names already exists")
        pass  # File already exists, no need to create it
    
    with open(class_list_file, 'r') as reader:
        classes_ = [l.strip() for l in reader.readlines()]  # Use strip to remove newline characters
        print(classes_)

    split_number = int(re.findall(r'\d+', Path(split_path).stem)[0])
    for d in ["train", "val"]:
        DatasetCatalog.register(f"{split_number}insect_" + d, lambda d=d: detectron2_get_dicts(os.path.join(split_path, d, "images"), os.path.join(split_path, d, "labels")))
        MetadataCatalog.get(f"{split_number}insect_" + d).set(thing_classes=classes_)

    train_var = DatasetCatalog.get(f'{split_number}insect_train')
    val_var = DatasetCatalog.get(f'{split_number}insect_val')
    train_len = len(train_var)
    val_len = len(val_var)
    print(f"Train len: {train_len}")
    print(f"Val len: {val_len}")
          
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config["2_model"]))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config["2_model"])
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(MetadataCatalog.get(f"{split_number}insect_train").thing_classes) # 1
    cfg.MODEL.DEVICE = config["2_device"]
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256 # 128 # parameter used to calculate class loss during training https://github.com/facebookresearch/detectron2/issues/1341
    # https://medium.com/@hirotoschwert/digging-into-detectron-2-part-5-6e220d762f9
    cfg.DATASETS.TRAIN = (f'{split_number}insect_train',)
    cfg.DATASETS.TEST = (f'{split_number}insect_val',)  # it seems you need to put your validation as DATASETS.TEST https://github.com/facebookresearch/detectron2/issues/286
    cfg.TEST.EVAL_PERIOD = 100 # int(len(train_var) / cfg.SOLVER.IMS_PER_BATCH // 10 * 10) # config["2_eval_period"]
    
    cfg.DATALOADER.NUM_WORKERS = config["2_workers"]
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False # https://github.com/facebookresearch/detectron2/issues/4082
    
    cfg.SOLVER.IMS_PER_BATCH = config["2_batch_size"]
    cfg.SOLVER.BASE_LR = config["2_base_lr"]
    cfg.SOLVER.MAX_ITER = config["2_iterations"]
    cfg.SOLVER.STEPS = [] # in this way the learning rate will not be decayed (as in YOLO)
    cfg.SOLVER.CHECKPOINT_PERIOD = config["2_check_period"] # int(cfg.SOLVER.MAX_ITER / 2) # https://github.com/facebookresearch/detectron2/issues/1295
    cfg.SEED = config["2_seed"]
    cfg.OUTPUT_DIR = str(split_path / config["2_experiment_name"])
    # for no augmentation
    # cfg.INPUT.RANDOM_FLIP="none"
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    print("Setup up finished!")

    return cfg
    
def detectron2_train(cfg, config):
    start = time.time()
    config["2_trainer"] == "new"
    trainer = NewTrainer(cfg)
    trainer.register_hooks([ValLossHook(cfg)])
    periodic_writer_hook = [hook for hook in trainer._hooks if isinstance(hook, hooks.PeriodicWriter)]
    all_other_hooks = [hook for hook in trainer._hooks if not isinstance(hook, hooks.PeriodicWriter)]
    trainer._hooks = all_other_hooks + periodic_writer_hook
        
    print("Start training")
    trainer.resume_or_load(resume=False) 
    trainer.train()
    
    f = open(os.path.join(cfg.OUTPUT_DIR, "config.yml"), 'w')
    f.write(cfg.dump())
    f.close()
    
    end = time.time()
    print(f"Finish with:{round((end - start), 2)} second, num_workers={cfg.DATALOADER.NUM_WORKERS}")
    gc.collect()
    torch.cuda.empty_cache()

def detectron2_val(split_path, config, model_weight="model_final"):
    cfg = get_cfg()
    cfg.merge_from_file(str(split_path / config["2_experiment_name"] / "config.yml"))
    metrics_df = pd.read_json(os.path.join(cfg.OUTPUT_DIR, "metrics.json"), orient="records", lines=True) 
    metrics_df = metrics_df[~metrics_df["total_loss"].isna()]
    
    # image to check loss function
    fig, ax = plt.subplots()
    ax.plot(metrics_df["iteration"], metrics_df["total_loss"], label="train")
    ax.plot(metrics_df["iteration"], metrics_df["val_total_loss"], label="validation")
    ax.legend(loc='upper right')
    ax.set_title("Loss curve")
    output_file = os.path.join(cfg.OUTPUT_DIR, "loss_curve.png")
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.show()

    class_list_file = split_path / "class.names"
    with open(class_list_file, 'r') as reader:
        classes_ = [l.strip() for l in reader.readlines()]  # Use strip to remove newline characters
        print(classes_)

    split_number = int(re.findall(r'\d+', Path(split_path).stem)[0])
    for d in ["train", "val"]:
        DatasetCatalog.register(f"{split_number}insect_" + d, lambda d=d: detectron2_get_dicts(os.path.join(split_path, d, "images"), os.path.join(split_path, d, "labels")))
        MetadataCatalog.get(f"{split_number}insect_" + d).set(thing_classes=classes_)
    
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, f"{model_weight}.pth") # or choose another checkpoint.pth
    cfg.MODEL.DEVICE = config["2_device"]
    predictor = DefaultPredictor(cfg)
    output_folder = os.path.join(cfg.OUTPUT_DIR, "validation/")
    evaluator = COCOEvaluator(f"{split_number}insect_val", ("bbox",), False, output_dir=output_folder)
    val_loader = build_detection_test_loader(cfg, f"{split_number}insect_val")
    result = inference_on_dataset(predictor.model, val_loader, evaluator)
    with open(os.path.join(output_folder, "validation_results.pkl"), 'wb') as pickle_file:
        pickle.dump(result['bbox'], pickle_file) 

    df = pd.DataFrame([result["bbox"]])
    df['Experiment'] = Path(cfg.OUTPUT_DIR).stem
    df['Split'] = int(re.findall(r'\d+', Path(split_path).stem)[0])
    df = df[['Experiment', 'Split'] + [col for col in df.columns if col not in ['Experiment', 'Split']]]
    df.to_csv(os.path.join(output_folder, "validation.csv"), index=False)
    # Training with a validation dataset?: https://github.com/facebookresearch/detectron2/issues/4368

    # predict on images
    print("predicting images")
    val_dataset = DatasetCatalog.get(f'{split_number}insect_val')
    val_images_dir = os.path.join(output_folder, "images")
    if not os.path.exists(val_images_dir):
        os.makedirs(val_images_dir,exist_ok=True)

    for d in val_dataset:
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)
        v = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(f"{split_number}insect_val"), instance_mode=ColorMode.SEGMENTATION)
        output = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        image_path = os.path.join(val_images_dir, os.path.basename(d["file_name"]))
        cv2.imwrite(image_path, output.get_image()[:,:,::-1])
    
    
    return metrics_df, result, cfg

def detectron2_predict(split_path, config, image_path=None):
    cfg = get_cfg()
    cfg.merge_from_file(str(split_path / config["2_experiment_name"]/ "config.yml"))
    # cfg.merge_from_file('march_experiments/01_crop_10folds/split_1/faster_rcnn_R_50_FPN_1xmay/config.yml')
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") # or choose another checkpoint.pth
    cfg.MODEL.DEVICE = config["2_device"]
    
    predictor = DefaultPredictor(cfg)

    # print/save images
    # single image
    if image_path:
        image = cv2.imread(image_path)
        outputs = predictor(im)
        threshold = config["2_conf_score"]
        pred_classes = outputs["instances"].pred_classes.tolist()
        scores = outputs["instances"].scores.tolist()
        bboxes = outputs["instances"].pred_boxes

        for j, bbox in enumerate(bboxes):
            bbox = bbox.tolist()
            score = scores[j]
            pred = pred_classes[j]
            if score > threshold:
                x1, y1, x2, y2 = [int(i) for i in bbox]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 5)
                
        cv2.imwrite(str(split_path / config["2_experiment_name"] / Path(image_path).name), image)

    evaluator = COCOEvaluator("insect_val", ("bbox",), False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "insect_val")
    result = inference_on_dataset(predictor.model, val_loader, evaluator)
    
    with open(os.path.join(cfg.OUTPUT_DIR, "validation_results.pkl"), 'wb') as pickle_file:
        pickle.dump(result['bbox'], pickle_file) 
    with open('notebook/validation_results.pkl', 'rb') as f:
        output_dict = pickle.load(f)
    
    data = {
        'experiment': ["PROVA_DETECTRON"],
        'mAP50': [output_dict['AP50']],
        'mAP75': [output_dict['AP75']],
        'mAP50_95': [output_dict['AP']],
        'AP-ST': [output_dict['AP-ST']],
        'AP-OI': [output_dict['AP-OI']]
    }

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(cfg.OUTPUT_DIR, "porvrrr.csv"), index=False)

    
    return result
    
## Stop EC2 instance with code
def stop_instance():
    aws_access_key_id = 'AKIAZU4DRSON6PGMLQOQ'
    aws_secret_access_key = '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'
    region_name = 'eu-central-1'
    instance_id = 'i-005b56ca09f762e4e'
    
    # Create an EC2 client
    ec2 = boto3.client('ec2', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=region_name)
    
    # Stop the instance
    response = ec2.stop_instances(InstanceIds=[instance_id])
    
    # Print the response
    print(response)
    