# Detection of *Scaphoideus titanus* and *Orientus ishidae* on yellow sticky traps
This repository contains scripts to run YOLOv8 and Detectron2 for detecting insects on yellow sticky traps.

`config.yaml`: This configuration file sets up the parameters for an experiment involving k-fold cross-validation for object detection tasks. It includes settings for project directories, dataset paths, insect IDs, image enhancement types, and the number of splits for k-fold cross-validation.

`config_template.yaml`
`config_template_0.yaml`
`config_template_1.yaml`
`config_template_0+1.yaml`
These YAML configuration files provide basic information required for an object detection model to understand the classes it will be working with.

`training_val_config`: This configuration file sets up the parameters for training and validation for object detection models using YOLOv8 and Detectron2 frameworks. It includes settings for project directories, training parameters, data augmentation values, validation settings, and specific configurations for Detectron2. Key functionalities include image cropping, brightness adjustment, YOLO model training and evaluation, and data split for training and validation using YAML configuration files.

`utils.py`: This Python script integrates various computer vision functionalities, from data preprocessing (label filtering, image augmentation), model training and evaluation, K-fold cross-validation setup, and dataset handling

## YOLO
`k_fold_split.py`: This script performs a k-fold split of a dataset for object detection tasks, based on the configurations specified in a YAML file. It sets up necessary paths, initializes logging, and calls a utility function to split the dataset and save the results in the designated experiment directory.

```
python kfold_split.py config.yaml
```

`k_fold_training.py`: This script performs k-fold cross-validation training for object detection models using configurations specified in a YAML file. It iterates over split configuration files, trains the models, and logs the progress and results.

```
python kfold_training.py training_val_config.yaml
```

`kfold_validation.py`: This script performs validation on k-fold splits of a dataset for object detection, using configurations specified in a YAML file. It sets up necessary paths, initializes logging, and calls a utility function to validate and save results for each split configuration file.

```
python kfold_validation.py training_val_config.yaml
```
## Faster R-CNN
`kfold_detectron.py`: This script performs k-fold cross-validation training for object detection models using Detectron2, based on configurations specified in a YAML file. It sets up necessary paths, initializes logging, configures Detectron2 for each split, and trains the models, logging progress and results.

```
python kfold_detectron.py training_val_config.yaml
```

## Contact

If you encounter any issues with the project or for questions/suggestions, you can reach out to us at:
- Email: [giorgio.checola@fmach.it](mailto:giorgio.checola@fmach.it)
- Research and Innovation Centre, Digital Agriculture Unit, Fondazione Edmund Mach, Via Edmund Mach, 1, 38098 San Michele all'Adige TN, Italy.
