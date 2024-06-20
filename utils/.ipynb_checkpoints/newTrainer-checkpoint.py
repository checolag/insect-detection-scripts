import os

from detectron2.data import DatasetMapper
import detectron2.data.transforms as T
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data.build import build_detection_train_loader

# from ravijoTensorboardwriter import CustomTensorboardXWriter


class NewTrainer(DefaultTrainer):
  @classmethod
  def build_train_loader(cls, cfg):
    """
    Train loader with custom data augmentation
    """
    augmentations = [
        T.ResizeShortestEdge(short_edge_length=(640, 672, 704, 736, 768, 800), max_size=1333, sample_style="choice"), # 1080
        
        # T.RandomBrightness(0.8, 1.8),
        # T.RandomContrast(0.6, 1.3),
        # T.RandomSaturation(0.8, 1.4),
        # T.RandomRotation(angle=[90, 90]),
        # T.RandomLighting(0.7),
        # T.RandomFlip(prob=0.4, horizontal=False, vertical=True),
    
        T.RandomFlip(),
    ]
    mapper = DatasetMapper(cfg, is_train=True, augmentations=augmentations)
    return build_detection_train_loader(cfg, mapper=mapper)
    # return build_detection_train_loader(cfg)

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    if output_folder is None:
      output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    return COCOEvaluator(dataset_name, cfg, True, output_folder)