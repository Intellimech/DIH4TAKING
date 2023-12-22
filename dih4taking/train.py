import os
import sys
import copy
import random
import cv2 as cv

import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

from detectron2_ext.custom_trainer import CustomTrainer

train_images = r"images" #images path
train_labels = r"labels" #labels path - COCO format

experiments_dir = r"experiments" #experiments output folder

def setup_training():
    from train_cfg import cfg

    dataset_name = cfg.DATASETS.TRAIN[0]
    register_coco_instances(dataset_name, {}, train_labels, train_images)

    experiments = os.listdir(experiments_dir)
    experiment_number = max([int(exp_name.split("_")[1]) for exp_name in experiments], default=-1) +1
    experiment_folder = f"experiment_{experiment_number}"

    output_dir = os.path.join(experiments_dir, experiment_folder)
    os.makedirs(output_dir, exist_ok=True)

    cfg.OUTPUT_DIR = output_dir

    cfg_yaml = cfg.dump()

    cfg_yaml_path=os.path.join(output_dir, "train_cfg.yaml")
    with open(cfg_yaml_path, "w") as f:
        f.write(cfg_yaml)

    return cfg

if __name__ == "__main__":
    cfg = setup_training()
    trainer = CustomTrainer(cfg) 
    
    trainer.resume_or_load(resume=False)
    trainer.train()