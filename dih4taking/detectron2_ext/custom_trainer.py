from detectron2.engine import DefaultTrainer
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data import transforms as T, build_detection_train_loader

from .custom_augmentations import RandomToGray

class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):

        augs = [
            T.RandomBrightness(0.9, 1.1),
            T.RandomSaturation(intensity_min=0.9, intensity_max=1.1),
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            RandomToGray(prob=0.5),
        ]

        mapper=DatasetMapper(cfg, is_train=True, augmentations=augs)

        return build_detection_train_loader(cfg, mapper=mapper)
