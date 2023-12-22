

from dataclasses import dataclass

@dataclass
class Image:
    file_name:str
    height:int
    width:int
    id:int

@dataclass
class Category:
    supercategory:str
    id:int
    name:str


@dataclass
class Annotation:
    segmentation:list[float]
    area:float
    iscrowd:int
    image_id:int
    bbox:list[int]
    category_id:int
    id:int


class COCODataset:
    def __init__(self):
        self.images:list[Image] = []
        self.categories:list[Category] = []
        self.annotations:list[Annotation] = []

    def set_categories(self, categories:list[Category]):
        self.categories = categories

    def add_record(self, image:Image, annotations:list[Annotation]):
        self.images.append(image)
        self.annotations.append(annotations)

