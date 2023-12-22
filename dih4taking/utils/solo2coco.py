import os
import time
import json
import shutil

from dataclasses import asdict
from itertools import groupby

import cv2 as cv
import numpy as np

import pycocotools.mask as cocomask

from .framedata import FrameData
from .cocodatasetutils import Image, Annotation, Category
from .cocodatasetutils import COCODataset



ANNOTATION_FILE = "annotation_definitions.json"
METADATA_FILE = "metadata.json"
METRIC_FILE = "metric_definitions.json"
SENSOR_FILE = "sensor_definitions.json"
FILE_PREFIX = "step"
SEQUENCE_PREFIX = "sequence"


class SOLO2COCOConverter:
    '''
    support just 1 sensor with id: "camera"
    '''

    def __init__(self, root_solo:str, root_coco:str, prefix:str="train", suffix:str=""):
        self._root_solo = root_solo
        self._root_coco = root_coco
        self._prefix = prefix

        self._images_folder = os.path.join(self._root_coco, "coco"+suffix, self._prefix, "images")
        self._labels_folder = os.path.join(self._root_coco, "coco"+suffix, self._prefix, "labels")

        self._setup_converter()

        self.visiblity_th:float = 1.0

    def get_n_steps_in_sequence(self, k_seq:int):
        files = os.listdir(os.path.join(self._root_solo, self._get_sequence_folder_name(k_seq)))
        n_steps = len([x for x in files if "frame_data" in x])
        return n_steps


    def _create_dirs(self):
        os.makedirs(self._images_folder, exist_ok=True)
        os.makedirs(self._labels_folder, exist_ok=True)

    def _get_sequence_folder_name(self, k_seq:int):
        return f"{SEQUENCE_PREFIX}.{k_seq}"
    

    def _load_metadata(self):
        with open(os.path.join(self._root_solo, METADATA_FILE), "r") as f:
            self.metadata:dict = json.load(f)

    def _load_annotation_definitions(self):
        with open(os.path.join(self._root_solo, ANNOTATION_FILE), "r") as f:
            self.annotations_definition:dict = json.load(f)


    def parse_frame(self, k_seq:int, k_step:int):
        sequence_folder_name = self._get_sequence_folder_name(k_seq)
        frame_data_file_name = f"{FILE_PREFIX}{k_step}.frame_data.json"

        frame_data = FrameData()
        frame_data.parse_json(os.path.join(self._root_solo, sequence_folder_name, frame_data_file_name))

        return frame_data
    
    def _process_rgb_image(self, k_seq:int, k_step:int, frame_data:FrameData):
        rgb_img_name = f"{FILE_PREFIX}{k_step}.{frame_data.sensor_id}.png"
        rgb_img_name_unique = f"{SEQUENCE_PREFIX}{k_seq}.{rgb_img_name}"

        rgb_img_path = os.path.join(self._root_solo, self._get_sequence_folder_name(k_seq), rgb_img_name)
        shutil.copy(rgb_img_path, os.path.join(self._images_folder, rgb_img_name_unique))

        record = Image(rgb_img_name_unique, frame_data.dimension.dimension[1], frame_data.dimension.dimension[0], frame_data.frame_number)
        return record
    
    def _load_segmentation_img(self, k_seq:int, k_step:int, sensor_id:str):
        img_name = f"{FILE_PREFIX}{k_step}.{sensor_id}.InstanceSegmentation.png"
        seg_img = cv.imread(os.path.join(self._root_solo, self._get_sequence_folder_name(k_seq), img_name))
        seg_img = cv.cvtColor(seg_img, cv.COLOR_BGR2RGB)

        return seg_img
    
    def _binary_mask_to_rle(self, binary_mask:np.ndarray):
        rle = {'counts': [], 'size': list(binary_mask.shape)}
        counts = rle.get('counts')
        for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
            if i == 0 and value == 1:
                counts.append(0)
            counts.append(len(list(elements)))
        return rle

    def _compute_segmentation_map(self, seg_img:cv.Mat, color:list[int]):
        '''
        color: RGBA
        '''

        if seg_img.shape[-1] == 4:
            ins_color = (*color,)
        else:
            ins_color = (*color[0:-1],)

        instance_mask = np.all(seg_img == ins_color, axis=-1).astype(np.uint8)
        # segmentation = cocomask.encode(np.asfortranarray(instance_mask))
        # segmentation["counts"] =  cocomask.decode(segmentation["counts"])
        segmentation = self._binary_mask_to_rle(instance_mask)

        return segmentation
    
    def _process_frame_instances(self, k_seq:int, k_step:int, frame_data:FrameData):
        assert len(frame_data.bbox) == len(frame_data.occlusion) == len(frame_data.rendering) == len(frame_data.instance)

        frame_data_zip = zip(frame_data.rendering, frame_data.occlusion, frame_data.bbox, frame_data.instance)


        annotations:list[Annotation] = []


        seg_img = self._load_segmentation_img(k_seq, k_step, frame_data.sensor_id)
        for (rendering, occlusion, bbox, instance) in frame_data_zip:
            if occlusion.percentVisible < self.visiblity_th:
                continue

            area = rendering.visiblePixels
            image_id = frame_data.frame_number
            category_id = rendering.labelId
            segmentation = self._compute_segmentation_map(seg_img, instance.color)

            annotation = Annotation(
                segmentation=segmentation,
                area=area,
                iscrowd=0,
                image_id=image_id,
                bbox=[bbox.origin[0], bbox.origin[1], bbox.dimension[0], bbox.dimension[1]],
                category_id=category_id,
                id=instance.instanceId
            )

            annotations.append(annotation)

        return annotations
    

    
    def get_categories(self):
        categories:list[Category] = []
        bbox_spec = next((x for x in self.annotations_definition["annotationDefinitions"] if x["id"] == "BoundingBox"), None)


        if bbox_spec is not None:
            for label in bbox_spec["spec"]:
                category = Category("default", label["label_id"], label["label_name"])
                categories.append(category)

        return categories
    
    def process_frame(self, k_seq:int, k_step:int, frame_data:FrameData):
        img = self._process_rgb_image(k_seq, k_step, frame_data)
        t1 = time.perf_counter()
        annotations = self._process_frame_instances(k_seq, k_step, frame_data)
        t2 = time.perf_counter()
        print(f"{k_seq=}, {k_step=}: {(t2-t1)=}")

        return img, annotations


    def convert(self):
        dataset = COCODataset()
        total_sequences = self.metadata["totalSequences"]
        
        instance_unique_id=0

        for k_seq in range(total_sequences):
            frames_per_iteration = self.get_n_steps_in_sequence(k_seq)
            for k_step in range(frames_per_iteration):

                frame_data = self.parse_frame(k_seq, k_step)
                img, annotations = self.process_frame(k_seq, k_step, frame_data)

                for ann in annotations:
                    ann.id = instance_unique_id
                    instance_unique_id+=1


                dataset.add_record(img, annotations)


        categories = self.get_categories()

        dataset.set_categories(categories)

        return dataset
    
    def export_labels(self, dataset:COCODataset):
        with open(os.path.join(self._labels_folder, "labels.txt"), "w") as f:
            coco_json = {
                "images": dataset.images,
                "annotations": [x for xx in dataset.annotations for x in xx],
                "categories": dataset.categories,
                }
            
            for k, v in coco_json.items():
                coco_json[k] = [asdict(x) for x in v]

            json.dump(coco_json, f)



    def _setup_converter(self):
        self._create_dirs()
        self._load_annotation_definitions()
        self._load_metadata()



