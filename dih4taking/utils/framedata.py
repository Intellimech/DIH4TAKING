import os
import json
import dataclasses
from dataclasses import dataclass


METRICS_KEY = "metrics"
OCCLUSION_ID= "Occlusion"
ANNOTATIONS_KEY = "annotations"
CAPTURES_KEY = "captures"
CAMERA_ID = "camera"
DIMENSION_KEY = "dimension"
INSTANCE_SEGMENTATION_ID = "InstanceSegmentation"
SEMANTIC_SEGMENTATION_ID = "SemanticSegmentation"
BOUNDING_BOX_ID = "BoundingBox"
RENDERED_INFO_ID = "RenderedObjectInfo"


@dataclass
class ObjectOcclusion:
    instanceId: int
    percentVisible:float
    percentInFrame:float
    visibilityInFrame:float

@dataclass
class ObjectRenderedInfo:
    labelId:int
    instanceId:int
    color:list[int]
    visiblePixels:int
    parentInstanceId:int
    childrenInstanceIds:list[int]
    labels:list[str]


@dataclass
class ObjectSemantic:
    labelName:str
    pixelValue:list[int]

@dataclass
class ObjectInstance:
    instanceId:int
    labelId:int
    labelName:str
    color:list[int]

@dataclass
class ObjectBoundingBox:
    instanceId:int
    labelId:int
    labelName:str
    origin:list[int]
    dimension:list[int] #width,height


@dataclass
class ImageDimension:
    dimension:list[int]


class FrameData:
    def __init__(self):
        self.frame_data = ""

        self.rendering:list[ObjectRenderedInfo] = []
        self.occlusion:list[ObjectOcclusion] = []
        self.instance:list[ObjectInstance] = []
        self.semantic:list[ObjectSemantic] = []
        self.bbox:list[ObjectBoundingBox] = []
        self.dimension:ImageDimension = [0,0]
        self.frame_number:int = -1
        self.sensor_id = CAMERA_ID

    def parse_json(self, frame_data_path:str):
        with open(frame_data_path, "r") as f:
            self.frame_data = json.load(f)

        self.frame_number = self.frame_data["frame"]

        self._get_rendering_info()
        self._get_occlusion()
        self._get_instance_segmentation()
        self._get_semantic_segmentation()
        self._get_bounding_box()
        self._get_dimension()


    def _get_capture_data(self):
        capture_data = next((x for x in self.frame_data[CAPTURES_KEY] if x["id"] == CAMERA_ID))
        return capture_data

    def _get_rendering_info(self):
        rendering_data = next((x for x in self.frame_data[METRICS_KEY] if x["id"] == RENDERED_INFO_ID), [])
        if "values" in rendering_data:
            for value in rendering_data["values"]:
                data = {x:value[x] for x in ObjectRenderedInfo.__annotations__.keys()}
                rendering = ObjectRenderedInfo(**data)
                self.rendering.append(rendering)

    def _get_occlusion(self):
        occlusion_data = next((x for x in self.frame_data[METRICS_KEY] if x["id"] == OCCLUSION_ID), [])
        if "values" in occlusion_data:
            for value in occlusion_data["values"]:
                data = {x:value[x] for x in ObjectOcclusion.__annotations__.keys()}
                occlusion = ObjectOcclusion(**data)
                self.occlusion.append(occlusion)

    def _get_instance_segmentation(self):
        capture_data = self._get_capture_data()
        instance_data = next((x for x in capture_data[ANNOTATIONS_KEY] if x["id"] == INSTANCE_SEGMENTATION_ID), [])

        if "instances" in instance_data:
            for inst in instance_data["instances"]:
                data = {x:inst[x] for x in ObjectInstance.__annotations__.keys()}
                instance = ObjectInstance(**data)
                self.instance.append(instance)

    def _get_semantic_segmentation(self):
        capture_data = self._get_capture_data()
        semantic_data = next((x for x in capture_data[ANNOTATIONS_KEY] if x["id"] == SEMANTIC_SEGMENTATION_ID), [])
        if "instances" in semantic_data:
            for inst in semantic_data["instances"]:
                data = {x:inst[x] for x in ObjectSemantic.__annotations__.keys()}
                semantic = ObjectSemantic(**data)
                self.semantic.append(semantic)

    def _get_bounding_box(self):
        capture_data = self._get_capture_data()
        bbox_data = next((x for x in capture_data[ANNOTATIONS_KEY] if x["id"] == BOUNDING_BOX_ID), [])
        if "values" in bbox_data:
            for value in bbox_data["values"]:
                data = {x:value[x] for x in ObjectBoundingBox.__annotations__.keys()}
                bbox = ObjectBoundingBox(**data)
                self.bbox.append(bbox)


    def _get_dimension(self):
        capture_data = self._get_capture_data()
        dimension = ImageDimension(dimension=capture_data[DIMENSION_KEY])
        self.dimension = dimension






            

    

