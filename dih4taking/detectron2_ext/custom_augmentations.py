import cv2 as cv

from detectron2.data import transforms as T
from fvcore.transforms.transform import Transform

class ToGrayTransform(Transform):
    def __init__(self):

        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray_as_rgb = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
        
        return gray_as_rgb

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation

    def inverse(self):
        return T.NoOpTransform()



class RandomToGray(T.Augmentation):
    def __init__(self, prob:float=0.5):
        self.prob = prob
        super().__init__()

    def get_transform(self, image):
        if self._rand_range() < self.prob:
            return ToGrayTransform()
        else:
            return T.NoOpTransform()