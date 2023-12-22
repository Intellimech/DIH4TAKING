from dataclasses import dataclass

import torch
torch._C._jit_set_bailout_depth(0) #altrimenti la seconda run del modello (ottenuto tramite tracing) è 100x più lenta
import cv2 as cv
import numpy as np

import torchvision #necessary import, even if unused, otherwise model won't run

from .inference_utils import paste_mask_in_image_fast

@dataclass
class PredictionInstance:
    score:float
    box:list[int] #[x0,y0,x1,y1]
    mask:np.ndarray


@dataclass
class ImagePrediction:
    shape:tuple[int]
    instances:list[PredictionInstance]

class Predictor2D:
    def __init__(self, model_path:str, inference_size:int):
        '''
        model_path: path to torch model exported by tracing
        inference_size: size of the image, if rectangular will be cropped to be square
        '''
        self._model_path:str = model_path
        self._inference_size:int = inference_size
        self._model = None

    def load_model(self):
        self._model = torch.jit.load(self._model_path)


    def predict(self, images:list[cv.Mat], treshold:float, bgr:bool=True):
        '''
        bgr: image input format, if false, it is assumed RGB
        '''
        predictions:list[ImagePrediction] = []
        images_starting_size = [img.shape for img in images]

        images_crop_coords = [self._get_crop_to_square_coordinates(img) for img in images]

        if not isinstance(images, list):
            img_l = [images]
        else:
            img_l = images

        img_l = [self._crop_img(img, coords) for img, coords in zip(images, images_crop_coords)]
        img_l = [self._resize_square_img(img, self._inference_size, copy=False) for img in img_l]
        if not bgr:
            img_l = [cv.cvtColor(img, cv.COLOR_RGB2BGR) for img in img_l]
        img_l = [torch.as_tensor(img.astype("float32").transpose(2, 0, 1)) for img in img_l]

        height, width = self._inference_size, self._inference_size
        inputs = ({"image": img, "height": torch.as_tensor(height), "width": torch.as_tensor(width)} for img in img_l)

        with torch.no_grad():
            preds = self._model(inputs)


        for pred, starting_sz, crop_coords in zip(preds, images_starting_size, images_crop_coords):
            scores = pred["scores"]
            boxes = pred["pred_boxes"]
            masks = pred["pred_masks"]

            masks = masks[scores>=treshold]
            boxes = boxes[scores>=treshold]
            scores = scores[scores>=treshold]

            img_preds:list[PredictionInstance] = []
            for box_t, score_t, mask_t in zip(boxes, scores, masks):
                mask = paste_mask_in_image_fast(mask_t, box_t, self._inference_size, self._inference_size)
                mask = mask.to("cpu").detach().numpy()

                box = box_t.to("cpu").detach().numpy()
                box = [int(x) for x in box]
                score = score_t.to("cpu").detach().item()

                mask_wrt_starting_img = np.zeros(starting_sz[:2], dtype=np.uint8)
                short_edge_sz = min(starting_sz[:2])
                mask_pred_resized = self._resize_square_img(mask, short_edge_sz, copy=False)

                l,r,t,b = crop_coords
                mask_wrt_starting_img[t:b,l:r] = mask_pred_resized

                #TODO: bnd box has to be adapated to original image size
                img_preds.append(PredictionInstance(score, [0,0,0,0], mask_wrt_starting_img))
            
            predictions.append(ImagePrediction(starting_sz, img_preds))

        return predictions




    def _get_crop_to_square_coordinates(self, img:cv.Mat):
        h, w = img.shape[:2]
        l=0
        r=w
        t=0
        b=h


        margin=abs(w-h)

        if not h==w:
            if w>h:
                l = margin // 2
                r = w-margin // 2
            else:
                t = margin // 2
                b = h-margin // 2

        return [l,r,t,b]

    def _crop_img(self, image:cv.Mat, crop_coordinates:list[int], copy:bool=True):
        img = image
        if copy:
            img = image.copy()

        l,r,t,b = crop_coordinates
        img = img[t:b, l:r]
        return img


    def _resize_square_img(self, image:cv.Mat, size_to:int, copy:bool = True):
        img = image
        if copy:
            img = image.copy()

        size_from = img.shape[0]

        if size_from == size_to: return img

        if size_from > size_to:
            interpolation=cv.INTER_AREA
        else:
            interpolation=cv.INTER_CUBIC
        
        img = cv.resize(img, dsize=(size_to, size_to), interpolation=interpolation)
        return img










