# DIH4TAKING
The code that can be found in this repository has been developed during the DIH4TAKING project.

A Neural Network is used to detect and segment objects starting from a RBG image. Objects mask are used to segment a depth map aligned with the RGB image. The pointcloud (RGB + DEPTH) comes from a 3D camera.
The model is trained Detectron2: https://github.com/facebookresearch/detectron2 

The best object to pick is chosen among the one identified by the model (neural network)


In this repository it is possible to find three main scripts:

train.py for training a semantic segmentation model based on the detectron 2 framework
export_by_scripting.py for converting the model to a format that can be read natively by PyTorch
inference.py for the identification of the best object to pick. Object center of mass and main axes directions will be extracted
