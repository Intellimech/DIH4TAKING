# DIH4TAKING
The code that can be found in this repository has been developed during the DIH4TAKING project. This is the solution proposed to achieve the goals of the Perception Pipeline: select an object in a bin for a Pick & Place task, starting from the PointCloud obtained by a 3D camera.
## Perception Pipeline ##
This pipeline is an end-to-end solution for:
* Training an A.I model capable of identifying and segmenting arbitrary objects in an 2D (color) image
* Segmenting the Pointcloud exploiting objects masks obtained from the A.I model
* Calculating objects center of mass and principal axes directions

The pipeline has been tested both on virtual scenes (Virtual 3D Camera) and real scenes (Real 3D Camera) 

### A.I Model ###
Starting from the pointcloud (RGB + DEPTH) coming from a 3D camera, a Neural Network is used to detect and segment objects in the color image. Then, objects masks are used to segment the depth map aligned with the RGB image. 

The model is trained using Detectron2 from Meta: https://github.com/facebookresearch/detectron2.

#### A.I Model Training ####
The model is trained on a dataset composed of synthetic images generated using Unity Perception Package: https://github.com/Unity-Technologies/com.unity.perception and following the Domain Randomization Methodology: https://arxiv.org/abs/1703.06907. The CAD model of the component is required



### Repository Description ###
In this repository it is possible to find three main scripts:

* train.py
  - For training the A.I model using Detectron 2 on the images generated with the Unity Perception Package. Real images can be used as well, as long as labels are in COCO format.
* export_by_scripting.py
  - For exporting the model to PyTorch
* inference.py
  - For the selection of the object to pick.

The script "inference.py" contains the algorithm for:
* Object segmentation
* Pointcloud segmentation
* Object position identification
* Object orientation identification (principal axes)
