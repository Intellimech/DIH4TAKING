import os
import time
import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R


from utils.predictor import Predictor2D
from utils.pcd_handler import PointCloudHandler

print(os.getcwd())

model_path = r"model\model.ts"
inf_size = 1200
predictor = Predictor2D(model_path, inf_size)
predictor.load_model()

import os
import open3d as o3d



img_path = r"data\image.jpg"
pcd_path = r"data\pointcloud.pcd"

img = cv.imread(img_path)


pcd = o3d.io.read_point_cloud(pcd_path, remove_nan_points=False, format= "pcd")



height, width = img.shape[:2]
preds = predictor.predict([img], 0.8)

pcdh = PointCloudHandler()

for pred in preds:
    pcd_mask = np.zeros(pred.shape[:2], dtype=np.uint8)

    overlay = np.zeros(pred.shape, dtype=np.uint8)*255
    for instance in pred.instances:
        overlay[(instance.mask==1).nonzero()] = (0,255,0)

        pcd_mask = np.bitwise_or(pcd_mask, instance.mask)
        

        
    img = cv.addWeighted(img, 1, overlay, 0.2,0)

    cv.imwrite("output.jpg", img)
    cv.imwrite("mask.jpg", pcd_mask*255)


    t1 = time.perf_counter()
    points = np.asarray(pcd.points)
    points = points.reshape((height, width, 3))
    points[(pcd_mask==0).nonzero()] = (np.nan, np.nan, np.nan)



    points = points.reshape((-1, 3))
    elements_to_remove = np.all(np.bitwise_not(np.isnan(points)), axis=-1)
    points = points[elements_to_remove]



    pcd_points = np.asarray(pcd.points)
    pcd_points = pcd_points.reshape((height, width, 3))
    obj_pts = []
    obj_cogs = []
    obj_pca = []
    obj_rot = []

    for instance in pred.instances:
        _, _points, _ = pcdh.segment_planar_points(instance.mask, pcd_points)
        _obj_pca = pcdh.get_longitudinal_axis(_points)

        obj_pca.append(_obj_pca)
        obj_pts.append(_points)
        obj_cogs.append(pcdh.compute_obj_cog_from_points(_points))


    opcd_objs = o3d.geometry.PointCloud()

    opcd_objs.points = o3d.utility.Vector3dVector(np.vstack(obj_pts))

    closest_obj_idx = np.argmin(np.linalg.norm(np.vstack(obj_cogs), axis=1))

    spheres_geometry = o3d.geometry.TriangleMesh()
    spheres_geometry_closest = o3d.geometry.TriangleMesh()




    for idx, obj_cog in enumerate(obj_cogs):
        _components, _ = obj_pca[idx]

        chirality = np.dot(np.cross(_components[0], _components[1]), _components[2])
        if np.isclose(chirality, -1):
            _components[2]*=-1

        if _components[2][2] < 0:
            _components[1:] *= -1

        _components = _components / np.linalg.norm(_components, axis=1, keepdims=True)
        r = _components.transpose()

        rm = R.from_matrix(r)
        
        obj_rot.append(rm)



