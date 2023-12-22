from dataclasses import dataclass

import numpy as np
import open3d as o3d

from numpy import typing as npt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

@dataclass
class ObjectMask:
    mask:npt.NDArray #(H,W)

class PointCloudHandler:
    def __init__(self):
        pass

    def _get_inliers_from_mask(self, mask:npt.NDArray, pcd:npt.NDArray):
        '''
        mask: np.array - (H, W) or (H x W,)
        pcd: np.array - (H, W, 3[xyz]) or (HxW,3)
        '''
        inliers_idx = mask.nonzero()
        inliers = pcd[inliers_idx]
        return inliers_idx, inliers
    
    def compute_obj_cog_from_mask(self, mask:npt.NDArray, pcd:npt.NDArray):
        _, obj_pts = self._get_inliers_from_mask(mask, pcd)
        obj_cog = self.comput_obj_cog_from_points(obj_pts)

        return obj_cog
    
    def compute_obj_cog_from_points(self, obj_pts:npt.NDArray):
        obj_cog = obj_pts.mean(axis=0)
        return obj_cog

    

    # def segment_planar_points(self, masks:list[npt.NDArray], pcd:npt.NDArray):
    #     points_on_planes:list[npt.NDArray] = []
    #     opcd = o3d.geometry.PointCloud()
    #     for mask in masks:
    #         obj_pts_idx, obj_pts = self._get_inliers_from_mask(mask, pcd)
    #         obj_pts_linear = obj_pts.reshape(-1,3)
    #         opcd.points = o3d.utility.Vector3dVector(obj_pts_linear)
    #         _, inliers_idx = opcd.segment_plane(distance_threshold=0.001,
    #                                         ransac_n=10,
    #                                         num_iterations=1000)
            
    #     #     points_on_planes.extend(obj_pts_linear[inliers])
        
    #     # points_on_planes = list(np.unique(points_on_planes, axis=0))

    #     return points_on_planes
    

    def segment_planar_points(self, mask:npt.NDArray, pcd:npt.NDArray, try_to_reshape:bool=True):
        assert mask.ndim == 2
        pcd_matrix = pcd
        if pcd.ndim <=2:
            if try_to_reshape:
                pcd_matrix=pcd.reshape((mask.shape[:2], 3))
            else:
                assert pcd.ndim==3


        opcd = o3d.geometry.PointCloud()

        obj_pts_idx, obj_pts = self._get_inliers_from_mask(mask, pcd_matrix)
        opcd.points = o3d.utility.Vector3dVector(obj_pts)

        plane_model, obj_idx_seg = opcd.segment_plane(distance_threshold=0.001,
                                        ransac_n=10,
                                        num_iterations=1000)
        
        mask_refined = np.zeros_like(mask)
        mask_linear = np.ravel_multi_index(obj_pts_idx, mask.shape)[obj_idx_seg]
        mask_refined[np.unravel_index(mask_linear, mask.shape)] = 1
        return mask_refined, obj_pts[obj_idx_seg], plane_model
    

    def get_longitudinal_axis(self, obj_pts:npt.NDArray):
        scaler = MinMaxScaler()
        scaler.fit_transform(obj_pts)
        pca = PCA(n_components=3)
        # pca.fit(scaler.fit_transform(obj_pts))
        pca.fit(obj_pts)

        return pca.components_, pca.explained_variance_
    

        








        