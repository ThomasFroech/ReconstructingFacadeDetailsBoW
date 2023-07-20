import open3d as o3d
import numpy as np


def applyArtificialNoise(input_points):

    noise_strength = 0.2

    noisy_points = np.asarray(input_points) + noise_strength * np.random.randn(*np.asarray(input_points).shape)

    noisy_pcd = o3d.geometry.PointCloud()
    noisy_pcd.points = o3d.utility.Vector3dVector(noisy_points)
    o3d.visualization.draw_geometries([noisy_pcd])

    return noisy_pcd

