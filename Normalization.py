# This function is going to be used to
# normalize the point cloud derived from
# the CAD model.

import PCA
import numpy as np
from numpy.linalg import inv
import open3d as o3d


def normalize(pointArray):
    # this function is used to perform a pose normalization of the input
    # point data.
    # Step 1: Translation: move the whole point cloud to the
    #         Coordinates of the mean value
    # Calculation of the mean-values:
    xmean = np.mean(pointArray[:, 0])
    ymean = np.mean(pointArray[:, 1])
    zmean = np.mean(pointArray[:, 2])
    # print("Mean Point: ", xmean, " ", ymean, " ", zmean)
    # Performing the actual translation
    length = len(pointArray[:, 0])
    pointArrayTranslated = np.zeros((length, 3))

    a = 0
    for i in pointArray[:, 0]:
        pointArrayTranslated[a, 0] = pointArray[a, 0] - xmean
        pointArrayTranslated[a, 1] = pointArray[a, 1] - ymean
        pointArrayTranslated[a, 2] = pointArray[a, 2] - zmean
        a = a + 1

    # Step 3: Scale invariance
    # finding the indices of the maximal coordinate values of the point cloud in each direction in order to find
    # A factor to scale the point cloud's furthest point to a standard distance
    ind_maxValuesXYZ = np.zeros((1, 3))
    ind_maxValuesXYZ[0, 0] = np.argmax(np.absolute(pointArrayTranslated[:, 0]))
    ind_maxValuesXYZ[0, 1] = np.argmax(np.absolute(pointArrayTranslated[:, 1]))
    ind_maxValuesXYZ[0, 2] = np.argmax(np.absolute(pointArrayTranslated[:, 2]))
    # print("Maximal value indices: ", ind_maxValuesXYZ)
    # finding the actual values, and the largest one of these values
    a = int(ind_maxValuesXYZ[0, 0])
    b = int(ind_maxValuesXYZ[0, 1])
    c = int(ind_maxValuesXYZ[0, 2])
    maxValuesXYZ = np.ones((1, 3))
    maxValuesXYZ[0, 0] = pointArrayTranslated[a, 0]
    maxValuesXYZ[0, 1] = pointArrayTranslated[b, 1]
    maxValuesXYZ[0, 2] = pointArrayTranslated[c, 2]
    # Calculation of tha factor that the whole point cloud should be scaled with
    # so that scale invariance is achieved
    ind_maxMaxValuesXYZ = np.argmax(np.absolute(maxValuesXYZ))
    scalingFactor = 10 / (maxValuesXYZ[0, ind_maxMaxValuesXYZ])
    # print("Scaling factor: ", scalingFactor)
    # Applying of the scaling factor:
    pos = 0
    pointArrayTraRotSca = np.zeros((length, 3))
    # print("TEST: ", pointArrayTraRot)
    for m in pointArrayTranslated[:, 0]:
        pos2 = 0
        for n in pointArrayTranslated[0, :]:
            pointArrayTraRotSca[pos, pos2] = scalingFactor * pointArrayTranslated[pos, pos2]
            pos2 = pos2 + 1
        pos = pos + 1
    # write the nupy array to an o3d point cloud file
    # print("test: ", pointArrayTraRotSca)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointArrayTraRotSca)
    o3d.io.write_point_cloud("Test_normalized.ply", pcd)
    return pcd
