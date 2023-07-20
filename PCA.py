# This file contains the PCA functionality that is used to get the
# principal axes (eigenvectors of the covarianc ematrix) and the
# respective scale factors (eigenvalues of the covariance matric)

# The function has to be given an n x 3 Array of xyz-Point data
import numpy as np


def performPCA(pointArray):
    length = len(pointArray[:, 0])
    # Calculation of the mean-values
    xmean = np.mean(pointArray[:, 0])
    ymean = np.mean(pointArray[:, 1])
    zmean = np.mean(pointArray[:, 2])
    # Calculation of the elements of the main diagonal
    dxx = pow(np.std(pointArray[:, 0]), 2)
    dyy = pow(np.std(pointArray[:, 1]), 2)
    dzz = pow(np.std(pointArray[:, 2]), 2)
    # Calculation of the off-diagonal elements
    dxy = 0
    dxz = 0
    dyz = 0
    c = 0
    for k in pointArray[:, 0]:
        dxy = dxy + ((pointArray[c, 0] - xmean) * ((pointArray[c, 1]) - ymean))
        dxz = dxz + ((pointArray[c, 0] - xmean) * ((pointArray[c, 2]) - zmean))
        dyz = dyz + ((pointArray[c, 1] - ymean) * ((pointArray[c, 2]) - zmean))
    # Setting up the covariance Matrix
    covarianceMatrix = np.ones((3, 3))
    covarianceMatrix[0, 0] = dxx
    covarianceMatrix[1, 1] = dyy
    covarianceMatrix[2, 2] = dzz
    covarianceMatrix[0, 1] = dxy / (length - 1)
    covarianceMatrix[0, 2] = dxz / (length - 1)
    covarianceMatrix[1, 0] = dxy / (length - 1)
    covarianceMatrix[2, 0] = dxz / (length - 1)
    covarianceMatrix[2, 1] = dyz / (length - 1)
    covarianceMatrix[1, 2] = dyz / (length - 1)
    print(covarianceMatrix)
    # Calculation of the Eigenvalues of the Covariancematrix
    eigenvalues, eigenvectors = np.linalg.eigh(covarianceMatrix)
    return eigenvalues, eigenvectors
