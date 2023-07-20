# This Python File will contain code in order to create an image of the Pointcloud in order to
# extract features from this projected image
import numpy as np
from PIL import Image as im
import cv2


# This function creates a quadratic binary image from an input point cloud.
def create2DBinaryImage(InputCloud, numberOfColumns, numberOfRows, planeSpec):
    # Input Cloud: The Input point cloud
    # numberOfColumns: The number of pixels in the column direction (--> Axis 0)
    # numberOfRows: the number of pixels in the  row direction (--> Axis 1)
    # planeSpec: Specifies to which of the plane spanned by the objects
    # principal axes the point cloud should be projected (Point Cloud Normalization has to be performed beforehand)

    if planeSpec == 'xy' or planeSpec == 'yx':
        print("PlaneSpec: ", planeSpec)
        points = np.asarray(InputCloud.points)[:, 0:2]
        print("Points: ", points)
        # Step1 create a zeros Array that is as large as the resulting image should be
        startimage = np.zeros((numberOfRows + 150, numberOfColumns + 150), dtype=np.uint8)
        # it has to be this datatype so that the array can be accepted by the opencv funstion

        # Step 2 : Sample the projected point cloud
        # Step 2.1: Calculate the necessary sampling distance
        maxExtentX = 10 #max(abs(points[:, 0])) # Todo: Hier muss man mal schauen
        # print("Ind_maxExtentX: ", maxExtentX)
        maxExtentY = 10#max(abs(points[:, 1])) # Todo Hier muss man mal schauen
        # print("Ind_maxExtentY: ", maxExtentY)
        maxValues = [maxExtentX, maxExtentY]
        gesMax = max(maxValues)
        print("gesMax: ", gesMax)
        pixelSize = (2 * gesMax) / numberOfColumns
        print("Pixel Size: ", pixelSize)
        # Step 3 : Create a Grid in order to sample the projected point cloud
        grid = np.ceil(points / pixelSize)
        # print("Grid: ", grid)
        # Translate the matrix to the center so that there are only positive values
        shape = np.shape(grid)
        grid_trans = np.zeros(shape)
        counter1 = 0
        for i in grid[:, 0]:
            grid_trans[counter1, 0] = grid[counter1, 0] + (numberOfRows + 75) / 2
            counter1 = counter1 + 1
        counter2 = 0
        for j in grid[:, 1]:
            grid_trans[counter2, 1] = grid[counter2, 1] + (numberOfColumns+75) / 2
            counter2 = counter2 + 1
        print("grid max: ", max(grid_trans[:, 0]), "  ", max(grid_trans[:, 1]))
        print("grid min: ", min(grid_trans[:, 0]), "  ", min(grid_trans[:, 1]))
        print("Grid_Trans: ", grid_trans)
        # Assign the values to the respective raster cells
        counter = 0
        for i in grid_trans[:, 0]:
            a = int(grid_trans[counter, 0])
            # print("a: ", a)
            b = int(grid_trans[counter, 1])
            # print("b: ", b)
            startimage[a + 20, b + 20] = 250
            # print("yoo")
            counter = counter + 1
        # write the image to a test file in order to inspect the result
        data = im.fromarray(startimage)
        data.show()

    elif planeSpec == 'xz' or planeSpec == 'zx':
        print("PlaneSpec: ", planeSpec)
        points1 = np.asarray(InputCloud.points)[:, 0]
        # print("Points 1: ", points1)
        points2 = np.asarray(InputCloud.points)[:, 2]
        # print("points 2: ", points2)
        counter = 0
        points = np.ones((len(points1), 2))
        for i in points1:
            points[counter, 0] = points1[counter]
            points[counter, 1] = points2[counter]
            counter = counter + 1
        # Step1 create a zeros Array that is as large as the resulting image should be
        startimage = np.zeros((numberOfRows + 150, numberOfColumns + 150), dtype=np.uint8)
        # Step 2 : Sample the projected point cloud
        # Step 2.1: Calculate the necessary sampling distance
        maxExtentX = 10 #max(abs(points[:, 0])) # Todo Hier muss man mal schauen
        # print("Ind_maxExtentX: ", maxExtentX)
        maxExtentY = 10 #max(abs(points[:, 1])) #Todo Hier muss man mal schauen
        # print("Ind_maxExtentY: ", maxExtentY)
        maxValues = [maxExtentX, maxExtentY]
        gesMax = max(maxValues)
        print("gesMax: ", gesMax)
        pixelSize = (2 * gesMax) / numberOfColumns
        print("Pixel Size: ", pixelSize)
        # Step 3 : Create a Grid in order to sample the projected point cloud
        grid = np.ceil(points / pixelSize)
        # print("Grid: ", grid)
        # Translate the matrix to the center so that there are only positive values
        shape = np.shape(grid)
        grid_trans = np.zeros(shape)
        counter1 = 0
        for i in grid[:, 0]:
            grid_trans[counter1, 0] = grid[counter1, 0] + (numberOfRows + 75) / 2
            counter1 = counter1 + 1
        counter2 = 0
        for j in grid[:, 1]:
            grid_trans[counter2, 1] = grid[counter2, 1] + (numberOfColumns + 75) / 2
            counter2 = counter2 + 1
        print("grid max: ", max(grid_trans[:, 0]), "  ", max(grid_trans[:, 1]))
        print("grid min: ", min(grid_trans[:, 0]), "  ", min(grid_trans[:, 1]))
        print("Grid_Trans: ", grid_trans)
        # Assign the values to the respective raster cells
        counter = 0
        for i in grid_trans[:, 0]:
            a = int(grid_trans[counter, 0])
            # print("a: ", a)
            b = int(grid_trans[counter, 1])
            # print("b: ", b)
            startimage[a + 20, b + 20] = 250
            # print("yoo")
            counter = counter + 1
        # write the image to a test file in order to inspect the result
        data = im.fromarray(startimage)
        data.show()
    elif planeSpec == 'yz' or planeSpec == 'zy':
        print("PlaneSpec: ", planeSpec)
        points = np.asarray(InputCloud.points)[:, 1:]
        print("Points: ", points)
        # Step1 create a zeros Array that is as large as the resulting image should be
        startimage = np.zeros((numberOfRows + 150, numberOfColumns + 150), dtype=np.uint8)
        # Step 2 : Sample the projected point cloud
        # Step 2.1: Calculate the necessary sampling distance
        maxExtentX = 10#max(abs(points[:, 0])) # Todo: hier muss man mal schauen
        # print("Ind_maxExtentX: ", maxExtentX)
        maxExtentY = 10#max(abs(points[:, 1]))# Todo: hier muss man mal schauen
        # print("Ind_maxExtentY: ", maxExtentY)
        maxValues = [maxExtentX, maxExtentY]
        gesMax = max(maxValues)
        print("gesMax: ", gesMax)
        pixelSize = (2 * gesMax) / numberOfColumns
        print("Pixel Size: ", pixelSize)
        # Step 3 : Create a Grid in order to sample the projected point cloud
        grid = np.ceil(points / pixelSize)
        # print("Grid: ", grid)
        # Translate the matrix to the center so that there are only positive values
        shape = np.shape(grid)
        grid_trans = np.zeros(shape)
        counter1 = 0
        for i in grid[:, 0]:
            grid_trans[counter1, 0] = grid[counter1, 0] + (numberOfRows+75) / 2
            counter1 = counter1 + 1
        counter2 = 0
        for j in grid[:, 1]:
            grid_trans[counter2, 1] = grid[counter2, 1] + (numberOfColumns+75
                                                           ) / 2
            counter2 = counter2 + 1
        print("grid max: ", max(grid_trans[:, 0]), "  ", max(grid_trans[:, 1]))
        print("grid min: ", min(grid_trans[:, 0]), "  ", min(grid_trans[:, 1]))
        print("Grid_Trans: ", grid_trans)
        # Assign the values to the respective raster cells
        counter = 0
        for i in grid_trans[:, 0]:
            a = int(grid_trans[counter, 0])
            # print("a: ", a)
            b = int(grid_trans[counter, 1])
            # print("b: ", b)
            startimage[a + 20, b + 20] = int(250)
            # print("yoo")
            counter = counter + 1
        # write the image to a test file in order to inspect the result
        data = im.fromarray(startimage)  # Todo: maybe this step is not necessary after all
        data.show()  # Just for making a small visualization in order to check for correctness
    else:
        print("Please select a valid plane to which the Pointcloud shall be projected")

    return startimage


# This part of the code is just for choosing the right projection in order to find the
# front view of the window's point cloud
def calculate2DCovarianceMatrix(inputPoints):
    n = len(inputPoints[:, 0])  # the number of points
    # Compute the mean of each dimension
    mean_x_step1 = np.mean(inputPoints[:, 0])
    mean_y_step1 = np.mean(inputPoints[:, 1])
    # Compute the covariance matrix
    cov_matrix = np.zeros((2, 2))
    for i in range(n):
        x_dev = inputPoints[i, 0] - mean_x_step1
        y_dev = inputPoints[i, 1] - mean_y_step1
        cov_matrix[0, 0] += x_dev * x_dev
        cov_matrix[1, 1] += y_dev * y_dev
        cov_matrix[0, 1] += x_dev * y_dev
    cov_matrix[1, 0] = cov_matrix[0, 1]
    cov_matrix /= (n - 1)
    return cov_matrix

# This code is used in order to find the view of the pointcloud that yields the maximum covariance.
# Unfortunately, the View with the maximum covariance is not always the front view
def calculateMaximumCovariance(InputCloud):
    # setting up an empty list in order to save the calculated covariances for the 3 projections
    covresult = []
    # this function is used in order to calculate the covariance of th projected pointcloud
    spec = ''
    # Step 1: Claculate the covariance matrix for the pointcloud projected in xy
    points_step1 = np.asarray(InputCloud.points)[:, 0:2]
    # calculate the covariance matrix
    cov_matrix_step1 = calculate2DCovarianceMatrix(points_step1)
    covresult.append(cov_matrix_step1)
    # Step 2: Claculate the covariance matrix for the pointcloud projected in xz
    points1 = np.asarray(InputCloud.points)[:, 0]
    # print("Points 1: ", points1)
    points2 = np.asarray(InputCloud.points)[:, 2]
    # print("points 2: ", points2)
    counter = 0
    points_step2 = np.ones((len(points1), 2))
    for i in points1:
        points_step2[counter, 0] = points1[counter]
        points_step2[counter, 1] = points2[counter]
        counter = counter + 1
    # calculate the covariance matrix
    cov_matrix_step2 = calculate2DCovarianceMatrix(points_step2)
    covresult.append(cov_matrix_step2)
    # Step 3: Calculate the covariance matrix for the pointcloud projected in yz
    points_step3 = np.asarray(InputCloud.points)[:, 1:]
    # Calculate the covariance matrix
    cov_matrix_step3 = calculate2DCovarianceMatrix(points_step3)
    covresult.append(cov_matrix_step3)
    covarianceValues=[]
    for covmat in covresult:
        covarianceValue = abs((covmat[0, 1] + covmat[1, 0]) / 2)
        covarianceValues.append(covarianceValue)
    # Step 4: find out which of the projections yields the highest covariance
    max_index = covarianceValues.index(max(covarianceValues))
    # n order to do so , the covariance matrix with the highes off diagonal elements must be found
    # Step 4.1: Iterate through all the elements in the convresult
    if max_index == 0:
        spec = 'xy'
    elif max_index ==1:
        spec = 'xz'
    elif max_index == 2:
        spec = 'zy'

    return spec

def manualSelection():
    choice = input("Enter some text: ")
    return choice

# Todo: This Code was written with th Help of ChatGPT
def DouglasPeucker(image):

    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Set the tolerance for the Douglas-Peucker algorithm
    # Todo: This parameter can be adjusted in order to define the abount of smoothness
    tolerance = 7

    # Iterate over each contour and simplify it using the Douglas-Peucker algorithm
    simplified_contours = []
    for contour in contours:
        simplified_contour = cv2.approxPolyDP(contour, tolerance, True)
        simplified_contours.append(simplified_contour)

    # Create a new image with the same dimensions as the original binary image
    output_image = np.zeros(image.shape, dtype=np.uint8)
    # Draw the simplified contours on the output image
    cv2.drawContours(output_image, simplified_contours, -1, (255, 0, 150), 1)

    # Display the output image
    cv2.imshow('Simplified image', output_image)
    cv2.waitKey(0)

    return output_image