import cv2 as cv
import numpy as np
import matplotlib as plt
from matplotlib import pyplot as plt2
from skimage.feature import hog
from skimage import data, exposure
import math


# Todo: This function is not used in the project
# This function is used in order to extract SIFT keypoints and the respective descriptor
# from an image given as an arbitrarily large
# numpy-array (8-bit integer). It makes use of the OpenCV Library
def extractSIFT(image):
    # print("Image: ", type(image[0, 0]))
    # Creation of a SIFT-Object
    sift = cv.SIFT_create()
    # of the Keypoints and the descriptors
    kp, des = sift.detectAndCompute(image, None)
    # print("kp: ", kp)
    # print("des: ", des)
    cv.imshow("An example image", image)
    cv.waitKey(0)
    img = cv.drawKeypoints(image, kp, image, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imshow("An example image", img)
    cv.waitKey(0)
    returnvalue = [kp, des]
    return returnvalue


# Todo: Discuss if this can be used or if it is unpractical --> Discuss this with Olaf
def extractMSER(image):
    vis = image.copy()
    mser = cv.MSER_create()
    # defining an MSER-object
    regions, _ = mser.detectRegions(image)
    for p in regions:
        # Finding the maximal indices in order to define the bounding box
        xmax, ymax = np.amax(p, axis=0)
        xmin, ymin = np.amin(p, axis=0)
        cv.rectangle(vis, (xmin, ymax), (xmax, ymin), (0, 255, 0), 1)
    cv.imshow("An example image", vis)
    cv.waitKey(0)
    print("regions: ", regions)
    print("_: ", _)
    return vis


# Todo: This function is not to be used in the project
# this Function is used to extract orb features and keypoints
def extractORB(image):
    # Initiate ORB detector
    orb = cv.ORB_create()
    # find the keypoints with ORB
    kp = orb.detect(image, None)
    # compute the descriptors with ORB
    kp, des = orb.compute(image, kp)
    #print("kp: ", kp)
    #print("des: ", des)

    returnvalue = [kp, des]
    cv.imshow("An example image", image)
    cv.waitKey(0)
    img2 = cv.drawKeypoints(image, kp, None, color=(0, 255, 0), flags=0)
    plt2.imshow(img2)
    plt2.show()
    print("Number of Keypoints: ", len(kp))
    return returnvalue

# Todo: This function was added after the report submission
def denseFeatureSampling(image):
    orb = cv.ORB_create()

    # Specify dense keypoints
    step_size = 10
    keypoints = []
    for x in range(0, image.shape[1], step_size):
        for y in range(0, image.shape[0], step_size):
            keypoints.append(cv.KeyPoint(x, y, step_size))

    # Compute descriptors at dense keypoints
    keypoints, descriptors = orb.compute(image, keypoints)

    # Draw keypoints on image
    img_with_keypoints = cv.drawKeypoints(image, keypoints, None, color=(0, 255, 0))

    # Display image with keypoints
    cv.imshow('Image with keypoints', img_with_keypoints)
    cv.waitKey(0)

    returnvalue = [keypoints, descriptors]
    return returnvalue
# Todo: End of the added function


# This function is used in order to calculate the statistical moments of an input image
def extractMoments(image):

    moments = cv.HuMoments(cv.moments(image)).flatten()
    print("Moments: ", moments)
    return moments


def extractHOG(image):
    print("Image type: ", type(image))
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(25, 25),
                        cells_per_block=(1, 1), visualize=True, feature_vector=True, channel_axis=None)
    print("length= ", len(fd))

    #cv.imshow('HOG image', hog_image)

    # Warten auf Benutzereingabe
    #cv.waitKey(0)

    # just for testing

    counter=0
    counter2=0
    for i in fd:
        if fd[counter] != 0:
            print("fd: ", fd[counter])
            counter2=counter2+1
    counter=counter+1
    print("Counter: ", counter2)
    print("hog_image: ", hog_image)
    fig, (ax1, ax2) = plt2.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    ax1.axis('off')
    ax1.imshow(image, cmap=plt2.cm.gray)
    ax1.set_title('Input image')
    #Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt2.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt2.show()
    return fd

# this function is going to be used in order to match the input image with a template and count the
# number of occurences as well as output the certainty of the matching
# Todo: Diese methode funktioniert nur sehr schlecht und wird daher nicht verwendet. Die Templates sind nicht Scale oder Rotationsinvariant
def matchCrossTemplate(image):
    # loading the template
    template = cv.imread('C:\\Users\\thoma\\Documents\\Master_GUG\\Project_Photogrammetry\\Templates\\cross.jpg', 0)
    w, h = template.shape[::-1]
    img = image
    img2 = img.copy()
    # All the 6 methods for comparison in a list
    methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
               'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
    for meth in methods:
        img = img2.copy()
        method = eval(meth)
        # Apply template Matching
        res = cv.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv.rectangle(img, top_left, bottom_right, 255, 2)
        plt2.subplot(121), plt2.imshow(res, cmap='gray')
        plt2.title('Matching Result'), plt2.xticks([]), plt2.yticks([])
        plt2.subplot(122), plt2.imshow(img, cmap='gray')
        plt2.title('Detected Point'), plt2.xticks([]), plt2.yticks([])
        plt2.suptitle(meth)
        plt2.show()

    return 0

def getDilation(image, kernelSize):
    kernel = np.ones((kernelSize, kernelSize), np.uint8)
    dilation = cv.dilate(image, kernel, iterations=1)

    #cv.imshow('Bild laplacian', dilation)

    # Warten auf Benutzereingabe
    #cv.waitKey(0)
    return dilation

def getLaplace(image):
    laplacian = np.asarray(cv.Laplacian(image, cv.CV_64F), dtype=np.uint8)


    cv.imshow('Bild laplacian', laplacian)

    # Warten auf Benutzereingabe
    cv.waitKey(0)

    return laplacian

# This function counts the straight lines that appear in an image. the Code was inspired by:
# https://stackoverflow.com/questions/60586283/how-to-count-lines-in-an-image-with-python-opencv
def countLines(image):
    # The application of these function was moved elsewhere!
    #dilated = getDilation(image, 12)
    laplacian = image

    try:
        # Todo: testing through different values
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 15  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 65  # minimum number of pixels making up a line
        max_line_gap = 30  # maximum gap in pixels between connectable line segments
        lines = cv.HoughLinesP(laplacian, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
        # calculate the distances between points (x1,y1), (x2,y2) :
        distance = []
        for line in lines:
            distance.append(np.linalg.norm(line[:, :2] - line[:, 2:]))
        # print('max distance:', max(distance), '\nmin distance:', min(distance))
        # Adjusting the best distance
        bestDistance = (max(distance) + min(distance)) / 2
        numberOfLines = []
        count = 0
        for x in distance:
            if x > bestDistance:
                numberOfLines.append(x)
                count = count + 1
                # print('Number of lines:', count)

        line_image = np.copy(laplacian) * 0  # creating a blank to draw lines on
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
         #Finally, draw the lines on your srcImage.
         #Draw the lines on the  image
        lines_edges = cv.addWeighted(laplacian, 0.8, line_image, 1, 0)
        cv.imshow('Detected lines', lines_edges)
        cv.waitKey(0)

        return count
    except:
        print("Error")
        return 0

# This function is used in order to calculate the 2d bounding box of theobject in the image
# this function only works when there is only one object in the image
# it is ensured, that it is the largest Contour that is being used.
def boundingBox2D(image):
    contours, hierarchy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    print("contours")
    dict = {}
    xval=[]
    yval=[]
    wval=[]
    hval=[]
    counter = 0
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        xval.append(x)
        yval.append(y)
        wval.append(w)
        hval.append(h)
        area = w * h
        dict[counter] = area
        counter = counter + 1

    max_key, max_val = max(dict.items(), key=lambda x: x[1])
    cv.rectangle(image, (xval[max_key], yval[max_key]), (xval[max_key] + wval[max_key], yval[max_key] + hval[max_key]), (255, 0, 0), 7)
    # Display the image with bounding boxes
    cv.imshow('Bounding Boxes', image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return [xval[max_key], yval[max_key], wval[max_key], hval[max_key]]

# this function is used in order to calculate the circularity of the bounding box
def BB2dCircularity(image):
    bb = boundingBox2D(image)
    width = bb[2]
    height = bb[3]
    area = width * height
    perimeter = 2 * (width + height)
    circularity = (4 * math.pi * area) / (perimeter ** 2)
    return circularity

# this function is used in order to calculate the squareness of the bounding box
def BB2dSquareness(image):
    bb = boundingBox2D(image)
    width = bb[2]
    height = bb[3]
    squareness = min(width, height) / max(width, height)
    return squareness

def get2dConvexHull(image):
    contours, hierarchy = cv.findContours(image, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)

    # find the longest contour
    dict = {}
    counter = 0
    for contour in contours:
        dict[counter] = len(contour)
        counter = counter + 1
    max_key, max_val = max(dict.items(), key=lambda x: x[1])

    # Calculate the convex hull
    convexHull = cv.convexHull(contours[max_key])

    return convexHull

def ch2dSquareness(image):

    # Get the convex hull
    convexHull = get2dConvexHull(image)

    # calculate the perimeter of the object's convex hull
    perimeter = cv.arcLength(convexHull, True)

    # Calculate the area of the object's convex hull
    area = cv.contourArea(convexHull)

    # Calculate the side length of an equal area square
    sidelength = math.sqrt(area)

    # calculate the perimeter of the equal area square
    perimeter_EASquare = 4 * sidelength

    # calculate the ratio
    squareness = perimeter_EASquare / perimeter

    return squareness

# Todo: Überlegen ob wirklich sinnvoll!!!
def ch2dbb2dAreaRatio(image):
    # Get the convex hull
    convexHull = get2dConvexHull(image)

    # Calculate the area of the object's convex hull
    CHarea = cv.contourArea(convexHull)

    # get the object's bounding box
    bb= boundingBox2D(image)

    # calculate the area of the boundingbox
    bbarea = bb[2] * bb[3]

    ratio = CHarea / bbarea

    return ratio

# Todo: Überlegen ob wirklich sinnvoll!!!
def ch2dbb2dPerimeterRatio(image):
    # Get the convex hull
    convexHull = get2dConvexHull(image)

    # calculate the perimeter of the object's convex hull
    perimeterCH = cv.arcLength(convexHull, True)

    # get the object's bounding box
    bb = boundingBox2D(image)

    # get the perimeter of the object's bounding box
    perimeterBB = (2 * bb[2]) + (2 * bb[3])

    # Calculate the ratio
    ratio = perimeterCH / perimeterBB

    return ratio

# This function is used to calculate the cohesion of the object in the 2d binary image
# Todo: unfortunately, the "cv.findcontour()" function is very unstable and absolutely not robust to noise
def Cohesion2D(image):

    # get the convex hull
    convexHull = get2dConvexHull(image)

    #Print the convex hull
    cv.drawContours(image, [convexHull], 0, (255, 0, 0), 6)
    cv.imshow('Convex Hull', image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Calculate the area of the object
    area = cv.contourArea(convexHull)

    # Calculate the perimeter of the object
    perimeter = cv.arcLength(convexHull, True)

    # Calculate the Area of a circle with the same perimeter
    radius = perimeter / (2 * math.pi)
    areaCircle = math.pi * math.pow(radius, 2)

    # Calculate the cohesion of the object as the reatio of the Area of the "same-perimeter-circle" with
    # the actual area of the object
    # Todo: hiermuss gecheckt werden ob der quotient richtig herum ist!
    cohesion = area / areaCircle
    return cohesion

# This function is used to count the number of contours that were found in the image
def numberOfContours(image):
    contours, hierarchy = cv.findContours(image, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    noc = len(contours)
    return noc

# this function is used in order to calculate the perimeter/area ratio of the object's bounding box
def PARatioBB2d(image):
    # get the bounding box
    bb = boundingBox2D(image)
    width = bb[2]
    height = bb[3]

    # Calculate the area of the bounding box
    areaBB = width * height

    # Calculate the perimeter of the bounding box
    perimeterBB = (2 * bb[2]) + (2 * bb[3])

    # Calculate the ratio
    ratio = perimeterBB / areaBB

    return ratio

# this function is used in order to calculate the perimeter/area ratio of the object's convex hull
def PARatioCH2d(image):
    # Todo: muss noch in die Codebook und Inference Pipeline eingefügt werden
    # Get the convex hull
    convexHull = get2dConvexHull(image)

    # calculate the perimeter of the object's convex hull
    perimeterCH = cv.arcLength(convexHull, True)

    # Calculate the area of the object
    areaCH = cv.contourArea(convexHull)

    ratio = perimeterCH / areaCH

    return ratio
