import os
import open3d as o3d
import numpy as np

import Global_Features
import Normalization
import twoDImage
import twoDImage as tdi
import twoDFeatures
import Global_Features as gf
import cv2 as cv
import copy


# This function is going to be used in roder to select the front view of the window.
def selectImages(image1, image2, image3):
    # todo: to be implemented
    return 0


# This file is going to combine all the functionalities related to the codebook
def createLocalFeatureSpace(inputcloud, outputDir, s1=0, s2=0, s3=0):
    ## Parameters to specify the used features in oder to use the respective feature to create the codebook set the
    # respective s-number to one
    # s1: ORB descriptors
    # s2: SIFT descriptors
    # s3: MSER

    # ATTENTION: It can only ONE descriptor bes selected at a time!

    # Step 1: Creation of the featureSpace .txt file

    dir = str(outputDir + 'LocalFeatureSpace.txt')
    f = open(dir,
             mode='a')
    # writing some sort of simple feature space coding into the first line for the FeatureSpace Document
    # f.write('%d' % s1 + '%d' % s2 + '%d' % s3 + "\n") # Todo: For-Schleife wieder hier einfügen und dann auskommentieren

    # Generate the 2d-images from the normalized point cloud
    #spec = tdi.calculateMaximumCovariance(inputcloud)
    image1 = tdi.create2DBinaryImage(inputcloud, 640, 640, 'xy')
    choice = tdi.manualSelection()
    if choice == "n":
        print("The previously shown image is not chosen")
        image1 = tdi.create2DBinaryImage(inputcloud, 640, 640, 'xz')
        choice = tdi.manualSelection()
        if choice == "n":
            print("The previously shown image is not chosen")
            image1 = tdi.create2DBinaryImage(inputcloud, 640, 640, 'yz')
    image1d = twoDFeatures.getDilation(image1, 20)
    image1dlx = twoDFeatures.getLaplace(image1d)
    print("Jetzt kommt DP!!!!!!!!!!!")
    image1dl = twoDImage.DouglasPeucker(image1dlx)

    if s1 == 1 or s2 == 1:
        if s1 == 1:
            orb_k1, orb_desc_1 = twoDFeatures.extractORB(image1dl)
        # Todo: dense feature sampling was added after the report submission
        if s2 == 1:
            orb_k1, orb_desc_1 = twoDFeatures.denseFeatureSampling(image1dl)

        try:
            for desc1 in orb_desc_1:
                counter = 0
                str1 = ""
                for i in desc1:
                    str1 = str1 + ('%17.5f' % desc1[counter] + ";")
                    counter = counter + 1
                f.write(str1 + "\n")
        except:
            x = 5

    f.close()
    return 0


# This function will be used in order to perform a clusterng in the featurespace that is spanned by the previously
# extracted local feature descriptors.
def clusteringInLocalFeatureSpace(featureSpaceMatrix, algSpec, k):
    # filename: specify the name of the local faturespace .txt file
    # algSpec:  specify which clustering algorithm should be used
    #           possible settings:
    #           algSpec = 'kMeans' makes use of the kMeans clustering Algorithm
    #           k = the number of cluster centers

    print("Testmatrix: ", featureSpaceMatrix.shape)
    # featureSpaceMatrixTransposed = np.transpose(featureSpaceMatrix)

    # perform the clustering algorithm
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv.kmeans(np.float32(featureSpaceMatrix), k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    # print("Center: ", center)

    return center


def createGlobalFeatureSpace(inputCloud, outputDir, s1=0, s2=0, s3=0, s4=0, s5=0, s6=0, s7=0, s8=0, s9=0, s10=0, s11=0,
                             s12=0, s13=0, s14=0,
                             s15=0,
                             s16=0, s17=0, s18=0, s19=0, s20=0, s21=0, s22=0, s23=0, s24=0, s25=0, s26=0, s27=0, s28=0, s29=0, s30=0, s31=0, s32=0, s33=0, s34=0, s35=0):
    # Parameters to specify the used features in oder to use the respective feature to create the codebook set the
    # respective s-number to one


    # Step 1: Creation of the featureSpace .txt file
    dir = str(outputDir + 'GlobalFeatureSpace.txt')
    f = open(dir,
             mode='a')

    # writing some sort of simple feature space coding into the first line for the FeatureSpace Document
    # f.write(
    #    '%d' % s1 + '%d' % s2 + '%d' % s3 + '%d' % s4 + '%d' % s5 + '%d' % s6 + '%d' % s7 + '%d' % s8 + '%d' % s9 + '%d' % s10 + '%d' % s11 + '%d' % s12 + '%d' % s13 + '%d' % s14 + '%d' % s15 + '%d' % s16 + '%d' % s17 + '%d' % s18 + '%d' % s19 + '%d' % s20 + '%d' % s21 + '%d' % s22 + '%d' % s23 + "\n")

    # Todo: For-Schleife wieder hier einfügen und dann auskommentieren
    featureVector = []

    # Distinguish the different cases
    if s1 == 1:
        vobb = gf.volumeOfBoundingBox(inputCloud)
        print("Volume of BoundingBox: ", vobb)
        featureVector.append(vobb)
    if s2 == 1:
        abb = gf.areaOfBoundingBox(inputCloud)
        print("Area of Bounding Box: ", abb)
        featureVector.append(abb)
    if s3 == 1:
        volch = gf.volumeOfConvexHull(inputCloud)
        print("Volume of Convex Hull: ", volch)
        featureVector.append(volch)
    if s4 == 1:
        aoch = gf.areaOfConvexHull(inputCloud)
        print("Area of Convex Hull: ", aoch)
        featureVector.append(aoch)
    if s5 == 1:
        bbAvRatio = gf.bbAVRatio(inputCloud)
        print("bbAVRatio: ", bbAvRatio)
        featureVector.append(bbAvRatio)
    if s6 == 1:
        bbCubeness = gf.bbCubeness(inputCloud)
        print("BBCubeness: ", bbCubeness)
        featureVector.append(bbCubeness)
    if s7 == 1:
        chAvRatio = gf.chAVRatio(inputCloud)
        print("CHAVRatio: ", chAvRatio)
        featureVector.append(chAvRatio)
    if s8 == 1:
        chCubeness = gf.chCubeness(inputCloud)
        print("chCubeness: ", chCubeness)
        featureVector.append(chCubeness)
    if s9 == 1:
        chCircularity = gf.chCircularity(inputCloud)
        print("chCircularity: ", chCircularity)
        featureVector.append(chCircularity)
    if s10 == 1:
        convexity = gf.convexity(inputCloud)
        print("Nicht implementiert")
        featureVector.append(convexity)
    if s11 == 1:
        chBb_VolumeRatio = gf.chBbAreaRatio(inputCloud)
        print("chBb_VolumeRatio: ", chBb_VolumeRatio)
        featureVector.append(chBb_VolumeRatio)
    if s12 == 1:
        chBb_AreaRatio = gf.chBbAreaRatio(inputCloud)
        print("chBb_AreaRatio: ", chBb_AreaRatio)
        featureVector.append(chBb_AreaRatio)
    if s13 == 1:
        fractalDimension = gf.fractalDimension(inputCloud)
        print("Fractal Dimension: ", fractalDimension)
        featureVector.append(fractalDimension)
    if s14 != 0:  # Todo: Muss noch implementiert werden
        print(0)

    if s15 == 1:
        cohesion = gf.cohesion(inputCloud)
        print("Cohesion: ", cohesion)
        featureVector.append(cohesion)

    if s33 == 1:
        square2d = Global_Features.squarenessCH2d(inputCloud)
        print("Squareness of the 2D projected Pointcloud: ", square2d)
        featureVector.append(square2d)

    if s34 == 1:
        circ2d = Global_Features.circularityCH2d(inputCloud)
        print("Circularity of the 2D projected Pointcloud: ", circ2d)
        featureVector.append(circ2d)

    if s35 == 1:
        bbchratt = Global_Features.bb2dch2dAreaRatio(inputCloud)
        print("Perimeter ratio of 2d Convex Hull and 2d Bounding Box: ", bbchratt)
        featureVector.append(bbchratt)

    if s15 == 1 or s16 == 1 or s17 == 1 or s18 == 1 or s19 == 1 or s20 == 1 or s21 == 1 or s22 == 1 or s23 == 1 or s24 == 1 or s25 == 1 or s27 == 1 or s28 == 1 or s29 == 1 or s30 == 1 or s31 == 1 or s32 == 1:
        image1 = tdi.create2DBinaryImage(inputCloud, 640, 640, 'xy')
        choice = tdi.manualSelection()
        if choice == "n":
            print("The previously shown image is not chosen")
            image1 = tdi.create2DBinaryImage(inputCloud, 640, 640, 'xz')
            choice = tdi.manualSelection()
            if choice == "n":
                print("The previously shown image is not chosen")
                image1 = tdi.create2DBinaryImage(inputCloud, 640, 640, 'yz')
        image1d = twoDFeatures.getDilation(image1, 20)
        image1dlx = twoDFeatures.getLaplace(image1d)
        print("Jetzt kommt DP!!!!!!!!!!!")
        image1dl = twoDImage.DouglasPeucker(image1dlx)


        if s16 == 1:
         # Todo: eventuell kann man auch das dilated / laplace image zur extraktion verwenden
         # Todo: dieses wäre jedoch dann weitaus weiter von der ursprünglichen Punktwolke entfernt
            momentsImage1dl = twoDFeatures.extractMoments(image1dl)

            # Todo: diese funktion muss noch dringend getestet werden
            for moment in momentsImage1dl:
                featureVector.append(moment)


        if s22 == 1:
            number_of_lines1 = twoDFeatures.countLines(image1dl)

            print("Number of Lines1: ", number_of_lines1)

            featureVector.append(number_of_lines1)

        if s23 == 1:

            hog_desc_1 = twoDFeatures.extractHOG(image1dl)

            for hog1 in hog_desc_1:
                featureVector.append(hog1)

        # Calculation of the circularity of the 2d-boundingBox
        if s24 == 1:
            image1dla = copy.deepcopy(image1dl)
            bb2dcirc = twoDFeatures.BB2dCircularity(image1dla)
            print("2D - Circularity of the Bounding Box: ",  bb2dcirc)
            featureVector.append(bb2dcirc)

        # Calculation of the squareness of the 2D-bounding box
        if s25 == 1:
            image1dlb = copy.deepcopy(image1dl)
            bb2dcube = twoDFeatures.BB2dSquareness(image1dlb)
            print("2D - Cubeness of the Bounding Box: ", bb2dcube)
            featureVector.append(bb2dcube)

        if s26 == 1:
            image1dlc = copy.deepcopy(image1dl)
            cohe2d = twoDFeatures.Cohesion2D(image1dlc)
            print("2D - Cohesion: ", cohe2d)
            featureVector.append(cohe2d)

        if s27 == 1:
            image1dd = copy.deepcopy(image1dl)
            noc = twoDFeatures.numberOfContours(image1dd)
            print("Number of contours: ", noc)
            featureVector.append(noc)

        if s28 == 1:
            image1dlg = copy.deepcopy(image1dl)
            ch2dsquare = twoDFeatures.ch2dSquareness(image1dlg)
            print("2d convex Hull squareness: ",  ch2dsquare)
            featureVector.append(ch2dsquare)

        if s29 == 1:
            image1dlh = copy.deepcopy(image1dl)
            ch2dbb2dAreaRat = twoDFeatures.ch2dbb2dAreaRatio(image1dlh)
            print("Area ratio of 2d convex hull and 2d bounding box", ch2dbb2dAreaRat)
            featureVector.append(ch2dbb2dAreaRat)

        if s30 == 1:
            image1dli = copy.deepcopy(image1dl)
            ch2dbb2dPerimeterRat = twoDFeatures.ch2dbb2dPerimeterRatio(image1dli)
            print("Perimeter ratio of 2d convex hull and 2d bounding box", ch2dbb2dPerimeterRat)
            featureVector.append(ch2dbb2dPerimeterRat)

        if s31 == 1:
            image1dlj = copy.deepcopy(image1dl)
            bbav = twoDFeatures.PARatioBB2d(image1dlj)
            print("Perimeter-area-ratio of the bounding box:  ", bbav)
            featureVector.append(bbav)

        if s32 == 1:
            image1dlk = copy.deepcopy(image1dl)
            chav = twoDFeatures.PARatioCH2d(image1dlk)
            print("Perimeter-area-ratio of the convex hull:  ", chav)
            featureVector.append(chav)

        # Writing the feature vector into the FeatureSpace File
    string2write = ''
    counter = 0
    for i in featureVector:
        if counter == 0:
            string2write = ('%20.12f' % i)
            counter = counter + 1
        else:
            string2write = string2write + ';' + ('%20.12f' % i)
            counter = counter + 1
        # print("String2Write: ", string2write)
    string2write = string2write + '\n'
    f.write(string2write)
    f.close()

    return 0


def generateCodebook(inputCloud, outputDir):
    # create a new .txt file that is going to codebook
    dir = str(outputDir + 'Codebook.txt')
    f = open(dir, mode='a')

    # extract lokal features and perform a clustering in the local featurespace to find the codewords
    # for the lokal features
    filename1 = outputDir + 'LocalFeatureSpace.txt'
    datei = open(filename1, 'r')
    #lengthOfFile = len(datei.readlines())
    featureSpaceMatrix = np.ones((1, 33))
    linecount = 0
    for zeile in datei:
        featurevector = np.ones((1, 33))
        print("Feature Vector: ", featurevector)
        temp = zeile.split(";")
        counter = 0
        for desc_elem in temp:
            try:
                desc_elem_float = float(desc_elem)
                print("desc_elem_float: ", desc_elem_float)
                featurevector[0, counter] = desc_elem_float
            except:
                continue
            counter = counter + 1
            print("Counter: ", counter)

        print(featurevector)
        featureSpaceMatrix = np.vstack((featureSpaceMatrix, featurevector))
        linecount = linecount+1
    if linecount > 0:
        print("This happens")
        centers = clusteringInLocalFeatureSpace(featureSpaceMatrix, 'kMeans', 25)
    else:
        centers = []
    print("Dies ist ein ganz doofer test", len(centers))
    # Write to the previously generated .txt file
    for center in centers:
        print(" Es klappt")
        for c_dim in center:
            string2write = str(str(c_dim) + ';')
            f.write(string2write)
        f.write("\n")
    f.close()
    return 0
