# this python file is going to contain all the functionalities that are
# necessary for the inference of the code
import Global_Features as gf
import twoDImage
import twoDImage as tdi
import twoDFeatures
import numpy as np
import math
import cv2 as cv
import copy



def representObject(filename, inputCloud, outputdirectory, repName, s1=0, s2=0, s3=0, s4=0, s5=0, s6=0, s7=0, s8=0,
                    s9=0, s10=0,
                    s11=0,
                    s12=0, s13=0, s14=0,
                    s15=0,
                    s16=0, s17=0, s18=0, s19=0, s20=0, s21=0, s22=0, s23=0, s24=0, s25=0, s26=0, s27=0, s28=0, s29=0, s30=0, s31=0, s32=0, s33=0, s34=0, s35=0, sl1=0, sl2=0, sl3=0):
    # This function is going to be used in order to represent an input object by means of the
    # Codebook that was previously generated.

    # Step 1 extracting the features from the object
    # Step 1.1 Extracting the global features
    # Distinguish the different cases Todo: This Code is similar to the code found in the 'Codebook.py' file eventuell
    #                                 Todo: kann dieser mit einem zusätzlichen parameter "Writefile = False" noch
    #                                 Todo: in die Codebook Construction umgebettet werden
    globalfeatureVector = []
    if s1 == 1:
        vobb = gf.volumeOfBoundingBox(inputCloud)
        print("Volume of BoundingBox: ", vobb)
        globalfeatureVector.append(vobb)
    if s2 == 1:
        abb = gf.areaOfBoundingBox(inputCloud)
        print("Area of Bounding Box: ", abb)
        globalfeatureVector.append(abb)
    if s3 == 1:
        volch = gf.volumeOfConvexHull(inputCloud)
        print("Volume of Convex Hull: ", volch)
        globalfeatureVector.append(volch)
    if s4 == 1:
        aoch = gf.areaOfConvexHull(inputCloud)
        print("Area of Convex Hull: ", aoch)
        globalfeatureVector.append(aoch)
    if s5 == 1:
        bbAvRatio = gf.bbAVRatio(inputCloud)
        print("bbAVRatio: ", bbAvRatio)
        globalfeatureVector.append(bbAvRatio)
    if s6 == 1:
        bbCubeness = gf.bbCubeness(inputCloud)
        print("BBCubeness: ", bbCubeness)
        globalfeatureVector.append(bbCubeness)
    if s7 == 1:
        chAvRatio = gf.chAVRatio(inputCloud)
        print("CHAVRatio: ", chAvRatio)
        globalfeatureVector.append(chAvRatio)
    if s8 == 1:
        chCubeness = gf.chCubeness(inputCloud)
        print("chCubeness: ", chCubeness)
        globalfeatureVector.append(chCubeness)
    if s9 == 1:
        chCircularity = gf.chCircularity(inputCloud)
        print("chCircularity: ", chCircularity)
        globalfeatureVector.append(chCircularity)
    if s10 == 1:
        convexity = gf.convexity(inputCloud)
        print("Nicht implementiert")
        globalfeatureVector.append(convexity)
    if s11 == 1:
        chBb_VolumeRatio = gf.chBbAreaRatio(inputCloud)
        print("chBb_VolumeRatio: ", chBb_VolumeRatio)
        globalfeatureVector.append(chBb_VolumeRatio)
    if s12 == 1:
        chBb_AreaRatio = gf.chBbAreaRatio(inputCloud)
        print("chBb_AreaRatio: ", chBb_AreaRatio)
        globalfeatureVector.append(chBb_AreaRatio)
    if s13 == 1:
        fractalDimension = gf.fractalDimension(inputCloud)
        print("Fractal Dimension: ", fractalDimension)
        globalfeatureVector.append(fractalDimension)
    if s14 != 0:  # Todo: Muss noch implementiert werden
        print(0)

    if s24 == 1:
        cohesion = gf.cohesion(inputCloud)
        print("Cohesion: ", cohesion)
        globalfeatureVector.append(cohesion)

    if s33 == 1:
        square2d = gf.squarenessCH2d(inputCloud)
        print("Squareness of the 2D projected Pointcloud: ", square2d)
        globalfeatureVector.append(square2d)

    if s34 == 1:
        circ2d = gf.circularityCH2d(inputCloud)
        print("Circularity of the 2D projected Pointcloud: ", circ2d)
        globalfeatureVector.append(circ2d)

    if s35 == 1:
        bbchratt = gf.bb2dch2dAreaRatio(inputCloud)
        print("Perimeter ratio of 2d Convex Hull and 2d Bounding Box: ", bbchratt)
        globalfeatureVector.append(bbchratt)

    if s15 == 1 or s16 == 1 or s17 == 1 or s18 == 1 or s19 == 1 or s20 == 1 or s21 == 1 or s22 == 1 or s23 == 1 or s25 == 1 or s26==1 or s27==1 or s28 == 1 or s29 == 1 or s30 == 1 or s31 == 1 or s32 == 1:
        # Todo hier findet ein test statt
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

        # Extraction of the statistical moments of the image
        if s16 == 1:
            # Todo: eventuell kann man auch das dilated / laplace image zur extraktion verwenden
            # Todo: dieses wäre jedoch dann weitaus weiter von der ursprünglichen Punktwolke entfernt
            momentsImage1dl = twoDFeatures.extractMoments(image1dl)

            # Todo: diese funktion muss noch dringend getestet werden
            for moment in momentsImage1dl:
                globalfeatureVector.append(moment)

        # extraction of the number of lines in the image
        if s22 == 1:
            number_of_lines1 = twoDFeatures.countLines(image1dl)
            print("Number of Lines1: ", number_of_lines1)
            globalfeatureVector.append(number_of_lines1)

        # extraction of the HOG feautres of the image
        if s23 == 1:
            hog_desc_1 = twoDFeatures.extractHOG(image1dl)

            for hog1 in hog_desc_1:
                globalfeatureVector.append(hog1)

        if s24 == 1:
            image1dla = copy.deepcopy(image1dl)
            bb2dcirc = twoDFeatures.BB2dCircularity(image1dla)
            globalfeatureVector.append(bb2dcirc)

        # Calculation of the squareness of the 2D-bounding box
        if s25 == 1:
            image1dlb = copy.deepcopy(image1dl)
            bb2dcube = twoDFeatures.BB2dSquareness(image1dlb)
            globalfeatureVector.append(bb2dcube)

        if s26 == 1:
            image1dlc = copy.deepcopy(image1dl)
            cohe2d = twoDFeatures.Cohesion2D(image1dlc)
            globalfeatureVector.append(cohe2d)

        if s27 == 1:
            image1dd = copy.deepcopy(image1dl)
            noc = twoDFeatures.numberOfContours(image1dd)
            globalfeatureVector.append(noc)


        if s28 == 1:
            image1dlg = copy.deepcopy(image1dl)
            ch2dsquare = twoDFeatures.ch2dSquareness(image1dlg)
            print("2d convex Hull squareness: ", ch2dsquare)
            globalfeatureVector.append(ch2dsquare)

        if s29 == 1:
            image1dlh = copy.deepcopy(image1dl)
            ch2dbb2dAreaRat = twoDFeatures.ch2dbb2dAreaRatio(image1dlh)
            print("Area ratio of 2d convex hull and 2d bounding box", ch2dbb2dAreaRat)
            globalfeatureVector.append(ch2dbb2dAreaRat)

        if s30 == 1:
            image1dli = copy.deepcopy(image1dl)
            ch2dbb2dPerimeterRat = twoDFeatures.ch2dbb2dPerimeterRatio(image1dli)
            print("Perimeter ratio of 2d convex hull and 2d bounding box", ch2dbb2dPerimeterRat)
            globalfeatureVector.append(ch2dbb2dPerimeterRat)
        if s31 == 1:
            image1dlj = copy.deepcopy(image1dl)
            bbav = twoDFeatures.PARatioBB2d(image1dlj)
            print("Perimeter-area-ratio of the bounding box:  ", bbav)
            globalfeatureVector.append(bbav)

        if s32 == 1:
            image1dlk = copy.deepcopy(image1dl)
            chav = twoDFeatures.PARatioCH2d(image1dlk)
            print("Perimeter-area-ratio of the convex hull:  ", chav)
            globalfeatureVector.append(chav)


    print(len(globalfeatureVector))
    # Step 1.2: Extraction of the local features:
    localFeatureMatrix = np.ones((1, 33))
    if sl1 == 1 or sl2 == 1:
        # Todo hier findet ein test statt, leider ist allem anschein nach die Perspektive mit der höchsten
        # todo: Kovarianz nicht immer die Fronotansicht des jwewiligen Objektes
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
        if sl1 == 1:
            orb_k1, orb_desc_1 = twoDFeatures.extractORB(image1dl)
        if sl2 == 1:
            orb_k1, orb_desc_1 = twoDFeatures.denseFeatureSampling(image1dl)

        # Step 2: performing the vector quantization for the local features
        quantizationResultComplete = np.ones((1, 2))
        if sl1 == 1 or sl2 == 1 or sl3 == 1:  # Todo: diese If bedingung muss nochmal gecheckt werden!!
            try:
                for i in orb_desc_1:
                    quantization_result = vectorQuantization(i, outputdirectory)
                    quantizationResultComplete = np.vstack((quantizationResultComplete, quantization_result))
                print("Complete Quantization result: ", quantizationResultComplete)
            except:
                print("No feature found")
    #    try:
    #        for j in orb_desc_2:
    #            quantization_result = vectorQuantization(j, outputdirectory)
    #            quantizationResultComplete = np.vstack((quantizationResultComplete, quantization_result))
    #        print("Complete Quantization result: ", quantizationResultComplete)
    #    except:
    #        print("No feature found")
    #    try:
    #        for k in orb_desc_3:
    #            quantization_result = vectorQuantization(k, outputdirectory)
    #            quantizationResultComplete = np.vstack((quantizationResultComplete, quantization_result))
    #        print("Complete Quantization result: ", quantizationResultComplete)
    #    except:
    #        print("No feature found")
    # Todo: hier muss noch geprüft werden ob eine if-Bendingung notwendig ist oder nicht!
    # Step 3: Count the occurences of each "word" and create the histogram of word occurences
    # Step 3.1 generate the histogram first for the local features
        histogram_tmp_ges = np.histogram(quantizationResultComplete[:, 0], bins=range(26))
        histogram_tmp = histogram_tmp_ges[0]
        print("Histogram: ", histogram_tmp)
    if sl1 == 0:
        histogram_tmp = []
    # Step 3.2 append the global features to the histogram
    for i in range(len(globalfeatureVector)):
        histogram_tmp = np.hstack((histogram_tmp, globalfeatureVector[i]))

    # Step 4 creating a .txt file where the represented object is stored
    dir = str(outputdirectory + repName)
    f = open(dir, mode='a')
    f.write(filename + ':')
    for elem in histogram_tmp:
        f.write(str(elem) + ';')
    f.write('\n')
    return 0


def vectorQuantization(inputVector, directory):
    # this code is going to be used in order to implement a vector Quantization
    # it compares the euclidean distance of the input vector to all the vectors
    # that are contained in the 'Codebook.txt' file in th directory. The output
    # is going to be the vector that is closest to the input vector.

    # Step 1: reading the classes from the codebook and put all the classes into a codeword matrix
    filename1 = directory + 'Codebook.txt'
    datei = open(filename1, 'r')
    codewordMatrix = np.ones((1, 32))
    for line in datei:
        codewordVector = np.ones((1, 32))
        splittedLine = line.split(";")
        counter = 0
        for elem in splittedLine:
            try:
                desc_elem_float = float(elem)
                # print("desc_elem_float: ", desc_elem_float)
                codewordVector[0, counter] = desc_elem_float
            except:
                continue
            counter = counter + 1
        print("feature Vector: ", codewordVector)
        codewordMatrix = np.vstack((codewordMatrix, codewordVector))
    codewordMatrix = codewordMatrix[1:, :]
    print("Codewordmatrix", codewordMatrix.shape)
    # Creating a matrix (number of codewords , 2) that is going to hold the indices to the codewords
    # and the respective euclidean distance between the codeword to the vector that shall be quantized
    index_Matrix = np.ones((len(codewordMatrix[:, 1]), 2))
    print("IndexMartix Shape: ", index_Matrix.shape)
    # Iterate through all the codewords that were read frome the 'Codebook.txt' file
    index = 0  # Todo: the indexing is probably not necessary and can be omitted
    for codewordcounter in range(len(codewordMatrix[:, 1])):
        print("Codewordcounter: ", codewordcounter)
        # Calculate the euclidean distance between the vector that should be quantized
        # and each of the codewords. the distance is then stored in a respective matrix
        sum_tmp = 0
        for dim in range(len(codewordMatrix[codewordcounter,
                             :]) - 1):
            print("dim: ", dim)
            print("Inputvector: ", inputVector.shape)
            sum_temp = sum_tmp + pow(codewordMatrix[codewordcounter, :][dim] - inputVector[dim], 2)
        index_Matrix[index, 0] = index
        euclidean_distance = math.sqrt(sum_temp)
        print("Euclidean Distance: ", euclidean_distance)
        index_Matrix[index, 1] = euclidean_distance
        index = index + 1
    # find value & index with the smallest euclidean distance
    minimum = np.argmin(abs(index_Matrix[:, 1]))
    print("IndexMatrix: ", index_Matrix)
    print("Minimum: ", minimum)
    return index_Matrix[minimum, :]


def histogramComparison(h1, h2, dist, w1=0, w2=0, w3=0, w4=0):
    # this function will be used in order to compare two Histograms and return the respective
    # histogranm distance
    # Input parameters:
    #                   h1: the first histogram
    #                   h2: the second histogram
    #                   dist: the used histogram distance
    #                           possible choices:  - Minkowski Distance p=2
    #                                              - Chi-Square Distance
    #                                              - Kullback Leibler Divergence
    #                                              - Jensen Shannon Divergence
    #                                              - Earth Movers Distance
    #                                              - Combined

    def HistogramNormalization(hg1, hg2):
        # Todo: muss noch implementiert werden
        h1n = 0
        h2n = 0
        return h1n, h2n

    # This is just a small helper function that calculates the entropy as an intermediate step during
    # the calculation of the Jensen-Shannon Divergence
    def entropy(h, m):
        h = h / np.sum(h)
        m = m / np.sum(m)
        entropy = np.sum(h * np.log2(h / m))
        return entropy

    # Function to calculate the Minkowski Distance between two given Histograms
    def MinkowskiDistance(hg1, hg2):
        sum = 0
        for i in range(len(h1)):
            sum += math.pow(abs(hg1[i] - hg2[i]), 2)
        distance = math.sqrt(sum)
        return distance

    # Function to calculate the Chi-Square distance between two given Histograms
    def ChiSquareDistance(hg1, hg2):
        hg1a = hg1.astype(np.float32)
        hg2a = hg2.astype(np.float32)
        distance = cv.compareHist(hg1a, hg2a, cv.HISTCMP_CHISQR)
        return distance

    # Function to calculate the Kullback-Leibler divergence for two Histograms
    def KullbackLeiblerDivergence(hg1, hg2):
        hg1a = np.asarray(hg1, dtype=np.float)
        hg2a = np.asarray(hg2, dtype=np.float)
        # For loop for smoothening to ensure robustness
        for i in range(len(hg1)):
            if hg1a[i] == 0:
                hg1a[i] = 1e-8
            if hg2a[i] == 0:
                hg2a[i] = 1e-8
        hg1a /= np.sum(hg1a)
        hg2a /= np.sum(hg2a)
        distance = np.sum(np.where(hg1a != 0, hg1a * np.log(hg1a / hg2a), 0))
        return distance

    # Function to calculate the Jensen-Shannon divergence for two given histograms
    def JensenShannonDivergence(hg1, hg2):
        hg1a = np.asarray(hg1, dtype=np.float)
        hg2a = np.asarray(hg2, dtype=np.float)
        # For loop for smoothening to ensure robustness
        for i in range(len(hg1a)):
            if hg1a[i] == 0:
                hg1a[i] = 1e-8
            if hg2a[i] == 0:
                hg2a[i] = 1e-8
        m = 0.5 * (hg1a + hg2a)
        distance = 0.5 * (entropy(hg1a, m) + entropy(hg2a, m))
        return distance

    # Distinguishing the different user's choices
    if dist == 'Minkowski Distance p=2':
        print('Choice: Minkowski Distance p=2')
        finalDistance = MinkowskiDistance(h1, h2)

    elif dist == 'Chi-Square Distance':
        print('Choice: Chi-Square Distance')
        finalDistance = ChiSquareDistance(h1, h2)

    elif dist == 'Kullback Leibler Divergence':
        print('Choice: Kullback Leibler Divergence')
        finalDistance = KullbackLeiblerDivergence(h1, h2)

    elif dist == 'Jensen Shannon Divergence':
        print('Choise: Jensen Shannon Divergence')
        finalDistance = JensenShannonDivergence(h1, h2)

    elif dist == 'Combined':
        d1 = MinkowskiDistance(h1, h2)
        d2 = ChiSquareDistance(h1, h2)
        d3 = KullbackLeiblerDivergence(h1, h2)
        d4 = JensenShannonDivergence(h1, h2)
        # Calculation of the final distance as a weighted average of all the distances
        finalDistance = ((w1 * d1) + (w2 * d2) + (w3 * d3) + (w4 * d4)) / 4

    else:
        print("Please select a valid histogram distance!")
        finalDistance = 0

    return finalDistance


def infere(outputDir, threshold):
    # Open the inference file
    infFile = r'C:\Users\thoma\Documents\Master_GUG\Project_Photogrammetry\Code_Output\representation_Inference.txt'
    datei2 = open(infFile, 'r')
    # Compare the histogram to all the others from the file
    dir = str(outputDir + "Resultat.txt")
    f = open(dir, mode='a')

    counter = 0
    for zeile1 in datei2:
        dict = {}
        parts1 = zeile1.split(':')
        parts1split = parts1[1].split(';')
        parts1split.pop()
        histogramMod = np.asarray(np.float32(parts1split))
        counter2 = 0
        modFile = r'C:\Users\thoma\Documents\Master_GUG\Project_Photogrammetry\Code_Output\representation_Model.txt'
        datei1 = open(modFile, 'r')
        for zeile2 in datei1:
            parts2 = zeile2.split(':')
            parts2split = parts2[1].split(';')
            parts2split.pop()
            histogrammInf = np.asarray(np.float32(parts2split))
            dist = histogramComparison(histogramMod, histogrammInf, 'Jensen Shannon Divergence', w1=1, w2=1, w3=2, w4=3)
            if dist < threshold:
                dict[str(parts1[0] + ' with ' + parts2[0]) + ':::'] = dist
            elif dist > threshold:
                dict[str(parts1[0] + ' with ' + ' Larger distance ' + parts2[0]) + ':::'] = dist
            counter2 = counter2 + 1
            print("counter2: ", counter2)
        counter = counter + 1
        # Write all the distances to a textfile
        print(counter)
        f.write(str(dict) + '\n')

    f.close()

    return 0


def analyzeResult(outputDirectory):
    # read the "Result.txt" file
    filename1 = r'C:\Users\thoma\Documents\Master_GUG\Project_Photogrammetry\Code_Output\Resultat.txt'
    datei = open(filename1, 'r')
    dir = str(outputDirectory + 'Matchings.txt')
    f = open(dir, mode='a')
    counter = 0

    for zeile in datei:
        reconstruceded_dict = {}
        print("Zeile Nummer ", counter, " : ", len(zeile))
        splittedLine = zeile.split(',')
        for i in splittedLine:
            # print("i: ", i )
            splitted_i = i.split(':')
            key = splitted_i[0]
            print("Key: ", key)
            value = splitted_i[4]
            print("Value: ", value)
            if '}\n' in value:
                value = value.replace('}\n', '')
                reconstruceded_dict[key] = np.float32(value)
            elif '}' in value:
                value = value.replace('}', '')
                reconstruceded_dict[key] = np.float32(value)
            else:
                reconstruceded_dict[key] = np.float32(value)
        print("Reconstructed Dict: ", reconstruceded_dict)
        # Find the value  with the largest numerical value and output the respective key
        max_key = min(reconstruceded_dict, key=reconstruceded_dict.get)
        # print("min key: ", max_key)
        f.write(str(max_key)+'('+ str(reconstruceded_dict[max_key])+')')
        f.write('\n')
        counter = counter + 1

    # Todo: muss noch implementiert werden.
    return 0
