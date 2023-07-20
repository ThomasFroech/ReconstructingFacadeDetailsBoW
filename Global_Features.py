# In this file, functions in order to extract several global features are defined
import math
import open3d as o3d
import numpy as np
import math
import FractalDimension as fd
import twoDImage as tdi
from PIL import Image as im
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

# This function simply outputs the volume of the bounding box
def volumeOfBoundingBox(inputCloud):
    # get the oriented bounding box
    obb = inputCloud.get_oriented_bounding_box()
    # print(obb)
    # Calculation of the volume
    volume = (2 * obb.extent[0]) * (2 * obb.extent[1]) * (2 * obb.extent[2])
    # print("bbVolume= ", volume)
    return volume


# this function calculates the Surface Area of the bounding box
def areaOfBoundingBox(inputCloud):
    # get the oriented bounding box
    obb = inputCloud.get_oriented_bounding_box()
    # calculating the Area
    bbArea = 2 * (((2 * obb.extent[0]) * (2 * obb.extent[1])) + ((2 * obb.extent[1]) * (2 * obb.extent[2])) + (
            (2 * obb.extent[2]) * (2 * obb.extent[0])))
    # print("bbArea: ", bbArea)
    return bbArea


# This function is used to calculate the center of mass of a give Point Cloud
def getCnterOfMass(InputCloud, _):
    sumX = 0
    sumY = 0
    sumZ = 0
    length = len(_)
    counter = 0
    for i in _:
        sumX = sumX + InputCloud.points[_[counter]][0]
        sumY = sumY + InputCloud.points[_[counter]][1]
        sumZ = sumZ + InputCloud.points[_[counter]][2]
        xMean = sumX / length
        yMean = sumY / length
        zMean = sumZ / length
        counter = counter + 1
    centerOfMass = [[xMean, yMean, zMean]]
    # print("com: ", centerOfMass)
    return centerOfMass


# This function calculates the volume of the convex hull
def volumeOfConvexHull(InputCloud):
    # Get the convex hull
    hull, _ = InputCloud.compute_convex_hull()

    # Extracting all the triangles from the convex hull
    triangles = np.asarray(hull.triangles)
    # Iterating through all the Triangles in th convex hull
    counter = 0
    centerOfMass = getCnterOfMass(InputCloud, _)
    # Calculate the volume of the irregular tetrahedron formed by a triangle and the center of mass
    # for all of the triangles in the convex hull
    # 1. retrieve the triangles
    triangles = np.asarray(hull.triangles)
    # Iterating over all the triangles
    volume = 0
    counter = 0
    for i in triangles:
        # Todo: das Aufstellen der Matrix ist noch deutlich
        # Todo: zu verbessern um die Effizienz des Codes zu steigern
        # Todo: eventuell Numpy funktionen dazu verwenden (recherchieren welche am besten dazu geeignet sind!)
        matrix = np.zeros((4, 4))
        matrix[:, 3] = 1
        matrix[0, 0] = InputCloud.points[(_[triangles[counter][0]])][0]
        matrix[0, 1] = InputCloud.points[(_[triangles[counter][0]])][1]
        matrix[0, 2] = InputCloud.points[(_[triangles[counter][0]])][2]
        matrix[1, 0] = InputCloud.points[(_[triangles[counter][1]])][0]
        matrix[1, 1] = InputCloud.points[(_[triangles[counter][1]])][1]
        matrix[1, 2] = InputCloud.points[(_[triangles[counter][1]])][2]
        matrix[2, 0] = InputCloud.points[(_[triangles[counter][2]])][0]
        matrix[2, 1] = InputCloud.points[(_[triangles[counter][2]])][1]
        matrix[2, 2] = InputCloud.points[(_[triangles[counter][2]])][2]
        matrix[3, 0] = centerOfMass[0][0]
        matrix[3, 1] = centerOfMass[0][1]
        matrix[3, 2] = centerOfMass[0][2]
        # print("Matrix: ", matrix)
        volume = volume + np.abs(((1 / math.factorial(3)) * np.linalg.det(matrix)))
        # print("Volume: ", volume)
        counter = counter + 1
    return volume


# this function calculates the area of the surface of the convex hull
def areaOfConvexHull(InputCloud):
    # Get the convex hull
    hull, _ = InputCloud.compute_convex_hull()
    # Extracting all the triangles from the convex hull
    triangles = np.asarray(hull.triangles)
    # Iterating through all the Triangles in th convex hull
    chArea = 0
    counter = 0

    for i in triangles:
        # get from the different indices to the actual point coordinates of the
        # different triangle's vertex points
        trianglePoint_1 = InputCloud.points[(_[triangles[counter][0]])]
        # print("T1: ", trianglePoint_1)
        trianglePoint_2 = InputCloud.points[(_[triangles[counter][1]])]
        # print("T2: ", trianglePoint_2)
        trianglePoint_3 = InputCloud.points[(_[triangles[counter][2]])]
        # print("T3: ", trianglePoint_3)
        # Calculation of the side lengths
        sideA = np.linalg.norm(trianglePoint_1 - trianglePoint_2)
        # print("SidaA: ", sideA)
        sideB = np.linalg.norm(trianglePoint_2 - trianglePoint_3)
        # print("SidaB: ", sideB)
        sideC = np.linalg.norm(trianglePoint_1 - trianglePoint_3)
        # print("SidaC: ", sideC)
        # Claculation of the half perimeter
        halfPerim = 0.5 * (sideA + sideB + sideC)
        # Use Heron's law
        chArea = chArea + (math.sqrt(halfPerim * (halfPerim - sideA) * (halfPerim - sideB) * (halfPerim - sideC)))
        counter = counter + 1
    # print("Area: ", chArea)
    return chArea

# This Function is used in order to calculate the area / Volume ratio of the bounding box
def bbAVRatio(InputCloud):
    area = areaOfBoundingBox(InputCloud)
    volume = volumeOfBoundingBox(InputCloud)
    ratio = area / volume
    # print("AV-Ratio: ", ratio)
    return ratio


# this function compares the Surfce-Area of the Bounding Box to the Surface-Area of a
# cube with the same volume
def bbCubeness(InputCloud):
    bbVolume = volumeOfBoundingBox(InputCloud)
    bbSurface = areaOfBoundingBox(InputCloud)
    # Calculation of the area of a cube with the same volume
    length = math.sqrt(bbVolume)
    cubeSurface = 6 * (length * length)
    cubeness = bbSurface / cubeSurface
    # print(cubeness)
    return cubeness


# this function compares the Surface-area of the convex hull with a cube of the same volume
def chAVRatio(InputCloud):
    volumeCH = volumeOfConvexHull(InputCloud)
    areaCH = areaOfConvexHull(InputCloud)
    chCubeness = areaCH / volumeCH
    return chCubeness

# this function compares the Surfce-Area of the Convex_Hull to the Surface-Area of a
# cube with the same volume
def chCubeness(InputCloud):
    volumeCH = volumeOfConvexHull(InputCloud)
    areaCH = areaOfConvexHull(InputCloud)
    # Calculation of the area of a cube with the same volume
    length = math.sqrt(volumeCH)
    cubeSurface = 6 * (length * length)
    cubeness = areaCH / cubeSurface
    return cubeness


# This function is used to calculate the circularity of the convex hull of the point cloud
# it measures the volume deviation between the convex  hull and is equal volume hemisphere
# Todo: WRONGGG!!!!!!!
def chCircularity(InputCloud):


    # Todo! Alt Falscher code
    volumeCH = volumeOfConvexHull(InputCloud)
    areaCH = areaOfConvexHull(InputCloud)
    # Calculation of the volume of a sphere with the same  surface area

    # Calculate the radius of a shpere withe the same area as the convex hull
    sphereRadius = math.sqrt(areaCH / (4 * math.pi))

    # Calculate the volume of the sphere using the radius
    sphereSurface = (4 / 3) * math.pi * (sphereRadius ** 3)

    # calculation of the actual circularity value
    chCircularity = areaCH / sphereSurface

    return chCircularity


# Todo: Volume can not be calculated because mesh is not always 3 dimensional
def convexity(InputCloud):
    return 0


def cohesion(InputCloud):
    # get the convex hull
    hull, _ = InputCloud.compute_convex_hull()
    # Step 1: get the volume of the convex hull
    volumeCH = volumeOfConvexHull(InputCloud)
    # Step 2: find the radius of the sphere that has the same volume as the convex hull
    # Calculation of the surface-area of a sphere with the same volume
    radius = math.pow((volumeCH / ((4 / 3) * math.pi)), (1 / 3))
    print("radius: ", radius)
    # Step 3: get the center of mass of the pointcloud
    # Calculate the distance from the center of mass
    centerOfmass = getCnterOfMass(InputCloud, _)
    avg_dist_in_circle = 0
    avg_dist_ges = 0
    # Iterate over all the points in the convex hull
    counter = 0
    for index in _:
        # print("Point: ", InputCloud.points[index])
        # Get the point coordainets
        x = InputCloud.points[index][0]
        # print("x: ", x)
        y = InputCloud.points[index][1]
        # print("y: ", y)
        z = InputCloud.points[index][2]
        # print("z: ", z)
        dist = math.sqrt(
            math.pow(x - centerOfmass[0][0], 2) + math.pow(y - centerOfmass[0][1], 2) + math.pow(z - centerOfmass[0][2],
                                                                                                 2))
        # print("Dist: ", dist)
        if dist <= radius:
            avg_dist_in_circle = avg_dist_in_circle + dist
            avg_dist_ges = avg_dist_ges + dist
        if dist > radius:
            avg_dist_ges = avg_dist_ges + dist
        counter = counter + 1
        # print("AVGDC: ", (avg_dist_in_circle / counter))
        # print("AVGDG: ", (avg_dist_ges / counter))
    cohesion = (avg_dist_in_circle / counter) / (avg_dist_ges / counter)

    return cohesion


# Todo: This function is not going to be finished, as it gets too complicated
# Todo: and will take too much time an effort to be finished
# This function is used to calculate the exchange of the
# concex hull of the input cloud. The exchange measures how much of the volume
# inside a sphere is exchanged with the volume outside it to create the polyhedron.
def exchange(InputCloud):
    # Get the convex hull
    hull, _ = InputCloud.compute_convex_hull()
    # Step 1: get the volume of the convex hull
    volumeCH = volumeOfConvexHull(InputCloud)
    # Step 2: find the radius of the sphere that has the same volume as the convex hull
    # Calculation of the surface-area of a sphere with the same volume
    radius = math.pow((volumeCH / ((4 / 3) * math.pi)), (1 / 3))
    # Step 3: get the center of mass of the pointcloud
    cog = getCnterOfMass(InputCloud)
    # Step 4: Calculate The Exchange.
    volume_intersect = 0
    counter = 0
    # Iteriere über alle dreiecke
    triangles = np.asarray(hull.triangles)
    for triang in triangles:
        # transfromiere die koordinaten der Vertices in Spärische Koordinaten
        x1 = InputCloud.points[(_[triangles[counter][0]])][0]
        y1 = InputCloud.points[(_[triangles[counter][0]])][1]
        z1 = InputCloud.points[(_[triangles[counter][0]])][2]
        x2 = InputCloud.points[(_[triangles[counter][1]])][0]
        y2 = InputCloud.points[(_[triangles[counter][1]])][1]
        z2 = InputCloud.points[(_[triangles[counter][1]])][2]
        x3 = InputCloud.points[(_[triangles[counter][2]])][0]
        y3 = InputCloud.points[(_[triangles[counter][2]])][1]
        z3 = InputCloud.points[(_[triangles[counter][2]])][2]
        # Calculation of the radius
        r1 = math.sqrt((x1 * x1) + (y1 * y1) + (z1 * z1))
        r2 = math.sqrt((x2 * x2) + (y2 * y2) + (z2 * z2))
        r3 = math.sqrt((x3 * x3) + (y3 * y3) + (z3 * z3))
        # Calculation of the angle theta
        theta1 = math.acos(z2 / r2)
        theta2 = math.acos(z1 / r1)
        theta3 = math.acos(z3 / r3)
        # Calculation of tha angle phi
        phi1 = math.atan2(y1, x1)
        phi1 = math.atan2(y2, x2)
        phi1 = math.atan2(y3, x3)
        cpunter = counter + 1

    return 0


# This function is used to calculate the ratio of the volume of the convex hull and the bounding box
def chBb_VolumeRatio(InputCloud):
    volumeCH = volumeOfConvexHull(InputCloud)
    volumeBb = volumeOfBoundingBox(InputCloud)
    ratio = volumeCH / volumeBb
    return ratio


# This function is used to calculate the ratio of the surface-area of the convex hull and the surface area of the
# bounding box
def chBbAreaRatio(InputCloud):
    surfaceAreaCH = areaOfConvexHull(InputCloud)
    surfaceAreaBB = areaOfBoundingBox(InputCloud)
    ratio = surfaceAreaCH / surfaceAreaBB
    return ratio


# Todo: muss immernoch implementiert werden! --> Use the fractal Dimension instead!
def fractality(InputCloud):
    fractality = 0
    return fractality


# This function is used to calculate the fractal dimension of the point cloud
# by applying the box counting algorithm. The code that is used here was written by
# the Chat GPT Chatbot from OpenAi. The code was checked briefly for plausibility
# but was not investigated in detail
def fractalDimension(InputCloud):
    hull, _ = InputCloud.compute_convex_hull()
    pointsInCH = np.ones((len(_), 3))
    counter = 0
    for element in _:
        pointsInCH[counter, 0] = InputCloud.points[element][0]
        pointsInCH[counter, 1] = InputCloud.points[element][1]
        pointsInCH[counter, 2] = InputCloud.points[element][2]
        counter = counter + 1
    resolutions, num_boxes = fd.box_count(pointsInCH, 0.001, 5, 500)
    fractal_dimension = fd.calc_fractal_dimension(resolutions, num_boxes)
    return fractal_dimension


# This function is used to calculate the fractal dimension of the 3D point cloud after it was projected
# to one of the coordinate planes
# Instructions for use:
# 'xy' for the projection to the xy plane
# 'xz' for the projection to the xz plane
# 'yz' for the projection to the yz plane
# Todo: eventuell noch an die konvexe Hülle anpassen!
def fractalDimensionProjections(InputCloud, planeSpecifications):
    # print("Test: ", np.asarray(InputCloud.points))
    if planeSpecifications == 'xy' or planeSpecifications == 'yx':
        points = np.asarray(InputCloud.points)[:, 0:2]
        # print("Points: ", points)
        resolutions, num_boxes = fd.box_count(np.asarray(points), 0.01, 1, 50)
        fractal_dimension = fd.calc_fractal_dimension(resolutions, num_boxes)
        # print("Plane: ", planeSpecifications)
    elif planeSpecifications == 'xz' or planeSpecifications == 'zx':
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
        # print("Points: ", points)
        resolutions, num_boxes = fd.box_count(np.asarray(points), 0.01, 1, 50)
        fractal_dimension = fd.calc_fractal_dimension(resolutions, num_boxes)
        # print("Plane: ", planeSpecifications)
    elif planeSpecifications == 'yz' or planeSpecifications == 'zy':
        points = np.asarray(InputCloud.points)[:, 1:2]
        # print("Points: ", points)
        resolutions, num_boxes = fd.box_count(np.asarray(points), 0.01, 1, 50)
        fractal_dimension = fd.calc_fractal_dimension(resolutions, num_boxes)
        # print("Plane: ", planeSpecifications)
    else:
        print("Please select a valid plane to which the Pointcloud shall be projected")
        fractal_Dimension = 0
    return fractal_dimension






def calculatePerimeter(points, hull):
    # get the vertices of the convex hull
    hull_vertices = points[hull.vertices]

    # calculate the length of each edge of the convex hull
    edge_lengths = np.linalg.norm(np.diff(hull_vertices, axis=0), axis=1)

    # calculate the perimeter of the convex hull
    perimeter = np.sum(edge_lengths)
    return perimeter

def circularityCH2d(inputCloud):
    # This part of the code is necessary in order to finde the front view
    image1 = tdi.create2DBinaryImage(inputCloud, 640, 640, 'xy')
    choice1 = tdi.manualSelection()
    if choice1 == "y":
        spec = 'xy'

    if choice1 == "n":
        print("The previously shown image is not chosen")
        image1 = tdi.create2DBinaryImage(inputCloud, 640, 640, 'xz')
        choice2 = tdi.manualSelection()
        if choice2 == "y":
            spec = 'xz'
        if choice2 == "n":
            print("The previously shown image is not chosen")
            image1 = tdi.create2DBinaryImage(inputCloud, 640, 640, 'yz')
            spec = 'yz'


    # distinguish the different projection cases
    if spec == 'xy':
        # Projecting the point cloud
        points = np.asarray(inputCloud.points)[:, 0:2]

        # compute the convex hull
        hull = ConvexHull(points)

        # get the area of the convex hull
        area = hull.volume

        # get the perimeter of the convex hull
        perimeter = calculatePerimeter(points, hull)

        # Calculate the Area of a circle with the same perimeter
        radius = perimeter / (2 * math.pi)
        areaCircle = math.pi * math.pow(radius, 2)

        # calculate the ratio
        ratio = area / areaCircle

    elif spec == 'xz':
        points1 = np.asarray(inputCloud.points)[:, 0]
        points2 = np.asarray(inputCloud.points)[:, 2]
        counter = 0
        points = np.ones((len(points1), 2))

        for i in points1:
            points[counter, 0] = points1[counter]
            points[counter, 1] = points2[counter]
            counter = counter + 1

        # compute the convex hull
        hull = ConvexHull(points)

        # get the area of the convex hull
        area = hull.volume

        # get the perimeter of the convex hull
        perimeter = calculatePerimeter(points, hull)

        # Calculate the Area of a circle with the same perimeter
        radius = perimeter / (2 * math.pi)
        areaCircle = math.pi * math.pow(radius, 2)

        # Calculate the ratio
        ratio = area / areaCircle

    elif spec == 'yz':
        # Projecting the point cloud
        points = np.asarray(inputCloud.points)[:, 1:]

        # compute the convex hull
        hull = ConvexHull(points)

        # get the area of the convex hull
        area = hull.volume

        # get the perimeter of the convex hull
        perimeter = calculatePerimeter(points, hull)

        # Calculate the Area of a circle with the same perimeter
        radius = perimeter / (2 * math.pi)
        areaCircle = math.pi * math.pow(radius, 2)

        # Calculate the ratio
        ratio = area / areaCircle

    return ratio


def squarenessCH2d(inputCloud):
    # todo: to be implemented
    # This part of the code is necessary in order to finde the front view
    image1 = tdi.create2DBinaryImage(inputCloud, 640, 640, 'xy')
    choice1 = tdi.manualSelection()
    if choice1 == "y":
        spec = 'xy'

    if choice1 == "n":
        print("The previously shown image is not chosen")
        image1 = tdi.create2DBinaryImage(inputCloud, 640, 640, 'xz')
        choice2 = tdi.manualSelection()
        if choice2 == "y":
            spec = 'xz'
        if choice2 == "n":
            print("The previously shown image is not chosen")
            image1 = tdi.create2DBinaryImage(inputCloud, 640, 640, 'yz')
            spec = 'yz'


    # distinguish the different projection cases
    if spec == 'xy':
        # Projecting the point cloud
        points = np.asarray(inputCloud.points)[:, 0:2]

        # compute the convex hull
        hull = ConvexHull(points)

        # get the area of the convex hull
        area = hull.volume

        # get the perimeter of the convex hull
        perimeter = calculatePerimeter(points, hull)

        # Calculate the Area of a square with the same perimeter
        sidelength = math.sqrt(area)
        perimSquare = 4 * sidelength

        # calculate the ratio
        ratio = perimSquare / perimeter

    elif spec == 'xz':
        points1 = np.asarray(inputCloud.points)[:, 0]
        points2 = np.asarray(inputCloud.points)[:, 2]
        counter = 0
        points = np.ones((len(points1), 2))

        for i in points1:
            points[counter, 0] = points1[counter]
            points[counter, 1] = points2[counter]
            counter = counter + 1

        # compute the convex hull
        hull = ConvexHull(points)

        # get the area of the convex hull
        area = hull.volume

        # get the perimeter of the convex hull
        perimeter = calculatePerimeter(points, hull)

        # Calculate the Area of a square with the same perimeter
        sidelength = math.sqrt(area)
        perimSquare = 4 * sidelength

        # calculate the ratio
        ratio = perimSquare / perimeter

    elif spec == 'yz':
        # Projecting the point cloud
        points = np.asarray(inputCloud.points)[:, 1:]

        # compute the convex hull
        hull = ConvexHull(points)

        # get the area of the convex hull
        area = hull.volume

        # get the perimeter of the convex hull
        perimeter = calculatePerimeter(points, hull)

        # Calculate the Area of a square with the same perimeter
        sidelength = math.sqrt(area)
        perimSquare = 4 * sidelength

        # calculate the ratio
        ratio = perimSquare / perimeter
    return ratio

# This function is used in order to calculate the ratio of the area of the convex hull
# and the area of the bounding box of the projected pointcloud
def bb2dch2dAreaRatio(inputCloud):
    # This part of the code is necessary in order to finde the front view
    image1 = tdi.create2DBinaryImage(inputCloud, 640, 640, 'xy')
    choice1 = tdi.manualSelection()
    if choice1 == "y":
        spec = 'xy'

    if choice1 == "n":
        print("The previously shown image is not chosen")
        image1 = tdi.create2DBinaryImage(inputCloud, 640, 640, 'xz')
        choice2 = tdi.manualSelection()
        if choice2 == "y":
            spec = 'xz'
        if choice2 == "n":
            print("The previously shown image is not chosen")
            image1 = tdi.create2DBinaryImage(inputCloud, 640, 640, 'yz')
            spec = 'yz'

    # distinguish the different projection cases
    if spec == 'xy':
        # Projecting the point cloud
        points = np.asarray(inputCloud.points)[:, 0:2]

        print("das sind die Punkte! ", points)
        # compute the convex hull
        hull = ConvexHull(points)

        # get the area of the convex hull
        areaCH = hull.volume

        # find minimum and maximum x and y coordinates of points on hull
        min_x = min(points[hull.vertices, 0])
        max_x = max(points[hull.vertices, 0])
        min_y = min(points[hull.vertices, 1])
        max_y = max(points[hull.vertices, 1])

        # bounding box dimensions
        width = max_x - min_x
        height = max_y - min_y

        # get the area of the bounding box
        areaBB = width * height

        # Calculate the ratio
        ratio = areaCH / areaBB

    elif spec == 'xz':
        points1 = np.asarray(inputCloud.points)[:, 0]
        points2 = np.asarray(inputCloud.points)[:, 2]
        counter = 0
        points = np.ones((len(points1), 2))

        for i in points1:
            points[counter, 0] = points1[counter]
            points[counter, 1] = points2[counter]
            counter = counter + 1

        # compute the convex hull
        hull = ConvexHull(points)

        # get the area of the convex hull
        areaCH = hull.volume

        # find minimum and maximum x and y coordinates of points on hull
        min_x = min(points[hull.vertices, 0])
        max_x = max(points[hull.vertices, 0])
        min_y = min(points[hull.vertices, 1])
        max_y = max(points[hull.vertices, 1])

        # bounding box dimensions
        width = max_x - min_x
        height = max_y - min_y

        # get the area of the bounding box
        areaBB = width * height

        # Calculate the ratio
        ratio = areaCH / areaBB

    elif spec == 'yz':
        # Projecting the point cloud
        points = np.asarray(inputCloud.points)[:, 1:]

        # compute the convex hull
        hull = ConvexHull(points)

        # get the area of the convex hull
        areaCH = hull.volume

        # find minimum and maximum x and y coordinates of points on hull
        min_x = min(points[hull.vertices, 0])
        max_x = max(points[hull.vertices, 0])
        min_y = min(points[hull.vertices, 1])
        max_y = max(points[hull.vertices, 1])

        # bounding box dimensions
        width = max_x - min_x
        height = max_y - min_y

        # get the area of the bounding box
        areaBB = width * height

        # Calculate the ratio
        ratio = areaCH / areaBB

    return ratio