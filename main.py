import CodeBook as CB
import os
import argparse
import Inference
import open3d as o3d
import numpy as np
import Normalization
import ArtificialNoise
import twoDImage as tdi

# Read the environment variables
# -i : Directory of the folder that contains the .pcd files that are used to create the codebook
# -o : Directory of th folder that will contain the different .txt files

PARSER = argparse.ArgumentParser(description='Project_Froech')
PARSER.add_argument('-i', '--directory',
                    help='Directory containing CityGML file(s).', required=True)
PARSER.add_argument('-o', '--results',
                    help='Directory where the OBJ file(s) should be written.', required=True)
PARSER.add_argument('-inf', '--inference',
                    help='Directory which contains the files on which the code should infere.', required=True)

ARGS = vars(PARSER.parse_args())
input_directory = os.path.join(ARGS['directory'], '')
print("Input dircetory: ", input_directory)
output_directory = os.path.join(ARGS['results'], '')
print("Output dircetory: ", output_directory)
inference_directory = os.path.join(ARGS['inference'], '')
print("Inference dircetory: ", inference_directory)

inputDir = input_directory
FileCounter = 0
for filename in os.listdir(inputDir):
    print("filename: ", filename)
    # create an empty feature vector
    featureVector = []
    # read the file
    file = os.path.join(input_directory, filename)
    pcd = o3d.io.read_point_cloud(file)
    # Get the points of the PointCloud
    points = np.asarray(pcd.points)
    print("Number of points: ", len(points))
    # normlize the points
    normalized_points = Normalization.normalize(points)
    # stuff = Normalization.normalize(points) # Todo: this is a test to find out whats going on!
    # Get number of points in point cloud
    num_points = np.asarray(pcd.points).shape[0]
    # Calculate the bounding box volume
    bounding_box = pcd.get_axis_aligned_bounding_box()
    bounding_box_volume = bounding_box.volume()
    # Calculate the point density
    point_density = num_points / bounding_box_volume
    # perfroma a downsmapling of the pointcloud
    downpcd = normalized_points.voxel_down_sample(voxel_size=0.05)
    stuff, pointsrem = downpcd.remove_radius_outlier(nb_points=16, radius=0.5)
    print("Stufftype: ", len(stuff.points))
    repName = "representation_model.txt"
    Inference.representObject(filename, stuff, output_directory, repName, s23=1, sl1=1)
    CB.createLocalFeatureSpace(stuff, output_directory, s1=1,)
    CB.createGlobalFeatureSpace(stuff, output_directory, s23=1)
CB.generateCodebook(input_directory, output_directory)

#for filename in os.listdir(inference_directory):
#    try:
#        print("filename: ", filename)
#        # create an empty feature vector
#        featureVector = []
#        # read the file
#        file = os.path.join(inference_directory, filename)
#        pcd = o3d.io.read_point_cloud(file)
#        points = np.asarray(pcd.points)
#        print("Number of points: ", len(points))
#        # Get number of points in point cloud
#        num_points = np.asarray(pcd.points).shape[0]
#        # Calculate the bounding box volume
#        bounding_box = pcd.get_axis_aligned_bounding_box()
#        bounding_box_volume = bounding_box.volume()
#        # Calculate the point density
#        point_density = num_points / bounding_box_volume
#        # normlize the points
#        normalized_points = Normalization.normalize(points)
#        #stuff = Normalization.normalize(points) # Todo: This is just an experimental code to find out#        #normalized_points_noise = ArtificialNoise.applyArtificialNoise(normalized_points.points)#
#
#        z_mean = pcd.get_center()[2]
#        print("Z_mean: ", z_mean)
#        #perfroma a downsmapling of the pointcloud
#        if z_mean < 14:
#            downpcd = normalized_points.voxel_down_sample(voxel_size=0.05)
#            stuff, pointsrem = downpcd.remove_radius_outlier(nb_points=16, radius=0.5)
#        else:
#            stuff, pointsrem = normalized_points.remove_radius_outlier(nb_points=16, radius=0.5)
#
#       print("Stufftype: ", len(stuff.points))
#        repName = "representation_Inference.txt"
#        Inference.representObject(filename, stuff, output_directory, repName, sl1=1, s23=1)
#    except:
#        #This part of the code is used in order to assign a match to the
#        #Windows where an error occures
#        print("An error has occcured!")
#        dir = output_directory + "Resultat.txt"
#        dict2wr = {filename: "no match"}
#        f = open(dir, mode='a')
#        f.write(str(dict2wr))
#        f.write("\n")
#        f.close()
#threshold = 0.4
#Inference.infere(output_directory, threshold)
#Inference.analyzeResult(output_directory)
