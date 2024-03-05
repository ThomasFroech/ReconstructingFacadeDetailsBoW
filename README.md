
This repository contains the code, that was used in the paper: Reconstructing façade details using MLS point clouds and Bag-of-Words approach
# Project Photogrammetry - Reconstructing façade details using MLS point clouds and Bag-of-Words approach

## Introduction

The idea of this project is to make use of the Bag of Words approach [Curka et. al., 2004] in order reconstruct facade details using MLS point clouds. the general idea is described in the following schematic graphic based on [Memon et al., 2019]. 

<div align="center">
  <img src="uploads/623aad7443aa0256c6ed074a1504f95a/Introductional_Graphic.png" alt="Introductional_Graphic" width="600">
  <br>
  <em>Figure 1: General overview, based on [Memon et al., 2019]</em>
</div>
<br></br>
The actuality and relevance of this idea results from the increased availability of MLS point clouds and large CAD databases [Trimble Inc, 2021] in recent years. 

## Methodology

### Overview
The following diagram gives an overview on the implementation in this project. The gereal structure can be divided into two parts:
<ul><li>The training part. The codebook is construcetd here. This process consists of several individual steps. These invlove (among others) pose normalization, feature extraction, and a clustering in feature space. </li><li>The inference part. Here , the description of the windows from the MLS point cloud takes place as well as the histogram comparison between the codeword representation of the model and the object.</li></ul>
<br></br>
<div align="center">
  <img src="uploads/18af7f9514e2da1eec73fe7e4a98b9a7/Gesamter_Prozess.drawio.png" alt="Introductional_Graphic" width="600">
  <br>
  <em>Figure 2: Schematic diagram of the used methodology</em>
</div>


### The Bag of Words Approach 

The general bag of words approach was introduced in 1986 by Salton and McGill. It was originally developed in order to classify texts by creating histograms of relative word occurrences. [Salton & McGill, 1986] This approach was later extended to images by Csurka et. al. in 2004. The concept there is to extract local features from an Image, build a dictionary (codebook) and create a histogram of occurences of the "visual codewords" for each image in the following step. The construction of the codebook invelves several different steps for which specific considerations have to be made [Csurka et. al., 2004]:

<ul><li>Feature extraction: Which features should be used?</li><li>Vector quantization: How can it be performed?</li><li>Clustering: Which clustering algorithm should be used? What number of clusters would be beneficial?</li><li>Histogram comparison: Which histogram distance should be used?</li></ul>

### Augmentation of the bag of words approach by incorporating global features

A problem in using the bag of words approach when dealing with shapes is that shapes generally have a smaller number of distinct features. This means that geometric objects will be represented by many similar visual words [Bronstein et al., 2011]. A possibility to circumvent this problem is to incorporate global features into the representation of the object. This way it is possible to make use of global information on the object during the process. There have been different studies that make use of this concept:

<ul><li>[Zhu et al., 2016]</li><li>[Memon et al., 2019]</li></ul>

The realization of the general concept of incorporation of global features in this project is schematically shown in the following diagram:

<div align="center">
  <img src="uploads/127a1d17bac909a89ae205568563f45c/Incorporation_of_Global_Features.drawio.png" alt="Introductional_Graphic" width="300">
  <br>
  <em>Figure 3: Incorporation of global features</em>
</div>



### CAD Model Selection

Large CAD libraries are one of the foundations of the concept that this project is based on. The CAD models are selected from the SketchUp 3D Warehouse [Trimble Inc, 2021]. For experimental purposes, only a small number of windows is selected. The individual windows are chosen on the basis of their visual appearance according to the windows that appear in  the TUM Facade dataset. A model that does not correspond to any window that appears in the TUM Facade dataset is selecte additionally. 

The selected models are manually edited in order to add or remove window bars. Furthermore, curtains or similar structures are removed from all of the chosen CAD Models.

In order to make the analysis of the window bars possible, window glass is also removed from all of the models.Through this step, the models correspond more to the actual measurement conditions, since the pulses of the laser scanner usually penetrate window glass. The following figure shows the edited windows that are used in this project:

<div align="center">
  <img src="uploads/c83f12b1fcda79dc61653c27568d0cb6/Alle_Fenster.png" alt="Introductional_Graphic" width="600">
  <br>
  <em>Figure 4: Edited CAD window models</em>
</div>


### CAD Model Sampling

One of the key problems in this project is that the features that are used to describe the model and the features that describe the measured window have to correspond to each other as much as possible. In this project, the CAD models are transferred from the CAD domain to the point cloud domain to ensure this correspondence. This transfer is realized by perfoming an equidistant sampling of the CAD models in FME with the PointCloudCombiner application. During this process, the sampling distance is a critical parameter. It has to be chosen in such a way, that the resulting point cloud represents the CAD model accurately. Large sampling distances will lead to inclomplete representations, while too small sampling distances will result in an unnecessary large number of points.

<div align="center">
  <img src="uploads/c3c00dbece44fe5ba1fc2c20b33d4cae/Sampled_Window.JPG" alt="Introductional_Graphic" width="200">
  <br>
  <em>Figure 5: Sampled CAD Model</em>
</div>

### Pose normalization

The first step is to perform a pose normalization. In order to apply the  techniques in the following steps, certain invariances must be guaranteed. These following invariances can be guaranteed by a sophisticated pose normalization:

<ul><li>Translation invariance</li><li>Rotation invariance</li><li>Scale invariance</li></ul>

The Translation invariance can generally be achieved by various different techniques. In this project, it is realized by performing a translation of the mean coordinate point to the orogin of the coordinate system. The translation parameters are simply the mean x,y and z values.

The scale invarianve is achieved by scaling the pointclouds th such a way that the maximal distance of a point from the origin of the coordinate system is equal to 10. The scaling factor is calculated from The largest coordinates in terms of absolute value.

The most common method in order to achieve rotational invariance is to calculate the principal axes of the object and align them with the coordinate system. Often, the principal axes of objects are calculated by using principal component analysis (PCA) [Martens & Blankenbach, 2020]. This concept was implemented in this project experimentally, but has proven not to be unsuitable for use, here. The reason is that PCA is very sensitive to towards noise and point sampling density variations [Martens & Blankenbach, 2020]. Both disturbing fasctors are present in the pointclouds that are used in this project. Because of that, the point clouds are are aligned to the coordinate axes manually. There would be other options for alignemnt, for example a rectilinearity based approach by [Lian et al., 2009] that are not investigate further in this project.

### Downsampling and Outlier Removal

In order to increase the speed of the computations a downsampling of the pointclouds is performed. for this purpose, the respective Open3D functionality ("normalized_points.voxel_down_sample(voxel_size=0.05")) is used. in this project, the voxel size that is used in this function is set to 0.5. 
For the removal of outliers and the reudction of noise, the Open3D functionality ".remove_radius_outlier(nb_points=16, radius=0.5))" is used. the parameter settings for this function were determined by a viusal trial and error approach. The settings were adapted until the binary image that was generated from the resulting projected point cloud showed a satisfactory appearance.

### 2D Image Generation

A facade detail (or more specifically, a window) that is extracted from an MLS point cloud is generally available as a 3 dimensional collection of points. When thinking about how we as humans perceive facade details (or again more specifically windows), we notice that we mostly rely on the front view of the window. Most windows dont have prominent 3 dimensional structures that can be used for identification. Because of thís, only the front view of the windows is considered in this project. 
For practicability and simplified feature extraction, a 2D binary image is created from the pointcloud. The process of creating such images consits of the following two steps:

1. Projection of the 3D point cloud into a 2D point cloud. Because of the manual pose normalization that is performed in the previous step, the front view is already parallel to one of the coordinate planes. This simplifys the projection of the pointcloud.
2. Then the image is created from the projected point cloud. A zero-matrix that represents the 2D structure of the binary image is to 1 for every cell that corresponds to a projected point

Since there are three coordinate planes that the front view could be parallel to, the correct one has to be found. In this project, a simple manual selection is applied for this purpose. During this selection process, the user is presented the images generated from the three different projection directions subsequently. The user can either discard or accept the presented views. Another, more elegant  prossibility that is implemented experimentally but not used in the project, is to find the correct projection direction by finding the view with the maximum covariance. 

The following graphic shows an example of the binary 2D images generated from a model


<div align="center">
  <img src="uploads/247dd220ddd651b7c4e6bb9e9296ccce/3_Images.png" alt="Introductional_Graphic" width="700">
  <br>
  <em>Figure 6: Projected imaged from all 3 different views</em>
</div>

### 2D Image Processing

In general, the points clouds that are aquired by MLS are corrupted by a certain level of noise and will show point density variations. In order to be able to extract meaningful information from the binary images that are created in the previous step, different processing steps are necessary. The following processing steps are applied:

<ul><li>Morphological Operators:   Dilation


This step is necessary beacause the initial image consits of a collection of individually separated points. in order to perform an edge detection later on, the represented object must be a single homogenous area in the image. This is achieved by applying a dilation with a sufficiently large kernel size. In this project a kernel size of 20 pixels is used. The following graphic gives an example of the dilated image:

<div align="center">
  <img src="uploads/891cc3ab1b78fe495daa3a9324e4cb6a/Dilated_image.JPG" alt="Introductional_Graphic" width="250">
  <br>
  <em>Figure 7: Dilated Image</em>
</div>


</li><li>Edge Detection:   Laplace Filter

The geometry (and also the topology) of the window is mainly described by the edges and corners of the window. Therefore it is reasonable to perform an edge detection filter to the previously dilated image before extracting features from it. The following graphic gives an example of the image after the application of the laplace filter: 

<div align="center">
  <img src="uploads/3ea7feeebe02c36e5fc338aa2f5db90f/Laplacian_image.JPG" alt="Introductional_Graphic" width="250">
  <br>
  <em>Figure 8: Laplacian Image</em>
</div>

</li><li>Line Simplification: Douglas-Peucker Algorithm

Often, noise will lead to a lot of irregularities in the images. In order to increase robustness to such noise, the Douglas-peucker algorithm [Douglas & Peucker, 1973] is applied to the contours that are found in the images. The following graphic shows an example of such a simplification. On the left side there is the image without the simplification by Douglas-Peucker, on the right side there is the same image after the application of the Douglas-Peucker algorithm. For the implementation of the algorithm, the OpenCV Library is used.

<div align="center">
  <img src="uploads/3a2b8335c57d0cbfc03361d592c7f89a/DP_Auswirkung.png" width="400">
  <br>
  <em>Figure 9: Application of the Douglas-Peucker Algorithm</em>
</div>
<br><br/>
The following figure demonstrates how the extraction of meaningful interest points can be enhanced by applying this method. In the left image, a lot of the detected interest points are not related to the general structure of the window, but are solely detected because of the presence of noise. In the right image, where the Douglas-Peucker algorithm was applied, more meaningful interest points are found.

<br></br> 
<div align="center">
  <img src="uploads/25f3c582cb9b4f40479d15b11704a31d/Comparison_DP.png" width="400">
  <br>
  <em>Figure 10: Extraction of meaningful interest points</em>
</div>
</li>
</ul>

### Feature Extraction

In order to construct the codebook and describe objects by means of this codebook, features have to be extracted (vgl. Figure 2). In this project, the extraction of the features takes place in the point cloud domain. Various features were implemented but not all of them are used, as some have proven to be unfit for use in the framework of this project. The following table summarizes all the implemented features:


| Feature | Feature Type | Acquisition Source | Explanation | Additional information|
|--------------|-----------|-----------|-----------|-----------|
| AV-Ratio of the Bounding Box | global | 3D point cloud| Ratio between the area and the volume of the bounding box | the bounding box is calculated by using the Open3D functionality "get_oriented_bounding_box()"| 
| Cubeness of the Bounding Box | global | 3D point cloud | The Bounding Box Cubeness compares the surface area of the bounding box to the surface area of a cube with the same volume | <ul><li>the bounding box is  calculated by using the Open3D functionality ".get_oriented_bounding_box()"</li><li>Inspired by [Labetski et al., 2022]</li><ul>|
| AV-Ratio of the Convex Hull | global | 3D point cloud | Ratio between the area and the volume of the convex hull | the convex hull is  calculated by using the Open3D functionality ".compute_convex_hull()"|
| Cubeness of the Convex Hull | global | 3D point cloud |The Cubeness of the Convex Hull compares the surface area of the convex hull to the surface area of a cube with the same volume| <ul><li>the convex hull is  calculated by using the Open3D functionality ".compute_convex_hull()"</li><li>Inspired by [Labetski et al., 2022]</li></ul> |
| Circularity of the Convex Hull | global | 3D point cloud | The Circularity of the Convex Hull measures the volume deviation between the convex  hull and is equal area hemisphere |<ul><li>the convex hull is  calculated by using the Open3D functionality ".compute_convex_hull()"</li><li>Inspired by [Labetski et al., 2022]</li></ul>|
| Cohesion of the Convex Hull | global | 3D point cloud | <ul><li>The cohesion is a measure of overall accessibility from all points to others within a polyhedron.</li><li>"The Cohesion Index is the ratio of the average distance-squared among all points in an equalarea circle and the average distance-squared among all points in the shape" [Angel et al., 2010]</li></ul> | - |
| Exchange of the Convex Hull | global | 3d point cloud | "[The exchange] measures how much of the area inside a circle is exchanged with the area outside it to create a given shape" [Angel et al., 2010] | The calculation of this feature was attempted, but has proven to be much to complex in order to be completed within the scope of this project. A proper geometry engine would be necessary to perform this task.|
| Convex Hull to Bounding Box Area Ratio | global | 3D point cloud | The ratio of the area of convex hull and the area of the bounding box | <ul><li>the bounding box is  calculated by using the Open3D functionality ".get_oriented_bounding_box()"</li><li>the convex hull is  calculated by using the Open3D functionality ".compute_convex_hull()"</li><ul>|
| Convex Hull to Bounding Box Volume Ratio | global | 3D point cloud | The ratio of the vlume of convex hull and the volume of the bounding box | <ul><li>the bounding box is  calculated by using the Open3D functionality ".get_oriented_bounding_box()"</li><li>the convex hull is  calculated by using the Open3D functionality ".compute_convex_hull()"</li><ul>|
| Fractal Dimension of the Point Cloud | global | 3D point cloud | - | It has turned out, that the fractal dimension of the point cloud is dependent on the ratio of the box size that is used during the box counting algorithm and the density of the point cloud. Because of that the use of the fractal dimension was not pursued any further in this project. It could however be possible to find a way on how to adapt the box size according to the density of the point cloud. This is a challenging task, because the point density of the point cloud that represents a window, is not only dependent on the acquisition conditions, but on the shape and structure of a window, too. To find the true density of points that actually represent parts of the building, knowledge about the structure of the window would be necessary.|
| Squareness of the 2D convex hull | global | <ul><li>2D pointcloud (projected 3D point cloud)</li><li>2d image</li></ul> | The 2D squareness of the convex hull measures the perimeter deviation between the 2D convex hull and a square with the same area | Inspired by [Labetski et al., 2022] |
| Squareness of the 2D bounding box | global | 2D pointcloud (projected 3D point cloud) | The 2D squareness of the bounding box measures the perimeter deviation between the 2D convex hull and a square with the same area | Inspired by [Labetski et al., 2022] |
|Circularity of the 2D projected Pointcloud| global | 2D pointcloud (projected 3D point cloud) | -|-|-|
| 2d Convex hull 2d bounding box area ratio | global | 2d image| measures the area deviation of the 2d convex hull and the 2d bounding nox | - |
| 2d Convex hull 2d bounding box perimeter ratio | global | <ul><li>2D pointcloud (projected 3D point cloud)</li><li>2d image</li></ul> | measures the perimeter deviation of the 2d convex hull and the 2d bounding nox | - |
| Cohesion of the 2d convex hull | global | 2d image|  <ul><li>The cohesion is a measure of overall accessibility from all points to others within a polyhedron.</li><li>"The Cohesion Index is the ratio of the average distance-squared among all points in an equalarea circle and the average distance-squared among all points in the shape" [Angel et al., 2010]</li></ul> | - |
| Number of straight lines in the image | global | 2D image | <ul><li>Number of lies detected in the image</li><li>The lines are detected by applying the hough transform</li><li>5 parameters can be varied: <ul><li>The distance resolution in pixels of the Hough grid</li><li>The angular resolution in radians of the Hough grid</li><li>The minimum number of votes (intersections in Hough grid cell)</li><li>The minimum number of pixels making up a line</li><li>The maximum gap in pixels between connectable line segments</li></ul></li></ul> |<ul><li>The different parameters were set by a trial and error approach</li><li>The calculation is done via the OpenVC-Library</li></ul>|
| Histogram of Oriented Gradients | semi-global | 2D - image | - | <ul><li>The calculation is done via the OpenVC-Library</li><li>Based on [Dalal & Triggs, 2005]</li></ul> |
| Hu - Moments | global | 2D - image | <ul><li>"[The Hu moments are a set of] two-dimensional moment invariants for planar geometric figures" [Hu, 1962] </li><li>The Hu-moments are invariant to: <ul><li>Translation</li><li>Rotation</li><li>Scaling</li></ul></li></ul> | <ul><li>The calculation is done via the OpenVC-Library</li><li>Based on [Hu, 1962]</li></ul>|
| Orb-descriptor | local | 2D image | "very fast binary descriptor based on BRIEF [...] which is rotation invariant and resistant to noise" [Rublee et al., 2011] |<ul><li>Only the descriptor is used in this project, the position of the keypoint itsself is not used.</li><li>"two orders of magnitude faster than SIFT" [Rublee et al., 2011]</li><li>The calculation is done via the OpenVC-Library</li><li>Based on [Rublee et al., 2011]</li></ul> |

### Codebook Construction

### Histogram Comparison

The final step in the process that is described in figure 2 is the comparison of the histograms that represent the occurence of the visual words. There are a number of ways in which such a comparison can be realized. For this project several different histogram distances are implemeted. The following table summarizes them: 
<br></br>
| Name|  Formula |Further information |
|--------------|-----------|-----------|
| Minkowski Distance | <ul><li><img src="uploads/77640ae75da3dbfde3dcc8384d1f8a16/Minkowski_Distance.JPG" alt="Introductional_Graphic" width="150"></li><li>[Cha, 2008 ]</li></ul>| In this project, the parameter p if fixed to 2.|
| Chi-Square Distance |<ul><li><img src="uploads/fbcf0ef7c8bca1569fffdf7709659b9f/Chi-Square-Distance.JPG" alt="Introductional_Graphic" width="152"></li><li>[Cha, 2008 ]</li></ul>| For the calculation of the Chi-Square distance, the respective function in the OpenCV Library is used. |
| Kullback-Leibler Divergence |<ul><li><img src="uploads/5275c68f1d9f3b638dcc4db00977ef83/Kullback-Leibler_Divergence.JPG" alt="Introductional_Graphic" width="170"></li><li>[Joyce, 2014]</li></ul>| - |
| Jensen-Shannon Divergence |<ul><li>![Jensen-Shannon_Divergence](uploads/f15a9a3c43700f241270cb6d3b3bae07/Jensen-Shannon_Divergence.JPG)</li><li> [Fuglede & Topsøe, 2004]</li></ul>| D is the Kullback Leibler Divergence |
| Combined Histogram Distance | - | A weighted average of all the previously listed histogram distances.| 

### Implementation

The implemetation is done in Python 3.10. In this project various openly accessible python libraries are used.The following are the most important ones:
<br></br>
<ul>
<li>
Open3D ([➔ website](http://www.open3d.org/)) 
</li>
<li>
OpenCV ([➔ website](https://docs.opencv.org/4.x/index.html))
</li>
<li>
NumPy ([➔ website](https://numpy.org/))
</li>
<li>
Math ([➔ website](https://docs.python.org/3/library/math.html))
</li>
<li>
SkImage ([➔ website](https://scikit-image.org/))
</li>
<li>
PIL ([➔ website](https://python-pillow.org/))
</li>
</ul>

<br></br>
There are a number of different files that contain specific functionalities. 
The following table lists all the implemented files and briefly explains their functionality

| Name|  Functionality |Further information |
|-----------|-----------|-----------|
|main.py|The main file that is used in order to run the code. In this file, the input-, output- and inference directories are defined an |/|
|Normalization.py|This file contains the functionalities that are used in order to achieve translation and scale invariance of a point cloud.|<ul><li>The scaling is set in such a way that the largest extent in a coordinate direction is set to 10</li><li>The trasnslation is performed in such a way that the mean point of the point cloud becomes the new origin of the local coordinate system</li><li>Initially, a funtionality for rotation invariance based on PCA was also implemented here. It was later moitted because of the sensitivity of PCA towards asymmetry and point density variations</li></ul>| 
|PCA.py| This file contains a functionality to perform a Principal Component Analysis (PCA) on a given Point cloud. Input are the points as an numpy array, output are the eigenvectors and their respective eigenvalues. |During the experiments for this project, PCA was not used.|
|twoDImage.py|This file contains the functionalities that are necessary to create binary two dimensional images from the projected pointclouds. These funtionalities include: <ul><li>The image generation. Input: the coordinate axes to be projected to, the image size and the input Open3D point cloud</li><li>A function to calculate the covariance of a two dimensional point cloud (discarded in this project)</li><li>A method that calculates the projection direction with the maximum covariance (discarded for this project, not clear if it actually works)</li><li>A functionality for the manual selection of a projection direction</li><li>The functionalit for the Douglas-Peucker algorithm</li></ul>|The code in the oether files has been adapted in order to work on the manual selection of the best projection directio. It would be a laboreous task to make it work for the automatic projection direction detection|
|Global_Features.py| This file contains functionalities to calculate global features for a given pointcloud. | This file contains a lot of unused and unfinished code.|
|twoDFeatures.py| This file contains the functionalities that are used in order to extract features from the binary images that are created by the functions in the twoDimage.py file. The functions here include: <ul><li>ORB</li><li>SIFT (not used in this project)</li><li>MSER (Not used in this project)</li><li>HOG</li><li>Dilation function</li><li>Laplace function</li><li>Line detection & counting</li><li>Template matching (unused and unfinished)</li><li>Function for the extraction of the Hu-Moments</li><li>the functions for the extraction of the global features from the 2d bounding box and convex hull</li></ul>|/|
|FractalDimension.py| This file contains the functionality to calculate the fractal dimension of a given pointcloud by applying the boxcounting algorithm.|This functionality is not used in this project because of several issues.|
|CodeBook.py| This file contains all the functionalities that are used in order to create the codebook from a set of given model-pointclouds. these functionalities include: <ul><li>A function to create the local feature space and write it into a respective .txt file</li><li>A function to create the global feature space and write it into a respective .txt file</li><li>A functoinality to perform a clustering in the clocal feaure space</li><li>a function do generate the codebook and write it to a respective .txt file</li></ul>|As an interface between the different functions, the .txt file is chosen because of several advantages: <ul><li>Possibility of re-use of single files</li><li>Simplified manipulation of individual files for debugging and testing purposes</li><li>Simplified Archiving of results</li></ul>|
| Inference.py| This file contains all the functionalities that are necessary in order to perform the actual object representations by means of the codebook. The function that are implemented in this file include:<ul><li>A function to represent a point cloud by means of the codebook</li><li>A function vor vector quantization</li><li></li>A function for the histogram comparison that calculates the distance of a given gistogram to all the histograms of the model representations and writes them into the "Result.txt" file.</li><li>auxiliary functions for the histogram distance calculation </li><li>A function that analyzes the result, finds the best matches and writes them into the "Matchings.txt" file.</li></ul>|/|
|ArtificialNoise.py| This file contains a function that applies random noise of a certain defined strength to an input pointcloud | THis functionality is used in the small scale experiments in this project.| 

### Discarded Ideas

<ul><li>Template matching: 

One idea was to count the number of occurences of certain patterns in the 2D binary images. For experimental purposes a template matchin algorithm was implemented that was insteded to match specially designed cross patterns with the binary images an count the numer of found matches. There are several issues why this approach was discarded from this project:

<ul><li>Invariance to rotation and scaling takes a high effort to construct.</li><li>Sensitivity to noise</li><li>For different models, different new templates might have to be crafted.</li></ul>
The following grapic contains a selection of cross patterns that were crafted in order to test this functionality:

<br><br>

<div align="center">
  <img src="uploads/f984e45a9b4931ed1400b4b3d2607cf8/Templates.png" alt="Introductional_Graphic" width="250">
  <br>
  <em>Figure 11: Selection of templates</em>
</div>

</li><li> The number of found Contours

Another feature implemente but not used is to count the number of found contours in the images. The idea here is that distinguishing feature betwee windows with windowbars and windows without widnow bars is overall numer of found contours in the image. For windows with window bars, the number of contours will theoretically be higher than for windowas without window bars. With this idea, the perfomance of the discrimination based on the interior structure of the windows should have been improved. The implementation  of the contour detection and counting is based on the OpenCV library. The results are hevily affected by noise and sparsity, which makes the feature unfit  for use in this project. 

</li><li>

3D Insterest Point Detectors

There is the idea to use 3D Interest point detectors and their respective Descriptors in order to extract local features directly from the 3d point cloud. Examples for such detectors would be Harris-3D [Sipiran & Bustos, 2011], SIFT 3D [Scovanner et al., 2007] or the USIP detector which makes use of deep leraning approaches [Li & Lee, 2019]. This idea was discarded, because the often very irregular shape of the windows from the MLS point clouds does not often correspond to the very regular 3D-shape of the windows that are sampled from the CAD models. This means that the extracted keypoints would most probably not correspond to the keypoints found in the model in most cases. The following graphic illustrates this problem:

<div align="center">
  <img src="uploads/968733d0bcc7ceb74a44a096ded2557b/Symmetrical_and_Asymmetrical.png" alt="Introductional_Graphic" width="250">
  <br>
  <em>Figure 12: Window from MLS point cloud and window sampled from CAD</em>
</div>

</li></ul>

## Experiments

### Small Scale Experiments

<ul>
<li>
The first experiment that is conducted has the goal to identify any grave implementation mistakes that could have been made. In order to assess this, the Inference is done on exactly the same pointclouds that are used in order to create the codebook. In theory, the same features should be extracted from all of these pointclouds, and the resulting matching distances should always be zero while the matching should be 100% correct. Several exeriments for different combinations of features and different histogram distances have shown that this is exactly the case. 
</li>
<li>
The second set of experiments is conducted in order to assess the performans of different features in the presence of noise. Different combinations of features are used and their performance is measured. For the inference, the point clouds that are used to generate the codebook are overlaid with random noise of various strengths. The following graphic schematically describes the experiments:
<br><br>
<div align="center">
  <img src="uploads/136cfb67422d7d08ffecd514bd98dbfa/Noisy_Cloud_Schema.drawio.png" alt="Introductional_Graphic" width="400">
  <br>
  <em>Figure 13: Schema of the second experiment</em>
</div>
<br></br>
As test windows, the same windows as in figure 3 are used. In the test dataset, consisting of the windows with added noise, each of these windowas appears 4 times. That leads to a total number of 32 test windows.
the results of these tests are summarized in the following table:

</li>
</ul>

| Feature combination |  noise level |Histogram Distance | Performance |
|--------------|-----------|-----------|-----------|
| Only local features: ORB | 0.2 | Chi-Square-Distance | ![Confusion_Matrix_1__1_.drawio_1_](uploads/95fe34fcc165608372583ec6c7d0d78d/Confusion_Matrix_1__1_.drawio_1_.png)<ul><li>Overall Accuracy: 0.469</li><li>Kappa-Coefficient: 0.356 (~ "Weak")</li></ul>| It can be seen that the performance of the whole method is very bad. |
| Only global features: HOG | 0.2 | Chi-Square-Distance |![Confusion_Matrix_1__1_.drawio_2_](uploads/7139fe61fa1a3c49313d0cf4652aaccc/Confusion_Matrix_1__1_.drawio_2_.png)<ul><li>Overall Accuracy: 0.750</li><li>Kappa-Coefficient: 0.714 (~ "Good")</li></ul>|
| <ul><li>ORB</li><li>HOG</li></ul>| 0.2 | Chi-Square-Distance |![Confusion_Matrix_1__1_.drawio](uploads/0ce2b760c045f823a3dce5a31e30f678/Confusion_Matrix_1__1_.drawio.png)<ul><li>Overall Accuracy: 0.688</li><li>Kappa-Coefficient: 0.646 (~ "Good")</li></ul>|
| <ul><li>ORB</li><li>HOG</li></ul>| 0.4 | Chi-Square-Distance |![Unbenanntes_Diagramm.drawio](uploads/dacb1f9cb840855a879bb90b9c48359f/Unbenanntes_Diagramm.drawio.png)<ul><li>Overall Accuracy: 0.500</li><li>Kappa-Coefficient: 0.428 (~ "Moderate")</li></ul>|
| <ul><li>ORB</li><li>HOG</li><li>Hu-Moments</li></ul>| 0.2 |Chi-Square-Distance|![Confusion_Matrix_1__1_.drawio_2_](uploads/c20ad2712d648ecd6c6eff8b1b31c193/Confusion_Matrix_1__1_.drawio_2_.png)<ul><li>Overall Accuracy: 0.781</li><li>Kappa-Coefficient: 0.750 (~ "Good")</li></ul>|
| <ul><li>ORB</li><li>HOG</li><li>Hu-Moments</li></ul>| 0.4 |Chi-Square-Distance|![Confusion_Matrix_1__1_.drawio_2_](uploads/084fdbafe5c6db4ccf75f807b9547376/Confusion_Matrix_1__1_.drawio_2_.png)<ul><li>Overall Accuracy: 0.375</li><li>Kappa-Coefficient: 0.278 (~ "Weak")</li></ul>|

### Discussion of the Small Scale Experiments
<ul>
<li>
The first thing that can be observed is that the quality of the matching rapidly decreases with the noise level. Generally, the experiments that were conducted on the higher noise level show a lower overall accuracy.  
</li>
<br></br>
<li>
Apparently the matching is generally more stable and better for some windows. For others, there are large differences in the matching itsself as well as in the quality of the matching. This observation is strengthened by the examination of the user's and producer's accuracy of the individual window types. the following to tables list the user's and producer's accuracies for the individual window types. Each row of the table represets a single experiment. the order of the experiments is the same as in the table above.
<br><br>
<div align="center">
  <img src="uploads/9cd6b4d96a99bc20026473befd7ff651/Users-Accuracy.png" alt="Introductional_Graphic" width="500">
  <br>
</div>
<br></br>
<br><br>
<div align="center">
  <img src="uploads/9f50e0b9052e36377c197f7c9c226fde/Producers-Accuracy.png" alt="Introductional_Graphic" width="500">
  <br>
</div>
<br></br>
This table shows that the arched window with no bars and the two octagon-shaped windows are  matched in the most stable manner compared to the other window types. In general, the rectangular and quadratic windows show less good user's and producers accuracies in these experiments. A look into the variances and standard deviations of the user's and pproducer's accuracies that are listed in the table below, strengthen this observation.
<br><br>
<div align="center">
  <img src="uploads/e9a53d9651d00db38e1d6cec5a1e678c/Unbenanntes_Diagramm.drawio.png" alt="Introductional_Graphic" width="650">
  <br>
</div>
<br></br>
The reason for this might be that the windows that are detected in a more stable way (the arched window with no cross and the octagon shaped windows) have properties that can be described by the used features in a more precise way than the other windows. The windows, for which the performance is weaker, probably have properties that can be described by the used features in a less precise way. Therfore the description of these windows will probably have a very similar description that clould lead to a wrong classification in the presence of noise.
</li>
<br></br>
<li>
The perormance of the local features (ORB) on it's own is very poorly. If the local features are augmented by incorporating the (semi-) global hog features into the process, the overall performance is increased.
</li>
</ul>

## Experiments on the TUM Facade Dataset

The next set of experiments is performed on a set of 42 windows that are extracted from the TUM-Facade dataset. In this dataset only large rectangular and arched windows occur.
<br></br>
| Feature combination |Histogram Distance | Dataset Type | Performance |
|--------------|-----------|-----------|--------------|
|<ul><li>HOG</li></ul>| Jensen-Shannon Divergence | no molding | ![Unbenanntes_Diagramm.drawio](uploads/8be5f48f5029c6f99fbfa7e6caa42e78/Unbenanntes_Diagramm.drawio.png)<ul><li>Overall Accuracy: 0.36</li></ul>
|<ul><li>ORB</li><li>HOG</li></ul>|Minkowski-Distance| no molding |![Unbenanntes_Diagramm.drawio](uploads/200566b3f61ae26296f0db47eb233a11/Unbenanntes_Diagramm.drawio.png)<ul><li>Overall Accuracy: 0.405</li></ul>|
|<ul><li>ORB</li><li>HOG</li><li>Douglas-Peucker tolerance set from 10 to 7</li></ul>|Jensen-Shannon Divergence| no molding |![Unbenanntes_Diagramm.drawio](uploads/51228f872faa8506070e6d6b91b54b06/Unbenanntes_Diagramm.drawio.png)<ul><li>Overall Accuracy: 0.52</li></ul>|
|<ul><li>ORB</li><li>HOG</li></ul>|Jensen-Shannon Divergence| no molding |![Unbenanntes_Diagramm.drawio](uploads/8096f9ba78bd04f95c68d77f39429500/Unbenanntes_Diagramm.drawio.png)<ul><li>Overall Accuracy: 0.57</li></ul>|
|<ul><li>ORB</li><li>HOG</li></ul>|Jensen-Shannon Divergence| with molding |![Unbenanntes_Diagramm.drawio](uploads/6078a6ede2020dc7f6c9c587d86ea41c/Unbenanntes_Diagramm.drawio.png)<ul><li>Overall Accuracy: 0.57</li></ul>|
|<ul><li>ORB</li><li>HOG</li><li>Squareness of the 2D projected Pointcloud</li><li>Circularity of the 2D projected Pointcloud</li><li>Perimeter ratio of 2d Convex Hull and 2d Bounding Box</li></ul>|Jensen-Shannon Divergence| with molding |![Unbenanntes_Diagramm.drawio](uploads/a01e156ec9f24da40ccba94a168496d4/Unbenanntes_Diagramm.drawio.png)<ul><li>Overall Accuracy: 0.524</li></ul>|

<br></br>
It has to be mentioned that an experiment where only local features (ORB) were used is not inluded in this list. The reason is that the performance was very poorly. The use of different histogram distances lead towards strong biases towards certain window types in this experiment.

## Discussion of the Experiments on the TUM Facade Dataset

<ul>
<li>Quality strongly dependent on:
<ul>
<li>Choice of features. The best results are achieved by using a combination von local features and (semi-) global HOG features.</li>
<li>Choice histogram distance. the best results are achieved by using the jensen shannon divergence</li>
<li>Filter radius & number of neighbours. In this project, the filter radius  that is used during the outlier removal is set to 0.5. The number of neighbouring points is set to 16</li>
<li>Douglas-Peucker tolerance. The best results were achieved with a douglas Peucker tolerance of 15</li>
<li>Number of K-Means clusters. In this project, 25 clusters are usen in any of the experiments.</li>
<li>Dilation kernel size. The best results were achieved by using a dilation kernel size of 20 pixels</li>
<li>Vector quantization method. In this project, only the minkowski distance is used for the vector quantization For this the parameter p is set to 2.</li>
</ul>
</li>
<br></br>
<li>
An investigation in the position of the windows in the facade reveals that, in the experiment with the best performnce, the falsely identified windows are not distributed randomly. The following graphic shows that most of the falsly classified windows are located in the second floor of the building:
<br><br>
<div align="center">
  <img src="uploads/2e2123a20c853dff3e14a6236f025e0b/TUM_Facade_Performance.png" alt="Introductional_Graphic" width="800">
  <br>
  <em>Figure 14: Correctly and falsely matched windows in the facade</em>
</div>
<br></br>

The reason for this might be that the point density of the MLS point cloud is lower in the upper parts of the facade due to the higher distance from the sensor. This, in combination with assumingly unfavorable parameter settings during the outlier removal, might lead to the loss of structures that are important for the correct identification of the window. The following graphic supports this assumption. It shows the binary image that is created from the 4th window from the right in the second floor of the building:
<br><br>
<div align="center">
  <img src="uploads/8023b9b15946934dfc66bcc48b3c1da8/Bad_Window_Comparison.png" width="400">
  <br>
  <br>
  <em>Figure 15: Sparse window from the second floor</em>
</div>
<br></br>

</li>
</ul>
<br></br>

## Outlook

There are many ways in which this project can be extended improved and further developed.Future work could focus on various different aspects. 

There is the possibility of performing experiments on larger stets of data. This way, deeper insight into the behaviour of the approach under a larger variation of circumstances could be gained. This project focuses on the concept itsself, and experimets are only conducted on a relatively small scale. Therfore the reliability of the insight that is gained by the analysis of the different experiments is relatively limited. Reliable statistical data would be one of the foundations of a further development of the approach introduced in this project. Future work could help to set this foundation. 

Another objective could be to minimize the influence of noise and sparsity of the MLS Point clouds. The quality of the resulting matching is, as the experiments that are conducted in this projects show, largely dependent on the chosen parameters. Future work could also aim at optimizing the choice of parameters in order to maximize the matching quality with respect to these two disturbing factors. A way to achieve this could be by focusing on the "scene gist" or in other words on more high level information, rather than on very detailed low level information.

A further possibility to increase the performance of the approach could be to use a dense grid of keypoints in order to extract local features instead of using interest point detectors like ORB or SIFT.

Right now the whole approach is implemented in a very experimental way. Future work could also aim at making the concept mor applyable for everyday use. This might also invole implementation in lower level programming language to increase the temporal performance of the whole process. The construction of a sophisticated user interface could also contribute to further development of this approach.

There is also the possibility of altering this approach, or to use a related approach instead. For example the approach could be modified in a way, as describen in the following diagram:
<br><br>
<div align="center">
  <img src="uploads/cd028c30c0b6f0847cff0206fb9f6f52/Unbenanntes_Diagramm.drawio_6_.png" width="700">
  <br>
  <em>Figure 16: Schematic Description of an altered version of the apprach</em>
</div>
<br></br>

This approach could be used to reduce computation time for larger datasets and databases. The approach that is used in this project has the disadvantage that there has to be a similarity comparison between every of the N windows in the dataset and every of the M windows in the CAD library (N * M  histogram distance calculations). The computational effort will become very large for larger datasets and databases. The number of histogram distance calculations in the approach that is introduced in the previous graphic will be just (L * M) with L being the number of window types in the dataset. This number will generally be smaller, because we can assume that for a the assumption L < N holds for any sensible clustering algorithm. In th worst case, where there are no identical windows in the dataset, there will still maximally be (L * M) = (N * M) histogram distance calculations necessary. In the best case, when there is just one type of window in a dataset, there will be just one histogram distance calculation necessary.


## List of Figures
<ul>
<li>
Figure 1: General overview, based on [Memon et al., 2019]
</li>
<li>
Figure 2: Schematic diagram of the used methodology
</li>
<li>
Figure 3: Incorporation of global features
</li>
<li>
Figure 4: Edited CAD window models
</li>
<li>
Figure 5: Sampled CAD Model
</li>
<li>
Figure 6: Projected imaged from all 3 different views
</li>
<li>
Figure 7: Dilated Image
</li>
<li>
Figure 8: Laplacian Image
</li>
<li>
Figure 9: Application of the Douglas-Peucker Algorithm
</li>
<li>
Figure 10: Extraction of meaningful interest points
</li>
<li>
Figure 11: Selection of templates
</li>
<li>
Figure 12: Window from MLS point cloud and window sampled from CAD
</li>
<li>
Figure 13: Schema of the second experiment
</li>
<li>
Figure 14: Schematic diagram of the used methodology
</li>
<li>
Figure 15: Sparse window from the second floor
</li>
<li>
Figure 16: Schematic Description of the an altered version of the apprach
</li>
</ul>

## References
<ul>
<li>Bronstein AM, Bronstein MM, Guibas LJ, Ovsjanikov M (2011) Shape google: geometric words and expressions for invariant shape retrieval. ACM Transactions on Graphics 30(1): 1-20
</li>
<li>
Cha SH (2008) Taxonomy of nominal type histogram distance measures. In: Long C, Sohrab SH (eds). AMERICAN CONFERENCE ON APPLIED MATHEMATICS (MATH '08), Harvard, Massachusetts: 325-330
</li>
<li>
Csurka G, Dance CR, Fan L, Willamowski J, Bray C (2004) Visual categorization with bags of keypoints. Workshop on statistical learning in computer vision, ECCV. 1(1-22): 1-2
</li>
<li>
Douglas DH, Peucker TK (1973) Algorithms for the reduction of the number of point required to represent a digitized line or its caricature. Cartographica: the international journal for geographic information and geovisualization 10(2): 112-122
</li>
<li>
Fuglede B, Topsøe (2004) Jensen-Shannon divergence and Hilbert space embedding. In International Symposium on Information Theory, 2004, Chicago, IL, USA. IEEE: 1-6
</li>
<li>
Hu MK (1962) Visual pattern recognition by moment invariants. IRE Transactions on Information Theory 8(2): 179-187
</li>
<li>
Joyce JM (2011) Kullback-Leibler Divergence, International Encyclopedia of Statistical Science. Springer, Berlin, Heidelberg: 720-722
</li>
<li>
Kang Z, Yang J (2018) A probabilistic graphical model for the classification of mobile LiDAR point clouds. ISPRS Journal of Photogrammetry and Remote Sensing 143 (2018): 108-123
</li>
<li>
Labetski A, Vitalis S, Bijecki F, Ohori KA, Stoter J (2022) 3D building metrics for urban morphology. International Journal of Geographical Information Science, 37(1): 36-67.
</li>
<li>
Martens J, Blankenbach J (2020) An evaluation of pose-normalization algorithms for point clouds introducing a novel histogram-based approach. Advanced Engineering Informatics 46(2020): 101132
</li>
<li>
Memon SA, Mahmood T, Akhtar F, Azeem M, Shaukat Z (2019) 3D shape retrieval using bag of words approaches. In 2019 2nd International Conference on Computing, Mathematics and Engineering Technologies (iCoMET).Sukkur, Pakistan: IEEE: 1-7
</li>
<li>
Li J, Lee GH (2019) USIP: unsupervised stable interest point detection from 3d point clouds. In: Lee KM, Forsyth T, Pollefeys M, Tang X (eds). Seoul, Korea: IEEE: 361-370
</li>
<li>
Lian Z, Godil A, Sun X (2010) Visual similarity 3d shape retrieval using bag-of-features. In: Pernot JS, Spagnuolo M, Falcidieno B, Veron P (eds) 2010 Shape modelling International Conference. Los Alamitos: IEEE: 25-36 
</li>
<li>
Salton G, McGill MJ (1986) Introduction to Modern Information Retrieval. New York, NY, USA: McGraw-Hill, Inc.
</li>
<li>
Scovanner P, Saad A, Saha M (2007) A 3-dimensional sift descriptor and its application to action recognition. In Lienhart R, Prasad AR (eds) 15th ACM international conference on Multimedia. Augsburg, Germany. Association for Computing Machinery, New York, NY, United States: 631-640
</li>
<li>
Sipiran I, Bustos B (2011) Harris 3D: a robust extension of the Harris operator for interest point detection on 3D meshes. The Visual Computer 27(2011): 963–976
</li>
<li>
Trimble Inc (2021) SketchUp 3D Warehouse. https://3dwarehouse.sketchup.com/ (4 February 2021)
</li>
<li>
Wysocki O, Hoegner L, Stilla U (2022) TUM-Façade: Reviewing and Enriching Point Cloud Benchmarks for Façade Segmentation. Int. Arch. Photogramm. Remote Sens. Spatial Inform. Sci. XLVI-2/W1-2022, 529–536
</li>
<li>
Zhu Q, Zhong Y, Zhao B, Xia GS, Zhang L (2016) Bag-of-visual-words scene classifier with local features for high spatial resolution remote sensing imagery. IEEE Geoscience and Remote Sensing Letters 13(3): 747-751
</li>
</ul>

