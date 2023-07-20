# Todo: IMPORTANT: This Code was written by the Chat GPT
# Todo: that was developed by OpenAI (See: https://chat.openai.com)

import numpy as np


def box_count(points, min_resolution, max_resolution, num_steps):
    """
    Calculates the fractal dimension of a point cloud using the Box-Counting method.

    Parameters
    ----------
    points : array-like
        The point cloud, represented as an NxD array, where N is the number of points and D is the dimensionality.
    min_resolution : float
        The minimum box size to use.
    max_resolution : float
        The maximum box size to use.
    num_steps : int
        The number of steps to use between the minimum and maximum box sizes.

    Returns
    -------
    resolutions : array-like
        The size of the boxes used, in the same order as the number of points.
    num_boxes : array-like
        The number of boxes with at least one point.
    """
    # print("Points: ", np.shape(points))
    # Generate an array of box sizes to use
    resolutions = np.logspace(np.log10(min_resolution), np.log10(max_resolution), num_steps)
    # print("Resolution: ", np.shape(resolutions))
    # Initialize array to store the number of boxes with at least one point
    num_boxes = np.zeros(num_steps)

    for i, resolution in enumerate(resolutions):
        # Create a grid of boxes
        #print(points/resolution)
        grid = np.ceil(points / resolution)

        # Count the number of boxes with at least one point
        num_boxes[i] = len(np.unique(grid, axis=0))

    return resolutions, num_boxes


def calc_fractal_dimension(resolutions, num_boxes):
    """
    Calculates the fractal dimension of a point cloud from the resolutions and number of boxes.

    Parameters
    ----------
    resolutions : array-like
        The size of the boxes used, in the same order as the number of points.
    num_boxes : array-like
        The number of boxes with at least one point.

    Returns
    -------
    fractal_dimension : float
        The fractal dimension of the point cloud.
    """
    # Convert resolutions and num_boxes to logarithmic scale
    log_resolutions = np.log(resolutions)
    log_num_boxes = np.log(num_boxes)

    # Fit a straight line to the log-log plot using linear regression
    slope, intercept = np.polyfit(log_resolutions, log_num_boxes, 1)

    # The fractal dimension is the slope of the fitted line
    fractal_dimension = -slope

    return fractal_dimension
