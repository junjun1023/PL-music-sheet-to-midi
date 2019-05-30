import numpy as np

"""
    Returns distance between two points in cartesian coordinate system.
"""
def distance(point1, point2):

    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)