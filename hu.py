import cv2
import numpy as np
from numpy import linalg

"""
    Gets the clef from the first staff.
    param image: image to get the clef from
    param staff: First staff from the image.
    return:
"""
def get_clef(image, staff):
    width = image.shape[0]

"""
    Uses Hu moments to classify the clef - violin or bass.
    return: A string indicating the clef
"""
