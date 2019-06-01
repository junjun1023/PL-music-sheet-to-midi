import cv2
import numpy as np

import config

"""
    Prepares the image for the next transformation. Applies threshold and canny edge detection.
    return: Matrices of image after each step.
"""
def preprocess_image(image):
    if config.VERBOSE:
        print("Preprocessing image.")
    gray = image.copy()
    _, thresholded = cv2.threshold(gray, config.THRESHOLD_MIN, config.THRESHOLD_MAX, cv2.THRESH_BINARY)
    element = np.ones((3, 3))
    thresholded = cv2.erode(thresholded, element)
    edges = cv2.Canny(thresholded, 10, 100, apertureSize=3)
    return edges, thresholded

"""
    Detects lines present in the picture and adds ones that are horizontal enough to a list.
    :param hough: result of Hough Transform function.
    :param image: Main image
    :param nlines: How many lines we want to process.
    :return: A list of horizontal lines and an image with lines drawn on it.
"""
def detect_lines(hough, image, nlines):
    if config.VERBOSE:
        print("Detecting lines.")
    all_lines = set()
    width, height = image.shape
    # convert to color image so that you can see the lines
    lines_image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for result_arr in hough[:nlines]:
        rho = result_arr[0][0]
        theta = result_arr[0][1]
        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a * rho
        y0 = b * rho
        shape_sum = width + height
        x1 = int(x0 + shape_sum * (-b))
        y1 = int(y0 + shape_sum * a)
        x2 = int(x0 - shape_sum * (-b))
        y2 = int(y0 - shape_sum * a)

        start = (x1, y1)
        end = (x2, y2)
        diff = y2 - y1
        if abs(diff) < config.LINES_ENDPOINTS_DIFFERENCE:
            all_lines.add(int((start[1] + end[1]) / 2))
            cv2.line(lines_image_color, start, end, (0, 0, 255), 2)

    if config.SAVING_IMAGES_STEPS:
        cv2.imwrite("output/5lines.png", lines_image_color)

    return all_lines, lines_image_color