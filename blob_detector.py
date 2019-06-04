import cv2
import numpy as np

from config import *

def detect_blobs(input_image, staffs):
    if VERBOSE:
        print("Detecting blobs.")
    im_with_blobs = input_image.copy() #複製圖像

    im_inv = (255 - im_with_blobs) # 最大图像灰度值减去原图像，即可得到反转的图像（黑白對調）
    # kernel通过将其与不同数量的相邻像素组合来告诉如何更改任何给定像素的值
    kernel = cv2.getStructuringElement(ksize=(1, int(im_inv.shape[0] / 500)), shape=cv2.MORPH_RECT) 
    horizontal_lines = cv2.morphologyEx(im_inv, cv2.MORPH_OPEN, kernel) #morphologyEx可以理解為消除物體之外的區域
    horizontal_lines = (255 - horizontal_lines)

    kernel = cv2.getStructuringElement(ksize=(int(im_inv.shape[1] / 350), 1), shape=cv2.MORPH_RECT)
    vertical_lines = cv2.morphologyEx(255 - horizontal_lines, cv2.MORPH_OPEN, kernel)
    vertical_lines = (255 - vertical_lines)

    if SAVING_IMAGES_STEPS:
        cv2.imwrite("output/8a_lines_horizontal_removed.png", horizontal_lines)
        cv2.imwrite("output/8a_lines_vertical_removed.png", vertical_lines)

    im_with_blobs = vertical_lines
    im_with_blobs = cv2.cvtColor(im_with_blobs, cv2.COLOR_GRAY2BGR)
    print(im_with_blobs)
    #基本上就是一堆設定可以讓黑黑的部分被圈 標記起來
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 225
    params.maxArea = 1500
    params.filterByCircularity = True
    params.minCircularity = 0.6
    params.filterByConvexity = True
    params.minConvexity = 0.9
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    detector = cv2.SimpleBlobDetector_create(params)
    print(detector)
    keypoints = detector.detect(im_with_blobs)
    print(keypoints)
    cv2.drawKeypoints(im_with_blobs, keypoints=keypoints, outImage=im_with_blobs, color=(0, 0, 255),
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    if SAVING_IMAGES_STEPS:
        cv2.imwrite("output/8b_with_blobs.jpg", im_with_blobs)

    staff_diff = 3 / 5 * (staffs[0].max_range - staffs[0].min_range)
    bins = [x for sublist in [[staff.min_range - staff_diff, staff.max_range + staff_diff] for staff in staffs] for x in
            sublist]
    bins.sort()
    print(bins)
    keypoints_staff = np.digitize([key.pt[1] for key in keypoints], bins)
    print(keypoints_staff)
    sorted_notes = sorted(list(zip(keypoints, keypoints_staff)), key=lambda tup: (tup[1], tup[0].pt[0]))

    im_with_numbers = im_with_blobs.copy()

    for idx, tup in enumerate(sorted_notes):
        cv2.putText(im_with_numbers, str(idx), (int(tup[0].pt[0]), int(tup[0].pt[1])),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(255, 0, 0))
        cv2.putText(im_with_blobs, str(tup[1]), (int(tup[0].pt[0]), int(tup[0].pt[1])),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(255, 0, 0))
    if SAVING_IMAGES_STEPS:
        cv2.imwrite("output/8c_with_numbers.jpg", im_with_numbers)
        cv2.imwrite("output/8d_with_staff_numbers.jpg", im_with_blobs)

    if VERBOSE:
        print("Keypoints length : " + str(len(keypoints)))

    print(sorted_notes)
    return sorted_notes