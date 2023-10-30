# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np

binary_image = cv2.imread('Binary_scan.png', cv2.IMREAD_GRAYSCALE)

#img = cv2.imread('grayscale_image.png')

grayscale_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)


orb = cv2.ORB_create(50)

kp, des = orb.detectAndCompute(grayscale_image, None)

print("The set of keypoints are:\n")

print(len(kp))

img2 = cv2.drawKeypoints(grayscale_image, kp, None, flags=None)

cv2.imwrite('Keypoints_scan_ORB.jpg', img2)
cv2.imwrite('Keypoints_scan_ORB.png', img2)

cv2.imshow("ORB", img2)

cv2.waitKey(0)