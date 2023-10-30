# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


binary_image_SCAN = cv2.imread('Binary_scan.png', cv2.IMREAD_GRAYSCALE)
binary_image_MAP = cv2.imread('Binary_map.png', cv2.IMREAD_GRAYSCALE)

#img = cv2.imread('grayscale_image.png')

grayscale_image_SCAN = cv2.cvtColor(binary_image_SCAN, cv2.COLOR_GRAY2BGR)
grayscale_image_MAP = cv2.cvtColor(binary_image_MAP, cv2.COLOR_GRAY2BGR)

orb_scan = cv2.ORB_create(50)
orb_map = cv2.ORB_create(50)


kp, des = orb_scan.detectAndCompute(grayscale_image_SCAN, None)
kp1, des1 = orb_scan.detectAndCompute(grayscale_image_MAP, None)

print("The number keypoints in SCAN are:\n")

print(len(kp))

print("\n The number keypoints in SCAN are:\n")

print(len(kp1))

img2 = cv2.drawKeypoints(grayscale_image_SCAN, kp, None, flags=None)
img3 = cv2.drawKeypoints(grayscale_image_MAP, kp, None, flags=None)

cv2.imwrite('Keypoints_scan_ORB.jpg', img2)
cv2.imwrite('Keypoints_scan_ORB.png', img2)


cv2.imwrite('Keypoints_map_ORB.jpg', img3)
cv2.imwrite('Keypoints_map_ORB.png', img3)

###############################################################################
# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des,des1)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.
img4 = cv2.drawMatches(img2,kp,img3,kp1,matches[:15],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

###############################################################################
cv2.imshow("ORB_scan", img2)
cv2.imshow("ORB_map", img3)
cv2.imshow("ORB_match", img4)

cv2.imwrite('Keypoints_map_ORB_matched.png', img4)

cv2.waitKey(0)

#plt.imshow(img4),plt.show()