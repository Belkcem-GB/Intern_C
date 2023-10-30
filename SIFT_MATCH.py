# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 21:54:54 2023

@author: HP
"""

# import required libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

# read two input images as grayscale
img1 = cv2.imread('Binary_scan.png',0)
img2 = cv2.imread('Binary_map.png',0)

# Initiate SIFT detector
sift = cv2.SIFT_create()

# detect and compute the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher()

# Match descriptors.
matches = bf.match(des1,des2)

# sort the matches based on distance
matches = sorted(matches, key=lambda val: val.distance)

# Draw first 50 matches.
out = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)


cv2.imwrite('Keypoints_map_SIFT_matched.png', out)

cv2.waitKey(0)
#plt.imshow(out), plt.show()