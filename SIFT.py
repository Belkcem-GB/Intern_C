# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 12:09:47 2023

@author: HP
"""

import cv2

# Loading the image
img_binary_scan = cv2.imread('Binary_scan.png')
img_binary_map = cv2.imread('Binary_map.png')

# Converting image to grayscale
gray_scan= cv2.cvtColor(img_binary_scan,cv2.COLOR_BGR2GRAY)
gray_map= cv2.cvtColor(img_binary_map,cv2.COLOR_BGR2GRAY)

# Applying SIFT detector
sift = cv2.SIFT_create(50)
kp = sift.detect(gray_scan, None)

sift_2 = cv2.SIFT_create(50)
kp_2 = sift.detect(gray_map, None)

# Marking the keypoint on the image using circles
img_binary_scan_KP =cv2.drawKeypoints(gray_scan ,
					kp ,
					img_binary_scan ,
					flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

img_binary_map_KP =cv2.drawKeypoints(gray_map ,
					kp_2 ,
					img_binary_map ,
					flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

print("The set of keypoints ftom the map are:\n")

print(len(kp_2))

print("\nThe set of keypoints ftom the scan are:\n")

print(len(kp))

#save the image scan with drawed keypoints
cv2.imwrite('Keypoints_scan_SIFT.jpg', img_binary_scan_KP)
cv2.imwrite('Keypoints_map_SIFT.jpg', img_binary_map_KP)

