# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 11:03:10 2023

@author: HP
"""
'''
import cv2
import numpy as np

# Load the binary 2D image (white object on black background)
image = cv2.imread('gray.jpg', cv2.IMREAD_GRAYSCALE)

# Calculate the distance transform (Euclidean distance)
dist_transform = cv2.distanceTransform(image, distanceType=cv2.DIST_L2, maskSize=5)

# Invert the distance map to get the signed distance field
sdf = -dist_transform

print(np.max(sdf))

# Normalize the SDF to the range [-1, 1]
sdf = sdf

# Now, the 'sdf' variable contains the signed distance field of the object in the image.
'''

'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the binary image
binary_map = cv2.imread('Binary_map.png', 0)
binary_scan = cv2.imread('Binary_scan.png', 0)

# Calculate the distance transform
dist_transform = cv2.distanceTransform(binary_map, distanceType=cv2.DIST_L2, maskSize=5)
dist_transform_ = cv2.distanceTransform(binary_scan, distanceType=cv2.DIST_L2, maskSize=5)

# Invert the distance transform
sdf_map = -dist_transform
sdf_scan = -dist_transform_

# Optional: Normalize the SDF values to a specific range (e.g., -1 to 1)
normalized_sdf = (sdf_map - sdf_map.min()) / (sdf_map.max() - sdf_map.min()) * 2 - 1
normalized_sdf_ = (sdf_scan - sdf_scan.min()) / (sdf_map.max() - sdf_scan.min()) * 2 - 1

# Display the SDF map
plt.imshow(normalized_sdf, cmap='gray')
plt.colorbar()
plt.show()

plt.imshow(normalized_sdf_, cmap='gray')
plt.colorbar()
plt.show()
'''

################################### SIFT MAP AND SCAN GOO ##############################

import cv2 
import numpy as np 
  
# Load the input image and make it grayscale. 
image_scan = cv2.imread('grayscale_scan_0.09.png') 
gray_scan = cv2.cvtColor(image_scan, cv2.COLOR_BGR2GRAY) 

image_map = cv2.imread('grayscale_map_0.09.png') 
gray_map = cv2.cvtColor(image_map, cv2.COLOR_BGR2GRAY)
  
# Create a binary image by throttling the image. 
ret1, thresh1 = cv2.threshold(gray_scan, 105, 255, cv2.THRESH_BINARY) 
ret2, thresh2 = cv2.threshold(gray_map, 105, 255, cv2.THRESH_BINARY) 
  
#Determine the distance transform.  [cv2.DIST_L2 \\ cv2.DIST_LABEL_PIXEL]
dist_scan = cv2.distanceTransform(thresh1, cv2.DIST_L2, 0) 
dist_map = cv2.distanceTransform(thresh2, cv2.DIST_L2, 0) 
  
# Make the distance transform normal. 
dist_output_scan = cv2.normalize(dist_scan, None, 0, 1.0, cv2.NORM_MINMAX) 
dist_output_map = cv2.normalize(dist_map, None, 0, 1.0, cv2.NORM_MINMAX)

# Initialize the SIFT detector
sift = cv2.SIFT_create()

#######################################################################
# Convert the dist_output to an 8-bit, single-channel image
dist_output_scan = (dist_output_scan * 255).astype(np.uint8)
dist_output_map = (dist_output_map * 255).astype(np.uint8)
#######################################################################
# Detect and compute keypoints and descriptors in the distance transform image
kp1, des1 = sift.detectAndCompute(dist_output_scan, None)
kp2, des2 = sift.detectAndCompute(dist_output_map, None)

# Draw the keypoints on the distance transform image
output_image_scan = cv2.drawKeypoints(dist_output_scan, kp1, outImage=None)
output_image_map = cv2.drawKeypoints(dist_output_map, kp2, outImage=None)


###############################################################################
# create BFMatcher object [cv2.NORM_L2 \\ cv2.NORM_HAMMING]
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.
img4 = cv2.drawMatches(output_image_scan,kp1,output_image_map,kp2,matches[:15],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

###############################################################################
cv2.imshow('MATCHED', img4)0

# Display the result
cv2.imshow('SIFT on Distance Transform SCAN', output_image_scan)
  
# Display the distance transform 
cv2.imshow('Distance Transform SCAN', dist_output_scan) 

cv2.imshow('ORIGINAL SCAN', image_scan)

##############################################################################
# Display the result
cv2.imshow('SIFT on Distance Transform MAP', output_image_map)
  
# Display the distance transform 
cv2.imshow('Distance Transform MAP', dist_output_map) 

cv2.imshow('ORIGINAL MAP', image_map)
cv2.waitKey(0) 

###############################################################################
