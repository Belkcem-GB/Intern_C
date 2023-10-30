import numpy as np
import cv2
import matplotlib.pyplot as plt
'''
# Read the input images as grayscale
img1 = cv2.imread('Binary_scan.png', 0)
img2 = cv2.imread('Binary_map.png', 0)

# Apply Gaussian blur to reduce noise
img1_blurred = cv2.GaussianBlur(img1, (5, 5), 0)
img2_blurred = cv2.GaussianBlur(img2, (5, 5), 0)

# Initiate SIFT detector
sift = cv2.SIFT_create()

# Detect and compute the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1_blurred, None)
kp2, des2 = sift.detectAndCompute(img2_blurred, None)

# Create BFMatcher object
bf = cv2.BFMatcher()

# Match descriptors
matches = bf.match(des1, des2)

# Sort the matches based on distance
matches = sorted(matches, key=lambda val: val.distance)

# Draw first 50 matches
out = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)

cv2.imwrite('Keypoints_(5X5_)SIFT_matched_GAUSSIAN_BLUR.png', out)

# Find good matches
good_matches = [m for m in matches if m.distance < 0.15 * matches[0].distance]
print("GOOD matches:", len(good_matches))

# Check if you have enough good matches to calculate the homography
if len(good_matches) >= 4:
    # Get the source and destination points for the homography
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Calculate the homography matrix
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warp the first image onto the second
    h, w = img1.shape
    warped_img1 = cv2.warpPerspective(img1, M, (w + img2.shape[1], h + img2.shape[0]))

    # Combine the two images
    result = np.copy(warped_img1)
    result[0:img2.shape[0], 0:img2.shape[1]] = img2

    # Save the stitched image
    cv2.imwrite('Stitched_Image.png', result)

    cv2.imshow('Stitched Image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Not enough good matches to create a stitched image.")
'''
##############################################################################################
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 00:32:22 2023

@author: HP
"""
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Read the input images as grayscale
img1 = cv2.imread('Binary_scan.png', 0)
img2 = cv2.imread('Binary_map.png', 0)

# Apply Gaussian blur to reduce noise
img1_blurred = cv2.GaussianBlur(img1, (5, 5), 0)
img2_blurred = cv2.GaussianBlur(img2, (5, 5), 0)

# Initiate SIFT detector
sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)

# Detect and compute the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1_blurred, None)
kp2, des2 = sift.detectAndCompute(img2_blurred, None)

# Create BFMatcher object
bf = cv2.BFMatcher()

# Match descriptors
matches = bf.match(des1, des2)

# Sort the matches based on distance
matches = sorted(matches, key=lambda val: val.distance)

# Draw first 50 matches
out = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)

cv2.imwrite('Keypoints_(__5X5)SIFT_matched_GAUSSIAN_BLUR.png', out)

cv2.waitKey(0)
# plt.imshow(out), plt.show()
'''
###############################################################################################
import numpy as np
import cv2

# Load the two input images
img1 = cv2.imread('Binary_scan.png', 0)
img2 = cv2.imread('Binary_map.png', 0)

# Apply Gaussian blur to reduce noise
img1_blurred = cv2.GaussianBlur(img1, (5, 5), 0)
img2_blurred = cv2.GaussianBlur(img2, (5, 5), 0)

# Initialize variables to track the best parameter values
best_num_keypoints = 0
best_nOctaveLayers = 0
best_contrastThreshold = 0
best_edgeThreshold = 0
best_sigma = 0
best_stitched_image = None
best_good_matches = []  # Track the best set of good matches

# Loop through different parameter combinations
for num_keypoints in [0, 500, 1000]:
    for nOctaveLayers in [2, 3, 4]:
        for contrastThreshold in [0.02, 0.04, 0.06]:
            for edgeThreshold in [5, 10, 15]:
                for sigma in [1.2, 1.6, 2.0]:
                    # Create SIFT detector with the current parameter combination
                    sift = cv2.SIFT_create(
                        nfeatures=num_keypoints,
                        nOctaveLayers=nOctaveLayers,
                        contrastThreshold=contrastThreshold,
                        edgeThreshold=edgeThreshold,
                        sigma=sigma
                    )

                    # Detect and compute keypoints and descriptors with SIFT
                    kp1, des1 = sift.detectAndCompute(img1_blurred, None)
                    kp2, des2 = sift.detectAndCompute(img2_blurred, None)

                    # Create BFMatcher object
                    bf = cv2.BFMatcher()

                    # Match descriptors
                    matches = bf.knnMatch(des1, des2, k=2)
                    print("\n", matches)

                    # Apply Lowe's ratio test to get good matches
                    good_matches = []
                    print("\n", good_matches)
                    for m, n in matches:
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)

                    # Check if you have enough good matches to calculate the homography
                    if len(good_matches) >= 4:
                        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                        # Attempt to calculate the homography matrix
                        try:
                            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                            if M is not None:
                                M = M.astype(np.float32)
                                h, w = img1.shape
                                warped_img1 = cv2.warpPerspective(img1, M, (w + img2.shape[1], h + img2.shape[0]))
                                result = np.copy(warped_img1)
                                result[0:img2.shape[0], 0:img2.shape[1]] = img2
                                num_good_matches = len(good_matches)

                                if num_good_matches > best_num_keypoints:
                                    best_num_keypoints = num_good_matches
                                    best_nOctaveLayers = nOctaveLayers
                                    best_contrastThreshold = contrastThreshold
                                    best_edgeThreshold = edgeThreshold
                                    best_sigma = sigma
                                    best_stitched_image = result
                                    best_good_matches = good_matches
                        except cv2.error:
                            continue

# Save the stitched image using the best parameter combination
if best_stitched_image is not None:
    cv2.imwrite('Best_Stitched_Image.png', best_stitched_image)
