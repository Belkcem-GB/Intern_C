import numpy as np
import cv2
import matplotlib.pyplot as plt

# Read the input images as grayscale
img1 = cv2.imread('grayscale_scan_0.2.png', 0)
img2 = cv2.imread('grayscale_map_0.2.png', 0)

# Apply Gaussian blur to reduce noise
img1_blurred = cv2.GaussianBlur(img1, (5, 5), 0)
img2_blurred = cv2.GaussianBlur(img2, (5, 5), 0)

# Initiate SRUF detector
# Initiate SIFT detector
sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.06, edgeThreshold=5, sigma=2)

# Detect and compute the keypoints and descriptors with SRUF
kp1, des1 = sift.detectAndCompute(img1_blurred, None)
kp2, des2 = sift.detectAndCompute(img2_blurred, None)

print(" kp1: \n", len(kp1))
print("kp2: \n", len(kp2))

# Create BFMatcher object
bf = cv2.BFMatcher()

# Match descriptors
matches = bf.match(des1, des2)

# Sort the matches based on distance
matches = sorted(matches, key=lambda val: val.distance)

# Draw first 50 matches
out = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)

cv2.imwrite('Keypoints_0.2_(5X5)SIFT_matched_GAUSSIAN_BLUR.png', out)
cv2.imshow("Keypoints_0.2_(5X5)_matched_GAUSSIAN_BLUR.png", out)

cv2.waitKey(0)
