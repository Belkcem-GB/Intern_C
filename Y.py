import cv2
import numpy as np

# Load the reference and distorted binary images
reference_image = cv2.imread('grayscale_map_0.09.png', 0)  # Load reference image in grayscale
distorted_image = cv2.imread('grayscale_scan_0.09.png', 0)  # Load distorted image in grayscale

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Find keypoints and descriptors in both images
keypoints_reference, descriptors_reference = sift.detectAndCompute(reference_image, None)
keypoints_distorted, descriptors_distorted = sift.detectAndCompute(distorted_image, None)

# Initialize the FLANN-based matcher
flann_matcher = cv2.FlannBasedMatcher_create()

# Match descriptors using KNN (k-nearest neighbors) with a ratio test
matches = flann_matcher.knnMatch(descriptors_reference, descriptors_distorted, k=2)

# Apply Lowe's ratio test to select good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

print(len(good_matches))

# Calculate the transformation matrix using RANSAC
if len(good_matches) >= 4:
    src_pts = np.float32([keypoints_reference[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([keypoints_distorted[m.trainIdx].pt for m in good_matches])
    transformation_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
else:
    print("Not enough good matches to calculate a transformation matrix.")
    transformation_matrix = None

# Apply the transformation to the distorted image
if transformation_matrix is not None:
    height, width = reference_image.shape
    warped_image = cv2.warpPerspective(distorted_image, transformation_matrix, (width, height))
else:
    warped_image = None

# Visualize the matched and aligned images
if warped_image is not None:
    cv2.imshow("Warped Image", warped_image)
    cv2.imshow("Reference Image", reference_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Unable to align the images due to insufficient matches.")

# You can now use image similarity metrics to quantify the matching quality.
