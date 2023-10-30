'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

def extract_features_using_sift(image):
    esdf_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image = esdf_normalized.astype(np.uint8)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def visualize_sift_keypoints(image):
    keypoints, descriptors = extract_features_using_sift(image)
    esdf_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image = esdf_normalized.astype(np.uint8)
    out_image = cv2.drawKeypoints(image, keypoints, None)
    plt.imshow(out_image)
    plt.show()

def compute_esdf_map(preprocessed_grid):
    # Convert unknown cells (-1) to free space (0)
    preprocessed_grid = np.where(preprocessed_grid == -1, 100, preprocessed_grid)
    binary_grid = (preprocessed_grid == 100).astype(int)
    esdf = distance_transform_edt(1 - binary_grid)
    return esdf

# Example usage
if __name__ == "__main__":
    # Load an image (replace 'your_image.jpg' with your actual image path)
    image = cv2.imread('PLY_IMG_MAP.png', cv2.IMREAD_GRAYSCALE)

    # Visualize SIFT keypoints
    visualize_sift_keypoints(image)

    # Compute ESDF map
    esdf_map = compute_esdf_map(image)

    # Normalize ESDF map to the range [0, 1]
    esdf_normalized = cv2.normalize(esdf_map, None, 0, 1, cv2.NORM_MINMAX)

    # Scale the normalized ESDF map for visualization
    esdf_visual = (esdf_normalized * 255).astype(np.uint8)

    # Display the ESDF map using cv2.imshow
    cv2.imshow("ESDF Map", esdf_visual)
    cv2.waitKey(0)

    # You can further process or display the ESDF map as needed
    #plt.imshow(esdf_map, cmap='viridis')
    #plt.colorbar()
    #plt.show()
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

def extract_features_using_sift(image):
    esdf_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image = esdf_normalized.astype(np.uint8)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def visualize_sift_keypoints(image):
    keypoints, descriptors = extract_features_using_sift(image)
    esdf_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image = esdf_normalized.astype(np.uint8)
    out_image = cv2.drawKeypoints(image, keypoints, None)
    plt.imshow(out_image)
    plt.show()

def compute_esdf_map(preprocessed_grid):
    # Convert unknown cells (-1) to free space (0)
    preprocessed_grid = np.where(preprocessed_grid == -1,80, preprocessed_grid)
    binary_grid = (preprocessed_grid == 80).astype(int)
    esdf = distance_transform_edt(1 - binary_grid)
    return esdf

# Example usage
if __name__ == "__main__":
    # Load an image (replace 'your_image.jpg' with your actual image path)
    image = cv2.imread("grayscale_map_0.09.png", cv2.IMREAD_GRAYSCALE)

    # Visualize SIFT keypoints
    visualize_sift_keypoints(image)

    # Compute ESDF map
    esdf_map = compute_esdf_map(image)

    # Define your minimum and maximum values
    min_val = 0.0
    max_val = 523.7594867875903

    # Normalize ESDF map using your specific min and max values
    esdf_normalized = cv2.normalize(esdf_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Display the ESDF map using cv2.imshow
    cv2.imshow("ESDF Map", esdf_normalized)
    cv2.waitKey(0)

    # You can further process or display the ESDF map as needed
    #plt.imshow(esdf_map, cmap='viridis')
    #plt.colorbar()
    #plt.show()
