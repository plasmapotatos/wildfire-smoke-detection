import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

def compare_images(image1, image2):
    # Convert images to grayscale
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute SSIM (Structural Similarity Index)
    ssim_index = ssim(gray_image1, gray_image2)

    # Compute normalized correlation coefficient
    correlation = cv2.matchTemplate(gray_image1, gray_image2, cv2.TM_CCORR_NORMED)[0][0]

    # Compute histogram intersection
    hist1 = cv2.calcHist([gray_image1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray_image2], [0], None, [256], [0, 256])
    hist_intersection = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)

    # Use a weighted combination of the metrics
    similarity = 0.4 * ssim_index + 0.4 * correlation + 0.2 * hist_intersection
    return similarity

def match_images(randomized_images, target_images):
    num_randomized = len(randomized_images)
    num_target = len(target_images)

    # Compute similarity matrix
    similarity_matrix = np.zeros((num_randomized, num_target))
    for i in range(num_randomized):
        for j in range(num_target):
            similarity_matrix[i][j] = compare_images(randomized_images[i], target_images[j])

    # Solve the assignment problem to find the best matching
    row_ind, col_ind = linear_sum_assignment(-similarity_matrix)

    # Create a dictionary mapping indices in randomized_images to indices in target_images
    matching_indices = {row: col for row, col in zip(row_ind, col_ind)}

    # Sort randomized_images based on the matching
    sorted_randomized_images = [randomized_images[i] for i in sorted(matching_indices.keys())]

    return sorted_randomized_images

# Example usage
if __name__ == "__main__":
    # Assuming you have lists of randomized images and target images
    randomized_images = [...]  # List of randomized images
    target_images = [...]       # List of target images

    # Call match_images function to sort the randomized images
    sorted_randomized_images = match_images(randomized_images, target_images)

    # Now sorted_randomized_images contains the randomized images sorted to match the target images
