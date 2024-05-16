import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from utils.image_utils import resize_images, load_images_from_directory, pil_to_cv2, parse_xml

def compare_images(image1, image2):
    # Convert images to grayscale
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute SSIM (Structural Similarity Index)
    ssim_index = ssim(gray_image1, gray_image2)
    #print("SSIM:", ssim_index)
    # Compute normalized correlation coefficient
    correlation = cv2.matchTemplate(gray_image1, gray_image2, cv2.TM_CCORR_NORMED)[0][0]
    #print("Correlation:", correlation)
    # Compute histogram intersection
    # hist1 = cv2.calcHist([gray_image1], [0], None, [256], [0, 256])
    # hist2 = cv2.calcHist([gray_image2], [0], None, [256], [0, 256])
    # hist_intersection = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
    # print("Histogram Intersection:", hist_intersection)

    # Use a weighted combination of the metrics
    similarity = 0.5 * ssim_index + 0.5 * correlation #+ 0.2 * hist_intersection
    return similarity

def match_images(randomized_images, target_images):
    num_randomized = len(randomized_images)
    num_target = len(target_images)

    # Compute similarity matrix
    similarity_matrix = np.zeros((num_randomized, num_target))
    for i in range(num_randomized):
        for j in range(num_target):
            similarity_matrix[i][j] = compare_images(randomized_images[i], target_images[j])
    # print("Similarity Matrix:")
    # for row in similarity_matrix:
    #     print(row)
    matched_images = []
    for i in range(num_randomized):
        max_similarity = 0
        max_index = 0
        for j in range(num_target):
            if similarity_matrix[i][j] > max_similarity:
                max_similarity = similarity_matrix[i][j]
                max_index = j
        matched_images.append(max_index)

    # # Solve the assignment problem to find the best matching
    # row_ind, col_ind = linear_sum_assignment(-similarity_matrix)

    # # Create a dictionary mapping indices in randomized_images to indices in target_images
    # matching_indices = {row: col for row, col in zip(row_ind, col_ind)}

    # # Sort randomized_images based on the matching
    # sorted_randomized_images = [randomized_images[i] for i in sorted(matching_indices.keys())]

    return matched_images

def partition_similar_images(image_list, threshold):
    partitions = []
    current_partition = [image_list[0]]

    for i in range(1, len(image_list)):
        similarity = compare_images(image_list[i-1][0], image_list[i][0])
        #print(f"Similarity for {i - 1} and {i}: {similarity}")
        if similarity >= threshold:
            current_partition.append(image_list[i])
        else:
            partitions.append(current_partition)
            current_partition = [image_list[i]]
    
    partitions.append(current_partition)  # Add the last partition
    
    return partitions

def get_bounding_box_area(bbox):
    xmin, ymin, xmax, ymax = bbox
    return (xmax - xmin) * (ymax - ymin)

def sort_images_by_bbox_area(images, image_names, xml_dir):
    images_with_bbox = []
    for i, image_name in enumerate(image_names):
        xml_file = os.path.join(xml_dir, image_name + '.xml')
        if os.path.exists(xml_file):
            path, bboxs = parse_xml(xml_file)
            bbox = bboxs[0]
            area = get_bounding_box_area(bbox)
            images_with_bbox.append((images[i], image_name, area))
    images_with_bbox.sort(key=lambda x: x[2])
    return [image for image, _, _ in images_with_bbox]

image_directory = "temp"
target_directory = "raw_data"
xml_directory = "xmls"

# Example usage
if __name__ == "__main__":
    images, image_names = load_images_from_directory(image_directory)
    images = [pil_to_cv2(image) for image in images]
    combined_images = []
    for i, image in enumerate(images):
        combined_images.append((image, image_names[i]))
    #print(image_names)
    similar_images = partition_similar_images(combined_images, 0.8)
    target_directories = sorted(os.listdir(target_directory))

    cnt = 0
    string = "a"
    for i, partition in enumerate(similar_images):
        #print(len(partition), partition[0])
        imgs = [img for img, img_name in partition]
        img_names = [img_name for img, img_name in partition]
        sorted_images = sort_images_by_bbox_area(imgs, img_names, xml_directory)
        print(len(sorted_images))
        for img in sorted_images:
            cv2.imwrite(f"sorted_images/{string}.jpg", img)
            string += 'a'
            cnt += 1
        # all_images = load_images_from_directory(os.path.join(target_directory, target_directories[i]))
        # target_images = []
        # for img in all_images:
        #     # Check if the filename contains a "+"
        #     if "+" in img.filename:
        #         target_images.append(img)
        # target_images = [pil_to_cv2(image) for image in target_images]
        # target_images = resize_images(target_images, 640, 480)
        # randomized_images = partition

        # print(compare_images(partition[0], target_images[0]))
        
        # sorted_randomized_images = match_images(partition, target_images)
        # print(sorted_randomized_images)
