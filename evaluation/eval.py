import os
import numpy as np

from shutil import move
from utils.parse_xml import parse_xml
from PIL import Image
from utils.image_utils import add_border, stitch_images, overlay_bbox

# Define paths
results_dir = "results/raw"
xmls_dir = "xmls"
evaluation_dir = "./results/llava/processed"

# Define tile dimensions
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
TILE_WIDTH = IMAGE_WIDTH // 4
TILE_HEIGHT = IMAGE_HEIGHT // 4


def get_tile_box(tile_index, image_width, image_height, num_rows, num_columns):
    # Calculate the width and height of each tile
    tile_width = image_width // num_columns
    tile_height = image_height // num_rows

    # Calculate the row and column of the tile
    row = tile_index // num_columns
    col = tile_index % num_columns

    # Calculate the coordinates of the bounding box
    xmin = col * tile_width
    xmax = xmin + tile_width
    ymin = row * tile_height
    ymax = ymin + tile_height

    return (xmin, ymin, xmax, ymax)


def bbox_intersect(bbox1, bbox2):
    """
    Check if two bounding boxes intersect.

    Arguments:
    bbox1 : tuple
        Bounding box coordinates of the first box in the format (xmin, xmax, ymin, ymax).
    bbox2 : tuple
        Bounding box coordinates of the second box in the format (xmin, xmax, ymin, ymax).

    Returns:
    bool
        True if the bounding boxes intersect, False otherwise.
    """
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2

    # Check for intersection
    return not (xmax1 < xmin2 or xmin1 > xmax2 or ymax1 < ymin2 or ymin1 > ymax2)


def process_tiles(tiles, tiled_image_path, num_columns, num_rows, bbox=None):
    """
    Process positive tiles by adding a red border and stitching them together.

    Arguments:
    tiles : list
        List of positive tile indices.
    tiled_image_path : str
        Path to the tiled image directory.
    num_columns : int
        Number of columns in the tiled image grid.
    num_rows : int
        Number of rows in the tiled image grid.

    Returns:
    PIL.Image
        The stitched image with positive tiles highlighted.
    """
    # Load all tiled images
    image_array = [
        [
            Image.open(f"{tiled_image_path}/tiled_image_{i}_{j}.jpg")
            for j in range(num_columns)
        ]
        for i in range(num_rows)
    ]

    # Add a red border to positive tiles
    for tile_index in tiles:
        row = tile_index // num_columns
        col = tile_index % num_columns
        image_array[row][col] = add_border(image_array[row][col])

    # Stitch the images together
    stitched_image = stitch_images(
        image_array,
        num_columns,
        num_rows,
    )

    if bbox:
        stitched_image = overlay_bbox(stitched_image, bbox)

    return stitched_image


# Function to classify results and move files
def classify_results(results_dir, xmls_dir, evaluation_dir):
    num_true_positive = 0
    num_false_positive = 0
    num_false_negative = 0

    for results_file in os.listdir(results_dir):
        if results_file.endswith("_results.txt"):
            image_id = results_file.split("_")[0]
            xml_file = os.path.join(xmls_dir, image_id + ".xml")
            results_path = os.path.join(results_dir, results_file)
            if os.path.exists(xml_file):
                # Parse the XML file and read the results
                _, bounding_boxes = parse_xml(xml_file)
                bounding_box = bounding_boxes[0] if bounding_boxes else None
                with open(results_path, "r") as f:
                    results = f.read().splitlines()
                positive_tiles = [
                    idx for idx, result in enumerate(results) if result == "Yes"
                ]

                # Classify the results
                target_dir = ""
                if bounding_box and positive_tiles:
                    if any(
                        not bbox_intersect(
                            get_tile_box(tile, IMAGE_WIDTH, IMAGE_HEIGHT, 4, 4),
                            bounding_box,
                        )
                        for tile in positive_tiles
                    ):
                        target_dir = os.path.join(evaluation_dir, "false_positive")
                        num_false_positive += 1
                    else:
                        target_dir = os.path.join(evaluation_dir, "true_positive")
                        num_true_positive += 1
                elif bounding_box and not positive_tiles:
                    target_dir = os.path.join(evaluation_dir, "false_negative")
                    num_false_negative += 1

                # Move the results file to the target directory
                if target_dir:
                    tile_image_path = os.path.join("images", "tiled_images", image_id)
                    processed_image = process_tiles(
                        positive_tiles, tile_image_path, 4, 4, bounding_box
                    )
                    target_file = os.path.join(target_dir, image_id + ".jpg")
                    os.makedirs(target_dir, exist_ok=True)
                    processed_image.save(target_file)
                else:
                    print(f"Skipping {results_file} as no bounding box found.")

    print(f"True positives: {num_true_positive}")
    print(f"False positives: {num_false_positive}")
    print(f"False negatives: {num_false_negative}")


if __name__ == "__main__":
    # Call the function to classify results
    classify_results(results_dir, xmls_dir, evaluation_dir)
