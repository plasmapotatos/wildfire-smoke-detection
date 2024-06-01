import ast
import json
import os
import sys

import numpy as np
from gradio_client import Client
from PIL import Image
from tqdm import tqdm

from utils.image_utils import (
    add_border,
    extract_and_calculate_horizon,
    extract_and_parse_coordinates,
    extract_tiles_from_horizon,
    overlay_bbox,
    stitch_image_with_bboxes,
    stitch_images,
    tile_image,
    union_bounding_box,
)
from utils.prompts import PALIGEMMA_DETECT_PROMPT, PALIGEMMA_SEGMENT_PROMPT, LLAVA_PROMPT
from utils.request_utils import prompt_llava, prompt_llava_next, prompt_paligemma

# Constants
num_rows = 4
num_cols = 4
series_folder = "splits"
model_name = "llava"
mode = "horizon"
output_folder = f"series_results/{model_name}/{mode}"
horizon_y_sum = 0
num_images = 0

if model_name == "paligemma":
    client = Client("http://127.0.0.1:7860/")


def extract_answer(output):
    answer = output.split("<output>")[1].split("<output/>")[0].strip()
    return answer

def get_bounding_boxes_and_tiles(results, tile_boxes):
    """
    This function takes in an array of results and an array of tile boxes,
    and returns two arrays: one of bounding boxes scaled to (0, 0) and the
    other of the original tile boxes that correspond to a result of "yes".

    :param results: List of results, each being "yes" or "no"
    :param tile_boxes: List of tile boxes, each in the format [xmin, ymin, xmax, ymax, ...]
    :return: Tuple containing two lists:
             - List of scaled bounding boxes
             - List of original tile boxes with result "yes"
    """
    scaled_bounding_boxes = []
    original_tile_boxes = []

    for result, tile in zip(results, tile_boxes):
        if result == "yes":
            xmin, ymin, xmax, ymax = tile
            scaled_box = [0, 0, xmax - xmin, ymax - ymin]
            scaled_bounding_boxes.append(scaled_box)
            original_tile_boxes.append(tile)

    return scaled_bounding_boxes, original_tile_boxes

def run_on_image_paligemma(image, dist_above, dist_below, tile_width, tile_number):
    # Get horizon
    segment_response = prompt_paligemma(
        PALIGEMMA_SEGMENT_PROMPT, images=[image], client=client
    )[0]
    horizon_y_new = extract_and_calculate_horizon(
        segment_response, image.width, image.height
    )
    global horizon_y_sum
    global num_images
    # horizon_y_sum += horizon_y_new
    # num_images += 1
    # horizon_y = horizon_y_sum // num_images
    # if horizon_y < image.height // 2 - 100 or horizon_y > image.height // 2 + 100:
    #     print(
    #         f"Possibly incorrect horizon detection: {horizon_y} out of {image.height}"
    #     )
    # print("new: ", horizon_y_new)
    # print("sum: ", horizon_y_sum)
    # print("num: ", num_images)
    horizon_y = image.height // 2
    # Get tiled images
    extracted_tiles, tile_boxes = extract_tiles_from_horizon(
        image, horizon_y, dist_above, dist_below, tile_width, tile_number
    )
    for i, tile in enumerate(extracted_tiles):
        tile.save(f"test/tile_{i}.jpg")
    # Run detection on each tiled image
    detect_responses = prompt_paligemma(
        PALIGEMMA_DETECT_PROMPT, images=extracted_tiles, client=client
    )
    # Extract and parse coordinates
    bboxes = [
        extract_and_parse_coordinates(
            detect_response, extracted_tiles[0].width, extracted_tiles[0].height
        )
        for detect_response in detect_responses
    ]

    stitched_image = stitch_image_with_bboxes(image, bboxes, tile_boxes, union=True)

    return stitched_image, bboxes, tile_boxes

def run_on_image_llava(image, dist_above, dist_below, tile_width, tile_number):
    horizon_y = image.height // 2
    # Get tiled images
    extracted_tiles, tile_boxes = extract_tiles_from_horizon(
        image, horizon_y, dist_above, dist_below, tile_width, tile_number
    )
    for i, tile in enumerate(extracted_tiles):
        tile.save(f"test/tile_{i}.jpg")
    # Run detection on each tiled image
    detect_responses = ast.literal_eval(prompt_llava(LLAVA_PROMPT, images=extracted_tiles))

    parsed_responses = [response.lower() for response in detect_responses]

    print(parsed_responses)

    # Extract bounding boxes
    bounding_boxes, new_tile_boxes = get_bounding_boxes_and_tiles(parsed_responses, tile_boxes)

    print(bounding_boxes)
    stitched_image = stitch_image_with_bboxes(image, bounding_boxes, new_tile_boxes, union=True)
    return stitched_image, bounding_boxes, new_tile_boxes


def run_on_folder(
    image_folder, output_folder, dist_above, dist_below, tile_width, tile_number
):
    # Create output folders if they don't exist
    stitched_folder = os.path.join(output_folder, "stitched")
    bounding_box_folder = os.path.join(output_folder, "bounding_boxes")
    tile_box_folder = os.path.join(output_folder, "tile_boxes")
    os.makedirs(stitched_folder, exist_ok=True)
    os.makedirs(bounding_box_folder, exist_ok=True)
    os.makedirs(tile_box_folder, exist_ok=True)

    # Get list of image files
    image_files = os.listdir(image_folder)
    global horizon_y_sum
    global num_images
    horizon_y_sum = 0
    num_images = 0
    for image_file in tqdm(image_files):
        # check if image is an image
        if not image_file.endswith(".jpg") and not image_file.endswith(".jpeg"):
            continue

        # check if image is already processed

        if os.path.exists(
            os.path.join(stitched_folder, f"{os.path.splitext(image_file)[0]}.jpg")
        ):
            tqdm.write(f"Skipping image {image_file} as it is already processed.")
            continue
        # Load image
        image_path = os.path.join(image_folder, image_file)
        image_name = os.path.splitext(image_file)[0]
        image = Image.open(image_path)

        tqdm.write(f"Processing image: {image_file}")
        if model_name == "paligemma":
            stitched_image, bounding_boxes, tile_boxes = run_on_image_paligemma(
                image, dist_above, dist_below, tile_width, tile_number
            )
        if model_name == "llava":
            stitched_image, bounding_boxes, tile_boxes = run_on_image_llava(
                image, dist_above, dist_below, tile_width, tile_number
            )

        # Save stitched image
        stitched_image.save(os.path.join(stitched_folder, f"{image_name}.jpg"))

        # Save bounding boxes
        with open(
            os.path.join(bounding_box_folder, f"{image_name}_bounding_boxes.txt"), "w"
        ) as f:
            f.write(str(bounding_boxes))

        # Save tile boxes
        with open(
            os.path.join(tile_box_folder, f"{image_name}_tile_boxes.txt"), "w"
        ) as f:
            f.write(str(tile_boxes))


def run_on_series_folders(
    series_folder,
    output_folder,
    dist_above,
    dist_below,
    tile_width,
    tile_number,
    num_series=None,
    mode="tiled",
):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    series_folders = os.listdir(series_folder)
    series_folders = sorted(series_folders)

    # Iterate over each folder in the series_folders path
    for folder_name in tqdm(series_folders[:num_series]):
        folder_path = os.path.join(series_folder, folder_name)

        # Check if the path is a directory
        if os.path.isdir(folder_path):
            # Define the corresponding output path for this folder
            output_path = os.path.join(output_folder, folder_name)
            print(output_path)
            # Ensure the output path exists
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            # Run the run_on_folder function
            tqdm.write(f"Processing folder: {folder_name}")
            run_on_folder(
                folder_path,
                output_path,
                dist_above,
                dist_below,
                tile_width,
                tile_number,
            )


# # Process images
# run_on_series_folders(
#     series_folder, output_folder, prompt, num_rows, num_cols, mode="tiled"
# )

image = Image.open("test/false_negative.jpg")

# Specify parameters
dist_above = 400  # Example distance above horizon
dist_below = 400  # Example distance below the horizon
tile_number = 5  # Example number of tiles
tile_width = image.width // 4  # Example tile width

image, _, _ = run_on_image_llava(image, dist_above, dist_below, tile_width, tile_number)
image.save("test/stitched.jpg")

run_on_series_folders(
    series_folder,
    output_folder,
    dist_above,
    dist_below,
    tile_width,
    tile_number,
)
