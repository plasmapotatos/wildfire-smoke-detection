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
    extract_and_parse_coordinates,
    overlay_bbox,
    stitch_images,
    tile_image,
    union_bounding_box,
)
from utils.request_utils import prompt_llava, prompt_llava_next, prompt_paligemma

# Constants
num_rows = 4
num_cols = 4
series_folder = "splits"
output_folder = f"series_results/paligemma"
prompt = """detect en Detect the smoke in the image"""
model_name = "paligemma"

client = Client("http://127.0.0.1:7860/")


def extract_answer(output):
    answer = output.split("<output>")[1].split("<output/>")[0].strip()
    return answer


def run_on_folder(
    image_folder, output_folder, prompt, num_rows=None, num_cols=None, mode="tiled"
):
    # Create output folders if they don't exist
    stitched_folder = os.path.join(output_folder, "stitched")
    raw_folder = os.path.join(output_folder, "raw")
    os.makedirs(stitched_folder, exist_ok=True)
    os.makedirs(raw_folder, exist_ok=True)

    # Get list of image files
    image_files = os.listdir(image_folder)
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
        image = Image.open(image_path)

        if mode == "tiled":
            # Split image into tiles
            tiled_images = tile_image(image, num_rows, num_cols)
            tile_width = tiled_images[0][0].width
            tile_height = tiled_images[0][0].height

            results = []
            tqdm.write(f"Processing image: {image_file}")

            flattened_images = [image for row in tiled_images for image in row]

            # Send all tiles to LLAVA model
            # output = ast.literal_eval(prompt_llava_next(prompt, images=flattened_images))
            if model_name == "paligemma":
                output = prompt_paligemma(
                    prompt, images=flattened_images, client=client
                )
            coords = [
                extract_and_parse_coordinates(result, tile_width, tile_height)
                for result in output
            ]

            bbox = union_bounding_box(
                coords, num_rows, num_cols, tile_width, tile_height
            )

            if bbox:
                image = overlay_bbox(image, bbox)
            # Save stitched image
            image.save(
                os.path.join(stitched_folder, f"{os.path.splitext(image_file)[0]}.jpg")
            )

            # Save results
            with open(
                os.path.join(
                    raw_folder, f"{os.path.splitext(image_file)[0]}_results.txt"
                ),
                "w",
            ) as f:
                if coords:
                    f.write(str(bbox))
        if mode == "frame":
            if model_name == "paligemma":
                output = prompt_paligemma(
                    prompt, image_paths=[image_path], client=client
                )[0]
            width = image.width
            height = image.height
            coords = extract_and_parse_coordinates(output, width, height)
            if coords:
                image = overlay_bbox(image, coords)
            # Save stitched image
            image.save(
                os.path.join(stitched_folder, f"{os.path.splitext(image_file)[0]}.jpg")
            )

            # Save results
            with open(
                os.path.join(
                    raw_folder, f"{os.path.splitext(image_file)[0]}_results.txt"
                ),
                "w",
            ) as f:
                if bbox:
                    f.write(str(bbox))


def run_on_series_folders(
    series_folder,
    output_folder,
    prompt,
    num_rows,
    num_cols,
    num_series=None,
    mode="tiled",
):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    series_folders = os.listdir(series_folder)
    series_folders = sorted(series_folders)

    if mode == "tiled":
        # Create the tiled folder
        output_folder = os.path.join(output_folder, mode, f"{num_rows}x{num_cols}")
        os.makedirs(output_folder, exist_ok=True)
    if mode == "frame":
        output_folder = os.path.join(output_folder, mode)
        os.makedirs(output_folder, exist_ok=True)

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
            run_on_folder(folder_path, output_path, prompt, num_rows, num_cols)


# Process images
run_on_series_folders(
    series_folder, output_folder, prompt, num_rows, num_cols, mode="tiled"
)
