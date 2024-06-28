import ast
import json
import os
import sys
import re

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
from utils.prompts import (
    LLAVA_PROMPT,
    PALIGEMMA_DETECT_PROMPT,
    PALIGEMMA_SEGMENT_PROMPT,
    PHI3_ASSISTANT,
    PHI3_PROMPT,
    GPT4_BASIC_PROMPT,
    GPT4_REASONING_PROMPT,
)
from utils.request_utils import (
    prompt_llava,
    prompt_llava_next,
    prompt_paligemma,
    prompt_phi3,
    prompt_gpt4,
)

# Constants
image = Image.open("test/nemo_test.jpg")

# Specify parameters
dist_above = 400  # Example distance above horizon
dist_below = 300  # Example distance below the horizon
tile_number = 5  # Example number of tiles
num_tiles = 4
tile_width = image.width // num_tiles  # Example tile width
num_rows = 4
num_cols = 4
series_folder = "actual_trial"
model_name = "gpt4"
mode = "horizon"
output_folder = f"budget_results/{model_name}/{mode}/{tile_number}x{num_tiles}"
horizon_y_sum = 0
num_images = 0

if model_name == "paligemma" or model_name == "phi3":
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
        if result[:3] == "yes":
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
    # horizon_y_new = extract_and_calculate_horizon(
    #     segment_response, image.width, image.height
    # )
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

    union_bbox, stitched_image = stitch_image_with_bboxes(
        image, bboxes, tile_boxes, union=True
    )

    return detect_responses, union_bbox, stitched_image, bboxes, tile_boxes


def run_on_image_phi3(image, dist_above, dist_below, tile_width, tile_number):
    # Get horizon
    # segment_response = prompt_paligemma(
    #     PALIGEMMA_SEGMENT_PROMPT, images=[image], client=client
    # )[0]
    # horizon_y_new = extract_and_calculate_horizon(
    #     segment_response, image.width, image.height
    # )
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
    detect_responses = prompt_phi3(
        PHI3_PROMPT, PHI3_ASSISTANT, images=extracted_tiles, client=client
    )
    print(detect_responses)
    # Extract and parse coordinates
    parsed_responses = [response.lower() for response in detect_responses]

    print(parsed_responses)

    # Extract bounding boxes
    bounding_boxes, new_tile_boxes = get_bounding_boxes_and_tiles(
        parsed_responses, tile_boxes
    )

    print(bounding_boxes)
    union_bbox, stitched_image = stitch_image_with_bboxes(
        image, bounding_boxes, new_tile_boxes, union=True
    )
    return parsed_responses, union_bbox, stitched_image, bounding_boxes, new_tile_boxes


def run_on_image_llava(image, dist_above, dist_below, tile_width, tile_number):
    horizon_y = image.height // 2
    # Get tiled images
    extracted_tiles, tile_boxes = extract_tiles_from_horizon(
        image, horizon_y, dist_above, dist_below, tile_width, tile_number
    )
    for i, tile in enumerate(extracted_tiles):
        tile.save(f"test/tile_{i}.jpg")
    # Run detection on each tiled image
    detect_responses = ast.literal_eval(
        prompt_llava(LLAVA_PROMPT, images=extracted_tiles)
    )

    parsed_responses = [response.lower() for response in detect_responses]

    print(parsed_responses)

    # Extract bounding boxes
    bounding_boxes, new_tile_boxes = get_bounding_boxes_and_tiles(
        parsed_responses, tile_boxes
    )

    print(bounding_boxes)
    union_bbox, stitched_image = stitch_image_with_bboxes(
        image, bounding_boxes, new_tile_boxes, union=True
    )
    return parsed_responses, union_bbox, stitched_image, bounding_boxes, new_tile_boxes

def parse_reasoning_output(output):
    # Regex pattern to match <output>...</output> or <output/> or <output />
    pattern = r'<output>(.*?)<output>'
    
    # Find all matches
    matches = re.findall(pattern, output.lower(), re.DOTALL)
    if len(matches) == 0 or all([match == "" for match in matches]):
        print("No matches found", output)
    print(output, matches)
    output = matches[0].lower() if matches else "no"
    return output

def run_on_image_gpt4(image, dist_above, dist_below, tile_width, tile_number, prompt_mode="reasoning"):
    horizon_y = image.height // 2
    # Get tiled images
    extracted_tiles, tile_boxes = extract_tiles_from_horizon(
        image, horizon_y, dist_above, dist_below, tile_width, tile_number
    )
    for i, tile in enumerate(extracted_tiles):
        tile.save(f"test/tile_{i}.jpg")
    # Run detection on each tiled image
    if prompt_mode == "reasoning":
        detect_responses = prompt_gpt4(GPT4_REASONING_PROMPT, images=extracted_tiles)
    elif prompt_mode == "basic":
        detect_responses = prompt_gpt4(GPT4_BASIC_PROMPT, images=extracted_tiles)

    if prompt_mode == "reasoning":
        parsed_responses = [parse_reasoning_output(response) for response in detect_responses]
    elif prompt_mode == "basic":
        parsed_responses = [response.lower() for response in detect_responses]

    print(parsed_responses)

    # Extract bounding boxes
    bounding_boxes, new_tile_boxes = get_bounding_boxes_and_tiles(
        parsed_responses, tile_boxes
    )

    union_bbox, stitched_image = stitch_image_with_bboxes(
        image, bounding_boxes, new_tile_boxes, union=True
    )
    return parsed_responses, union_bbox, stitched_image, bounding_boxes, new_tile_boxes, detect_responses

def middle_half(array):
    # Step 1: Sort the array
    sorted_array = sorted(array)
    
    # Step 2: Calculate the necessary indices
    n = len(sorted_array)
    half_length = (n + 1) // 2  # Equivalent to ceil(n / 2)
    quarter_length = (half_length + 1) // 2  # Equivalent to ceil(half_length / 2)
    
    start_index = quarter_length
    end_index = n - quarter_length
    
    # Step 3: Slice and return the middle half
    return sorted_array[start_index:end_index]

def run_on_folder(
    image_folder, output_folder, dist_above, dist_below, tile_width, tile_number
):
    # Create output folders if they don't exist
    stitched_folder = os.path.join(output_folder, "stitched")
    bounding_box_folder = os.path.join(output_folder, "bounding_boxes")
    tile_box_folder = os.path.join(output_folder, "tile_boxes")
    results_folder = os.path.join(output_folder, "results")
    union_bounding_box_folder = os.path.join(output_folder, "union_bounding_boxes")
    raw_output_folder = os.path.join(output_folder, "raw_output")
    os.makedirs(stitched_folder, exist_ok=True)
    os.makedirs(bounding_box_folder, exist_ok=True)
    os.makedirs(tile_box_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(union_bounding_box_folder, exist_ok=True)
    os.makedirs(raw_output_folder, exist_ok=True)
    # Get list of image files
    image_files = os.listdir(image_folder)
    global horizon_y_sum
    global num_images
    horizon_y_sum = 0
    num_images = 0
    halved_image_files = middle_half(sorted(image_files))
    print(halved_image_files)
    for image_file in tqdm(halved_image_files):
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
            (
                parsed_results,
                union_bounding_box,
                stitched_image,
                bounding_boxes,
                tile_boxes,
            ) = run_on_image_paligemma(
                image, dist_above, dist_below, tile_width, tile_number
            )
        if model_name == "llava":
            (
                parsed_results,
                union_bounding_box,
                stitched_image,
                bounding_boxes,
                tile_boxes,
            ) = run_on_image_llava(
                image, dist_above, dist_below, tile_width, tile_number
            )
        if model_name == "gpt4":
            (
                parsed_results,
                union_bounding_box,
                stitched_image,
                bounding_boxes,
                tile_boxes,
                raw_output,
            ) = run_on_image_gpt4(
                image, dist_above, dist_below, tile_width, tile_number
            )
        if model_name == "phi3":
            parsed_results, union_bounding_box, stitched_image, bounding_boxes, tile_boxes = (
                run_on_image_phi3(
                    image, dist_above, dist_below, tile_width, tile_number
                )
            )

        # Save stitched image
        stitched_image.save(os.path.join(stitched_folder, f"{image_name}.jpg"))

        # Save bounding boxes
        with open(
            os.path.join(bounding_box_folder, f"{image_name}_bounding_boxes.txt"), "w"
        ) as f:
            f.write(str(bounding_boxes))

        # Save union bounding box
        with open(
            os.path.join(
                union_bounding_box_folder, f"{image_name}_union_bounding_box.txt"
            ),
            "w",
        ) as f:
            f.write(str(union_bounding_box))

        # Save tile boxes
        with open(
            os.path.join(tile_box_folder, f"{image_name}_tile_boxes.txt"), "w"
        ) as f:
            f.write(str(tile_boxes))

        # Save results
        with open(os.path.join(results_folder, f"{image_name}_results.txt"), "w") as f:
            f.write(str(parsed_results))

        # Save raw output
        if model_name == "gpt4":
            with open(
                os.path.join(raw_output_folder, f"{image_name}_raw_output.txt"), "w"
            ) as f:
                f.write(str(raw_output))


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

# image = Image.open("trial/false_positive/20180728_FIRE_rm-w-mobo-c/1532815746_+00120.jpg")
# _, _, image, _, _ = run_on_image_gpt4(
#     image, dist_above, dist_below, tile_width, tile_number, prompt_mode="reasoning"
# )
# image.save("test/stitched.jpg")

run_on_series_folders(
    series_folder,
    output_folder,
    dist_above,
    dist_below,
    tile_width,
    tile_number,
    num_series=30,
    mode=mode,
)

# folder_path = "splits/validation"

# run_on_folder(
#     folder_path,
#     output_folder,
#     dist_above,
#     dist_below,
#     tile_width,
#     tile_number,
# )
