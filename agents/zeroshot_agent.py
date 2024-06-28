import ast
import json
import os
import sys

import numpy as np
from gradio_client import Client
from PIL import Image
from tqdm import tqdm

from utils.image_utils import add_border, stitch_images, tile_image
from utils.prompts import (
    GPT4_BASIC_PROMPT,
    LLAVA_PROMPT,
    PALIGEMMA_DETECT_PROMPT,
    PHI3_ASSISTANT,
    PHI3_PROMPT,
)
from utils.request_utils import (
    prompt_gpt4,
    prompt_llava,
    prompt_llava_next,
    prompt_paligemma,
    prompt_phi3,
)

# Constants
num_rows = 1
num_cols = 1
series_folder = "splits/test"
model_name = os.environ.get("MODEL_NAME", "gpt4")
output_folder = f"results/{model_name}/tiled/{num_rows}x{num_cols}"


if model_name == "paligemma" or model_name == "phi3":
    client = Client("http://127.0.0.1:7860/")


def extract_answer(output):
    answer = output.split("<output>")[1].split("<output/>")[0].strip()
    return answer


def run_on_folder(image_folder, output_folder, prompt, num_rows, num_cols):
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
            os.path.join(
                stitched_folder, f"{os.path.splitext(image_file)[0]}_stitched.jpg"
            )
        ):
            tqdm.write(f"Skipping image {image_file} as it is already processed.")
            continue
        # Load image
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path)

        # Split image into tiles
        tiled_images = tile_image(image, num_rows, num_cols)

        results = []
        tqdm.write(f"Processing image: {image_file}")

        flattened_images = [image for row in tiled_images for image in row]

        # Send all tiles to LLAVA model
        # output = ast.literal_eval(prompt_llava_next(prompt, images=flattened_images))
        if model_name == "phi3":
            output = prompt_phi3(
                prompt, PHI3_ASSISTANT, images=flattened_images, client=client
            )
        if model_name == "llava":
            output = ast.literal_eval(prompt_llava(prompt, images=flattened_images))
        if model_name == "llava-next":
            output = ast.literal_eval(
                prompt_llava_next(prompt, images=flattened_images)
            )
        if model_name == "paligemma":
            output = prompt_paligemma(prompt, images=flattened_images, client=client)
        if model_name == "gpt4":
            output = prompt_gpt4(prompt, images=flattened_images)

        output = [result.lower() for result in output]

        for row in range(num_rows):
            for col in range(num_cols):
                # tqdm.write(f"Processing tile {row}, {col}")
                # # Send tile image to LLAVA model
                # output = prompt_llava_next(prompt, image=tiled_images[row][col])
                # output = ast.literal_eval(output)
                # answer = extract_answer(output[row * num_cols + col]).lower()
                answer = output[row * num_cols + col].lower()
                if model_name == "paligemma":
                    result = answer != ""
                else:
                    result = answer == "yes"
                # if result:
                #     tqdm.write(f"Tile {row}, {col} contains smoke")
                results.append(result)

                # Border the tile if result is "yes"
                if result:
                    tiled_images[row][col] = add_border(
                        tiled_images[row][col], border_size=2, border_color=(255, 0, 0)
                    )
                else:
                    tiled_images[row][col] = add_border(
                        tiled_images[row][col],
                        border_size=2,
                        border_color=(255, 255, 255),
                    )

        # Stitch tiles back together
        stitched_image = stitch_images(tiled_images)
        # Save stitched image
        stitched_image.save(
            os.path.join(
                stitched_folder, f"{os.path.splitext(image_file)[0]}_stitched.jpg"
            )
        )

        # Save results
        with open(
            os.path.join(raw_folder, f"{os.path.splitext(image_file)[0]}_results.txt"),
            "w",
        ) as f:
            for result in results:
                f.write("yes\n" if result else "no\n")


def run_on_series_folders(
    series_folder, output_folder, prompt, num_rows, num_cols, num_series=None
):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    series_folders = os.listdir(series_folder)
    series_folders = sorted(series_folders)
    print(series_folders[:num_series])

    # Iterate over each folder in the series_folders path
    for folder_name in tqdm(series_folders[:num_series]):
        folder_path = os.path.join(series_folder, folder_name)

        # Check if the path is a directory
        if os.path.isdir(folder_path):
            # Define the corresponding output path for this folder
            output_path = os.path.join(output_folder, folder_name)

            # Ensure the output path exists
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            # Run the run_on_folder function
            tqdm.write(f"Processing folder: {folder_name}")
            run_on_folder(folder_path, output_path, prompt, num_rows, num_cols)


# Process images
if __name__ == "__main__":
    if model_name == "paligemma":
        run_on_series_folders(series_folder, output_folder, PALIGEMMA_DETECT_PROMPT, num_rows, num_cols)
    elif model_name == "phi3":
        run_on_series_folders(series_folder, output_folder, PHI3_PROMPT, num_rows, num_cols)
    elif model_name == "llava":
        run_on_series_folders(series_folder, output_folder, LLAVA_PROMPT, num_rows, num_cols)
    elif model_name == "gpt4":
        run_on_series_folders(series_folder, output_folder, GPT4_BASIC_PROMPT, num_rows, num_cols)