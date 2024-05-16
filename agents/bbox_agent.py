import os
import json
import ast
import sys
from tqdm import tqdm

import numpy as np
from PIL import Image
from utils.image_utils import tile_image, add_border, stitch_images
from utils.llava_request import prompt_llava_next, prompt_llava


def extract_answer(output):
    answer = output.split("<output>")[1].split("<output/>")[0].strip()
    return answer


def process_images(image_folder, output_folder, prompt, num_rows, num_cols):
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
        output = ast.literal_eval(prompt_llava(prompt, images=flattened_images))
        output = [result.lower() for result in output]
        print(output)

        for row in range(num_rows):
            for col in range(num_cols):
                # tqdm.write(f"Processing tile {row}, {col}")
                # # Send tile image to LLAVA model
                # output = prompt_llava_next(prompt, image=tiled_images[row][col])
                # output = ast.literal_eval(output)
                # answer = extract_answer(output[row * num_cols + col]).lower()
                answer = output[row * num_cols + col].lower()
                result = answer == "yes"
                if result:
                    tqdm.write(f"Tile {row}, {col} contains smoke")
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


# Constants
num_rows = 4
num_cols = 4
image_folder = "raw_data/20160619_FIRE_lp-e-iqeye"
output_folder = f"series_results/llava/{num_rows}x{num_cols}/20160619_FIRE_lp-e-iqeye"
prompt = """You are a proficient smoke detector at a fire tower. Does the following image contain wildfire smoke? Look carefully, and distinguish between clouds and smoke. Output "yes" only if there is smoke, and "no" only if there is no smoke. Be conservative, and only output "yes" if you are sure there is smoke. Reason out your logic, and enclose it in <Reasoning> <Reasoning/>. *****Do NOT go over 50 words*****. If you find yourself repeating yourself in your reasoning, stop your reasoning immediately. Then, output one line which is either "yes" or "no", enclosing it in <Output> <Output/>.

*****Example Image with Smoke Reasoning*****
The image shows a large plume of dark smoke rising from the mountains, which is indicative of a wildfire. The smoke is distinct from the surrounding clouds and vegetation, and its presence suggests a fire is occurring. The smoke is not a natural occurrence in this context, and its presence is a clear indication of a wildfire. The smoke is not a cloud, as it is not associated with precipitation or weather patterns.
*****End of Example Image with Smoke Reasoning*****

*****Example Image without Smoke Reasoning*****
The image shows a clear blue sky with no visible signs of smoke or haze. The sky is devoid of any particles or discoloration that would indicate the presence of wildfire smoke. The absence of any visible signs of smoke or haze leads to the conclusion that there is no wildfire smoke present in the image
*****End of Example Image without Smoke Reasoning*****

Remember to keep your reasoning concise, not more than 50 words, and end with a <Output> tag enclosing your final answer, for example <Output>yes<Output/> or <Output>no<Output/>."""
model_name = "llava-next"
test = """"You are a proficient smoke detector at a fire tower. Does the following image contain wildfire smoke? Look carefully, and distinguish between clouds and smoke. Reason out your logic. Then, output one line which is either "yes" or "no".
"""

# Process images
process_images(image_folder, output_folder, test, num_rows, num_cols)
