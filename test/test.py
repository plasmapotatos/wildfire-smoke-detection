import ast
import json
import os
import sys
import re

import numpy as np
from gradio_client import Client
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

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

image = Image.open("test/1468870572_-02400.jpg")
dist_above = 400  # Example distance above horizon
dist_below = 300  # Example distance below the horizon
tile_number = 5  # Example number of tiles
num_tiles = 4
tile_width = image.width // num_tiles  # Example tile width
num_rows = 4
num_cols = 4
horizon_y = image.height // 2  # Example horizon y coordinate
x = tile_number

# Generate a list of colors from a colormap
colors = [tuple(int(c * 255) for c in plt.cm.viridis(i)[:3]) for i in np.linspace(0, 1, x)]
print(colors)
extracted_tiles, tile_boxes = extract_tiles_from_horizon(
    image, horizon_y, dist_above, dist_below, tile_width, tile_number
)
for i, tile in enumerate(extracted_tiles):
    tile.save(f"test/tile_{i}.jpg")
# image = overlay_bbox(image, (0, 0, image.width, image.height), color = (255, 0, 0), thickness=10)
for i, tile_box in enumerate(tile_boxes):
    image = overlay_bbox(image, tile_box, color = (255, 255, 255))

image.save("test/stitched.jpg")