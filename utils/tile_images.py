import os

from PIL import Image
from utils.get_images import get_images


def tile_image(image, rows, columns):
    # Get the dimensions of the original image
    original_width, original_height = image.size

    # Calculate the width and height of each tile
    tile_width = original_width // columns
    tile_height = original_height // rows

    # Initialize a 2D list to store the tiled images
    tiled_images = [[None] * columns for _ in range(rows)]

    # Split the original image into tiles
    for i in range(rows):
        for j in range(columns):
            # Calculate the coordinates for cropping each tile
            left = j * tile_width
            upper = i * tile_height
            right = left + tile_width
            lower = upper + tile_height

            # Crop the tile from the original image
            tile = image.crop((left, upper, right, lower))

            # Store the tile in the 2D list
            tiled_images[i][j] = tile

    return tiled_images


images_path = "./images/tiled_raw_images"
if not os.path.exists(images_path):
    os.makedirs(images_path)

source_images_path = "../smoke-detection/raw_data/20160604_FIRE_rm-n-mobo-c"
images = get_images(source_images_path)


for image in images:
    tiled_images = tile_image(image, 4, 4)
    filename = image.filename.split("/")[-1].split(".")[0]
    tiled_images_path = f"{images_path}/{filename}"
    if not os.path.exists(tiled_images_path):
        os.makedirs(tiled_images_path)
    for i, row in enumerate(tiled_images):
        for j, tile in enumerate(row):
            tile.save(f"{tiled_images_path}/tiled_image_{i}_{j}.jpg")
