import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import cv2
import os
import numpy as np
import re


# Function to extract, parse, and scale the coordinate string in the format <locXXXX> in a specific order
def extract_and_parse_coordinates(text, width, height):
    # Regex pattern to match <locXXXX> parts
    pattern = r"<loc\d{4}>"
    # Find all parts that match the pattern
    matches = re.findall(pattern, text)

    if len(matches) != 4:
        return None

    # Extract integer values from each <locXXXX> part
    values = [int(match[4:-1]) for match in matches]  # Remove '<loc' and '>'

    # Reorder the coordinates to (ymin, xmin, ymax, xmax)
    ymin, xmin, ymax, xmax = values

    # Scaling factor (assuming original values are in the range 0-1024)
    scale_x = width / 1024
    scale_y = height / 1024

    # Scale the coordinates
    xmin = int(xmin * scale_x)
    ymin = int(ymin * scale_y)
    xmax = int(xmax * scale_x)
    ymax = int(ymax * scale_y)

    return xmin, ymin, xmax, ymax


def add_border(
    image,
    border_size=5,
    border_color=(255, 0, 0),
):
    """
    Add a border around the image without increasing its size.

    Arguments:
    image : PIL.Image
        The input image to which the border will be added.
    border_size : int
        The size of the border to be added.
    border_color : tuple, optional
        Color of the border in RGB format. Default is black (0, 0, 0).

    Returns:
    PIL.Image
        The image with the border added.
    """
    # Create a drawing object
    draw = ImageDraw.Draw(image)

    # Get image dimensions
    width, height = image.size

    # Draw top border
    draw.rectangle([0, 0, width, border_size], fill=border_color)
    # Draw bottom border
    draw.rectangle([0, height - border_size, width, height], fill=border_color)
    # Draw left border
    draw.rectangle([0, 0, border_size, height], fill=border_color)
    # Draw right border
    draw.rectangle([width - border_size, 0, width, height], fill=border_color)

    return image


def stitch_images(image_array):
    """
    Stitch together images arranged in a grid format.

    Arguments:
    image_array : list
        2D array of images to be stitched together.
    Returns:
    PIL.Image
        The stitched image.
    """
    num_rows = len(image_array)
    num_columns = len(image_array[0])

    image_width, image_height = image_array[0][0].size

    stitched_width = image_width * num_columns
    stitched_height = image_height * num_rows

    stitched_image = Image.new("RGB", (stitched_width, stitched_height))

    for i in range(num_rows):
        for j in range(num_columns):
            paste_x = j * image_width
            paste_y = i * image_height
            stitched_image.paste(image_array[i][j], (paste_x, paste_y))

    return stitched_image


def overlay_bbox(image, bbox, color=(0, 255, 0), thickness=2):
    """
    Overlay a bounding box onto an image.

    Arguments:
    image : PIL.Image
        The input image onto which the bounding box will be overlaid.
    bbox : tuple
        Bounding box coordinates in the format (xmin, ymin, xmax, ymax).
    color : tuple, optional
        Color of the bounding box outline in RGB format. Default is red (255, 0, 0).
    thickness : int, optional
        Thickness of the bounding box outline. Default is 2 pixels.

    Returns:
    PIL.Image
        The image with the bounding box overlaid.
    """
    # Create a drawing object
    draw = ImageDraw.Draw(image)

    # Extract bounding box coordinates
    xmin, ymin, xmax, ymax = bbox

    # Draw bounding box
    for i in range(thickness):
        draw.rectangle(
            [min(xmax - i, xmin + i), min(ymax - i, ymin + i), xmax - i, ymax - i],
            outline=color,
        )

    return image


def draw_horizontal_line(image, x, line_color=(255, 0, 0), line_thickness=1):
    """
    Draws a horizontal line across the image at the specified x-coordinate.

    Args:
    - image: PIL Image object.
    - x: The x-coordinate where the line should be drawn.
    - line_color: Tuple representing the RGB color of the line. Default is red.
    - line_thickness: Thickness of the line. Default is 1.

    Returns:
    - PIL Image object with the horizontal line drawn.
    """
    draw = ImageDraw.Draw(image)
    width, height = image.size
    draw.line((0, x, width, x), fill=line_color, width=line_thickness)
    return image


def resize_images(images, width, height):
    resized_images = []
    for image in images:
        resized_image = cv2.resize(image, (width, height))
        resized_images.append(resized_image)
    return resized_images


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


def load_images_from_directory(directory):
    image_list = []
    image_names = []
    # List all files in the directory
    files = os.listdir(directory)
    # Sort files alphabetically
    files.sort()
    # Iterate through files
    for filename in files:
        # Check if file is an image (you may want to add more image file extensions)
        if filename.endswith((".jpg", ".jpeg", ".png", ".gif")):
            # Open the image file
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path)
            # Append image to list
            image_list.append(image)
            image_names.append(filename.split(".")[0])
    return image_list, image_names


def pil_to_cv2(pil_image):
    """
    Converts a PIL (Python Imaging Library) image to OpenCV format.

    Args:
        pil_image (PIL.Image): The PIL image to be converted.

    Returns:
        numpy.ndarray: The image in OpenCV format (BGR color space).
    """
    # Convert PIL image to numpy array
    np_array = np.array(pil_image)

    # Convert RGB to BGR (OpenCV uses BGR color space)
    bgr_array = cv2.cvtColor(np_array, cv2.COLOR_RGB2BGR)

    return bgr_array


def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    filename = root.find("filename").text
    folder = root.find("folder").text
    path = root.find("path").text
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)
    boxes = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        xmin = int(float(box.find("xmin").text))
        ymin = int(float(box.find("ymin").text))
        xmax = int(float(box.find("xmax").text))
        ymax = int(float(box.find("ymax").text))
        boxes.append((xmin, ymin, xmax, ymax))
    return path, boxes


if __name__ == "__main__":
    # Load an image
    image = Image.open("raw_data/20160604_FIRE_rm-n-mobo-c/1465066440_+00840.jpg")

    tiled_images = tile_image(image, 4, 4)

    # save each tiled image in tile_row_col
    for row in range(4):
        for col in range(4):
            tiled_images[row][col].save(f"tile_{row}_{col}.jpg")
