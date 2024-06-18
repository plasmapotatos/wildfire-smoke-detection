import os
import re
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ImageDraw


# Function to extract, parse, and scale the coordinate string in the format <locXXXX> in a specific order
def extract_and_parse_coordinates(text, width, height):
    # Regex pattern to match <locXXXX> parts
    pattern = r"<loc\d{4}>"
    # Find all parts that match the pattern
    matches = re.findall(pattern, text)

    if len(matches) < 4:  # TODO: change how to handle this
        return None

    # Extract integer values from each <locXXXX> part
    values = [int(match[4:-1]) for match in matches]  # Remove '<loc' and '>'

    # Reorder the coordinates to (ymin, xmin, ymax, xmax)
    ymin, xmin = values[:2]
    ymax, xmax = values[-2:]

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


def overlay_bbox(image, bbox, color=(0, 255, 0), thickness=5):
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
        Thickness of the bounding box outline. Default is 5 pixels.

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


def get_union_bounding_box(bounding_boxes):
    """
    Calculate the union bounding box that covers all the input bounding boxes.

    Parameters:
    - bounding_boxes (list of tuples): A list of tuples specifying the bounding box coordinates (left, upper, right, lower).

    Returns:
    - union_bbox (tuple): A tuple specifying the union bounding box coordinates (left, upper, right, lower).
    """
    # Initialize the extreme values
    min_xmin = float("inf")
    min_ymin = float("inf")
    max_xmax = float("-inf")
    max_ymax = float("-inf")

    # Iterate over the bounding boxes and update the extreme values
    for bbox in bounding_boxes:
        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox

            # Update the extreme values for the union bounding box
            if xmin < min_xmin:
                min_xmin = xmin
            if ymin < min_ymin:
                min_ymin = ymin
            if xmax > max_xmax:
                max_xmax = xmax
            if ymax > max_ymax:
                max_ymax = ymax

    # Check if we found any valid bounding boxes
    if (
        min_xmin == float("inf")
        or min_ymin == float("inf")
        or max_xmax == float("-inf")
        or max_ymax == float("-inf")
    ):
        return None  # No valid bounding boxes

    # Return the smallest covering bounding box
    return (min_xmin, min_ymin, max_xmax, max_ymax)


def stitch_image_with_bboxes(image, bounding_boxes, tiled_boxes, union=False):
    """
    Overlays bounding boxes on the original image at their relative positions specified by tiled_boxes.

    Parameters:
    - image (PIL.Image.Image): The original image.
    - bounding_boxes (list of tuples): A list of tuples specifying the bounding box coordinates (left, upper, right, lower).
    - tiled_boxes (list of tuples): A list of tuples specifying the relative coordinates (x, y) for each bounding box.

    Returns:
    - stitched_image (PIL.Image.Image): The image with all bounding boxes overlaid at their relative positions.
    """
    # Create a copy of the image to draw on to avoid modifying the original image
    stitched_image = image.copy()

    adjusted_bboxs = []
    # Iterate over the bounding boxes and their corresponding tiled boxes
    for bbox, tbox in zip(bounding_boxes, tiled_boxes):
        # Calculate the position of the bounding box on the original image
        # x_offset and y_offset are the offsets from the tiled box's position
        x_offset = tbox[0]
        y_offset = tbox[1]
        if bbox is None:
            continue

        # Adjust the bounding box coordinates by adding the offsets
        adjusted_bbox = (
            bbox[0] + x_offset,
            bbox[1] + y_offset,
            bbox[2] + x_offset,
            bbox[3] + y_offset,
        )

        adjusted_bboxs.append(adjusted_bbox)
        # Overlay the bounding box on the stitched image using the overlay_bbox function
        if not union:
            stitched_image = overlay_bbox(stitched_image, adjusted_bbox)
    union_bbox = None

    if union:
        union_bbox = get_union_bounding_box(adjusted_bboxs)
        if union_bbox is not None:
            stitched_image = overlay_bbox(stitched_image, union_bbox)

    return union_bbox, stitched_image


def union_bounding_box(bounding_boxes, num_rows, num_cols, tile_width, tile_height):
    # Initialize the extreme values
    min_xmin = float("inf")
    min_ymin = float("inf")
    max_xmax = float("-inf")
    max_ymax = float("-inf")

    # Area of a single tile
    tile_area = tile_width * tile_height
    threshold_area = 0.95 * tile_area

    # Iterate over the bounding boxes and update the extreme values
    for i, bbox in enumerate(bounding_boxes):
        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox

            # Calculate the bounding box area
            bbox_area = (xmax - xmin) * (ymax - ymin)

            # Ignore bounding boxes that cover more than 95% of the tile area
            if bbox_area > threshold_area:
                continue

            # Calculate the tile's position in the larger image
            row = i // num_cols
            col = i % num_cols

            # Calculate the offset for the current tile
            x_offset = col * tile_width
            y_offset = row * tile_height

            # Translate the bounding box coordinates to the larger image
            translated_xmin = xmin + x_offset
            translated_ymin = ymin + y_offset
            translated_xmax = xmax + x_offset
            translated_ymax = ymax + y_offset

            # Update the extreme values for the union bounding box
            if translated_xmin < min_xmin:
                min_xmin = translated_xmin
            if translated_ymin < min_ymin:
                min_ymin = translated_ymin
            if translated_xmax > max_xmax:
                max_xmax = translated_xmax
            if translated_ymax > max_ymax:
                max_ymax = translated_ymax

    # Check if we found any valid bounding boxes
    if (
        min_xmin == float("inf")
        or min_ymin == float("inf")
        or max_xmax == float("-inf")
        or max_ymax == float("-inf")
    ):
        return None  # No valid bounding boxes

    # Return the smallest covering bounding box
    return (min_xmin, min_ymin, max_xmax, max_ymax)


def extract_and_calculate_horizon(input_str, image_width, image_height):
    # Convert the loc values from strings to integers
    xmin, ymin, xmax, ymax = extract_and_parse_coordinates(
        input_str, image_width, image_height
    )

    # Calculate the adjusted xmax
    horizon_y = ymax

    return horizon_y


def extract_tiles_from_horizon(
    image, horizon_y, dist_above, dist_below, tile_width, tile_number
):
    # Calculate the total height of the tiles region
    total_height = dist_above + dist_below


    # Calculate the starting y-coordinate for extraction
    start_y = max(0, horizon_y - dist_above)

    # Calculate the ending y-coordinate for extraction
    end_y = min(image.height, horizon_y + dist_below)

    # Initialize an array to store the tiles
    tiles = []
    tile_boxes = []

    if tile_number == 1:
        # Extract the tile from the image
        tile_box = (0, start_y, image.width, end_y)
        tile_boxes.append(tile_box)
        tile = image.crop(tile_box)
        tiles.append(tile)
        return tiles, tile_boxes

    # Iterate through the x-coordinates
    for x in range(
        0, image.width - tile_width + 1, (image.width - tile_width) // (tile_number - 1)
    ):
        # Extract the tile from the image
        tile_box = (x, start_y, x + tile_width, end_y)

        tile_boxes.append(tile_box)

        # Extract the tile from the image
        tile = image.crop(tile_box)
        # Append the tile to the tiles array
        tiles.append(tile)

    return tiles, tile_boxes


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
    image = Image.open("test/false_negatives/1.jpg")

    # Specify parameters
    horizon_y = image.height // 2  # Example horizon x value
    dist_above = 400  # Example distance above horizon
    dist_below = 400  # Example distance below the horizon
    tile_number = 7  # Example number of tiles
    tile_width = image.width // 4  # Example tile width

    # Extract tiles
    extracted_tiles, tile_boxes = extract_tiles_from_horizon(
        image, horizon_y, dist_above, dist_below, tile_width, tile_number
    )

    # Save or further process the extracted tiles as needed
    # For example, to save each tile as a separate image:
    for i, tile in enumerate(extracted_tiles):
        tile.save(f"test/tile_{i}.jpg")  # Save each tile with a unique name
