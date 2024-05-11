from PIL import Image, ImageDraw


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


def stitch_images(image_array, num_columns, num_rows):
    """
    Stitch together images arranged in a grid format.

    Arguments:
    image_array : list
        2D array of images to be stitched together.
    num_columns : int
        Number of columns in the grid.
    num_rows : int
        Number of rows in the grid.

    Returns:
    PIL.Image
        The stitched image.
    """
    expected_length = num_columns * num_rows
    if len(image_array) * len(image_array[0]) != expected_length:
        raise ValueError(
            "The length of the image array does not match the expected length."
        )

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


if __name__ == "__main__":
    # Load an image
    image = Image.open("./test_smoke.jpg")

    # Add a border to the image
    bordered_image = add_border(image, border_size=10, border_color=(255, 0, 0))

    # Display the image
    bordered_image.save("./bordered_image.jpg")
