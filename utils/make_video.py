import os

import cv2


def create_animation(
    image_folder, output_file, frame_duration=100, resize_width=None, resize_height=None
):
    # Get list of image files in the folder
    image_files = [
        os.path.join(image_folder, img)
        for img in sorted(os.listdir(image_folder))
        if img.endswith(".jpg") or img.endswith(".jpeg")
    ]

    # Sort images alphabetically
    image_files.sort()

    # Read the first image to get dimensions
    frame = cv2.imread(image_files[0])
    height, width, _ = frame.shape

    # If both resize_width and resize_height are provided, resize the images
    if resize_width and resize_height:
        width, height = resize_width, resize_height

    # Define video writer
    video = cv2.VideoWriter(
        output_file,
        cv2.VideoWriter_fourcc(*"mp4v"),
        1 / (frame_duration / 1000),
        (width, height),
    )

    # Write frames
    for image_path in image_files:
        frame = cv2.imread(image_path)
        if resize_width and resize_height:
            frame = cv2.resize(frame, (resize_width, resize_height))
        video.write(frame)

    # Release resources
    cv2.destroyAllWindows()
    video.release()


# Example usage
image_series_path = "series_results/paligemma/tiled/4x4"
image_folder_path = "sorted_images"  # Replace with your image folder path
output_path = "videos/paligemma/tiled/4x4"
output_video = "test_sorted.mp4"
frame_duration_ms = 250  # Time interval for each frame in milliseconds
resize_width = 640  # New width of the resized images (optional)
resize_height = 480  # New height of the resized images (optional)

series_folders = sorted(os.listdir(image_series_path))
if not os.path.exists(output_path):
    os.makedirs(output_path)

for series_folder in series_folders:
    image_folder_path = os.path.join(image_series_path, series_folder, "stitched")
    output_video = os.path.join(output_path, f"{series_folder}.mp4")
    if os.path.exists(output_video):
        print(f"Skipping {output_video} as it already exists.")
        continue
    create_animation(
        image_folder_path, output_video, frame_duration_ms, resize_width, resize_height
    )
