import os
import json
from tqdm import tqdm


def evaluate_tiles(tiled_results, image_label):
    """
    Evaluate the classification of a tiled image.

    :param tiled_results: List of lists representing the 4x4 tiled results (each value is 'yes' or 'no')
    :param image_label: True if the image is labeled as having fire, False otherwise
    :return: One of ['true_positive', 'false_positive', 'true_negative', 'false_negative']
    """
    has_fire = any("yes" in row for row in tiled_results)

    if has_fire and image_label:
        return "true_positive"
    elif has_fire and not image_label:
        return "false_positive"
    elif not has_fire and not image_label:
        return "true_negative"
    elif not has_fire and image_label:
        return "false_negative"


def evaluate_series(series_folder):
    """
    Evaluate a series of images in a folder.

    :param series_folder: Path to the series folder containing multiple directories of image series
    """
    stats = []

    for series_dir in tqdm(sorted(os.listdir(series_folder))):
        series_path = os.path.join(series_folder, series_dir)
        # print(series_path)
        if not os.path.isdir(series_path):
            continue

        raw_folder = os.path.join(series_path, "raw")
        if not os.path.exists(raw_folder):
            continue

        # Initialize statistics
        series_stats = {
            "folder_name": series_dir,
            "true_positive": 0,
            "false_positive": 0,
            "true_negative": 0,
            "false_negative": 0,
            "least_seconds_after_smoke_tp": None,
        }

        for file_name in os.listdir(raw_folder):
            if file_name.endswith("_results.txt"):
                file_path = os.path.join(raw_folder, file_name)
                with open(file_path, "r") as file:
                    # Read the tiled results
                    tiled_results = [line.strip().split() for line in file]

                # Extract image label and time from file name
                parts = file_name.split("_")
                label_part = parts[1]
                image_label = label_part.startswith("+")
                # print(image_label)
                # print(parts, parts[1][0:])
                try:
                    time_seconds = int(parts[1][0:])
                except:
                    continue

                # Evaluate the image
                evaluation = evaluate_tiles(tiled_results, image_label)
                series_stats[evaluation] += 1

                if evaluation == "true_positive":
                    if (
                        series_stats["least_seconds_after_smoke_tp"] is None
                        or time_seconds < series_stats["least_seconds_after_smoke_tp"]
                    ):
                        series_stats["least_seconds_after_smoke_tp"] = time_seconds

        # Append the statistics for the series
        stats.append(series_stats)

    # Save the statistics to a JSON file
    output_path = os.path.join(series_folder, "series_evaluation_stats.json")
    with open(output_path, "w") as json_file:
        json.dump(stats, json_file, indent=4)


# Example usage
series_folder_path = "series_results/llava/4x4"
evaluate_series(series_folder_path)
