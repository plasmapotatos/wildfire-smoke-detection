import json
import os
from ast import literal_eval

from tqdm import tqdm

model = "paligemma"


def evaluate_tiles(tiled_results, image_label):
    """
    Evaluate the classification of a tiled image.

    :param tiled_results: List of lists representing the 4x4 tiled results (each value is 'yes' or 'no')
    :param image_label: True if the image is labeled as having fire, False otherwise
    :return: One of ['true_positive', 'false_positive', 'true_negative', 'false_negative']
    """
    tiled_results = literal_eval(tiled_results)
    if model == "paligemma":
        if all(row == None for row in tiled_results):
            has_fire = False
        else:
            has_fire = True
    else:
        has_fire = any("yes" in row for row in tiled_results)

    if has_fire and image_label:
        return "true_positive"
    elif has_fire and not image_label:
        return "false_positive"
    elif not has_fire and not image_label:
        return "true_negative"
    elif not has_fire and image_label:
        return "false_negative"


def evaluate_frame(results_string, image_label):
    """
    Evaluates the results based on the provided results string and image label.

    Parameters:
    results_string (str): The results string to evaluate.
    image_label (str): The image label to evaluate.

    Returns:
    str: 'true_positive', 'false_positive', 'true_negative', or 'false_negative'.
    """
    if results_string and image_label:
        return "true_positive"  # True Positive: Both are not falsy
    elif results_string and not image_label:
        return "false_positive"  # False Positive: results_string is not falsy and image_label is falsy
    elif not results_string and not image_label:
        return "true_negative"  # True Negative: Both are falsy
    elif not results_string and image_label:
        return "false_negative"  # False Negative: results_string is falsy and image_label is not falsy


def evaluate_series(series_folder, mode="tiled"):
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

        raw_folder = os.path.join(series_path, "bounding_boxes")
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
            if file_name.endswith("_bounding_boxes.txt"):

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
                if mode == "tiled":
                    file_path = os.path.join(raw_folder, file_name)
                    with open(file_path, "r") as file:
                        # Read the tiled results
                        tiled_results = [line.strip().split() for line in file]
                    if mode == "horizon":
                        with open(file_path, "r") as file:
                            # Read the tiled results
                            tiled_results = file.read().strip()
                    # Evaluate the image
                    evaluation = evaluate_tiles(tiled_results, image_label)
                    series_stats[evaluation] += 1

                elif mode == "horizon":
                    file_path = os.path.join(raw_folder, file_name)
                    with open(file_path, "r") as file:
                        # Read the tiled results
                        tiled_results = file.read().strip()
                    # Evaluate the image
                    evaluation = evaluate_tiles(tiled_results, image_label)
                    series_stats[evaluation] += 1
                elif mode == "frame":
                    file_path = os.path.join(raw_folder, file_name)
                    with open(file_path, "r") as file:
                        # Read the frame results
                        results_string = file.read().strip()

                    # Evaluate the image
                    evaluation = evaluate_frame(results_string, image_label)
                    series_stats[evaluation] += 1

                # Update the least seconds after smoke for true positives

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
model_name = "paligemma"
mode = "horizon"
series_folder_path = f"series_results/{model_name}/{mode}"
evaluate_series(series_folder_path, mode=mode)
