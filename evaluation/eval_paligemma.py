import json
import os

from shapely.geometry import box

from utils.parse_nemo import calculate_union_bounding_box, get_bounding_boxes


# Function to read bounding boxes from a JSON file
def read_annotated_bounding_box(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
        objects = data["objects"]
        bounding_boxes = get_bounding_boxes(objects)
        union_box = calculate_union_bounding_box(bounding_boxes)
    return union_box


# Function to read bounding boxes from a result txt file
def read_result_bounding_box(txt_path):
    with open(txt_path, "r") as f:
        line = f.readline().strip()
        if line == "None" or line == "":
            return None
        # Removing parentheses and splitting by comma
        line = line.strip("()")
        xmin, ymin, xmax, ymax = map(float, line.split(","))
    return (xmin, ymin, xmax, ymax)


# Function to check if two bounding boxes overlap
def bounding_boxes_overlap(box1, box2):
    box1_poly = box(box1[0], box1[1], box1[2], box1[3])
    box2_poly = box(box2[0], box2[1], box2[2], box2[3])
    return box1_poly.intersects(box2_poly)


# Function to calculate evaluation metrics
def calculate_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    return accuracy, precision, recall, f1


# Directory paths
ann_dir = "nemo/val/ann"
results_dir = "series_results/nemo/llava/horizon/5x4/union_bounding_boxes"
evaluation_results = {}

# Counters for evaluation metrics
tp, fp, fn = 0, 0, 0

# Process each JSON file in the annotation directory
for json_file in sorted(os.listdir(ann_dir)):
    if json_file.endswith(".json"):
        json_path = os.path.join(ann_dir, json_file)
        txt_file = json_file.split(".")[0] + "_union_bounding_box.txt"
        txt_path = os.path.join(results_dir, txt_file)
        print("Processing:", json_file)
        # Get annotated bounding box
        annotated_box = read_annotated_bounding_box(json_path)
        # Get result bounding box
        result_box = read_result_bounding_box(txt_path)
        if result_box is None:
            fn += 1
            evaluation_results[json_file] = "False Negative"
        elif bounding_boxes_overlap(annotated_box, result_box):
            tp += 1
            evaluation_results[json_file] = "True Positive"
        else:
            fp += 1
            evaluation_results[json_file] = "False Positive"

# Calculate metrics
accuracy, precision, recall, f1 = calculate_metrics(tp, fp, fn)

# Save evaluation results
with open("evaluation_results.json", "w") as f:
    json.dump(evaluation_results, f, indent=4)

# Print summary
summary = {
    "True Positives": tp,
    "False Positives": fp,
    "False Negatives": fn,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
}
print(json.dumps(summary, indent=4))
