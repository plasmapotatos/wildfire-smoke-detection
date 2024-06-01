import json


def load_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def process_data(data, num_worst_tp, num_worst_tn):
    total_true_positive = 0
    total_false_positive = 0
    total_true_negative = 0
    total_false_negative = 0
    total_least_seconds = 0
    count_least_seconds = 0

    min_true_positive = float("inf")
    min_true_negative = float("inf")
    min_tp_folder = None
    min_tn_folder = None

    for entry in data:
        total_true_positive += entry["true_positive"]
        total_false_positive += entry["false_positive"]
        total_true_negative += entry["true_negative"]
        total_false_negative += entry["false_negative"]

        if entry["true_positive"] < min_true_positive:
            min_true_positive = entry["true_positive"]
            min_tp_folder = entry["folder_name"]

        if entry["true_negative"] < min_true_negative:
            min_true_negative = entry["true_negative"]
            min_tn_folder = entry["folder_name"]

        if entry["least_seconds_after_smoke_tp"] is not None:
            total_least_seconds += entry["least_seconds_after_smoke_tp"]
            count_least_seconds += 1

    total_predictions = (
        total_true_positive
        + total_false_positive
        + total_true_negative
        + total_false_negative
    )

    true_positive_percent = (total_true_positive / total_predictions) * 100
    false_positive_percent = (total_false_positive / total_predictions) * 100
    true_negative_percent = (total_true_negative / total_predictions) * 100
    false_negative_percent = (total_false_negative / total_predictions) * 100
    average_least_seconds = (
        total_least_seconds / count_least_seconds if count_least_seconds > 0 else None
    )

    accuracy = (total_true_positive + total_true_negative) / total_predictions
    precision = (
        total_true_positive / (total_true_positive + total_false_positive)
        if (total_true_positive + total_false_positive) > 0
        else 0
    )
    recall = (
        total_true_positive / (total_true_positive + total_false_negative)
        if (total_true_positive + total_false_negative) > 0
        else 0
    )
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    worst_true_positives = sorted(data, key=lambda x: x["true_positive"])[:num_worst_tp]
    worst_true_negatives = sorted(data, key=lambda x: x["true_negative"])[:num_worst_tn]

    return {
        "true_positive_percent": true_positive_percent,
        "false_positive_percent": false_positive_percent,
        "true_negative_percent": true_negative_percent,
        "false_negative_percent": false_negative_percent,
        "average_least_seconds": average_least_seconds,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "min_true_positive": min_true_positive,
        "min_tp_folder": min_tp_folder,
        "min_true_negative": min_true_negative,
        "min_tn_folder": min_tn_folder,
        "worst_true_positives": worst_true_positives,
        "worst_true_negatives": worst_true_negatives,
    }


def save_results_to_json(results, output_file):
    with open(output_file, "w") as file:
        json.dump(results, file, indent=4)


def main():
    file_path = "series_results/llava/horizon/series_evaluation_stats.json"
    output_file = "processed_results.json"

    num_worst_tp = 5  # Number of worst-performing true positives
    num_worst_tn = 5  # Number of worst-performing true negatives

    data = load_json(file_path)
    results = process_data(data, num_worst_tp, num_worst_tn)

    save_results_to_json(results, output_file)

    print("Combined Data Results:")
    print(f"True Positive Percent: {results['true_positive_percent']:.2f}%")
    print(f"False Positive Percent: {results['false_positive_percent']:.2f}%")
    print(f"True Negative Percent: {results['true_negative_percent']:.2f}%")
    print(f"False Negative Percent: {results['false_negative_percent']:.2f}%")
    print(
        f"Average Least Seconds After Smoke TP: {results['average_least_seconds']:.2f}"
        if results["average_least_seconds"] is not None
        else "Average Least Seconds After Smoke TP: None"
    )
    print(f"Accuracy: {results['accuracy']:.2f}")
    print(f"Precision: {results['precision']:.2f}")
    print(f"Recall: {results['recall']:.2f}")
    print(f"F1 Score: {results['f1_score']:.2f}")
    print(
        f"Minimum True Positive: {results['min_true_positive']} in folder {results['min_tp_folder']}"
    )
    print(
        f"Minimum True Negative: {results['min_true_negative']} in folder {results['min_tn_folder']}"
    )
    print(
        f"Worst-performing True Positives (Top {num_worst_tp}): {[entry['folder_name'] for entry in results['worst_true_positives']]}"
    )
    print(
        f"Worst-performing True Negatives (Top {num_worst_tn}): {[entry['folder_name'] for entry in results['worst_true_negatives']]}"
    )


if __name__ == "__main__":
    main()
