
import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image
from transformers import pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)


# Directories
DATA = Path("datasets/selected")
OUTPUT = Path("results")
OUTPUT.mkdir(exist_ok=True)

# 0 = Real, 1 = Fake
images = (
    [(p, 0) for p in sorted((DATA / "real").glob("*.jpg"))]
    + [(p, 1) for p in sorted((DATA / "fake").glob("*.jpg"))]
)

if not images or len({label for _, label in images}) != 2:
    raise ValueError("Both real and fake image folders must contain images.")

# Load pretrained deepfake detector
print("Loading Deepfake Detector...", flush=True)

detector = pipeline(
    "image-classification",
    model="prithivMLmods/Deep-Fake-Detector-v2-Model",
    device=-1,
)

actual = []
predicted = []
scores = []
times = []
records = []

# Evaluate every image
for i, (path, label) in enumerate(images, start=1):
    with Image.open(path) as image:
        image = image.convert("RGB")

        start = time.perf_counter()

        results = detector(
            image,
            function_to_apply="softmax"
        )

        elapsed = time.perf_counter() - start

    fake_score = next(
        result["score"]
        for result in results
        if result["label"].lower() == "deepfake"
    )

    prediction = int(fake_score >= 0.5)

    actual.append(label)
    predicted.append(prediction)
    scores.append(fake_score)
    times.append(elapsed)

    records.append([
        str(path),
        label,
        prediction,
        fake_score,
        elapsed
    ])

    if i % 50 == 0 or i == len(images):
        print(f"Processed {i}/{len(images)} images", flush=True)


# Confusion matrix
matrix = confusion_matrix(actual, predicted, labels=[0, 1])
tn, fp, fn, tp = matrix.ravel()

# Calculate metrics
metrics = {
    "Accuracy": accuracy_score(actual, predicted),
    "Precision": precision_score(actual, predicted, zero_division=0),
    "Recall": recall_score(actual, predicted, zero_division=0),
    "F1": f1_score(actual, predicted, zero_division=0),
    "Balanced Accuracy": balanced_accuracy_score(actual, predicted),
    "False Positive Rate": fp / (fp + tn),
    "False Negative Rate": fn / (fn + tp),
    "ROC-AUC": roc_auc_score(actual, scores),
    "Average Precision": average_precision_score(actual, scores),
    "Average Inference Time (s)": sum(times) / len(times),
}

# Print and save results
lines = [
    "=== DEEPFAKE DETECTION EVALUATION ===",
    f"Total Images: {len(images)}",
    f"True Negatives: {tn}",
    f"False Positives: {fp}",
    f"False Negatives: {fn}",
    f"True Positives: {tp}",
    "",
]

for name, value in metrics.items():
    lines.append(f"{name}: {value:.4f}")

report = "\n".join(lines)
print("\n" + report)

(OUTPUT / "metrics.txt").write_text(report)

# Save individual predictions
with open(OUTPUT / "predictions.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "image", "actual", "predicted",
        "deepfake_score", "inference_seconds"
    ])
    writer.writerows(records)

# Confusion matrix chart
ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=["Real", "Fake"]
).plot(values_format="d")

plt.title("Deepfake Detection Confusion Matrix")
plt.savefig(OUTPUT / "confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.close()

# ROC curve
RocCurveDisplay.from_predictions(actual, scores)
plt.title("Deepfake Detection ROC Curve")
plt.savefig(OUTPUT / "roc_curve.png", dpi=200, bbox_inches="tight")
plt.close()

# Precision-recall curve
PrecisionRecallDisplay.from_predictions(actual, scores)
plt.title("Deepfake Detection Precision-Recall Curve")
plt.savefig(
    OUTPUT / "precision_recall_curve.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()

print("\nEvaluation complete! Results saved to results/")
