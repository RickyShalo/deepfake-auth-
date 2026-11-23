
import random
from pathlib import Path

import pyarrow.parquet as pq


SOURCE = Path("datasets/140k/data")
OUTPUT = Path("datasets/selected")
N = 500

rng = random.Random(42)
samples = {0: [], 1: []}
counts = {0: 0, 1: 0}

files = sorted(SOURCE.glob("test-*.parquet"))

if not files:
    raise FileNotFoundError(f"No test Parquet files found in {SOURCE}")

# Randomly select images from both classes
for file in files:
    print(f"Reading {file.name}...")

    parquet = pq.ParquetFile(file)

    for batch in parquet.iter_batches(
        batch_size=128,
        columns=["image", "label"]
    ):
        for row in batch.to_pylist():
            label = row["label"]
            if label not in samples:
                continue

            image = row["image"]["bytes"]
            counts[label] += 1

            # Reservoir sampling: every image has
            # an equal probability of being selected.
            if len(samples[label]) < N:
                samples[label].append(image)
            else:
                index = rng.randrange(counts[label])

                if index < N:
                    samples[label][index] = image

# Save the selected images
for label, name in [(0, "real"), (1, "fake")]:
    if len(samples[label]) < N:
        raise ValueError(f"Not enough {name} images")

    folder = OUTPUT / name
    folder.mkdir(parents=True, exist_ok=True)

    for i, image in enumerate(samples[label], start=1):
        path = folder / f"{name}_{i:04d}.jpg"
        path.write_bytes(image)

    print(f"Saved {N} {name} images to {folder}")

print("\nExtraction complete! 1,000 images saved.")
