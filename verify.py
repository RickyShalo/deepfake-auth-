
import sys

import torch
import torch.nn.functional as F
from PIL import Image

from face_utils import (
    ENROLL_DIR,
    FAKE_THRESHOLD,
    MATCH_THRESHOLD,
    clean_name,
    get_embedding,
    get_fake_score,
    preprocess,
)


def verify_user(name, image):
    # Login is accepted only if the photo is NOT flagged as a deepfake
    # AND the face matches the enrolled template
    name = clean_name(name)
    path = ENROLL_DIR / f"{name}.pt"

    if not path.exists():
        return False, [f"{name} is not enrolled."]

    try:
        face, cropped = preprocess(image)
    except ValueError as error:
        return False, [f"Rejected: {error}"]

    # Step 1: deepfake check
    fake_score = get_fake_score(cropped)

    if fake_score >= FAKE_THRESHOLD:
        return False, [
            "Rejected: photo looks like a deepfake.",
            f"Deepfake score: {fake_score:.4f} (limit {FAKE_THRESHOLD})",
        ]

    # Step 2: face verification
    template = torch.load(path)
    similarity = F.cosine_similarity(
        template, get_embedding(face), dim=0
    ).item()

    messages = [
        f"Deepfake score: {fake_score:.4f} (limit {FAKE_THRESHOLD})",
        f"Cosine similarity: {similarity:.4f} (need {MATCH_THRESHOLD})",
    ]

    if similarity >= MATCH_THRESHOLD:
        return True, [f"Accepted: welcome back, {name}!"] + messages

    return False, ["Rejected: face does not match."] + messages


if __name__ == "__main__":
    # Usage: python verify.py <name> photo.jpg
    if len(sys.argv) != 3:
        print("Usage: python verify.py <name> <photo>")
        sys.exit(1)

    success, messages = verify_user(sys.argv[1], Image.open(sys.argv[2]))

    print("\n".join(messages))
