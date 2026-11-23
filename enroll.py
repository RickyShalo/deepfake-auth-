
import sys

import torch
import torch.nn.functional as F
from PIL import Image

from face_utils import (
    ENROLL_DIR,
    FAKE_THRESHOLD,
    clean_name,
    get_embedding,
    get_fake_score,
    preprocess,
)


MIN_PHOTOS = 3


def enroll_user(name, images):
    # Enrollment flow:
    # 1. find the face in each photo
    # 2. skip photos the detector thinks are deepfakes
    # 3. average the FaceNet embeddings into one face template
    # 4. save it to enrollments/<name>.pt
    name = clean_name(name)
    path = ENROLL_DIR / f"{name}.pt"

    # Do not let someone overwrite another user's face
    if path.exists():
        return False, [f"{name} is already enrolled."]

    embeddings = []
    messages = []

    for i, image in enumerate(images, start=1):
        try:
            face, cropped = preprocess(image)
        except ValueError as error:
            messages.append(f"Photo {i}: skipped ({error})")
            continue

        fake_score = get_fake_score(cropped)

        if fake_score >= FAKE_THRESHOLD:
            messages.append(
                f"Photo {i}: skipped (deepfake score {fake_score:.2f})"
            )
            continue

        embeddings.append(get_embedding(face))
        messages.append(
            f"Photo {i}: accepted (deepfake score {fake_score:.2f})"
        )

    if len(embeddings) < MIN_PHOTOS:
        messages.append(
            f"Need at least {MIN_PHOTOS} good photos but got "
            f"{len(embeddings)}. {name} was not enrolled."
        )
        return False, messages

    template = torch.stack(embeddings).mean(dim=0)
    template = F.normalize(template, dim=0)

    torch.save(template, path)

    messages.append(f"Enrolled {name} using {len(embeddings)} photos.")
    return True, messages


if __name__ == "__main__":
    # Usage: python enroll.py <name> photo1.jpg photo2.jpg photo3.jpg
    if len(sys.argv) < 3:
        print("Usage: python enroll.py <name> <photo1> <photo2> <photo3> ...")
        sys.exit(1)

    images = [Image.open(path) for path in sys.argv[2:]]
    success, messages = enroll_user(sys.argv[1], images)

    print("\n".join(messages))
