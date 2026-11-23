
from pathlib import Path

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from transformers import pipeline


# Enrolled face templates are saved here (one .pt file per user)
ENROLL_DIR = Path("enrollments")
ENROLL_DIR.mkdir(exist_ok=True)

# Thresholds (starting values, we still need to tune these)
MATCH_THRESHOLD = 0.6  # cosine similarity needed to accept a face
FAKE_THRESHOLD = 0.5   # deepfake score at or above this gets rejected
FACE_CONFIDENCE = 0.95 # MTCNN detections below this are ignored

# Load both pretrained models once so every request can reuse them
print("Loading FaceNet...", flush=True)
mtcnn = MTCNN(image_size=160, margin=12)
facenet = InceptionResnetV1(pretrained="vggface2").eval()

print("Loading Deepfake Detector...", flush=True)
detector = pipeline(
    "image-classification",
    model="prithivMLmods/Deep-Fake-Detector-v2-Model",
    device=-1,
)


def preprocess(image):
    # Same steps as test_models.py:
    # face tensor for FaceNet, cropped face for the detector
    image = image.convert("RGB")

    boxes, probs = mtcnn.detect(image)

    # Ignore low-confidence detections (MTCNN sometimes finds
    # tiny "faces" in the background around 0.8 confidence)
    if boxes is not None:
        boxes = boxes[probs >= FACE_CONFIDENCE]

    if boxes is None or len(boxes) != 1:
        raise ValueError("expected exactly one face in the photo")

    # Crop the face from the box we kept
    face = mtcnn.extract(image, boxes, None)

    x1, y1, x2, y2 = boxes[0]
    cropped = image.crop((x1, y1, x2, y2))

    return face, cropped


def get_embedding(face):
    with torch.no_grad():
        embedding = facenet(face.unsqueeze(0))

    return embedding[0]


def get_fake_score(cropped):
    results = detector(cropped, function_to_apply="softmax")

    return next(
        result["score"]
        for result in results
        if result["label"].lower() == "deepfake"
    )


def clean_name(name):
    name = name.strip().lower()

    if not name.isalnum():
        raise ValueError("name can only use letters and numbers")

    return name
