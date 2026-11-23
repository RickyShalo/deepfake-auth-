
from PIL import Image
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from transformers import pipeline


# Load pretrained models
print("Loading FaceNet...")
mtcnn = MTCNN(image_size=160, margin=12)
facenet = InceptionResnetV1(pretrained="vggface2").eval()

print("Loading Deepfake Detector...")
detector = pipeline(
    "image-classification",
    model="prithivMLmods/Deep-Fake-Detector-v2-Model",
    device=-1
)


def preprocess(path):
    image = Image.open(path).convert("RGB")

    boxes, _ = mtcnn.detect(image)

    if boxes is None or len(boxes) != 1:
        raise ValueError(f"Expected exactly one face in {path}")

    face = mtcnn(image)

    x1, y1, x2, y2 = boxes[0]
    cropped = image.crop((x1, y1, x2, y2))

    return face, cropped


# Preprocess both images
enroll, enroll_crop = preprocess("samples/enroll.png")
probe, probe_crop = preprocess("samples/probe.png")


# Model 1: Facial verification
with torch.no_grad():
    embedding1 = facenet(enroll.unsqueeze(0))
    embedding2 = facenet(probe.unsqueeze(0))

    similarity = F.cosine_similarity(
        embedding1, embedding2
    ).item()

print("\n=== FACE VERIFICATION ===")
print(f"Cosine similarity: {similarity:.4f}")


# Model 2: Deepfake detection
print("\n=== DEEPFAKE DETECTION ===")

for name, image in [
    ("Enrollment", enroll_crop),
    ("Probe", probe_crop)
]:
    print(f"\n{name}:")

    results = detector(
        image,
        function_to_apply="softmax"
    )

    for result in results:
        print(f"{result['label']}: {result['score']:.4f}")


print("\nBoth model inference tests completed!")
