
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SYSTEM = platform.system()
VENV = ROOT / ".venv"

if SYSTEM == "Windows":
    PYTHON = VENV / "Scripts" / "python.exe"
    UV_NAME = "uv.exe"
else:
    PYTHON = VENV / "bin" / "python"
    UV_NAME = "uv"


def run(*args):
    subprocess.run(args, cwd=ROOT, check=True)


def get_uv():
    # Use an existing uv installation when available
    paths = [
        shutil.which("uv"),
        Path.home() / ".local" / "bin" / UV_NAME,
        Path.home() / ".cargo" / "bin" / UV_NAME,
    ]

    for path in paths:
        if path and Path(path).is_file():
            return str(path)

    print("Installing uv...")

    # Download the official installer for this operating system
    if SYSTEM == "Windows":
        url = "https://astral.sh/uv/install.ps1"
        suffix = ".ps1"
    else:
        url = "https://astral.sh/uv/install.sh"
        suffix = ".sh"

    with tempfile.TemporaryDirectory() as directory:
        installer = Path(directory) / f"install{suffix}"

        urllib.request.urlretrieve(url, installer)

        if SYSTEM == "Windows":
            run(
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy", "Bypass",
                "-File", str(installer),
            )
        else:
            run("sh", str(installer))

    # Find uv after installation
    paths = [
        shutil.which("uv"),
        Path.home() / ".local" / "bin" / UV_NAME,
        Path.home() / ".cargo" / "bin" / UV_NAME,
    ]

    for path in paths:
        if path and Path(path).is_file():
            return str(path)

    raise RuntimeError("Could not locate uv after installation.")


def main():
    if SYSTEM not in ("Windows", "Darwin", "Linux"):
        raise RuntimeError(f"Unsupported operating system: {SYSTEM}")

    print(f"Detected operating system: {SYSTEM}")

    # 1. Find or install uv
    uv = get_uv()

    # 2. Install compatible Python
    print("\nInstalling Python 3.11...")
    run(uv, "python", "install", "3.11")

    # 3. Create virtual environment
    print("\nCreating virtual environment...")
    run(uv, "venv", "--python", "3.11", str(VENV))

    # 4. Install dependencies
    print("\nInstalling dependencies...")
    run(
        uv, "pip", "install",
        "--python", str(PYTHON),
        "-r", str(ROOT / "requirements.txt"),
    )

    # 5. Download pretrained models
    print("\nDownloading pretrained models...")

    model_code = """
from facenet_pytorch import InceptionResnetV1
from transformers import pipeline

print("Loading FaceNet...", flush=True)
InceptionResnetV1(pretrained="vggface2").eval()

print("Loading Deepfake Detector...", flush=True)
pipeline(
    "image-classification",
    model="prithivMLmods/Deep-Fake-Detector-v2-Model",
    device=-1,
)

print("Both models ready!")
"""

    run(str(PYTHON), "-c", model_code)

    # 6. Prepare sample directory
    (ROOT / "samples").mkdir(exist_ok=True)

    print("\nSetup completed successfully!")
    print("Add enroll.png and probe.png to samples/")
    print(f"Run: {PYTHON} test_models.py")


if __name__ == "__main__":
    try:
        main()
    except (subprocess.CalledProcessError, OSError, RuntimeError) as error:
        print(f"\nSetup failed: {error}", file=sys.stderr)
        sys.exit(1)
