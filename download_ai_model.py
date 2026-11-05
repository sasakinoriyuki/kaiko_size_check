"""Download an immutable official checkpoint; no camera images are uploaded."""
import hashlib
from pathlib import Path
import urllib.request

from ai_measurement import MODEL_SHA256, MODEL_URL


def main():
    destination = Path(__file__).resolve().parent / "models/mobile_sam.pt"
    destination.parent.mkdir(exist_ok=True)
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == MODEL_SHA256:
        print("AI model is already installed.")
        return
    partial = destination.with_suffix(".partial")
    try:
        print("Downloading the official MobileSAM checkpoint (about 39 MB)...")
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response, partial.open("wb") as f:
            while block := response.read(1024*1024):
                f.write(block)
        if hashlib.sha256(partial.read_bytes()).hexdigest() != MODEL_SHA256:
            raise ValueError("The downloaded model checksum did not match.")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    print("AI model installed.")


if __name__ == "__main__":
    main()
