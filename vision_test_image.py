"""Explicit still-image cascade demonstration; no image or window opens on import."""
import argparse
from pathlib import Path
from vision_data import ROOT, load_cv2


def run(image_path=ROOT / "peeps_bunch.jpg", cascade_path=ROOT / "classifier/cascade.xml", *, cv2_module=None):
    image_path = Path(image_path).resolve(strict=True)
    cascade_path = Path(cascade_path).resolve(strict=True)
    cv2 = cv2_module if cv2_module is not None else load_cv2()
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"could not decode image: {image_path}")
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise ValueError(f"could not load cascade: {cascade_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for x, y, width, height in faces:
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
    try:
        cv2.imshow("Faces", image)
        cv2.waitKey(0)
    finally:
        cv2.destroyAllWindows()
    return len(faces)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=ROOT / "peeps_bunch.jpg")
    parser.add_argument("--cascade", default=ROOT / "classifier/cascade.xml")
    args = parser.parse_args(argv)
    print(f"Detections: {run(args.image, args.cascade)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
