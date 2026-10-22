"""Explicit camera demonstration for an EigenFace model; no capture on import."""
import argparse
import json
import math
from pathlib import Path
from vision_data import FACE_SIZE, ROOT, load_cv2


def load_labels(path):
    if path is None:
        return {}
    with Path(path).open(encoding="utf-8") as source:
        metadata = json.load(source)
    if metadata.get("schema_version") != 1 or metadata.get("face_size") != list(FACE_SIZE) or metadata.get("grayscale") is not True:
        raise ValueError("labels file must describe the 100x100 grayscale training workflow")
    labels = metadata.get("labels")
    if not isinstance(labels, dict) or not labels:
        raise ValueError("labels file has no label mapping")
    if any(not key.isdigit() or not isinstance(name, str) or not name for key, name in labels.items()):
        raise ValueError("invalid label mapping")
    return {int(key): name for key, name in labels.items()}


def run(model_path, cascade_path=ROOT / "cascade140.xml", labels_path=None,
        camera=0, threshold=5400.0, *, cv2_module=None):
    model_path = Path(model_path).resolve(strict=True)
    cascade_path = Path(cascade_path).resolve(strict=True)
    labels = load_labels(labels_path)
    if type(camera) is not int or camera < 0:
        raise ValueError("camera must be a nonnegative device index")
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be a positive finite EigenFace distance")
    cv2 = cv2_module if cv2_module is not None else load_cv2()
    if not hasattr(cv2, "face") or not hasattr(cv2.face, "EigenFaceRecognizer_create"):
        raise RuntimeError("EigenFace prediction requires OpenCV's contrib face module")
    model = cv2.face.EigenFaceRecognizer_create()
    model.read(str(model_path))
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise ValueError(f"could not load cascade: {cascade_path}")
    capture = cv2.VideoCapture(camera)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"could not open camera index {camera}")
        while True:
            success, frame = capture.read()
            if not success:
                break
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for x, y, width, height in cascade.detectMultiScale(gray, 1.3, 5):
                face = cv2.resize(gray[y:y + height, x:x + width], FACE_SIZE)
                label, distance = model.predict(face)
                # This is an uncalibrated distance threshold, not a probability.
                text = labels.get(int(label), f"Label {label}") if distance < threshold else "Unknown"
                # Separate detections must not share a single smoothing history.
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if cv2.getWindowProperty("Face Recognition", cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        try:
            capture.release()
        finally:
            cv2.destroyAllWindows()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="EigenFace model; none is included in this repository")
    parser.add_argument("--cascade", default=ROOT / "cascade140.xml")
    parser.add_argument("--labels", help="labels.json created together with this model; omit for unidentified numeric labels")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=5400.0, help="historical uncalibrated distance cutoff")
    args = parser.parse_args(argv)
    run(args.model, args.cascade, args.labels, args.camera, args.threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
