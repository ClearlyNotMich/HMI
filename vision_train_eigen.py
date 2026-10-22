"""Explicit EigenFace training from annotated crops; original images are read-only."""
import argparse
import json
from pathlib import Path
from vision_data import (
    ANNOTATIONS, FACE_SIZE, POSITIVE_DIR, ROOT, box_in_bounds,
    fresh_output_directory, load_cv2, read_annotations, require_complete, write_json,
)


def prepare_training_data(audit, cv2):
    """Decode valid crops; assign stable labels to their first person folder."""
    samples, issues = [], []
    for row in audit.rows:
        parts = Path(row.relative).parts
        if len(parts) < 2:
            issues.append(f"{row.relative}: expected a person subdirectory")
            continue
        image = cv2.imread(str(row.image), cv2.IMREAD_GRAYSCALE)
        if image is None:
            issues.append(f"{row.relative}: image could not be decoded")
            continue
        for index, box in enumerate(row.boxes):
            if not box_in_bounds(box, image.shape):
                issues.append(f"{row.relative}: rectangle {index + 1} is outside image bounds")
                continue
            x, y, width, height = box
            crop = cv2.resize(image[y:y + height, x:x + width], FACE_SIZE)
            samples.append((parts[0], row.relative, index, crop))
    labels = {name: index for index, name in enumerate(sorted({s[0] for s in samples}))}
    return samples, labels, issues


def train(dataset=POSITIVE_DIR, annotations=ANNOTATIONS, output=ROOT / "outputs/vision/eigen",
          allow_incomplete=False, *, cv2_module=None, numpy_module=None):
    audit = read_annotations(annotations, dataset)
    require_complete(audit, allow_incomplete)
    cv2 = cv2_module if cv2_module is not None else load_cv2()
    if not hasattr(cv2, "face") or not hasattr(cv2.face, "EigenFaceRecognizer_create"):
        raise RuntimeError("EigenFace training requires OpenCV's contrib face module")
    samples, labels, image_issues = prepare_training_data(audit, cv2)
    if image_issues and not allow_incomplete:
        raise ValueError("image validation failed: " + "; ".join(image_issues))
    if not samples:
        raise ValueError("no valid annotated face crops are available")
    if numpy_module is None:
        import numpy as numpy_module
    output = fresh_output_directory(output, (dataset,))
    crops = output / "crops"
    crops.mkdir()
    for number, (_, _, _, image) in enumerate(samples):
        if not cv2.imwrite(str(crops / f"{number:06d}.png"), image):
            raise OSError("OpenCV could not write a generated face crop")
    model = cv2.face.EigenFaceRecognizer_create()
    model.train([sample[3] for sample in samples],
                numpy_module.asarray([labels[sample[0]] for sample in samples], dtype=numpy_module.int32))
    model.save(str(output / "eigenface_model.yml"))
    if not (output / "eigenface_model.yml").is_file():
        raise OSError("OpenCV did not write the trained model")
    metadata = {
        "schema_version": 1, "face_size": list(FACE_SIZE), "grayscale": True,
        "labels": {str(label): name for name, label in labels.items()},
        "annotation_audit": audit.summary(), "issues": audit.issues + image_issues,
        "training_samples": [{"image": relative, "rectangle": index + 1, "label": labels[person]}
                             for person, relative, index, _ in samples],
    }
    write_json(output / "labels.json", metadata)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=POSITIVE_DIR)
    parser.add_argument("--annotations", default=ANNOTATIONS)
    parser.add_argument("--output", default=ROOT / "outputs/vision/eigen",
                        help="new output directory; never overwritten")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="explicitly accept reported missing/invalid inputs")
    args = parser.parse_args(argv)
    metadata = train(args.dataset, args.annotations, args.output, args.allow_incomplete)
    print(json.dumps({"training_samples": len(metadata["training_samples"]),
                      "annotation_audit": metadata["annotation_audit"],
                      "image_or_annotation_issues": len(metadata["issues"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
