"""Prepare source copies and explicitly invoke legacy OpenCV cascade utilities.

Historical scripts noted OpenCV 3.4.3. Training utilities must be installed
separately. No images are resized in place or read during module import.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from vision_data import (
    ANNOTATIONS, NEGATIVE_DIR, POSITIVE_DIR, ROOT, box_in_bounds,
    fresh_output_directory, image_files, load_cv2, read_annotations,
    require_complete, write_json,
)


def prepare(dataset=POSITIVE_DIR, annotations=ANNOTATIONS, negatives=NEGATIVE_DIR,
            output=ROOT / "outputs/vision/cascade", allow_incomplete=False, *, cv2_module=None):
    audit = read_annotations(annotations, dataset)
    require_complete(audit, allow_incomplete)
    cv2 = cv2_module if cv2_module is not None else load_cv2()
    positives, negative_paths, issues = [], [], []
    for row in audit.rows:
        image = cv2.imread(str(row.image), cv2.IMREAD_GRAYSCALE)
        if image is None or any(not box_in_bounds(box, image.shape) for box in row.boxes):
            issues.append(f"{row.relative}: unreadable image or out-of-bounds rectangle")
        else:
            positives.append(row)
    for path in image_files(negatives):
        if cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) is None:
            issues.append(f"negative image {path.name}: could not be decoded")
        else:
            negative_paths.append(path)
    if issues and not allow_incomplete:
        raise ValueError("image validation failed: " + "; ".join(issues))
    if not positives or not negative_paths:
        raise ValueError("both positive and negative images are required")
    output = fresh_output_directory(output, (dataset, negatives))
    (output / "positive").mkdir()
    (output / "negative").mkdir()
    # Numeric aliases avoid whitespace in OpenCV's unquoted list-file paths.
    with (output / "positives.txt").open("x", encoding="utf-8") as description:
        for number, row in enumerate(positives):
            relative = f"positive/{number:06d}{row.image.suffix.lower()}"
            shutil.copyfile(row.image, output / relative)
            coordinates = " ".join(str(value) for box in row.boxes for value in box)
            description.write(f"{relative} {len(row.boxes)} {coordinates}\n")
    with (output / "negatives.txt").open("x", encoding="utf-8") as description:
        for number, path in enumerate(negative_paths):
            relative = f"negative/{number:06d}{path.suffix.lower()}"
            shutil.copyfile(path, output / relative)
            description.write(relative + "\n")
    manifest = {"schema_version": 1, "sample_width": 48, "sample_height": 48,
                "positive_samples": sum(len(row.boxes) for row in positives),
                "positive_images": len(positives), "negative_images": len(negative_paths),
                "annotation_audit": audit.summary(), "issues": audit.issues + issues,
                "positive_sources": [row.relative for row in positives]}
    write_json(output / "manifest.json", manifest)
    return manifest


def _manifest(run_directory):
    root = Path(run_directory).resolve(strict=True)
    with (root / "manifest.json").open(encoding="utf-8") as source:
        manifest = json.load(source)
    for field in ("positive_samples", "negative_images", "sample_width", "sample_height"):
        if type(manifest.get(field)) is not int or manifest[field] <= 0:
            raise ValueError(f"invalid prepared-run manifest: {field}")
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported prepared-run manifest")
    for name in ("positives.txt", "negatives.txt"):
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    return root, manifest


def create_samples(run_directory, executable="opencv_createsamples", *, runner=None):
    root, manifest = _manifest(run_directory)
    vector = root / "positives.vec"
    if vector.exists() or vector.is_symlink():
        raise FileExistsError(vector)
    command = [str(executable), "-info", "positives.txt", "-num", str(manifest["positive_samples"]),
               "-w", str(manifest["sample_width"]), "-h", str(manifest["sample_height"]), "-vec", "positives.vec"]
    (runner if runner is not None else subprocess.run)(command, cwd=root, check=True)
    if not vector.is_file():
        raise OSError("opencv_createsamples did not produce positives.vec")


def train_classifier(run_directory, num_positive, stages=15, threads=16,
                     executable="opencv_traincascade", *, runner=None):
    root, manifest = _manifest(run_directory)
    if type(num_positive) is not int or not 0 < num_positive <= manifest["positive_samples"]:
        raise ValueError("num-positive must be positive and no larger than the prepared sample count")
    if type(stages) is not int or stages <= 0 or type(threads) is not int or threads <= 0:
        raise ValueError("stages and threads must be positive integers")
    if not (root / "positives.vec").is_file():
        raise FileNotFoundError(root / "positives.vec")
    (root / "classifier").mkdir(exist_ok=False)
    command = [str(executable), "-data", "classifier", "-vec", "positives.vec", "-bg", "negatives.txt",
               "-numPos", str(num_positive), "-numNeg", str(manifest["negative_images"]),
               "-numStages", str(stages), "-w", str(manifest["sample_width"]),
               "-h", str(manifest["sample_height"]), "-numThreads", str(threads)]
    (runner if runner is not None else subprocess.run)(command, cwd=root, check=True)
    if not (root / "classifier/cascade.xml").is_file():
        raise OSError("opencv_traincascade did not produce classifier/cascade.xml")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    preparation = commands.add_parser("prepare", help="copy validated input images into a new run folder")
    preparation.add_argument("--dataset", default=POSITIVE_DIR)
    preparation.add_argument("--annotations", default=ANNOTATIONS)
    preparation.add_argument("--negatives", default=NEGATIVE_DIR)
    preparation.add_argument("--output", default=ROOT / "outputs/vision/cascade")
    preparation.add_argument("--allow-incomplete", action="store_true")
    samples = commands.add_parser("samples", help="explicitly run opencv_createsamples")
    samples.add_argument("--run-dir", required=True)
    samples.add_argument("--executable", default="opencv_createsamples")
    training = commands.add_parser("train", help="explicitly run opencv_traincascade")
    training.add_argument("--run-dir", required=True)
    training.add_argument("--num-positive", required=True, type=int, help="choose for available samples; historical 340 is not assumed")
    training.add_argument("--stages", type=int, default=15)
    training.add_argument("--threads", type=int, default=16)
    training.add_argument("--executable", default="opencv_traincascade")
    args = parser.parse_args(argv)
    if args.operation == "prepare":
        print(json.dumps(prepare(args.dataset, args.annotations, args.negatives, args.output, args.allow_incomplete), indent=2))
    elif args.operation == "samples":
        create_samples(args.run_dir, args.executable)
    else:
        train_classifier(args.run_dir, args.num_positive, args.stages, args.threads, args.executable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
