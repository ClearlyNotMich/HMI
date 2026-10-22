"""Portable, standard-library-only handling of historical vision inputs."""
from dataclasses import dataclass, field
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
POSITIVE_DIR = ROOT / "image/p"
NEGATIVE_DIR = ROOT / "image/n"
ANNOTATIONS = ROOT / "positives.txt"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
FACE_SIZE = (100, 100)  # OpenCV uses (width, height).


@dataclass(frozen=True)
class Annotation:
    image: Path
    relative: str
    boxes: tuple
    line: int


@dataclass
class AnnotationAudit:
    rows: list = field(default_factory=list)
    total_rows: int = 0
    found_rows: int = 0
    missing_rows: int = 0
    invalid_rows: int = 0
    duplicate_rows: int = 0
    issues: list = field(default_factory=list)

    def summary(self):
        return {"total_rows": self.total_rows, "found_rows": self.found_rows,
                "missing_rows": self.missing_rows, "invalid_rows": self.invalid_rows,
                "duplicate_rows": self.duplicate_rows, "usable_rows": len(self.rows),
                "usable_rectangles": sum(len(row.boxes) for row in self.rows)}


def resolve_image_path(raw, dataset):
    """Remap legacy .../image/p paths; otherwise require a dataset-relative path."""
    dataset = Path(dataset).resolve()
    normalized = re.sub(r"/+", "/", raw.replace(chr(92), "/"))
    candidate = Path(normalized)
    if candidate.is_absolute():
        try:
            return candidate.resolve().relative_to(dataset).as_posix()
        except ValueError:
            pass
    parts = normalized.split("/")
    markers = [i for i in range(len(parts) - 1)
               if [part.casefold() for part in parts[i:i + 2]] == ["image", "p"]]
    if markers:
        parts = parts[markers[-1] + 2:]
    elif candidate.is_absolute() or re.match(r"^[A-Za-z]:", normalized):
        raise ValueError("absolute path is outside the selected dataset")
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("image path must remain inside the selected dataset")
    try:
        return dataset.joinpath(*parts).resolve().relative_to(dataset).as_posix()
    except ValueError as error:
        raise ValueError("image path escapes the selected dataset") from error


def parse_annotation_line(line):
    """Parse a JPEG/PNG path followed by a count and rectangle integers.

    The image suffix fixes the path boundary even when coordinates resemble a
    second count. Keep the original path substring, including repeated spaces.
    """
    match = re.fullmatch(r"(.+\.(?:jpe?g|png))\s+([+-]?\d+(?:\s+[+-]?\d+)*)\s*",
                         line.rstrip("\r\n"), re.IGNORECASE)
    if match is None:
        raise ValueError("expected JPEG/PNG path, positive rectangle count, and exactly 4 integers per rectangle")
    numbers = [int(value) for value in match.group(2).split()]
    count, values = numbers[0], numbers[1:]
    if count < 1 or len(values) != 4 * count:
        raise ValueError("expected path, positive rectangle count, and exactly 4 integers per rectangle")
    return match.group(1), tuple(tuple(values[offset:offset + 4])
                                 for offset in range(0, len(values), 4))


def read_annotations(path=ANNOTATIONS, dataset=POSITIVE_DIR):
    """Return existing unique valid rows and explicit issue counts.

    Pixel bounds are checked later when images are decoded. A row can be both
    missing and invalid, so issue counts can overlap.
    """
    dataset = Path(dataset).resolve(strict=True)
    if not dataset.is_dir():
        raise ValueError("dataset must be a directory")
    audit, seen = AnnotationAudit(), set()
    with Path(path).open(encoding="utf-8-sig") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            audit.total_rows += 1
            try:
                raw, boxes = parse_annotation_line(line)
                relative = resolve_image_path(raw, dataset)
            except ValueError as error:
                audit.invalid_rows += 1
                audit.issues.append(f"line {number}: {error}")
                continue
            image = dataset / relative
            exists = image.is_file()
            if exists:
                audit.found_rows += 1
            else:
                audit.missing_rows += 1
                audit.issues.append(f"line {number}: missing {relative}")
            duplicate = relative in seen
            seen.add(relative)
            if duplicate:
                audit.duplicate_rows += 1
                audit.issues.append(f"line {number}: duplicate image {relative}")
            invalid = any(x < 0 or y < 0 or w <= 0 or h <= 0 for x, y, w, h in boxes)
            if invalid:
                audit.invalid_rows += 1
                audit.issues.append(f"line {number}: invalid rectangle in {relative}")
            if exists and not duplicate and not invalid:
                audit.rows.append(Annotation(image, relative, boxes, number))
    audit.rows.sort(key=lambda row: row.relative)
    return audit


def image_files(directory):
    directory = Path(directory).resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("image input must be a directory")
    result = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if not path.resolve().is_relative_to(directory):
                raise ValueError(f"image symlink escapes input directory: {path.name}")
            result.append(path)
    return result


def box_in_bounds(box, shape):
    x, y, width, height = box
    return (x >= 0 and y >= 0 and width > 0 and height > 0
            and x + width <= shape[1] and y + height <= shape[0])


def require_complete(audit, allow_incomplete):
    if audit.issues and not allow_incomplete:
        raise ValueError(f"annotation issues: {audit.summary()}; audit with t.py, or explicitly use --allow-incomplete")


def fresh_output_directory(path, input_directories=()):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path = path.resolve()
    for directory in input_directories:
        if path.is_relative_to(Path(directory).resolve()):
            raise ValueError("output must be outside the input image directories")
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, ensure_ascii=False)
        output.write("\n")


def load_cv2():
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required for this explicit vision operation") from error
    return cv2
