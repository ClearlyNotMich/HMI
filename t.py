"""Audit annotation paths/counts without decoding images or importing OpenCV."""
import argparse
import json
from vision_data import ANNOTATIONS, POSITIVE_DIR, read_annotations


def check_paths(file_path=ANNOTATIONS, dataset=POSITIVE_DIR):
    return read_annotations(file_path, dataset)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default=ANNOTATIONS)
    parser.add_argument("--dataset", default=POSITIVE_DIR)
    parser.add_argument("--details", action="store_true", help="include individual issues")
    args = parser.parse_args(argv)
    audit = check_paths(args.annotations, args.dataset)
    result = audit.summary()
    if args.details:
        result["issues"] = audit.issues
    print(json.dumps(result, indent=2))
    return 1 if audit.issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
