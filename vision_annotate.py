"""Explicit GUI annotation into a new file; input images are never overwritten."""
import argparse
from pathlib import Path
from vision_data import POSITIVE_DIR, ROOT, box_in_bounds, image_files, load_cv2


def rectangle_from_drag(start, end, shape):
    """Normalize reverse drags, clamp to image bounds, and reject empty boxes."""
    x1, x2 = sorted((max(0, min(shape[1], start[0])), max(0, min(shape[1], end[0]))))
    y1, y2 = sorted((max(0, min(shape[0], start[1])), max(0, min(shape[0], end[1]))))
    box = (x1, y1, x2 - x1, y2 - y1)
    return box if box_in_bounds(box, shape) else None


def annotate(dataset=POSITIVE_DIR, output=ROOT / "outputs/vision/annotations/positives.txt", *, cv2_module=None):
    dataset = Path(dataset).resolve(strict=True)
    files = image_files(dataset)
    if not files:
        raise ValueError("no input images found")
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output = output.resolve()
    if output.is_relative_to(dataset):
        raise ValueError("annotation output must be outside the input image directory")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    cv2 = cv2_module if cv2_module is not None else load_cv2()
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    # Write only to a new file; partial annotations are retained if interrupted.
    with output.open("x", encoding="utf-8") as description:
        try:
            for path in files:
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f"could not decode input image: {path}")
                boxes = []
                state = {"start": None}

                def draw_rectangle(event, x, y, flags, parameter):
                    if event == cv2.EVENT_LBUTTONDOWN:
                        state["start"] = (x, y)
                    elif event == cv2.EVENT_LBUTTONUP and state["start"] is not None:
                        box = rectangle_from_drag(state["start"], (x, y), image.shape)
                        state["start"] = None
                        if box is not None:
                            boxes.append(box)

                cv2.namedWindow("image", cv2.WINDOW_NORMAL)
                cv2.setMouseCallback("image", draw_rectangle)
                quit_all = False
                while True:
                    preview = image.copy()
                    for x, y, width, height in boxes:
                        cv2.rectangle(preview, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    cv2.imshow("image", preview)
                    key = cv2.waitKey(20) & 0xFF
                    if key in (27, ord("q")):
                        quit_all = key == ord("q")
                        break
                    if cv2.getWindowProperty("image", cv2.WND_PROP_VISIBLE) < 1:
                        quit_all = True
                        break
                if boxes:
                    relative = path.relative_to(dataset).as_posix()
                    coordinates = " ".join(str(value) for box in boxes for value in box)
                    description.write(f"{relative} {len(boxes)} {coordinates}\n")
                    description.flush()
                    written += 1
                cv2.destroyAllWindows()
                if quit_all:
                    break
        finally:
            cv2.destroyAllWindows()
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__ + " Drag boxes; Escape advances, q finishes.")
    parser.add_argument("--dataset", default=POSITIVE_DIR)
    parser.add_argument("--output", default=ROOT / "outputs/vision/annotations/positives.txt")
    args = parser.parse_args(argv)
    print(f"Annotated images: {annotate(args.dataset, args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
