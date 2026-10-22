"""Offline vision checks: fake images/models/capture, no OpenCV or real devices."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import t
import vision_annotate as annotate
import vision_data as data
import vision_test as camera
import vision_test_image as still
import vision_train as cascade
import vision_train_eigen as eigen


class FakeImage:
    def __init__(self, width=128, height=128):
        self.shape = (height, width)
        self.size = width * height

    def __getitem__(self, item):
        y, x = item
        return FakeImage(x.stop - x.start, y.stop - y.start)

    def copy(self):
        return FakeImage(self.shape[1], self.shape[0])


def fake_cv():
    model = Mock()
    model.save.side_effect = lambda path: Path(path).write_bytes(b"fake-model")
    model.predict.return_value = (0, 100.0)
    detector = Mock()
    detector.empty.return_value = False
    detector.detectMultiScale.return_value = [(10, 20, 30, 40), (70, 50, 20, 20)]
    capture = Mock()
    capture.isOpened.return_value = True
    capture.read.return_value = (True, FakeImage())
    cv = SimpleNamespace(
        IMREAD_GRAYSCALE=0, COLOR_BGR2GRAY=6, FONT_HERSHEY_SIMPLEX=0,
        WINDOW_NORMAL=0, WND_PROP_VISIBLE=4, EVENT_LBUTTONDOWN=1, EVENT_LBUTTONUP=4,
        imread=Mock(return_value=FakeImage()), imwrite=Mock(),
        resize=Mock(side_effect=lambda image, size, **kw: FakeImage(*size) if size else image),
        cvtColor=Mock(side_effect=lambda image, _: image),
        face=SimpleNamespace(EigenFaceRecognizer_create=Mock(return_value=model)),
        CascadeClassifier=Mock(return_value=detector),
        VideoCapture=Mock(return_value=capture),
        rectangle=Mock(), putText=Mock(), imshow=Mock(), waitKey=Mock(return_value=ord("q")),
        destroyAllWindows=Mock(), getWindowProperty=Mock(return_value=1),
        namedWindow=Mock(), setMouseCallback=Mock(),
    )

    def write(path, image):
        Path(path).write_bytes(b"fake-crop")
        return True

    cv.imwrite.side_effect = write
    return cv, model, detector, capture


class VisionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "positive inputs"
        self.negatives = self.root / "negative inputs"
        for person in ("alpha", "zeta"):
            (self.dataset / person).mkdir(parents=True)
            (self.dataset / person / "photo.jpg").write_bytes(person.encode())
        self.negatives.mkdir()
        (self.negatives / "background.jpg").write_bytes(b"background")
        self.annotations = self.root / "annotations.txt"
        self.annotations.write_text("zeta/photo.jpg 1 10 20 30 40\nalpha/photo.jpg 1 5 6 20 30\n")
        self.cv, self.model, self.detector, self.capture = fake_cv()
        self.model_path = self.root / "model.yml"
        self.model_path.write_bytes(b"model")
        self.cascade_path = self.root / "cascade.xml"
        self.cascade_path.write_bytes(b"cascade")

    def test_legacy_windows_paths_spaces_and_multiple_rectangles(self):
        raw = r"F:\\Code\\HMIvoice\\image\\p\\alpha\\photo.jpg 2 1 2 3 4 5 6 7 8"
        self.annotations.write_text(raw + "\n")
        audit = data.read_annotations(self.annotations, self.dataset)
        self.assertEqual(audit.summary()["usable_rectangles"], 2)
        self.assertEqual(audit.rows[0].relative, "alpha/photo.jpg")
        raw, boxes = data.parse_annotation_line("a person/photo.jpg 1 0 0 5 6")
        self.assertEqual(raw, "a person/photo.jpg")
        self.assertEqual(boxes, ((0, 0, 5, 6),))

    def test_missing_invalid_duplicate_and_count_mismatch_are_reported(self):
        self.annotations.write_text(
            "alpha/photo.jpg 1 1 2 3 4\nalpha/photo.jpg 1 1 2 3 4\n"
            "missing/a.jpg 1 1 2 0 4\nzeta/photo.jpg 2 1 2 3 4\n"
        )
        audit = data.read_annotations(self.annotations, self.dataset)
        self.assertEqual(audit.summary(), {
            "total_rows": 4, "found_rows": 2, "missing_rows": 1,
            "invalid_rows": 2, "duplicate_rows": 1, "usable_rows": 1, "usable_rectangles": 1,
        })
        with self.assertRaisesRegex(ValueError, "allow-incomplete"):
            data.require_complete(audit, False)
        data.require_complete(audit, True)

    def test_multiple_rectangles_round_trip_when_coordinate_resembles_count(self):
        boxes = ((0, 0, 10, 1), (5, 5, 10, 10))
        encoded = "alpha/photo.jpg 2 " + " ".join(str(value) for box in boxes for value in box)
        self.assertEqual(data.parse_annotation_line(encoded), ("alpha/photo.jpg", boxes))
        self.annotations.write_text(encoded + "\n")
        self.assertEqual(data.read_annotations(self.annotations, self.dataset).rows[0].boxes, boxes)

    def test_path_with_repeated_spaces_round_trips_without_normalization(self):
        relative = "person  1/a  photo.JPG"
        image = self.dataset / relative
        image.parent.mkdir()
        image.write_bytes(b"photo")
        encoded = relative + " 1 0 0 10 10"
        self.assertEqual(data.parse_annotation_line(encoded)[0], relative)
        self.annotations.write_text(encoded + "\n")
        audit = data.read_annotations(self.annotations, self.dataset)
        self.assertEqual(audit.rows[0].relative, relative)
        self.assertEqual(audit.rows[0].image, image)

    def test_annotation_paths_cannot_escape_dataset(self):
        for raw in ("../outside.jpg", "/outside/photo.jpg", "C:/outside/photo.jpg"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                data.resolve_image_path(raw, self.dataset)
        outside = self.root / "outside.jpg"
        outside.write_bytes(b"outside")
        (self.dataset / "outside.jpg").symlink_to(outside)
        with self.assertRaises(ValueError):
            data.resolve_image_path("outside.jpg", self.dataset)

    def test_fresh_outputs_reject_existing_input_and_dangling_symlinks(self):
        existing = self.root / "existing"
        existing.mkdir()
        (existing / "keep").write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            data.fresh_output_directory(existing)
        with self.assertRaises(ValueError):
            data.fresh_output_directory(self.dataset / "new-output", (self.dataset,))
        destination = self.root / "not-created"
        link = self.root / "dangling"
        link.symlink_to(destination, target_is_directory=True)
        with self.assertRaises(FileExistsError):
            data.fresh_output_directory(link)
        with self.assertRaises(FileExistsError):
            annotate.annotate(self.dataset, link, cv2_module=self.cv)
        self.assertFalse(destination.exists())
        self.assertEqual((existing / "keep").read_bytes(), b"keep")

    def test_eigen_labels_are_deterministic_and_original_inputs_preserved(self):
        audit = data.read_annotations(self.annotations, self.dataset)
        samples, labels, issues = eigen.prepare_training_data(audit, self.cv)
        self.assertEqual(labels, {"alpha": 0, "zeta": 1})
        self.assertFalse(issues)
        self.assertTrue(all(sample[3].shape == (100, 100) for sample in samples))
        output = self.root / "eigen"
        numpy = SimpleNamespace(asarray=lambda values, dtype: values, int32="int32")
        result = eigen.train(self.dataset, self.annotations, output,
                             cv2_module=self.cv, numpy_module=numpy)
        self.assertEqual(result["labels"], {"0": "alpha", "1": "zeta"})
        self.assertEqual(camera.load_labels(output / "labels.json"), {0: "alpha", 1: "zeta"})
        self.assertEqual(self.model.train.call_args.args[1], [0, 1])
        self.assertEqual((self.dataset / "alpha/photo.jpg").read_bytes(), b"alpha")
        with self.assertRaises(FileExistsError):
            eigen.train(self.dataset, self.annotations, output,
                        cv2_module=self.cv, numpy_module=numpy)

    def test_out_of_bounds_and_decode_failures_do_not_create_training_output(self):
        self.annotations.write_text("alpha/photo.jpg 1 100 100 50 50\n")
        output = self.root / "bad-training"
        with self.assertRaisesRegex(ValueError, "outside image bounds"):
            eigen.train(self.dataset, self.annotations, output, cv2_module=self.cv)
        self.assertFalse(output.exists())
        self.cv.imread.return_value = None
        with self.assertRaisesRegex(ValueError, "could not be decoded"):
            eigen.train(self.dataset, self.annotations, output, cv2_module=self.cv)
        self.assertFalse(output.exists())

    def test_incomplete_training_requires_explicit_opt_in(self):
        self.annotations.write_text("missing/photo.jpg 1 0 0 10 10\nalpha/photo.jpg 1 0 0 10 10\n")
        output = self.root / "incomplete"
        with self.assertRaisesRegex(ValueError, "annotation issues"):
            eigen.train(self.dataset, self.annotations, output, cv2_module=self.cv)
        self.cv.face.EigenFaceRecognizer_create.assert_not_called()
        result = eigen.train(self.dataset, self.annotations, output, True, cv2_module=self.cv,
                             numpy_module=SimpleNamespace(asarray=lambda values, dtype: values, int32="int32"))
        self.assertEqual(result["annotation_audit"]["missing_rows"], 1)
        self.assertEqual(len(result["training_samples"]), 1)

    def test_cascade_preparation_copies_sources_without_resizing(self):
        output = self.root / "prepared run"
        manifest = cascade.prepare(self.dataset, self.annotations, self.negatives, output, cv2_module=self.cv)
        self.assertEqual(manifest["positive_samples"], 2)
        self.assertEqual(manifest["negative_images"], 1)
        self.cv.resize.assert_not_called()
        self.assertEqual((output / "positive/000000.jpg").read_bytes(), b"alpha")
        self.assertEqual((self.dataset / "alpha/photo.jpg").read_bytes(), b"alpha")
        self.assertEqual((output / "positives.txt").read_text().splitlines()[0],
                         "positive/000000.jpg 1 5 6 20 30")
        with self.assertRaises(FileExistsError):
            cascade.prepare(self.dataset, self.annotations, self.negatives, output, cv2_module=self.cv)

    def test_cascade_commands_use_lists_checked_exit_status_and_refuse_overwrite(self):
        output = self.root / "prepared run"
        cascade.prepare(self.dataset, self.annotations, self.negatives, output, cv2_module=self.cv)
        runner = Mock(side_effect=lambda command, **kwargs: (kwargs["cwd"] / "positives.vec").write_bytes(b"vec"))
        cascade.create_samples(output, executable="tool with spaces", runner=runner)
        command = runner.call_args.args[0]
        self.assertIsInstance(command, list)
        self.assertEqual(command[0], "tool with spaces")
        self.assertEqual(runner.call_args.kwargs, {"cwd": output, "check": True})
        with self.assertRaises(FileExistsError):
            cascade.create_samples(output, runner=runner)
        with self.assertRaises(ValueError):
            cascade.train_classifier(output, 340, runner=runner)
        training = Mock(side_effect=lambda command, **kwargs: (kwargs["cwd"] / "classifier/cascade.xml").write_bytes(b"trained"))
        cascade.train_classifier(output, 1, runner=training)
        self.assertEqual(training.call_args.args[0][training.call_args.args[0].index("-numPos") + 1], "1")
        with self.assertRaises(FileExistsError):
            cascade.train_classifier(output, 1, runner=training)

    def test_annotation_drag_and_generated_output_round_trip(self):
        self.assertEqual(annotate.rectangle_from_drag((50, 60), (10, 20), (128, 128)), (10, 20, 40, 40))
        self.assertIsNone(annotate.rectangle_from_drag((10, 10), (10, 20), (128, 128)))
        self.assertEqual(annotate.rectangle_from_drag((-4, -8), (12, 15), (10, 10)), (0, 0, 10, 10))

        def mouse(window, callback):
            callback(self.cv.EVENT_LBUTTONDOWN, 10, 20, None, None)
            callback(self.cv.EVENT_LBUTTONUP, 40, 60, None, None)

        self.cv.setMouseCallback.side_effect = mouse
        output = self.root / "new-annotations.txt"
        self.assertEqual(annotate.annotate(self.dataset, output, cv2_module=self.cv), 1)
        audit = data.read_annotations(output, self.dataset)
        self.assertEqual(audit.rows[0].boxes, ((10, 20, 30, 40),))
        self.assertEqual((self.dataset / "alpha/photo.jpg").read_bytes(), b"alpha")

    def test_camera_keeps_detection_boxes_separate_and_releases_resources(self):
        camera.run(self.model_path, self.cascade_path, cv2_module=self.cv)
        self.assertEqual(self.cv.rectangle.call_args_list[0].args[1:3], ((10, 20), (40, 60)))
        self.assertEqual(self.cv.rectangle.call_args_list[1].args[1:3], ((70, 50), (90, 70)))
        self.capture.release.assert_called_once()
        self.cv.destroyAllWindows.assert_called_once()

    def test_closed_camera_window_exits_without_another_frame(self):
        self.cv.waitKey.return_value = -1
        self.cv.getWindowProperty.return_value = 0
        camera.run(self.model_path, self.cascade_path, cv2_module=self.cv)
        self.capture.read.assert_called_once()
        self.capture.release.assert_called_once()

    def test_camera_errors_and_interrupts_release_resources(self):
        for error in (RuntimeError("prediction failed"), KeyboardInterrupt()):
            self.cv, self.model, self.detector, self.capture = fake_cv()
            self.model.predict.side_effect = error
            with self.subTest(error=type(error).__name__), self.assertRaises(type(error)):
                camera.run(self.model_path, self.cascade_path, cv2_module=self.cv)
            self.capture.release.assert_called_once()
            self.cv.destroyAllWindows.assert_called_once()

    def test_missing_model_and_invalid_configuration_do_not_open_camera(self):
        with self.assertRaises(FileNotFoundError):
            camera.run(self.root / "missing", self.cascade_path, cv2_module=self.cv)
        with self.assertRaises(ValueError):
            camera.run(self.model_path, self.cascade_path, threshold=float("nan"), cv2_module=self.cv)
        self.cv.VideoCapture.assert_not_called()

    def test_failed_camera_open_is_released(self):
        self.capture.isOpened.return_value = False
        with self.assertRaisesRegex(RuntimeError, "could not open camera"):
            camera.run(self.model_path, self.cascade_path, cv2_module=self.cv)
        self.capture.release.assert_called_once()
        self.cv.destroyAllWindows.assert_called_once()

    def test_unreadable_still_image_is_rejected_before_processing(self):
        self.cv.imread.return_value = None
        with self.assertRaisesRegex(ValueError, "could not decode"):
            still.run(self.model_path, self.cascade_path, cv2_module=self.cv)
        self.cv.cvtColor.assert_not_called()
        self.cv.imshow.assert_not_called()

    def test_still_display_error_closes_windows(self):
        self.cv.imshow.side_effect = RuntimeError("display failed")
        with self.assertRaisesRegex(RuntimeError, "display failed"):
            still.run(self.model_path, self.cascade_path, cv2_module=self.cv)
        self.cv.destroyAllWindows.assert_called_once()

    def test_audit_cli_exit_status_reports_incomplete_inputs(self):
        self.annotations.write_text("missing/photo.jpg 1 0 0 5 5\n")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = t.main(["--annotations", str(self.annotations), "--dataset", str(self.dataset)])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(output.getvalue())["missing_rows"], 1)


class ImportTests(unittest.TestCase):
    def test_import_and_help_are_inert_without_optional_dependencies(self):
        root = Path(__file__).resolve().parents[1]
        script = """
import builtins, importlib, sys
sys.path.insert(0, sys.argv[1])
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'cv2', 'numpy'}:
        raise AssertionError('Optional dependency imported: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
for name in ('vision_data', 'vision_annotate', 'vision_train', 'vision_train_eigen',
             'vision_test', 'vision_test_image', 't'):
    module = importlib.import_module(name)
    if hasattr(module, 'main'):
        try:
            module.main(['--help'])
        except SystemExit as error:
            assert error.code == 0
"""
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-B", "-c", script, str(root)],
                                    cwd=directory, text=True, capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
