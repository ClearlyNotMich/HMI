"""Offline voice checks with synthetic text, paths, headers, and model doubles."""

import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import voice


class MetricTests(unittest.TestCase):
    def test_wer_counts_insertions_deletions_substitutions_and_repeats(self):
        for reference, hypothesis, expected in (
            ("a b", "a b", 0.0),
            ("a b", "a", 0.5),
            ("a b", "a x b", 0.5),
            ("a b", "a c", 0.5),
            ("a a b", "a b", 1 / 3),
            ("a", "x y z", 3.0),
            ("a b", "", 1.0),
        ):
            with self.subTest(reference=reference, hypothesis=hypothesis):
                self.assertAlmostEqual(
                    voice.calculate_wer(reference.split(), hypothesis.split()),
                    expected,
                )
                self.assertAlmostEqual(
                    voice.calculate_wer2(reference.split(), hypothesis.split()),
                    expected,
                )

    def test_empty_reference_behavior_is_explicit(self):
        self.assertEqual(voice.calculate_wer([], []), 0.0)
        with self.assertRaisesRegex(ValueError, "empty reference"):
            voice.calculate_wer([], ["word"])

    def test_normalization_preserves_original_punctuation_policy(self):
        self.assertEqual(voice.preprocess("  Our,  THE!\ncan't  "), "our the cant")

    def test_output_uses_original_tokens_not_padded_heuristic_alignment(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = voice.output(" A, b! ", "x a b y")
        self.assertEqual(result["wer"], 1.0)
        self.assertEqual(
            result["character_distance"],
            voice.levenshtein_distance("a b", "x a b y"),
        )
        self.assertIn("Heuristic alignment", output.getvalue())
        self.assertNotIn("Macro", output.getvalue())
        self.assertNotIn("AWER", output.getvalue())

    def test_aligned_diagnostics_exclude_blanks_and_allow_unequal_lengths(self):
        self.assertEqual(
            voice.calculate_recall_precision_fscore(["a", ""], ["a", ""]),
            (1.0, 1.0, 1.0),
        )
        self.assertEqual(
            voice.calculate_recall_precision_fscore(["a", "b"], ["a"]),
            (1.0, 0.5, 2 / 3),
        )
        self.assertEqual(
            voice.calculate_recall_precision_fscore([""], ["", ""]),
            (0.0, 0.0, 0.0),
        )
        with contextlib.redirect_stdout(io.StringIO()):
            voice.calculate_for_word(["a"], ["a", "b"], ["a"])


class InputTests(unittest.TestCase):
    def test_workbook_rows_accept_numeric_ids_extra_columns_and_empty_rows(self):
        rows = [
            ("ID", "Sentence", "extra"),
            (1.0, "First sentence", None),
            (None, None, None),
            ("2", "Second sentence", "ignored"),
        ]
        self.assertEqual(
            voice._transcriptions_from_rows(rows),
            {1: "First sentence", 2: "Second sentence"},
        )

    def test_invalid_workbook_rows_are_not_silently_dropped(self):
        for rows in (
            [],
            [("ID", "Other")],
            [("ID", "ID", "Sentence")],
            [("ID", "Sentence")],
            [("ID", "Sentence"), (1, "hello"), (1, "duplicate")],
            [("ID", "Sentence"), (None, "hello")],
            [("ID", "Sentence"), (1.5, "hello")],
            [("ID", "Sentence"), (True, "hello")],
            [("ID", "Sentence"), (0, "hello")],
            [("ID", "Sentence"), (1, "!? ")],
            [("ID", "Sentence"), (1, None)],
            [("ID", "Sentence"), (1, "hello"), (None, None), ("bad ID", "text")],
        ):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                voice._transcriptions_from_rows(rows)

    def test_explicit_bundled_workbook_footer_is_not_a_reference(self):
        rows = [
            ("ID", "Sentence", "extra "),
            (1, "Synthetic reference", None),
            (None, None, None),
            ("type:", "case A", "case B"),
            ("metadata", "synthetic note", "another note"),
        ]
        self.assertEqual(
            voice._transcriptions_from_rows(rows),
            {1: "Synthetic reference"},
        )

    def test_excel_loader_closes_workbook_when_validation_fails(self):
        sheet = Mock()
        sheet.iter_rows.return_value = iter([("ID", "Sentence"), (1, None)])
        workbook = Mock(worksheets=[sheet])
        load = Mock(return_value=workbook)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "references.xlsx"
            path.write_bytes(b"synthetic placeholder; reader is mocked")
            with patch.dict(
                sys.modules, {"openpyxl": SimpleNamespace(load_workbook=load)}
            ):
                with self.assertRaises(ValueError):
                    voice.load_transcriptions_from_excel(path)
        workbook.close.assert_called_once_with()
        sheet.iter_rows.assert_called_once_with(values_only=True)

    def test_task_lookup_rejects_missing_or_empty_references(self):
        self.assertEqual(
            voice.select_transcription_for_audio_file("task12_a.WAV", {12: "hello"}),
            "hello",
        )
        for filename, references in (
            ("recording.wav", {1: "hello"}),
            ("task2_a.wav", {1: "hello"}),
            ("task1_a.wav", {1: ""}),
        ):
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                voice.select_transcription_for_audio_file(filename, references)

    def test_wav_listing_is_sorted_nonrecursive_and_case_insensitive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "task2.wav").touch()
            (root / "task1.WAV").touch()
            (root / "other.mp3").touch()
            (root / "not_a_file.wav").mkdir()
            self.assertEqual(
                voice.load_wav_files_from_folder(root),
                ["task1.WAV", "task2.wav"],
            )

    def test_missing_or_empty_audio_directory_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                voice.load_wav_files_from_folder(root / "missing")
            with self.assertRaisesRegex(ValueError, "No WAV"):
                voice.load_wav_files_from_folder(root)

    def test_audio_duration_rejects_invalid_headers_formats_and_zero_duration(self):
        info = Mock()
        with patch.dict(sys.modules, {"soundfile": SimpleNamespace(info=info)}):
            info.return_value = SimpleNamespace(format="WAV", duration=2.5)
            self.assertEqual(voice.get_audio_duration("fake.wav"), 2.5)
            for format_name, duration in (
                ("MP3", 1),
                ("WAV", 0),
                ("WAV", float("nan")),
            ):
                info.return_value = SimpleNamespace(
                    format=format_name, duration=duration
                )
                with (
                    self.subTest(format=format_name, duration=duration),
                    self.assertRaises(ValueError),
                ):
                    voice.get_audio_duration("fake.wav")
            info.side_effect = RuntimeError("invalid audio")
            with self.assertRaisesRegex(ValueError, "Cannot read WAV header"):
                voice.get_audio_duration("fake.wav")


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for name in ("task2_a.wav", "task1_a.wav"):
            (self.root / name).write_bytes(
                b"synthetic fixture; header reader is mocked"
            )

    def test_all_inputs_are_validated_before_one_model_is_loaded(self):
        events = []
        model = Mock()
        model.transcribe_file.side_effect = ["a", "a b"]

        def duration(path):
            events.append(path.name)
            return 2.0

        def create(device, cache):
            events.append("model")
            return model

        with (
            patch.object(
                voice,
                "load_transcriptions_from_excel",
                return_value={1: "a b", 2: "a b"},
            ),
            patch.object(voice, "get_audio_duration", side_effect=duration),
            patch.object(voice, "create_asr_model", side_effect=create) as factory,
            patch.object(voice.time, "perf_counter", side_effect=[10, 11, 20, 22]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            results = voice.run_evaluation(self.root, model_cache=self.root / "cache")
        self.assertEqual(events, ["task1_a.wav", "task2_a.wav", "model"])
        factory.assert_called_once_with("cpu", self.root / "cache")
        self.assertEqual([item["wer"] for item in results], [0.5, 0.0])
        self.assertEqual([item["rtf"] for item in results], [0.5, 1.0])
        self.assertEqual(
            [call.args[0] for call in model.transcribe_file.call_args_list],
            [str(self.root / "task1_a.wav"), str(self.root / "task2_a.wav")],
        )
        self.assertFalse((self.root / "cache").exists())

    def test_late_missing_reference_prevents_header_reads_and_model_creation(self):
        with (
            patch.object(
                voice, "load_transcriptions_from_excel", return_value={1: "hello"}
            ),
            patch.object(voice, "get_audio_duration") as duration,
            patch.object(voice, "create_asr_model") as factory,
            self.assertRaisesRegex(ValueError, "No reference for task 2"),
        ):
            voice.run_evaluation(self.root)
        duration.assert_not_called()
        factory.assert_not_called()

    def test_late_invalid_wav_prevents_model_creation(self):
        with (
            patch.object(
                voice,
                "load_transcriptions_from_excel",
                return_value={1: "hello", 2: "hi"},
            ),
            patch.object(
                voice, "get_audio_duration", side_effect=[1.0, ValueError("bad header")]
            ),
            patch.object(voice, "create_asr_model") as factory,
            self.assertRaisesRegex(ValueError, "bad header"),
        ):
            voice.run_evaluation(self.root)
        factory.assert_not_called()

    def test_invalid_device_is_rejected_before_inputs_or_model_are_loaded(self):
        with patch.object(voice, "create_asr_model") as factory:
            with self.assertRaisesRegex(ValueError, "device must"):
                voice.run_evaluation(self.root, device="invalid")
        factory.assert_not_called()

    def test_factory_uses_current_api_original_model_id_and_selected_device(self):
        encoder_decoder = Mock()
        modules = {
            "speechbrain.inference.ASR": SimpleNamespace(
                EncoderDecoderASR=encoder_decoder
            )
        }
        with patch.dict(sys.modules, modules):
            result = voice.create_asr_model("cpu", self.root / "cache")
        encoder_decoder.from_hparams.assert_called_once_with(
            source="speechbrain/asr-transformer-transformerlm-librispeech",
            savedir=str(self.root / "cache"),
            run_opts={"device": "cpu"},
        )
        self.assertIs(result, encoder_decoder.from_hparams.return_value)

    def test_unavailable_cuda_does_not_try_loading_a_model(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with self.assertRaisesRegex(ValueError, "CUDA is unavailable"):
                voice.create_asr_model("cuda", self.root / "cache")

    def test_cli_defaults_to_cpu_and_passes_explicit_paths(self):
        with patch.object(voice, "run_evaluation") as run:
            status = voice.main(["--audio-dir", str(self.root)])
        self.assertEqual(status, 0)
        run.assert_called_once_with(
            self.root,
            voice.DEFAULT_REFERENCE,
            device="cpu",
            model_cache=voice.DEFAULT_MODEL_CACHE,
        )

    def test_cli_requires_audio_directory(self):
        with (
            patch.object(voice, "run_evaluation") as run,
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            voice.main([])
        self.assertEqual(raised.exception.code, 2)
        run.assert_not_called()

    def test_import_and_help_do_not_load_dependencies_change_environment_or_write(self):
        script = """
import builtins
import os
import runpy
import sys

real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'speechbrain', 'torch', 'torchaudio', 'soundfile', 'openpyxl', 'pandas', 'numpy', 'pydub'}:
        raise AssertionError('Optional dependency imported: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
before = dict(os.environ)
path, mode = sys.argv[1:]
if mode == 'help':
    sys.argv = [path, '--help']
    try:
        runpy.run_path(path, run_name='__main__')
    except SystemExit as error:
        assert error.code == 0
else:
    runpy.run_path(path, run_name='voice_import_check')
assert dict(os.environ) == before
"""
        for mode in ("import", "help"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        "-c",
                        script,
                        str(Path(voice.__file__).resolve()),
                        mode,
                    ],
                    cwd=directory,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
