# HMI — speech and vision coursework

Python experiments in automatic speech recognition, face detection and EigenFace
recognition. The repository contains evaluation scripts, recorded inputs,
annotations and historical cascade-training artifacts. It is a coursework
snapshot, with no measured accuracy or deployment-readiness claim.

| Component | What it does |
| --- | --- |
| `voice.py` | Transcribes WAV files with a pretrained SpeechBrain English model and compares the output with task references from an Excel workbook. |
| `vision_train_eigen.py` | Crops annotated faces and trains an OpenCV EigenFace recognizer with a saved label mapping. |
| `vision_test.py` | Explicit webcam demo using a trained EigenFace model and a cascade detector. |
| `vision_test_image.py` | Runs a cascade detector on a still image. |
| `vision_annotate.py` | Interactive rectangle annotation tool. |
| `vision_train.py` | Prepares input lists and invokes the legacy OpenCV cascade-training programs through explicit commands. |
| `t.py` | Audits annotation paths and rectangles against available inputs. |

## Inspect without models or hardware

Use Python 3.10 or newer from the repository root:

```bash
python -B -m unittest discover -s tests -v
python voice.py --help
python vision_train_eigen.py --help
python vision_train.py --help
python vision_test.py --help
python vision_test_image.py --help
python vision_annotate.py --help
python t.py --help
```

Imports and help do not open a camera, create models, alter the input dataset or
download weights. Tests use synthetic inputs and fake dependencies. No full
SpeechBrain inference, model training or webcam session was run for the cleanup.

## Speech evaluation

Create a virtual environment and install the speech dependencies:

```bash
python -m venv .venv
# Activate .venv with the command appropriate for your shell.
python -m pip install -r requirements-voice.txt
```

The dependency list is not a tested environment lock. PyTorch and torchaudio must
be a compatible pair for your platform; follow the
[PyTorch installation instructions](https://pytorch.org/get-started/locally/).
The model and its behavior are described in the
[SpeechBrain model card](https://huggingface.co/speechbrain/asr-transformer-transformerlm-librispeech).
The first inference run can download model files into `pretrained_models/`.

Run evaluation explicitly on the committed WAV directory:

```bash
python voice.py --audio-dir "Speech Recognition Dataset/speechrecognition Dataset v2/wav" --device cpu
```

Use `--reference PATH` for a different workbook containing `ID` and `Sentence`
columns, `--model-cache PATH` for a different model cache, and `--device cuda`
only with a suitable PyTorch/CUDA environment. Filenames must contain a `taskN`
identifier that has a reference sentence. The script supports WAV inputs; it
does not convert or overwrite the committed MP3 recordings.
The default workbook is inside the v2 dataset directory and covers all 12 tasks.
The older root workbook contains only five references, including an incomplete
fifth sentence; it is retained as historical material.

Word error rate is word-token Levenshtein distance divided by the reference word
count, after lowercase and punctuation/whitespace normalization. It is computed
on the original normalized tokens, before the fuzzy display alignment. WER can
exceed 1 when there are many insertions. Alignment-based precision/recall values
are diagnostics, not macro-averaged classification scores.
Two empty texts have WER 0; a nonempty hypothesis with an empty reference raises
an error. Evaluation inputs must have nonempty references.

The old script cancelled insertion/deletion counts in one formula and applied
another formula to padded alignment output. Its historical outputs should not
be compared directly with the corrected metric as if they were equivalent. No
new dataset benchmark has been computed. Upstream LibriSpeech model-card results
do not describe these recordings.

## Vision experiments

Install `requirements-vision.txt` in an environment containing only one OpenCV
wheel variant. The desktop contrib package supplies `cv2.face` and GUI support;
headless or base-only OpenCV wheels do not provide the same setup.

The checked-in portrait dataset contains 169 images across seven label folders.
The primary annotation file references 357 images: 188 are absent. The scripts
resolve the historical Windows paths against the local repository, report
missing or invalid annotations, and use only validated available inputs. They
do not reconstruct the missing experiment data.

First audit the original annotation list without OpenCV:

```bash
python t.py
```

An exit status of 1 is expected for the incomplete original list: 169 usable
rows, 188 missing images and two invalid rectangles among the missing rows.
Training refuses those issues unless `--allow-incomplete` is explicitly supplied:

```bash
python vision_train_eigen.py --allow-incomplete --output outputs/vision/eigen
python vision_test.py --model outputs/vision/eigen/eigenface_model.yml --labels outputs/vision/eigen/labels.json
```

The second command opens the selected camera; press `q` to finish. A still-image
detector and the annotation tool are separate explicit commands:

```bash
python vision_test_image.py --image peeps_bunch.jpg --cascade haarcascade_frontalface_default.xml
python vision_annotate.py --output outputs/vision/annotations/positives.txt
```

In the annotation window, drag a rectangle, press Escape for the next image or
`q` to finish. Rectangles are written to the new output file.

Training uses fresh output directories; annotation uses a new file. Existing
outputs are refused. Input images and original annotation files stay unchanged.
Failures can leave partial new outputs for inspection; use a different output
path for a new attempt. The old EigenFace model is not included, so
train a new model and retain its accompanying label mapping before using the
recognition demo. A prediction's distance is not a probability, and the
historical threshold has not been calibrated or evaluated here.

The cascade preparation/training workflow needs separate `opencv_createsamples`
and `opencv_traincascade` binaries from an appropriate OpenCV 3.4 build. Installing
the Python OpenCV package does not supply these programs; see the
[OpenCV training guide](https://docs.opencv.org/4.x/dc/d88/tutorial_traincascade.html).
Prepare a separate cascade run with:

```bash
python vision_train.py prepare --allow-incomplete --output outputs/vision/cascade
python vision_train.py samples --run-dir outputs/vision/cascade
python vision_train.py train --help
```

`samples` and `train` invoke the external programs. For `train`, choose
`--num-positive` for the actual prepared sample count, leaving training margin;
the old setting of 340 positive samples is not supported by the available data.
Use a new run directory for a separate training attempt.

The Windows `run.bat` wrapper forwards arguments to `vision_train.py` without
requiring administrator elevation.

## Attribution, data and license status

The original 14 commits use the author identity `kozdonM`; their history and
authorship fields are preserved. Dataset contributors and the division of course
work are not established by commit metadata alone.

See [DATA_PROVENANCE.md](DATA_PROVENANCE.md) for the input inventory, missing data,
participant/publication permission requirements and historical-copy caveat.
[THIRD_PARTY.md](THIRD_PARTY.md) distinguishes supplied OpenCV cascades and
SpeechBrain models from project work. Their notices are retained. No new blanket
license is assigned to the code or datasets.
