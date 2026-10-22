# Third-party components and license status

## OpenCV cascades

`haarcascade_frontalface_default.xml` and `haarcascade_eye.xml` contain their own
OpenCV/Intel copyright, redistribution conditions and contributor notices in
their XML headers. Those notices and the model bytes are preserved. These files
are pretrained detector assets, not models created by this project.
The eye cascade credits Shameem Hameed; the frontal-face cascade credits
Rainer Lienhart.

OpenCV's [face-recognition documentation](https://docs.opencv.org/4.x/da/d60/tutorial_face_main.html)
describes the EigenFace recognizer used here. Its
[cascade-training documentation](https://docs.opencv.org/4.x/dc/d88/tutorial_traincascade.html)
explains the legacy command-line training tools. The detection/training library
and algorithms are upstream work; this repository contains coursework scripts
that use them.

The smaller `cascade.xml`, `cascade140.xml`, `classifier/` and `clas 140/` files
are retained historical experiment artifacts. Their complete training inputs,
evaluation and ownership are not established by the files alone; no performance
or new licensing claim is made for them.

## SpeechBrain model

The speech script uses
[`speechbrain/asr-transformer-transformerlm-librispeech`](https://huggingface.co/speechbrain/asr-transformer-transformerlm-librispeech).
The publisher's model card identifies Apache-2.0 licensing and describes an
English recognizer pretrained on LibriSpeech. Its published LibriSpeech scores
are upstream results, not evaluations of this repository's recordings.

Model files are obtained by SpeechBrain only when inference is explicitly run.
The cleanup removes nine machine-specific cache symlinks from the current tree;
they did not contain checkpoint weights. Original history also mentions
`speechbrain/asr-crdnn-commonvoice-14-en`, but the cleaned-up evaluation script
does not select that model.

## Project code and datasets

No repository-wide license was present in the original project. This cleanup
does not grant a new blanket license for the coursework, portraits, recordings,
spreadsheets, transcripts or experiment artifacts. Dependencies and pretrained
assets retain their own terms. Public availability alone does not establish
permission to reuse those separately authored materials.

See [DATA_PROVENANCE.md](DATA_PROVENANCE.md) for the dataset inventory and the
limits of the available provenance information.
