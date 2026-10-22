# Data and historical artifacts

The original repository is preserved as a record of speech and vision coursework.
Dataset collection, ownership, participant consent and redistribution terms have
not yet been documented in the repository. No new dataset license is assigned by
this cleanup.

## Current inputs

| Material | Contents and limitations |
| --- | --- |
| `Speech Recognition Dataset/speechrecognition Dataset v2/` | 96 MP3 and 96 WAV recordings, a transcript and an Excel reference sheet. Filenames distinguish task IDs and recording conditions. These labels do not independently verify speaker identity or recording provenance. |
| `audio_old/` | 96 WAV recordings that are byte-identical to their matching dataset WAV files. This is a duplicate working copy, not an independent test set. |
| `speechrecognition dataset details.xlsx` | Older root workbook with only five task references; the fifth sentence is a single character. It does not cover all 12 dataset tasks. The cleaned-up speech script defaults to the 12-task workbook inside the v2 dataset directory. |
| `Speech Recognition Dataset.zip` and `demo dataset.zip` | The dataset archive contains the 96 MP3s, 96 WAVs, workbook and transcript; the demo archive contains a transcript only. Both are retained without replacement. |
| `image/p/` | 169 portrait images in seven `persoon…` folders. Folder labels are training labels, not verified identities. |
| `image/n/` and `peeps_bunch.jpg` | 32 negative training images and a separate demonstration image. |
| `positives.txt`, `positives copy.txt`, `negatives.txt` | Original machine-specific annotation lists. The primary positive list has 357 rows, of which 169 resolve to current images and 188 reference missing files. The alternate list is retained as historical material. |
| Cascade XML and stage files | Historical pretrained and experiment artifacts; see [THIRD_PARTY.md](THIRD_PARTY.md). |

The cleanup keeps these input bytes unchanged. New outputs belong in separate
directories. It does not manufacture missing training images or a benchmark split.
The two workbooks contain prompts and recording-condition labels, but no dataset
license or participant-consent statement was found in them.
They also contain speaker-identifying notes and creator/editor metadata. Those
values are not reproduced here; permission to redistribute the workbooks must
cover that identifying information too.

## History and attribution

The original 14 commits, from December 2023 to January 2024, use the author name
`kozdonM` and the literal author-email field `kozdonM`. That field is not a valid
email address, and GitHub did not associate those commits with a user during the
review. Commit hashes, author fields and dates are preserved; no attribution is
rewritten. The repository is hosted by Michal Kozdon's `ObviouslyNotMich` account.
This history does not establish sole authorship of supplied teaching materials,
datasets or third-party models.

Earlier commits contain additional working artifacts, including audio and model
training files that are not in the current tree. Removing a file in a normal
cleanup commit does not remove its historical copies. Publication and transfer
permission therefore need to cover the retained history as well as current files.

## Validation limits

The cleanup's software tests use synthetic inputs and fake external dependencies.
They do not transcribe the recordings, identify portrait subjects, validate
participant consent, establish recognition accuracy or reproduce the historical
experiments. Dataset source, required credit and any redistribution restrictions
must be supplied by the owner before a professional-account transfer is finalized.
