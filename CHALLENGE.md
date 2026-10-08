# Image self-supervised learning challenge

Train an image encoder without labels, export it to ONNX, and upload it on the
[challenge page](https://galilai-group.github.io/stable-challenge-image/). The organizers embed fixed evaluation
images with your model and score how well simple linear probes on those embeddings classify them.

> **Submission at a glance**
>
> - **Format:** a single `.onnx` file. Not a PyTorch checkpoint, zip, or weights without the architecture.
> - **Size cap: 20 MiB per file.** Larger uploads are rejected. Check with `ls -lh model.onnx`.
> - **Limit:** up to 5 submissions per day.
> - **Only your latest submission counts:** the leaderboard uses your most recent successfully evaluated model,
>   not your best. A worse resubmission replaces a better one.
> - **Leaderboard updates:** the leaderboard is refreshed periodically (likely every few days), not immediately
>   after each submission.
> - **Before uploading:** `uv run test_submission.py model.onnx` must pass. The submission page repeats these
>   checks on your file and won't upload a file that fails them.

## Quick start

The environment is managed with [uv](https://docs.astral.sh/uv/getting-started/installation/). `uv sync` creates
`.venv` from [pyproject.toml](pyproject.toml), with the exact versions pinned in `uv.lock`.

```bash
git clone https://github.com/galilai-group/stable-challenge-image.git
cd stable-challenge-image
uv sync                                                    # create .venv with all dependencies
uv run training_data.py                                    # downloads data/train.parquet on first run: 7,500 images, 96x96
uv run test_submission.py example_submission/model.onnx    # check a model before submitting it
```

Use `uv run` in front of any command (e.g. `uv run python my_training.py`), or activate the environment with
`source .venv/bin/activate`. Add packages you need for training with `uv add <package>`.

## Training data

`load_training_data()` downloads [`train.parquet`](https://drive.google.com/file/d/1QRi4DL_S6CGoeSS2rhdEA1ujJMOLdMQF/view?usp=sharing)
(122 MB) into `data/` the first time it runs; you can also download it there yourself. It holds 7,500 images. Format is two columns: `image` (96×96 RGB) and `label` (a class id 0–29, or -1 for unlabelled). Only 10% of
the images are labelled. The file does not say which domain an image comes from.

Load it with `load_training_data()` (or `load_training_data("path/to/train.parquet")` for a copy elsewhere) from [training_data.py](training_data.py). Pass
`labelled_only=True` to keep only the labelled 10%.

The data is meant for self-supervised learning. You may use the labels for monitoring, for example with
stable-pretraining's `OnlineProbe`, but not in the training objective.

## Evaluation

The probes are fit by the organizers. Your model only produces embeddings. Each embedding dimension is
standardized before the probe is fit, so the overall scale of your embeddings doesn't matter. The evaluation runs
on CPU and is deterministic.

[evaluate.py](evaluate.py) runs the same embedding and ridge probe on any parquet file with `image` and `label`
columns, and reports the probe's leave-one-out accuracy. For example, `uv run evaluate.py model.onnx --data
data/train.parquet` scores your model on the labelled training images.


## Submission

Upload a single ONNX file that matches the model format below. Before submitting, make sure
`uv run test_submission.py model.onnx` passes (see below).

- **Size cap: 20 MiB.** The submission page rejects larger files. In float32 that is roughly 5 million parameters,
  so a standard ResNet-18 (11.7M parameters, ~45 MB) is too big.
- **Up to 5 submissions per day.** Every upload counts, even one whose evaluation later fails, so validate locally
  first.
- **Only your latest submission counts.** The leaderboard ranks the most recent model that evaluated successfully,
  not your best, so only resubmit a model you expect to beat your current one. A submission that fails evaluation
  appears as "Failed", with public diagnostics under "View logs".
- **Periodic leaderboard updates.** The leaderboard is refreshed periodically (likely every few days), so a new
  submission may not appear on it right away.

## Model format

An ONNX file with:

- **input** `image`: float32, `[B, 3, 96, 96]`, with a dynamic batch size `B`
- **output** `embedding`: float32, `[B, 1024]`

**Preprocessing is done by the evaluator, not your model.** Each image is RGB scaled to [0, 1] and then normalised
with the ImageNet mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)` before it reaches your model. Train
with the same normalisation, and don't normalise again inside the exported model.

For example, `export_onnx(encoder, path)` in [export_onnx.py](export_onnx.py) exports a PyTorch encoder. If the
encoder outputs fewer than 1024 features it zero-pads them, which doesn't change the probe. It also checks the
file with onnxruntime. Call `encoder.eval()` before exporting.

## Validate submission

Check your model before uploading it:

```bash
uv run test_submission.py path/to/model.onnx
```

It runs the same steps as the official evaluation, including the 20 MiB size check, and explains how to fix any
problem it finds. The checks are listed at the top of [test_submission.py](test_submission.py). Once it passes,
upload the file on the [challenge page](https://galilai-group.github.io/stable-challenge-image/).

The submission page runs the same checks again in your browser when you choose the file, except the final
`evaluate.py` run, and only uploads files that pass. A file it rejects is not uploaded and doesn't count toward your
daily attempts. Run `test_submission.py` locally anyway: it is quicker to iterate on, and it is the reference if
the two ever disagree.
