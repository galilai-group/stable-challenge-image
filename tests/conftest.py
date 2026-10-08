"""Tiny synthetic datasets in the same on-disk format as the real training and evaluation data.

Each class has its own mean colour plus noise, so a reasonable encoder plus the ridge probe does well above chance.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from datasets import Dataset, Features, Image, Value

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "example_submission" / "model.onnx"
SAMPLE = ROOT / "data" / "sample.parquet"
CLASSES, SIZE = 3, 96


def images(labels, rng, offset=0, size=SIZE):
    colours = np.random.default_rng(offset).uniform(40, 215, size=(max(labels) + 1, 3))
    x = colours[labels][:, None, None, :] + rng.normal(0, 25, size=(len(labels), size, size, 3))
    return list(np.clip(x, 0, 255).astype(np.uint8))


@pytest.fixture(scope="session")
def eval_data(tmp_path_factory):
    """12 labelled images per class plus 5 unlabelled ones (label -1, must be skipped), like the training data."""
    path, rng = tmp_path_factory.mktemp("eval") / "eval.parquet", np.random.default_rng(0)
    labels = np.repeat(np.arange(CLASSES), 12)
    Dataset.from_dict(
        {"image": images(labels, rng) + images(np.zeros(5, int), rng, offset=9), "label": labels.tolist() + [-1] * 5},
        features=Features({"image": Image(), "label": Value("int64")}),
    ).to_parquet(path)
    return path


def run(script, *args, cwd):
    """Run a repo script with this interpreter from `cwd` (the challenge runner uses a temp dir)."""
    return subprocess.run([sys.executable, str(ROOT / script), *map(str, args)], cwd=cwd,
                          capture_output=True, text=True, timeout=600)


def last_json(stdout):
    return json.loads(stdout.strip().splitlines()[-1])
