"""Load the training set: 7,500 images (96x96 RGB) from Imagenette, Galaxy10 DECaLS and EuroSAT, ~10% labelled.

With no path, ``train.parquet`` (122 MB) is downloaded from DATA_URL into ``data/`` on first use:

    from training_data import load_training_data
    ds = load_training_data()                                # columns: image (PIL), label (int, -1 = unlabelled)
    labelled = load_training_data(labelled_only=True)
    ds = load_training_data("path/to/train.parquet")         # an existing copy; never downloads

    uv run training_data.py [PATH]                           # print a summary
"""

import os
import shutil
import sys
import urllib.request

from datasets import Dataset

DATA_URL = "https://drive.google.com/file/d/1QRi4DL_S6CGoeSS2rhdEA1ujJMOLdMQF/view?usp=sharing"
DOWNLOAD_URL = "https://drive.usercontent.google.com/download?id=1QRi4DL_S6CGoeSS2rhdEA1ujJMOLdMQF&export=download&confirm=t"
DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "train.parquet")
UNLABELLED = -1


def download_training_data(path=DEFAULT_PATH):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"Downloading training data to {path} ...", file=sys.stderr)
    tmp = path + ".part"
    with urllib.request.urlopen(DOWNLOAD_URL) as response, open(tmp, "wb") as f:
        shutil.copyfileobj(response, f)
    with open(tmp, "rb") as f:
        if f.read(4) != b"PAR1":
            os.remove(tmp)
            raise RuntimeError(f"Download did not return a parquet file. Download train.parquet manually from {DATA_URL}")
    os.replace(tmp, path)
    return path


def load_training_data(path=None, labelled_only=False):
    if path is None:
        path = DEFAULT_PATH
        if not os.path.isfile(path):
            download_training_data(path)
    elif not os.path.isfile(path):
        raise FileNotFoundError(f"No training data at {path!r}. Download train.parquet from {DATA_URL}, "
                                "or call load_training_data() with no path to download it automatically")
    ds = Dataset.from_parquet(path)
    if labelled_only:
        ds = ds.filter(lambda labels: [l != UNLABELLED for l in labels], input_columns="label", batched=True)
    return ds


if __name__ == "__main__":
    ds = load_training_data(*sys.argv[1:2])
    n_labelled = sum(l != UNLABELLED for l in ds["label"])
    print(ds, f"labelled: {n_labelled}/{len(ds)}")
