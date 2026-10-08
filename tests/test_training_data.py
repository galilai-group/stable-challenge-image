import numpy as np
import pytest
from conftest import images
from datasets import Dataset, Features, Image, Value

import training_data as td


@pytest.fixture
def parquet(tmp_path):
    labels = [3, td.UNLABELLED, 17, td.UNLABELLED, td.UNLABELLED]
    path = tmp_path / "train.parquet"
    Dataset.from_dict({"image": images(np.zeros(len(labels), int), np.random.default_rng(0)), "label": labels},
                      features=Features({"image": Image(), "label": Value("int64")})).to_parquet(path)
    return str(path), labels


def test_load_training_data(parquet):
    path, labels = parquet
    ds = td.load_training_data(path)
    assert ds["label"] == labels
    assert {(im.mode, im.size) for im in ds["image"]} == {("RGB", (96, 96))}


def test_load_labelled_only(parquet):
    path, labels = parquet
    assert td.load_training_data(path, labelled_only=True)["label"] == [3, 17]


def test_missing_file_points_to_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="drive.google.com"):
        td.load_training_data(str(tmp_path / "train.parquet"))


def test_no_path_downloads_to_default(parquet, tmp_path, monkeypatch):
    src, labels = parquet
    dest = str(tmp_path / "data" / "train.parquet")
    monkeypatch.setattr(td, "DEFAULT_PATH", dest)
    monkeypatch.setattr(td, "DOWNLOAD_URL", "file://" + src)
    assert td.load_training_data()["label"] == labels
    monkeypatch.setattr(td, "DOWNLOAD_URL", "file:///nonexistent")  # second call reuses the cached file
    assert td.load_training_data()["label"] == labels
