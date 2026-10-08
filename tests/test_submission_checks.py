"""test_submission.py on the example submission and on the kinds of broken files participants upload."""

import gzip
import shutil
import zipfile

import numpy as np
import onnx
import pytest
from conftest import MODEL
from onnx import TensorProto, helper, numpy_helper

import test_submission as ts


def check(path, capsys, *extra):
    code = ts.main([str(path), *extra])
    return code, capsys.readouterr().out


def toy_model(path, input_name="image", input_shape=("batch", 3, 96, 96), input_type=TensorProto.FLOAT, dim=1024,
              output_shape=None, output_name="embedding", nan=False, constant=False, opset=17,
              ir_version=8):
    """image -> mean over pixels -> linear -> embedding, with knobs to break each part of the format."""
    w = np.zeros((3, dim), np.float32) if constant else np.random.default_rng(0).normal(size=(3, dim)).astype(np.float32)
    nodes = [helper.make_node("ReduceMean", [input_name], ["pooled"], axes=[2, 3], keepdims=0)]
    if input_type != TensorProto.FLOAT:
        nodes = [helper.make_node("Cast", [input_name], ["x"], to=TensorProto.FLOAT),
                 helper.make_node("ReduceMean", ["x"], ["pooled"], axes=[2, 3], keepdims=0)]
    nodes.append(helper.make_node("MatMul", ["pooled", "w"], ["z"]))
    inits = [numpy_helper.from_array(w, "w")]
    if nan:
        inits.append(numpy_helper.from_array(np.array(np.nan, np.float32), "nan"))
        nodes.append(helper.make_node("Mul", ["z", "nan"], [output_name]))
    else:
        nodes.append(helper.make_node("Identity", ["z"], [output_name]))
    graph = helper.make_graph(
        nodes, "toy", [helper.make_tensor_value_info(input_name, input_type, list(input_shape))],
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, list(output_shape or (input_shape[0], dim)))],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = ir_version
    onnx.save(model, str(path))
    return path


def test_example_submission_passes(capsys):
    code, out = check(MODEL, capsys)
    assert code == 0, out
    assert "All checks passed. The model is ready to submit." in out and "WARN" not in out


def test_valid_toy_model_passes(tmp_path, capsys):
    code, out = check(toy_model(tmp_path / "m.onnx"), capsys)
    assert code == 0, out


@pytest.mark.parametrize("make, message", [
    (lambda p: None, "There is no file"),
    (lambda p: p.mkdir(), "is a folder"),
    (lambda p: p.write_bytes(b""), "empty"),
    (lambda p: p.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\n"), "Git LFS pointer"),
    (lambda p: p.write_text("<!DOCTYPE html><html>Google Drive</html>"), "HTML"),
    (lambda p: p.write_bytes(b"\x80\x04\x95" + bytes(100)), "PyTorch checkpoint"),
    (lambda p: p.write_bytes(b"\x89HDF\r\n\x1a\n" + bytes(100)), "HDF5"),
    (lambda p: p.write_text("hello"), "text file"),
    (lambda p: p.write_bytes(MODEL.read_bytes()[:5000]), "not a readable ONNX model"),
])
def test_wrong_files(tmp_path, capsys, make, message):
    path = tmp_path / "model.onnx"
    make(path)
    code, out = check(path, capsys)
    assert code == 1 and message in out and "NOT ready" in out, out


def test_zipped_onnx(tmp_path, capsys):
    path = tmp_path / "model.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.write(MODEL, "submission/model.onnx")
    code, out = check(path, capsys)
    assert code == 1 and "zip archive containing 'submission/model.onnx'" in out


def test_torch_save_zip(tmp_path, capsys):
    path = tmp_path / "model.pt"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("archive/data.pkl", b"\x80\x02")
        z.writestr("archive/version", b"3")
    code, out = check(path, capsys)
    assert code == 1 and "PyTorch checkpoint" in out


def test_gzipped_onnx(tmp_path, capsys):
    path = tmp_path / "model.onnx.gz"
    with gzip.open(path, "wb") as f:
        f.write(MODEL.read_bytes())
    code, out = check(path, capsys)
    assert code == 1 and "gzip" in out


def test_size_limit(capsys):
    code, out = check(MODEL, capsys, "--max-mb", "0.5")
    assert code == 1 and "over the 0.5 MiB upload limit" in out


def test_default_size_limit_is_20_mib(tmp_path, capsys):
    path = toy_model(tmp_path / "m.onnx")
    model = onnx.load(str(path))
    model.graph.initializer.append(numpy_helper.from_array(np.zeros(21 * 1024**2 // 4, np.float32), "padding"))
    onnx.save(model, str(path))
    code, out = check(path, capsys)
    assert code == 1 and "over the 20 MiB upload limit" in out


def test_other_extension_warns(tmp_path, capsys):
    path = tmp_path / "model.bin"
    shutil.copy(MODEL, path)
    code, out = check(path, capsys)
    assert code == 0 and "WARN" in out and "'.bin'" in out


def test_external_data(tmp_path, capsys):
    path = tmp_path / "model.onnx"
    onnx.save(onnx.load(str(MODEL)), str(path), save_as_external_data=True, all_tensors_to_one_file=True,
              location="model.onnx.data", size_threshold=0)
    code, out = check(path, capsys)
    assert code == 1 and "separate files" in out and "onnx.save(onnx.load(" in out


@pytest.mark.parametrize("kwargs, message", [
    (dict(input_name="input"), "named 'input', but it must be named 'image'"),
    (dict(input_type=TensorProto.FLOAT16), "type float16"),
    (dict(input_type=TensorProto.UINT8), "type uint8"),
    (dict(input_shape=("batch", 96, 96, 3)), "channels-last"),
    (dict(input_shape=("batch", 3, 224, 224)), "height 224"),
    (dict(input_shape=("batch", 1, 96, 96)), "channels 1"),
    (dict(input_shape=(1, 3, 96, 96)), "batch size of 'image' is fixed to 1"),
    (dict(dim=512), "512 features, but it must have exactly 1024. Pad it"),
    (dict(dim=2048), "2048 features"),
])
def test_wrong_signature(tmp_path, capsys, kwargs, message):
    code, out = check(toy_model(tmp_path / "m.onnx", **kwargs), capsys)
    assert code == 1 and message in out, out


def test_extra_input(tmp_path, capsys):
    path = toy_model(tmp_path / "m.onnx")
    m = onnx.load(str(path))
    m.graph.input.append(helper.make_tensor_value_info("labels", TensorProto.INT64, ["batch"]))
    onnx.save(m, str(path))
    code, out = check(path, capsys)
    assert code == 1 and "2 inputs ('image', 'labels')" in out


def test_weights_listed_as_inputs_are_fine(tmp_path, capsys):
    """Old exporters (ONNX IR < 4) list every weight as a graph input too."""
    path = toy_model(tmp_path / "m.onnx")
    m = onnx.load(str(path))
    m.graph.input.append(helper.make_tensor_value_info("w", TensorProto.FLOAT, [3, 1024]))
    onnx.save(m, str(path))
    code, out = check(path, capsys)
    assert "2 inputs" not in out


def test_wrong_output_rank(tmp_path, capsys):
    path = toy_model(tmp_path / "m.onnx", output_shape=("batch", 1024, 1, 1))
    code, out = check(path, capsys)
    assert code == 1 and "x.flatten(1)" in out


def test_output_name_warns(tmp_path, capsys):
    code, out = check(toy_model(tmp_path / "m.onnx", output_name="logits"), capsys)
    assert code == 0 and "named 'logits' instead of 'embedding'" in out


def test_nan_embeddings(tmp_path, capsys):
    code, out = check(toy_model(tmp_path / "m.onnx", nan=True), capsys)
    assert code == 1 and "NaN or infinite" in out and "division" in out


def test_constant_embeddings_warn(tmp_path, capsys):
    code, out = check(toy_model(tmp_path / "m.onnx", constant=True), capsys)
    assert code == 0 and "same embedding" in out


def test_unsupported_opset(tmp_path, capsys):
    code, out = check(toy_model(tmp_path / "m.onnx", opset=99), capsys)
    assert code == 1 and "opset_version=17" in out, out


def test_too_new_ir_version(tmp_path, capsys):
    code, out = check(toy_model(tmp_path / "m.onnx", ir_version=99), capsys)
    assert code == 1 and "m.ir_version = 10" in out, out
