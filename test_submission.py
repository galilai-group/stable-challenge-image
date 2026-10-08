"""Check an ONNX submission before uploading it.

    uv run test_submission.py path/to/model.onnx

Runs the same steps as the official evaluation, in order, and stops at the first problem with an explanation of
how to fix it:

    1. the file is an ONNX model within the 20 MiB upload size limit (--max-mb), not a PyTorch checkpoint,
       a zip, a Git LFS pointer, a web page saved from a share link, ...
    2. the ONNX graph is valid and stored in a single file (no separate external-data weights file)
    3. it has exactly one input, "image", float32 [batch, 3, 96, 96] with a dynamic batch size, and its first
       output is a float [batch, 1024] embedding
    4. onnxruntime loads it on CPU (standard ONNX operators, a supported opset and IR version)
    5. it returns finite embeddings of the right shape for batch sizes 1, 3 and 4
    6. evaluate.py runs end to end on the three images in data/sample.parquet. The score on them is not
       meaningful.

Exit code 0 means the model is ready to submit. Lines marked "WARN" don't block submission but point at likely
problems, for example identical embeddings for every image, or BatchNorm/dropout exported in training mode (call
model.eval() before exporting).
"""

import argparse
import os
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))  # so `import evaluate` works from any directory

import evaluate  # noqa: E402

SAMPLE = ROOT / "data" / "sample.parquet"
MAX_MB = 20  # upload limit of the submission service, in MiB
SIZE, DIM = evaluate.SIZE, evaluate.DIM
EXPORT_HINT = ("The easiest fix is to export with export_onnx(encoder, path) from export_onnx.py, which sets the "
               "input/output names, a dynamic batch size and zero-pads the embedding to 1024 features.")


class Failed(Exception):
    """A check failed. The message tells the participant what is wrong and how to fix it."""


def warn(ctx, msg):
    ctx["warnings"].append(msg)
    print(f"      WARN  {msg}")


def fmt_dims(dims):
    return "unknown" if dims is None else "[" + ", ".join("?" if d is None else str(d) for d in dims) + "]"


def tensor_info(value_info):
    """(element type name, dims) of a graph input/output; dims entries are ints, names (dynamic) or None."""
    import onnx

    if not value_info.type.HasField("tensor_type"):
        return None, None
    t = value_info.type.tensor_type
    dtype = onnx.TensorProto.DataType.Name(t.elem_type)
    if not t.HasField("shape"):
        return dtype, None
    dims = []
    for d in t.shape.dim:
        dims.append(d.dim_value if d.HasField("dim_value") else (d.dim_param or None))
    return dtype, dims


# --------------------------------------------------------------------------------------------- 1. file


def sniff(path, head):
    """Explain what the file is if it is clearly not an ONNX model, else None."""
    pytorch = ("This looks like a PyTorch checkpoint (torch.save / TorchScript), not an ONNX model. Load your "
               "encoder in PyTorch and export it with export_onnx(encoder, 'model.onnx') from export_onnx.py.")
    if head.startswith(b"version https://git-lfs"):
        return ("This is a Git LFS pointer file, not the model itself. Run `git lfs pull` (or download the file "
                "directly) to get the real model.")
    if head.lstrip()[:1] == b"<":
        return ("This is an HTML/XML page, not a model. This usually happens when downloading from a share link "
                "(Google Drive, Colab, Dropbox, ...) saves the web page instead of the file. Download the file "
                "itself, for example from the browser's download button.")
    if head.startswith(b"PK"):
        try:
            names = zipfile.ZipFile(path).namelist()
        except zipfile.BadZipFile:
            return "This is a damaged zip archive, not an ONNX model."
        onnx_files = [n for n in names if n.lower().endswith(".onnx")]
        if onnx_files:
            return (f"This is a zip archive containing {onnx_files[0]!r}. Unzip it and check/submit the .onnx file "
                    f"itself.")
        if any(n.endswith(("data.pkl", "constants.pkl", "version")) for n in names):
            return pytorch
        return "This is a zip archive, not an ONNX model. Submit the .onnx file itself."
    if head.startswith(b"\x1f\x8b"):
        return "This is a gzip-compressed file. Decompress it (gunzip) and submit the .onnx file itself."
    if head[:1] == b"\x80" and head[1:2] in (b"\x02", b"\x03", b"\x04", b"\x05"):
        return pytorch.replace("(torch.save / TorchScript)", "or another Python pickle")
    if head.startswith(b"\x89HDF"):
        return ("This is an HDF5 file (for example a Keras .h5 model), not ONNX. Convert it with tf2onnx, or export "
                "your PyTorch encoder with export_onnx.py.")
    if head.startswith(b"\x93NUMPY"):
        return "This is a NumPy array file (.npy), not an ONNX model."
    if len(head) >= 10 and head[8:10] == b'{"':
        return ("This looks like a safetensors file, which holds only weights. Load the weights into your model "
                "in PyTorch and export it with export_onnx.py.")
    try:
        text = head.decode("utf-8")
        if all(c.isprintable() or c.isspace() for c in text):
            return f"This is a text file, not an ONNX model. It starts with: {text[:80]!r}"
    except UnicodeDecodeError:
        pass
    return None


def check_file(ctx):
    path = ctx["path"]
    if not path.exists():
        raise Failed(f"There is no file at {str(path)!r}. Check the path; relative paths start from the current "
                     f"directory, {os.getcwd()}.")
    if path.is_dir():
        if (path / "saved_model.pb").exists():
            raise Failed("This is a TensorFlow SavedModel folder. Convert it to ONNX with tf2onnx.")
        found = sorted(path.glob("*.onnx"))
        hint = f" Did you mean {str(found[0])!r}?" if found else ""
        raise Failed(f"{str(path)!r} is a folder. Pass the path of the .onnx file instead.{hint}")
    size = path.stat().st_size
    if size == 0:
        raise Failed("The file is empty (0 bytes). The export or the copy did not finish; export the model again.")
    with open(path, "rb") as f:
        head = f.read(512)
    problem = sniff(path, head)
    if problem:
        raise Failed(problem)
    if size > ctx["max_mb"] * 1024**2:
        raise Failed(f"The file is {size / 1024**2:,.1f} MiB, over the {ctx['max_mb']:,g} MiB upload limit, so the "
                     f"submission page will reject it. Use a smaller encoder (about {ctx['max_mb'] / 4:,g} million "
                     f"float32 parameters at most).")
    if path.suffix.lower() != ".onnx":
        warn(ctx, f"The file name ends in {path.suffix or '(no extension)'!r} rather than '.onnx'. That's fine for "
                  f"evaluation, but double-check that this is the file you meant.")
    return f"{path.name}, {size / 1024**2:.1f} MiB"


# --------------------------------------------------------------------------------------------- 2. graph


def check_graph(ctx):
    import onnx
    from onnx.external_data_helper import uses_external_data

    try:
        model = onnx.load(str(ctx["path"]), load_external_data=False)
    except Exception as e:
        raise Failed(f"The file is not a readable ONNX model ({type(e).__name__}: {e}). It may be truncated by an "
                     f"interrupted export, copy or download, or not be ONNX at all. Export it again.") from None
    if not model.graph.node and not model.graph.output:
        raise Failed("The file parses but contains no model graph, so it is probably not an ONNX model. Export it "
                     "again with export_onnx.py.")
    external = [t.name for t in model.graph.initializer if uses_external_data(t)]
    if external:
        raise Failed(f"The weights ({len(external)} tensors, e.g. {external[0]!r}) are stored in separate files next "
                     f"to the .onnx (ONNX 'external data'), but only the .onnx file is uploaded. Merge them into one "
                     f"file with:\n"
                     f"    import onnx; onnx.save(onnx.load('{ctx['path'].name}'), 'merged.onnx')\n"
                     f"This only works for models under 2 GB.")
    try:
        onnx.checker.check_model(model)
    except Exception as e:
        warn(ctx, f"onnx.checker reports a problem: {str(e).strip().splitlines()[0]}. Evaluation only needs "
                  f"onnxruntime to run the model, which is checked below.")
    ctx["model"] = model
    opsets = {o.domain or "ai.onnx": o.version for o in model.opset_import}
    ctx["opset"] = opsets.get("ai.onnx")
    return f"{len(model.graph.node)} nodes, opset {ctx['opset']}"


# --------------------------------------------------------------------------------------------- 3. inputs/outputs


def check_signature(ctx):
    graph = ctx["model"].graph
    weights = {t.name for t in graph.initializer}
    inputs = [i for i in graph.input if i.name not in weights]  # old exporters also list weights as inputs
    if not inputs:
        raise Failed("The model has no inputs. It must take one input named 'image'. " + EXPORT_HINT)
    if len(inputs) > 1:
        names = ", ".join(repr(i.name) for i in inputs)
        raise Failed(f"The model has {len(inputs)} inputs ({names}), but evaluation passes only one, 'image'. Make "
                     f"forward() take a single image tensor (extra arguments such as labels or masks become extra "
                     f"inputs when exporting). " + EXPORT_HINT)
    inp = inputs[0]
    if inp.name != "image":
        raise Failed(f"The input is named {inp.name!r}, but it must be named 'image'. Pass input_names=['image'] to "
                     f"torch.onnx.export. " + EXPORT_HINT)
    dtype, dims = tensor_info(inp)
    if dtype is None:
        raise Failed("The input 'image' is not a tensor. It must be a float32 tensor [batch, 3, 96, 96].")
    if dtype != "FLOAT":
        raise Failed(f"The input 'image' has type {dtype.lower()}, but evaluation passes float32. Export the model "
                     f"in float32 (no model.half() or quantised inputs) and with a float32 example input.")
    if dims is None:
        warn(ctx, "The input shape is not recorded in the model; it is tested by running the model below.")
    else:
        expected = f"[batch, 3, {SIZE}, {SIZE}]"
        if len(dims) != 4:
            hint = {3: " It looks like the batch dimension is missing.",
                    2: " It looks like the model expects flattened pixels; flatten inside the model instead."}
            raise Failed(f"The input 'image' has shape {fmt_dims(dims)} ({len(dims)} dimensions), but evaluation "
                         f"passes {expected}.{hint.get(len(dims), '')}")
        batch, c, h, w = dims
        if (c, h, w) == (SIZE, SIZE, 3):
            raise Failed(f"The input 'image' is channels-last {fmt_dims(dims)}, but evaluation passes channels-first "
                         f"{expected}. Remove the permute from your preprocessing or do it inside the model.")
        for name, value, want in (("channels", c, 3), ("height", h, SIZE), ("width", w, SIZE)):
            if isinstance(value, int) and value != want:
                hint = (" To use a backbone trained at another resolution, resize inside the model, e.g. with "
                        "F.interpolate.") if name != "channels" else ""
                raise Failed(f"The input 'image' has {name} {value}, but evaluation passes {expected} (RGB, "
                             f"{SIZE}x{SIZE}).{hint}")
        if isinstance(batch, int):
            raise Failed(f"The batch size of 'image' is fixed to {batch}, but evaluation passes batches of up to 256 "
                         f"images and a smaller last batch. Make it dynamic: pass dynamic_axes={{'image': {{0: "
                         f"'batch'}}}} to torch.onnx.export. " + EXPORT_HINT)

    outputs = list(graph.output)
    if not outputs:
        raise Failed("The model has no outputs. " + EXPORT_HINT)
    out = outputs[0]
    if len(outputs) > 1:
        warn(ctx, f"The model has {len(outputs)} outputs; evaluation uses only the first one, {out.name!r}, as the "
                  f"embedding. Make sure that's the encoder output (not, e.g., a projector output or logits).")
    if out.name != "embedding":
        warn(ctx, f"The output is named {out.name!r} instead of 'embedding'. Evaluation still uses it, but pass "
                  f"output_names=['embedding'] to keep to the format.")
    dtype, dims = tensor_info(out)
    if dtype is None:
        raise Failed(f"The output {out.name!r} is not a tensor (it may be a list or dict). Return a single tensor "
                     f"[batch, 1024] from forward().")
    if dtype.startswith(("INT", "UINT", "BOOL")):
        raise Failed(f"The output {out.name!r} has type {dtype.lower()}. That looks like class predictions (e.g. "
                     f"argmax), but evaluation needs float embeddings [batch, 1024].")
    if dtype != "FLOAT":
        warn(ctx, f"The output has type {dtype.lower()} rather than float32. Evaluation accepts it, but float32 is "
                  f"the expected format.")
    if dims is not None:
        if len(dims) != 2:
            hint = {4: " For a feature map such as [batch, C, 1, 1], flatten it inside the model: x.flatten(1).",
                    3: " For token embeddings [batch, tokens, dim], pool them, e.g. the CLS token x[:, 0] or the "
                       "mean x.mean(1).",
                    1: " It looks like the batch dimension was lost, e.g. by a squeeze or a mean over the batch."}
            raise Failed(f"The output has shape {fmt_dims(dims)}, but it must be [batch, {DIM}].{hint.get(len(dims), '')}")
        if isinstance(dims[1], int) and dims[1] != DIM:
            fix = ("Pad it with zeros, which doesn't change the probe. " + EXPORT_HINT if dims[1] < DIM else
                   f"Reduce it to at most {DIM} features (e.g. a smaller last layer) and zero-pad the rest.")
            raise Failed(f"The embedding has {dims[1]} features, but it must have exactly {DIM}. {fix}")
    return f"input image {fmt_dims(tensor_info(inp)[1])} -> output {out.name} {fmt_dims(dims)}"


# --------------------------------------------------------------------------------------------- 4. onnxruntime


def check_load(ctx):
    import onnxruntime as ort

    try:
        sess = ort.InferenceSession(str(ctx["path"]), providers=["CPUExecutionProvider"])
    except Exception as e:
        msg = str(e)
        if "IR version" in msg:
            hint = ("The model file format (ONNX IR version) is newer than onnxruntime "
                    f"{ort.__version__} reads, usually because a newer onnx/torch wrote it. Lower it with:\n"
                    f"    import onnx; m = onnx.load('{ctx['path'].name}'); m.ir_version = 10; "
                    f"onnx.save(m, '{ctx['path'].name}')")
        elif "opset" in msg.lower() and ("support" in msg.lower() or "version" in msg.lower()):
            hint = (f"The model uses ONNX opset {ctx.get('opset')}, newer than onnxruntime {ort.__version__} "
                    f"supports. Export with opset_version=17.")
        elif "No Op registered" in msg or "custom" in msg.lower() or "domain" in msg.lower():
            hint = ("The model uses an operator that onnxruntime doesn't have (a custom op or a non-standard "
                    "domain). Evaluation uses plain onnxruntime on CPU, so the model must use standard ONNX ops; "
                    "replace the custom layer (e.g. a fused CUDA kernel) with plain PyTorch ops before export.")
        else:
            hint = "Re-export the model, or simplify the layer named in the error above."
        raise Failed(f"onnxruntime {ort.__version__} could not load the model on CPU:\n    {msg.strip()}\n{hint}")
    if [i.name for i in sess.get_inputs()] != ["image"]:  # the exact check evaluate.py makes
        names = [i.name for i in sess.get_inputs()]
        raise Failed(f"onnxruntime sees the inputs {names}, but the model must have exactly one input, 'image'. "
                     + EXPORT_HINT)
    ctx["session"] = sess
    return f"onnxruntime {ort.__version__}, CPU"


# --------------------------------------------------------------------------------------------- 5. inference


def explain_submission_error(e):
    msg = str(e)
    if msg.startswith("Embedding shape"):
        return f"{msg}. " + EXPORT_HINT
    if "NaN or infinite" in msg:
        return (f"{msg}. Look for divisions by zero, log(0), normalising by a zero standard deviation, or float16 "
                f"overflow. Also check that the inputs are what the model expects: RGB in [0, 1], then normalised "
                f"with the ImageNet mean/std.")
    if msg.startswith("Model failed"):
        return (f"{msg}\nThe model crashed while running. If it works for the batch size you exported with but not "
                f"others, a shape was hard-coded during export (e.g. x.view(2, -1)); use x.flatten(1) or "
                f"x.reshape(x.shape[0], -1) and export with a dynamic batch size.")
    return msg


def images(n, seed=0):
    """`n` 96x96 uint8 RGB images: the sample images first, then noise."""
    from datasets import Dataset, disable_progress_bars

    disable_progress_bars()
    out = [np.asarray(im.convert("RGB")) for im in Dataset.from_parquet(str(SAMPLE))["image"]][:n]
    rng = np.random.default_rng(seed)
    out += [rng.integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8) for _ in range(n - len(out))]
    return out


def check_inference(ctx):
    sess = ctx["session"]
    imgs = images(7)
    try:
        t0 = time.time()
        z = evaluate.embed(sess, {"image": imgs}, batch=4)  # batches of 4 and 3
        seconds = time.time() - t0
        single = evaluate.embed(sess, {"image": imgs[:1]}, batch=1)
        again = evaluate.embed(sess, {"image": imgs}, batch=4)
    except evaluate.SubmissionError as e:
        raise Failed(explain_submission_error(e)) from None

    if z.std(0).max() < 1e-6:
        warn(ctx, "Every image gets the same embedding, so the probe can only guess (chance accuracy). The encoder "
                  "may have collapsed during training, or the model ignores its input.")
    if not np.allclose(z, again, rtol=1e-3, atol=1e-5):
        warn(ctx, "The model gives different embeddings for the same images on different runs. It probably "
                  "contains randomness such as dropout; call model.eval() before exporting.")
    elif not np.allclose(z[:1], single, rtol=1e-3, atol=1e-4):
        warn(ctx, "An image's embedding depends on the other images in its batch. This usually means BatchNorm was "
                  "exported in training mode; call model.eval() before exporting.")
    return f"embeddings {z.shape[1:]} for batch sizes 1, 3 and 4; {1000 * seconds / len(imgs):.1f} ms per image"


# --------------------------------------------------------------------------------------------- 6. evaluate.py


def check_evaluate(ctx):
    try:
        with np.errstate(all="ignore"):  # collapsed embeddings make the probe divide by zero; warned about above
            r = evaluate.evaluate(str(ctx["path"]), str(SAMPLE))
    except evaluate.SubmissionError as e:
        raise Failed(explain_submission_error(e)) from None
    if not np.isfinite(r["score"]):
        raise Failed(f"evaluate.py ran but returned a score of {r['score']}. Check the warnings above.")
    return f"evaluate.py ran in {r['seconds']['total']} s (the score on 3 sample images is not meaningful)"


# --------------------------------------------------------------------------------------------- main

STEPS = [
    ("File is an ONNX model", check_file),
    ("ONNX graph is valid", check_graph),
    ("Input and output format", check_signature),
    ("onnxruntime loads the model", check_load),
    ("Model runs on sample images", check_inference),
    ("evaluate.py runs on sample data", check_evaluate),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Check an ONNX model before submitting it.")
    ap.add_argument("onnx_model_path", help="path to your .onnx file")
    ap.add_argument("--max-mb", type=float, default=MAX_MB, help=f"upload size limit in MiB (default {MAX_MB})")
    a = ap.parse_args(argv)
    ctx = {"path": Path(a.onnx_model_path).expanduser(), "max_mb": a.max_mb, "warnings": []}
    print(f"Checking {ctx['path']}\n")
    for i, (title, check) in enumerate(STEPS, 1):
        print(f"[{i}/{len(STEPS)}] {title}", flush=True)
        try:
            note = check(ctx)
        except Failed as e:
            print("      FAIL")
            print("\n" + "\n".join("  " + line for line in str(e).splitlines()))
            print("\nThe model is NOT ready to submit. Fix the problem above and run this check again.")
            return 1
        except Exception:
            print("      FAIL  unexpected error:\n")
            print(traceback.format_exc())
            print("The model is NOT ready to submit. The error above most likely comes from the model; if you "
                  "think it's a bug in this script, report it together with this output.")
            return 1
        print(f"      OK    {note}")
    n = len(ctx["warnings"])
    print(f"\nAll checks passed{f' with {n} warning(s) above' if n else ''}. The model is ready to submit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
