"use strict";
// Browser version of test_submission.py. The submission page runs it in a Web Worker on the chosen file and only
// uploads files that pass, so a broken model doesn't use up one of the participant's daily attempts.
//
// Keep the checks and messages in step with test_submission.py, which stays the reference. Step 6 (evaluate.py on
// data/sample.parquet) runs only locally.
//
// Protocol: the page posts {file, max_bytes, base_url}; this script posts
//   {type: "steps", steps: [title, ...], command}       once, first
//   {type: "step", index, status: "running"|"ok"|"fail", note?}
//   {type: "warn", index, message}
//   {type: "done", ok, message?}                        the file passed (ok) or must be fixed (message says how)
//   {type: "unavailable", message}                      the check itself could not run; the page doesn't block
//
// submission_check_samples.bin holds the images of data/sample.parquet as uint8 [3, 96, 96, 3]. Regenerate it with
//   uv run python -c "import numpy as np; from datasets import Dataset; np.stack([np.asarray(im.convert('RGB'))
//   for im in Dataset.from_parquet('data/sample.parquet')['image']]).tofile('docs/submission_check_samples.bin')"

const ORT_VERSION = "1.30.0";  // same as the evaluator's onnxruntime
const ORT_URL = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const COMMAND = "uv run test_submission.py model.onnx";
const SIZE = 96, DIM = 1024;
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
const EXPORT_HINT = "The easiest fix is to export with export_onnx(encoder, path) from export_onnx.py, which sets the " +
  "input/output names, a dynamic batch size and zero-pads the embedding to 1024 features.";

class Failed extends Error {}  // a problem with the model; the message says how to fix it
class Unavailable extends Error {}  // the check itself could not run in this browser
class SubmissionError extends Error {}  // raised by embed, like evaluate.SubmissionError

const post = message => self.postMessage(message);

function warn(ctx, message) {
  ctx.warnings.push(message);
  post({type: "warn", index: ctx.step, message});
}

const fmtDims = dims => dims === null ? "unknown" : "[" + dims.map(d => d === null ? "?" : String(d)).join(", ") + "]";
const pyShape = dims => "(" + dims.join(", ") + (dims.length === 1 ? ",)" : ")");
const mib = bytes => (bytes / 1024 ** 2).toLocaleString("en-US", {minimumFractionDigits: 1, maximumFractionDigits: 1});
const num = x => x.toLocaleString("en-US", {maximumSignificantDigits: 6});

// --------------------------------------------------------------------------------------------- ONNX protobuf

// Reads just the parts of an ONNX ModelProto the checks need, skipping weight data, so a 20 MiB file parses quickly.
// Field numbers are from onnx.proto.
class Proto {
  constructor(bytes, start = 0, end = bytes.length) { this.b = bytes; this.pos = start; this.end = end; }

  varint() {
    let result = 0n, shift = 0n;
    for (;;) {
      if (this.pos >= this.end || shift > 63n) throw new Error("truncated or malformed varint");
      const byte = this.b[this.pos++];
      result |= BigInt(byte & 0x7f) << shift;
      if (!(byte & 0x80)) return BigInt.asIntN(64, result);
      shift += 7n;
    }
  }

  // Yields [field number, wire type, value], where value is a number for varints and a sub-reader for
  // length-delimited fields. Other wire types are skipped.
  *fields() {
    while (this.pos < this.end) {
      const key = Number(this.varint()), field = key >>> 3, wire = key & 7;
      if (wire === 0) yield [field, wire, Number(this.varint())];
      else if (wire === 2) {
        const length = Number(this.varint());
        if (length < 0 || this.pos + length > this.end) throw new Error("truncated length-delimited field");
        yield [field, wire, new Proto(this.b, this.pos, this.pos + length)];
        this.pos += length;
      } else if (wire === 1 || wire === 5) {
        this.pos += wire === 1 ? 8 : 4;
        if (this.pos > this.end) throw new Error("truncated fixed-width field");
      } else throw new Error(`unsupported wire type ${wire}`);
    }
  }

  string() { return new TextDecoder().decode(this.b.subarray(this.pos, this.end)); }
}

function parseModel(bytes) {
  const model = {ir_version: null, opsets: {}, graph: null};
  for (const [field, wire, value] of new Proto(bytes).fields()) {
    if (field === 1 && wire === 0) model.ir_version = value;
    else if (field === 8 && wire === 2) {
      let domain = "", version = null;
      for (const [f, w, v] of value.fields()) {
        if (f === 1 && w === 2) domain = v.string();
        else if (f === 2 && w === 0) version = v;
      }
      model.opsets[domain || "ai.onnx"] = version;
    } else if (field === 7 && wire === 2) model.graph = parseGraph(value);
  }
  return model;
}

function parseGraph(proto) {
  const graph = {nodes: 0, inputs: [], outputs: [], initializers: []};
  for (const [field, wire, value] of proto.fields()) {
    if (wire !== 2) continue;
    if (field === 1) graph.nodes++;
    else if (field === 5) {
      let name = "", external = false;
      for (const [f, w, v] of value.fields()) {
        if (f === 8 && w === 2) name = v.string();
        else if (f === 14 && w === 0) external = v === 1;  // data_location == EXTERNAL
      }
      graph.initializers.push({name, external});
    } else if (field === 11 || field === 12) (field === 11 ? graph.inputs : graph.outputs).push(parseValueInfo(value));
  }
  return graph;
}

function parseValueInfo(proto) {
  const info = {name: "", tensor: false, dtype: null, dims: null};
  for (const [field, wire, value] of proto.fields()) {
    if (field === 1 && wire === 2) info.name = value.string();
    else if (field === 2 && wire === 2) {
      for (const [f, w, tensor] of value.fields()) {
        if (f !== 1 || w !== 2) continue;  // TypeProto.tensor_type
        info.tensor = true;
        for (const [tf, tw, tv] of tensor.fields()) {
          if (tf === 1 && tw === 0) info.dtype = DTYPES[tv] || `TYPE_${tv}`;
          else if (tf === 2 && tw === 2) {
            info.dims = [];
            for (const [sf, sw, dim] of tv.fields()) {
              if (sf !== 1 || sw !== 2) continue;
              let d = null;
              for (const [df, dw, dv] of dim.fields()) {
                if (df === 1 && dw === 0) d = dv;
                else if (df === 2 && dw === 2 && d === null) d = dv.string() || null;
              }
              info.dims.push(d);
            }
          }
        }
        info.dtype ??= "UNDEFINED";
      }
    }
  }
  return info;
}

const DTYPES = ["UNDEFINED", "FLOAT", "UINT8", "INT8", "UINT16", "INT16", "INT32", "INT64", "STRING", "BOOL",
  "FLOAT16", "DOUBLE", "UINT32", "UINT64", "COMPLEX64", "COMPLEX128", "BFLOAT16", "FLOAT8E4M3FN", "FLOAT8E4M3FNUZ",
  "FLOAT8E5M2", "FLOAT8E5M2FNUZ", "UINT4", "INT4", "FLOAT4E2M1", "FLOAT8E8M0", "UINT2", "INT2"];

// --------------------------------------------------------------------------------------------- 1. file

function zipNames(bytes) {
  // The end-of-central-directory record is in the last 64 KiB; it points at the list of file names.
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let i = bytes.length - 22; i >= Math.max(0, bytes.length - 65557); i--) {
    if (view.getUint32(i, true) !== 0x06054b50) continue;
    const names = [], count = view.getUint16(i + 10, true);
    let p = view.getUint32(i + 16, true);
    for (let k = 0; k < count; k++) {
      if (p + 46 > bytes.length || view.getUint32(p, true) !== 0x02014b50) return null;
      const n = view.getUint16(p + 28, true), extra = view.getUint16(p + 30, true), comment = view.getUint16(p + 32, true);
      names.push(new TextDecoder().decode(bytes.subarray(p + 46, p + 46 + n)));
      p += 46 + n + extra + comment;
    }
    return names;
  }
  return null;
}

function sniff(bytes, head) {
  const starts = prefix => typeof prefix === "string"
    ? [...prefix].every((c, i) => head[i] === c.charCodeAt(0)) : prefix.every((b, i) => head[i] === b);
  const pytorch = "This looks like a PyTorch checkpoint (torch.save / TorchScript), not an ONNX model. Load your " +
    "encoder in PyTorch and export it with export_onnx(encoder, 'model.onnx') from export_onnx.py.";
  if (starts("version https://git-lfs"))
    return "This is a Git LFS pointer file, not the model itself. Run `git lfs pull` (or download the file " +
      "directly) to get the real model.";
  const firstNonSpace = head.find(b => ![0x20, 0x09, 0x0a, 0x0b, 0x0c, 0x0d].includes(b));
  if (firstNonSpace === 0x3c)
    return "This is an HTML/XML page, not a model. This usually happens when downloading from a share link " +
      "(Google Drive, Colab, Dropbox, ...) saves the web page instead of the file. Download the file " +
      "itself, for example from the browser's download button.";
  if (starts("PK")) {
    const names = zipNames(bytes);
    if (names === null) return "This is a damaged zip archive, not an ONNX model.";
    const onnx = names.filter(n => n.toLowerCase().endsWith(".onnx"));
    if (onnx.length)
      return `This is a zip archive containing '${onnx[0]}'. Unzip it and check/submit the .onnx file itself.`;
    if (names.some(n => ["data.pkl", "constants.pkl", "version"].some(s => n.endsWith(s)))) return pytorch;
    return "This is a zip archive, not an ONNX model. Submit the .onnx file itself.";
  }
  if (starts([0x1f, 0x8b])) return "This is a gzip-compressed file. Decompress it (gunzip) and submit the .onnx file itself.";
  if (head[0] === 0x80 && [2, 3, 4, 5].includes(head[1]))
    return pytorch.replace("(torch.save / TorchScript)", "or another Python pickle");
  if (starts([0x89, 0x48, 0x44, 0x46]))
    return "This is an HDF5 file (for example a Keras .h5 model), not ONNX. Convert it with tf2onnx, or export " +
      "your PyTorch encoder with export_onnx.py.";
  if (starts([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]))  // \x93NUMPY
    return "This is a NumPy array file (.npy), not an ONNX model.";
  if (head.length >= 10 && head[8] === 0x7b && head[9] === 0x22)
    return "This looks like a safetensors file, which holds only weights. Load the weights into your model " +
      "in PyTorch and export it with export_onnx.py.";
  try {
    const text = new TextDecoder("utf-8", {fatal: true}).decode(head);
    if (/^(?:\P{C}|\s)*$/u.test(text))
      return `This is a text file, not an ONNX model. It starts with: '${text.slice(0, 80)}'`;
  } catch { /* not UTF-8 text */ }
  return null;
}

async function checkFile(ctx) {
  const {file} = ctx;
  if (file.size === 0)
    throw new Failed("The file is empty (0 bytes). The export or the copy did not finish; export the model again.");
  ctx.bytes = new Uint8Array(await file.arrayBuffer());
  const problem = sniff(ctx.bytes, ctx.bytes.subarray(0, 512));
  if (problem) throw new Failed(problem);
  const maxMb = ctx.maxBytes / 1024 ** 2;
  if (file.size > ctx.maxBytes)
    throw new Failed(`The file is ${mib(file.size)} MiB, over the ${num(maxMb)} MiB upload limit, so the ` +
      `submission page will reject it. Use a smaller encoder (about ${num(maxMb / 4)} million float32 parameters ` +
      `at most).`);
  const dot = file.name.lastIndexOf(".");
  const suffix = dot > 0 ? file.name.slice(dot) : "";
  if (suffix.toLowerCase() !== ".onnx")
    warn(ctx, `The file name ends in '${suffix || "(no extension)"}' rather than '.onnx'. That's fine for ` +
      `evaluation, but double-check that this is the file you meant.`);
  return `${file.name}, ${mib(file.size)} MiB`;
}

// --------------------------------------------------------------------------------------------- 2. graph

function checkGraph(ctx) {
  let model;
  try {
    model = parseModel(ctx.bytes);
  } catch (e) {
    throw new Failed(`The file is not a readable ONNX model (${e.message}). It may be truncated by an interrupted ` +
      `export, copy or download, or not be ONNX at all. Export it again.`);
  }
  const graph = model.graph || {nodes: 0, inputs: [], outputs: [], initializers: []};
  if (!graph.nodes && !graph.outputs.length)
    throw new Failed("The file parses but contains no model graph, so it is probably not an ONNX model. Export it " +
      "again with export_onnx.py.");
  const external = graph.initializers.filter(t => t.external).map(t => t.name);
  if (external.length)
    throw new Failed(`The weights (${external.length} tensors, e.g. '${external[0]}') are stored in separate files ` +
      `next to the .onnx (ONNX 'external data'), but only the .onnx file is uploaded. Merge them into one file ` +
      `with:\n    import onnx; onnx.save(onnx.load('${ctx.file.name}'), 'merged.onnx')\n` +
      `This only works for models under 2 GB.`);
  ctx.model = model;
  ctx.opset = model.opsets["ai.onnx"] ?? null;
  return `${graph.nodes} nodes, opset ${ctx.opset ?? "None"}`;
}

// --------------------------------------------------------------------------------------------- 3. inputs/outputs

function checkSignature(ctx) {
  const graph = ctx.model.graph;
  const weights = new Set(graph.initializers.map(t => t.name));
  const inputs = graph.inputs.filter(i => !weights.has(i.name));  // old exporters also list weights as inputs
  if (!inputs.length) throw new Failed("The model has no inputs. It must take one input named 'image'. " + EXPORT_HINT);
  if (inputs.length > 1)
    throw new Failed(`The model has ${inputs.length} inputs (${inputs.map(i => `'${i.name}'`).join(", ")}), but ` +
      `evaluation passes only one, 'image'. Make forward() take a single image tensor (extra arguments such as ` +
      `labels or masks become extra inputs when exporting). ` + EXPORT_HINT);
  const inp = inputs[0];
  if (inp.name !== "image")
    throw new Failed(`The input is named '${inp.name}', but it must be named 'image'. Pass input_names=['image'] ` +
      `to torch.onnx.export. ` + EXPORT_HINT);
  if (!inp.tensor) throw new Failed("The input 'image' is not a tensor. It must be a float32 tensor [batch, 3, 96, 96].");
  if (inp.dtype !== "FLOAT")
    throw new Failed(`The input 'image' has type ${inp.dtype.toLowerCase()}, but evaluation passes float32. ` +
      `Export the model in float32 (no model.half() or quantised inputs) and with a float32 example input.`);
  const dims = inp.dims;
  if (dims === null) warn(ctx, "The input shape is not recorded in the model; it is tested by running the model below.");
  else {
    const expected = `[batch, 3, ${SIZE}, ${SIZE}]`;
    if (dims.length !== 4) {
      const hint = {3: " It looks like the batch dimension is missing.",
        2: " It looks like the model expects flattened pixels; flatten inside the model instead."}[dims.length] || "";
      throw new Failed(`The input 'image' has shape ${fmtDims(dims)} (${dims.length} dimensions), but evaluation ` +
        `passes ${expected}.${hint}`);
    }
    const [batch, c, h, w] = dims;
    if (c === SIZE && h === SIZE && w === 3)
      throw new Failed(`The input 'image' is channels-last ${fmtDims(dims)}, but evaluation passes channels-first ` +
        `${expected}. Remove the permute from your preprocessing or do it inside the model.`);
    for (const [name, value, want] of [["channels", c, 3], ["height", h, SIZE], ["width", w, SIZE]]) {
      if (typeof value === "number" && value !== want) {
        const hint = name !== "channels" ? " To use a backbone trained at another resolution, resize inside the " +
          "model, e.g. with F.interpolate." : "";
        throw new Failed(`The input 'image' has ${name} ${value}, but evaluation passes ${expected} (RGB, ` +
          `${SIZE}x${SIZE}).${hint}`);
      }
    }
    if (typeof batch === "number")
      throw new Failed(`The batch size of 'image' is fixed to ${batch}, but evaluation passes batches of up to 256 ` +
        `images and a smaller last batch. Make it dynamic: pass dynamic_axes={'image': {0: 'batch'}} to ` +
        `torch.onnx.export. ` + EXPORT_HINT);
  }

  const outputs = graph.outputs;
  if (!outputs.length) throw new Failed("The model has no outputs. " + EXPORT_HINT);
  const out = outputs[0];
  if (outputs.length > 1)
    warn(ctx, `The model has ${outputs.length} outputs; evaluation uses only the first one, '${out.name}', as the ` +
      `embedding. Make sure that's the encoder output (not, e.g., a projector output or logits).`);
  if (out.name !== "embedding")
    warn(ctx, `The output is named '${out.name}' instead of 'embedding'. Evaluation still uses it, but pass ` +
      `output_names=['embedding'] to keep to the format.`);
  if (!out.tensor)
    throw new Failed(`The output '${out.name}' is not a tensor (it may be a list or dict). Return a single tensor ` +
      `[batch, 1024] from forward().`);
  if (/^(INT|UINT|BOOL)/.test(out.dtype))
    throw new Failed(`The output '${out.name}' has type ${out.dtype.toLowerCase()}. That looks like class ` +
      `predictions (e.g. argmax), but evaluation needs float embeddings [batch, 1024].`);
  if (out.dtype !== "FLOAT")
    warn(ctx, `The output has type ${out.dtype.toLowerCase()} rather than float32. Evaluation accepts it, but ` +
      `float32 is the expected format.`);
  if (out.dims !== null) {
    if (out.dims.length !== 2) {
      const hint = {4: " For a feature map such as [batch, C, 1, 1], flatten it inside the model: x.flatten(1).",
        3: " For token embeddings [batch, tokens, dim], pool them, e.g. the CLS token x[:, 0] or the mean x.mean(1).",
        1: " It looks like the batch dimension was lost, e.g. by a squeeze or a mean over the batch."}[out.dims.length];
      throw new Failed(`The output has shape ${fmtDims(out.dims)}, but it must be [batch, ${DIM}].${hint || ""}`);
    }
    const features = out.dims[1];
    if (typeof features === "number" && features !== DIM) {
      const fix = features < DIM ? "Pad it with zeros, which doesn't change the probe. " + EXPORT_HINT
        : `Reduce it to at most ${DIM} features (e.g. a smaller last layer) and zero-pad the rest.`;
      throw new Failed(`The embedding has ${features} features, but it must have exactly ${DIM}. ${fix}`);
    }
  }
  return `input image ${fmtDims(inp.dims)} -> output ${out.name} ${fmtDims(out.dims)}`;
}

// --------------------------------------------------------------------------------------------- 4. onnxruntime

// onnxruntime-web wraps errors in its C API call and the C++ source location; keep only the message itself.
const ortMessage = e => String(e?.message || e).trim()
  .replace(/^(?:Can't create a session\.|failed to call OrtRun\(\)\.)\s*ERROR_CODE: \d+, ERROR_MESSAGE: /, "")
  .replace(/^\S+\.(?:h|cc|cpp):\d+ [^(]*\([^)]*\)(?: const)? /, "");

// A one-node Identity model, loaded first so a runtime that can't start is never blamed on the participant's model.
const TINY_MODEL = "CAgSADo3ChAKAXgSAXkiCElkZW50aXR5EgF0Wg8KAXgSCgoICAESBAoCCAFiDwoBeRIKCggIARIECgIIAUIECgAQEQ==";

let runtime = null;
function startRuntime() {
  runtime ??= (async () => {
    importScripts(ORT_URL + "ort.wasm.min.js");
    ort.env.wasm.wasmPaths = ORT_URL;
    ort.env.wasm.numThreads = 1;  // threads need cross-origin isolation, which the submission page doesn't have
    ort.env.logLevel = "error";
    const tiny = Uint8Array.from(atob(TINY_MODEL), c => c.charCodeAt(0));
    await (await ort.InferenceSession.create(tiny)).release();
  })();
  return runtime;
}

async function checkLoad(ctx) {
  try {
    await startRuntime();
  } catch (e) {
    throw new Unavailable(`onnxruntime-web could not start in this browser (${e?.message || e}).`);
  }
  let sess;
  try {
    sess = await ort.InferenceSession.create(ctx.bytes, {executionProviders: ["wasm"]});
  } catch (e) {
    const msg = ortMessage(e), lower = msg.toLowerCase();
    let hint;
    if (msg.includes("IR version"))
      hint = `The model file format (ONNX IR version) is newer than onnxruntime ${ORT_VERSION} reads, usually ` +
        `because a newer onnx/torch wrote it. Lower it with:\n    import onnx; m = onnx.load('${ctx.file.name}'); ` +
        `m.ir_version = 10; onnx.save(m, '${ctx.file.name}')`;
    else if (lower.includes("opset") && (lower.includes("support") || lower.includes("version")))
      hint = `The model uses ONNX opset ${ctx.opset ?? "None"}, newer than onnxruntime ${ORT_VERSION} supports. ` +
        `Export with opset_version=17.`;
    else if (msg.includes("No Op registered") || lower.includes("not a registered") || lower.includes("custom") ||
             lower.includes("domain"))
      hint = "The model uses an operator that onnxruntime doesn't have (a custom op or a non-standard domain). " +
        "Evaluation uses plain onnxruntime on CPU, so the model must use standard ONNX ops; replace the custom " +
        "layer (e.g. a fused CUDA kernel) with plain PyTorch ops before export.";
    else hint = "Re-export the model, or simplify the layer named in the error above.";
    throw new Failed(`onnxruntime ${ORT_VERSION} could not load the model:\n    ${msg}\n${hint}`);
  }
  if (sess.inputNames.length !== 1 || sess.inputNames[0] !== "image")
    throw new Failed(`onnxruntime sees the inputs [${sess.inputNames.map(n => `'${n}'`).join(", ")}], but the ` +
      `model must have exactly one input, 'image'. ` + EXPORT_HINT);
  ctx.session = sess;
  return `onnxruntime-web ${ORT_VERSION}, in this browser`;
}

// --------------------------------------------------------------------------------------------- 5. inference

function explainSubmissionError(e) {
  const msg = e.message;
  if (msg.startsWith("Embedding shape")) return `${msg}. ` + EXPORT_HINT;
  if (msg.includes("NaN or infinite"))
    return `${msg}. Look for divisions by zero, log(0), normalising by a zero standard deviation, or float16 ` +
      `overflow. Also check that the inputs are what the model expects: RGB in [0, 1], then normalised with the ` +
      `ImageNet mean/std.`;
  if (msg.startsWith("Model failed"))
    return `${msg}\nThe model crashed while running. If it works for the batch size you exported with but not ` +
      `others, a shape was hard-coded during export (e.g. x.view(2, -1)); use x.flatten(1) or ` +
      `x.reshape(x.shape[0], -1) and export with a dynamic batch size.`;
  return msg;
}

function halfToFloat(h) {
  const sign = h & 0x8000 ? -1 : 1, exp = (h >> 10) & 0x1f, frac = h & 0x3ff;
  if (exp === 0) return sign * 2 ** -14 * (frac / 1024);
  if (exp === 31) return frac ? NaN : sign * Infinity;
  return sign * 2 ** (exp - 15) * (1 + frac / 1024);
}

function floats(tensor) {
  if (tensor.type === "float16" && tensor.data instanceof Uint16Array) return Float64Array.from(tensor.data, halfToFloat);
  return Float64Array.from(tensor.data, Number);
}

// Same as evaluate.embed: uint8 HWC images -> RGB in [0, 1] -> ImageNet mean/std -> float32 NCHW, in batches.
async function embed(sess, imgs, batch) {
  const pixels = SIZE * SIZE, out = [];
  for (let i = 0; i < imgs.length; i += batch) {
    const block = imgs.slice(i, i + batch), x = new Float32Array(block.length * 3 * pixels);
    block.forEach((im, n) => {
      for (let p = 0; p < pixels; p++)
        for (let c = 0; c < 3; c++) x[(n * 3 + c) * pixels + p] = (im[p * 3 + c] / 255 - MEAN[c]) / STD[c];
    });
    const shape = [block.length, 3, SIZE, SIZE];
    let z;
    try {
      const result = await sess.run({image: new ort.Tensor("float32", x, shape)});
      z = result[sess.outputNames[0]];
    } catch (e) {
      throw new SubmissionError(`Model failed on float32 input of shape ${pyShape(shape)}: ${ortMessage(e)}`);
    }
    if (z.dims.length !== 2 || z.dims[0] !== block.length || z.dims[1] !== DIM)
      throw new SubmissionError(`Embedding shape ${pyShape(z.dims)}, expected (batch, ${DIM})`);
    const values = floats(z);
    if (!values.every(Number.isFinite)) throw new SubmissionError("Embeddings contain NaN or infinite values");
    for (let n = 0; n < block.length; n++) out.push(values.subarray(n * DIM, (n + 1) * DIM));
  }
  return out;
}

// `n` 96x96 uint8 RGB images: the sample images first, then noise.
async function images(ctx, n) {
  let samples;
  try {
    const r = await fetch(new URL("submission_check_samples.bin", ctx.baseUrl));
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    samples = new Uint8Array(await r.arrayBuffer());
  } catch (e) {
    throw new Unavailable(`The sample images could not be downloaded (${e.message}).`);
  }
  const size = SIZE * SIZE * 3, out = [];
  for (let i = 0; i + size <= samples.length && out.length < n; i += size) out.push(samples.subarray(i, i + size));
  let seed = 0x9e3779b9;
  const random = () => { seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; return (seed >>> 0) & 255; };
  while (out.length < n) out.push(Uint8Array.from({length: size}, random));
  return out;
}

const allclose = (a, b, rtol, atol) => a.every((row, i) => row.every((v, j) => Math.abs(v - b[i][j]) <= atol + rtol * Math.abs(b[i][j])));

async function checkInference(ctx) {
  const sess = ctx.session, imgs = await images(ctx, 7);
  let z, single, again, seconds;
  try {
    const t0 = performance.now();
    z = await embed(sess, imgs, 4);  // batches of 4 and 3
    seconds = (performance.now() - t0) / 1000;
    single = await embed(sess, imgs.slice(0, 1), 1);
    again = await embed(sess, imgs, 4);
  } catch (e) {
    if (e instanceof SubmissionError) throw new Failed(explainSubmissionError(e));
    throw e;
  }
  let maxStd = 0;
  for (let j = 0; j < DIM; j++) {
    const mean = z.reduce((s, row) => s + row[j], 0) / z.length;
    maxStd = Math.max(maxStd, Math.sqrt(z.reduce((s, row) => s + (row[j] - mean) ** 2, 0) / z.length));
  }
  if (maxStd < 1e-6)
    warn(ctx, "Every image gets the same embedding, so the probe can only guess (chance accuracy). The encoder " +
      "may have collapsed during training, or the model ignores its input.");
  if (!allclose(z, again, 1e-3, 1e-5))
    warn(ctx, "The model gives different embeddings for the same images on different runs. It probably contains " +
      "randomness such as dropout; call model.eval() before exporting.");
  else if (!allclose(z.slice(0, 1), single, 1e-3, 1e-4))
    warn(ctx, "An image's embedding depends on the other images in its batch. This usually means BatchNorm was " +
      "exported in training mode; call model.eval() before exporting.");
  return `embeddings (${DIM},) for batch sizes 1, 3 and 4; ${(1000 * seconds / imgs.length).toFixed(1)} ms per image`;
}

// --------------------------------------------------------------------------------------------- main

const STEPS = [
  ["File is an ONNX model", checkFile],
  ["ONNX graph is valid", checkGraph],
  ["Input and output format", checkSignature],
  ["onnxruntime loads the model", checkLoad],
  ["Model runs on sample images", checkInference],
];

async function run({file, max_bytes, base_url}) {
  post({type: "steps", steps: STEPS.map(([title]) => title), command: COMMAND});
  startRuntime().catch(() => {});  // download onnxruntime while the first checks run; failures surface in step 4
  const ctx = {file, maxBytes: max_bytes, baseUrl: base_url, warnings: [], step: 0};
  for (const [index, [, check]] of STEPS.entries()) {
    ctx.step = index;
    post({type: "step", index, status: "running"});
    try {
      const note = await check(ctx);
      post({type: "step", index, status: "ok", note});
    } catch (e) {
      if (e instanceof Unavailable) { post({type: "unavailable", message: e.message}); return; }
      post({type: "step", index, status: "fail"});
      post({type: "done", ok: false, message: e instanceof Failed ? e.message
        : `Unexpected error: ${e?.message || e}\nThis most likely comes from the model. If you think it's a bug ` +
          `in the check, run ${COMMAND} and report its output to the organizers.`});
      return;
    }
  }
  post({type: "done", ok: true});
}

self.onmessage = event => { run(event.data); };
