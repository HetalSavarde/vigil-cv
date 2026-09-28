"""
VIGIL-CV :: demo model trainer

Trains a tiny softmax (linear) classifier with plain numpy gradient descent
-- no PyTorch/TensorFlow dependency needed for a 32x32 3-class toy problem --
then hand-builds a real ONNX graph (onnx.helper) so the exported artifact is a
genuine .onnx file VIGIL-CV's F2 module loads and runs with onnxruntime,
exactly like a real contributed model.
"""
import hashlib
import json
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.features import CLASSES, flatten_features, load_image_array  # noqa: E402


def load_training_data(dataset_dir):
    with open(os.path.join(dataset_dir, "labels.json")) as f:
        records = json.load(f)
    X, y = [], []
    for r in records:
        img = load_image_array(os.path.join(dataset_dir, "images", r["filename"]))
        X.append(flatten_features(img))
        y.append(CLASSES.index(r["label"]))
    return np.stack(X), np.array(y), records


def train_softmax(X, y, n_classes=3, lr=0.5, epochs=300, seed=0, l2=1e-3):
    rng = np.random.RandomState(seed)
    n_features = X.shape[1]
    W = rng.randn(n_features, n_classes).astype(np.float32) * 0.01
    b = np.zeros(n_classes, dtype=np.float32)
    Y_onehot = np.eye(n_classes, dtype=np.float32)[y]
    n = X.shape[0]

    for epoch in range(epochs):
        logits = X @ W + b
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        grad_logits = (probs - Y_onehot) / n
        grad_W = X.T @ grad_logits + l2 * W
        grad_b = grad_logits.sum(axis=0)
        W -= lr * grad_W
        b -= lr * grad_b

    preds = np.argmax(X @ W + b, axis=1)
    acc = float((preds == y).mean())
    return W, b, acc


def export_onnx(W: np.ndarray, b: np.ndarray, out_path: str, model_name="vigil_demo_classifier"):
    """Hand-build a minimal ONNX graph: logits = X @ W + b ; probs = softmax(logits)."""
    n_features, n_classes = W.shape
    X_in = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, n_features])
    Y_out = helper.make_tensor_value_info("probs", TensorProto.FLOAT, [None, n_classes])

    W_init = helper.make_tensor("W", TensorProto.FLOAT, W.shape, W.flatten().tolist())
    b_init = helper.make_tensor("b", TensorProto.FLOAT, b.shape, b.flatten().tolist())

    matmul_node = helper.make_node("MatMul", ["input", "W"], ["matmul_out"])
    add_node = helper.make_node("Add", ["matmul_out", "b"], ["logits"])
    softmax_node = helper.make_node("Softmax", ["logits"], ["probs"], axis=1)

    graph = helper.make_graph(
        [matmul_node, add_node, softmax_node],
        model_name,
        [X_in],
        [Y_out],
        initializer=[W_init, b_init],
    )
    model = helper.make_model(graph, producer_name="vigil-cv-demo")
    model.opset_import[0].version = 13
    model.ir_version = 9  # keep compatible with widely-deployed onnxruntime versions
    onnx.checker.check_model(model)
    onnx.save(model, out_path)
    return out_path


def weight_digest(W: np.ndarray, b: np.ndarray) -> str:
    """Stable 'model identity' hash used by F2/F3 as the weight digest."""
    h = hashlib.sha256()
    h.update(W.tobytes())
    h.update(b.tobytes())
    return h.hexdigest()


def train_and_export(dataset_dir, out_path, seed=0, epochs=300):
    X, y, _ = load_training_data(dataset_dir)
    W, b, acc = train_softmax(X, y, seed=seed, epochs=epochs)
    export_onnx(W, b, out_path)
    digest = weight_digest(W, b)
    meta = {"train_accuracy": acc, "weight_digest": digest, "seed": seed, "epochs": epochs}
    with open(out_path + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Trained model -> {out_path} | train_acc={acc:.3f} | digest={digest[:16]}...")
    return out_path, meta


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base, "..", "data", "clean_dataset")
    out_path = os.path.join(base, "reference_model.onnx")
    train_and_export(dataset_dir, out_path, seed=0, epochs=300)
