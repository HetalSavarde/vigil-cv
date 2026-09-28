"""
VIGIL-CV :: F2 - Model Integrity Check

Implements FR2.1-FR2.8 (scaled to prototype):
- Loads an ONNX model and runs it against a FIXED reference probe battery
  (synthetic, seeded, reproducible -- not drawn from the training set).
- Generates a behavioral fingerprint from the reference outputs.
- Compares a candidate fingerprint to the declared reference fingerprint.
- Flags likely substitution/modification when deviation exceeds threshold.
- Reports access level honestly (black-box only in this prototype: FR2.6).
"""
import hashlib
import json
import os
import sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.features import IMG_SIZE  # noqa: E402

DEVIATION_THRESHOLD = 0.15  # cosine-distance threshold for "substituted" flag
N_PROBES = 24


def build_probe_battery(seed: int = 1337, n_probes: int = N_PROBES) -> np.ndarray:
    """
    A FIXED, reproducible synthetic input battery (FR2.1). Not drawn from any
    real dataset -- this is what makes the behavioral fingerprint a property
    of the *model*, not of whatever data happens to be lying around.
    """
    rng = np.random.RandomState(seed)
    return rng.rand(n_probes, IMG_SIZE * IMG_SIZE * 3).astype(np.float32)


def run_battery(model_path: str, battery: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    out = sess.run(None, {input_name: battery})[0]
    return out  # shape: (n_probes, n_classes)


def fingerprint(outputs: np.ndarray) -> np.ndarray:
    """Behavioral fingerprint = flattened, rounded probability vector across the battery."""
    return np.round(outputs, 4).reshape(-1)


def fingerprint_hash(fp: np.ndarray) -> str:
    return hashlib.sha256(fp.tobytes()).hexdigest()


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 1.0
    return float(1.0 - (a @ b) / denom)


def model_weight_digest(model_path: str) -> str:
    """Simple identity digest of the model file bytes (stand-in for a weight digest)."""
    with open(model_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_reference_fingerprint(model_path: str, seed: int = 1337) -> dict:
    battery = build_probe_battery(seed=seed)
    outputs = run_battery(model_path, battery)
    fp = fingerprint(outputs)
    return {
        "battery_seed": seed,
        "n_probes": battery.shape[0],
        "fingerprint_hash": fingerprint_hash(fp),
        "fingerprint_vector": fp.tolist(),
        "weight_digest": model_weight_digest(model_path),
    }


def assess_model(candidate_model_path: str, reference_fp: dict) -> dict:
    """
    Compares candidate model behavior against a declared reference fingerprint.
    Returns a VIGIL-CV finding dict (see report schema) plus raw evidence.
    """
    seed = reference_fp["battery_seed"]
    battery = build_probe_battery(seed=seed, n_probes=reference_fp["n_probes"])
    outputs = run_battery(candidate_model_path, battery)
    cand_fp = fingerprint(outputs)
    ref_fp = np.array(reference_fp["fingerprint_vector"], dtype=np.float32)

    distance = cosine_distance(cand_fp, ref_fp)
    exact_hash_match = fingerprint_hash(cand_fp) == reference_fp["fingerprint_hash"]
    cand_digest = model_weight_digest(candidate_model_path)
    identity_match = cand_digest == reference_fp["weight_digest"]

    substituted = distance > DEVIATION_THRESHOLD or not identity_match
    confidence = float(min(0.99, max(0.5, distance / (DEVIATION_THRESHOLD * 4) + 0.5))) if substituted else float(
        max(0.5, 1.0 - distance * 3)
    )
    severity = "high" if distance > DEVIATION_THRESHOLD * 2 or not identity_match else (
        "medium" if substituted else "low"
    )

    finding = {
        "flag_id": f"F2-{cand_digest[:10]}",
        "module": "model_integrity",
        "affected_asset": f"model:{cand_digest[:16]}",
        "reason": (
            "Candidate model's behavioral fingerprint deviates materially from the "
            "declared reference fingerprint on the fixed probe battery; weight "
            "identity digest does not match the declared reference model."
            if substituted else
            "Candidate model's behavioral fingerprint matches the declared reference "
            "within tolerance; weight identity digest matches."
        ),
        "evidence": {
            "fingerprint_cosine_distance": round(distance, 6),
            "deviation_threshold": DEVIATION_THRESHOLD,
            "exact_fingerprint_hash_match": exact_hash_match,
            "weight_identity_match": identity_match,
            "reference_weight_digest": reference_fp["weight_digest"][:16] + "...",
            "candidate_weight_digest": cand_digest[:16] + "...",
            "n_probes": int(battery.shape[0]),
        },
        "confidence": round(confidence, 3),
        "severity": severity,
        "recommended_disposition": "quarantine" if severity == "high" else ("review" if substituted else "accept"),
        "access_level_used": "black_box",
        "limitations": (
            "Black-box behavioral probing only; no white-box parameter/activation "
            "statistics were computed for this candidate (FR2.6). Detection is "
            "based on aggregate behavioral deviation, not neuron-level analysis."
        ),
        "is_substituted": substituted,
    }
    return finding


if __name__ == "__main__":
    ref = build_reference_fingerprint("models/reference_model.onnx")
    print(json.dumps(assess_model("models/reference_model.onnx", ref), indent=2))
