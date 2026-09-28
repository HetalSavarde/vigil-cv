"""
VIGIL-CV :: F6 - Attack Simulator

Implements FR6.1-FR6.7 (scaled to prototype). All scenarios are seeded and
parameterized so every run is reproducible (FR6.6), and every simulated
attack is clearly labeled as a TEST SCENARIO (FR6.7) -- this module never
touches anything outside its own working copy of the demo dataset/model.
"""
import json
import os
import random
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.features import CLASSES, load_image_array  # noqa: E402
from models.train_model import export_onnx, train_softmax, weight_digest  # noqa: E402


def make_poisoned_copy(clean_dataset_dir: str, out_dir: str, seed: int = 7) -> dict:
    """
    FR6.1 label-flipping + FR6.2 near-duplicate flooding + FR6.3 trigger patch,
    all injected into ONE contributor's batch so F1's aggregation has a
    concentrated signal to surface (this is the realistic "one bad actor"
    scenario, not uniformly-random noise across everyone).
    """
    rng = random.Random(seed)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    shutil.copytree(clean_dataset_dir, out_dir)

    with open(os.path.join(out_dir, "labels.json")) as f:
        records = json.load(f)

    target_contributor = "contributor_B"
    target_ids = [r for r in records if r["contributor_id"] == target_contributor]

    log = {"scenario": "data_poisoning", "seed": seed, "actions": []}

    # --- Label flipping: flip 6 of this contributor's labels ---
    flip_targets = rng.sample(target_ids, k=min(6, len(target_ids)))
    for r in flip_targets:
        other_labels = [c for c in CLASSES if c != r["label"]]
        new_label = rng.choice(other_labels)
        log["actions"].append({
            "type": "label_flip", "sample_id": r["sample_id"],
            "from": r["label"], "to": new_label,
        })
        r["label"] = new_label

    # --- Near-duplicate flooding: clone 5 images with near-zero perturbation ---
    dup_sources = rng.sample(target_ids, k=min(5, len(target_ids)))
    img_dir = os.path.join(out_dir, "images")
    next_idx = len(records)
    for src in dup_sources:
        img = Image.open(os.path.join(img_dir, src["filename"])).convert("RGB")
        arr = np.asarray(img).astype(np.int16)
        arr += rng.randint(-2, 2)  # near-imperceptible perturbation
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        new_fname = f"flood_{next_idx:05d}.png"
        Image.fromarray(arr).save(os.path.join(img_dir, new_fname))
        new_record = {
            "sample_id": f"s{next_idx:05d}",
            "filename": new_fname,
            "label": src["label"],
            "contributor_id": target_contributor,
            "batch_id": src["batch_id"],
        }
        records.append(new_record)
        log["actions"].append({
            "type": "near_duplicate_flood", "sample_id": new_record["sample_id"],
            "source_sample_id": src["sample_id"],
        })
        next_idx += 1

    # --- Trigger patch: stamp a small fixed high-frequency checkerboard patch ---
    trigger_targets = rng.sample(target_ids, k=min(6, len(target_ids)))
    for r in trigger_targets:
        path = os.path.join(img_dir, r["filename"])
        img = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(img)
        px, py, size, cell = 1, 1, 8, 2
        for yy in range(0, size, cell):
            for xx in range(0, size, cell):
                color = (0, 0, 0) if (xx // cell + yy // cell) % 2 == 0 else (255, 255, 255)
                draw.rectangle([px + xx, py + yy, px + xx + cell - 1, py + yy + cell - 1], fill=color)
        img.save(path)
        log["actions"].append({"type": "trigger_patch", "sample_id": r["sample_id"]})

    with open(os.path.join(out_dir, "labels.json"), "w") as f:
        json.dump(records, f, indent=2)
    with open(os.path.join(out_dir, "ATTACK_LOG.json"), "w") as f:
        json.dump(log, f, indent=2)

    log["_meta_target_contributor"] = target_contributor
    return log


def make_substituted_model(clean_dataset_dir: str, reference_model_meta_path: str, out_path: str,
                            seed: int = 99) -> dict:
    """
    FR6.4: produce a modified/candidate model that behaves differently from
    the declared reference -- trained with a different seed / on the
    poisoned label set, so its decision boundary genuinely diverges rather
    than faking a "substitution" flag.
    """
    from models.train_model import load_training_data
    X, y, _ = load_training_data(clean_dataset_dir)
    # Different seed AND corrupted supervision (label noise) => a genuinely
    # different, worse-behaved model, not a cosmetic file swap.
    rng = np.random.RandomState(seed)
    noisy_y = y.copy()
    flip_mask = rng.rand(len(y)) < 0.4
    noisy_y[flip_mask] = rng.randint(0, 3, size=flip_mask.sum())

    W, b, acc = train_softmax(X, noisy_y, seed=seed, epochs=300)
    export_onnx(W, b, out_path)
    digest = weight_digest(W, b)
    meta = {"train_accuracy": acc, "weight_digest": digest, "seed": seed,
            "scenario": "model_substitution", "label_noise_fraction": 0.4}
    with open(out_path + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def tamper_sealed_record(record: dict, tamper_type: str = "output") -> dict:
    """
    FR6.5: tamper with a previously-sealed inference record. Returns a
    mutated copy (the caller keeps the untampered original for comparison).
    tamper_type: "output" | "input_hash" | "config"
    """
    tampered = json.loads(json.dumps(record))  # deep copy
    if tamper_type == "output":
        current = tampered["output"].get("predicted_label", CLASSES[0])
        others = [c for c in CLASSES if c != current]
        tampered["output"]["predicted_label"] = random.choice(others)
        tampered["output"]["confidence"] = round(random.uniform(0.8, 0.99), 3)
    elif tamper_type == "input_hash":
        tampered["input_hash"] = "0" * 64
    elif tamper_type == "config":
        tampered["preprocessing_config"]["resize"] = 999
    tampered["_tamper_type"] = tamper_type  # demo-only marker, not part of real seal
    return tampered


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log = make_poisoned_copy(
        os.path.join(base, "data", "clean_dataset"),
        os.path.join(base, "data", "poisoned_dataset"),
    )
    print(json.dumps(log, indent=2))
