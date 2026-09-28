"""
VIGIL-CV :: F1 - Training-Data Integrity Check

Implements a scaled prototype of FR1.1-FR1.8:
- FR1.2 near-duplicate / duplicate flooding via perceptual hash (Hamming distance)
- FR1.3 label flipping via k-NN class-consistency in embedding space
- FR1.4 OOD detection via embedding-space distance from per-class centroids
- FR1.5 trigger-like pattern detection via localized high-variance patch scan
- FR1.6 contributor/batch-level risk aggregation
- FR1.7 every finding carries reason/evidence/confidence/severity/disposition
- FR1.8 findings are phrased as observed anomalies, not claims of intent
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.features import foreground_embedding, hamming, load_image_array, phash  # noqa: E402

PHASH_DUP_THRESHOLD = 1        # hamming distance <= this => near-duplicate
LABEL_FLIP_KNN = 5             # neighbors considered for class-consistency
LABEL_FLIP_MIN_DISAGREE = 0.6  # fraction of disagreeing neighbors to flag
OOD_ZSCORE_THRESHOLD = 3.0     # centroid distance z-score to flag as OOD
TRIGGER_VARIANCE_THRESHOLD = 3.2  # local-patch variance multiple over image avg


def _load_all(dataset_dir):
    with open(os.path.join(dataset_dir, "labels.json")) as f:
        records = json.load(f)
    imgs, embs, phashes = {}, {}, {}
    for r in records:
        sid = r["sample_id"]
        arr = load_image_array(os.path.join(dataset_dir, "images", r["filename"]))
        imgs[sid] = arr
        embs[sid] = foreground_embedding(arr)
        phashes[sid] = phash(arr)
    return records, imgs, embs, phashes


def _detect_near_duplicates(records, phashes):
    flags = []
    ids = [r["sample_id"] for r in records]
    seen_pairs = set()
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            d = hamming(phashes[a], phashes[b])
            if d <= PHASH_DUP_THRESHOLD:
                pair = tuple(sorted((a, b)))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                confidence = round(1.0 - d / (PHASH_DUP_THRESHOLD + 1), 3)
                flags.append({
                    "flag_id": f"F1-DUP-{a}-{b}",
                    "module": "data_integrity",
                    "affected_asset": f"{a},{b}",
                    "reason": "Near-duplicate images detected via perceptual-hash similarity, "
                              "consistent with duplicate flooding.",
                    "evidence": {"phash_hamming_distance": int(d), "threshold": PHASH_DUP_THRESHOLD},
                    "confidence": confidence,
                    "severity": "medium" if d <= PHASH_DUP_THRESHOLD // 2 else "low",
                    "recommended_disposition": "review",
                    "access_level_used": "not_applicable",
                    "limitations": "Perceptual hashing catches near-identical images; "
                                   "semantically-duplicate-but-visually-different content is not covered.",
                })
    return flags


def _detect_label_flips(records, embs):
    flags = []
    ids = [r["sample_id"] for r in records]
    labels = {r["sample_id"]: r["label"] for r in records}
    emb_matrix = np.stack([embs[i] for i in ids])
    for idx, sid in enumerate(ids):
        dists = np.linalg.norm(emb_matrix - emb_matrix[idx], axis=1)
        dists[idx] = np.inf
        nn_idx = np.argsort(dists)[:LABEL_FLIP_KNN]
        neighbor_labels = [labels[ids[k]] for k in nn_idx]
        disagree_frac = sum(1 for l in neighbor_labels if l != labels[sid]) / len(neighbor_labels)
        if disagree_frac >= LABEL_FLIP_MIN_DISAGREE:
            flags.append({
                "flag_id": f"F1-FLIP-{sid}",
                "module": "data_integrity",
                "affected_asset": sid,
                "reason": "Sample's declared label is inconsistent with the labels of its "
                          "nearest neighbors in embedding space, consistent with label flipping "
                          "or systematic mislabelling.",
                "evidence": {
                    "declared_label": labels[sid],
                    "neighbor_labels": neighbor_labels,
                    "disagreement_fraction": round(disagree_frac, 3),
                    "k": LABEL_FLIP_KNN,
                },
                "confidence": round(min(0.95, 0.5 + disagree_frac / 2), 3),
                "severity": "high" if disagree_frac >= 0.8 else "medium",
                "recommended_disposition": "quarantine" if disagree_frac >= 0.8 else "review",
                "access_level_used": "not_applicable",
                "limitations": "Consistency check assumes the reference embedding space is "
                               "itself trustworthy and that neighbors are drawn from clean data.",
            })
    return flags


def _detect_ood(records, embs):
    flags = []
    by_class = defaultdict(list)
    for r in records:
        by_class[r["label"]].append(r["sample_id"])
    for label, ids in by_class.items():
        mat = np.stack([embs[i] for i in ids])
        centroid = mat.mean(axis=0)
        dists = np.linalg.norm(mat - centroid, axis=1)
        mu, sigma = dists.mean(), dists.std() + 1e-6
        for sid, d in zip(ids, dists):
            z = (d - mu) / sigma
            if z >= OOD_ZSCORE_THRESHOLD:
                flags.append({
                    "flag_id": f"F1-OOD-{sid}",
                    "module": "data_integrity",
                    "affected_asset": sid,
                    "reason": "Sample lies far outside the embedding-space distribution of its "
                              "declared class relative to the reference distribution.",
                    "evidence": {"centroid_distance": round(float(d), 4), "z_score": round(float(z), 3)},
                    "confidence": round(min(0.95, 0.5 + (z - OOD_ZSCORE_THRESHOLD) / 10), 3),
                    "severity": "medium",
                    "recommended_disposition": "review",
                    "access_level_used": "not_applicable",
                    "limitations": "OOD scoring is relative to the declared reference class "
                                   "distribution; a genuinely rare-but-legitimate sample can "
                                   "also trigger this check.",
                })
    return flags


def _detect_triggers(records, imgs):
    """Scan for a small, unusually high-variance/high-frequency localized patch
    (a simple stand-in for a visual backdoor trigger patch)."""
    flags = []
    patch = 8
    for r in records:
        sid = r["sample_id"]
        gray = imgs[sid].mean(axis=2)
        h, w = gray.shape
        img_var = gray.var() + 1e-6
        max_local_var = 0.0
        max_loc = None
        for y in range(0, h - patch, 2):
            for x in range(0, w - patch, 2):
                block = gray[y:y + patch, x:x + patch]
                v = block.var()
                if v > max_local_var:
                    max_local_var = v
                    max_loc = (x, y)
        ratio = max_local_var / img_var
        if ratio >= TRIGGER_VARIANCE_THRESHOLD:
            flags.append({
                "flag_id": f"F1-TRIG-{sid}",
                "module": "data_integrity",
                "affected_asset": sid,
                "reason": "A small localized high-frequency patch was detected that is "
                          "disproportionate to overall image texture, consistent with a "
                          "feasible visual trigger/backdoor pattern.",
                "evidence": {
                    "local_variance_ratio": round(float(ratio), 3),
                    "threshold": TRIGGER_VARIANCE_THRESHOLD,
                    "patch_location_xy": max_loc,
                    "patch_size": patch,
                },
                "confidence": round(min(0.9, 0.5 + (ratio - TRIGGER_VARIANCE_THRESHOLD) / 10), 3),
                "severity": "high",
                "recommended_disposition": "quarantine",
                "access_level_used": "not_applicable",
                "limitations": "Heuristic patch-variance scan; does not confirm the pattern "
                               "actually causes targeted model behavior (see F2 trigger-probing).",
            })
    return flags


def aggregate_contributor_risk(records, flags):
    """FR1.6: roll sample-level flags up to contributor/batch level."""
    flagged_ids = set()
    for f in flags:
        for asset in str(f["affected_asset"]).split(","):
            flagged_ids.add(asset)

    by_contrib = defaultdict(lambda: {"total": 0, "flagged": 0, "batches": set()})
    for r in records:
        c = by_contrib[r["contributor_id"]]
        c["total"] += 1
        c["batches"].add(r["batch_id"])
        if r["sample_id"] in flagged_ids:
            c["flagged"] += 1

    risk = {}
    for contributor, stats in by_contrib.items():
        rate = stats["flagged"] / stats["total"] if stats["total"] else 0.0
        risk[contributor] = {
            "total_samples": stats["total"],
            "flagged_samples": stats["flagged"],
            "flag_rate": round(rate, 3),
            "batches": sorted(stats["batches"]),
            "risk_level": "high" if rate >= 0.3 else ("medium" if rate >= 0.1 else "low"),
        }
    return risk


def run_f1(dataset_dir: str) -> dict:
    records, imgs, embs, phashes = _load_all(dataset_dir)
    flags = []
    flags += _detect_near_duplicates(records, phashes)
    flags += _detect_label_flips(records, embs)
    flags += _detect_ood(records, embs)
    flags += _detect_triggers(records, imgs)
    contributor_risk = aggregate_contributor_risk(records, flags)
    return {
        "module": "data_integrity",
        "n_samples_scanned": len(records),
        "findings": flags,
        "contributor_risk": contributor_risk,
    }


if __name__ == "__main__":
    result = run_f1(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "clean_dataset"))
    print(f"Scanned {result['n_samples_scanned']} samples, {len(result['findings'])} findings")
    print(json.dumps(result["contributor_risk"], indent=2))
