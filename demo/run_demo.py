"""
VIGIL-CV :: DEMO ORCHESTRATOR

Runs the full "catch the bad guy" demo narrative end-to-end:

  ACT 0  Baseline   - clean dataset + clean model, shown as the "known good" control
  ACT 1  Poison     - inject label-flips / near-dup flooding / trigger patches -> F1 catches it
  ACT 2  Substitute - swap in a behaviorally-different model -> F2 catches it
  ACT 3  Tamper     - seal an inference, then edit it after the fact -> F3 catches it
  ACT 4  Report     - correlate everything (F5), verify the audit log (F7), emit the report

Run with:  python3 demo/run_demo.py
Outputs land in output/ : report.json, audit_log.jsonl, sealed_records.jsonl
These are what dashboard/app.py (F8) reads and displays.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audit.audit_log import AuditLog  # noqa: E402
from data.generate_dataset import generate  # noqa: E402
from detectors.f1_data_integrity import run_f1  # noqa: E402
from detectors.f2_model_integrity import assess_model, build_reference_fingerprint  # noqa: E402
from detectors.f3_provenance import build_f3_finding, seal_inference, verify_record  # noqa: E402
from detectors.f5_correlation import correlate  # noqa: E402
from models.train_model import train_and_export  # noqa: E402
from report.report_generator import build_report  # noqa: E402
from simulator.attack_simulator import (  # noqa: E402
    make_poisoned_copy,
    make_substituted_model,
    tamper_sealed_record,
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
MODELS_DIR = os.path.join(BASE, "models")
OUTPUT_DIR = os.path.join(BASE, "output")

CLEAN_DATASET = os.path.join(DATA_DIR, "clean_dataset")
POISONED_DATASET = os.path.join(DATA_DIR, "poisoned_dataset")
REFERENCE_MODEL = os.path.join(MODELS_DIR, "reference_model.onnx")
SUBSTITUTED_MODEL = os.path.join(MODELS_DIR, "substituted_model.onnx")


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def banner(title):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


PAUSE = False


def beat(msg="Press Enter to continue..."):
    if PAUSE:
        input(f"\n    >>> {msg}")
    else:
        time.sleep(0.4)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    audit = AuditLog(os.path.join(OUTPUT_DIR, "audit_log.jsonl"))
    sealed_records_path = os.path.join(OUTPUT_DIR, "sealed_records.jsonl")
    open(sealed_records_path, "w").close()  # fresh run

    # ---------------------------------------------------------------- ACT 0
    banner("ACT 0 -- SETUP: clean baseline dataset + trained reference model")
    if not os.path.exists(os.path.join(CLEAN_DATASET, "labels.json")):
        generate(CLEAN_DATASET, n_per_class=40)
    else:
        print("Clean dataset already present, reusing it.")
    if not os.path.exists(REFERENCE_MODEL):
        train_and_export(CLEAN_DATASET, REFERENCE_MODEL, seed=0, epochs=300)
    else:
        print("Reference model already trained, reusing it.")
    audit.append("setup", {"clean_dataset": CLEAN_DATASET, "reference_model": REFERENCE_MODEL})

    baseline_f1 = run_f1(CLEAN_DATASET)
    print(f"\n[F1 baseline] scanned {baseline_f1['n_samples_scanned']} clean samples -> "
          f"{len(baseline_f1['findings'])} findings (expected: small honest noise floor)")
    for contributor, stats in baseline_f1["contributor_risk"].items():
        print(f"    {contributor}: flag_rate={stats['flag_rate']:.3f}  risk={stats['risk_level']}")
    audit.append("f1_baseline_run", {
        "n_samples": baseline_f1["n_samples_scanned"],
        "n_findings": len(baseline_f1["findings"]),
    })

    reference_fp = build_reference_fingerprint(REFERENCE_MODEL)
    baseline_f2 = assess_model(REFERENCE_MODEL, reference_fp)
    print(f"\n[F2 baseline] self-check on reference model -> "
          f"substituted={baseline_f2['is_substituted']}  confidence={baseline_f2['confidence']}  "
          f"disposition={baseline_f2['recommended_disposition']}")
    audit.append("f2_baseline_run", {"is_substituted": baseline_f2["is_substituted"]})
    beat("Baseline established. Press Enter to start the attack...")

    # ---------------------------------------------------------------- ACT 1
    banner("ACT 1 -- POISON THE DATA (secretly)")
    print("Injecting: label flips, near-duplicate flooding, and trigger patches\n"
          "into contributor_B's batch, undeclared...")
    attack_log = make_poisoned_copy(CLEAN_DATASET, POISONED_DATASET, seed=7)
    print(f"  -> {len(attack_log['actions'])} covert actions taken against "
          f"{attack_log['_meta_target_contributor']} (see data/poisoned_dataset/ATTACK_LOG.json)")
    beat("Attack injected. Press Enter to run the F1 scan...")

    print("\nRunning F1 data-integrity check on the now-poisoned dataset (as if we didn't know)...")
    f1_result = run_f1(POISONED_DATASET)
    print(f"  -> {len(f1_result['findings'])} findings "
          f"(baseline was {len(baseline_f1['findings'])})")
    for contributor, stats in f1_result["contributor_risk"].items():
        marker = "  <-- FLAGGED" if stats["risk_level"] != "low" else ""
        print(f"    {contributor}: flag_rate={stats['flag_rate']:.3f}  risk={stats['risk_level']}{marker}")
    audit.append("f1_run", {
        "dataset": "poisoned_dataset",
        "n_findings": len(f1_result["findings"]),
        "high_risk_contributors": [c for c, s in f1_result["contributor_risk"].items()
                                    if s["risk_level"] == "high"],
    })
    beat("Act 1 caught. Press Enter to swap the model...")

    # ---------------------------------------------------------------- ACT 2
    banner("ACT 2 -- SWAP IN A TAMPERED MODEL (secretly)")
    print("Training a behaviorally-different candidate model (different seed + "
          "40% label-noise) and presenting it as the deployed model...")
    sub_meta = make_substituted_model(CLEAN_DATASET, REFERENCE_MODEL + ".meta.json",
                                       SUBSTITUTED_MODEL, seed=99)
    print(f"  -> substituted model trained (train_acc={sub_meta['train_accuracy']:.3f})")

    print("\nRunning F2 behavioral fingerprint check against the declared reference...")
    f2_finding = assess_model(SUBSTITUTED_MODEL, reference_fp)
    print(f"  -> is_substituted={f2_finding['is_substituted']}  "
          f"cosine_distance={f2_finding['evidence']['fingerprint_cosine_distance']}  "
          f"confidence={f2_finding['confidence']}  severity={f2_finding['severity']}  "
          f"disposition={f2_finding['recommended_disposition']}")
    audit.append("f2_run", {
        "model": "substituted_model.onnx",
        "is_substituted": f2_finding["is_substituted"],
        "confidence": f2_finding["confidence"],
    })
    beat("Act 2 caught. Press Enter to tamper an inference...")

    # ---------------------------------------------------------------- ACT 3
    banner("ACT 3 -- EDIT AN AI ANSWER AFTER THE FACT (secretly)")
    sample_img = sorted(glob.glob(os.path.join(CLEAN_DATASET, "images", "circle_*.png")))[0]
    print(f"Running one real inference on {os.path.basename(sample_img)} and sealing the result...")
    output = {"predicted_label": "circle", "confidence": 0.97}
    sealed = seal_inference(
        sample_img,
        model_weight_digest=reference_fp["weight_digest"],
        preprocessing_config={"resize": 32, "normalize": True},
        output=output,
        sequence_no=1,
    )
    with open(sealed_records_path, "a") as f:
        f.write(json.dumps(sealed) + "\n")

    v_before = verify_record(sealed, sample_img)
    print(f"  -> verify BEFORE tampering: valid={v_before['valid']}  "
          f"failed_components={v_before['failed_components']}")
    beat("Sealed and verified clean. Press Enter to tamper it...")

    print("\nNow silently editing the stored output field (circle -> square)...")
    tampered = tamper_sealed_record(sealed, tamper_type="output")
    with open(sealed_records_path, "a") as f:
        f.write(json.dumps(tampered) + "\n")

    v_after = verify_record(tampered, sample_img)
    print(f"  -> verify AFTER tampering: valid={v_after['valid']}  "
          f"failed_components={v_after['failed_components']}  <-- CAUGHT")

    f3_findings = [
        build_f3_finding(sealed, v_before),
        build_f3_finding(tampered, v_after),
    ]
    audit.append("f3_run", {
        "records_checked": 2,
        "valid_before": v_before["valid"],
        "valid_after_tamper": v_after["valid"],
        "failed_components": v_after["failed_components"],
    })
    beat("Act 3 caught. Press Enter to see the correlated report...")

    # ---------------------------------------------------------------- ACT 4
    banner("ACT 4 -- THE REPORT: correlate everything and show the evidence")
    f5_composites = correlate(f1_result, f2_finding, f3_findings)
    print(f"[F5] {len(f5_composites)} correlated composite finding(s) generated")
    for c in f5_composites:
        print(f"  -> {c['flag_id']}: {c['reason'][:90]}...")

    audit.append("f5_run", {"n_composites": len(f5_composites)})
    audit_verification = audit.verify_chain()
    print(f"\n[F7] audit log chain verification -> intact={audit_verification['chain_intact']} "
          f"({audit_verification['n_entries']} entries)")

    report = build_report(f1_result, f2_finding, f3_findings, f5_composites, audit_verification)
    report_path = os.path.join(OUTPUT_DIR, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=_json_default)

    print(f"\nFull assurance report written -> {report_path}")
    print(f"Severity summary: {report['summary_by_severity']}")
    print(f"\nOpen the dashboard to view everything visually:\n"
          f"  streamlit run dashboard/app.py")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VIGIL-CV demo orchestrator")
    parser.add_argument("--pause", action="store_true",
                         help="Wait for Enter between acts (use this while screen-recording).")
    args = parser.parse_args()
    PAUSE = args.pause
    main()
