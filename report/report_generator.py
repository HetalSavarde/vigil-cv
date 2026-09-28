"""
VIGIL-CV :: Assurance Report Generator (SRS Section 6 schema)
"""
import time
from collections import Counter


def build_report(f1_result, f2_finding, f3_findings, f5_composites, audit_verification) -> dict:
    all_findings = list(f1_result["findings"]) + [f2_finding] + list(f3_findings) + list(f5_composites)

    by_severity = Counter(f["severity"] for f in all_findings)
    by_module = Counter(f["module"] for f in all_findings)

    report = {
        "report_generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "coverage_statement": (
            "This run assessed training-data integrity (near-duplicate flooding, "
            "label flipping, out-of-distribution samples, feasible trigger patterns), "
            "model behavioral integrity against a declared reference fingerprint "
            "(black-box only), inference provenance sealing/verification, and "
            "cross-module correlation. Distribution-shift assessment was not run in "
            "this demo scenario."
        ),
        "supported_attack_classes": [
            "near_duplicate_flooding", "label_flipping", "out_of_distribution_samples",
            "feasible_trigger_patterns", "model_substitution", "behavioral_deviation",
            "inference_tampering", "inference_replay",
        ],
        "unsupported_or_untested_attack_classes": [
            "deep neuron-level backdoor reconstruction",
            "white-box parameter/activation analysis (no white-box access in this run)",
            "distribution shift from operational drift (module not run in this scenario)",
        ],
        "assumptions": [
            "A known-good reference dataset distribution and model fingerprint were available.",
            "Demo dataset/model are synthetic and team-generated.",
            "Attack detection is representative, not exhaustive (per SRS Section 10).",
        ],
        "summary_by_severity": dict(by_severity),
        "summary_by_module": dict(by_module),
        "n_samples_scanned": f1_result["n_samples_scanned"],
        "contributor_risk": f1_result["contributor_risk"],
        "findings": all_findings,
        "correlated_findings": f5_composites,
        "verification_results": {
            "audit_log": audit_verification,
            "inference_records": [
                {"record_id": f["affected_asset"], "valid": f["severity"] == "low"}
                for f in f3_findings
            ],
        },
    }
    return report
