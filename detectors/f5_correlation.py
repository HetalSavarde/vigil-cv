"""
VIGIL-CV :: F5 - Cross-Module Correlation

Implements FR5.1-FR5.4: when independent modules (F1 data integrity,
F2 model integrity, F3 provenance) flag findings that trace back to the same
contributor/model/asset, produce one transparent composite finding whose
confidence/severity is explainable from its component signals -- not a new
opaque score.
"""


def correlate(f1_result: dict, f2_finding: dict, f3_findings: list) -> list:
    composites = []

    # Correlate: a high-risk contributor (F1) AND a substituted model (F2)
    # in the same run is a materially stronger signal than either alone.
    high_risk_contributors = [
        c for c, stats in f1_result.get("contributor_risk", {}).items()
        if stats["risk_level"] in ("medium", "high")
    ]
    if high_risk_contributors and f2_finding.get("is_substituted"):
        composites.append({
            "flag_id": "F5-COMPOSITE-DATA-MODEL",
            "module": "correlated",
            "affected_asset": ",".join(high_risk_contributors) + f",{f2_finding['affected_asset']}",
            "reason": (
                "Elevated contributor-level data-integrity risk was observed in the same "
                "run as a detected model substitution/behavioral deviation. Independently, "
                "either signal alone is only suggestive; together they materially raise "
                "confidence that this run reflects a coordinated integrity compromise "
                "rather than isolated noise."
            ),
            "evidence": {
                "contributing_signals": [
                    {"module": "data_integrity", "contributors": high_risk_contributors},
                    {"module": "model_integrity", "flag_id": f2_finding["flag_id"],
                     "cosine_distance": f2_finding["evidence"]["fingerprint_cosine_distance"]},
                ],
            },
            "confidence": round(min(0.97, 0.5 + f2_finding["confidence"] / 2), 3),
            "severity": "high",
            "recommended_disposition": "quarantine",
            "access_level_used": "black_box",
            "limitations": "Correlation strengthens confidence but does not itself prove "
                           "causal linkage between the flagged data and the flagged model.",
        })

    # Correlate: any failed F3 verification alongside a substituted model.
    failed_provenance = [f for f in f3_findings if f["severity"] == "high"]
    if failed_provenance and f2_finding.get("is_substituted"):
        composites.append({
            "flag_id": "F5-COMPOSITE-MODEL-INFERENCE",
            "module": "correlated",
            "affected_asset": f2_finding["affected_asset"] + "," + ",".join(
                f["affected_asset"] for f in failed_provenance
            ),
            "reason": (
                "A substituted/deviating model was detected in the same run as one or more "
                "inference records that failed cryptographic verification. This combination "
                "is consistent with an actor both replacing the model and altering its "
                "outputs after the fact."
            ),
            "evidence": {
                "contributing_signals": [
                    {"module": "model_integrity", "flag_id": f2_finding["flag_id"]},
                    {"module": "inference_provenance",
                     "failed_records": [f["affected_asset"] for f in failed_provenance]},
                ],
            },
            "confidence": 0.95,
            "severity": "high",
            "recommended_disposition": "quarantine",
            "access_level_used": "not_applicable",
            "limitations": "Correlation is based on co-occurrence within the same assurance "
                           "run, not a proven attacker link between the two events.",
        })

    return composites
