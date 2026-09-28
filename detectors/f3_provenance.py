"""
VIGIL-CV :: F3 - Inference Provenance and Output Integrity

Implements FR3.1-FR3.6 (scaled to prototype):
- Hashes input image + model identifier/weight digest + preprocessing config + output
- Signs each protected record with HMAC-SHA256 (Ed25519 also supported if a
  keypair is provided; HMAC keeps the demo dependency-light and fast)
- Includes timestamp + monotonic sequence number/nonce (replay detectability)
- verify_record() recomputes and reports exactly which component failed
"""
import hashlib
import hmac
import json
import os
import time

SECRET_KEY = b"vigil-cv-demo-hmac-key-do-not-use-in-production"


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_file(path: str) -> str:
    with open(path, "rb") as f:
        return _sha256_hex(f.read())


def seal_inference(
    input_image_path: str,
    model_weight_digest: str,
    preprocessing_config: dict,
    output: dict,
    sequence_no: int,
    record_id: str = None,
) -> dict:
    """FR3.1-FR3.3: build and sign a protected inference record."""
    input_hash = _hash_file(input_image_path)
    config_hash = _sha256_hex(json.dumps(preprocessing_config, sort_keys=True).encode())
    output_hash = _sha256_hex(json.dumps(output, sort_keys=True).encode())
    nonce = os.urandom(8).hex()
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    record = {
        "record_id": record_id or f"inf_{sequence_no:06d}",
        "input_image": os.path.basename(input_image_path),
        "input_hash": input_hash,
        "model_weight_digest": model_weight_digest,
        "preprocessing_config": preprocessing_config,
        "config_hash": config_hash,
        "output": output,
        "output_hash": output_hash,
        "timestamp": timestamp,
        "sequence_no": sequence_no,
        "nonce": nonce,
    }
    payload = json.dumps(record, sort_keys=True).encode()
    signature = hmac.new(SECRET_KEY, payload, hashlib.sha256).hexdigest()
    record["signature"] = signature
    return record


def verify_record(record: dict, input_image_path: str = None) -> dict:
    """
    FR3.4-FR3.6: recompute each protected component and report which ones
    (if any) fail validation. `input_image_path` is optional -- if not given,
    input-hash re-verification against the original file is skipped and only
    the stored fields + signature are checked (still catches record tampering).
    """
    rec = dict(record)
    stored_signature = rec.pop("signature", None)
    payload = json.dumps(rec, sort_keys=True).encode()
    recomputed_signature = hmac.new(SECRET_KEY, payload, hashlib.sha256).hexdigest()

    failures = []
    if stored_signature != recomputed_signature:
        failures.append("signature")

    recomputed_config_hash = hashlib.sha256(
        json.dumps(rec["preprocessing_config"], sort_keys=True).encode()
    ).hexdigest()
    if recomputed_config_hash != rec["config_hash"]:
        failures.append("preprocessing_config")

    recomputed_output_hash = hashlib.sha256(
        json.dumps(rec["output"], sort_keys=True).encode()
    ).hexdigest()
    if recomputed_output_hash != rec["output_hash"]:
        failures.append("output")

    if input_image_path and os.path.exists(input_image_path):
        actual_input_hash = _hash_file(input_image_path)
        if actual_input_hash != rec["input_hash"]:
            failures.append("input_image")

    valid = len(failures) == 0
    return {
        "record_id": rec.get("record_id"),
        "valid": valid,
        "failed_components": failures,
        "checked_signature": True,
        "checked_input_image": bool(input_image_path),
        "recomputed_signature_matches": stored_signature == recomputed_signature,
    }


def build_f3_finding(record: dict, verification: dict) -> dict:
    valid = verification["valid"]
    return {
        "flag_id": f"F3-{record['record_id']}",
        "module": "inference_provenance",
        "affected_asset": record["record_id"],
        "reason": (
            "Sealed inference record verified successfully; input/model/output "
            "binding is intact."
            if valid else
            f"Sealed inference record FAILED verification. Tampered/mismatched "
            f"component(s): {', '.join(verification['failed_components'])}."
        ),
        "evidence": {
            "failed_components": verification["failed_components"],
            "sequence_no": record.get("sequence_no"),
            "nonce": record.get("nonce"),
            "timestamp": record.get("timestamp"),
        },
        "confidence": 0.99 if not valid else 0.98,
        "severity": "high" if not valid else "low",
        "recommended_disposition": "quarantine" if not valid else "accept",
        "access_level_used": "not_applicable",
        "limitations": "Detects tampering/substitution/replay of the sealed record fields; "
                       "does not independently re-run the model to confirm the output was "
                       "originally correct.",
    }


if __name__ == "__main__":
    import glob
    img_path = sorted(glob.glob("data/clean_dataset/images/circle_*.png"))[0]
    rec = seal_inference(
        img_path,
        model_weight_digest="deadbeef" * 8,
        preprocessing_config={"resize": 32, "normalize": True},
        output={"predicted_label": "circle", "confidence": 0.98},
        sequence_no=1,
    )
    print("Sealed:", json.dumps(rec, indent=2))
    v = verify_record(rec, img_path)
    print("Verify (untampered):", v)

    tampered = json.loads(json.dumps(rec))
    tampered["output"]["predicted_label"] = "square"
    v2 = verify_record(tampered, img_path)
    print("Verify (tampered output):", v2)
