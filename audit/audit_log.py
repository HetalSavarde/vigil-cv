"""
VIGIL-CV :: F7 - Tamper-Evident Audit Log

Implements FR7.1-FR7.5: an append-only, hash-chained JSON-lines log.
Each entry embeds the hash of the previous entry, so altering, inserting,
or removing any record breaks the chain from that point forward -- exactly
the same tamper-evidence pattern used for F3, applied to the system's own
activity log.
"""
import hashlib
import json
import os
import time

GENESIS_HASH = "0" * 64


def _entry_hash(entry_without_hash: dict) -> str:
    payload = json.dumps(entry_without_hash, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


class AuditLog:
    def __init__(self, path: str):
        self.path = path
        if not os.path.exists(path):
            open(path, "w").close()

    def _last_hash(self) -> str:
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            return GENESIS_HASH
        with open(self.path, "r") as f:
            lines = [l for l in f.readlines() if l.strip()]
        if not lines:
            return GENESIS_HASH
        return json.loads(lines[-1])["entry_hash"]

    def append(self, event_type: str, details: dict) -> dict:
        prev_hash = self._last_hash()
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event_type": event_type,
            "details": details,
            "prev_hash": prev_hash,
        }
        entry["entry_hash"] = _entry_hash(entry)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return entry

    def read_all(self):
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            return [json.loads(l) for l in f if l.strip()]

    def verify_chain(self) -> dict:
        entries = self.read_all()
        expected_prev = GENESIS_HASH
        breaks = []
        for i, entry in enumerate(entries):
            stored_hash = entry.get("entry_hash")
            recomputed = _entry_hash({k: v for k, v in entry.items() if k != "entry_hash"})
            if entry.get("prev_hash") != expected_prev:
                breaks.append({"index": i, "reason": "prev_hash mismatch (record inserted/removed/reordered)"})
            if stored_hash != recomputed:
                breaks.append({"index": i, "reason": "entry_hash mismatch (record content altered)"})
            expected_prev = stored_hash if stored_hash == recomputed else recomputed
        return {
            "n_entries": len(entries),
            "chain_intact": len(breaks) == 0,
            "breaks": breaks,
        }


if __name__ == "__main__":
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), "audit.jsonl")
    log = AuditLog(path)
    log.append("f1_run", {"n_samples": 120, "n_findings": 3})
    log.append("f2_run", {"model": "reference_model.onnx", "substituted": False})
    print("Verify (clean):", log.verify_chain())

    # tamper with line 2 directly on disk
    with open(path) as f:
        lines = f.readlines()
    entry = json.loads(lines[1])
    entry["details"]["substituted"] = True  # attacker edits history
    lines[1] = json.dumps(entry) + "\n"
    with open(path, "w") as f:
        f.writelines(lines)
    print("Verify (tampered):", log.verify_chain())
