# VIGIL-CV

### Vision Integrity & Governance Intelligence Layer for Computer Vision

**A Trust Layer for Computer Vision Pipelines. Can we trust the data, the model, and the predictions?**

> **Status:** in development. `main` currently holds a **demo prototype** (see [Demo prototype](#demo-prototype)). The full tool is being built on feature branches. The complete specification is in [`docs/`](docs/).

**Demo video:** `[add link here]`

---

## What is VIGIL-CV?

VIGIL-CV is an **offline, model-agnostic assurance layer** that sits alongside an existing computer-vision pipeline and independently checks whether it can be trusted. It does not replace or modify the pipeline. It takes a dataset, a model, and inference records, and returns evidence-backed findings, a report, and a tamper-evident audit trail.

In multi-contributor pipelines, many parties supply data and models, so no single source can be assumed honest. VIGIL-CV checks the three places where a pipeline can be quietly compromised:

1. **Training data**: poisoned, mislabeled, duplicated, out-of-distribution or backdoor-triggered samples
2. **Trained models**: substituted or behaviorally altered models
3. **Inference outputs**: results edited, swapped or replayed after the fact

```
Training Dataset ─────┐
                      │
Candidate Model ──────┼──> VIGIL-CV Assurance Engine ──> Evidence + Report
                      │                                  + Audit Log
Inference Records ────┘                                  + Dashboard
```

---

## What it does

| ID | Module | What it does |
|---|---|---|
| **F1** | Training-Data Integrity | Detects near-duplicate flooding, label flipping, out-of-distribution samples and trigger-like patterns, and rolls sample-level evidence up to contributor, source and batch risk |
| **F2** | Model Integrity | Compares a model against a declared reference using a fixed behavioral battery to catch substitution or anomalous behavior. Works black-box, with optional white-box statistics when internals are available |
| **F3** | Inference Provenance | Cryptographically binds each inference to its input, model identity, configuration, output, timestamp and sequence, and detects alteration, substitution or replay |
| **F4** | Distribution-Shift Detection | Detects deviation from a declared reference distribution, and separates probable operational drift (terrain, season, sensor, lighting) from suspicious manipulation where the evidence supports it |
| **F5** | Cross-Module Correlation | Combines independent signals that point at the same sample, contributor, model or record into transparent composite findings |
| **F6** | Attack Simulator | Reproducibly injects representative poisoning, backdoor, substitution and tampering scenarios, so detection can be tested and demonstrated |
| **F7** | Tamper-Evident Audit Log | Append-only, hash-chained log of system activity and findings, verifiable end to end |
| **F8** | Analyst Dashboard | Local dashboard showing every finding with its evidence, confidence, severity, limitations and recommended action |

---

## Evidence, not black-box scores

Every finding follows one schema: what was flagged, **why**, the **evidence**, a **confidence** value, a **severity** (low, medium, high), a recommended **disposition** (accept, review, quarantine), the **access level** used (white-box, black-box or not applicable), and its **limitations**. Findings describe observed anomalies; they never claim malicious intent, and the final decision stays with a human analyst.

Every report also states its **coverage**: which attack classes were tested, which were not, and which checks were unavailable. Nothing is reported that wasn't measured.

### Design principles

1. Evidence before conclusion
2. No fabricated certainty
3. Reference-based assurance
4. Cryptographic verifiability
5. Reproducibility
6. Human oversight
7. Explicit coverage

---

## Environment and stack

- **Fully offline**: no cloud services or external APIs
- Windows or Linux, CPU-based (GPU optional), Python 3
- Inputs: image datasets (simplified COCO/YOLO), ONNX and PyTorch/TorchScript models
- Stack: Python, ONNX Runtime, PyTorch, scikit-learn, NumPy, OpenCV, Pillow, `cryptography` (HMAC / Ed25519), JSON-lines audit storage, Streamlit

## Scope and limitations

VIGIL-CV is designed to make a pipeline's integrity claims **measurable, explainable and cryptographically verifiable**. It does not claim that a pipeline is universally secure. It does not guarantee detection of every attack class, perform neuron-level forensic reconstruction, or require retraining of the audited model. It never presents white-box findings when only black-box access exists.

---

## Demo prototype

The demo on `main` is a simplified working slice of the tool. A built-in attack simulator plays the attacker, and the detectors are never told what was done.

| Act | Secret attack | Caught by |
|---|---|---|
| 1 | Poison one contributor's training images | F1 |
| 2 | Swap in a behaviorally different model | F2 |
| 3 | Edit a stored inference result | F3 |
| 4 | (none) Correlate findings, verify the audit log, produce the report | F5, F7 |

**Simplifications in the demo:** synthetic shape images instead of real photos, a small NumPy-trained classifier exported to a real `.onnx` file, HMAC signing with a **hardcoded demo key** (not for real use), thresholds tuned to the demo data, and no F4. Detection is representative, not exhaustive.

### Run it

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

```bash
# Windows                              # macOS / Linux
py demo\run_demo.py --pause            python3 demo/run_demo.py --pause
py -m streamlit run dashboard\app.py   python3 -m streamlit run dashboard/app.py
```

`--pause` waits for Enter between acts. Streamlit may ask for an email on first launch; press Enter to skip. The audit log accumulates across runs, so to reset:

```bash
# Windows (cmd)                        # macOS / Linux
rmdir /s /q data\clean_dataset         rm -rf data/clean_dataset data/poisoned_dataset \
rmdir /s /q data\poisoned_dataset            output models/*.onnx*
rmdir /s /q output
del /q models\*.onnx*
```

### Demo layout

```
data/generate_dataset.py       synthetic dataset          detectors/f1_data_integrity.py
models/train_model.py          model + ONNX export        detectors/f2_model_integrity.py
simulator/attack_simulator.py  F6 attacks                 detectors/f3_provenance.py
audit/audit_log.py             F7 audit log               detectors/f5_correlation.py
report/report_generator.py     report                     dashboard/app.py   F8 dashboard
demo/run_demo.py               runs the whole demo
```

---

## Roadmap

- Real datasets (COCO/YOLO subsets) and real ONNX and PyTorch/TorchScript models
- F4 distribution-shift detection
- White-box model statistics where internals are accessible
- Ed25519 signing with per-installation key management
- A packaged, downloadable offline application

## Branches and tags

- `main`: demo prototype
- tag `v0.1-demo`: permanent marker of the demo
- `feature/*`: the full tool, merged into `main` as each part is ready