# Software Requirements Specification (SRS)

# VIGIL-CV

### Vision Integrity & Governance Intelligence Layer for Computer Vision

**A Trust Layer for Computer Vision Pipelines**
*Can we trust the data, model, and predictions?*

---

**Problem Statement ID:** SIH26228
**Problem Statement Title:** Trustworthy Computer Vision Integrity Assurance for Data, Models and Inference Outputs in Multi-Contributor Pipelines
**Theme:** Blockchain & Cybersecurity
**Category:** Software
**Document Version:** 2.0 (Final)
**Prepared for:** Smart India Hackathon

---

## 1. Introduction

### 1.1 Purpose

This document specifies the requirements for **VIGIL-CV** — a model-agnostic, offline software framework that independently evaluates the trustworthiness of a computer-vision pipeline across three major lifecycle stages:

1. **Training data**
2. **Trained models**
3. **Inference outputs**

VIGIL-CV is designed as an independent assurance layer rather than a replacement for the existing CV pipeline. It produces evidence-based findings that can identify or indicate data poisoning, label flipping, trigger/backdoor patterns, duplicate flooding, out-of-distribution content, model substitution or behavioral deviation, inference tampering/replay, and distribution shift.

VIGIL-CV does not assume that every contributing source is trusted. Each finding must contain supporting evidence, confidence/severity information, access assumptions, limitations, and a recommended human disposition.

### 1.2 Scope

VIGIL-CV shall:

- Ingest a dataset containing images and labels/metadata and identify suspicious samples and contributor/batch-level risk.
- Detect near-duplicate flooding, label flipping/systematic mislabelling, out-of-distribution samples, and feasible trigger-like patterns.
- Ingest a trained model and assess anomalous or substituted behavior using a fixed behavioral reference battery.
- Support black-box assessment and, when model internals are available, optional white-box parameter/activation statistics.
- Cryptographically bind every protected inference to its input, model identity/weight digest, preprocessing/inference configuration, output, timestamp, and sequence/nonce information.
- Detect post-hoc alteration, substitution, or replay of sealed inference records.
- Detect distribution shift between a declared reference distribution and incoming data, including operational changes such as terrain, season, sensor, illumination, and acquisition conditions.
- Use available evidence to distinguish probable operational drift from suspicious manipulation where such a distinction is supportable.
- Correlate independent findings across modules when they point to the same sample, contributor, batch, model, or inference record.
- Maintain a tamper-evident, hash-chained audit log of system activity and findings.
- Provide an analyst-facing local dashboard with evidence, confidence/severity, limitations, and recommended disposition.
- Provide a reproducible attack simulator for representative poisoning, backdoor, model substitution, and inference tampering scenarios.
- Operate fully offline with no dependency on cloud services or external APIs.

VIGIL-CV will **not**:

- Guarantee detection of every possible attack class.
- Perform full neuron-level or deep white-box forensic reconstruction.
- Require retraining of the contributed model for baseline integrity assessment.
- Claim white-box findings when only black-box access is available.
- Provide production-scale multi-user access control or long-term distributed storage in the prototype.

### 1.3 Intended Audience

- Development team
- Hackathon judges/evaluators
- System operators
- Security/CV analysts
- Future maintainers and extenders

### 1.4 Definitions and Abbreviations

| Term | Meaning |
|---|---|
| CV | Computer Vision |
| VIGIL-CV | Vision Integrity & Governance Intelligence Layer for Computer Vision (this system) |
| OOD | Out-of-Distribution data that differs materially from the expected distribution |
| Label flipping | Deliberate modification of a sample's ground-truth label |
| Backdoor / Trigger | A hidden pattern intended to cause targeted model behavior when present |
| White-box access | Access to model weights, architecture, parameters, or activations |
| Black-box access | Access limited primarily to model inputs and outputs |
| Provenance | Verifiable record of origin and processing history |
| Behavioral fingerprint | Reference signature derived from model outputs on a fixed input battery |
| Distribution shift | A material change between reference and incoming data distributions |
| Hash chain | A sequence of records where each entry cryptographically references the previous entry |
| Disposition | Recommended action: accept, review, or quarantine |

### 1.5 Product Identity

| Element | Description |
|---|---|
| **Name** | VIGIL-CV |
| **Expansion** | Vision Integrity & Governance Intelligence Layer for Computer Vision |
| **Tagline** | A Trust Layer for Computer Vision Pipelines |
| **Core Question** | Can we trust the data, model, and predictions? |
| **Positioning** | An independent, offline assurance layer — not a CV pipeline replacement, a verifier that sits alongside it |

---

## 2. Overall Description

### 2.1 Product Perspective

VIGIL-CV is a **standalone, offline assurance tool** that sits alongside an existing computer-vision pipeline. It takes a dataset, model, and optionally inference records as inputs and produces an assurance report and audit trail as outputs. It does not modify the audited CV pipeline.

Conceptually:

```text
Training Dataset ─────┐
                       │
Candidate Model ───────┼──> VIGIL-CV Assurance Engine ──> Evidence + Report
                       │                                    + Audit Log
Inference Records ─────┘                                    + Dashboard
```

### 2.2 Product Functions

| ID | Function | Description |
|---|---|---|
| F1 | Training-Data Integrity | Detect suspicious samples and aggregate risk to contributor/batch/source level |
| F2 | Model Integrity | Detect anomalous, substituted, or backdoor-like model behavior |
| F3 | Inference Provenance | Cryptographically seal and verify inference records |
| F4 | Distribution-Shift Detection | Detect and characterize deviation from a reference distribution |
| F5 | Cross-Module Correlation | Combine related signals into transparent composite findings |
| F6 | Attack Simulator | Reproducibly inject representative attack scenarios |
| F7 | Tamper-Evident Audit Log | Maintain and verify a hash-chained append-only log |
| F8 | Analyst Dashboard | Present findings, evidence, confidence/severity, limitations, and disposition |

### 2.3 User Classes

| User | Description |
|---|---|
| Analyst / Reviewer | Inspects evidence and decides the final disposition |
| System Operator | Runs assurance checks and attack simulations |
| Evaluator | Reviews architecture, evidence, attack coverage, and live demonstration |

### 2.4 Operating Environment

- Fully offline / air-gapped
- Windows/Linux standard machine
- Python 3.x
- CPU-based prototype; GPU optional
- No external authentication or hosted service dependency

### 2.5 Design Constraints

- Model-agnostic design
- No mandatory retraining for baseline assessment
- Common CV formats such as simplified COCO/YOLO input
- ONNX and PyTorch/TorchScript support as available
- Graceful black-box fallback
- Local cryptographic operations
- Transparent evidence instead of unexplained risk scores

### 2.6 Assumptions

- A known-good reference dataset and/or model fingerprint is available.
- Demonstration datasets/models are public or team-generated.
- Prototype scale is small-to-moderate.
- Attack detection is representative rather than exhaustive.

---

## 3. System Features and Functional Requirements

### 3.1 F1 — Training-Data Integrity Check

**Purpose:** Identify suspicious or anomalous training samples and aggregate evidence into contributor/batch/source-level risk.

**Requirements:**
- **FR1.1:** Accept an image folder with labels/metadata and optional contributor, source, or batch ID.
- **FR1.2:** Detect near-duplicate or duplicate flooding using perceptual hashing and/or embedding similarity.
- **FR1.3:** Detect label flipping and systematic mislabelling using class-consistency and embedding-based checks.
- **FR1.4:** Detect OOD samples using embedding-space distance from a reference distribution.
- **FR1.5:** Detect feasible trigger-like visual patterns, such as localized high-frequency or repeated patches.
- **FR1.6:** Aggregate sample-level evidence into contributor/source/batch risk.
- **FR1.7:** Each finding shall include reason, affected sample/source, evidence metric, confidence, severity, and disposition.
- **FR1.8:** The system shall distinguish an observed anomaly from a claim of malicious intent.

### 3.2 F2 — Model Integrity Check

**Purpose:** Assess whether a supplied model behaves consistently with a declared reference model/fingerprint.

**Requirements:**
- **FR2.1:** Load ONNX models and run them against a fixed reference battery.
- **FR2.2:** Generate a behavioral fingerprint from reference outputs.
- **FR2.3:** Compare a candidate model fingerprint against the expected reference fingerprint.
- **FR2.4:** Detect likely model substitution or modification through behavioral deviation.
- **FR2.5:** Where white-box access is available, calculate basic parameter/activation statistics.
- **FR2.6:** Where only black-box access is available, clearly report that white-box checks were unavailable/skipped.
- **FR2.7:** Where feasible, perform trigger-oriented behavioral probing against suspicious candidate models.
- **FR2.8:** Every model finding shall state access level, confidence, evidence, and limitations.

### 3.3 F3 — Inference Provenance and Output Integrity

**Purpose:** Create a verifiable cryptographic binding between an inference and the exact inputs/configuration/model/output that produced it.

**Requirements:**
- **FR3.1:** Hash the input image, model identifier/weight digest, preprocessing/inference configuration, and output.
- **FR3.2:** Sign each protected record using HMAC or Ed25519.
- **FR3.3:** Include timestamp and monotonic sequence number/nonce to make replay detectable.
- **FR3.4:** Provide a verification function for stored records.
- **FR3.5:** Altering any protected component shall cause verification failure.
- **FR3.6:** The verification result shall identify which component(s) failed validation when feasible.

### 3.4 F4 — Distribution-Shift and Anomaly Assessment

**Purpose:** Detect material deviation from a declared reference distribution and characterize likely causes.

**Requirements:**
- **FR4.1:** Compute an embedding-space or statistical distribution distance.
- **FR4.2:** Produce a calibrated risk/confidence indication for the observed shift.
- **FR4.3:** Consider operational changes including terrain, season, sensor, illumination, weather, and acquisition conditions.
- **FR4.4:** Use available heuristics to distinguish gradual/natural drift from abrupt or suspicious-looking changes.
- **FR4.5:** Explicitly state when available evidence is insufficient to distinguish drift from manipulation.

### 3.5 F5 — Cross-Module Correlation

**Purpose:** Combine independent signals that point to the same asset or event.

**Requirements:**
- **FR5.1:** Correlate findings by sample, contributor, batch, model, or inference record.
- **FR5.2:** When independent modules flag the same asset, produce a transparent composite finding.
- **FR5.3:** Composite confidence/severity must be explainable and traceable to the underlying findings.
- **FR5.4:** The dashboard shall show the individual signals that contributed to a composite finding.

### 3.6 F6 — Attack Simulator

**Purpose:** Provide reproducible demonstrations of representative attack classes.

**Requirements:**
- **FR6.1:** Inject label-flipped samples into a clean dataset.
- **FR6.2:** Inject near-duplicate samples.
- **FR6.3:** Embed a simple visual trigger into a subset of images and, where feasible, produce a model exhibiting corresponding behavior.
- **FR6.4:** Substitute a clean reference model with a modified/candidate model.
- **FR6.5:** Tamper with a previously sealed inference record.
- **FR6.6:** Store scenario parameters so every demonstration can be reproduced.
- **FR6.7:** Clearly label simulated attacks as test scenarios rather than real-world incidents.

### 3.7 F7 — Tamper-Evident Audit Log

**Requirements:**
- **FR7.1:** Each audit entry shall contain the previous entry hash.
- **FR7.2:** Provide whole-chain verification.
- **FR7.3:** Record findings, dispositions, inference-sealing events, and verification events.
- **FR7.4:** Store the audit trail locally in human-inspectable JSON-lines format.
- **FR7.5:** Verification shall detect altered, inserted, or removed records within the protected chain.

### 3.8 F8 — Analyst Dashboard

**Requirements:**
- **FR8.1:** Show affected asset, reason, evidence, confidence/severity, access level, and disposition.
- **FR8.2:** Show a dynamically generated coverage statement for the current run.
- **FR8.3:** Provide audit-log and inference-record verification views.
- **FR8.4:** Visually distinguish single-module findings from correlated findings.
- **FR8.5:** Run entirely locally.
- **FR8.6:** Make clear that recommendations support human review and do not constitute an automatic forensic conclusion.

---

## 4. External Interface Requirements

### 4.1 User Interface

A local web dashboard, preferably implemented with Streamlit, accessible through `localhost`, branded as **VIGIL-CV**.

### 4.2 Software Interfaces

**Dataset Input:**
- Folder-based images
- Labels/metadata in simplified JSON/CSV or supported COCO/YOLO subset

**Model Input:**
- ONNX (`.onnx`)
- PyTorch/TorchScript (`.pt`) where supported

**Outputs:**
- Structured assurance report in JSON
- Hash-chained audit log in JSON-lines
- Human-readable dashboard findings

### 4.3 Hardware Interface

Standard CPU execution for prototype scale. GPU optional.

### 4.4 Communication Interface

No network communication is required during assurance execution.

---

## 5. Non-Functional Requirements

| Category | Requirement |
|---|---|
| Offline Operation | Complete workflow operates without internet/cloud access |
| Model-Agnosticism | Detection logic is not tied to one CV architecture |
| Extensibility | New detector modules can be added without rewriting the core pipeline |
| Transparency | Every finding has human-readable evidence |
| Tamper-Evidence | Sealed records and audit logs are cryptographically verifiable |
| Performance | Hundreds of images should be processed within a few minutes on demo hardware |
| Reproducibility | Attack scenarios and assurance runs use documented parameters |
| Honest Limitations | Unsupported attack classes and unavailable access levels are explicitly reported |
| Human Oversight | Recommendations assist analysts; they do not replace human disposition |

---

## 6. Assurance Report Schema

Each finding shall use the following structure:

```json
{
  "flag_id": "string",
  "module": "data_integrity | model_integrity | inference_provenance | distribution_shift | correlated",
  "affected_asset": "sample_id | contributor_id | batch_id | model_id | inference_record_id",
  "reason": "human-readable explanation",
  "evidence": {
    "metric_name": "value"
  },
  "confidence": 0.0,
  "severity": "low | medium | high",
  "recommended_disposition": "accept | review | quarantine",
  "access_level_used": "white_box | black_box | not_applicable",
  "timestamp": "ISO 8601 string",
  "limitations": "string"
}
```

The overall report shall include:
- Coverage statement
- Supported attack classes
- Unsupported/untested attack classes
- Assumptions
- Summary by severity
- Summary by module
- Correlated findings
- Verification results

---

## 7. System Architecture

```text
 ┌──────────────────┐   ┌──────────────────┐   ┌────────────────────┐
 │ Dataset Input     │   │ Model Input      │   │ Inference Records  │
 │ COCO/YOLO subset  │   │ ONNX / PyTorch   │   │ input+model+output │
 └────────┬─────────┘   └────────┬─────────┘   └──────────┬─────────┘
          │                      │                         │
          ▼                      ▼                         ▼
 ┌──────────────────┐   ┌──────────────────┐   ┌────────────────────┐
 │ F1 Data           │   │ F2 Model         │   │ F3 Provenance      │
 │ Integrity         │   │ Integrity        │   │ Seal + Verify      │
 └────────┬─────────┘   └────────┬─────────┘   └──────────┬─────────┘
          │                      │                         │
          └──────────────┬───────┴──────────────┬──────────┘
                         ▼                      ▼
                ┌──────────────────┐   ┌─────────────────────┐
                │ F4 Distribution   │──▶│ F5 Correlation       │
                │ Shift Detection   │   │ + Composite Flags    │
                └──────────────────┘   └──────────┬──────────┘
                                                  ▼
                                        ┌─────────────────────┐
                                        │ F7 Hash-Chained      │
                                        │ Audit Log             │
                                        └──────────┬──────────┘
                                                   ▼
                                        ┌─────────────────────┐
                                        │ F8 VIGIL-CV Dashboard │
                                        └─────────────────────┘

              ┌────────────────────────────────────────┐
              │ F6 Attack Simulator                     │
              │ poisoning / backdoor / substitution /  │
              │ inference tampering                     │
              └────────────────────────────────────────┘
```

---

## 8. Technology Stack

| Layer | Technology |
|---|---|
| Core | Python 3.x |
| Model Inference | ONNX Runtime, PyTorch |
| Embeddings / Anomaly Detection | scikit-learn, NumPy |
| Image Processing | OpenCV, Pillow |
| Cryptography | `hashlib`, `cryptography`, HMAC / Ed25519 |
| Audit Storage | JSON-lines + hash chain |
| Dashboard | Streamlit |
| Dataset Formats | Simplified COCO/YOLO |
| Model Formats | ONNX / TorchScript |

---

## 9. Use Cases

| Use Case | Actor | Description |
|---|---|---|
| UC1 | System Operator | Run full assurance pipeline |
| UC2 | Analyst | Review flagged findings |
| UC3 | Analyst | Verify a sealed inference record |
| UC4 | System Operator | Run an attack simulation |
| UC5 | Analyst/Evaluator | Verify audit-log integrity |
| UC6 | Analyst | Inspect coverage and limitations |

---

## 10. Prototype Coverage Statement

### Supported
- Near-duplicate detection
- Label-flipping/systematic mislabelling checks
- Basic OOD detection
- Black-box behavioral fingerprinting
- White-box parameter/activation statistics when accessible
- Model substitution/modification detection through behavioral deviation
- Basic trigger-oriented probing where feasible
- Cryptographic inference sealing and tamper detection
- Basic distribution-distance drift detection
- Cross-module correlation
- Hash-chained audit-log verification
- Reproducible poisoning, backdoor, substitution, and tampering demonstrations

### Explicit Limitations
- Full COCO/YOLO parsing is not required for the prototype; a declared subset may be used.
- Deep neuron-level backdoor reconstruction is out of scope.
- Detection of every possible poisoning/backdoor technique is not guaranteed.
- Operational drift versus malicious manipulation cannot always be distinguished reliably.
- Production-scale performance and multi-user access control are out of scope.
- Attack simulation demonstrates representative scenarios and does not prove coverage of all real-world attacks.

---

## 11. Security and Trust Principles

VIGIL-CV shall follow these principles:

1. **Evidence before conclusion** — every finding must point to measurable evidence.
2. **No fabricated certainty** — unavailable checks are reported as unavailable.
3. **Reference-based assurance** — model and distribution comparisons require declared references.
4. **Cryptographic verifiability** — protected records must be independently verifiable.
5. **Reproducibility** — demonstrations use scripted, documented scenarios.
6. **Human oversight** — final disposition remains with the analyst.
7. **Explicit coverage** — the system states what it did and did not test.

---

## 12. Expected Outcome

The completed prototype shall provide a local, offline assurance layer — **VIGIL-CV** — capable of taking a representative CV dataset, model, and inference records and producing:

- Evidence-backed integrity findings
- Contributor/source-level risk indicators
- Model behavioral comparison
- Inference tamper verification
- Distribution-shift indicators
- Correlated composite findings
- A reproducible attack demonstration
- A tamper-evident audit trail
- A human-readable assurance dashboard
- A coverage statement describing supported checks and limitations

The central objective is not to claim that a CV pipeline is universally secure. It is to make its **integrity claims measurable, explainable, and cryptographically verifiable**.

---

*End of Document*
