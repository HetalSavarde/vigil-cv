"""
VIGIL-CV :: F8 - Assurance Dashboard (Streamlit)

Reads output/report.json, output/audit_log.jsonl, output/sealed_records.jsonl
(produced by demo/run_demo.py) and renders them as an interactive dashboard:
tabs per module, evidence-first finding cards, contributor risk table, and a
live "re-verify" button for the audit log and sealed inference records.

Run with:  streamlit run dashboard/app.py
"""
import json
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from audit.audit_log import AuditLog  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE, "output")
REPORT_PATH = os.path.join(OUTPUT_DIR, "report.json")
AUDIT_PATH = os.path.join(OUTPUT_DIR, "audit_log.jsonl")
SEALED_PATH = os.path.join(OUTPUT_DIR, "sealed_records.jsonl")

SEVERITY_COLOR = {"high": "🔴", "medium": "🟠", "low": "🟢"}

st.set_page_config(page_title="VIGIL-CV Assurance Dashboard", layout="wide")


def load_report():
    if not os.path.exists(REPORT_PATH):
        return None
    with open(REPORT_PATH) as f:
        return json.load(f)


def load_sealed_records():
    if not os.path.exists(SEALED_PATH):
        return []
    with open(SEALED_PATH) as f:
        return [json.loads(l) for l in f if l.strip()]


def finding_card(f):
    icon = SEVERITY_COLOR.get(f.get("severity", "low"), "⚪")
    with st.container(border=True):
        st.markdown(f"**{icon} `{f['flag_id']}`** — {f['module']}  "
                    f"&nbsp;&nbsp; confidence **{f['confidence']}** &nbsp;&nbsp; "
                    f"→ *{f['recommended_disposition']}*")
        st.write(f["reason"])
        with st.expander("Evidence"):
            st.json(f.get("evidence", {}))
        if f.get("limitations"):
            st.caption(f"⚠️ Limitations: {f['limitations']}")


st.title("🛡️ VIGIL-CV — Assurance Dashboard")
st.caption("Offline CV pipeline trust & integrity assurance — evidence-first, not black-box scores.")

report = load_report()

if report is None:
    st.warning("No report found yet. Run `python3 demo/run_demo.py` first, then reload this page.")
    st.stop()

# ---- top summary row ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Samples scanned", report["n_samples_scanned"])
c2.metric("Total findings", len(report["findings"]))
sev = report["summary_by_severity"]
c3.metric("High severity", sev.get("high", 0))
c4.metric("Correlated findings", len(report["correlated_findings"]))

st.divider()

tabs = st.tabs([
    "📊 Overview", "🖼️ F1 Data Integrity", "🧠 F2 Model Integrity",
    "🔏 F3 Inference Provenance", "🔗 F5 Correlated", "📜 F7 Audit Log", "📄 Full Report",
])

# ---- Overview ----
with tabs[0]:
    st.subheader("Coverage statement")
    st.info(report["coverage_statement"])

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Supported attack classes**")
        for a in report["supported_attack_classes"]:
            st.markdown(f"- {a}")
    with colB:
        st.markdown("**Not tested / out of scope this run**")
        for a in report["unsupported_or_untested_attack_classes"]:
            st.markdown(f"- {a}")

    st.subheader("Contributor risk")
    risk_rows = []
    for contributor, stats in report["contributor_risk"].items():
        risk_rows.append({
            "contributor": contributor,
            "total_samples": stats["total_samples"],
            "flagged_samples": stats["flagged_samples"],
            "flag_rate": stats["flag_rate"],
            "risk_level": stats["risk_level"],
            "batches": ", ".join(stats["batches"]),
        })
    st.table(risk_rows)

# ---- F1 ----
with tabs[1]:
    st.subheader("Training-data integrity findings")
    f1_findings = [f for f in report["findings"] if f["module"] == "data_integrity"]
    if not f1_findings:
        st.success("No findings.")
    for f in f1_findings:
        finding_card(f)

# ---- F2 ----
with tabs[2]:
    st.subheader("Model behavioral integrity")
    f2_findings = [f for f in report["findings"] if f["module"] == "model_integrity"]
    for f in f2_findings:
        finding_card(f)

# ---- F3 ----
with tabs[3]:
    st.subheader("Sealed inference records")
    sealed_records = load_sealed_records()
    f3_findings = [f for f in report["findings"] if f["module"] == "inference_provenance"]
    for f in f3_findings:
        finding_card(f)

    st.markdown("---")
    st.markdown("**Live re-verify** (recomputes hash + signature right now)")
    if sealed_records:
        from detectors.f3_provenance import verify_record
        for i, rec in enumerate(sealed_records):
            cols = st.columns([3, 1])
            cols[0].code(f"{rec['record_id']}  |  output={rec['output']}  |  seq={rec['sequence_no']}")
            if cols[1].button("Verify now", key=f"verify_{i}"):
                v = verify_record(rec)
                if v["valid"]:
                    st.success(f"✅ VALID — no tampering detected for {rec['record_id']}")
                else:
                    st.error(f"❌ TAMPERED — failed: {v['failed_components']}")

# ---- F5 ----
with tabs[4]:
    st.subheader("Cross-module correlated findings")
    if not report["correlated_findings"]:
        st.info("No correlated composite findings in this run.")
    for f in report["correlated_findings"]:
        finding_card(f)

# ---- F7 ----
with tabs[5]:
    st.subheader("Tamper-evident audit log")
    if os.path.exists(AUDIT_PATH):
        log = AuditLog(AUDIT_PATH)
        entries = log.read_all()
        st.write(f"{len(entries)} entries")
        st.dataframe([
            {"timestamp": e["timestamp"], "event_type": e["event_type"], "details": json.dumps(e["details"])}
            for e in entries
        ], width="stretch", hide_index=True)
        if st.button("Re-verify chain now"):
            v = log.verify_chain()
            if v["chain_intact"]:
                st.success(f"✅ Chain intact across {v['n_entries']} entries")
            else:
                st.error(f"❌ Chain broken: {v['breaks']}")
    else:
        st.info("No audit log found yet.")

# ---- Full report ----
with tabs[6]:
    st.subheader("Raw assurance report (JSON)")
    st.json(report)
    st.download_button("Download report.json", data=json.dumps(report, indent=2),
                        file_name="vigil_cv_report.json", mime="application/json")
