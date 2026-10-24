"""Streamlit entry point. S1: the deployment probe; the S6 demo replaces it (docs/plan.md).

Streamlit Community Cloud runs this file without installing the package, so the two lines below
put ``src/`` on the import path; dependencies come from the requirements.txt next to this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from shopsage_rec.serving.probe import ProbeReport, run_probe

st.set_page_config(page_title="ShopSage probe", layout="centered")
st.title("ShopSage Recommender")
st.caption(
    "Deployment probe: a Milvus Lite collection the size of the demo's catalogue (20,108 items, "
    "128-d dense and sparse vectors, synthetic data), queried by dense, sparse and RRF hybrid "
    "search. The recommender demo replaces this page."
)


@st.cache_resource(show_spinner="Building the Milvus Lite collection and timing queries ...")
def _probe() -> ProbeReport:
    return run_probe()


if st.button("Run the probe again"):
    _probe.clear()
rows = _probe().rows()
st.table({"measure": [label for label, _ in rows], "value": [value for _, value in rows]})
