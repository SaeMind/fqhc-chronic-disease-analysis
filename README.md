# FQHC Intelligence Platform

Local analysis of geography, health-center quality, and national survey-based outreach scenarios. Historical NAMCS and demonstration artifacts are retained in Git at baseline commit `04370105300b7cc32499a34bfb6e99e7589521f4`. The prior model and financial claims are retired pending provenance review.

## Run locally

From the repository root, use Python 3.14. The verified input snapshots are already present in this work copy under `data/public/`.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt
make PYTHON=.venv/bin/python verify test
make PYTHON=.venv/bin/python run
```

Open [the local dashboard](http://127.0.0.1:8501). The server binds to loopback and disables Streamlit usage reporting.

If input files are missing, run `python3 scripts/fetch_public.py` to acquire the exact files in [the provenance ledger](evidence/sources.json). The HRSA site feed is a moving snapshot and the CDC export can change byte order. Keep the verified local snapshots if you need to reproduce this run. A changed upstream file requires review and a new ledger entry. The app blocks results when an input is missing, modified, or fails an analytical gate.

## Modules

| Module | Unit of analysis | Data and interpretation |
| --- | --- | --- |
| Need versus Capacity | County or county equivalent | CDC PLACES crude diagnosed diabetes prevalence with its published 95% interval, 2019–2023 ACS B01003 population, CDC/ATSDR national SVI 2022 and HRSA active permanent awardee-site density. Site density measures physical presence and serves as a proxy for capacity. |
| Peer-Benchmarked Quality | 2025 HRSA H80 awardee organization | UDS Table 7 hypertension control and poor glycemic status or missing assessment, compared against other awardees in the same urban/rural setting and records-reviewed band. |
| Outreach Prioritization | Held-out NHANES survey sample | National survey-based proxy using NHANES 2017–March 2020 public-use files. A cross-sectional model explores fixed-capacity scenarios and weighted net benefit for a single elevated HbA1c measurement. It has no clinical or external FQHC validation. |

See [PLAN.md](PLAN.md) for mappings and gate rules and [REVIEW_PACKET.md](REVIEW_PACKET.md) for verified results and limitations.
