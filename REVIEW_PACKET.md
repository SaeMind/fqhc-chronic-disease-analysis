# Project 3 review packet

**Local evidence branch** `codex/fqhc-evidence`

**Baseline** `04370105300b7cc32499a34bfb6e99e7589521f4`

**Verification date** November 19, 2026

## Review decision

G1–G5 passed for the pinned snapshots in this checkout. The app is ready for local review. The three modules support exploratory analysis and the NHANES module is prominently labeled a national survey-based proxy.

| Gate | Verified result |
| --- | --- |
| G1 · Provenance | Nine official-source files matched the recorded sizes and SHA-256 hashes. The acquisition script verified all nine existing local snapshots. |
| G2 · Geographic joins | All 3,144 PLACES counties joined to both ACS and SVI. 13,160 of 13,161 eligible HRSA sites matched. The excluded California site had a blank county FIPS. |
| G3 · UDS benchmarking | The workbook roster reconciled to 1,356 awardees. Hypertension control had 1,354 reportable and benchmarked centers. Poor glycemic status had 1,345 reportable centers and 1,344 benchmarked centers. |
| G4 · NHANES proxy | 6,646 complete cases, with 4,527 training and 2,119 holdout records across disjoint masked clusters. The 15 holdout clusters contained 106 positive observations. |
| G5 · Local app | Six tests passed. Streamlit integration exercised all three modules, the filters and slider, the proxy label, and the G1 failure state. A loopback server health check and app HTTP request passed, and the test server was stopped. |

See the [source ledger](evidence/sources.json), [machine-readable verification output](evidence/verification.json), and [acceptance tests](tests/test_platform.py). The initial geography gate rejected PLACES intervals with space padding. The parser was corrected and the same bound checks then passed. The initial dashboard tests exposed an import-path mismatch, which was corrected before the final pass.

## Verified NHANES holdout diagnostics

| MEC-weighted diagnostic | Result |
| --- | ---: |
| ROC AUC | 0.7483 |
| Average precision | 0.0733 |
| Brier score | 0.03031 |
| Positive share | 3.164% |

The results describe a single cluster holdout of a complete-case, pre-pandemic survey subset. Average precision should be read alongside the positive share. The decision curve uses hypothetical false-positive costs and the capacity comparison uses unweighted survey records. Clinical utility, diagnostic performance, subgroup equity, and transfer to FQHC patients have not been established.

## Historical evidence and remaining limitations

- The baseline Streamlit app loaded historical NAMCS model files. Separate development code retained a synthetic training path. The local branch removes these paths and the active README, portfolio, and ROI claims. Prior notebooks, figures, model binaries and reports can be audited at the baseline commit. This review did not authenticate their modeling results.
- M1 uses a November 2026 HRSA directory snapshot alongside older need estimates. Counts describe permanent awardee site presence. Staffing, capacity, travel across county lines, mobile and seasonal coverage, and FQHC look-alike sites are outside this measure.
- M2 uses records reviewed as the denominator and uses observed urban/rural setting and volume for descriptive peer groups. The flags support review of unfavorable tails. They convey neither significance nor case-mix-adjusted quality.
- M3 is cross-sectional and excludes incomplete records. Missing pregnancy status is permitted when NHANES did not flag a pregnancy. MEC weighting does not remove complete-case selection effects. A single HbA1c result carries no confirmed diagnosis and the module exposes aggregate results only.
- The source site feed may roll forward. Preserve the locally downloaded, ignored files for exact reproduction. The hash gate rejects changed bytes until the ledger and results are reviewed for a new snapshot.

Reproduce from the repository root with `make verify` and `make test`. Start the dashboard with `make run`. See [README.md](README.md) for environment setup and [PLAN.md](PLAN.md) for source mappings and acceptance rules.
