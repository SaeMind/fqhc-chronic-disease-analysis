"""Local Streamlit entry point."""

import pandas as pd
import streamlit as st

from src.platform_data import GateError, build, capacity, decision_curve, verify_sources

MODULES = ["Need versus Capacity", "Peer-Benchmarked Quality", "Outreach Prioritization"]
PROXY = "NHANES national survey-based proxy · 2017–March 2020"


@st.cache_data(show_spinner="Validating and analyzing the public data…")
def load_results(sources):
    # The source ledger is a cache input so an approved vintage change invalidates results.
    return build()


def geography_page(counties, audit):
    st.title(MODULES[0])
    st.caption("CDC PLACES 2024 · Census ACS 2019–2023 · CDC/ATSDR SVI 2022 · HRSA sites November 2026")
    st.write("Compare modeled adult diabetes prevalence and social vulnerability with the density "
             "of matched, active permanent FQHC awardee sites across U.S. counties.")
    state = st.selectbox("State or District of Columbia", sorted(counties["State"].unique()))
    view = counties.loc[counties["State"].eq(state)]
    st.scatter_chart(view, x="Matched sites per 100,000 residents", y="Diabetes prevalence (%)",
                     color="SVI percentile (0–1)")
    st.dataframe(view, hide_index=True, column_config={
        "Matched sites per 100,000 residents": st.column_config.NumberColumn(format="%.2f"),
        "Diabetes prevalence (%)": st.column_config.NumberColumn(format="%.1f"),
    })
    st.caption(f"{audit['matched sites']:,} of {audit['eligible sites']:,} eligible site records matched "
               "a county nationally. A zero means zero matched site records. Site density is a proxy "
               "for physical presence. Staffing, appointment availability, catchment boundaries, "
               "cross-county care and estimates from different years can affect interpretation.")


def quality_page(centers, audit):
    st.title(MODULES[1])
    st.caption("HRSA Uniform Data System · 2025 Health Center Program awardees")
    measure = st.selectbox("Quality measure", list(centers))
    data = centers[measure]
    direction = "Higher rates are favorable." if measure == "Hypertension control" else "Lower rates are favorable."
    st.write(f"{direction} Rates use the published numerator divided by the published number "
             "of records reviewed. The peer group uses the same urban/rural status and review-volume band.")
    state = st.selectbox("State or territory", ["All", *sorted(data["HealthCenterState"].dropna().unique())])
    review_only = st.checkbox("Show peer review flags only")
    view = data if state == "All" else data.loc[data["HealthCenterState"].eq(state)]
    if review_only:
        view = view.loc[view["Peer review flag"].eq("Review")]
    view = view.rename(columns={"BHCMISID": "Center ID", "HealthCenterName": "Health center",
                                "HealthCenterState": "State", "UrbanRuralFlag": "Setting"})
    st.dataframe(view, hide_index=True, column_config={
        "Rate (%)": st.column_config.NumberColumn(format="%.1f"),
        "Peer median (%)": st.column_config.NumberColumn(format="%.1f"),
    })
    counts = audit[measure]
    st.caption(f"{counts['benchmarked']:,} of {counts['roster']:,} awardees have a peer benchmark "
               "for this measure. A flag marks the unfavorable tail of the peer distribution. "
               "This is a descriptive comparison without case-mix adjustment or significance testing.")


def outreach_page(holdout, audit):
    st.title(MODULES[2])
    st.warning(PROXY)
    st.write("Explore how a fixed number of outreach slots would perform in held-out NHANES "
             "survey records. The outcome is a single measured HbA1c of 6.5% or greater among "
             "eligible adults without self-reported diagnosed diabetes.")
    st.caption("Cross-sectional research scenario only. The score has no external FQHC validation "
               "and is unsuitable for patient selection or diagnosis.")
    metrics = audit["NHANES held-out metrics"]
    for column, (label, value) in zip(st.columns(len(metrics)), metrics.items()):
        column.metric(label, f"{value:.3f}" if label != "Weighted positive share" else f"{value:.1%}")
    st.subheader("Fixed-capacity scenario")
    slots = st.slider("Outreach slots among held-out survey records", 1, len(holdout), min(100, len(holdout)))
    scenario = capacity(holdout, slots)
    st.table(pd.DataFrame({
        "Strategy": ["Highest model scores", "Random selection (expected)"],
        "Positive records among selected": [scenario["observed positives selected"],
                                            round(scenario["expected positives under random selection"], 2)],
    }))
    st.caption("Capacity counts use individual survey records without weights. The random strategy "
               "uses the expected yield at the same number of slots.")
    st.subheader("Decision curve")
    st.line_chart(decision_curve(holdout), x_label="Score threshold", y_label="Weighted net benefit")
    st.caption("Net benefit applies a false-positive penalty implied by each threshold. Held-out "
               "metrics and the curve use MEC weights. They are descriptive for the held-out subset.")


st.set_page_config(page_title="FQHC Intelligence Platform", layout="wide")
st.sidebar.title("FQHC Intelligence Platform")
module = st.sidebar.radio("Analysis", MODULES)
try:
    sources = verify_sources()
    counties, centers, holdout, audit = load_results(sources)
except GateError as error:
    st.error(f"Analysis unavailable. {error}")
    st.stop()

with st.sidebar.expander("Public data sources"):
    for source in sources.values():
        st.markdown(f"[{source['use']}]({source['documentation']}) · {source['vintage']}")

if module == MODULES[0]:
    geography_page(counties, audit["G2"])
elif module == MODULES[1]:
    quality_page(centers, audit["G3"])
else:
    outreach_page(holdout, audit)
