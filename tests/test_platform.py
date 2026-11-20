from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src import platform_data as platform

APP = Path(__file__).resolve().parents[1] / "src" / "dashboard.py"


@pytest.fixture(scope="module")
def real():
    return platform.build()


def test_missing_or_corrupted_sources_fail_closed(tmp_path):
    with pytest.raises(platform.GateError, match="G1 .*Missing source"):
        platform.verify_sources(tmp_path)
    (tmp_path / "uds").write_bytes(b"not the workbook")
    with pytest.raises(platform.GateError, match="G1 .*Size mismatch"):
        platform.verify_sources(tmp_path)


def test_county_join_and_normalization(real):
    counties, _, _, audit = real
    assert counties["FIPS"].str.fullmatch(r"[0-9]{5}").all()
    assert counties["FIPS"].is_unique
    assert counties["Matched permanent awardee sites"].sum() == audit["G2"]["matched sites"]
    np.testing.assert_allclose(counties["Matched sites per 100,000 residents"],
                               100_000 * counties["Matched permanent awardee sites"] / counties["ACS population"])


def test_uds_rates_and_peer_exclusions(real):
    _, centers, _, _ = real
    for data in centers.values():
        assert data["BHCMISID"].is_unique
        assert data["Rate (%)"].between(0, 100).all()
        row = data.iloc[0]
        peers = data[(data["UrbanRuralFlag"] == row["UrbanRuralFlag"])
                     & (data["Review volume"] == row["Review volume"])
                     & (data["BHCMISID"] != row["BHCMISID"])]
        assert row["Other eligible peers"] == len(peers)
        assert row["Peer median (%)"] == peers["Rate (%)"].median()
        assert row["Rate (%)"] == pytest.approx(100 * row["Numerator"] / row["Records reviewed"])
    assert pd.isna(platform.uds_count(pd.Series(["--", "-", "0"])))[:2].all()


def test_holdout_capacity_and_decision_curve(real):
    _, _, test, audit = real
    assert test["SEQN"].is_unique
    assert test["positive"].sum() == audit["G4"]["test positives"]
    assert 0 <= audit["NHANES held-out metrics"]["ROC AUC"] <= 1
    assert platform.capacity(test, len(test))["observed positives selected"] == test["positive"].sum()
    with pytest.raises(ValueError):
        platform.capacity(test, 0)
    curve = platform.decision_curve(test)
    threshold = curve.index[0]
    prevalence = audit["NHANES held-out metrics"]["Weighted positive share"]
    assert curve.iloc[0]["Outreach to all"] == pytest.approx(
        prevalence - (1 - prevalence) * threshold / (1 - threshold))
    assert curve["Outreach to none"].eq(0).all()


def test_dashboard_routes_and_controls():
    app = AppTest.from_file(str(APP), default_timeout=45).run()
    assert not app.exception and not app.error
    assert app.title[0].value == "Need versus Capacity"
    app.selectbox[0].select("IL").run()
    assert app.dataframe[0].value["State"].eq("IL").all()
    app.sidebar.radio[0].set_value("Peer-Benchmarked Quality").run()
    app.selectbox[0].select("Poor glycemic status or missing assessment").run()
    app.checkbox[0].check().run()
    assert not app.exception and not app.error
    assert len(app.dataframe[0].value) and app.dataframe[0].value["Peer review flag"].eq("Review").all()
    app.sidebar.radio[0].set_value("Outreach Prioritization").run()
    app.slider[0].set_value(25).run()
    assert not app.exception and not app.error
    assert "national survey-based proxy" in app.warning[0].value
    assert len(app.metric) == 4
    assert "SEQN" not in app.table[0].value.columns


def test_dashboard_blocks_output_when_source_gate_fails(monkeypatch, tmp_path):
    verify_sources = platform.verify_sources
    monkeypatch.setattr(platform, "verify_sources", lambda: verify_sources(tmp_path))
    app = AppTest.from_file(str(APP), default_timeout=15).run()
    assert not app.exception
    assert "G1" in app.error[0].value
    assert not app.dataframe and not app.metric
