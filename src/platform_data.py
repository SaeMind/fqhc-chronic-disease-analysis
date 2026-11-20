"""Pinned public data inputs and analytical acceptance gates."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "public"
LEDGER = ROOT / "evidence" / "sources.json"
INPUTS = {"uds", "sites", "places", "acs", "svi", "demo", "a1c", "bmi", "diq"}

SITE_ID = "BPHC Assigned Number"
SITE_FIPS = "State and County Federal Information Processing Standard Code"
MEASURES = {
    "Hypertension control": ("Table7_1", "T7_Li_C2b", "T7_Li_C2c", True),
    "Poor glycemic status or missing assessment": ("Table7_2", "T7_Li_C3b", "T7_Li_C3f", False),
}
FEATURES = ["RIDAGEYR", "RIAGENDR", "BMXBMI"]


class GateError(ValueError):
    """An acceptance gate prevented results from being displayed."""


def require(condition, gate, message):
    if not condition:
        raise GateError(f"{gate} · {message}")


def verify_sources(directory=DATA):
    ledger = json.loads(LEDGER.read_text())
    require(set(ledger) == INPUTS, "G1", "Source ledger is incomplete.")
    for name, source in ledger.items():
        path = directory / name
        require(path.is_file(), "G1", f"Missing source {name}.")
        require(path.stat().st_size == source["bytes"], "G1", f"Size mismatch for {name}.")
        with path.open("rb") as file:
            actual = hashlib.file_digest(file, "sha256").hexdigest()
        require(actual == source["sha256"], "G1", f"SHA-256 mismatch for {name}.")
    return ledger


def unique(frame, key, gate):
    require(frame[key].notna().all() and frame[key].is_unique, gate,
            f"Missing or duplicated {key}.")


def county_keys(frame):
    return frame["FIPS"].str.fullmatch(r"[0-9]{5}", na=False)


def geography(directory):
    places = pd.read_csv(directory / "places", dtype={"countyfips": str}).rename(columns={
        "countyfips": "FIPS", "countyname": "County", "stateabbr": "State",
        "diabetes_crudeprev": "Diabetes prevalence (%)", "diabetes_crude95ci": "PLACES 95% interval",
    })
    states = set(places["State"]) - {"PR"}
    places = places[places["State"].isin(states)]
    prevalence = places["Diabetes prevalence (%)"]
    interval = places["PLACES 95% interval"].str.extract(
        r"^\(\s*([0-9.]+),\s*([0-9.]+)\)$"
    ).astype(float)
    require((interval[0].between(0, 100) & interval[1].between(0, 100)
             & interval[0].le(prevalence) & interval[1].ge(prevalence)).all(),
            "G2", "Invalid PLACES estimate or interval.")

    acs = pd.read_csv(directory / "acs", sep="|", dtype=str)
    acs = acs[acs["GEO_ID"].str.fullmatch(r"0500000US[0-9]{5}", na=False)].copy()
    acs["FIPS"] = acs["GEO_ID"].str[-5:]
    acs["ACS population"] = pd.to_numeric(acs["B01003_E001"])
    require(acs["ACS population"].gt(0).all(), "G2", "Invalid ACS denominator.")
    svi = pd.read_csv(directory / "svi", encoding="utf-8-sig",
                      usecols=["FIPS", "RPL_THEMES"], dtype={"FIPS": str})
    svi = svi.rename(columns={"RPL_THEMES": "SVI percentile (0–1)"})
    rank = svi["SVI percentile (0–1)"]
    require((rank.between(0, 1) | rank.eq(-999)).all(), "G2", "Invalid SVI percentile.")
    svi.loc[rank.eq(-999), "SVI percentile (0–1)"] = np.nan

    for frame in (places, acs, svi):
        unique(frame, "FIPS", "G2")
        require(county_keys(frame).all(), "G2", "Malformed county key.")
    counties = places.merge(acs[["FIPS", "ACS population"]], how="left", validate="one_to_one")
    counties = counties.merge(svi, how="left", validate="one_to_one")

    sites = pd.read_csv(directory / "sites", dtype=str, keep_default_na=False)
    unique(sites, SITE_ID, "G2")
    require(sites[SITE_ID].ne("").all(), "G2", "Empty HRSA site ID.")
    sites = sites.loc[
        sites["Site Status Description"].eq("Active")
        & sites["Health Center Type"].eq("Federally Qualified Health Center (FQHC)")
        & sites["Health Center Location Type Description"].eq("Permanent")
        & sites["Health Center Type Description"].isin(
            ["Service Delivery Site", "Administrative/Service Delivery Site"]
        )
        & sites["Site State Abbreviation"].isin(states)
    ].rename(columns={SITE_FIPS: "FIPS"})
    matched = sites["FIPS"].isin(counties["FIPS"])
    require(len(sites) > 0 and matched.mean() >= 0.95, "G2", "Site-to-county match below 95%.")
    counts = sites.loc[matched].groupby("FIPS").size()
    counties["Matched permanent awardee sites"] = counties["FIPS"].map(counts).fillna(0).astype(int)
    counties["Matched sites per 100,000 residents"] = (
        counties["Matched permanent awardee sites"] / counties["ACS population"] * 100_000
    )
    complete = counties[["ACS population", "SVI percentile (0–1)"]].notna().all(axis=1)
    require(len(counties) > 0 and complete.mean() >= 0.98, "G2", "County coverage below 98%.")
    audit = {
        "PLACES counties": len(counties), "counties with ACS and SVI": int(complete.sum()),
        "counties missing ACS": int(counties["ACS population"].isna().sum()),
        "counties missing SVI": int(counties["SVI percentile (0–1)"].isna().sum()),
        "eligible sites": len(sites), "matched sites": int(matched.sum()),
    }
    return counties.sort_values(["State", "County"]), audit


def sheet(workbook, name, skip_description=False):
    rows = workbook[name].values
    headers = next(rows)
    if skip_description:
        next(rows)
    return pd.DataFrame(rows, columns=headers).dropna(how="all")


def uds_count(values):
    clean = values.replace({"--": None, "-": None})
    count = pd.to_numeric(clean)
    require((count.dropna().ge(0) & count.dropna().mod(1).eq(0)).all(), "G3",
            "UDS totals contain invalid counts.")
    return count


def quality(directory):
    key = "BHCMISID"
    with (directory / "uds").open("rb") as file:
        workbook = load_workbook(file, read_only=True, data_only=True)
        roster = sheet(workbook, "HealthCenterInfo")
        year = sheet(workbook, "Coversheet").iloc[0]
        require(str(year["ReportingYear"]) == "2025"
                and roster["ReportingYear"].astype(str).eq("2025").all()
                and len(roster) == int(year["TotalNumberOfHealthCenters"]),
                "G3", "Workbook does not reconcile with the H80 2025 roster.")
        unique(roster, key, "G3")
        results, audit = {}, {}
        for measure, (tab, denominator, numerator, higher) in MEASURES.items():
            raw = sheet(workbook, tab, skip_description=True)[
                [key, "GrantNumber", denominator, numerator]
            ]
            unique(raw, key, "G3")
            require(set(raw[key]) == set(roster[key]), "G3", f"{tab} roster mismatch.")
            cols = [key, "GrantNumber", "HealthCenterName", "HealthCenterState", "UrbanRuralFlag"]
            data = roster[cols].merge(raw, on=[key, "GrantNumber"], validate="one_to_one")
            require(len(data) == len(roster), "G3", f"{tab} grant numbers do not reconcile.")
            data["Records reviewed"] = uds_count(data[denominator])
            data["Numerator"] = uds_count(data[numerator])
            reported = data[["Records reviewed", "Numerator"]].notna().all(axis=1)
            observed = data.loc[reported]
            require(observed["Numerator"].le(observed["Records reviewed"]).all(),
                    "G3", "UDS numerator exceeds denominator.")
            eligible = reported & data["Records reviewed"].ge(30)
            data = data.loc[eligible].copy()
            data["Rate (%)"] = 100 * data["Numerator"] / data["Records reviewed"]
            data["Review volume"] = pd.cut(
                data["Records reviewed"], [29, 249, 999, np.inf],
                labels=["30–249", "250–999", "1,000+"],
            )
            require(len(data) >= 0.85 * len(roster), "G3", f"{tab} usable coverage below 85%.")
            results[measure] = benchmark(data, higher)
            audit[measure] = {"roster": len(roster), "reportable": int(reported.sum()),
                              "benchmarked": int(results[measure]["Peer median (%)"].notna().sum()),
                              "eligible": len(data)}
        workbook.close()
    return results, audit


def benchmark(data, higher):
    data["Other eligible peers"] = 0
    for _, peers in data.loc[data["UrbanRuralFlag"].isin(["Rural", "Urban"])].groupby(
        ["UrbanRuralFlag", "Review volume"], observed=True
    ):
        data.loc[peers.index, "Other eligible peers"] = len(peers) - 1
        if len(peers) < 11:
            continue
        for index, row in peers.iterrows():
            reference = peers.loc[peers.index != index, "Rate (%)"]
            rate = row["Rate (%)"]
            data.loc[index, "Peer median (%)"] = reference.median()
            cutoff = reference.quantile(0.10 if higher else 0.90)
            data.loc[index, "Peer review flag"] = "Review" if (
                rate < cutoff if higher else rate > cutoff
            ) else "—"
    return data.drop(columns=["GrantNumber"])


def nhanes(directory):
    specifications = {
        "demo": ["RIDAGEYR", "RIAGENDR", "RIDEXPRG", "SDMVSTRA", "SDMVPSU", "WTMECPRP"],
        "diq": ["DIQ010"], "bmi": ["BMXBMI"], "a1c": ["LBXGH"],
    }
    frames = {}
    for name, columns in specifications.items():
        frame = pd.read_sas(directory / name, format="xport")[["SEQN", *columns]]
        unique(frame, "SEQN", "G4")
        frames[name] = frame
    cohort = frames["demo"]
    audit = {"demographic records": len(cohort)}
    for name in ("diq", "bmi", "a1c"):
        frame = frames[name]
        require(frame["SEQN"].isin(cohort["SEQN"]).all(), "G4", f"Orphaned NHANES {name} rows.")
        audit[f"{name} file records"] = len(frame)
        cohort = cohort.merge(frame, how="left", on="SEQN", validate="one_to_one")

    eligible = cohort.loc[
        cohort["RIDAGEYR"].ge(20) & cohort["DIQ010"].isin([2, 3])
        & cohort["RIDEXPRG"].ne(1) & cohort["WTMECPRP"].gt(0)
    ]
    cohort = eligible.dropna(subset=[*FEATURES, "LBXGH", "SDMVSTRA", "SDMVPSU"]).copy()
    require(np.isfinite(cohort[[*FEATURES, "LBXGH", "WTMECPRP"]]).all().all()
            and cohort["RIAGENDR"].isin([1, 2]).all()
            and cohort[["LBXGH", "BMXBMI"]].gt(0).all().all(), "G4", "Invalid cohort values.")
    cohort["positive"] = cohort["LBXGH"].ge(6.5)
    groups = cohort.groupby(["SDMVSTRA", "SDMVPSU"]).ngroup()
    train, test = next(GroupShuffleSplit(test_size=0.30, random_state=42).split(cohort, groups=groups))
    require(set(groups.iloc[train]).isdisjoint(groups.iloc[test]), "G4", "Survey cluster leakage.")
    audit.update({"eligible before complete-case filter": len(eligible), "complete cases": len(cohort),
                  "train": len(train), "test": len(test), "test clusters": groups.iloc[test].nunique()})
    train, test = cohort.iloc[train], cohort.iloc[test].copy()
    for split in (train, test):
        require(split["positive"].value_counts().reindex([True, False], fill_value=0).min() >= 20,
                "G4", "Too few held-out or training outcomes.")
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    weights = train["WTMECPRP"] / train["WTMECPRP"].mean()
    model.fit(train[FEATURES], train["positive"],
              standardscaler__sample_weight=weights, logisticregression__sample_weight=weights)
    test["score"] = model.predict_proba(test[FEATURES])[:, 1]
    require(test["score"].between(0, 1).all(), "G4", "Invalid held-out scores.")
    y, score, weights = test["positive"], test["score"], test["WTMECPRP"]
    metrics = {"ROC AUC": roc_auc_score(y, score, sample_weight=weights),
               "Average precision": average_precision_score(y, score, sample_weight=weights),
               "Brier score": brier_score_loss(y, score, sample_weight=weights),
               "Weighted positive share": float(np.average(y, weights=weights))}
    audit.update({"train positives": int(train["positive"].sum()), "test positives": int(y.sum())})
    return test[["SEQN", "positive", "WTMECPRP", "score"]], metrics, audit


def capacity(test, slots):
    if not 1 <= slots <= len(test):
        raise ValueError("Capacity must be between one and the evaluation cohort size.")
    selected = test.sort_values(["score", "SEQN"], ascending=[False, True]).head(slots)
    return {"slots": slots, "observed positives selected": int(selected["positive"].sum()),
            "expected positives under random selection": slots * float(test["positive"].mean())}


def decision_curve(test):
    y, weights = test["positive"], test["WTMECPRP"]
    rows = []
    for threshold in np.arange(1, 21) / 100:
        penalty = threshold / (1 - threshold)
        utility = y.astype(float) - (~y).astype(float) * penalty
        rows.append({"Threshold": threshold,
                     "Score-based outreach": np.average(utility * test["score"].ge(threshold), weights=weights),
                     "Outreach to all": np.average(utility, weights=weights), "Outreach to none": 0.0})
    return pd.DataFrame(rows).set_index("Threshold")


def build(directory=DATA):
    verify_sources(directory)
    try:
        counties, geographic_audit = geography(directory)
    except (ValueError, KeyError) as error:
        if isinstance(error, GateError):
            raise
        raise GateError(f"G2 · Invalid geographic source schema: {error}") from error
    try:
        centers, quality_audit = quality(directory)
    except (ValueError, KeyError) as error:
        if isinstance(error, GateError):
            raise
        raise GateError(f"G3 · Invalid UDS source schema: {error}") from error
    try:
        holdout, metrics, proxy_audit = nhanes(directory)
    except (ValueError, KeyError) as error:
        if isinstance(error, GateError):
            raise
        raise GateError(f"G4 · Invalid NHANES source schema: {error}") from error
    audit = {"G1": "All pinned inputs matched", "G2": geographic_audit,
             "G3": quality_audit, "G4": proxy_audit, "NHANES held-out metrics": metrics}
    return counties, centers, holdout, audit


if __name__ == "__main__":
    print(json.dumps(build()[-1], indent=2))
