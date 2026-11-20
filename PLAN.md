# Implementation and data validation

The evidence branch starts at `04370105300b7cc32499a34bfb6e99e7589521f4`. Preserve historical evidence in Git, remove legacy runtime paths, verify the public source snapshots, build the three modules in order, and exercise the local Streamlit entry point. The dashboard proceeds only when G1 through G4 pass. G5 is the integration acceptance test.

## G1 · Source integrity

The [provenance ledger](evidence/sources.json) records direct download URLs, vintages, acquisition date, filenames, documentation, sizes, and SHA-256 hashes. Match every input before parsing. The hashes document the locally acquired bytes. The directory is ignored by Git. HRSA assigns responsibility for reprocessing to the user under its [UDS terms](https://data.hrsa.gov/data/download). CDC files are public-use releases and are processed locally. The app exposes aggregate NHANES results only.

## G2 · County prevalence and site density

| Input | Grain and fields |
| --- | --- |
| CDC PLACES | County FIPS, name, state, crude diabetes prevalence and published confidence interval |
| Census ACS | B01003 population estimate for GEO_ID beginning with 0500000US |
| CDC/ATSDR SVI | National county FIPS and overall percentile RPL_THEMES. The published sentinel −999 becomes missing. |
| HRSA service sites | BPHC Assigned Number, state-county FIPS, site status, health-center type, site function and location type |

Preserve five-digit FIPS keys. Use the PLACES county universe for the 50 states and DC, enforce one-to-one ACS and SVI joins, and validate estimate and interval bounds. Join eligible active permanent awardee service sites, including combined administrative and service sites. Exclude purely administrative, mobile, seasonal, and look-alike sites. Require at least 95% site linkage and 98% combined ACS/SVI completeness. Keep absent ACS and SVI values missing. Calculate matched site count per 100,000 total residents and preserve PLACES estimates at their published precision.

The 2025 H80 workbook also contains HealthCenterZipCodes, with BHCMISID, GrantNumber, ReportingYear, ZipCode, ZipCodeType, and TotalNumberofPatients. Counts include suppression. USPS ZIP populations and county FIPS have different boundaries. The implementation uses the HRSA site snapshot with an explicit county FIPS field and leaves the ZIP table out of the density calculation.

## G3 · UDS quality

| Sheet | Organization mapping and fields |
| --- | --- |
| HealthCenterInfo | BHCMISID and GrantNumber, validated against ReportingYear, with center name/state and UrbanRuralFlag |
| Table7_1 | Total-line T7_Li_C2c controlled hypertension divided by T7_Li_C2b records reviewed |
| Table7_2 | Total-line T7_Li_C3f glycemic status assessment >9%, missing, or no test divided by T7_Li_C3b records reviewed |

The published counts and measure directions follow [HRSA Table 7 guidance](https://bphc.hrsa.gov/sites/default/files/bphc/data-reporting/table-7-fact-sheet-2025.pdf). Validate keys and roster size against the workbook cover, and check integer counts and numerator bounds. Treat the workbook markers `--` and `-` as missing and omit them from rates and peers. Require a minimum of 30 reviewed records per center and at least 85% usable coverage per measure. Compare each center with other eligible centers in the same setting and preselected review-volume band (30–249, 250–999, 1,000+), requiring at least 10 other peers. A flag indicates a rate strictly below the other peers’ 10th percentile for hypertension control or strictly above the 90th percentile for poor glycemic status. These thresholds are descriptive and carry no statistical significance or causal interpretation.

## G4 · NHANES proxy

Join P_DEMO, P_GHB, P_BMX and P_DIQ one-to-one by SEQN. Reject orphan or duplicate IDs. Use adults 20+ with no reported diagnosed diabetes (DIQ010 2 or 3), no flagged pregnancy (RIDEXPRG 1), positive MEC weights and complete predictors, outcome and survey design fields. Unknown or unavailable pregnancy status remains eligible. The outcome is a single LBXGH at or above 6.5%. [NIDDK describes the need for confirmation in clinical diagnosis](https://www.niddk.nih.gov/health-information/diagnostic-tests/a1c-test).

Use age, survey-recorded sex and measured BMI as features. Fit a weighted logistic regression with a weighted training-only standardizer. Split by masked strata and PSU pair with a fixed seed to hold out approximately 30% of clusters. Require disjoint clusters and at least 20 observations of each class in both splits. Use the pre-pandemic MEC exam weight WTMECPRP for fitting and descriptive holdout evaluation. No variance estimation or calibration adjustment is applied. The score measures association with a concurrent lab marker. It provides no estimate of prospective risk or outreach effectiveness.

For fixed capacity, select the highest holdout scores, break exact score ties by SEQN and compare the unweighted observed yield with the expected yield from the same number of randomly selected records. For thresholds 0.01 through 0.20, calculate weighted decision-curve net benefit as true-positive share minus false-positive share multiplied by threshold/(1 − threshold). Compare against selecting all or none. The threshold range is illustrative.

## G5 · Streamlit acceptance

Exercise all three routes, geography and quality filters, fixed-capacity slider, proxy labeling and blocking of tables and metrics on G1 failure. Re-run `make verify test` before revisions to the evidence packet.
