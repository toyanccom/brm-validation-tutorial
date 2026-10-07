# Matching validation units and prediction targets in repeated behavioral records

This repository accompanies a methodological tutorial prepared for Behavior
Research Methods. It provides an executable teaching example and the archived
analysis specification for a separate multiple object tracking (MOT) case study.
It does not imply acceptance, submission, preregistration, or external validation.

**Version 1.0.1, published 7 October 2026:** documented code, the complete
row-level simulated dataset, held-out predictions, expected outputs, and the empirical aggregate archive are
available in [toyanccom/brm-validation-tutorial](https://github.com/toyanccom/brm-validation-tutorial).
Use the files at the repository root. Cite a commit-specific snapshot rather
than a moving branch; GitHub's file permalink or commit history provides it.

The teaching example uses **entirely simulated observations**. No person,
participant record, or MOT response was used to generate them. Synthetic units
are labeled `SIM000`, `SIM001`, and so on. The generator settings are teaching
choices and were not calibrated to the human data.

## Run the complete teaching example

Use Python 3.12.14 and the recorded direct dependency versions:

```bash
python -m venv .venv
```

Activate that environment using the command appropriate for your platform.
Then, from this directory:

```bash
python -m pip install -r requirements.txt
python run_tutorial.py --output-dir outputs
python -m unittest -v test_tutorial.py
```

The tested run uses NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0, and
Matplotlib 3.10.8. Direct dependencies are pinned; transitive dependencies are
resolved by pip. Minor numerical or image differences across platforms remain
possible. The saved run records package versions and hashes for every CSV.

One command generates the artificial records, fits fold-contained pipelines,
saves held-out predictions, evaluates two outcome levels, resamples whole
groups with predictions fixed, and draws the comparison figure. Neither private
workbooks nor an account is required. Default settings are seed 20261005, 80
artificial groups, 30 trials per group, six balanced conditions, and five folds.
The sample sizes are unrelated to the 64 identifiers in each empirical archive.

## Read the two comparisons separately

1. Compare `RandomForest / row / trial` with `RandomForest / group / trial`.
   The model specification, predictors, generated sample, and trial target are
   held constant. The withholding unit changes. The fold audit reports the
   proportion of test groups represented in training. The row split describes
   additional trials of represented units; the group split describes omitted
   units. A row split is not a test of transfer to omitted units.
2. Compare `Ridge / group / trial` with `Ridge / group / group_mean`. These
   summaries use the same held-out predictions. Observations and predictions
   are averaged within group before the second score is computed. The target
   changes; there is no second model fit and no averaging of fold R-squared.

The artificial repeated numeric marker is independent of a unit's response
intercept in the generating population. A flexible model can use its repeated
values to recognize represented units. The group identifier itself is excluded
from predictors. This construction illustrates how shared unit regularities can
matter even when an identifier column is absent from the feature set. The
numerical size of the gap is specific to the chosen generator and estimators.

`R² = 1 - sum((observed - predicted)^2) / sum((observed - mean(observed))^2)`.
The denominator's evaluation-sample mean is a scoring reference. The separate
`MeanBaseline` predictor uses only the training fold's outcome mean. Neither
is an initial measured calibration response from a withheld unit.

## Inspect generated files

| File | Purpose |
| --- | --- |
| `synthetic_records.csv` | Generated inputs and outcomes, including group keys |
| `synthetic_predictions.csv` | One held-out prediction per row and configuration |
| `synthetic_fold_audit.csv` | Training and test counts and measured group overlap |
| `synthetic_metrics.csv` | Trial and group-average metrics, labeled SIMULATED |
| `synthetic_group_means.csv` | Paired group averages computed before scoring |
| `run_metadata.json` | Settings, versions, uncertainty interpretation, CSV hashes |
| `synthetic_comparisons.png` and `.pdf` | Reconstructed teaching figure |

The [`outputs/`](outputs/) directory now contains all five synthetic CSV files,
run metadata, and both comparison figures. Download the input observations
directly from [`outputs/synthetic_records.csv`](outputs/synthetic_records.csv).
The [`PUBLIC_DATA_DICTIONARY.md`](PUBLIC_DATA_DICTIONARY.md) describes every
column and how records and prediction vectors correspond. The `expected/`
directory retains the previously tested aggregate outputs. Every generated row
is artificial; the data origin is also recorded in the metrics and metadata.

The 400-resample intervals condition on already fitted predictions. They exclude
refitting, split selection, model selection, and transfer to another population.
They are not evidence that a particular model performs well on real MOT data.

## Adapt the pattern to a new study

Start from a tidy table with one observation per row. Replace the schema only
after specifying the unit excluded from training, when features become
available, and the target to be evaluated. `generate_records` is a teaching
input; an adapted empirical loader must separately audit repeated sessions,
duplicates, missing fields, and participant identity.

| Decision | Function | Check |
| --- | --- | --- |
| Independent unit and row construction | `generate_records` or your audited loader | Repeated sessions grouped at the relevant person or site level |
| Prediction-time feature availability | `FEATURES` | Exclude outcomes and later measurements not available at prediction |
| Training-contained preprocessing | `make_pipeline` and `validate` | Fit imputation, encoding, and scaling on training rows only |
| Withholding unit | `validate` | Inspect `test_group_overlap` and saved fold assignments |
| Scoring target and weights | `score_targets` | Average paired held-out observations and predictions before mean scoring |
| Conditional uncertainty | `conditional_intervals` | Whole-group resampling preserves repeated records and multiplicity |

Trial scoring weights each row equally; group-average scoring weights each group
equally. Unequal trial counts in another study make this weighting choice
substantive. The tests include an unequal-count example. They also perturb test
values to verify that fitted training transformations cannot change.

For new-session prediction when sessions are nested within people, splitting
only session labels can retain the same person in training. For future trials,
preserve chronology; the randomized teaching conditions do not demonstrate a
temporal deployment test. Any feature selection or hyperparameter tuning belongs
inside each outer training fold, with a separate inner validation process.

Relevant official implementation documentation:
[scikit-learn cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
and [pipelines](https://scikit-learn.org/stable/modules/compose.html).

## Empirical archive and access

`empirical_archive/` preserves the existing MOT code and aggregate tables with
their original names and recorded dependencies. Its `IJIE` paths are historical
labels. That archive is a secondary case study, not the generator of the
synthetic data. The empirical estimates were not refitted for this release.
The public repository contains text, code, and aggregate CSV files. The
manuscript's supplementary ZIP separately preserves the complete original
42-file empirical inventory and its binary figures. The original archive's
README and inventory describe that full archive; they do not expand the contents
or sharing permissions of the public subset.

To inspect the main empirical comparisons without the private workbooks:

```bash
python inspect_archived_results.py
```

To refit the empirical workflow, use the four authorized, byte-identical
workbooks and the separate requirements and commands in
`empirical_archive/README.md`. Synthetic inputs cannot be substituted for the
workbooks to reproduce empirical estimates. Source hashes are retained.

See [DATA_ACCESS.md](DATA_ACCESS.md) for the public file locations and the
separate, unresolved status of original human-data authorization. Author contact
is not the access mechanism for the publicly released synthetic files. The public repository
contains no real individual prediction vectors, original workbooks, or consent
forms. A Git commit fixes the code and aggregate-result snapshot. No DOI or
independent archival deposit has yet been completed. [ARCHIVING.md](ARCHIVING.md)
describes the remaining trusted-repository deposit; the public GitHub snapshot
must not be described as a Zenodo or OSF record.

## Relationship between manuscripts

The BRM tutorial owns the predictive-validation comparisons, target-level
scores, and the new synthetic teaching example. The main MOT manuscript owns
the local-versus-session-wide reference comparison and its relevant checks.
The revised MOT manuscript does not repeat the predictive tables. Both reuse
the same four empirical workbooks, and related preprints and the age manuscript
must be disclosed to editors. A separate exploratory draft concerns joint target
selection; those endpoints are not introduced into this validation tutorial.
Sharing observations does not make the studies
independent replications or remove the need to assess overlapping contributions.

## Reuse

New teaching code is distributed under the MIT license in `LICENSE_TUTORIAL.txt`.
This license covers `run_tutorial.py` and `test_tutorial.py`. Generated artificial
observations may be reused with attribution. It grants no rights to the human
source workbooks or third-party material. Preserved archive files retain their
original provenance; data sharing requires actual authorization.
