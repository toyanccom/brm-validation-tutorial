# Public synthetic data dictionary

The input dataset and all files in `outputs/` are entirely simulated. They contain
no participants, real MOT responses, real fitted parameters, or reconstructed
human distributions. They illustrate the validation and scoring workflow, and
cannot reproduce the empirical MOT estimates. Generator and estimator settings
are unchanged from version 1.0.0; version 1.0.1 adds public access to the files.

## Input observations

`outputs/synthetic_records.csv` contains 2,400 observations: 80 artificial groups,
30 trials per group, six conditions with five occurrences per group. Rows preserve
the generator's order. Seed 20261005 and the complete algorithm are in
`run_tutorial.py::generate_records`. Counts are teaching choices.

| Column | Meaning |
| --- | --- |
| `synthetic_group` | Artificial unit key `SIM000` through `SIM079`, used for withholding and whole-group resampling; excluded from model predictors |
| `trial` | Zero-based position 0 through 29 within an artificial group |
| `condition` | Artificial balanced condition `C0` through `C5`; no mapping to an empirical blur or contrast level |
| `unit_marker` | A repeated numeric value drawn independently of the outcome-generating intercept; 60 of 2,400 entries are missing by the fixed-seed teaching specification |
| `outcome` | An artificial integer count from 0 through 4, sampled from a binomial distribution; not a human tracking score |

The unit intercepts follow a specified normal distribution, numeric markers a
specified uniform distribution, and six equally spaced condition effects are
passed through a logistic link. None of these parameters was fitted to human
data. Only `condition` and `unit_marker` enter the predictive pipelines. Training
folds alone fit imputation, encoding, scaling, and estimators.

## Held-out predictions

`outputs/synthetic_predictions.csv` has 9,600 rows: one held-out prediction for
each of the 2,400 input observations in each of four model/scheme configurations.
Within a configuration, prediction row order matches the input table. The
generator's `trial` index can therefore be recovered by this correspondence; it
is not an additional predictor.

| Column | Meaning |
| --- | --- |
| `synthetic_group` | The artificial group belonging to the corresponding input observation |
| `observed` | The original artificial outcome for that observation |
| `predicted` | Held-out continuous prediction; no artificial clipping to the outcome range |
| `fold` | Validation fold recorded by the saved pipeline |
| `model` | `RandomForest`, `Ridge`, or `MeanBaseline` |
| `scheme` | `row` for five-fold shuffled row withholding, or `group` for five-fold GroupKFold |

The four configurations are RandomForest/row, RandomForest/group, Ridge/group,
and MeanBaseline/group. RandomForest uses 100 trees and minimum leaf size 5;
Ridge uses alpha 10. The MeanBaseline prediction uses the training fold's outcome
mean. Row withholding measures additional observations from represented units;
group withholding removes complete artificial units.

## Group means and validation audit

`outputs/synthetic_group_means.csv` has 320 rows, 80 artificial groups per
configuration. `observed` and `predicted` are paired within-group averages of
held-out outcomes and predictions before scoring; `model` and `scheme` identify
the configuration. These are not another model fit.

`outputs/synthetic_fold_audit.csv` has 20 rows, five folds per configuration.
It records `model`, `scheme`, `fold`, `training_rows`, `test_rows`,
`training_groups`, `test_groups`, and `test_group_overlap`. The overlap is the
proportion of test group keys represented in training. It is one for the tested
row split and zero for every tested group split.

## Summary results and provenance

`outputs/synthetic_metrics.csv` has eight rows: trial and group-mean scores for
each configuration. `data_origin` is `SIMULATED`; `n_scored` is the number of
rows or group averages scored. `MAE`, `RMSE`, and `R2` summarize the paired
held-out observations and predictions. `R2_CI_low` and `R2_CI_high` are
percentile bounds from 400 whole-group resamples with fitted predictions fixed.
They do not include refitting, split selection, tuning, or another population.
`mean_test_group_overlap` summarizes the fold audit. Negative R-squared values
are retained. Group-mean R-squared is computed after averaging pairs, never by
averaging fold R-squared.

`outputs/run_metadata.json` records settings, package versions, CSV hashes, and
the limited interpretation of conditional intervals. The PNG and PDF teaching
figures are derived from the synthetic metrics. Figure metadata may differ when
regenerated on another date; numeric CSV outputs matched the archived run at
relative and absolute tolerance 1e-12 on 7 October 2026. Five implementation
tests passed. `PUBLIC_DATA_VERIFICATION.json` records this check.

## Empirical case study

Files in `empirical_archive/results/IJIE/` are previously released aggregate
empirical summaries, not synthetic inputs. Their names retain historical journal
labels. Code and dictionaries are supplied in `empirical_archive/`; the original
individual workbooks and real individual prediction vectors are absent. Public
synthetic records cannot be substituted for those workbooks to refit empirical
estimates. See `DATA_ACCESS.md` for the distinct authorization status.
