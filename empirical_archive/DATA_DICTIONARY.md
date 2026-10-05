# Data and output dictionary

## Source structure

The program requires the unchanged four workbooks identified in `source_verification.csv`.
`participant` is a recorded identifier, not an independently verified natural person.
`trials.thisN` is the zero-based trial index; complete runs contain 0–79 in Experiment 1
and 0–47 in Experiment 2. Repeated runs are segmented at trial-index resets or changes
in the recorded date. Recorded `date` strings are not uniformly formatted timestamps.
`Age`, `Gender` and `frameRate` are recorded attributes, not new screening or calibration
measurements. `frameRate` is an estimated frame rate in Hz.

`corr_clicks`, `incorr_clicks` and `total_clicks` count correct, incorrect and total
selections. `all_clicks` is an eight-position selected/not-selected vector: positions
1–4 are targets and 5–8 are distractors. It is not a click-order record. Recovery is
`corr_clicks / 4`; response precision is `corr_clicks / total_clicks`. Image profile is
the number of unique basenames among the four target and four distractor image columns.

In Experiment 1, `n % 4 == 0` is the cluster-start normal reference, followed by full,
left and right field trials at lags 1, 2 and 3. `block` is `n // 16`, with values 0–4.
Its blur labels are 6, 8, 10, 12, 14 and contrast labels are −70, −30, +10, +50, +90;
these labels are confounded with fixed block order. `profile` is 1, 2, 4 or 8 unique
images. In Experiment 2, `schedule` in the Displays/Age outputs is the position index
`(n % 12) // 3`, from 0 to 3; IJIE uses the recorded `n_mods` label as the condition.
The `n_mods` labels are not verified counts of software states or transitions.
Experiment 2 has no local normal reference and its profiles occupy ordered blocks.

## Shared output fields

| Fields | Meaning |
| --- | --- |
| `experiment`, `feature` / `feature_family` | Dataset key: experiment 1/2 and blur/contrast. In the trial-permutation table, `feature` instead names the permuted predictor. |
| `labels` | Number of recorded identifiers contributing to that estimate. |
| `raw_rows`, `duplicate_extras`, `unique_rows` | Workbook rows, exact duplicate extras, and deduplicated rows. |
| `raw_repeat_identifiers`, `distinct_repeat_identifiers`, `unique_runs` | Apparent repetitions before deduplication, identifiers with more than one distinct run, and distinct runs. |
| `first_run_rows`, `non4_unique` | Rows in first complete runs; deduplicated rows with other than four selections. |
| `age_*`, `female`, `male` in `data_audit` | Aggregate distribution of recorded ages and gender labels in first-run samples; not a verified common cohort across files. |
| `fps_min`, `fps_max` | Minimum and maximum recorded frame-rate estimates. |
| `vector_mismatches` | Number of selection vectors failing the count identities. |
| `age_inconsistent_labels` | Identifiers with more than one recorded age across distinct runs. |
| `ci_low`, `ci_high`, `*_low`, `*_high` | Lower and upper 95% interval bounds, on the named estimate's scale. Methods differ by table as described below. |
| `converged`, `GEE_converged` | Whether the fitted GEE reported convergence. |
| `file`, `sha256`, `status` | Relative source path, expected/observed fingerprint and successful `MATCH`. |

Blank entries denote unavailable/not-applicable numeric values, not zeros. CSV rows
contain aggregate estimates, not individual records. `_private` intermediates generated
locally by the scripts are excluded from this distribution.

## IJIE estimates

`MAE` and `RMSE` are in correct-target units. `R2` is pooled out-of-fold (OOF) R-squared,
normally at trial level. The explicitly named participant-mean table averages both
observed and predicted trial values within identifier before computing all metrics.
The legacy comparison uses a separate aggregated target, with the same aggregate
target on both sides of that comparison; it is not the primary trial outcome.
`mean_test_participant_overlap` is the average fraction of test identifiers also found
in training, across legacy folds. `n` is the number of evaluated stress-test rows;
`held_level` specifies the held-out condition. `modeled_*` and `outcome_*` audit fields
describe the primary model's sample, mean and sample standard deviation.

`model` selects the fixed model; `feature_set` selects the fixed predictor subset.
Model settings are executable in `mot_repro_utils.py` and are not claimed to have been
preregistered. Default folds are five nonshuffled GroupKFold partitions by recorded
identifier. Preprocessing is fitted within training folds. Intervals resample 400
identifier clusters from fixed OOF predictions, seed 42; they do not refit the training
pipeline and do not quantify all model-selection or training uncertainty.

`intercept`, `slope` (or the `calibration_` prefixed names) fit observed trial scores to
OOF predictions. The intercept has correct-target units and the slope is dimensionless.
`predicted`, `fit` and the calibration band's bounds are in correct-target units; bands
are pointwise. `delta_MAE` is permuted minus unpermuted held-out MAE, not causal importance.
`fold` and `repeat` are zero-based. The row-permutation stream is keyed by
SeedSequence([42, experiment, blur=0/contrast=1, fold, feature index, repeat]); each of
five folds has eight repeats. The identifier-block stream inserts 99 after 42 and
uses block index instead of feature index. Age and gender move jointly in the demographic
block; frame rate moves between identifiers. `split_seed` is one of 17, 29, 43, 61, 89
for the separately refitted split sensitivities. Splits are not candidates selected
on reported test performance.
