# Aggregate output inventory

| File | Rows | Purpose |
| --- | ---: | --- |
| `calibration.csv` | 4 | Trial-level Ridge calibration intercept and slope with fixed-OOF intervals. |
| `calibration_bands.csv` | 160 | Calibration line and pointwise fixed-OOF bootstrap bands; Figure 5. |
| `data_audit.csv` | 4 | Aggregate source counts, repeat-run distinctions, response coherence and age distribution. |
| `leave_level_grouped_stress.csv` | 18 | Condition extrapolation stress tests combined with held-out identifiers. |
| `legacy_validation_comparison.csv` | 8 | Matched legacy aggregate target: row-wise versus identifier-grouped validation; Figure 2. |
| `participant_mean_metrics.csv` | 4 | Ridge performance after separately averaging observed and OOF-predicted scores within identifier. |
| `primary_model_performance.csv` | 20 | Five models in each of four datasets, with fixed-OOF R-squared intervals; Figure 3. |
| `ridge_feature_sets.csv` | 32 | Eight fixed Ridge feature sets per dataset. |
| `ridge_feature_sets_with_intervals.csv` | 32 | The same feature sets with fixed-OOF R-squared intervals; Figure 6. |
| `ridge_group_permutation.csv` | 320 | Identifier-level demographic-block and frame-rate permutation diagnostics. |
| `ridge_permutation_importance.csv` | 960 | Trial-row permutation diagnostics, by feature, held-out fold and independent repeat; Figure 7. |
| `ridge_split_sensitivity.csv` | 20 | Ridge refitted under five alternative identifier-grouped fold assignments. |
| `sensitivity_analyses.csv` | 12 | Exactly-four-response, first-trial-baseline and schedule-exclusion sensitivities. |
| `source_verification.csv` | 4 | Exact SHA-256 fingerprints of the four required source workbooks. |
