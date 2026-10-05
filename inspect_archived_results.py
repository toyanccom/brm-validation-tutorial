#!/usr/bin/env python3
"""Inspect reported aggregate comparisons without raw records or model fitting.

This reader joins existing result rows. It does not recreate source observations,
cross-validation predictions, confidence intervals, or participant-level plots.
"""
from pathlib import Path
import argparse
import csv
import sys


def load(root, name):
    with (root / 'results/IJIE' / f'{name}.csv').open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def select(data, e, family, **values):
    matches = [r for r in data if int(r['experiment']) == e
               and r['feature'] == family
               and all(r[k] == v for k, v in values.items())]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one aggregate row: {(e, family, values)}')
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-root', type=Path,
                        default=Path(__file__).resolve().parent / 'empirical_archive',
                        help='Path to the unchanged archived analysis directory.')
    args = parser.parse_args()
    primary = load(args.analysis_root, 'primary_model_performance')
    means = load(args.analysis_root, 'participant_mean_metrics')
    legacy = load(args.analysis_root, 'legacy_validation_comparison')
    writer = csv.writer(sys.stdout, lineterminator='\n')
    writer.writerow(['dataset', 'legacy_row_R2', 'legacy_group_R2',
                     'legacy_row_overlap', 'legacy_group_overlap',
                     'Ridge_trial_MAE', 'Ridge_trial_RMSE', 'Ridge_trial_R2',
                     'Ridge_trial_R2_CI_low', 'Ridge_trial_R2_CI_high',
                     'Ridge_identifier_mean_R2'])
    for experiment in (1, 2):
        for feature in ('blur', 'contrast'):
            row = select(legacy, experiment, feature, validation='rowwise')
            grouped = select(legacy, experiment, feature, validation='participant_grouped')
            trial = select(primary, experiment, feature, model='Ridge')
            average = select(means, experiment, feature, target='participant_mean')
            if float(row['mean_test_participant_overlap']) != 1.0:
                raise ValueError('The archived row-wise example no longer has complete overlap.')
            if float(grouped['mean_test_participant_overlap']) != 0.0:
                raise ValueError('The archived grouped example no longer has zero overlap.')
            # Formatting is the only transformation of the supplied estimates.
            numeric = [row['R2'], grouped['R2'],
                       row['mean_test_participant_overlap'], grouped['mean_test_participant_overlap'],
                       trial['MAE'], trial['RMSE'], trial['R2'],
                       trial['R2_CI_low'], trial['R2_CI_high'], average['R2']]
            writer.writerow([f'Exp{experiment} {feature}'] + [f'{float(v):.3f}' for v in numeric])


if __name__ == '__main__':
    main()
