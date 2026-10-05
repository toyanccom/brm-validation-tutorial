"""Meaningful checks of the teaching pipeline's validation and scoring claims."""
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from run_tutorial import (generate_records, validate, score_targets, FEATURES,
                          make_pipeline)


class TutorialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = generate_records(groups=20, trials=12)
        cls.paired, cls.folds, cls.fitted = validate(cls.frame)

    def test_generator_is_reproducible_and_bounded(self):
        pd.testing.assert_frame_equal(self.frame, generate_records(groups=20, trials=12))
        self.assertTrue(self.frame.outcome.between(0, 4).all())
        self.assertEqual(self.frame.synthetic_group.nunique(), 20)
        self.assertTrue(self.frame.synthetic_group.str.startswith("SIM").all())

    def test_whole_groups_are_withheld_and_each_row_is_predicted(self):
        self.assertTrue((self.folds.test_group_overlap == 0).all())
        self.assertTrue(np.isfinite(self.paired.predicted).all())
        for train, test, _ in self.fitted:
            self.assertFalse(set(self.frame.iloc[train].synthetic_group) &
                             set(self.frame.iloc[test].synthetic_group))
        self.assertTrue((self.paired.groupby("synthetic_group").fold.nunique() == 1).all())
        self.assertNotIn("synthetic_group", FEATURES)
        self.assertNotIn("trial", FEATURES)
        self.assertNotIn("outcome", FEATURES)

    def test_represented_group_question_has_measured_overlap(self):
        _, folds, _ = validate(self.frame, scheme="row")
        self.assertTrue((folds.test_group_overlap > 0).all())

    def test_test_values_cannot_change_fitted_preprocessing(self):
        train, test, original = self.fitted[0]
        perturbed = self.frame.copy()
        perturbed.loc[test, "unit_marker"] = 1e9
        perturbed.loc[test, "condition"] = "unseen_in_training"
        pipeline = make_pipeline("Ridge", 20261005)
        pipeline.fit(perturbed.iloc[train][FEATURES], perturbed.iloc[train].outcome)
        before = original.named_steps["preprocess"].named_transformers_["numeric"]
        after = pipeline.named_steps["preprocess"].named_transformers_["numeric"]
        np.testing.assert_allclose(before.named_steps["impute"].statistics_,
                                   after.named_steps["impute"].statistics_)
        np.testing.assert_allclose(before.named_steps["scale"].mean_,
                                   after.named_steps["scale"].mean_)
        np.testing.assert_allclose(original.predict(self.frame.iloc[train][FEATURES]),
                                   pipeline.predict(self.frame.iloc[train][FEATURES]))
        self.assertTrue(np.isfinite(pipeline.predict(perturbed.iloc[test][FEATURES])).all())

    def test_group_scores_average_predictions_before_scoring(self):
        # Unequal trial counts make weighting and group aggregation observable.
        paired = pd.DataFrame({"synthetic_group": ["A", "A", "A", "B", "C"],
                               "observed": [0., 2., 4., 1., 3.],
                               "predicted": [0., 0., 0., 2., 3.]})
        metrics, means = score_targets(paired)
        self.assertEqual(means.loc["A", "observed"], 2.)
        self.assertAlmostEqual(metrics["group_mean"]["R2"],
                               r2_score([2., 1., 3.], [0., 2., 3.]))
        self.assertNotAlmostEqual(metrics["trial"]["R2"],
                                  metrics["group_mean"]["R2"])


if __name__ == "__main__":
    unittest.main()
