"""Generate and analyze entirely simulated repeated records for a BRM tutorial.

Nothing in this file reads, reconstructs, or fits a model to human MOT records.
The seed and generator settings are fixed teaching choices, not estimates.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform

import numpy as np
import pandas as pd
import sklearn
import matplotlib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEFAULT_SEED = 20261005
CAT_FEATURES = ["condition"]
NUM_FEATURES = ["unit_marker"]
FEATURES = CAT_FEATURES + NUM_FEATURES


def generate_records(seed=DEFAULT_SEED, groups=80, trials=30):
    """Independent artificial units, each with six balanced trial conditions.

    A unit intercept affects its responses. An independently drawn repeated
    marker can serve as a fingerprint for represented units, but has no true
    predictive relationship with the intercept in the generating population.
    The marker is numeric, not the identifier used for withholding.
    """
    if groups < 10 or trials < 6 or trials % 6:
        raise ValueError("Need >=10 groups and a positive multiple of six trials.")
    rng = np.random.default_rng(seed)
    intercepts = rng.normal(0, 0.9, groups)
    markers = rng.uniform(-2, 2, groups)
    effects = np.linspace(-1.0, 1.0, 6)
    records = []
    for unit in range(groups):
        order = np.tile(np.arange(6), trials // 6)
        rng.shuffle(order)
        for trial, condition in enumerate(order):
            logit = 0.5 + intercepts[unit] + effects[condition]
            probability = 1 / (1 + np.exp(-logit))
            records.append({"synthetic_group": f"SIM{unit:03d}",
                            "trial": trial, "condition": f"C{condition}",
                            "unit_marker": markers[unit],
                            "outcome": int(rng.binomial(4, probability))})
    frame = pd.DataFrame(records)
    # A small fixed-seed missingness example makes the imputation step useful.
    missing = rng.choice(len(frame), size=len(frame) // 40, replace=False)
    frame.loc[missing, "unit_marker"] = np.nan
    return frame


def make_pipeline(model, seed):
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    preprocessing = ColumnTransformer([
        ("categorical", categorical, CAT_FEATURES),
        ("numeric", numeric, NUM_FEATURES),
    ])
    if model == "Ridge":
        estimator = Ridge(alpha=10.0)
    elif model == "RandomForest":
        estimator = RandomForestRegressor(
            n_estimators=100, min_samples_leaf=5, random_state=seed, n_jobs=1)
    else:
        raise ValueError(model)
    return Pipeline([("preprocess", preprocessing), ("model", estimator)])


def score(y, predictions):
    return {"MAE": float(mean_absolute_error(y, predictions)),
            "RMSE": float(mean_squared_error(y, predictions) ** 0.5),
            "R2": float(r2_score(y, predictions))}


def validate(frame, model="Ridge", scheme="group", seed=DEFAULT_SEED):
    """Fit a fresh preprocessing and model object separately inside each fold."""
    x = frame[FEATURES]
    y = frame["outcome"].to_numpy(float)
    groups = frame["synthetic_group"].to_numpy()
    if scheme == "group":
        splits = GroupKFold(n_splits=5).split(x, y, groups)
    elif scheme == "row":
        splits = KFold(n_splits=5, shuffle=True, random_state=seed).split(x, y)
    else:
        raise ValueError(scheme)
    predictions = np.full(len(frame), np.nan)
    assignment = np.full(len(frame), -1, int)
    visits = np.zeros(len(frame), int)
    folds, fitted = [], []
    for fold, (train, test) in enumerate(splits):
        represented = set(groups[train])
        test_groups = set(groups[test])
        overlap = len(represented & test_groups) / len(test_groups)
        if scheme == "group" and overlap:
            raise AssertionError("A withheld group is present in training.")
        if model == "MeanBaseline":
            pipeline = None
            predictions[test] = y[train].mean()
        else:
            pipeline = make_pipeline(model, seed + fold)
            pipeline.fit(x.iloc[train], y[train])
            predictions[test] = pipeline.predict(x.iloc[test])
        assignment[test] = fold
        visits[test] += 1
        folds.append({"model": model, "scheme": scheme, "fold": fold,
                      "training_rows": len(train), "test_rows": len(test),
                      "training_groups": len(represented),
                      "test_groups": len(test_groups),
                      "test_group_overlap": overlap})
        fitted.append((train, test, pipeline))
    if not np.all(visits == 1) or not np.isfinite(predictions).all():
        raise AssertionError("Every row must receive exactly one held-out prediction.")
    paired = pd.DataFrame({"synthetic_group": groups, "observed": y,
                           "predicted": predictions, "fold": assignment})
    return paired, pd.DataFrame(folds), fitted


def score_targets(paired):
    """Aggregate paired held-out outcomes and predictions before mean scoring.

    Group means receive equal weight. Trial metrics give equal weight to rows.
    These weights need not coincide in an application with unequal trial counts.
    """
    averages = paired.groupby("synthetic_group", sort=True)[
        ["observed", "predicted"]].mean()
    return {"trial": score(paired.observed, paired.predicted),
            "group_mean": score(averages.observed, averages.predicted)}, averages


def conditional_intervals(paired, repetitions=400, seed=DEFAULT_SEED + 1):
    """Resample whole groups and keep predictions fixed, without model refits."""
    rng = np.random.default_rng(seed)
    groups = np.array(sorted(paired.synthetic_group.unique()))
    locations = {g: np.flatnonzero(paired.synthetic_group.to_numpy() == g)
                 for g in groups}
    averages = paired.groupby("synthetic_group")[
        ["observed", "predicted"]].mean().reindex(groups)
    values = {"trial": [], "group_mean": []}
    for _ in range(repetitions):
        sampled = rng.integers(0, len(groups), len(groups))
        # Concatenation preserves multiplicity when a group is drawn twice.
        rows = np.concatenate([locations[groups[i]] for i in sampled])
        values["trial"].append(r2_score(paired.observed.iloc[rows],
                                       paired.predicted.iloc[rows]))
        means = averages.iloc[sampled]
        values["group_mean"].append(r2_score(means.observed, means.predicted))
    return {target: np.quantile(v, [0.025, 0.975]).tolist()
            for target, v in values.items()}


def run(output_dir, seed=DEFAULT_SEED, groups=80, trials=30):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = generate_records(seed, groups, trials)
    frame.to_csv(output / "synthetic_records.csv", index=False, float_format="%.12g")
    configurations = [("RandomForest", "row"), ("RandomForest", "group"),
                      ("Ridge", "group"), ("MeanBaseline", "group")]
    summaries, all_folds, all_predictions, all_means = [], [], [], []
    for model, scheme in configurations:
        paired, folds, _ = validate(frame, model, scheme, seed)
        metrics, means = score_targets(paired)
        intervals = conditional_intervals(paired, seed=seed + 1)
        for target, values in metrics.items():
            summaries.append({"data_origin": "SIMULATED", "model": model,
                              "scheme": scheme, "target": target,
                              "n_scored": len(frame) if target == "trial" else groups,
                              **values, "R2_CI_low": intervals[target][0],
                              "R2_CI_high": intervals[target][1],
                              "mean_test_group_overlap": folds.test_group_overlap.mean()})
        all_folds.append(folds)
        all_predictions.append(paired.assign(model=model, scheme=scheme))
        all_means.append(means.reset_index().assign(model=model, scheme=scheme))
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "synthetic_metrics.csv", index=False, float_format="%.12g")
    pd.concat(all_folds).to_csv(output / "synthetic_fold_audit.csv", index=False)
    pd.concat(all_predictions).to_csv(output / "synthetic_predictions.csv",
                                    index=False, float_format="%.12g")
    pd.concat(all_means).to_csv(output / "synthetic_group_means.csv", index=False,
                              float_format="%.12g")
    metadata = {
        "data_origin": "Entirely simulated; no human observations used",
        "seed": seed, "groups": groups, "trials_per_group": trials,
        "generator": {"intercept_mean": 0, "intercept_sd": 0.9,
                      "marker_distribution": "Uniform(-2,2), independent of intercept",
                      "condition_effects": np.linspace(-1, 1, 6).tolist(),
                      "logit_intercept": 0.5, "outcome": "Binomial(4, logistic(logit))",
                      "marker_missing_fraction": 0.025},
        "selection": "One fixed teaching specification; no tuning or search over seeds",
        "uncertainty": "400 whole-group percentile resamples, predictions fixed; no refits",
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "transfer": "Neither prospective nor external empirical validation",
    }
    metadata["output_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.glob("synthetic_*.csv"))}
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    make_figure(summary, output)
    print(summary[["model", "scheme", "target", "n_scored", "R2",
                   "mean_test_group_overlap"]].to_string(index=False))
    return summary


def make_figure(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.4), constrained_layout=True)
    settings = [[("RandomForest", "row", "trial"),
                 ("RandomForest", "group", "trial")],
                [("Ridge", "group", "trial"),
                 ("Ridge", "group", "group_mean")]]
    labels = [["Represented groups", "Withheld groups"],
              ["Individual trials", "Group averages"]]
    for ax, choices, ticklabels in zip(axes, settings, labels):
        rows = [summary[(summary.model == m) & (summary.scheme == s) &
                        (summary.target == t)].iloc[0] for m, s, t in choices]
        y = np.array([r.R2 for r in rows])
        ax.bar(np.arange(2), y, color=["#34658c", "#6c8373"], width=0.55)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xticks(np.arange(2), ticklabels)
        ax.set_ylabel("Held-out R²")
        for i, v in enumerate(y):
            ax.annotate(f"{v:.3f}", (i, v), xytext=(0, 6 if v >= 0 else -14),
                        textcoords="offset points", ha="center")
        ax.margins(y=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("A  Same model and target, different split", fontsize=8.5)
    axes[1].set_title("B  Same predictions, different target", fontsize=8.5)
    fig.suptitle("SIMULATED teaching example", fontweight="bold")
    fig.savefig(output / "synthetic_comparisons.png", dpi=300)
    fig.savefig(output / "synthetic_comparisons.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--groups", type=int, default=80)
    parser.add_argument("--trials-per-group", type=int, default=30)
    args = parser.parse_args()
    run(args.output_dir, args.seed, args.groups, args.trials_per_group)
