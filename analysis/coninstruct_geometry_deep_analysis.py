from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


MODELS = ("Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B")
MODEL_SAFE = {model: model.replace("/", "__") for model in MODELS}
LAYERS = tuple(range(32))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deep three-point geometry analysis for existing ConInstruct "
            "org/new/conflict hidden states."
        )
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/coninstruct_geometry_deep"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_final_label(text: str, allowed: Iterable[int]) -> int:
    allowed_set = set(allowed)
    if not isinstance(text, str) or not text.strip():
        return -1

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    answer_pattern = re.compile(
        r"(?:answer|result)\D*(-?\d+)\D*$",
        flags=re.IGNORECASE,
    )
    for line in reversed(lines[-5:]):
        match = answer_pattern.search(line)
        if match and int(match.group(1)) in allowed_set:
            return int(match.group(1))

    for line in reversed(lines[-3:]):
        match = re.search(r"(-?\d+)\D*$", line)
        if match and int(match.group(1)) in allowed_set:
            return int(match.group(1))
    return -1


def infer_prompt_order(
    instruction: str,
    org_constraint: str,
    new_constraint: str,
    configured_new_position: str,
) -> str:
    org_pos = instruction.find(org_constraint)
    new_pos = instruction.find(new_constraint)
    if org_pos >= 0 and new_pos >= 0:
        return "org_new" if org_pos < new_pos else "new_org"
    # The expanded instruction can merge or paraphrase the stored org constraint,
    # so exact substring matching often fails. The extraction metadata records
    # the deterministic injection position and is the reliable fallback.
    return {
        "after": "org_new",
        "before": "new_org",
    }.get(configured_new_position, "unknown")


def load_metadata_frame(path: Path) -> tuple[dict, pd.DataFrame]:
    metadata = load_json(path)
    frame = pd.DataFrame(metadata["examples"])
    frame["conflict_type_idx"] = frame["conflict_type_idx"].astype(int)
    frame["sample_id"] = frame["sample_id"].astype(int)
    frame["key"] = list(zip(frame["conflict_type_idx"], frame["sample_id"]))
    return metadata, frame


def load_pair_labels(root: Path, model: str) -> pd.DataFrame:
    rows: list[dict] = []
    eval_root = root / "evaluation_outputs" / "conflict_resolution" / model
    for type_dir in sorted(
        (path for path in eval_root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    ):
        conflict_type_idx = int(type_dir.name)
        for path in sorted(
            (p for p in type_dir.glob("*.json") if p.stem.isdigit()),
            key=lambda p: int(p.stem),
        ):
            data = load_json(path)
            entries = [value for key, value in data.items() if key != "llm_response"]
            if not entries:
                continue
            item = entries[0]
            raw_label = parse_final_label(item.get("evaluation_result", ""), (-1, 1, 2))
            order = item.get("instruction_order", "unknown")
            if order == "org_new":
                label = {1: 2, 2: 1}.get(raw_label, raw_label)
            else:
                label = raw_label
            rows.append(
                {
                    "conflict_type_idx": conflict_type_idx,
                    "sample_id": int(path.stem),
                    "pair_raw_label": raw_label,
                    "pair_label": label,
                    "pair_label_name": {1: "new", 2: "org", -1: "neither"}.get(
                        label, "unknown"
                    ),
                    "judge_instruction_order": order,
                }
            )
    return pd.DataFrame(rows)


def load_behavior_labels(root: Path, model: str) -> pd.DataFrame:
    rows: list[dict] = []
    eval_root = root / "evaluation_outputs" / "conflict_resolution" / "behavior" / model
    names = {
        1: "direct_no_ack",
        2: "asks_clarification",
        3: "self_resolves",
        4: "other",
        -1: "parse_failed",
    }
    for type_dir in sorted(
        (path for path in eval_root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    ):
        conflict_type_idx = int(type_dir.name)
        for path in sorted(
            (p for p in type_dir.glob("*.json") if p.stem.isdigit()),
            key=lambda p: int(p.stem),
        ):
            data = load_json(path)
            label = parse_final_label(data.get("evaluation_result", ""), (1, 2, 3, 4))
            rows.append(
                {
                    "conflict_type_idx": conflict_type_idx,
                    "sample_id": int(path.stem),
                    "behavior_label": label,
                    "behavior_label_name": names.get(label, "unknown"),
                }
            )
    return pd.DataFrame(rows)


def rows_for_type(frame: pd.DataFrame, conflict_type_idx: int, variant: str | None = None):
    subset = frame[frame["conflict_type_idx"] == conflict_type_idx]
    if variant is not None:
        subset = subset[subset["variant"] == variant]
    return subset.reset_index(drop=True)


def safe_norm(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    return np.maximum(np.linalg.norm(x, axis=axis), eps)


def compute_geometry(
    root: Path,
    model: str,
    conflict_examples: pd.DataFrame,
    org_new_examples: pd.DataFrame,
    configured_new_position: str,
) -> pd.DataFrame:
    hidden_root = (
        root / "outputs" / "coninstruct_hidden_states" / MODEL_SAFE[model]
    )
    conflict_root = hidden_root / "conflict_resolution"
    org_new_root = hidden_root / "conflict_resolution_org_new"
    output_frames: list[pd.DataFrame] = []

    for conflict_type_idx in sorted(conflict_examples["conflict_type_idx"].unique()):
        ctype = f"conflict_type_{int(conflict_type_idx)}"
        conflict_rows = rows_for_type(conflict_examples, conflict_type_idx)
        org_rows = rows_for_type(org_new_examples, conflict_type_idx, "org")
        new_rows = rows_for_type(org_new_examples, conflict_type_idx, "new")

        conflict_index = {key: idx for idx, key in enumerate(conflict_rows["key"])}
        org_index = {key: idx for idx, key in enumerate(org_rows["key"])}
        new_index = {key: idx for idx, key in enumerate(new_rows["key"])}
        keys = sorted(set(conflict_index) & set(org_index) & set(new_index))
        ci = np.array([conflict_index[key] for key in keys])
        oi = np.array([org_index[key] for key in keys])
        ni = np.array([new_index[key] for key in keys])

        metadata = conflict_rows.set_index("key").loc[keys].reset_index()
        metadata["prompt_order"] = [
            infer_prompt_order(
                instruction,
                org_constraint,
                new_constraint,
                configured_new_position,
            )
            for instruction, org_constraint, new_constraint in zip(
                metadata["instruction"],
                metadata["org_constraint"],
                metadata["new_constraint"],
            )
        ]
        metadata["last_prompt_constraint"] = metadata["prompt_order"].map(
            {"org_new": "new", "new_org": "org"}
        )
        metadata["org_constraint_chars"] = metadata["org_constraint"].str.len()
        metadata["new_constraint_chars"] = metadata["new_constraint"].str.len()

        for layer in LAYERS:
            hc = np.load(conflict_root / ctype / f"layer_{layer}.npy")[ci].astype(
                np.float64
            )
            ho = np.load(org_new_root / ctype / "org" / f"layer_{layer}.npy")[
                oi
            ].astype(np.float64)
            hn = np.load(org_new_root / ctype / "new" / f"layer_{layer}.npy")[
                ni
            ].astype(np.float64)

            endpoint = hn - ho
            endpoint_norm = safe_norm(endpoint)
            unit = endpoint / endpoint_norm[:, None]
            from_org = hc - ho
            position_t = np.einsum("ij,ij->i", from_org, endpoint) / (
                endpoint_norm**2
            )
            closest_on_line = ho + position_t[:, None] * endpoint
            orthogonal = hc - closest_on_line
            orthogonal_norm = safe_norm(orthogonal)

            midpoint = (ho + hn) / 2.0
            from_midpoint = hc - midpoint
            midpoint_norm = safe_norm(from_midpoint)
            signed_projection = np.einsum("ij,ij->i", from_midpoint, unit)

            distance_org = safe_norm(hc - ho)
            distance_new = safe_norm(hc - hn)
            hidden_scale = (safe_norm(ho) + safe_norm(hn) + safe_norm(hc)) / 3.0

            frame = metadata[
                [
                    "conflict_type_idx",
                    "sample_id",
                    "prompt_order",
                    "last_prompt_constraint",
                    "org_constraint_chars",
                    "new_constraint_chars",
                ]
            ].copy()
            frame["model_id"] = model
            frame["layer"] = layer
            frame["depth"] = layer / (len(LAYERS) - 1)
            frame["position_t"] = position_t
            frame["signed_margin"] = 2.0 * position_t - 1.0
            frame["nearest_side"] = np.where(position_t > 0.5, "new", "org")
            frame["outside_segment"] = (position_t < 0.0) | (position_t > 1.0)
            frame["offaxis_ratio"] = orthogonal_norm / endpoint_norm
            frame["log_offaxis_ratio"] = np.log1p(frame["offaxis_ratio"])
            frame["offaxis_fraction"] = orthogonal_norm / midpoint_norm
            frame["midpoint_distance_ratio"] = midpoint_norm / endpoint_norm
            frame["log_midpoint_distance_ratio"] = np.log1p(
                frame["midpoint_distance_ratio"]
            )
            frame["distance_org_ratio"] = distance_org / endpoint_norm
            frame["distance_new_ratio"] = distance_new / endpoint_norm
            frame["endpoint_separation"] = endpoint_norm
            frame["endpoint_relative_hidden"] = endpoint_norm / hidden_scale
            frame["signed_projection"] = signed_projection
            output_frames.append(frame)

    return pd.concat(output_frames, ignore_index=True)


def stage_for_layer(layer: int) -> str:
    if layer <= 9:
        return "early_00_09"
    if layer <= 21:
        return "middle_10_21"
    return "late_22_31"


def bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan, math.nan
    draws = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def build_sample_stage_frame(scores: pd.DataFrame) -> pd.DataFrame:
    frame = scores.copy()
    frame["stage"] = frame["layer"].map(stage_for_layer)
    frame["chosen_side"] = frame["pair_label_name"]
    frame["chosen_is_last"] = frame["chosen_side"] == frame["last_prompt_constraint"]
    frame["oriented_position"] = np.where(
        frame["pair_label_name"] == "new",
        frame["position_t"],
        1.0 - frame["position_t"],
    )
    frame["oriented_margin"] = 2.0 * frame["oriented_position"] - 1.0
    valid_labels = frame["pair_label_name"].isin(["org", "new"])
    frame.loc[~valid_labels, ["oriented_position", "oriented_margin"]] = np.nan

    aggregations = {
        "position_t": "mean",
        "signed_margin": "mean",
        "offaxis_ratio": "mean",
        "log_offaxis_ratio": "mean",
        "offaxis_fraction": "mean",
        "midpoint_distance_ratio": "mean",
        "log_midpoint_distance_ratio": "mean",
        "endpoint_relative_hidden": "mean",
        "oriented_position": "mean",
        "oriented_margin": "mean",
    }
    stage_frame = (
        frame.groupby(
            [
                "model_id",
                "conflict_type_idx",
                "sample_id",
                "stage",
                "prompt_order",
                "last_prompt_constraint",
                "pair_label",
                "pair_label_name",
                "behavior_label",
                "behavior_label_name",
            ],
            dropna=False,
        )
        .agg(aggregations)
        .reset_index()
    )
    return stage_frame


def build_trajectory_features(scores: pd.DataFrame) -> pd.DataFrame:
    id_columns = [
        "model_id",
        "conflict_type_idx",
        "sample_id",
        "prompt_order",
        "last_prompt_constraint",
        "pair_label",
        "pair_label_name",
        "behavior_label",
        "behavior_label_name",
    ]
    metrics = [
        "position_t",
        "signed_margin",
        "log_offaxis_ratio",
        "offaxis_fraction",
        "midpoint_distance_ratio",
        "log_midpoint_distance_ratio",
        "endpoint_relative_hidden",
    ]
    base = scores[id_columns].drop_duplicates(
        ["model_id", "conflict_type_idx", "sample_id"]
    )
    wide_parts = []
    for metric in metrics:
        part = scores.pivot_table(
            index=["model_id", "conflict_type_idx", "sample_id"],
            columns="layer",
            values=metric,
            aggfunc="first",
        )
        part.columns = [f"{metric}_layer_{int(layer):02d}" for layer in part.columns]
        wide_parts.append(part)
    wide = pd.concat(wide_parts, axis=1).reset_index()
    features = base.merge(
        wide,
        on=["model_id", "conflict_type_idx", "sample_id"],
        how="inner",
    )

    grouped = scores.sort_values("layer").groupby(
        ["model_id", "conflict_type_idx", "sample_id"], sort=False
    )
    trajectory_rows = []
    for key, group in grouped:
        group = group.sort_values("layer")
        t = group["position_t"].to_numpy()
        residual = group["log_offaxis_ratio"].to_numpy()
        delta_t = np.diff(t)
        delta_residual = np.diff(residual)
        trajectory_rows.append(
            {
                "model_id": key[0],
                "conflict_type_idx": key[1],
                "sample_id": key[2],
                "t_early_mean": t[:10].mean(),
                "t_middle_mean": t[10:22].mean(),
                "t_late_mean": t[22:].mean(),
                "t_late_minus_middle": t[22:].mean() - t[10:22].mean(),
                "residual_early_mean": residual[:10].mean(),
                "residual_middle_mean": residual[10:22].mean(),
                "residual_late_mean": residual[22:].mean(),
                "residual_late_minus_middle": residual[22:].mean()
                - residual[10:22].mean(),
                "max_abs_t_step": np.max(np.abs(delta_t)),
                "max_abs_t_step_layer": int(np.argmax(np.abs(delta_t)) + 1),
                "max_abs_residual_step": np.max(np.abs(delta_residual)),
                "max_abs_residual_step_layer": int(
                    np.argmax(np.abs(delta_residual)) + 1
                ),
                "residual_peak_layer": int(np.argmax(residual)),
                "residual_peak_value": float(np.max(residual)),
            }
        )
    return features.merge(
        pd.DataFrame(trajectory_rows),
        on=["model_id", "conflict_type_idx", "sample_id"],
        how="left",
    )


def numeric_feature_columns(frame: pd.DataFrame, prefixes: tuple[str, ...]) -> list[str]:
    return [
        column
        for column in frame.columns
        if any(column.startswith(prefix) for prefix in prefixes)
    ]


def grouped_cv(
    frame: pd.DataFrame,
    target: str,
    numeric_columns: list[str],
    categorical_columns: list[str],
    random_state: int,
    n_splits: int = 5,
) -> pd.DataFrame:
    valid = frame.dropna(subset=[target]).copy()
    y = valid[target].astype(int).to_numpy()
    groups = valid["sample_id"].to_numpy()
    X = valid[numeric_columns + categorical_columns]

    transformers = []
    if numeric_columns:
        transformers.append(("num", StandardScaler(), numeric_columns))
    if categorical_columns:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            )
        )
    preprocessor = ColumnTransformer(transformers)
    estimator = make_pipeline(
        preprocessor,
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=random_state,
        ),
    )
    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    rows = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
        estimator.fit(X.iloc[train_idx], y[train_idx])
        prediction = estimator.predict(X.iloc[test_idx])
        probability = estimator.predict_proba(X.iloc[test_idx])[:, 1]
        rows.append(
            {
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "balanced_accuracy": balanced_accuracy_score(y[test_idx], prediction),
                "roc_auc": roc_auc_score(y[test_idx], probability),
            }
        )
    return pd.DataFrame(rows)


def leave_type_out(
    frame: pd.DataFrame,
    target: str,
    numeric_columns: list[str],
    categorical_columns: list[str],
    random_state: int,
) -> pd.DataFrame:
    valid = frame.dropna(subset=[target]).copy()
    rows = []
    for held_out_type in sorted(valid["conflict_type_idx"].unique()):
        train = valid[valid["conflict_type_idx"] != held_out_type]
        test = valid[valid["conflict_type_idx"] == held_out_type]
        if train[target].nunique() < 2 or test[target].nunique() < 2:
            continue

        transformers = []
        if numeric_columns:
            transformers.append(("num", StandardScaler(), numeric_columns))
        if categorical_columns:
            transformers.append(
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    categorical_columns,
                )
            )
        estimator = make_pipeline(
            ColumnTransformer(transformers),
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=random_state,
            ),
        )
        estimator.fit(train[numeric_columns + categorical_columns], train[target])
        prediction = estimator.predict(test[numeric_columns + categorical_columns])
        probability = estimator.predict_proba(
            test[numeric_columns + categorical_columns]
        )[:, 1]
        rows.append(
            {
                "held_out_type": held_out_type,
                "n_train": len(train),
                "n_test": len(test),
                "balanced_accuracy": balanced_accuracy_score(
                    test[target], prediction
                ),
                "roc_auc": roc_auc_score(test[target], probability),
            }
        )
    return pd.DataFrame(rows)


def classifier_analyses(
    trajectory: pd.DataFrame,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    leave_type_frames = []

    feature_groups = {
        "conflict_type_only": ([], ["conflict_type_idx"]),
        "axis_trajectory": (
            numeric_feature_columns(trajectory, ("position_t_layer_",)),
            [],
        ),
        "offaxis_trajectory": (
            numeric_feature_columns(
                trajectory, ("log_offaxis_ratio_layer_", "offaxis_fraction_layer_")
            ),
            [],
        ),
        "full_geometry": (
            numeric_feature_columns(
                trajectory,
                (
                    "position_t_layer_",
                    "log_offaxis_ratio_layer_",
                    "offaxis_fraction_layer_",
                    "log_midpoint_distance_ratio_layer_",
                    "endpoint_relative_hidden_layer_",
                ),
            ),
            [],
        ),
        "full_geometry_plus_type": (
            numeric_feature_columns(
                trajectory,
                (
                    "position_t_layer_",
                    "log_offaxis_ratio_layer_",
                    "offaxis_fraction_layer_",
                    "log_midpoint_distance_ratio_layer_",
                    "endpoint_relative_hidden_layer_",
                ),
            ),
            ["conflict_type_idx"],
        ),
    }

    for model, model_frame in trajectory.groupby("model_id"):
        pair_frame = model_frame[
            model_frame["pair_label_name"].isin(["org", "new"])
        ].copy()
        pair_frame["target_pair_new"] = (
            pair_frame["pair_label_name"] == "new"
        ).astype(int)
        behavior_frame = model_frame[
            model_frame["behavior_label"].isin([1, 3])
        ].copy()
        behavior_frame["target_behavior_self_resolves"] = (
            behavior_frame["behavior_label"] == 3
        ).astype(int)

        for target_name, target_frame, target_column in [
            ("pair_choice", pair_frame, "target_pair_new"),
            (
                "behavior_1_vs_3",
                behavior_frame,
                "target_behavior_self_resolves",
            ),
        ]:
            for group_name, (numeric, categorical) in feature_groups.items():
                cv_results = grouped_cv(
                    target_frame,
                    target_column,
                    numeric,
                    categorical,
                    random_state,
                )
                summary_rows.append(
                    {
                        "model_id": model,
                        "target": target_name,
                        "feature_group": group_name,
                        "n_samples": len(target_frame),
                        "positive_rate": target_frame[target_column].mean(),
                        "balanced_accuracy_mean": cv_results[
                            "balanced_accuracy"
                        ].mean(),
                        "balanced_accuracy_std": cv_results[
                            "balanced_accuracy"
                        ].std(ddof=1),
                        "roc_auc_mean": cv_results["roc_auc"].mean(),
                        "roc_auc_std": cv_results["roc_auc"].std(ddof=1),
                    }
                )

                if group_name in {
                    "axis_trajectory",
                    "full_geometry",
                    "full_geometry_plus_type",
                }:
                    loto = leave_type_out(
                        target_frame,
                        target_column,
                        numeric,
                        [
                            column
                            for column in categorical
                            if column != "conflict_type_idx"
                        ],
                        random_state,
                    )
                    loto["model_id"] = model
                    loto["target"] = target_name
                    loto["feature_group"] = group_name
                    leave_type_frames.append(loto)

    return pd.DataFrame(summary_rows), pd.concat(leave_type_frames, ignore_index=True)


def cross_model_correlations(scores: pd.DataFrame) -> pd.DataFrame:
    four = scores[scores["model_id"] == MODELS[0]]
    nine = scores[scores["model_id"] == MODELS[1]]
    merged = four.merge(
        nine,
        on=["conflict_type_idx", "sample_id", "layer"],
        suffixes=("_4b", "_9b"),
    )
    rows = []
    for layer, group in merged.groupby("layer"):
        for metric in [
            "position_t",
            "signed_margin",
            "log_offaxis_ratio",
            "offaxis_fraction",
            "endpoint_relative_hidden",
        ]:
            correlation, p_value = spearmanr(
                group[f"{metric}_4b"],
                group[f"{metric}_9b"],
            )
            rows.append(
                {
                    "layer": layer,
                    "metric": metric,
                    "spearman_r": correlation,
                    "p_value": p_value,
                    "n": len(group),
                }
            )
    return pd.DataFrame(rows)


def cross_model_label_agreement(trajectory: pd.DataFrame) -> pd.DataFrame:
    four = trajectory[trajectory["model_id"] == MODELS[0]][
        [
            "conflict_type_idx",
            "sample_id",
            "pair_label_name",
            "behavior_label",
        ]
    ]
    nine = trajectory[trajectory["model_id"] == MODELS[1]][
        [
            "conflict_type_idx",
            "sample_id",
            "pair_label_name",
            "behavior_label",
        ]
    ]
    merged = four.merge(
        nine,
        on=["conflict_type_idx", "sample_id"],
        suffixes=("_4b", "_9b"),
    )
    valid_pair = merged[
        merged["pair_label_name_4b"].isin(["org", "new"])
        & merged["pair_label_name_9b"].isin(["org", "new"])
    ]
    valid_behavior = merged[
        merged["behavior_label_4b"].isin([1, 2, 3, 4])
        & merged["behavior_label_9b"].isin([1, 2, 3, 4])
    ]
    return pd.DataFrame(
        [
            {
                "comparison": "pair_org_vs_new",
                "n": len(valid_pair),
                "agreement": (
                    valid_pair["pair_label_name_4b"]
                    == valid_pair["pair_label_name_9b"]
                ).mean(),
            },
            {
                "comparison": "behavior_full",
                "n": len(valid_behavior),
                "agreement": (
                    valid_behavior["behavior_label_4b"]
                    == valid_behavior["behavior_label_9b"]
                ).mean(),
            },
        ]
    )


def stage_effects(
    stage_frame: pd.DataFrame,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> pd.DataFrame:
    rows = []
    metrics = [
        "position_t",
        "oriented_position",
        "log_offaxis_ratio",
        "offaxis_fraction",
        "log_midpoint_distance_ratio",
    ]
    for (model, stage), group in stage_frame.groupby(["model_id", "stage"]):
        for metric in metrics:
            for label_name, label_group in group.groupby("behavior_label_name"):
                if label_name not in {"direct_no_ack", "self_resolves"}:
                    continue
                mean, low, high = bootstrap_mean_ci(
                    label_group[metric].to_numpy(),
                    rng,
                    n_bootstrap,
                )
                rows.append(
                    {
                        "model_id": model,
                        "stage": stage,
                        "metric": metric,
                        "group": label_name,
                        "n": len(label_group),
                        "mean": mean,
                        "ci_low": low,
                        "ci_high": high,
                    }
                )

            direct = group[group["behavior_label"] == 1][metric].dropna()
            resolves = group[group["behavior_label"] == 3][metric].dropna()
            if len(direct) and len(resolves):
                statistic, p_value = mannwhitneyu(
                    direct,
                    resolves,
                    alternative="two-sided",
                )
                rows.append(
                    {
                        "model_id": model,
                        "stage": stage,
                        "metric": metric,
                        "group": "direct_vs_self_resolves_test",
                        "n": len(direct) + len(resolves),
                        "mean": math.nan,
                        "ci_low": math.nan,
                        "ci_high": math.nan,
                        "mannwhitney_u": statistic,
                        "p_value": p_value,
                    }
                )
    return pd.DataFrame(rows)


def benjamini_hochberg(p_values: pd.Series) -> np.ndarray:
    values = p_values.to_numpy(dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def controlled_behavior_effects(stage_frame: pd.DataFrame) -> pd.DataFrame:
    """Compare behavior 1 vs 3 after removing type and chosen-side means."""
    metrics = [
        "position_t",
        "log_offaxis_ratio",
        "offaxis_fraction",
        "log_midpoint_distance_ratio",
    ]
    valid = stage_frame[
        stage_frame["behavior_label"].isin([1, 3])
        & stage_frame["pair_label_name"].isin(["org", "new"])
    ].copy()
    rows = []
    for metric in metrics:
        centered_column = f"{metric}_centered"
        valid[centered_column] = valid[metric] - valid.groupby(
            [
                "model_id",
                "stage",
                "conflict_type_idx",
                "pair_label_name",
            ]
        )[metric].transform("mean")
        for (model, stage), group in valid.groupby(["model_id", "stage"]):
            direct = group[group["behavior_label"] == 1][centered_column]
            resolves = group[group["behavior_label"] == 3][centered_column]
            statistic, p_value = mannwhitneyu(
                direct,
                resolves,
                alternative="two-sided",
            )
            rows.append(
                {
                    "model_id": model,
                    "stage": stage,
                    "metric": metric,
                    "n_direct": len(direct),
                    "n_self_resolves": len(resolves),
                    "direct_mean_centered": direct.mean(),
                    "self_resolves_mean_centered": resolves.mean(),
                    "self_resolves_minus_direct": resolves.mean() - direct.mean(),
                    "mannwhitney_u": statistic,
                    "p_value": p_value,
                }
            )
    result = pd.DataFrame(rows)
    result["p_value_bh"] = benjamini_hochberg(result["p_value"])
    return result


def within_type_stage_auc(stage_frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ["position_t", "log_offaxis_ratio", "offaxis_fraction"]
    valid = stage_frame[stage_frame["pair_label_name"].isin(["org", "new"])].copy()
    rows = []
    for (model, stage, conflict_type_idx), group in valid.groupby(
        ["model_id", "stage", "conflict_type_idx"]
    ):
        target = (group["pair_label_name"] == "new").astype(int)
        if target.nunique() < 2:
            continue
        for metric in metrics:
            rows.append(
                {
                    "model_id": model,
                    "stage": stage,
                    "conflict_type_idx": conflict_type_idx,
                    "metric": metric,
                    "n": len(group),
                    "new_rate": target.mean(),
                    "roc_auc": roc_auc_score(target, group[metric]),
                }
            )
    return pd.DataFrame(rows)


def within_type_layer_auc(scores: pd.DataFrame) -> pd.DataFrame:
    metrics = ["position_t", "log_offaxis_ratio", "offaxis_fraction"]
    valid = scores[scores["pair_label_name"].isin(["org", "new"])].copy()
    rows = []
    for (model, layer, conflict_type_idx), group in valid.groupby(
        ["model_id", "layer", "conflict_type_idx"]
    ):
        target = (group["pair_label_name"] == "new").astype(int)
        if target.nunique() < 2:
            continue
        for metric in metrics:
            rows.append(
                {
                    "model_id": model,
                    "layer": layer,
                    "conflict_type_idx": conflict_type_idx,
                    "metric": metric,
                    "n": len(group),
                    "roc_auc": roc_auc_score(target, group[metric]),
                }
            )
    return pd.DataFrame(rows)


def geometry_extrema(layer_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "mean_position_t",
        "mean_log_offaxis_ratio",
        "fraction_outside_segment",
        "mean_endpoint_relative_hidden",
    ]
    for model, group in layer_frame.groupby("model_id"):
        for metric in metrics:
            minimum = group.loc[group[metric].idxmin()]
            maximum = group.loc[group[metric].idxmax()]
            rows.append(
                {
                    "model_id": model,
                    "metric": metric,
                    "min_layer": int(minimum["layer"]),
                    "min_value": minimum[metric],
                    "max_layer": int(maximum["layer"]),
                    "max_value": maximum[metric],
                }
            )
    return pd.DataFrame(rows)


def prompt_order_summary(trajectory: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in trajectory.groupby("model_id"):
        valid = group[group["pair_label_name"].isin(["org", "new"])].copy()
        valid["chosen_is_last"] = (
            valid["pair_label_name"] == valid["last_prompt_constraint"]
        )
        rows.append(
            {
                "model_id": model,
                "n_pair_labelled": len(valid),
                "prompt_order_unknown": int((valid["prompt_order"] == "unknown").sum()),
                "last_constraint_choice_rate": valid["chosen_is_last"].mean(),
                "new_choice_rate_when_new_last": (
                    valid.loc[
                        valid["last_prompt_constraint"] == "new",
                        "pair_label_name",
                    ]
                    == "new"
                ).mean(),
                "new_choice_rate_when_org_last": (
                    valid.loc[
                        valid["last_prompt_constraint"] == "org",
                        "pair_label_name",
                    ]
                    == "new"
                ).mean(),
            }
        )
    return pd.DataFrame(rows)


def layer_summary(scores: pd.DataFrame) -> pd.DataFrame:
    return (
        scores.groupby(["model_id", "layer"])
        .agg(
            n=("sample_id", "count"),
            mean_position_t=("position_t", "mean"),
            median_position_t=("position_t", "median"),
            fraction_new_side=("position_t", lambda values: (values > 0.5).mean()),
            fraction_outside_segment=("outside_segment", "mean"),
            mean_log_offaxis_ratio=("log_offaxis_ratio", "mean"),
            median_offaxis_ratio=("offaxis_ratio", "median"),
            mean_offaxis_fraction=("offaxis_fraction", "mean"),
            mean_midpoint_distance_ratio=("midpoint_distance_ratio", "mean"),
            median_midpoint_distance_ratio=("midpoint_distance_ratio", "median"),
            mean_log_midpoint_distance_ratio=(
                "log_midpoint_distance_ratio",
                "mean",
            ),
            mean_endpoint_relative_hidden=("endpoint_relative_hidden", "mean"),
        )
        .reset_index()
    )


def type_stage_summary(stage_frame: pd.DataFrame) -> pd.DataFrame:
    return (
        stage_frame.groupby(["model_id", "conflict_type_idx", "stage"])
        .agg(
            n=("sample_id", "count"),
            mean_position_t=("position_t", "mean"),
            fraction_new_side=("position_t", lambda values: (values > 0.5).mean()),
            mean_log_offaxis_ratio=("log_offaxis_ratio", "mean"),
            mean_offaxis_fraction=("offaxis_fraction", "mean"),
            mean_oriented_position=("oriented_position", "mean"),
        )
        .reset_index()
    )


def type_outcome_summary(trajectory: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, conflict_type_idx), group in trajectory.groupby(
        ["model_id", "conflict_type_idx"]
    ):
        pair_valid = group[group["pair_label_name"].isin(["org", "new"])]
        behavior_valid = group[group["behavior_label"].isin([1, 2, 3, 4])]
        rows.append(
            {
                "model_id": model,
                "conflict_type_idx": conflict_type_idx,
                "n": len(group),
                "new_choice_rate": (
                    pair_valid["pair_label_name"] == "new"
                ).mean(),
                "neither_rate": (group["pair_label_name"] == "neither").mean(),
                "self_resolves_rate": (
                    behavior_valid["behavior_label"] == 3
                ).mean(),
                "asks_clarification_rate": (
                    behavior_valid["behavior_label"] == 2
                ).mean(),
            }
        )
    return pd.DataFrame(rows)


def make_plots(
    output_dir: Path,
    layer_frame: pd.DataFrame,
    stage_frame: pd.DataFrame,
    cross_model: pd.DataFrame,
    classifier_summary: pd.DataFrame,
) -> None:
    colors = {MODELS[0]: "#1f77b4", MODELS[1]: "#d62728"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for model, group in layer_frame.groupby("model_id"):
        axes[0].plot(
            group["layer"],
            group["mean_position_t"],
            marker="o",
            markersize=3,
            label=model,
            color=colors[model],
        )
        axes[1].plot(
            group["layer"],
            group["mean_log_offaxis_ratio"],
            marker="o",
            markersize=3,
            label=model,
            color=colors[model],
        )
        axes[2].plot(
            group["layer"],
            group["fraction_outside_segment"],
            marker="o",
            markersize=3,
            label=model,
            color=colors[model],
        )
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Conflict position on org→new line")
    axes[0].set_ylabel("mean t (org=0, new=1)")
    axes[1].set_title("Orthogonal displacement")
    axes[1].set_ylabel("mean log(1 + off-axis / endpoint)")
    axes[2].set_title("Conflict outside endpoint segment")
    axes[2].set_ylabel("fraction t<0 or t>1")
    for axis in axes:
        axis.set_xlabel("layer")
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "overall_layer_geometry.png", dpi=220)
    plt.close(fig)

    behavior = stage_frame[
        stage_frame["behavior_label_name"].isin(["direct_no_ack", "self_resolves"])
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for model, model_group in behavior.groupby("model_id"):
        summary = (
            model_group.groupby(["stage", "behavior_label_name"])
            .agg(
                residual=("log_offaxis_ratio", "mean"),
                oriented=("oriented_position", "mean"),
            )
            .reset_index()
        )
        stages = ["early_00_09", "middle_10_21", "late_22_31"]
        for label, label_group in summary.groupby("behavior_label_name"):
            label_group = label_group.set_index("stage").loc[stages].reset_index()
            linestyle = "-" if label == "self_resolves" else "--"
            axes[0].plot(
                stages,
                label_group["residual"],
                marker="o",
                linestyle=linestyle,
                color=colors[model],
                label=f"{model} {label}",
            )
            axes[1].plot(
                stages,
                label_group["oriented"],
                marker="o",
                linestyle=linestyle,
                color=colors[model],
                label=f"{model} {label}",
            )
    axes[0].set_title("Off-axis geometry by behavior")
    axes[0].set_ylabel("mean log off-axis ratio")
    axes[1].set_title("Alignment with eventually chosen constraint")
    axes[1].set_ylabel("mean oriented t")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "behavior_stage_geometry.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for metric, group in cross_model.groupby("metric"):
        ax.plot(group["layer"], group["spearman_r"], marker="o", markersize=3, label=metric)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("layer")
    ax.set_ylabel("4B–9B Spearman correlation")
    ax.set_title("Cross-model consistency of normalized geometry")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "cross_model_geometry_correlation.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for axis, target in zip(axes, ["pair_choice", "behavior_1_vs_3"]):
        subset = classifier_summary[classifier_summary["target"] == target]
        x = np.arange(len(subset["feature_group"].unique()))
        groups = list(subset["feature_group"].unique())
        width = 0.35
        for offset, (model, model_group) in enumerate(subset.groupby("model_id")):
            indexed = model_group.set_index("feature_group").loc[groups]
            axis.bar(
                x + (offset - 0.5) * width,
                indexed["balanced_accuracy_mean"],
                width=width,
                yerr=indexed["balanced_accuracy_std"],
                label=model,
                color=colors[model],
                alpha=0.85,
            )
        axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
        axis.set_xticks(x)
        axis.set_xticklabels(groups, rotation=30, ha="right")
        axis.set_ylim(0.4, 0.85)
        axis.set_ylabel("grouped-CV balanced accuracy")
        axis.set_title(target)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "grouped_cv_summary.png", dpi=220)
    plt.close(fig)


def write_findings(
    output_dir: Path,
    pair_counts: pd.DataFrame,
    behavior_counts: pd.DataFrame,
    prompt_order: pd.DataFrame,
    type_outcomes: pd.DataFrame,
    layer_frame: pd.DataFrame,
    extrema_frame: pd.DataFrame,
    stage_effect_frame: pd.DataFrame,
    controlled_behavior_frame: pd.DataFrame,
    within_type_stage_frame: pd.DataFrame,
    classifier_summary: pd.DataFrame,
    leave_type_frame: pd.DataFrame,
    cross_model_frame: pd.DataFrame,
    cross_model_labels: pd.DataFrame,
) -> None:
    lines = [
        "# Existing-data ConInstruct geometry findings",
        "",
        "This report treats each sample as a three-point geometry "
        "(`org`, `new`, `conflict`). It does not interpret the single-sample "
        "org→new displacement as a reusable concept vector.",
        "",
        "## Label coverage",
        "",
        pair_counts.to_markdown(index=False),
        "",
        behavior_counts.to_markdown(index=False),
        "",
        "## Prompt construction",
        "",
        prompt_order.to_markdown(index=False, floatfmt=".4f"),
        "",
        "All current conflict prompts append the new constraint after the "
        "original expanded instruction. There is no order-swap counterfactual, "
        "so recency cannot be separated from new-constraint identity.",
        "",
        "## Outcome rates by conflict type",
        "",
        type_outcomes.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Overall geometry",
        "",
        extrema_frame.to_markdown(index=False, floatfmt=".4f"),
        "",
    ]

    for model in MODELS:
        subset = layer_frame[layer_frame["model_id"] == model]
        selected = subset[subset["layer"].isin([1, 5, 10, 15, 20, 25, 31])]
        lines.extend([f"### {model}", "", selected.to_markdown(index=False, floatfmt=".4f"), ""])

    lines.extend(
        [
            "## Behavior 1 versus behavior 3",
            "",
            stage_effect_frame.to_markdown(index=False, floatfmt=".5f"),
            "",
            "## Behavior effects after controlling conflict type and chosen side",
            "",
            controlled_behavior_frame.to_markdown(index=False, floatfmt=".5f"),
            "",
            "## Within-conflict-type output-choice AUC",
            "",
            (
                within_type_stage_frame.groupby(
                    ["model_id", "stage", "metric"]
                )["roc_auc"]
                .agg(["mean", "median", "std"])
                .reset_index()
                .to_markdown(index=False, floatfmt=".4f")
            ),
            "",
            "## Grouped-CV prediction",
            "",
            classifier_summary.to_markdown(index=False, floatfmt=".4f"),
            "",
            "## Leave-one-conflict-type-out prediction",
            "",
            (
                leave_type_frame.groupby(
                    ["model_id", "target", "feature_group"]
                )[["balanced_accuracy", "roc_auc"]]
                .agg(["mean", "std"])
                .reset_index()
                .to_markdown(index=False, floatfmt=".4f")
            ),
            "",
            "## Cross-model normalized-geometry consistency",
            "",
            (
                cross_model_frame.groupby("metric")["spearman_r"]
                .agg(["mean", "min", "max"])
                .reset_index()
                .to_markdown(index=False, floatfmt=".4f")
            ),
            "",
            cross_model_labels.to_markdown(index=False, floatfmt=".4f"),
            "",
            "## Interpretation limits",
            "",
            "- Hidden states are from the final prompt position before generation, "
            "not answer-prefix or generated-token states.",
            "- There is no neutral seed hidden state in the existing extraction, "
            "so the analysis cannot identify two independently estimated concept vectors.",
            "- Off-axis displacement can reflect conflict interaction, prompt length, "
            "or composition of two constraints. A compatible two-constraint control is "
            "required before calling it conflict-specific.",
            "- Evaluation labels are model-judge outputs and contain judge noise.",
            "- Every current conflict prompt places the new constraint last, "
            "so the existing runs cannot identify a causal recency effect.",
        ]
    )
    (output_dir / "findings.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_state)

    score_frames = []
    pair_count_frames = []
    behavior_count_frames = []

    for model in MODELS:
        hidden_root = (
            root / "outputs" / "coninstruct_hidden_states" / MODEL_SAFE[model]
        )
        conflict_metadata, conflict_examples = load_metadata_frame(
            hidden_root / "conflict_resolution" / "metadata.json"
        )
        _, org_new_examples = load_metadata_frame(
            hidden_root / "conflict_resolution_org_new" / "metadata.json"
        )
        pair_labels = load_pair_labels(root, model)
        behavior_labels = load_behavior_labels(root, model)

        geometry = compute_geometry(
            root,
            model,
            conflict_examples,
            org_new_examples,
            conflict_metadata["run_config"].get(
                "new_constraint_position",
                "unknown",
            ),
        )
        geometry = geometry.merge(
            pair_labels,
            on=["conflict_type_idx", "sample_id"],
            how="left",
        )
        geometry = geometry.merge(
            behavior_labels,
            on=["conflict_type_idx", "sample_id"],
            how="left",
        )
        score_frames.append(geometry)

        pair_counts = (
            pair_labels["pair_label_name"]
            .value_counts(dropna=False)
            .rename_axis("label")
            .reset_index(name="n")
        )
        pair_counts.insert(0, "model_id", model)
        pair_count_frames.append(pair_counts)
        behavior_counts = (
            behavior_labels["behavior_label_name"]
            .value_counts(dropna=False)
            .rename_axis("label")
            .reset_index(name="n")
        )
        behavior_counts.insert(0, "model_id", model)
        behavior_count_frames.append(behavior_counts)

    scores = pd.concat(score_frames, ignore_index=True)
    scores.to_csv(output_dir / "sample_layer_geometry.csv", index=False)

    layer_frame = layer_summary(scores)
    layer_frame.to_csv(output_dir / "layer_summary.csv", index=False)
    extrema_frame = geometry_extrema(layer_frame)
    extrema_frame.to_csv(output_dir / "geometry_extrema.csv", index=False)

    stage_frame = build_sample_stage_frame(scores)
    stage_frame.to_csv(output_dir / "sample_stage_geometry.csv", index=False)

    type_stage_frame = type_stage_summary(stage_frame)
    type_stage_frame.to_csv(output_dir / "type_stage_summary.csv", index=False)

    trajectory = build_trajectory_features(scores)
    trajectory.to_csv(output_dir / "sample_trajectory_features.csv", index=False)

    type_outcomes = type_outcome_summary(trajectory)
    type_outcomes.to_csv(output_dir / "type_outcome_summary.csv", index=False)

    prompt_order = prompt_order_summary(trajectory)
    prompt_order.to_csv(output_dir / "prompt_order_summary.csv", index=False)

    stage_effect_frame = stage_effects(
        stage_frame,
        rng,
        args.bootstrap_samples,
    )
    stage_effect_frame.to_csv(output_dir / "behavior_stage_effects.csv", index=False)
    controlled_behavior_frame = controlled_behavior_effects(stage_frame)
    controlled_behavior_frame.to_csv(
        output_dir / "controlled_behavior_effects.csv",
        index=False,
    )
    within_type_stage_frame = within_type_stage_auc(stage_frame)
    within_type_stage_frame.to_csv(
        output_dir / "within_type_stage_auc.csv",
        index=False,
    )
    within_type_layer_frame = within_type_layer_auc(scores)
    within_type_layer_frame.to_csv(
        output_dir / "within_type_layer_auc.csv",
        index=False,
    )

    classifier_summary, leave_type_frame = classifier_analyses(
        trajectory,
        args.random_state,
    )
    classifier_summary.to_csv(output_dir / "grouped_cv_summary.csv", index=False)
    leave_type_frame.to_csv(output_dir / "leave_type_out_results.csv", index=False)

    cross_model_frame = cross_model_correlations(scores)
    cross_model_frame.to_csv(output_dir / "cross_model_correlations.csv", index=False)
    cross_model_labels = cross_model_label_agreement(trajectory)
    cross_model_labels.to_csv(output_dir / "cross_model_label_agreement.csv", index=False)

    make_plots(
        output_dir,
        layer_frame,
        stage_frame,
        cross_model_frame,
        classifier_summary,
    )

    pair_counts = pd.concat(pair_count_frames, ignore_index=True)
    behavior_counts = pd.concat(behavior_count_frames, ignore_index=True)
    pair_counts.to_csv(output_dir / "pair_label_counts.csv", index=False)
    behavior_counts.to_csv(output_dir / "behavior_label_counts.csv", index=False)

    write_findings(
        output_dir,
        pair_counts,
        behavior_counts,
        prompt_order,
        type_outcomes,
        layer_frame,
        extrema_frame,
        stage_effect_frame,
        controlled_behavior_frame,
        within_type_stage_frame,
        classifier_summary,
        leave_type_frame,
        cross_model_frame,
        cross_model_labels,
    )
    print(f"Wrote analysis to {output_dir}")


if __name__ == "__main__":
    main()
