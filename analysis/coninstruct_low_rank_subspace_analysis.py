from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.utils.extmath import randomized_svd


MODELS = ("Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B")
MODEL_SAFE = {model: model.replace("/", "__") for model in MODELS}
LAYERS = tuple(range(32))
CONFLICT_TYPES = tuple(range(1, 10))
TYPE_COMPONENTS = {
    1: {"content"},
    2: {"keyword"},
    3: {"keyword", "phrase"},
    4: {"phrase"},
    5: {"length"},
    6: {"format"},
    7: {"style"},
    8: {"phrase", "content"},
    9: {"phrase", "style"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Adapt the low-rank behavior-subspace analysis from arXiv:2606.14388 "
            "to ConInstruct org/new matched hidden-state contrasts."
        )
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/coninstruct_low_rank_subspaces"),
    )
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--decision-rank", type=int, default=2)
    parser.add_argument(
        "--layers",
        default="0,8,16,24,31",
        help=(
            "Comma-separated layers to analyze. The default includes early, "
            "middle, the paper-style -8 layer (24 of 32), and final layer."
        ),
    )
    parser.add_argument(
        "--decision-pca-components",
        type=int,
        default=16,
        help=(
            "Number of pooled contrast PCs searched for label alignment. "
            "The paper ranks PCs by correlation with a continuous decision margin; "
            "we use the binary org/new outcome as an explicit approximation."
        ),
    )
    parser.add_argument("--bootstrap-layer", type=int, default=24)
    parser.add_argument("--bootstrap-samples", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    args.layer_indices = tuple(int(value) for value in args.layers.split(","))
    invalid_layers = set(args.layer_indices) - set(LAYERS)
    if invalid_layers:
        parser.error(f"Invalid layers: {sorted(invalid_layers)}")
    return args


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_final_label(text: str, allowed: tuple[int, ...]) -> int:
    if not isinstance(text, str) or not text.strip():
        return -1
    allowed_set = set(allowed)
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


def load_metadata_frame(path: Path) -> tuple[dict, pd.DataFrame]:
    metadata = load_json(path)
    frame = pd.DataFrame(metadata["examples"])
    frame["conflict_type_idx"] = frame["conflict_type_idx"].astype(int)
    frame["sample_id"] = frame["sample_id"].astype(int)
    frame["key"] = list(zip(frame["conflict_type_idx"], frame["sample_id"]))
    return metadata, frame


def load_pair_labels(root: Path, model: str) -> pd.DataFrame:
    rows = []
    eval_root = root / "evaluation_outputs" / "conflict_resolution" / model
    for type_dir in sorted(
        (path for path in eval_root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    ):
        conflict_type_idx = int(type_dir.name)
        for path in sorted(
            (candidate for candidate in type_dir.glob("*.json") if candidate.stem.isdigit()),
            key=lambda candidate: int(candidate.stem),
        ):
            data = load_json(path)
            entries = [value for key, value in data.items() if key != "llm_response"]
            if not entries:
                continue
            item = entries[0]
            raw_label = parse_final_label(item.get("evaluation_result", ""), (-1, 1, 2))
            order = item.get("instruction_order", "unknown")
            label = {1: 2, 2: 1}.get(raw_label, raw_label) if order == "org_new" else raw_label
            rows.append(
                {
                    "conflict_type_idx": conflict_type_idx,
                    "sample_id": int(path.stem),
                    "pair_label": label,
                    "pair_label_name": {1: "new", 2: "org", -1: "neither"}.get(
                        label, "unknown"
                    ),
                }
            )
    return pd.DataFrame(rows)


def load_behavior_labels(root: Path, model: str) -> pd.DataFrame:
    rows = []
    eval_root = root / "evaluation_outputs" / "conflict_resolution" / "behavior" / model
    for type_dir in sorted(
        (path for path in eval_root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    ):
        conflict_type_idx = int(type_dir.name)
        for path in sorted(
            (candidate for candidate in type_dir.glob("*.json") if candidate.stem.isdigit()),
            key=lambda candidate: int(candidate.stem),
        ):
            data = load_json(path)
            label = parse_final_label(data.get("evaluation_result", ""), (1, 2, 3, 4))
            rows.append(
                {
                    "conflict_type_idx": conflict_type_idx,
                    "sample_id": int(path.stem),
                    "behavior_label": label,
                }
            )
    return pd.DataFrame(rows)


def rows_for_type(
    frame: pd.DataFrame,
    conflict_type_idx: int,
    variant: str | None = None,
) -> pd.DataFrame:
    subset = frame[frame["conflict_type_idx"] == conflict_type_idx]
    if variant is not None:
        subset = subset[subset["variant"] == variant]
    return subset.reset_index(drop=True)


def fit_basis(
    matrix: np.ndarray,
    rank: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    usable_rank = min(rank, centered.shape[0] - 1, centered.shape[1])
    if usable_rank < 1:
        raise ValueError(f"Cannot fit rank-{rank} basis to shape {centered.shape}")
    _, singular_values, vt = randomized_svd(
        centered,
        n_components=usable_rank,
        random_state=random_state,
    )
    basis = vt.T
    denominator = float(np.square(centered).sum())
    explained_ratio = (
        float(np.square(singular_values).sum() / denominator)
        if denominator > 0
        else math.nan
    )
    return basis, singular_values, explained_ratio


def residualize(matrix: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return matrix - (matrix @ basis) @ basis.T


def subspace_metrics(
    basis_a: np.ndarray,
    basis_b: np.ndarray,
) -> dict[str, float]:
    singular_values = np.linalg.svd(basis_a.T @ basis_b, compute_uv=False)
    singular_values = np.clip(singular_values, 0.0, 1.0)
    angles = np.degrees(np.arccos(singular_values))
    return {
        "overlap": float(np.mean(np.square(singular_values))),
        "sigma_1": float(singular_values[0]),
        "sigma_2": float(singular_values[1]) if len(singular_values) > 1 else math.nan,
        "angle_1_deg": float(angles[0]),
        "angle_2_deg": float(angles[1]) if len(angles) > 1 else math.nan,
        "mean_angle_deg": float(angles.mean()),
        "shared_dimension_sigma_ge_0_5": int((singular_values >= 0.5).sum()),
    }


def aligned_type_data(
    hidden_root: Path,
    conflict_type_idx: int,
    layer: int,
    conflict_examples: pd.DataFrame,
    org_new_examples: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    org_rows = rows_for_type(org_new_examples, conflict_type_idx, "org")
    new_rows = rows_for_type(org_new_examples, conflict_type_idx, "new")
    org_index = {key: idx for idx, key in enumerate(org_rows["key"])}
    new_index = {key: idx for idx, key in enumerate(new_rows["key"])}
    keys = sorted(set(org_index) & set(new_index))

    ctype = f"conflict_type_{conflict_type_idx}"
    org = np.load(
        hidden_root / "conflict_resolution_org_new" / ctype / "org" / f"layer_{layer}.npy"
    )
    new = np.load(
        hidden_root / "conflict_resolution_org_new" / ctype / "new" / f"layer_{layer}.npy"
    )
    org_aligned = org[[org_index[key] for key in keys]].astype(np.float64)
    new_aligned = new[[new_index[key] for key in keys]].astype(np.float64)
    contrasts = new_aligned - org_aligned

    metadata = pd.DataFrame(
        {
            "conflict_type_idx": [key[0] for key in keys],
            "sample_id": [key[1] for key in keys],
        }
    ).merge(labels, on=["conflict_type_idx", "sample_id"], how="left")
    return contrasts, metadata


def identify_label_aligned_subspace(
    pooled_contrasts: np.ndarray,
    pooled_labels: np.ndarray,
    decision_rank: int,
    pca_components: int,
    random_state: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    centered = pooled_contrasts - pooled_contrasts.mean(axis=0, keepdims=True)
    components = min(
        pca_components,
        centered.shape[0] - 1,
        centered.shape[1],
    )
    _, singular_values, vt = randomized_svd(
        centered,
        n_components=components,
        random_state=random_state,
    )
    basis = vt.T
    scores = centered @ basis
    valid = np.isfinite(pooled_labels)

    rows = []
    for component_idx in range(components):
        if valid.sum() < 3 or np.std(scores[valid, component_idx]) == 0:
            correlation = math.nan
        else:
            correlation = pearsonr(
                scores[valid, component_idx],
                pooled_labels[valid],
            ).statistic
        rows.append(
            {
                "component": component_idx,
                "singular_value": singular_values[component_idx],
                "label_correlation": correlation,
                "abs_label_correlation": abs(correlation),
            }
        )
    diagnostics = pd.DataFrame(rows).sort_values(
        "abs_label_correlation",
        ascending=False,
    )
    selected = diagnostics.head(decision_rank)["component"].to_numpy(dtype=int)
    diagnostics["selected"] = diagnostics["component"].isin(selected)
    return basis[:, selected], diagnostics.sort_values("component")


def random_overlap_baseline(
    dimension: int,
    rank: int,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = []
    for _ in range(samples):
        qa, _ = np.linalg.qr(rng.normal(size=(dimension, rank)))
        qb, _ = np.linalg.qr(rng.normal(size=(dimension, rank)))
        values.append(np.square(qa.T @ qb).sum() / rank)
    return float(np.mean(values)), float(np.std(values, ddof=1))


def outcome_summary(
    pair_labels: pd.DataFrame,
    behavior_labels: pd.DataFrame,
) -> pd.DataFrame:
    merged = pair_labels.merge(
        behavior_labels,
        on=["conflict_type_idx", "sample_id"],
        how="outer",
    )
    rows = []
    for conflict_type_idx, group in merged.groupby("conflict_type_idx"):
        pair_valid = group[group["pair_label_name"].isin(["org", "new"])]
        behavior_valid = group[group["behavior_label"].isin([1, 2, 3, 4])]
        rows.append(
            {
                "conflict_type_idx": conflict_type_idx,
                "new_choice_rate": (
                    pair_valid["pair_label_name"] == "new"
                ).mean(),
                "self_resolves_rate": (
                    behavior_valid["behavior_label"] == 3
                ).mean(),
            }
        )
    return pd.DataFrame(rows)


def analyze_model(
    root: Path,
    output_dir: Path,
    model: str,
    rank: int,
    decision_rank: int,
    decision_pca_components: int,
    random_state: int,
    layers: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, dict]]:
    hidden_root = root / "outputs" / "coninstruct_hidden_states" / MODEL_SAFE[model]
    _, conflict_examples = load_metadata_frame(
        hidden_root / "conflict_resolution" / "metadata.json"
    )
    _, org_new_examples = load_metadata_frame(
        hidden_root / "conflict_resolution_org_new" / "metadata.json"
    )
    pair_labels = load_pair_labels(root, model)
    behavior_labels = load_behavior_labels(root, model)
    labels = pair_labels.merge(
        behavior_labels,
        on=["conflict_type_idx", "sample_id"],
        how="outer",
    )
    outcomes = outcome_summary(pair_labels, behavior_labels)
    outcomes["model_id"] = model

    overlap_rows = []
    category_rows = []
    decision_rows = []
    cache: dict[int, dict] = {}

    for layer in layers:
        type_data = {}
        pooled_contrasts = []
        pooled_label_values = []
        for conflict_type_idx in CONFLICT_TYPES:
            contrasts, metadata = aligned_type_data(
                hidden_root,
                conflict_type_idx,
                layer,
                conflict_examples,
                org_new_examples,
                labels,
            )
            binary_label = metadata["pair_label"].map({1: 1.0, 2: -1.0}).to_numpy()
            type_data[conflict_type_idx] = {
                "contrasts": contrasts,
                "metadata": metadata,
            }
            pooled_contrasts.append(contrasts)
            pooled_label_values.append(binary_label)

        pooled = np.vstack(pooled_contrasts)
        pooled_labels = np.concatenate(pooled_label_values)
        decision_basis, decision_diagnostics = identify_label_aligned_subspace(
            pooled,
            pooled_labels,
            decision_rank,
            decision_pca_components,
            random_state + layer,
        )
        decision_diagnostics["model_id"] = model
        decision_diagnostics["layer"] = layer
        decision_rows.append(decision_diagnostics)

        raw_bases = {}
        residual_bases = {}
        centered_contrasts = {}
        residualized_contrasts = {}

        for conflict_type_idx, values in type_data.items():
            contrasts = values["contrasts"]
            centered = contrasts - contrasts.mean(axis=0, keepdims=True)
            raw_basis, raw_singular_values, raw_explained = fit_basis(
                centered,
                rank,
                random_state + layer * 100 + conflict_type_idx,
            )
            residualized = residualize(centered, decision_basis)
            residual_basis, residual_singular_values, residual_explained = fit_basis(
                residualized,
                rank,
                random_state + 100000 + layer * 100 + conflict_type_idx,
            )
            raw_bases[conflict_type_idx] = raw_basis
            residual_bases[conflict_type_idx] = residual_basis
            centered_contrasts[conflict_type_idx] = centered
            residualized_contrasts[conflict_type_idx] = residualized

            coupling = subspace_metrics(raw_basis, decision_basis)
            category_rows.append(
                {
                    "model_id": model,
                    "layer": layer,
                    "conflict_type_idx": conflict_type_idx,
                    "n": len(contrasts),
                    "hidden_dimension": contrasts.shape[1],
                    "rank": rank,
                    "raw_topk_explained_ratio": raw_explained,
                    "residual_topk_explained_ratio": residual_explained,
                    "raw_singular_value_1": raw_singular_values[0],
                    "raw_singular_value_2": raw_singular_values[1],
                    "residual_singular_value_1": residual_singular_values[0],
                    "residual_singular_value_2": residual_singular_values[1],
                    "decision_coupling_overlap": coupling["overlap"],
                    "decision_coupling_mean_angle_deg": coupling["mean_angle_deg"],
                    "decision_coupling_min_angle_deg": coupling["angle_1_deg"],
                }
            )

        for index, type_a in enumerate(CONFLICT_TYPES):
            for type_b in CONFLICT_TYPES[index + 1 :]:
                raw_metrics = subspace_metrics(raw_bases[type_a], raw_bases[type_b])
                residual_metrics = subspace_metrics(
                    residual_bases[type_a],
                    residual_bases[type_b],
                )
                for space_name, metrics in [
                    ("raw", raw_metrics),
                    ("label_residualized", residual_metrics),
                ]:
                    overlap_rows.append(
                        {
                            "model_id": model,
                            "layer": layer,
                            "type_a": type_a,
                            "type_b": type_b,
                            "space": space_name,
                            **metrics,
                        }
                    )

        cache[layer] = {
            "decision_basis": decision_basis,
            "raw_bases": raw_bases,
            "residual_bases": residual_bases,
            "centered_contrasts": centered_contrasts,
            "residualized_contrasts": residualized_contrasts,
            "dimension": pooled.shape[1],
        }

    overlap_frame = pd.DataFrame(overlap_rows)
    category_frame = pd.DataFrame(category_rows)
    decision_frame = pd.concat(decision_rows, ignore_index=True)
    model_dir = output_dir / MODEL_SAFE[model]
    model_dir.mkdir(parents=True, exist_ok=True)
    overlap_frame.to_csv(model_dir / "pairwise_subspace_overlap.csv", index=False)
    category_frame.to_csv(model_dir / "category_subspace_metrics.csv", index=False)
    decision_frame.to_csv(model_dir / "label_aligned_pc_diagnostics.csv", index=False)
    outcomes.to_csv(model_dir / "outcome_summary.csv", index=False)
    return overlap_frame, category_frame, outcomes, cache


def bootstrap_stability(
    model: str,
    layer_cache: dict,
    rank: int,
    samples: int,
    random_state: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows = []
    decision_basis = layer_cache["decision_basis"]
    for conflict_type_idx in CONFLICT_TYPES:
        raw_reference = layer_cache["raw_bases"][conflict_type_idx]
        residual_reference = layer_cache["residual_bases"][conflict_type_idx]
        centered = layer_cache["centered_contrasts"][conflict_type_idx]
        for bootstrap_idx in range(samples):
            indices = rng.integers(0, len(centered), size=len(centered))
            sample = centered[indices]
            raw_basis, _, _ = fit_basis(
                sample,
                rank,
                random_state + bootstrap_idx + conflict_type_idx * 1000,
            )
            residual_sample = residualize(
                sample - sample.mean(axis=0, keepdims=True),
                decision_basis,
            )
            residual_basis, _, _ = fit_basis(
                residual_sample,
                rank,
                random_state + 100000 + bootstrap_idx + conflict_type_idx * 1000,
            )
            rows.extend(
                [
                    {
                        "model_id": model,
                        "conflict_type_idx": conflict_type_idx,
                        "bootstrap_idx": bootstrap_idx,
                        "space": "raw",
                        "overlap_with_reference": subspace_metrics(
                            raw_reference,
                            raw_basis,
                        )["overlap"],
                    },
                    {
                        "model_id": model,
                        "conflict_type_idx": conflict_type_idx,
                        "bootstrap_idx": bootstrap_idx,
                        "space": "label_residualized",
                        "overlap_with_reference": subspace_metrics(
                            residual_reference,
                            residual_basis,
                        )["overlap"],
                    },
                ]
            )
    return pd.DataFrame(rows)


def overlap_behavior_relationship(
    overlap_frame: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    outcome_lookup = outcomes.set_index("conflict_type_idx")
    rows = []
    for _, row in overlap_frame.iterrows():
        type_a = int(row["type_a"])
        type_b = int(row["type_b"])
        rows.append(
            {
                **row.to_dict(),
                "new_choice_rate_distance": abs(
                    outcome_lookup.loc[type_a, "new_choice_rate"]
                    - outcome_lookup.loc[type_b, "new_choice_rate"]
                ),
                "self_resolves_rate_distance": abs(
                    outcome_lookup.loc[type_a, "self_resolves_rate"]
                    - outcome_lookup.loc[type_b, "self_resolves_rate"]
                ),
            }
        )
    pair_frame = pd.DataFrame(rows)
    correlation_rows = []
    for (model, layer, space), group in pair_frame.groupby(
        ["model_id", "layer", "space"]
    ):
        for distance_name in [
            "new_choice_rate_distance",
            "self_resolves_rate_distance",
        ]:
            correlation, p_value = spearmanr(
                group["overlap"],
                -group[distance_name],
            )
            correlation_rows.append(
                {
                    "model_id": model,
                    "layer": layer,
                    "space": space,
                    "outcome_similarity": distance_name.replace("_distance", ""),
                    "spearman_r": correlation,
                    "p_value": p_value,
                    "n_pairs": len(group),
                }
            )
    return pd.DataFrame(correlation_rows)


def cross_model_matrix_correlations(overlap_frame: pd.DataFrame) -> pd.DataFrame:
    four = overlap_frame[overlap_frame["model_id"] == MODELS[0]]
    nine = overlap_frame[overlap_frame["model_id"] == MODELS[1]]
    merged = four.merge(
        nine,
        on=["layer", "type_a", "type_b", "space"],
        suffixes=("_4b", "_9b"),
    )
    rows = []
    for (layer, space), group in merged.groupby(["layer", "space"]):
        correlation, p_value = spearmanr(
            group["overlap_4b"],
            group["overlap_9b"],
        )
        rows.append(
            {
                "layer": layer,
                "space": space,
                "spearman_r": correlation,
                "p_value": p_value,
                "n_pairs": len(group),
            }
        )
    return pd.DataFrame(rows)


def named_component_overlap_test(overlap_frame: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import mannwhitneyu

    frame = overlap_frame.copy()
    frame["shares_named_component"] = [
        bool(TYPE_COMPONENTS[int(type_a)] & TYPE_COMPONENTS[int(type_b)])
        for type_a, type_b in zip(frame["type_a"], frame["type_b"])
    ]
    rows = []
    for (model, layer, space), group in frame.groupby(
        ["model_id", "layer", "space"]
    ):
        shared = group[group["shares_named_component"]]["overlap"]
        nonshared = group[~group["shares_named_component"]]["overlap"]
        statistic, p_value = mannwhitneyu(
            shared,
            nonshared,
            alternative="greater",
        )
        rows.append(
            {
                "model_id": model,
                "layer": layer,
                "space": space,
                "n_shared_pairs": len(shared),
                "shared_mean_overlap": shared.mean(),
                "nonshared_mean_overlap": nonshared.mean(),
                "shared_median_overlap": shared.median(),
                "nonshared_median_overlap": nonshared.median(),
                "mannwhitney_u": statistic,
                "p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def rank_sensitivity(
    caches: dict[str, dict[int, dict]],
    layer: int,
    random_state: int,
) -> pd.DataFrame:
    from scipy.stats import mannwhitneyu

    rows = []
    for model in MODELS:
        matrices = caches[model][layer]["centered_contrasts"]
        dimension = caches[model][layer]["dimension"]
        for rank in [1, 2, 3, 4, 5]:
            bases = {
                conflict_type_idx: fit_basis(
                    matrix,
                    rank,
                    random_state + rank * 100 + conflict_type_idx,
                )[0]
                for conflict_type_idx, matrix in matrices.items()
            }
            all_overlaps = []
            shared_overlaps = []
            nonshared_overlaps = []
            for index, type_a in enumerate(CONFLICT_TYPES):
                for type_b in CONFLICT_TYPES[index + 1 :]:
                    overlap = subspace_metrics(
                        bases[type_a],
                        bases[type_b],
                    )["overlap"]
                    all_overlaps.append(overlap)
                    if TYPE_COMPONENTS[type_a] & TYPE_COMPONENTS[type_b]:
                        shared_overlaps.append(overlap)
                    else:
                        nonshared_overlaps.append(overlap)
            statistic, p_value = mannwhitneyu(
                shared_overlaps,
                nonshared_overlaps,
                alternative="greater",
            )
            rows.append(
                {
                    "model_id": model,
                    "layer": layer,
                    "rank": rank,
                    "mean_overlap": np.mean(all_overlaps),
                    "median_overlap": np.median(all_overlaps),
                    "shared_component_mean_overlap": np.mean(shared_overlaps),
                    "nonshared_mean_overlap": np.mean(nonshared_overlaps),
                    "shared_vs_nonshared_u": statistic,
                    "shared_vs_nonshared_p_value": p_value,
                    "random_k_over_d": rank / dimension,
                }
            )
    return pd.DataFrame(rows)


def matrix_from_pairs(frame: pd.DataFrame, layer: int, space: str) -> np.ndarray:
    matrix = np.eye(len(CONFLICT_TYPES))
    subset = frame[(frame["layer"] == layer) & (frame["space"] == space)]
    for _, row in subset.iterrows():
        a = int(row["type_a"]) - 1
        b = int(row["type_b"]) - 1
        matrix[a, b] = row["overlap"]
        matrix[b, a] = row["overlap"]
    return matrix


def make_plots(
    output_dir: Path,
    overlap_frame: pd.DataFrame,
    category_frame: pd.DataFrame,
    stability_frame: pd.DataFrame,
    bootstrap_layer: int,
) -> None:
    colors = {"raw": "#d62728", "label_residualized": "#1f77b4"}

    summary = (
        overlap_frame.groupby(["model_id", "layer", "space"])["overlap"]
        .mean()
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for axis, model in zip(axes, MODELS):
        subset = summary[summary["model_id"] == model]
        for space, group in subset.groupby("space"):
            axis.plot(
                group["layer"],
                group["overlap"],
                marker="o",
                markersize=3,
                label=space,
                color=colors[space],
            )
        axis.set_title(model)
        axis.set_xlabel("layer")
        axis.set_ylabel("mean off-diagonal rank-2 overlap")
        axis.legend()
    fig.suptitle("Conflict-type subspace overlap across layers")
    fig.tight_layout()
    fig.savefig(output_dir / "mean_overlap_by_layer.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for row_idx, model in enumerate(MODELS):
        for col_idx, space in enumerate(["raw", "label_residualized"]):
            matrix = matrix_from_pairs(
                overlap_frame[overlap_frame["model_id"] == model],
                bootstrap_layer,
                space,
            )
            axis = axes[row_idx, col_idx]
            image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
            axis.set_xticks(range(9), labels=range(1, 10))
            axis.set_yticks(range(9), labels=range(1, 10))
            axis.set_xlabel("conflict type")
            axis.set_ylabel("conflict type")
            axis.set_title(f"{model}\n{space}, layer {bootstrap_layer}")
            fig.colorbar(image, ax=axis, fraction=0.046)
    fig.tight_layout()
    fig.savefig(output_dir / f"overlap_heatmaps_layer_{bootstrap_layer}.png", dpi=220)
    plt.close(fig)

    explained = (
        category_frame.groupby(["model_id", "layer"])[
            ["raw_topk_explained_ratio", "residual_topk_explained_ratio"]
        ]
        .mean()
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for axis, model in zip(axes, MODELS):
        subset = explained[explained["model_id"] == model]
        axis.plot(
            subset["layer"],
            subset["raw_topk_explained_ratio"],
            marker="o",
            markersize=3,
            label="raw",
        )
        axis.plot(
            subset["layer"],
            subset["residual_topk_explained_ratio"],
            marker="o",
            markersize=3,
            label="label residualized",
        )
        axis.set_title(model)
        axis.set_xlabel("layer")
        axis.set_ylabel("top-2 explained variance ratio")
        axis.legend()
    fig.suptitle("How low-rank are the conflict-type contrast matrices?")
    fig.tight_layout()
    fig.savefig(output_dir / "rank2_explained_variance.png", dpi=220)
    plt.close(fig)

    stability_summary = (
        stability_frame.groupby(["model_id", "conflict_type_idx", "space"])[
            "overlap_with_reference"
        ]
        .mean()
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for axis, model in zip(axes, MODELS):
        subset = stability_summary[stability_summary["model_id"] == model]
        x = np.arange(9)
        width = 0.38
        for offset, space in enumerate(["raw", "label_residualized"]):
            group = subset[subset["space"] == space].set_index(
                "conflict_type_idx"
            ).loc[list(CONFLICT_TYPES)]
            axis.bar(
                x + (offset - 0.5) * width,
                group["overlap_with_reference"],
                width,
                label=space,
                color=colors[space],
            )
        axis.set_xticks(x, labels=CONFLICT_TYPES)
        axis.set_xlabel("conflict type")
        axis.set_ylabel("bootstrap overlap with fitted subspace")
        axis.set_ylim(0, 1)
        axis.set_title(model)
        axis.legend()
    fig.suptitle(f"Rank-2 subspace bootstrap stability at layer {bootstrap_layer}")
    fig.tight_layout()
    fig.savefig(output_dir / f"bootstrap_stability_layer_{bootstrap_layer}.png", dpi=220)
    plt.close(fig)


def write_report(
    output_dir: Path,
    overlap_frame: pd.DataFrame,
    category_frame: pd.DataFrame,
    stability_frame: pd.DataFrame,
    random_frame: pd.DataFrame,
    cross_model_frame: pd.DataFrame,
    behavior_relation_frame: pd.DataFrame,
    component_test_frame: pd.DataFrame,
    rank_sensitivity_frame: pd.DataFrame,
    bootstrap_layer: int,
) -> None:
    overlap_summary = (
        overlap_frame.groupby(["model_id", "layer", "space"])["overlap"]
        .mean()
        .reset_index()
    )
    selected_overlap = overlap_summary[
        overlap_summary["layer"].isin([0, 8, 16, bootstrap_layer, 31])
    ]
    explained_summary = (
        category_frame.groupby(["model_id", "layer"])
        .agg(
            raw_rank2_explained=("raw_topk_explained_ratio", "mean"),
            residual_rank2_explained=("residual_topk_explained_ratio", "mean"),
            decision_coupling_overlap=("decision_coupling_overlap", "mean"),
            decision_coupling_angle_deg=("decision_coupling_mean_angle_deg", "mean"),
        )
        .reset_index()
    )
    selected_explained = explained_summary[
        explained_summary["layer"].isin([0, 8, 16, bootstrap_layer, 31])
    ]
    stability_summary = (
        stability_frame.groupby(["model_id", "space"])["overlap_with_reference"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    cross_selected = cross_model_frame[
        cross_model_frame["layer"].isin([0, 8, 16, bootstrap_layer, 31])
    ]
    behavior_selected = behavior_relation_frame[
        behavior_relation_frame["layer"].isin([0, 8, 16, bootstrap_layer, 31])
    ]
    component_selected = component_test_frame[
        component_test_frame["layer"].isin([0, 8, 16, bootstrap_layer, 31])
    ]

    lines = [
        "# ConInstruct low-rank subspace analysis",
        "",
        "Method adapted from Sharma et al., *A Low-Rank Subspace Analysis "
        "of LLM Interventions* (arXiv:2606.14388). Each ConInstruct conflict "
        "type is treated as a behavior category. For each matched sample, "
        "`delta = h_new - h_org`; centered deltas are stacked and a rank-2 "
        "PCA/SVD subspace is fitted.",
        "",
        "The paper identifies and removes decision-aligned PCs using a continuous "
        "model log-probability margin. That margin is unavailable here. The "
        "`label_residualized` analysis instead ranks pooled PCs by correlation "
        "with the binary final org/new judge label. It is an approximation and "
        "must not be described as an exact replication of decision residualization.",
        "",
        "## Mean pairwise conflict-type overlap",
        "",
        selected_overlap.to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Rank-2 explained variance and label coupling",
        "",
        selected_explained.to_markdown(index=False, floatfmt=".5f"),
        "",
        f"## Bootstrap stability at layer {bootstrap_layer}",
        "",
        stability_summary.to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Random rank-2 baseline",
        "",
        random_frame.to_markdown(index=False, floatfmt=".7f"),
        "",
        "## Cross-model correlation of overlap matrices",
        "",
        cross_selected.to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Exploratory relation between overlap and outcome-profile similarity",
        "",
        behavior_selected.to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Do taxonomically related conflict types share more geometry?",
        "",
        "Type components are defined directly from the ConInstruct taxonomy: "
        "content, keyword, phrase, length, format, and style. Examples include "
        "1↔8 (content), 3↔4 (phrase), and 7↔9 (style).",
        "",
        component_selected.to_markdown(index=False, floatfmt=".5f"),
        "",
        f"## Rank sensitivity at layer {bootstrap_layer}",
        "",
        rank_sensitivity_frame.to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Scope",
        "",
        "- This analysis tests whether conflict *families* occupy reproducible "
        "low-rank covariance subspaces. It does not estimate a reusable vector "
        "for any individual constraint.",
        "- Without model log-probability margins, the decision-related "
        "residualization is approximate.",
        "- Without activation projection interventions, overlap cannot be linked "
        "causally to cross-type behavioral effects as in the paper.",
        "- The current two models are checkpoints from one model family; "
        "cross-family replication remains necessary.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_state)

    overlap_frames = []
    category_frames = []
    outcome_frames = []
    caches = {}

    for model in MODELS:
        overlap, category, outcomes, cache = analyze_model(
            root,
            output_dir,
            model,
            args.rank,
            args.decision_rank,
            args.decision_pca_components,
            args.random_state,
            args.layer_indices,
        )
        overlap_frames.append(overlap)
        category_frames.append(category)
        outcome_frames.append(outcomes)
        caches[model] = cache

    overlap_frame = pd.concat(overlap_frames, ignore_index=True)
    category_frame = pd.concat(category_frames, ignore_index=True)
    outcomes = pd.concat(outcome_frames, ignore_index=True)
    overlap_frame.to_csv(output_dir / "all_pairwise_subspace_overlap.csv", index=False)
    category_frame.to_csv(output_dir / "all_category_subspace_metrics.csv", index=False)
    outcomes.to_csv(output_dir / "all_outcome_summaries.csv", index=False)

    stability_frames = []
    random_rows = []
    for model in MODELS:
        stability_frames.append(
            bootstrap_stability(
                model,
                caches[model][args.bootstrap_layer],
                args.rank,
                args.bootstrap_samples,
                args.random_state,
            )
        )
        dimension = caches[model][args.bootstrap_layer]["dimension"]
        mean, std = random_overlap_baseline(
            dimension,
            args.rank,
            1000,
            rng,
        )
        random_rows.append(
            {
                "model_id": model,
                "hidden_dimension": dimension,
                "rank": args.rank,
                "empirical_random_overlap_mean": mean,
                "empirical_random_overlap_std": std,
                "theoretical_k_over_d": args.rank / dimension,
            }
        )
    stability_frame = pd.concat(stability_frames, ignore_index=True)
    random_frame = pd.DataFrame(random_rows)
    stability_frame.to_csv(output_dir / "bootstrap_stability.csv", index=False)
    random_frame.to_csv(output_dir / "random_subspace_baseline.csv", index=False)

    cross_model_frame = cross_model_matrix_correlations(overlap_frame)
    cross_model_frame.to_csv(
        output_dir / "cross_model_overlap_matrix_correlations.csv",
        index=False,
    )

    behavior_relation_frames = []
    for model in MODELS:
        behavior_relation_frames.append(
            overlap_behavior_relationship(
                overlap_frame[overlap_frame["model_id"] == model],
                outcomes[outcomes["model_id"] == model],
            )
        )
    behavior_relation_frame = pd.concat(behavior_relation_frames, ignore_index=True)
    behavior_relation_frame.to_csv(
        output_dir / "overlap_outcome_similarity_correlations.csv",
        index=False,
    )
    component_test_frame = named_component_overlap_test(overlap_frame)
    component_test_frame.to_csv(
        output_dir / "named_component_overlap_test.csv",
        index=False,
    )
    rank_sensitivity_frame = rank_sensitivity(
        caches,
        args.bootstrap_layer,
        args.random_state,
    )
    rank_sensitivity_frame.to_csv(
        output_dir / f"rank_sensitivity_layer_{args.bootstrap_layer}.csv",
        index=False,
    )

    make_plots(
        output_dir,
        overlap_frame,
        category_frame,
        stability_frame,
        args.bootstrap_layer,
    )
    write_report(
        output_dir,
        overlap_frame,
        category_frame,
        stability_frame,
        random_frame,
        cross_model_frame,
        behavior_relation_frame,
        component_test_frame,
        rank_sensitivity_frame,
        args.bootstrap_layer,
    )
    print(f"Wrote low-rank subspace analysis to {output_dir}")


if __name__ == "__main__":
    main()
