from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


VARIANT_FIELDS = {
    "seed": "seed_instruction",
    "org": "original_instruction",
    "new": "new_instruction",
    "conflict": "conflict_instruction",
}
LABEL_NAMES = {1: "new", 2: "org", -1: "neither"}
EPS = 1e-12


@dataclass
class PilotItem:
    key: tuple[int, int]
    triplet: dict[str, Any]
    response: str
    label: int
    label_name: str
    response_path: str
    evaluation_path: str


def parse_int_list(value: str) -> list[int]:
    values = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated integer.")
    return values


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "y", "t"}:
        return True
    if normalized in {"0", "false", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract sample-local constraint-subspace trajectories by teacher-forcing "
            "the same conflict response under seed/original/new/conflict prompts."
        )
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument(
        "--triplets",
        type=Path,
        default=Path("datasets/constraint_triplets.jsonl"),
    )
    parser.add_argument(
        "--responses-root",
        type=Path,
        default=Path(
            "outputs/constraint_triplets/Qwen/Qwen3.5-4B/"
            "non_thinking/conflict"
        ),
    )
    parser.add_argument(
        "--evaluations-root",
        type=Path,
        default=Path(
            "evaluation_outputs/constraint_triplets/"
            "judge_gpt-4o-2024-11-20/Qwen/Qwen3.5-4B/"
            "non_thinking/conflict"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "analysis/constraint_trajectory/"
            "Qwen__Qwen3.5-4B/non_thinking"
        ),
    )
    parser.add_argument(
        "--layers",
        type=parse_int_list,
        default=parse_int_list("8,16,24,31"),
        help="Comma-separated zero-indexed decoder layers.",
    )
    parser.add_argument(
        "--prefixes",
        type=parse_int_list,
        default=parse_int_list("0,1,4,16,32,64,128,256,512,1024,2048,4096"),
        help="Response-prefix token counts. Prefix 0 is always added.",
    )
    parser.add_argument(
        "--max-response-tokens",
        type=int,
        default=4096,
        help=(
            "Maximum teacher-forced response length. Use 8192 for very long CoT "
            "when the model context window and GPU memory permit it."
        ),
    )
    parser.add_argument(
        "--include-final",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also extract the final available response-token position.",
    )
    parser.add_argument(
        "--labels",
        default="org,new",
        help="Comma-separated behavior labels to include: org,new,neither.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of examples after filtering. Use 0 for all.",
    )
    parser.add_argument(
        "--selection",
        choices=("stratified", "ordered", "random"),
        default="stratified",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--forward-batch-size",
        type=int,
        default=4,
        help=(
            "Number of prompt variants per forward pass. Multiples of four are "
            "recommended because each sample has four variants."
        ),
    )
    parser.add_argument(
        "--enable-thinking",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--strip-trailing-special-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    args.prefixes = sorted(set([0, *args.prefixes]))
    args.label_names = {
        label.strip() for label in args.labels.split(",") if label.strip()
    }
    invalid_labels = args.label_names - {"org", "new", "neither"}
    if invalid_labels:
        parser.error(f"Unsupported labels: {sorted(invalid_labels)}")
    if args.limit < 0:
        parser.error("--limit must be nonnegative.")
    if args.forward_batch_size < 4:
        parser.error("--forward-batch-size must be at least 4.")
    if args.max_response_tokens < 1:
        parser.error("--max-response-tokens must be positive.")
    return args


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_triplets(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    records: dict[tuple[int, int], dict[str, Any]] = {}
    for record in read_jsonl(path):
        key = (int(record["conflict_type_idx"]), int(record["sample_id"]))
        if key in records:
            raise ValueError(f"Duplicate triplet key {key} in {path}")
        records[key] = record
    return records


def matching_response_path(
    evaluation_path: Path,
    evaluations_root: Path,
    responses_root: Path,
) -> Path:
    return responses_root / evaluation_path.relative_to(evaluations_root)


def discover_items(
    triplets: dict[tuple[int, int], dict[str, Any]],
    responses_root: Path,
    evaluations_root: Path,
    label_names: set[str],
) -> list[PilotItem]:
    items: list[PilotItem] = []
    for evaluation_path in sorted(evaluations_root.rglob("*.json")):
        if evaluation_path.name == "results.json":
            continue
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        label = int(evaluation.get("label", -1))
        label_name = LABEL_NAMES.get(label, "unknown")
        if label_name not in label_names:
            continue

        key = (
            int(evaluation["conflict_type_idx"]),
            int(evaluation["sample_id"]),
        )
        if key not in triplets:
            raise KeyError(f"No triplet record for evaluation key {key}")

        response_path = matching_response_path(
            evaluation_path,
            evaluations_root,
            responses_root,
        )
        if not response_path.exists():
            raise FileNotFoundError(
                f"Missing response file corresponding to {evaluation_path}: "
                f"{response_path}"
            )
        response = json.loads(response_path.read_text(encoding="utf-8"))
        if (
            int(response["conflict_type_idx"]),
            int(response["sample_id"]),
        ) != key:
            raise ValueError(f"Response/evaluation key mismatch at {response_path}")

        items.append(
            PilotItem(
                key=key,
                triplet=triplets[key],
                response=response["response"],
                label=label,
                label_name=label_name,
                response_path=str(response_path),
                evaluation_path=str(evaluation_path),
            )
        )
    return items


def select_items(
    items: list[PilotItem],
    limit: int,
    strategy: str,
    seed: int,
) -> list[PilotItem]:
    if not limit or limit >= len(items):
        return sorted(items, key=lambda item: item.key)

    rng = random.Random(seed)
    if strategy == "ordered":
        return sorted(items, key=lambda item: item.key)[:limit]
    if strategy == "random":
        selected = list(items)
        rng.shuffle(selected)
        return sorted(selected[:limit], key=lambda item: item.key)

    groups: dict[tuple[int, str], list[PilotItem]] = defaultdict(list)
    for item in items:
        groups[(item.key[0], item.label_name)].append(item)
    for group in groups.values():
        rng.shuffle(group)

    selected: list[PilotItem] = []
    group_keys = sorted(groups)
    while len(selected) < limit:
        made_progress = False
        for group_key in group_keys:
            if groups[group_key] and len(selected) < limit:
                selected.append(groups[group_key].pop())
                made_progress = True
        if not made_progress:
            break
    return sorted(selected, key=lambda item: item.key)


def decoder_layers(model: Any) -> Any:
    candidates = [
        ("model.layers", lambda value: value.model.layers),
        ("model.language_model.layers", lambda value: value.model.language_model.layers),
        ("language_model.layers", lambda value: value.language_model.layers),
        ("transformer.h", lambda value: value.transformer.h),
    ]
    for _, accessor in candidates:
        try:
            layers = accessor(model)
        except AttributeError:
            continue
        if layers is not None:
            return layers
    raise AttributeError("Could not locate decoder layers on the loaded model.")


def forward_module_without_lm_head(model: Any) -> Any:
    if hasattr(model, "model") and isinstance(model.model, torch.nn.Module):
        return model.model
    if hasattr(model, "transformer") and isinstance(model.transformer, torch.nn.Module):
        return model.transformer
    return model


def input_device(model: Any) -> torch.device:
    try:
        embeddings = model.get_input_embeddings()
        if embeddings is not None:
            return embeddings.weight.device
    except (AttributeError, NotImplementedError):
        pass
    return next(model.parameters()).device


def resolve_model_path(model: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, model) if cache_dir else model


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any, str]:
    model_path = resolve_model_path(args.model, args.cache_dir)
    model_kwargs = {
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.torch_dtype:
        model_kwargs["torch_dtype"] = args.torch_dtype

    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    except (OSError, ValueError, TypeError) as causal_error:
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError:
            raise causal_error
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            **model_kwargs,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=args.trust_remote_code,
        legacy=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0
    return model.eval(), tokenizer, model_path


def format_prompt_ids(
    tokenizer: Any,
    instruction: str,
    enable_thinking: bool,
) -> list[int]:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": instruction},
    ]
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def response_token_ids(
    tokenizer: Any,
    response: str,
    max_tokens: int,
    strip_trailing_special_tokens: bool,
) -> list[int]:
    token_ids = tokenizer.encode(response, add_special_tokens=False)
    if strip_trailing_special_tokens:
        special_ids = set(tokenizer.all_special_ids)
        while token_ids and token_ids[-1] in special_ids:
            token_ids.pop()
    return token_ids[:max_tokens]


def prefix_slots(
    prompt_length: int,
    response_length: int,
    requested_prefixes: list[int],
    include_final: bool,
) -> tuple[list[int], list[int], list[str]]:
    actual_prefixes: list[int] = []
    positions: list[int] = []
    slot_names: list[str] = []
    for prefix in requested_prefixes:
        actual_prefixes.append(prefix if prefix <= response_length else -1)
        positions.append(
            prompt_length - 1
            if prefix == 0
            else min(prompt_length + prefix - 1, prompt_length + response_length - 1)
        )
        slot_names.append(f"prefix_{prefix}")
    if include_final:
        actual_prefixes.append(response_length)
        positions.append(
            prompt_length - 1
            if response_length == 0
            else prompt_length + response_length - 1
        )
        slot_names.append("final")
    return actual_prefixes, positions, slot_names


def pad_sequence_records(
    records: list[dict[str, Any]],
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_length = max(len(record["input_ids"]) for record in records)
    input_ids = torch.full(
        (len(records), max_length),
        pad_token_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros(
        (len(records), max_length),
        dtype=torch.long,
    )
    positions = torch.empty(
        (len(records), len(records[0]["positions"])),
        dtype=torch.long,
    )
    for row, record in enumerate(records):
        length = len(record["input_ids"])
        input_ids[row, :length] = torch.tensor(record["input_ids"], dtype=torch.long)
        attention_mask[row, :length] = 1
        positions[row] = torch.tensor(record["positions"], dtype=torch.long)
    return (
        input_ids.to(device),
        attention_mask.to(device),
        positions.to(device),
    )


def hidden_tensor_from_hook_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    raise TypeError(f"Unsupported decoder-layer output type: {type(output)!r}")


def forward_selected_positions(
    model: Any,
    records: list[dict[str, Any]],
    layer_indices: list[int],
    pad_token_id: int,
) -> dict[int, np.ndarray]:
    layers = decoder_layers(model)
    invalid = [layer for layer in layer_indices if layer < 0 or layer >= len(layers)]
    if invalid:
        raise ValueError(
            f"Requested invalid layers {invalid}; model has {len(layers)} decoder layers."
        )

    device = input_device(model)
    input_ids, attention_mask, positions = pad_sequence_records(
        records,
        pad_token_id,
        device,
    )
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = hidden_tensor_from_hook_output(output)
            gather_index = positions.to(hidden.device).unsqueeze(-1).expand(
                -1,
                -1,
                hidden.shape[-1],
            )
            captured[layer_index] = hidden.gather(1, gather_index).detach().float().cpu()

        return hook

    for layer_index in layer_indices:
        handles.append(layers[layer_index].register_forward_hook(make_hook(layer_index)))

    forward_module = forward_module_without_lm_head(model)
    forward_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    signature = inspect.signature(forward_module.forward)
    if "use_cache" in signature.parameters:
        forward_kwargs["use_cache"] = False
    if "return_dict" in signature.parameters:
        forward_kwargs["return_dict"] = True

    try:
        with torch.inference_mode():
            forward_module(**forward_kwargs)
    finally:
        for handle in handles:
            handle.remove()

    missing = set(layer_indices) - set(captured)
    if missing:
        raise RuntimeError(f"Hooks did not capture requested layers: {sorted(missing)}")
    return {layer: tensor.numpy() for layer, tensor in captured.items()}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= EPS:
        return math.nan
    return float(np.dot(a, b) / denominator)


def local_span_metrics(
    org_vector: np.ndarray,
    new_vector: np.ndarray,
    conflict_vector: np.ndarray,
) -> dict[str, float]:
    org_norm = float(np.linalg.norm(org_vector))
    new_norm = float(np.linalg.norm(new_vector))
    conflict_norm = float(np.linalg.norm(conflict_vector))
    if min(org_norm, new_norm, conflict_norm) <= EPS:
        return {
            "org_vector_norm": org_norm,
            "new_vector_norm": new_norm,
            "conflict_vector_norm": conflict_norm,
            "basis_cosine": math.nan,
            "basis_condition": math.inf,
            "org_partial": math.nan,
            "new_partial": math.nan,
            "org_alignment": math.nan,
            "new_alignment": math.nan,
            "subspace_fraction": math.nan,
            "residual_ratio": math.nan,
            "additive_cosine": math.nan,
            "additive_error_ratio": math.nan,
            "dominance_new_minus_org": math.nan,
        }

    unit_org = org_vector / org_norm
    unit_new = new_vector / new_norm
    basis = np.column_stack([unit_org, unit_new])
    coefficients, _, _, singular_values = np.linalg.lstsq(
        basis,
        conflict_vector,
        rcond=1e-6,
    )
    reconstruction = basis @ coefficients
    residual = conflict_vector - reconstruction
    projected_norm = float(np.linalg.norm(reconstruction))
    residual_norm = float(np.linalg.norm(residual))
    basis_condition = (
        float(singular_values[0] / singular_values[-1])
        if len(singular_values) == 2 and singular_values[-1] > EPS
        else math.inf
    )
    additive = org_vector + new_vector
    additive_norm = float(np.linalg.norm(additive))
    additive_denominator = max(conflict_norm + additive_norm, EPS)
    org_partial = float(coefficients[0] / conflict_norm)
    new_partial = float(coefficients[1] / conflict_norm)
    return {
        "org_vector_norm": org_norm,
        "new_vector_norm": new_norm,
        "conflict_vector_norm": conflict_norm,
        "basis_cosine": cosine(org_vector, new_vector),
        "basis_condition": basis_condition,
        "org_partial": org_partial,
        "new_partial": new_partial,
        "org_alignment": cosine(conflict_vector, org_vector),
        "new_alignment": cosine(conflict_vector, new_vector),
        "subspace_fraction": min(
            1.0,
            max(0.0, projected_norm**2 / max(conflict_norm**2, EPS)),
        ),
        "residual_ratio": residual_norm / conflict_norm,
        "additive_cosine": cosine(conflict_vector, additive),
        "additive_error_ratio": float(
            np.linalg.norm(conflict_vector - additive) / additive_denominator
        ),
        "dominance_new_minus_org": new_partial - org_partial,
    }


def endpoint_metrics(
    org_state: np.ndarray,
    new_state: np.ndarray,
    conflict_state: np.ndarray,
) -> dict[str, float]:
    axis = new_state - org_state
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= EPS:
        return {
            "endpoint_separation": axis_norm,
            "axis_position": math.nan,
            "off_axis_ratio": math.nan,
            "distance_to_org_ratio": math.nan,
            "distance_to_new_ratio": math.nan,
        }
    axis_position = float(
        np.dot(conflict_state - org_state, axis) / max(axis_norm**2, EPS)
    )
    closest = org_state + axis_position * axis
    return {
        "endpoint_separation": axis_norm,
        "axis_position": axis_position,
        "off_axis_ratio": float(np.linalg.norm(conflict_state - closest) / axis_norm),
        "distance_to_org_ratio": float(
            np.linalg.norm(conflict_state - org_state) / axis_norm
        ),
        "distance_to_new_ratio": float(
            np.linalg.norm(conflict_state - new_state) / axis_norm
        ),
    }


def behavior_metrics(metrics: dict[str, float], label_name: str) -> dict[str, float]:
    if label_name == "org":
        selected_partial = metrics["org_partial"]
        rejected_partial = metrics["new_partial"]
        selected_alignment = metrics["org_alignment"]
        rejected_alignment = metrics["new_alignment"]
    elif label_name == "new":
        selected_partial = metrics["new_partial"]
        rejected_partial = metrics["org_partial"]
        selected_alignment = metrics["new_alignment"]
        rejected_alignment = metrics["org_alignment"]
    else:
        return {
            "selected_partial": math.nan,
            "rejected_partial": math.nan,
            "selected_margin": math.nan,
            "selected_alignment": math.nan,
            "rejected_alignment": math.nan,
            "selected_alignment_margin": math.nan,
        }
    return {
        "selected_partial": selected_partial,
        "rejected_partial": rejected_partial,
        "selected_margin": selected_partial - rejected_partial,
        "selected_alignment": selected_alignment,
        "rejected_alignment": rejected_alignment,
        "selected_alignment_margin": selected_alignment - rejected_alignment,
    }


def compute_sample_rows(
    item: PilotItem,
    states: dict[str, dict[int, np.ndarray]],
    slot_prefixes: list[int],
    slot_names: list[str],
    layer_indices: list[int],
    response_tokens: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_prefixes: set[int] = set()
    valid_slots = []
    for slot_index, prefix in enumerate(slot_prefixes):
        if prefix < 0 or prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)
        valid_slots.append((slot_index, prefix, slot_names[slot_index]))

    prompt_slot = next(
        slot_index for slot_index, prefix, _ in valid_slots if prefix == 0
    )
    for layer in layer_indices:
        seed_prompt = states["seed"][layer][prompt_slot]
        fixed_org = states["org"][layer][prompt_slot] - seed_prompt
        fixed_new = states["new"][layer][prompt_slot] - seed_prompt

        for slot_index, prefix, slot_name in valid_slots:
            seed_state = states["seed"][layer][slot_index]
            org_state = states["org"][layer][slot_index]
            new_state = states["new"][layer][slot_index]
            conflict_state = states["conflict"][layer][slot_index]
            conflict_vector = conflict_state - seed_state

            common = {
                "conflict_type_idx": item.key[0],
                "sample_id": item.key[1],
                "conflict_type": item.triplet.get("conflict_type"),
                "task": item.triplet.get("task"),
                "behavior_label": item.label,
                "behavior_label_name": item.label_name,
                "prefix_tokens": prefix,
                "prefix_slot": slot_name,
                "is_prompt": prefix == 0,
                "is_final": prefix == response_tokens,
                "layer": layer,
                "response_path": item.response_path,
                "evaluation_path": item.evaluation_path,
            }

            dynamic = local_span_metrics(
                org_state - seed_state,
                new_state - seed_state,
                conflict_vector,
            )
            dynamic.update(endpoint_metrics(org_state, new_state, conflict_state))
            dynamic.update(behavior_metrics(dynamic, item.label_name))
            rows.append({**common, "basis_mode": "dynamic_local", **dynamic})

            fixed = local_span_metrics(fixed_org, fixed_new, conflict_vector)
            fixed.update(
                {
                    "endpoint_separation": math.nan,
                    "axis_position": math.nan,
                    "off_axis_ratio": math.nan,
                    "distance_to_org_ratio": math.nan,
                    "distance_to_new_ratio": math.nan,
                }
            )
            fixed.update(behavior_metrics(fixed, item.label_name))
            rows.append({**common, "basis_mode": "fixed_prompt", **fixed})
    return rows


def build_sequence_records(
    items: list[PilotItem],
    tokenizer: Any,
    prefixes: list[int],
    include_final: bool,
    max_response_tokens: int,
    enable_thinking: bool,
    strip_trailing_special_tokens: bool,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    sample_info: dict[tuple[int, int], dict[str, Any]] = {}
    for item in items:
        response_ids = response_token_ids(
            tokenizer,
            item.response,
            max_response_tokens,
            strip_trailing_special_tokens,
        )
        for variant, field in VARIANT_FIELDS.items():
            prompt_ids = format_prompt_ids(
                tokenizer,
                item.triplet[field],
                enable_thinking,
            )
            actual_prefixes, positions, slot_names = prefix_slots(
                len(prompt_ids),
                len(response_ids),
                prefixes,
                include_final,
            )
            records.append(
                {
                    "key": item.key,
                    "variant": variant,
                    "input_ids": [*prompt_ids, *response_ids],
                    "positions": positions,
                }
            )
            if item.key not in sample_info:
                sample_info[item.key] = {
                    "actual_prefixes": actual_prefixes,
                    "slot_names": slot_names,
                    "response_tokens": len(response_ids),
                }
    return records, sample_info


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_rows(handle: Any, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        handle.write(
            json.dumps(
                {key: json_safe(value) for key, value in row.items()},
                ensure_ascii=False,
            )
            + "\n"
        )
    handle.flush()


def completed_keys(path: Path) -> set[tuple[int, int]]:
    if not path.exists():
        return set()
    keys = set()
    for row in read_jsonl(path):
        keys.add((int(row["conflict_type_idx"]), int(row["sample_id"])))
    return keys


def materialize_csv(jsonl_path: Path, csv_path: Path) -> int:
    import pandas as pd

    frame = pd.read_json(jsonl_path, lines=True)
    frame.to_csv(csv_path, index=False)
    return len(frame)


def run_self_test() -> None:
    org = np.array([2.0, 0.0, 0.0])
    new = np.array([0.0, 3.0, 0.0])
    conflict = 0.25 * org + 0.75 * new + np.array([0.0, 0.0, 1.0])
    metrics = local_span_metrics(org, new, conflict)
    assert np.isclose(metrics["org_partial"], 0.25 * 2 / np.linalg.norm(conflict))
    assert np.isclose(metrics["new_partial"], 0.75 * 3 / np.linalg.norm(conflict))
    assert metrics["residual_ratio"] > 0

    endpoints = endpoint_metrics(
        np.array([0.0, 0.0]),
        np.array([2.0, 0.0]),
        np.array([1.5, 1.0]),
    )
    assert np.isclose(endpoints["axis_position"], 0.75)
    assert np.isclose(endpoints["off_axis_ratio"], 0.5)
    print("Geometry self-test passed.")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "trajectory_geometry.jsonl"
    csv_path = args.output_dir / "trajectory_geometry.csv"
    metadata_path = args.output_dir / "metadata.json"
    if args.overwrite:
        jsonl_path.unlink(missing_ok=True)
        csv_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
    elif jsonl_path.exists() and not args.resume:
        raise FileExistsError(
            f"{jsonl_path} already exists. Use --resume or --overwrite."
        )

    triplets = load_triplets(args.triplets)
    discovered = discover_items(
        triplets,
        args.responses_root,
        args.evaluations_root,
        args.label_names,
    )
    selected = select_items(
        discovered,
        args.limit,
        args.selection,
        args.random_seed,
    )
    done = completed_keys(jsonl_path) if args.resume else set()
    pending = [item for item in selected if item.key not in done]
    print(
        f"Discovered {len(discovered)} eligible examples; selected {len(selected)}; "
        f"pending {len(pending)}."
    )
    print("Selected labels:", Counter(item.label_name for item in selected))
    print(
        "Selected conflict types:",
        Counter(item.key[0] for item in selected),
    )
    if not pending:
        rows = materialize_csv(jsonl_path, csv_path)
        print(f"No pending samples. Materialized {rows} rows to {csv_path}")
        return

    model, tokenizer, resolved_model = load_model_and_tokenizer(args)
    samples_per_forward = max(1, args.forward_batch_size // len(VARIANT_FIELDS))
    output_mode = "a" if args.resume and jsonl_path.exists() else "w"

    with jsonl_path.open(output_mode, encoding="utf-8") as output_handle:
        for start in tqdm(
            range(0, len(pending), samples_per_forward),
            desc="Trajectory batches",
        ):
            batch_items = pending[start : start + samples_per_forward]
            sequence_records, sample_info = build_sequence_records(
                batch_items,
                tokenizer,
                args.prefixes,
                args.include_final,
                args.max_response_tokens,
                args.enable_thinking,
                args.strip_trailing_special_tokens,
            )
            captured = forward_selected_positions(
                model,
                sequence_records,
                args.layers,
                tokenizer.pad_token_id,
            )

            states_by_key: dict[
                tuple[int, int],
                dict[str, dict[int, np.ndarray]],
            ] = defaultdict(lambda: defaultdict(dict))
            for sequence_index, record in enumerate(sequence_records):
                for layer in args.layers:
                    states_by_key[record["key"]][record["variant"]][layer] = captured[
                        layer
                    ][sequence_index]

            for item in batch_items:
                info = sample_info[item.key]
                rows = compute_sample_rows(
                    item,
                    states_by_key[item.key],
                    info["actual_prefixes"],
                    info["slot_names"],
                    args.layers,
                    info["response_tokens"],
                )
                for row in rows:
                    row["response_tokens"] = info["response_tokens"]
                write_rows(output_handle, rows)

    row_count = materialize_csv(jsonl_path, csv_path)
    metadata = {
        "script": Path(__file__).name,
        "model": args.model,
        "resolved_model": resolved_model,
        "enable_thinking": args.enable_thinking,
        "triplets": str(args.triplets.resolve()),
        "responses_root": str(args.responses_root.resolve()),
        "evaluations_root": str(args.evaluations_root.resolve()),
        "layers": args.layers,
        "prefixes": args.prefixes,
        "include_final": args.include_final,
        "max_response_tokens": args.max_response_tokens,
        "selection": args.selection,
        "random_seed": args.random_seed,
        "selected_examples": len(selected),
        "selected_label_counts": dict(Counter(item.label_name for item in selected)),
        "selected_type_counts": {
            str(key): value
            for key, value in Counter(item.key[0] for item in selected).items()
        },
        "row_count": row_count,
        "basis_modes": ["dynamic_local", "fixed_prompt"],
        "notes": {
            "dynamic_local": (
                "At every response prefix, construct sample-specific org/new vectors "
                "relative to the seed prompt under the same teacher-forced response."
            ),
            "fixed_prompt": (
                "Hold the sample-specific prompt-end org/new vectors fixed and test "
                "whether they remain readable during generation."
            ),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {row_count} rows to {csv_path}")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
