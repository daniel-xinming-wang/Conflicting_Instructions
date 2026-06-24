from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub.errors import GatedRepoError


def hyphen_separated_values(value: str) -> list[str]:
    return [item for item in value.split("-") if item]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tokenize endpoint input text saved by "
            "generate_triplet_endpoint_hidden_states_vllm.py, run a forward "
            "pass, and save "
            "one [num_samples, hidden_size] .npy file per decoder layer."
        )
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--cache_dir", default="")
    parser.add_argument(
        "--input_root",
        default="./outputs/triplet_endpoint_hidden_states",
    )
    parser.add_argument(
        "--thinking_modes",
        default="thinking-non_thinking",
    )
    parser.add_argument(
        "--conditions",
        default="original-new-conflict",
    )
    parser.add_argument(
        "--conflict_type_idx",
        default="1-2-3-4-5-6-7-8-9",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--hidden_layers",
        default="all",
        help="'all' or hyphen-separated zero-based decoder layer indices.",
    )
    parser.add_argument(
        "--save_dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.thinking_modes = hyphen_separated_values(args.thinking_modes)
    args.conditions = hyphen_separated_values(args.conditions)
    args.conflict_type_indices = {
        int(value)
        for value in hyphen_separated_values(args.conflict_type_idx)
    }
    return args


def model_path(args: argparse.Namespace) -> str:
    return os.path.join(args.cache_dir, args.model) if args.cache_dir else args.model


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = model_path(args)
    kwargs = {
        "device_map": args.device_map,
        "torch_dtype": args.torch_dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    except (OSError, ValueError, GatedRepoError) as exc:
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError:
            raise exc
        model = AutoModelForImageTextToText.from_pretrained(path, **kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        path,
        padding_side="left",
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0
    return model.eval(), tokenizer


def num_hidden_layers(model: Any) -> int:
    if hasattr(model.config, "num_hidden_layers"):
        return int(model.config.num_hidden_layers)
    if hasattr(model.config, "text_config"):
        return int(model.config.text_config.num_hidden_layers)
    raise ValueError("Could not determine num_hidden_layers.")


def parse_hidden_layers(value: str, total_layers: int) -> list[int]:
    if value == "all":
        return list(range(total_layers))
    layers = [int(item) for item in hyphen_separated_values(value)]
    invalid = [layer for layer in layers if layer < 0 or layer >= total_layers]
    if invalid:
        raise ValueError(
            f"Hidden layers outside [0, {total_layers - 1}]: {invalid}"
        )
    return layers


def group_dir(
    args: argparse.Namespace,
    thinking_mode: str,
    condition: str,
    conflict_type_idx: int,
) -> Path:
    return (
        Path(args.input_root)
        / args.model
        / thinking_mode
        / condition
        / f"type_{conflict_type_idx}"
    )


def group_complete(
    directory: Path,
    hidden_layers: list[int],
) -> bool:
    return (
        (directory / "sample_ids.npy").exists()
        and (directory / "metadata.json").exists()
        and all(
            (directory / f"layer_{layer_idx}.npy").exists()
            for layer_idx in hidden_layers
        )
    )


def response_files(directory: Path) -> list[Path]:
    response_dir = directory / "responses"
    if not response_dir.exists():
        return []
    return sorted(response_dir.glob("*.json"), key=lambda path: int(path.stem))


def input_device(model: Any) -> torch.device:
    return model.get_input_embeddings().weight.device


def extract_group(
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    directory: Path,
    hidden_layers: list[int],
) -> None:
    files = response_files(directory)
    if not files:
        print(f"Skipping {directory}: no response records")
        return

    per_layer: dict[int, list[np.ndarray]] = {
        layer_idx: [] for layer_idx in hidden_layers
    }
    sample_ids: list[int] = []
    prefix_lengths: list[int] = []
    skipped_sample_ids: list[int] = []
    device = input_device(model)
    output_dtype = np.float16 if args.save_dtype == "float16" else np.float32

    for start in range(0, len(files), args.batch_size):
        batch_files = files[start : start + args.batch_size]
        records = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in batch_files
        ]
        valid_pairs = [
            (path, record)
            for path, record in zip(batch_files, records)
            if record.get("endpoint_input_text") is not None
            and record.get("endpoint_metadata", {}).get("status") == "text_saved"
        ]
        valid_paths = {path for path, _ in valid_pairs}
        skipped_sample_ids.extend(
            int(path.stem)
            for path in batch_files
            if path not in valid_paths
        )
        if not valid_pairs:
            continue
        batch_files = [pair[0] for pair in valid_pairs]
        endpoint_texts = [pair[1]["endpoint_input_text"] for pair in valid_pairs]
        sequences = [
            tokenizer.encode(text, add_special_tokens=False)
            for text in endpoint_texts
        ]
        max_length = max(len(sequence) for sequence in sequences)
        padded_ids = []
        attention_masks = []
        for sequence in sequences:
            pad_length = max_length - len(sequence)
            padded_ids.append(
                [tokenizer.pad_token_id] * pad_length + sequence
            )
            attention_masks.append([0] * pad_length + [1] * len(sequence))

        input_ids = torch.tensor(
            padded_ids,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.tensor(
            attention_masks,
            dtype=torch.long,
            device=device,
        )
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        decoder_states = outputs.hidden_states[1:]
        for layer_idx in hidden_layers:
            endpoint = decoder_states[layer_idx][:, -1, :]
            endpoint = endpoint.float().cpu().numpy().astype(output_dtype)
            per_layer[layer_idx].extend(endpoint)

        sample_ids.extend(int(path.stem) for path in batch_files)
        prefix_lengths.extend(len(sequence) for sequence in sequences)
        print(f"{directory}: processed {len(sample_ids)}/{len(files)}")

    if not sample_ids:
        print(f"Skipping {directory}: no valid endpoint input text")
        return

    directory.mkdir(parents=True, exist_ok=True)
    for layer_idx in hidden_layers:
        np.save(
            directory / f"layer_{layer_idx}.npy",
            np.stack(per_layer[layer_idx], axis=0),
        )
    np.save(
        directory / "sample_ids.npy",
        np.asarray(sample_ids, dtype=np.int64),
    )
    metadata = {
        "model": args.model,
        "num_samples": len(sample_ids),
        "sample_ids": sample_ids,
        "prefix_lengths": prefix_lengths,
        "skipped_sample_ids": skipped_sample_ids,
        "hidden_layers": hidden_layers,
        "save_dtype": args.save_dtype,
        "source": "generated_reasoning_text_transformers_forward",
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args)
    hidden_layers = parse_hidden_layers(
        args.hidden_layers,
        num_hidden_layers(model),
    )

    for thinking_mode in args.thinking_modes:
        for condition in args.conditions:
            for conflict_type_idx in sorted(args.conflict_type_indices):
                directory = group_dir(
                    args,
                    thinking_mode,
                    condition,
                    conflict_type_idx,
                )
                if (
                    group_complete(directory, hidden_layers)
                    and not args.overwrite
                ):
                    print(f"Skipping complete group: {directory}")
                    continue
                extract_group(
                    args=args,
                    model=model,
                    tokenizer=tokenizer,
                    directory=directory,
                    hidden_layers=hidden_layers,
                )


if __name__ == "__main__":
    main()
