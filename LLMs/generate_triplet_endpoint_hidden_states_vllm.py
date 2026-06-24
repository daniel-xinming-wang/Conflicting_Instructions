# -*- coding: UTF-8 -*-

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


INSTRUCTION_FIELDS = {
    "original": "original_instruction",
    "new": "new_instruction",
    "conflict": "conflict_instruction",
}

THINKING_MODES = {
    "thinking": True,
    "non_thinking": False,
}


def hyphen_separated_values(value: str) -> list[str]:
    return [item for item in value.split("-") if item]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate thinking and non-thinking ConInstruct responses with "
            "vLLM and save the textual model input ending at the "
            "reasoning-to-final boundary for a later Transformers forward pass."
        )
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--cache_dir", default="")
    parser.add_argument(
        "--input_filename",
        default="./datasets/constraint_triplets.jsonl",
    )
    parser.add_argument(
        "--output_root",
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
    parser.add_argument("--restart_idx", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--max_model_len", type=int, default=65536)
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--boundary_mode",
        choices=("auto", "qwen_think", "gemma_channel", "prompt_end"),
        default="auto",
    )
    parser.add_argument("--boundary_token", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.thinking_modes = hyphen_separated_values(args.thinking_modes)
    invalid_modes = set(args.thinking_modes) - set(THINKING_MODES)
    if invalid_modes:
        parser.error(f"Unsupported thinking modes: {sorted(invalid_modes)}")

    args.conditions = hyphen_separated_values(args.conditions)
    invalid_conditions = set(args.conditions) - set(INSTRUCTION_FIELDS)
    if invalid_conditions:
        parser.error(f"Unsupported conditions: {sorted(invalid_conditions)}")

    try:
        args.conflict_type_indices = {
            int(value)
            for value in hyphen_separated_values(args.conflict_type_idx)
        }
    except ValueError:
        parser.error("--conflict_type_idx must contain hyphen-separated integers.")
    return args


def model_path(args: argparse.Namespace) -> str:
    return os.path.join(args.cache_dir, args.model) if args.cache_dir else args.model


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def existing_response_is_valid(
    path: Path,
    expected_boundary_mode: str,
) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as file:
            record = json.load(file)
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(record, dict) or not isinstance(record.get("response"), str):
        return False
    endpoint_metadata = record.get("endpoint_metadata")
    if not isinstance(endpoint_metadata, dict):
        return False
    return endpoint_metadata.get("boundary_mode") == expected_boundary_mode



def model_family(model_name: str) -> str:
    normalized = model_name.lower()
    if "gemma" in normalized:
        return "gemma"
    if "qwen" in normalized:
        return "qwen"
    return "unknown"


def exact_token_ids(tokenizer: Any, token_text: str) -> list[int]:
    vocab = tokenizer.get_vocab()
    if token_text in vocab:
        return [int(vocab[token_text])]

    token_id = tokenizer.convert_tokens_to_ids(token_text)
    if token_id is not None and token_id != tokenizer.unk_token_id:
        return [int(token_id)]

    token_ids = tokenizer.encode(token_text, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Boundary token produced no token IDs: {token_text!r}")
    return [int(token_id) for token_id in token_ids]


def find_last_subsequence(
    sequence: list[int],
    target: list[int],
) -> int | None:
    if not target or len(target) > len(sequence):
        return None
    for start in range(len(sequence) - len(target), -1, -1):
        if sequence[start : start + len(target)] == target:
            return start
    return None


def resolve_boundary_token_ids(
    args: argparse.Namespace,
    tokenizer: Any,
    thinking_mode: str | None = None,
) -> tuple[str, str | None, list[int] | None]:
    if args.boundary_token is not None:
        return (
            "explicit_token",
            args.boundary_token,
            exact_token_ids(tokenizer, args.boundary_token),
        )

    mode = args.boundary_mode
    if mode == "auto":
        family = model_family(args.model)
        if family == "qwen":
            mode = "qwen_think"
        elif family == "gemma":
            if thinking_mode == "non_thinking":
                mode = "prompt_end"
            else:
                mode = "gemma_channel"
        else:
            raise ValueError(
                f"Cannot infer boundary for {args.model!r}; "
                "pass --boundary_token or --boundary_mode."
            )

    if mode == "prompt_end":
        return mode, None, None
    boundary_text = {
        "qwen_think": "</think>",
        "gemma_channel": "<channel|>",
    }[mode]
    return mode, boundary_text, exact_token_ids(tokenizer, boundary_text)


def format_prompt(
    tokenizer: Any,
    instruction: str,
    enable_thinking: bool,
) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": instruction},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def sample_dir(
    args: argparse.Namespace,
    item: dict[str, Any],
    thinking_mode: str,
    condition: str,
) -> Path:
    return (
        Path(args.output_root)
        / args.model
        / thinking_mode
        / condition
        / f"type_{item['conflict_type_idx']}"
    )


def build_generation_items(
    args: argparse.Namespace,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for source in read_jsonl(args.input_filename):
        sample_id = int(source["sample_id"])
        if sample_id <= args.restart_idx:
            continue
        if args.max_samples is not None and sample_id > args.max_samples:
            continue
        if source["conflict_type_idx"] not in args.conflict_type_indices:
            continue

        for thinking_mode in args.thinking_modes:
            enable_thinking = THINKING_MODES[thinking_mode]
            for condition in args.conditions:
                destination_dir = sample_dir(
                    args,
                    source,
                    thinking_mode,
                    condition,
                )
                response_path = destination_dir / "responses" / f"{sample_id}.json"
                boundary_mode, _, _ = resolve_boundary_token_ids(
                    args,
                    tokenizer,
                    thinking_mode,
                )
                if (
                    not args.overwrite
                    and existing_response_is_valid(response_path, boundary_mode)
                ):
                    continue

                instruction = source[INSTRUCTION_FIELDS[condition]]
                items.append(
                    {
                        "source": source,
                        "thinking_mode": thinking_mode,
                        "enable_thinking": enable_thinking,
                        "condition": condition,
                        "instruction": instruction,
                        "prompt": format_prompt(
                            tokenizer,
                            instruction,
                            enable_thinking,
                        ),
                        "response_path": response_path,
                    }
                )
    return items


def build_sampling_params(
    llm: Any,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any]]:
    sampling_params = llm.get_default_sampling_params()
    source = "model_generation_config"

    family = model_family(args.model)
    if family == "qwen":
        source = "qwen_model_card_shared_cot_control"
        sampling_params.temperature = 1.0
        sampling_params.top_p = 0.95
        sampling_params.top_k = 20
        sampling_params.min_p = 0.0
        sampling_params.presence_penalty = 1.5
        sampling_params.repetition_penalty = 1.0
    elif family == "gemma":
        source = "gemma_huggingface_generation_config_shared_cot_control"
        sampling_params.temperature = 1.0
        sampling_params.top_p = 0.95
        sampling_params.top_k = 64
        sampling_params.min_p = 0.0
        sampling_params.presence_penalty = 0.0
        sampling_params.repetition_penalty = 1.0

    sampling_params.max_tokens = args.max_tokens
    sampling_params.seed = args.seed
    resolved = {
        "source": source,
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p,
        "top_k": sampling_params.top_k,
        "min_p": sampling_params.min_p,
        "presence_penalty": sampling_params.presence_penalty,
        "repetition_penalty": sampling_params.repetition_penalty,
        "max_tokens": sampling_params.max_tokens,
        "seed": sampling_params.seed,
    }
    return sampling_params, resolved


def save_generation(
    args: argparse.Namespace,
    tokenizer: Any,
    item: dict[str, Any],
    request_output: Any,
    sampling_metadata: dict[str, Any],
    boundary_mode: str,
    boundary_text: str | None,
    boundary_token_ids: list[int] | None,
) -> str:
    completion = request_output.outputs[0]
    prompt_ids = [int(token_id) for token_id in request_output.prompt_token_ids]
    completion_ids = [int(token_id) for token_id in completion.token_ids]
    full_ids = prompt_ids + completion_ids

    if boundary_mode == "prompt_end":
        boundary_start = len(prompt_ids) - 1
        boundary_end = boundary_start
    else:
        boundary_start = find_last_subsequence(full_ids, boundary_token_ids)
        boundary_end = (
            boundary_start + len(boundary_token_ids) - 1
            if boundary_start is not None
            else None
        )

    if boundary_end is None:
        endpoint_status = "boundary_missing"
        boundary_source = None
        prefix_length = None
        endpoint_input_text = None
        reasoning = None
        final_answer = None
    else:
        boundary_source = (
            "prompt" if boundary_end < len(prompt_ids) else "completion"
        )
        expected_source = (
            "completion" if item["thinking_mode"] == "thinking" else "prompt"
        )
        if boundary_source != expected_source:
            endpoint_status = "unexpected_boundary_source"
        else:
            endpoint_status = "text_saved"
        endpoint_input_text = tokenizer.decode(
            full_ids[: boundary_end + 1],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        prefix_length = boundary_end + 1

        if boundary_source == "completion":
            completion_boundary_start = boundary_start - len(prompt_ids)
            completion_boundary_end = boundary_end - len(prompt_ids)
            reasoning_ids = completion_ids[:completion_boundary_start]
            final_answer_ids = completion_ids[completion_boundary_end + 1 :]
            reasoning = tokenizer.decode(
                reasoning_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            final_answer = tokenizer.decode(
                final_answer_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        else:
            # In non-thinking mode, the endpoint is in the prompt. For Qwen this
            # may correspond to an empty reasoning delimiter; for Gemma it is
            # the generation prompt end with no explicit thought channel.
            reasoning = ""
            final_answer = tokenizer.decode(
                completion_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

    raw_response = tokenizer.decode(
        completion_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    source = item["source"]
    record = {
        "model": args.model,
        "thinking_mode": item["thinking_mode"],
        "enable_thinking": item["enable_thinking"],
        "sample_id": source["sample_id"],
        "task": source.get("task"),
        "condition": item["condition"],
        "conflict_type_idx": source["conflict_type_idx"],
        "conflict_type": source["conflict_type"],
        "seed_instruction": source["seed_instruction"],
        "org_constraint": source["org_constraint"],
        "new_constraint": source["new_constraint"],
        "instruction": item["instruction"],
        "response": raw_response,
        "reasoning": reasoning,
        "final_answer": final_answer,
        "prompt_length": len(prompt_ids),
        "completion_length": len(completion_ids),
        "finish_reason": completion.finish_reason,
        "stop_reason": completion.stop_reason,
        "endpoint_input_text": endpoint_input_text,
        "endpoint_metadata": {
            "status": endpoint_status,
            "boundary_mode": boundary_mode,
            "boundary_text": boundary_text,
            "boundary_token_ids": boundary_token_ids,
            "boundary_start": boundary_start,
            "boundary_end": boundary_end,
            "boundary_source": boundary_source,
            "prefix_length": prefix_length,
        },
        "sampling": sampling_metadata,
    }
    response_path: Path = item["response_path"]
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return endpoint_status


def main() -> None:
    args = parse_args()
    from vllm import LLM

    resolved_model_path = model_path(args)
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model_path,
        trust_remote_code=args.trust_remote_code,
    )
    boundary_specs = {
        thinking_mode: resolve_boundary_token_ids(args, tokenizer, thinking_mode)
        for thinking_mode in args.thinking_modes
    }
    print(f"Boundary specs by thinking mode: {boundary_specs}")

    llm = LLM(
        model=resolved_model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        max_model_len=args.max_model_len,
    )
    sampling_params, sampling_metadata = build_sampling_params(llm, args)
    print(f"Sampling: {sampling_metadata}")

    generation_items = build_generation_items(args, tokenizer)
    print(f"Generating {len(generation_items)} responses")
    status_counts: dict[str, int] = {}

    for start in range(0, len(generation_items), args.batch_size):
        batch_items = generation_items[start : start + args.batch_size]
        outputs = llm.generate(
            [item["prompt"] for item in batch_items],
            sampling_params,
            use_tqdm=False,
        )
        for item, output in zip(batch_items, outputs):
            boundary_mode, boundary_text, boundary_token_ids = boundary_specs[
                item["thinking_mode"]
            ]
            status = save_generation(
                args=args,
                tokenizer=tokenizer,
                item=item,
                request_output=output,
                sampling_metadata=sampling_metadata,
                boundary_mode=boundary_mode,
                boundary_text=boundary_text,
                boundary_token_ids=boundary_token_ids,
            )
            status_counts[status] = status_counts.get(status, 0) + 1

        completed = min(start + len(batch_items), len(generation_items))
        print(
            f"Completed {completed}/{len(generation_items)}; "
            f"statuses={status_counts}"
        )


if __name__ == "__main__":
    main()
