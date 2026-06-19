# -*- coding: UTF-8 -*-

import argparse
import json
import os
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


INSTRUCTION_FIELDS = {
    "original": "original_instruction",
    "new": "new_instruction",
    "conflict": "conflict_instruction",
}


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "y", "t"}:
        return True
    if normalized in {"0", "false", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


def hyphen_separated_values(value):
    return [item for item in value.split("-") if item]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate responses for the original, new, and conflicting "
            "two-constraint ConInstruct prompts with vLLM."
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
        default="./outputs/constraint_triplets",
    )
    parser.add_argument(
        "--conditions",
        default="original-new-conflict",
        help="Hyphen-separated subset of: original, new, conflict.",
    )
    parser.add_argument(
        "--conflict_type_idx",
        default="1-2-3-4-5-6-7-8-9",
        help="Hyphen-separated conflict type ids.",
    )
    parser.add_argument("--restart_idx", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument(
        "--enable_thinking",
        "--enable-thinking",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help=(
            "Set Qwen thinking mode at runtime. Accepts true or false; passing "
            "the flag without a value means true."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate outputs that already exist.",
    )
    args = parser.parse_args()

    args.conditions = hyphen_separated_values(args.conditions)
    invalid_conditions = set(args.conditions) - set(INSTRUCTION_FIELDS)
    if invalid_conditions:
        parser.error(
            "Unsupported conditions: "
            + ", ".join(sorted(invalid_conditions))
        )

    try:
        args.conflict_type_indices = {
            int(value)
            for value in hyphen_separated_values(args.conflict_type_idx)
        }
    except ValueError:
        parser.error("--conflict_type_idx must contain hyphen-separated integers.")
    return args


def model_path(args):
    return os.path.join(args.cache_dir, args.model) if args.cache_dir else args.model


def output_path(args, item, condition):
    thinking_mode = "thinking" if args.enable_thinking else "non_thinking"
    return (
        Path(args.output_root)
        / args.model
        / thinking_mode
        / condition
        / f"type_{item['conflict_type_idx']}"
        / f"{item['sample_id']}.json"
    )


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def format_prompt(tokenizer, instruction, enable_thinking):
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


def build_generation_items(args, tokenizer):
    generation_items = []
    for item in read_jsonl(args.input_filename):
        sample_id = item["sample_id"]
        if sample_id <= args.restart_idx:
            continue
        if args.max_samples is not None and sample_id > args.max_samples:
            continue
        if item["conflict_type_idx"] not in args.conflict_type_indices:
            continue

        for condition in args.conditions:
            destination = output_path(args, item, condition)
            if destination.exists() and not args.overwrite:
                continue

            instruction = item[INSTRUCTION_FIELDS[condition]]
            generation_items.append(
                {
                    "source": item,
                    "condition": condition,
                    "instruction": instruction,
                    "prompt": format_prompt(
                        tokenizer,
                        instruction,
                        args.enable_thinking,
                    ),
                    "output_path": destination,
                }
            )
    return generation_items


def write_output(args, generation_item, response):
    source = generation_item["source"]
    output = {
        "model": args.model,
        "enable_thinking": args.enable_thinking,
        "sample_id": source["sample_id"],
        "task": source.get("task"),
        "condition": generation_item["condition"],
        "conflict_type_idx": source["conflict_type_idx"],
        "conflict_type": source["conflict_type"],
        "seed_instruction": source["seed_instruction"],
        "org_constraint": source["org_constraint"],
        "new_constraint": source["new_constraint"],
        "instruction": generation_item["instruction"],
        "response": response,
    }
    destination = generation_item["output_path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    print(args)

    path = model_path(args)
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=args.trust_remote_code,
    )
    llm = LLM(
        model=path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    generation_items = build_generation_items(args, tokenizer)
    print(f"Generating {len(generation_items)} responses")

    for start in range(0, len(generation_items), args.batch_size):
        batch_items = generation_items[start:start + args.batch_size]
        outputs = llm.generate(
            [item["prompt"] for item in batch_items],
            sampling_params,
            use_tqdm=False,
        )
        for item, output in zip(batch_items, outputs):
            write_output(args, item, output.outputs[0].text)

        completed = min(start + len(batch_items), len(generation_items))
        print(f"Completed {completed}/{len(generation_items)}")


if __name__ == "__main__":
    main()
