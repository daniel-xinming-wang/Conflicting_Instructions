# -*- coding: UTF-8 -*-

"""Evaluate constraint-triplet generations with the existing API judge."""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

from evaluation import (
    api_key_dict,
    extract_label,
    extract_label_conflict_resolution_behavior,
)
from gpt4 import OpenAIGPT4


VALID_CONDITIONS = {"original", "new", "conflict"}
FAILURE_RESPONSE = "Fail to connect to OpenAI API."
EVALUATOR_VERSIONS = {
    "conflict_resolution": "final_answer_only_v1",
    "conflict_resolution_behavior": "final_answer_only_behavior_v1",
}
BEHAVIOR_LABEL_NAMES = {
    1: "direct_answer_without_acknowledging_conflict",
    2: "asks_for_clarification_without_answering",
    3: "self_resolves_conflict_and_answers",
    4: "other",
    -1: "invalid_judge_output",
}


def strip_generation_markers(text):
    """Remove model-protocol markers surrounding an already extracted answer."""
    text = text.strip()
    leading_markers = (
        "<|message|>",
        "<|channel>final",
        "<|channel>assistant",
        "<|start|>assistant",
    )
    changed = True
    while changed:
        changed = False
        for marker in leading_markers:
            if text.startswith(marker):
                text = text[len(marker):].lstrip()
                changed = True

    trailing_pattern = re.compile(
        r"(?:<turn\|>|<\|end\|>|<\|end_of_text\|>|"
        r"<\|endoftext\|>|<\|eot_id\|>)+\s*$"
    )
    return trailing_pattern.sub("", text).strip()


def extract_final_answer(data):
    """Return only the final answer that should be shown to the judge.

    Non-thinking generations are already direct answers. Thinking generations
    must contain a recognized end-of-reasoning/channel delimiter. We
    deliberately do not fall back to the full response when that delimiter is
    absent, because doing so would cause the judge to score the CoT itself.
    """
    response = data["response"]
    if not data["enable_thinking"]:
        answer = strip_generation_markers(response)
        if not answer:
            return None, "direct_response", "empty_final_answer"
        return answer, "direct_response", None

    # Qwen/DeepSeek-style explicit end-of-thinking delimiter.
    if "</think>" in response:
        answer = strip_generation_markers(response.rsplit("</think>", 1)[1])
        if not answer:
            return None, "after_end_think", "empty_final_answer"
        return answer, "after_end_think", None

    # Gemma-style raw channel decoding observed in the saved generations:
    # <|channel>thought ... <channel|>FINAL_ANSWER<turn|>
    if "<channel|>" in response:
        answer = strip_generation_markers(response.rsplit("<channel|>", 1)[1])
        if not answer:
            return None, "after_channel_boundary", "empty_final_answer"
        return answer, "after_channel_boundary", None

    # Harmony-style explicit final channel, for compatibility with models that
    # preserve the full assistant protocol in decoded token IDs.
    final_channel_markers = (
        "<|start|>assistant<|channel>final<|message|>",
        "<|channel>final<|message|>",
        "<|channel>final",
    )
    for marker in final_channel_markers:
        if marker in response:
            answer = strip_generation_markers(response.rsplit(marker, 1)[1])
            if not answer:
                return None, "after_final_channel", "empty_final_answer"
            return answer, "after_final_channel", None

    return None, "unrecognized_thinking_format", "missing_final_answer_delimiter"


def hyphen_separated_values(value):
    return [item for item in value.split("-") if item]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate original/new/conflict response files under "
            "outputs/constraint_triplets with an API judge."
        )
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help=(
            "Any directory under outputs/constraint_triplets, including the "
            "constraint_triplets root, a model directory, or a thinking-mode directory."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Evaluation output directory. By default, the first path component "
            "named 'outputs' is replaced with 'evaluation_outputs'."
        ),
    )
    parser.add_argument(
        "--task",
        choices=["conflict_resolution", "conflict_resolution_behavior"],
        default="conflict_resolution",
        help=(
            "Evaluate which constraint is followed, or classify the model's "
            "conflict-resolution behavior."
        ),
    )
    parser.add_argument("--model", default="gpt-4o-2024-11-20")
    parser.add_argument(
        "--api_key",
        default=None,
        help="API key value. Prefer --api_key_env instead of putting a key in shell history.",
    )
    parser.add_argument(
        "--api_key_env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the API key.",
    )
    parser.add_argument(
        "--api_key_name",
        default="key",
        help="Fallback key name in evaluation.py's api_key_dict.",
    )
    parser.add_argument(
        "--base_url",
        default="https://api.openai.com/v1",
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument(
        "--reasoning_effort",
        choices=["minimal", "low", "medium", "high"],
        default="low",
        help="Reasoning effort for GPT-5-family judges.",
    )
    parser.add_argument("--wait_time", type=int, default=10)
    parser.add_argument("--retry_times", type=int, default=10)
    parser.add_argument(
        "--conditions",
        default=None,
        help=(
            "Hyphen-separated subset of original, new, conflict. Defaults to "
            "conflict for behavior evaluation and all three otherwise."
        ),
    )
    parser.add_argument(
        "--conflict_type_idx",
        default="1-2-3-4-5-6-7-8-9",
        help="Hyphen-separated conflict type ids.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Only evaluate files whose sample_id is at most this value.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate at most this many discovered files (useful for a smoke test).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Evaluate files again even when their evaluation output already exists.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Discover and validate inputs without making API requests.",
    )
    args = parser.parse_args()

    conditions_value = args.conditions or (
        "conflict"
        if args.task == "conflict_resolution_behavior"
        else "original-new-conflict"
    )
    args.conditions = hyphen_separated_values(conditions_value)
    invalid_conditions = set(args.conditions) - VALID_CONDITIONS
    if invalid_conditions:
        parser.error(
            "Unsupported conditions: " + ", ".join(sorted(invalid_conditions))
        )
    if (
        args.task == "conflict_resolution_behavior"
        and args.conditions != ["conflict"]
    ):
        parser.error(
            "--task conflict_resolution_behavior requires --conditions conflict."
        )

    try:
        args.conflict_type_indices = {
            int(value) for value in hyphen_separated_values(args.conflict_type_idx)
        }
    except ValueError:
        parser.error("--conflict_type_idx must contain hyphen-separated integers.")

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1.")
    return args


def judge_name_for_path(judge_model):
    return judge_model.strip("/").replace("/", "__")


def default_output_dir(input_dir, judge_model, task):
    path = Path(input_dir)
    parts = list(path.parts)
    try:
        output_index = parts.index("outputs")
    except ValueError as exc:
        raise ValueError(
            "--output_dir is required when --input_dir has no 'outputs' path component."
        ) from exc
    parts[output_index] = "evaluation_outputs"
    constraint_triplets_index = output_index + 1
    if (
        constraint_triplets_index >= len(parts)
        or parts[constraint_triplets_index] != "constraint_triplets"
    ):
        raise ValueError(
            "Default output mapping requires --input_dir to be under "
            "outputs/constraint_triplets. Otherwise, pass --output_dir."
        )
    parts.insert(
        constraint_triplets_index + 1,
        f"judge_{judge_name_for_path(judge_model)}",
    )
    if task == "conflict_resolution_behavior":
        parts.insert(
            constraint_triplets_index + 2,
            "conflict_resolution_behavior",
        )
    return Path(*parts)


def load_prompt_template(task):
    prompt_filename = {
        "conflict_resolution": "conflict_resolution_pair_evaluation.txt",
        "conflict_resolution_behavior": (
            "conflict_resolution_behavior_evaluation.txt"
        ),
    }[task]
    prompt_path = Path(__file__).resolve().parent.parent / "prompts" / prompt_filename
    return prompt_path.read_text(encoding="utf-8")


def validate_input(data, source_path):
    required_fields = {
        "model",
        "enable_thinking",
        "sample_id",
        "condition",
        "conflict_type_idx",
        "org_constraint",
        "new_constraint",
        "response",
    }
    missing = required_fields - set(data)
    if missing:
        raise ValueError(
            f"{source_path} is missing required fields: {', '.join(sorted(missing))}"
        )
    if data["condition"] not in VALID_CONDITIONS:
        raise ValueError(
            f"{source_path} has unsupported condition: {data['condition']!r}"
        )


def discover_items(args, input_dir, output_dir):
    items = []
    for source_path in sorted(input_dir.rglob("*.json")):
        if source_path.name == "results.json":
            continue
        with source_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        validate_input(data, source_path)

        if data["condition"] not in args.conditions:
            continue
        if int(data["conflict_type_idx"]) not in args.conflict_type_indices:
            continue
        if args.max_samples is not None and int(data["sample_id"]) > args.max_samples:
            continue

        relative_path = source_path.relative_to(input_dir)
        items.append(
            {
                "source_path": source_path,
                "output_path": output_dir / relative_path,
                "data": data,
            }
        )

    if args.limit is not None:
        items = items[: args.limit]
    return items


def instruction_order(data):
    """Alternate order deterministically to reduce positional judge bias."""
    order_key = (
        int(data["sample_id"])
        + int(data["conflict_type_idx"])
        + sum(ord(character) for character in data["condition"])
    )
    return "org_new" if order_key % 2 == 0 else "new_org"


def build_prompt(prompt_template, data, task, order=None):
    prompt = prompt_template.replace("##model_response##", data["response"])
    if task == "conflict_resolution_behavior":
        return prompt.replace("##instruction##", data["instruction"])

    if order == "org_new":
        instruction_1 = data["org_constraint"]
        instruction_2 = data["new_constraint"]
    else:
        instruction_1 = data["new_constraint"]
        instruction_2 = data["org_constraint"]
    return (
        prompt.replace("##instruction1##", instruction_1)
        .replace("##instruction2##", instruction_2)
    )


def evaluate_item(client, prompt_template, item, task):
    data = item["data"]
    evaluator_version = EVALUATOR_VERSIONS[task]
    judged_response, extraction_method, extraction_error = extract_final_answer(data)
    if extraction_error is not None:
        result = dict(data)
        result.update(
            {
                "judge_model": client.model,
                "evaluation_task": task,
                "evaluator_version": evaluator_version,
                "evaluation_status": "parse_failed",
                "response_extraction_method": extraction_method,
                "response_extraction_error": extraction_error,
                "judged_response": "",
            }
        )
        return result

    order = instruction_order(data) if task == "conflict_resolution" else None
    evaluation_result = client.generate(
        user_msg=build_prompt(
            prompt_template,
            {**data, "response": judged_response},
            task,
            order=order,
        )
    )
    if evaluation_result == FAILURE_RESPONSE:
        return None

    if task == "conflict_resolution":
        label = extract_label(evaluation_result, order)
        label_name = {
            1: "new_constraint",
            2: "org_constraint",
            -1: "neither",
        }[label]
    else:
        label = extract_label_conflict_resolution_behavior(evaluation_result)
        label_name = BEHAVIOR_LABEL_NAMES.get(label, "invalid_judge_output")

    result = dict(data)
    result.update(
        {
            "judge_model": client.model,
            "evaluation_task": task,
            "evaluator_version": evaluator_version,
            "evaluation_status": "evaluated",
            "response_extraction_method": extraction_method,
            "response_extraction_error": None,
            "judged_response": judged_response,
            "evaluation_result": evaluation_result,
            "label": label,
            "label_name": label_name,
        }
    )
    if order is not None:
        result["instruction_order"] = order
    return result


def read_completed_result(output_path, judge_model, task):
    if not output_path.exists():
        return None
    with output_path.open("r", encoding="utf-8") as file:
        result = json.load(file)
    required_fields = {
        "model",
        "enable_thinking",
        "condition",
        "evaluation_status",
        "evaluator_version",
        "judged_response",
    }
    if not required_fields.issubset(result):
        return None
    if result.get("judge_model") != judge_model:
        return None
    if result.get("evaluation_task") != task:
        return None
    if result.get("evaluator_version") != EVALUATOR_VERSIONS[task]:
        return None
    if result["evaluation_status"] == "evaluated" and "label" not in result:
        return None
    return result


def add_to_summary(counts, result, task):
    model = result["model"]
    mode = "thinking" if result["enable_thinking"] else "non_thinking"
    condition = result["condition"]
    conflict_type = f"type_{result['conflict_type_idx']}"
    bucket = counts[model][mode][condition][conflict_type]
    bucket["num"] += 1
    if task == "conflict_resolution":
        label_name = {
            1: "new_constraint",
            2: "org_constraint",
            -1: "neither",
        }[int(result["label"])]
        bucket[label_name] += 1
    else:
        label = int(result["label"])
        bucket[f"label_{label}"] += 1
        bucket[f"label_{label}_ids"].append(int(result["sample_id"]))


def finalize_summary(
    counts,
    judge_model,
    discovered,
    api_failed,
    parse_failed,
    task,
):
    summary = {
        "judge_model": judge_model,
        "evaluation_task": task,
        "evaluator_version": EVALUATOR_VERSIONS[task],
        "num_discovered": discovered,
        "num_evaluated": 0,
        "num_failed": api_failed + parse_failed,
        "num_api_failed": api_failed,
        "num_parse_failed": parse_failed,
        "models": {},
    }
    for model, modes in counts.items():
        summary["models"][model] = {}
        for mode, conditions in modes.items():
            summary["models"][model][mode] = {}
            for condition, conflict_types in conditions.items():
                condition_output = {"by_conflict_type": {}}
                if task == "conflict_resolution":
                    condition_totals = {
                        "num": 0,
                        "org_constraint": 0,
                        "new_constraint": 0,
                        "neither": 0,
                    }
                else:
                    condition_totals = {
                        "num": 0,
                        **{f"label_{label}": 0 for label in (1, 2, 3, 4, -1)},
                        **{
                            f"label_{label}_ids": []
                            for label in (1, 2, 3, 4, -1)
                        },
                    }
                for conflict_type, bucket in sorted(conflict_types.items()):
                    output_bucket = dict(bucket)
                    if task == "conflict_resolution":
                        for label_name in (
                            "org_constraint",
                            "new_constraint",
                            "neither",
                        ):
                            output_bucket[f"{label_name}_rate"] = (
                                bucket[label_name] / bucket["num"]
                            )
                            condition_totals[label_name] += bucket[label_name]
                    else:
                        for label in (1, 2, 3, 4, -1):
                            label_key = f"label_{label}"
                            ids_key = f"{label_key}_ids"
                            output_bucket[f"{label_key}_rate"] = (
                                bucket[label_key] / bucket["num"]
                            )
                            condition_totals[label_key] += bucket[label_key]
                            condition_totals[ids_key].extend(bucket[ids_key])
                    condition_totals["num"] += bucket["num"]
                    condition_output["by_conflict_type"][conflict_type] = output_bucket

                if task == "conflict_resolution":
                    rate_keys = ("org_constraint", "new_constraint", "neither")
                else:
                    rate_keys = tuple(
                        f"label_{label}" for label in (1, 2, 3, 4, -1)
                    )
                    condition_output["label_definitions"] = {
                        str(label): name
                        for label, name in BEHAVIOR_LABEL_NAMES.items()
                    }
                for key in rate_keys:
                    condition_totals[f"{key}_rate"] = (
                        condition_totals[key] / condition_totals["num"]
                    )
                condition_output["overall"] = condition_totals
                summary["num_evaluated"] += condition_totals["num"]
                summary["models"][model][mode][condition] = condition_output
    return summary


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else default_output_dir(input_dir, args.model, args.task)
    )
    items = discover_items(args, input_dir, output_dir)
    print(f"Discovered {len(items)} matching response files under {input_dir}")
    print(f"Evaluation outputs: {output_dir}")
    if args.dry_run:
        extraction_counts = Counter()
        parse_failures = []
        for item in items:
            _, method, error = extract_final_answer(item["data"])
            extraction_counts[(method, error)] += 1
            if error is not None and len(parse_failures) < 10:
                parse_failures.append(item["source_path"])
        print("Final-answer extraction summary:")
        for (method, error), count in sorted(extraction_counts.items()):
            print(f"  {method}: {count} (error={error})")
        if parse_failures:
            print("First parse failures:")
            for source_path in parse_failures:
                print(f"  {source_path}")
        print("First discovered files:")
        for item in items[:10]:
            print(item["source_path"])
        return
    if not items:
        raise ValueError("No matching triplet response files were found.")

    api_key = (
        args.api_key
        or os.environ.get(args.api_key_env)
        or api_key_dict.get(args.api_key_name)
    )
    if not api_key or api_key == "your_api_key":
        raise ValueError(
            f"No API key provided. Set {args.api_key_env}, pass --api_key, or "
            f"configure api_key_dict[{args.api_key_name!r}] in LLMs/evaluation.py."
        )

    client = OpenAIGPT4(
        api_key=api_key,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        wait_time=args.wait_time,
        retry_times=args.retry_times,
        base_url=args.base_url,
        reasoning_effort=args.reasoning_effort,
    )
    prompt_template = load_prompt_template(args.task)
    counts = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: (
                        {
                            "num": 0,
                            "org_constraint": 0,
                            "new_constraint": 0,
                            "neither": 0,
                        }
                        if args.task == "conflict_resolution"
                        else {
                            "num": 0,
                            **{
                                f"label_{label}": 0
                                for label in (1, 2, 3, 4, -1)
                            },
                            **{
                                f"label_{label}_ids": []
                                for label in (1, 2, 3, 4, -1)
                            },
                        }
                    )
                )
            )
        )
    )
    api_failed = 0
    parse_failed = 0

    for index, item in enumerate(items, start=1):
        result = None
        if not args.overwrite:
            result = read_completed_result(
                item["output_path"],
                args.model,
                args.task,
            )
        if result is None:
            result = evaluate_item(client, prompt_template, item, args.task)
            if result is None:
                api_failed += 1
                print(f"[{index}/{len(items)}] API request failed: {item['source_path']}")
                continue
            item["output_path"].parent.mkdir(parents=True, exist_ok=True)
            with item["output_path"].open("w", encoding="utf-8") as file:
                json.dump(result, file, ensure_ascii=False, indent=2)
            status = result["evaluation_status"]
        else:
            status = "cached"

        if result["evaluation_status"] == "parse_failed":
            parse_failed += 1
            print(
                f"[{index}/{len(items)}] parse_failed "
                f"({result['response_extraction_error']}): "
                f"{item['source_path']}"
            )
            continue

        add_to_summary(counts, result, args.task)
        print(f"[{index}/{len(items)}] {status}: {item['source_path']}")

    summary = finalize_summary(
        counts=counts,
        judge_model=args.model,
        discovered=len(items),
        api_failed=api_failed,
        parse_failed=parse_failed,
        task=args.task,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(f"Wrote summary to {results_path}")


if __name__ == "__main__":
    main()
