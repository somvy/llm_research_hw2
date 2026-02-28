import json
import os
import random
import sys

sys.path.insert(0, ".")

import numpy as np
from math import comb
from transformers import AutoTokenizer
from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from novel_ops import (
    parse_cot,
    reconstruct_operator,
    deserialize_expr,
    compute_reward,
    STEP_PATTERN,
)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
N_SAMPLES = 4
SPLITS = ["easy", "medium", "hard"]

SYSTEM_PROMPT = "You are a helpful math assistant."

MODELS = [
    {"name": "base", "adapter": None, "system_prompt": None},
    {
        "name": "checkpoint-500",
        "adapter": "outputs_novel_ops/checkpoint-500",
        "system_prompt": SYSTEM_PROMPT,
    },
    {
        "name": "checkpoint-final",
        "adapter": "novel_ops_grpo_lora/",
        "system_prompt": SYSTEM_PROMPT,
    },
]

SAMPLING_PARAMS = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=512,
    n=N_SAMPLES,
)

INTERVENTION_SAMPLING_PARAMS = SamplingParams(
    temperature=0,
    max_tokens=256,
    n=1,
)


def extract_answer(text: str) -> str:
    _, final = parse_cot(text)
    return str(final) if final is not None else ""


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def build_prompts(tokenizer, dataset, system_prompt):
    prompts = []
    for row in dataset:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": row["question"]})
        prompts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return prompts


def reconstruct_from_metadata(metadata):
    operators = [
        reconstruct_operator(om["symbol"], om["template_idx"], om["coeffs"])
        for om in metadata["operators"]
    ]
    op_lookup = {op.symbol: op for op in operators}
    expression = deserialize_expr(metadata["expression_tree"], op_lookup)
    return operators, expression


def build_intervention_data(outputs, prompts):
    data = []
    for out, prompt in zip(outputs, prompts):
        for comp in out.outputs:
            orig_ans = extract_answer(comp.text)
            if not orig_ans:
                continue
            matches = list(STEP_PATTERN.finditer(comp.text))
            if len(matches) < 2:
                continue
            step_idx = random.randint(0, len(matches) - 2)
            m = matches[step_idx]
            claimed = int(m.group(4))
            offset = random.choice([-7, -5, -3, 3, 5, 7])
            corrupted = claimed + offset
            prefix = comp.text[: m.start(4)] + str(corrupted) + "\n"
            data.append({"prompt": prompt + prefix, "orig_ans": orig_ans})
    return data


def evaluate_split(llm, tokenizer, dataset, system_prompt, lora_request=None):
    prompts = build_prompts(tokenizer, dataset, system_prompt)
    outputs = llm.generate(prompts, SAMPLING_PARAMS, lora_request=lora_request)

    per_question = []
    all_step_acc = []
    all_has_steps = []
    all_faithful_when_correct = []

    for out, row in zip(outputs, dataset):
        metadata = (
            json.loads(row["metadata"])
            if isinstance(row["metadata"], str)
            else row["metadata"]
        )
        operators, expression = reconstruct_from_metadata(metadata)
        n_correct = 0

        for comp in out.outputs:
            ans = extract_answer(comp.text).strip()
            is_correct = ans == str(row["answer"]).strip()
            if is_correct:
                n_correct += 1

            reward_info = compute_reward(comp.text, expression, operators)
            has_steps = reward_info.n_parsed_steps > 0
            all_has_steps.append(has_steps)
            if has_steps:
                step_acc = reward_info.n_correct_steps / reward_info.n_parsed_steps
                all_step_acc.append(step_acc)
                if is_correct:
                    all_faithful_when_correct.append(step_acc)

        per_question.append(n_correct)

    per_question = np.array(per_question)
    results = {}
    for k in [1, 2, 4]:
        if k > N_SAMPLES:
            continue
        scores = np.array([pass_at_k(N_SAMPLES, c, k) for c in per_question])
        results[f"best@{k}"] = {
            "mean": round(float(scores.mean()), 4),
            "std": round(float(scores.std()), 4),
        }

    results["faithfulness"] = {
        "has_steps_rate": round(float(np.mean(all_has_steps)), 4),
        "step_accuracy": round(float(np.mean(all_step_acc)), 4)
        if all_step_acc
        else None,
        "faithful_when_correct": round(float(np.mean(all_faithful_when_correct)), 4)
        if all_faithful_when_correct
        else None,
    }

    intervention_data = build_intervention_data(outputs, prompts)
    causal_faithfulness = None
    if intervention_data:
        int_prompts = [d["prompt"] for d in intervention_data]
        int_outputs = llm.generate(
            int_prompts, INTERVENTION_SAMPLING_PARAMS, lora_request=lora_request
        )
        changed = 0
        total = 0
        for cont, d in zip(int_outputs, intervention_data):
            new_ans = extract_answer(cont.outputs[0].text)
            if new_ans:
                total += 1
                if new_ans != d["orig_ans"]:
                    changed += 1
        causal_faithfulness = round(changed / total, 4) if total > 0 else None
    results["causal_faithfulness"] = causal_faithfulness

    return results


def main():
    ds = load_dataset("therem/novel-ops-reasoning")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    adapters = [
        os.path.abspath(m["adapter"])
        for m in MODELS
        if m["adapter"] and os.path.exists(m["adapter"])
    ]

    llm = LLM(
        model=MODEL_NAME,
        enable_lora=bool(adapters),
        max_lora_rank=64,
        max_model_len=2048,
        gpu_memory_utilization=0.9,
    )

    all_results = {}

    for model_cfg in MODELS:
        name = model_cfg["name"]
        adapter_path = model_cfg["adapter"]

        lora_request = None
        if adapter_path:
            lora_request = LoRARequest(name, 1, os.path.abspath(adapter_path))

        model_results = {}
        for split_name in SPLITS:
            res = evaluate_split(
                llm,
                tokenizer,
                ds[split_name],
                system_prompt=model_cfg["system_prompt"],
                lora_request=lora_request,
            )
            model_results[split_name] = res

        all_results[name] = model_results

    with open("eval_results.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
