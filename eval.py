import json
import os
import sys

sys.path.insert(0, ".")

import numpy as np
from math import comb
from transformers import AutoTokenizer
from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from novel_ops import parse_cot, reconstruct_operator, deserialize_expr, compute_reward

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
N_SAMPLES = 4
SPLITS = ["easy", "medium", "hard"]

REASONING_PROMPT = """\
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

MODELS = [
    {"name": "base", "adapter": None, "system_prompt": None, "use_xml": False},
    {
        "name": "checkpoint-250",
        "adapter": "outputs_novel_ops/checkpoint-250",
        "system_prompt": REASONING_PROMPT,
        "use_xml": True,
    },
    {
        "name": "checkpoint-500",
        "adapter": "outputs_novel_ops/checkpoint-500",
        "system_prompt": REASONING_PROMPT,
        "use_xml": True,
    },
]

SAMPLING_PARAMS = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=512,
    n=N_SAMPLES,
)


def extract_xml_answer(text: str) -> str:
    if "<answer>" not in text:
        return ""
    return text.split("<answer>")[-1].split("</answer>")[0].strip()


def extract_answer(text: str, use_xml: bool) -> str:
    if use_xml:
        xml_ans = extract_xml_answer(text)
        if xml_ans:
            return xml_ans
    _, final = parse_cot(text)
    if final is not None:
        return str(final)
    return ""


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


def evaluate_split(llm, tokenizer, dataset, system_prompt, use_xml, lora_request=None):
    prompts = build_prompts(tokenizer, dataset, system_prompt)
    outputs = llm.generate(prompts, SAMPLING_PARAMS, lora_request=lora_request)

    per_question = []
    all_step_acc = []
    all_has_steps = []
    all_faithful_when_correct = []

    for out, row in zip(outputs, dataset):
        metadata = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
        operators, expression = reconstruct_from_metadata(metadata)
        n_correct = 0

        for comp in out.outputs:
            ans = extract_answer(comp.text, use_xml=use_xml).strip()
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
        "step_accuracy": round(float(np.mean(all_step_acc)), 4) if all_step_acc else None,
        "faithful_when_correct": round(float(np.mean(all_faithful_when_correct)), 4) if all_faithful_when_correct else None,
    }
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
                use_xml=model_cfg["use_xml"],
                lora_request=lora_request,
            )
            model_results[split_name] = res

        all_results[name] = model_results

    with open("eval_results.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
