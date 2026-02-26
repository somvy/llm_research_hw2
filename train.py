import json
import re
import sys

sys.path.insert(0, ".")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from datasets import load_dataset, concatenate_datasets
from trl import GRPOConfig, GRPOTrainer


from base import Data
from novel_ops import NovelOpsVerifier


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def get_dataset():
    ds = load_dataset("therem/novel-ops-reasoning")
    combined = concatenate_datasets([ds[s] for s in ds])
    combined = combined.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
        }
    )
    return combined


def correctness_reward_func(
    prompts, completions, answer, metadata, **kwargs
) -> list[float]:
    responses = [c[0]["content"] for c in completions]
    rewards = []
    for resp, a, meta_str in zip(responses, answer, metadata):
        meta = json.loads(meta_str) if isinstance(meta_str, str) else meta_str
        data = Data(question="", answer=a, metadata=meta)
        try:
            info = verifier.compute_detailed_reward(data, resp)
            rewards.append(info.total_reward * 2.0)
        except Exception:
            extracted = extract_xml_answer(resp)
            rewards.append(2.0 if extracted.strip() == str(a).strip() else 0.0)
    q = prompts[0][-1]["content"]
    print(
        "-" * 20,
        f"Question:\n{q}",
        f"\nAnswer:\n{answer[0]}",
        f"\nResponse:\n{responses[0]}",
        f"\nExtracted:\n{extract_xml_answer(responses[0])}",
    )
    return rewards


def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [c[0]["content"] for c in completions]
    extracted = [extract_xml_answer(r) for r in responses]
    return [0.5 if re.match(r"^-?\d+$", r.strip()) else 0.0 for r in extracted]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [c[0]["content"] for c in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [c[0]["content"] for c in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    return [count_xml(c[0]["content"]) for c in completions]


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
lora_rank = 64

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto",
)

model = get_peft_model(
    model,
    LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type=TaskType.CAUSAL_LM,
    ),
)
model.enable_input_require_grads()
model.print_trainable_parameters()


SYSTEM_PROMPT = """\
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

verifier = NovelOpsVerifier()

dataset = get_dataset()


training_args = GRPOConfig(
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    num_generations=16,
    max_completion_length=512,
    max_steps=500,
    save_steps=250,
    max_grad_norm=0.1,
    report_to="wandb",
    output_dir="outputs_novel_ops",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=dataset,
)
trainer.train()

model.save_pretrained("novel_ops_grpo_lora")
tokenizer.save_pretrained("novel_ops_grpo_lora")

text = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": dataset[0]["question"]},
    ],
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(text, return_tensors="pt").to(model.device)
output_ids = model.generate(
    **inputs, max_new_tokens=1024, temperature=0.8, top_p=0.95, do_sample=True
)
print(
    "Test output:",
    tokenizer.decode(
        output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    ),
)
