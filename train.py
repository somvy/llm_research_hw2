import json
import os
import sys

os.environ["WANDB_SILENT"] = "true"
sys.path.insert(0, ".")

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from trl import GRPOConfig, GRPOTrainer

from base import Data
from novel_ops import NovelOpsEnv, NovelOpsVerifier, parse_cot


def extract_answer(text: str) -> str:
    _, final = parse_cot(text)
    return str(final) if final is not None else ""


def build_curriculum_dataset(env, system_prompt, schedule):
    rows = []
    prev_idx = 0
    for last_diff_idx, difficulty in schedule.items():
        ns = last_diff_idx - prev_idx
        data_list = env.generate(num_of_questions=ns, difficulty=difficulty)
        for data in data_list:
            rows.append(
                {
                    "prompt": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": data.question},
                    ],
                    "answer": data.answer,
                    "metadata": json.dumps(data.metadata),
                }
            )
        prev_idx = last_diff_idx
    return Dataset.from_list(rows)


def correctness_reward_func(
    prompts, completions, answer, metadata, **kwargs
) -> list[float]:
    responses = [c[0]["content"] for c in completions]
    rewards = []
    for resp, a, meta_str in zip(responses, answer, metadata):
        meta = json.loads(meta_str) if isinstance(meta_str, str) else meta_str
        data = Data(question="", answer=a, metadata=meta)
        info = verifier.compute_detailed_reward(data, resp, STEP_WEIGHT)
        rewards.append(info.total_reward * 2.0)
    return rewards


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
lora_rank = 64

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
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


SYSTEM_PROMPT = "You are a helpful math assistant."
STEP_WEIGHT = 0.8

verifier = NovelOpsVerifier()
env = NovelOpsEnv()

# last id: difficulty
CURRICULUM_SCHEDULE = {
    200: 1,
    400: 2,
    550: 3,
    700: 4,
    900: 5,
    1100: 6,
    # 1300: 7,
    # 1500: 8,
    # 1700: 9,
    # 1900: 10,
}

dataset = build_curriculum_dataset(env, SYSTEM_PROMPT, CURRICULUM_SCHEDULE)
MAX_STEPS = len(dataset)

print("Ds len:", len(dataset))
print(
    "Sample dataset:",
    "\n".join([dataset[idx]["prompt"][1]["content"] for idx in [0, 300, 500, 1000]]),
)
exit()

training_args = GRPOConfig(
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    beta=0.05,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    num_generations=8,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,
    # generation_batch_size=16,
    max_completion_length=512,
    use_vllm=True,
    vllm_mode="colocate",
    max_steps=MAX_STEPS,
    save_steps=500,
    max_grad_norm=0.1,
    report_to="wandb",
    output_dir="outputs_novel_ops",
    shuffle_dataset=False,
    scale_rewards=False,
    # sync_ref_model=True,
    # ref_model_sync_steps=200,
    logging_steps=10,
    log_completions=True,
    vllm_gpu_memory_utilization=0.5,
    num_completions_to_print=0,
    log_unique_prompts=True,
)


trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[correctness_reward_func],
    args=training_args,
    train_dataset=dataset,
)
trainer.train()

model.save_pretrained("novel_ops_grpo_lora")
tokenizer.save_pretrained("novel_ops_grpo_lora")


