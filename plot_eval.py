import json
import numpy as np
import matplotlib.pyplot as plt

with open("eval_results.json") as f:
    data = json.load(f)

models = ["base", "checkpoint-250", "checkpoint-500"]
model_labels = ["base", "ckpt-250", "ckpt-500"]
splits = ["easy", "medium", "hard"]
metrics = ["best@1", "best@2", "best@4"]

x = np.arange(len(models))
width = 0.25
offsets = [-width, 0, width]
colors = ["#4C72B0", "#DD8452", "#55A868"]

fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

for ax, split in zip(axes, splits):
    for i, (metric, offset, color) in enumerate(zip(metrics, offsets, colors)):
        means = [data[m][split][metric]["mean"] for m in models]
        stds = [data[m][split][metric]["std"] for m in models]
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            label=metric,
            color=color,
            capsize=4,
            error_kw={"linewidth": 1},
        )

    ax.set_title(split.capitalize(), fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("pass@k" if split == "easy" else "")
    ax.grid(axis="y", alpha=0.3)

axes[-1].legend(title="metric", loc="upper right")
fig.suptitle("Novel Ops — pass@k by difficulty and model", fontsize=14)
plt.tight_layout()
plt.savefig("eval_results.png", dpi=150)

faith_metrics = ["has_steps_rate", "step_accuracy", "faithful_when_correct"]
faith_labels = ["has steps", "step accuracy", "faithful|correct"]
faith_colors = ["#937860", "#C44E52", "#8172B2"]

fig2, axes2 = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

for ax, split in zip(axes2, splits):
    for i, (metric, offset, color, label) in enumerate(
        zip(faith_metrics, offsets, faith_colors, faith_labels)
    ):
        vals = [data[m][split]["faithfulness"][metric] or 0 for m in models]
        ax.bar(x + offset, vals, width, label=label, color=color)

    ax.set_title(split.capitalize(), fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate" if split == "easy" else "")
    ax.grid(axis="y", alpha=0.3)

axes2[-1].legend(loc="upper right")
fig2.suptitle("Novel Ops — faithfulness by difficulty and model", fontsize=14)
plt.tight_layout()
plt.savefig("eval_faithfulness.png", dpi=150)
print("saved eval_results.png + eval_faithfulness.png")