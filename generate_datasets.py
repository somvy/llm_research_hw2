import json
import sys

sys.path.insert(0, ".")

from datasets import Dataset, DatasetDict
from novel_ops import NovelOpsEnv

SPLITS = {
    "easy": {"difficulty": 2, "num": 500},
    "medium": {"difficulty": 5, "num": 500},
    "hard": {"difficulty": 8, "num": 500},
}
SEED = 0xCAFEBABE
REPO_ID = "therem/novel-ops-reasoning"


def generate_split(difficulty: int, num: int, seed: int) -> list[dict]:
    env = NovelOpsEnv(seed=seed)
    episodes = env.generate(num_of_questions=num, difficulty=difficulty)
    rows = []
    for ep in episodes:
        rows.append(
            {
                "question": ep.question,
                "answer": ep.answer,
                "difficulty": ep.difficulty,
                "metadata": json.dumps(ep.metadata, ensure_ascii=False),
            }
        )
    return rows


if __name__ == "__main__":
    splits = {}
    for name, cfg in SPLITS.items():
        print(
            f"Generating {name} split (difficulty={cfg['difficulty']}, n={cfg['num']})..."
        )
        rows = generate_split(cfg["difficulty"], cfg["num"], seed=SEED)
        splits[name] = Dataset.from_list(rows)
        print(f"  -> {len(rows)} samples")

    ds = DatasetDict(splits)
    print(ds)

    ds.push_to_hub(REPO_ID, private=False)
