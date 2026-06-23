#!/usr/bin/env python3
"""Prepare a small instruction dataset for LoRA fine-tuning."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def pick_text(row: dict, names: list[str]) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def convert_row(row: dict) -> dict | None:
    instruction = pick_text(row, ["instruction", "question", "query", "prompt", "input"])
    extra_input = pick_text(row, ["input", "context"])
    output = pick_text(row, ["output", "answer", "response", "completion"])

    if not instruction or not output:
        return None

    if extra_input and extra_input != instruction:
        instruction = f"{instruction}\n\nContext:\n{extra_input}"

    return {
        "instruction": instruction,
        "input": "",
        "output": output,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="medalpaca/medical_meadow_medical_flashcards")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="data/medical_flashcards_lora.jsonl")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    rows = [item for item in (convert_row(row) for row in ds) if item]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.max_samples]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
      for row in rows:
          f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} examples to {out_path}")


if __name__ == "__main__":
    main()
