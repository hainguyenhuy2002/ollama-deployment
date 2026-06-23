#!/usr/bin/env python3
"""Small LoRA SFT trainer for Ascend NPU."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


SYSTEM_PROMPT = "You are a careful clinical pharmacy assistant. Give concise and safe answers."


def load_ascend() -> bool:
    try:
        import torch_npu  # noqa: F401
    except Exception:
        return False
    return hasattr(torch, "npu") and torch.npu.is_available()


class JsonlSftDataset(Dataset):
    def __init__(self, path: str, tokenizer, cutoff_len: int):
        self.samples = []
        self.tokenizer = tokenizer
        self.cutoff_len = cutoff_len
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        row = self.samples[idx]
        user = row["instruction"]
        if row.get("input"):
            user = f"{user}\n\n{row['input']}"
        answer = row["output"]

        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        full_text = f"{prompt}{answer}<|im_end|>"

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        full = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.cutoff_len,
        )
        input_ids = full.input_ids
        labels = input_ids.copy()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids": input_ids,
            "attention_mask": full.attention_mask,
            "labels": labels,
        }


def collate(batch: list[dict], pad_token_id: int) -> dict:
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids, attention_mask, labels = [], [], []
    for item in batch:
        pad = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * pad)
        attention_mask.append(item["attention_mask"] + [0] * pad)
        labels.append(item["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data", default="data/medical_flashcards_lora.jsonl")
    parser.add_argument("--output-dir", default="outputs/qwen2.5-0.5b-medical-lora")
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--cutoff-len", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_npu = load_ascend()
    device = torch.device("npu:0" if use_npu else "cpu")
    dtype = torch.bfloat16 if use_npu else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model.to(device)
    model.train()
    model.print_trainable_parameters()

    dataset = JsonlSftDataset(args.data, tokenizer, args.cutoff_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_updates = args.max_steps
    warmup_steps = max(1, math.ceil(total_updates * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_updates)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "training_metrics.jsonl"
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(total=args.max_steps, desc="LoRA fine-tuning")
    while global_step < args.max_steps:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum
            loss.backward()

            if (global_step + 1) % args.grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            progress.update(1)

            if global_step % 10 == 0:
                raw_loss = float(loss.detach().cpu()) * args.grad_accum
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": global_step, "loss": raw_loss}) + "\n")
                progress.set_postfix(loss=f"{raw_loss:.4f}")

            if global_step % args.save_steps == 0:
                checkpoint = output_dir / f"checkpoint-{global_step}"
                model.save_pretrained(checkpoint)
                tokenizer.save_pretrained(checkpoint)

            if global_step >= args.max_steps:
                break

    progress.close()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved LoRA adapter to {output_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
