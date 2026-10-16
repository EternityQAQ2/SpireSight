from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="在 SFT adapter 基础上进行 DPO")
    parser.add_argument("--model", default="openbmb/MiniCPM5-2B")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--train", type=Path, default=Path("data/processed/dpo/train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/adapters/dpo"))
    parser.add_argument("--epochs", type=float, default=1.0)
    args = parser.parse_args()

    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    dataset = load_dataset("json", data_files={"train": str(args.train)})["train"]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map="auto",
        dtype="auto",
    )
    model = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=True)
    config = DPOConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        beta=0.1,
        logging_steps=5,
        save_strategy="epoch",
        max_length=4096,
        max_prompt_length=3072,
        report_to="none",
    )
    trainer = DPOTrainer(
        model=model,
        args=config,
        processing_class=tokenizer,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

