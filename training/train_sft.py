from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="使用 LoRA 进行 SFT")
    parser.add_argument("--model", default="openbmb/MiniCPM5-2B")
    parser.add_argument("--train", type=Path, default=Path("data/processed/sft/train.jsonl"))
    parser.add_argument("--validation", type=Path, default=Path("data/processed/sft/validation.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/adapters/sft"))
    parser.add_argument("--epochs", type=float, default=2.0)
    args = parser.parse_args()

    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    dataset = load_dataset(
        "json",
        data_files={"train": str(args.train), "validation": str(args.validation)},
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map="auto",
        dtype="auto",
    )
    config = SFTConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        max_length=4096,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        ),
    )
    trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

