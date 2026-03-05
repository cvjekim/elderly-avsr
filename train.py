import os
import argparse
import yaml
from pathlib import Path
from typing import Optional

import torch
import wandb
from accelerate import Accelerator
from transformers import (
    TrainingArguments,
    Trainer,
    set_seed,
)

from utils.wandb_utils import init_wandb
from utils.metrics import compute_wer


BASELINE_MODELS = [
    "auto_avsr",
    "av_hubert",
    "whisper_flamingo",
    "llama_avsr",
    "mms_llama",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AVSR Baseline Training")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=BASELINE_MODELS,
        help="Baseline model to train",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/workspace/data/jekim/avsr/kmsav/data/cropped",
        help="Root path to KMSAV cropped data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save checkpoints and logs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="avsr-baseline",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Weights & Biases run name",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def build_model(model_name: str, config: dict) -> torch.nn.Module:
    # - Dynamically import and instantiate the selected baseline model
    if model_name == "auto_avsr":
        from models.auto_avsr import build_model as _build
    elif model_name == "av_hubert":
        from models.av_hubert import build_model as _build
    elif model_name == "whisper_flamingo":
        from models.whisper_flamingo import build_model as _build
    elif model_name == "llama_avsr":
        from models.llama_avsr import build_model as _build
    elif model_name == "mms_llama":
        from models.mms_llama import build_model as _build
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return _build(config)


def build_dataset(model_name: str, data_root: str, config: dict, split: str = "train"):
    # - Load KMSAV dataset split with model-specific preprocessing
    from data import build_dataset as _build_dataset
    return _build_dataset(
        model_name=model_name,
        data_root=data_root,
        config=config,
        split=split,
    )


def build_collator(model_name: str, config: dict):
    # - Return model-specific data collator for batching
    from data import build_collator as _build_collator
    return _build_collator(model_name=model_name, config=config)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    # - Set reproducibility
    set_seed(args.seed)

    # - Initialize wandb
    run_name = args.wandb_run_name or f"{args.model_name}-{Path(args.config).stem}"
    init_wandb(
        project=args.wandb_project,
        run_name=run_name,
        config={**config, "model_name": args.model_name, "seed": args.seed},
    )

    # - Build model
    model = build_model(args.model_name, config)

    # - Build datasets
    train_dataset = build_dataset(args.model_name, args.data_root, config, split="train")
    eval_dataset = build_dataset(args.model_name, args.data_root, config, split="val")

    # - Build data collator
    data_collator = build_collator(args.model_name, config)

    # - Configure training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=config.get("batch_size", 4),
        per_device_eval_batch_size=config.get("eval_batch_size", 4),
        num_train_epochs=config.get("epochs", 30),
        learning_rate=config.get("learning_rate", 1e-4),
        weight_decay=config.get("weight_decay", 0.01),
        warmup_ratio=config.get("warmup_ratio", 0.1),
        lr_scheduler_type=config.get("lr_scheduler", "cosine"),
        fp16=config.get("fp16", True),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=config.get("save_total_limit", 3),
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        logging_steps=config.get("logging_steps", 50),
        report_to="wandb",
        seed=args.seed,
        dataloader_num_workers=config.get("num_workers", 4),
    )

    # - Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_wer,
    )

    # - Train
    trainer.train()

    # - Save final model
    trainer.save_model(os.path.join(args.output_dir, "final"))

    # - Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
