import os
import argparse
import yaml
from pathlib import Path
from typing import Optional

import torch
import wandb
from transformers import set_seed

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
    parser = argparse.ArgumentParser(description="AVSR Baseline Evaluation")
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
        help="Baseline model to evaluate",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/workspace/data/jekim/avsr/kmsav/data/cropped",
        help="Root path to KMSAV cropped data",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_outputs",
        help="Directory to save evaluation results",
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


def build_dataset(model_name: str, data_root: str, config: dict, split: str = "test"):
    from data import build_dataset as _build_dataset
    return _build_dataset(
        model_name=model_name,
        data_root=data_root,
        config=config,
        split=split,
    )


def build_collator(model_name: str, config: dict):
    from data import build_collator as _build_collator
    return _build_collator(model_name=model_name, config=config)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset,
    collator,
    config: dict,
    output_dir: str,
) -> dict:
    # - Run inference on the evaluation split
    # - Collect predictions and references
    # - Compute WER and log results
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.get("eval_batch_size", 4),
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        collate_fn=collator,
    )

    all_predictions = []
    all_references = []

    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = model.generate(**batch)
        all_predictions.extend(outputs["predictions"])
        all_references.extend(outputs["references"])

    # - Compute metrics
    results = compute_wer(
        predictions=all_predictions,
        references=all_references,
    )

    # - Save predictions to file
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "predictions.txt")
    with open(output_path, "w") as f:
        for pred, ref in zip(all_predictions, all_references):
            f.write(f"REF: {ref}\n")
            f.write(f"HYP: {pred}\n")
            f.write("\n")

    return results


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    set_seed(args.seed)

    # - Initialize wandb for eval tracking
    run_name = args.wandb_run_name or f"{args.model_name}-eval-{args.split}"
    init_wandb(
        project=args.wandb_project,
        run_name=run_name,
        config={**config, "model_name": args.model_name, "split": args.split},
    )

    # - Build model and load checkpoint
    model = build_model(args.model_name, config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))

    # - Build evaluation dataset
    eval_dataset = build_dataset(args.model_name, args.data_root, config, split=args.split)
    data_collator = build_collator(args.model_name, config)

    # - Run evaluation
    results = evaluate(
        model=model,
        dataset=eval_dataset,
        collator=data_collator,
        config=config,
        output_dir=args.output_dir,
    )

    # - Log results
    print(f"Evaluation Results ({args.split}):")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
    wandb.log(results)

    wandb.finish()


if __name__ == "__main__":
    main()
