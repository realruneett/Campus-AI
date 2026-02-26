#!/usr/bin/env python3
"""
Create Training Config
Reads the master config.yaml and generates an ai-toolkit compatible
YAML training config at configs/train_sdxl_lora.yaml.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_ai_toolkit_config(config: dict, output_path: str):
    """
    Generate an ai-toolkit compatible training config from master config.
    ai-toolkit expects a specific YAML schema for training SDXL LoRA.
    """
    sdxl_cfg = config.get("models", {}).get("sdxl", {})
    training_cfg = config.get("training", {})
    sdxl_lora_cfg = training_cfg.get("sdxl_lora", {})
    lora_cfg = sdxl_lora_cfg.get("lora", {})
    optim_cfg = sdxl_lora_cfg.get("optimizer", {})
    sched_cfg = sdxl_lora_cfg.get("scheduler", {})
    snr_cfg = sdxl_lora_cfg.get("min_snr_gamma", {})
    paths_cfg = config.get("paths", {})

    # Base model
    base_model = sdxl_cfg.get("repo_id", "stabilityai/stable-diffusion-xl-base-1.0")

    # Paths
    data_dir = os.path.abspath(paths_cfg.get("data", {}).get("train", "data/train"))
    output_dir = os.path.abspath(
        paths_cfg.get("models", {}).get("sdxl", {}).get("checkpoints", "models/sdxl/checkpoints")
    )
    log_dir = os.path.abspath(
        paths_cfg.get("logs", {}).get("tensorboard", "logs/tensorboard")
    )

    # LoRA params
    rank = lora_cfg.get("rank", 32)
    alpha = lora_cfg.get("alpha", 16)
    dropout = lora_cfg.get("dropout", 0.05)

    # Training params
    batch_size = sdxl_lora_cfg.get("batch_size", 1)
    grad_accum = sdxl_lora_cfg.get("gradient_accumulation_steps", 4)
    lr = optim_cfg.get("learning_rate", 1e-4)
    epochs = sdxl_lora_cfg.get("epochs", 4)
    max_steps = sdxl_lora_cfg.get("max_steps", 12800)
    warmup_steps = sched_cfg.get("warmup_steps", 100)
    weight_decay = optim_cfg.get("weight_decay", 0.01)
    
    betas = optim_cfg.get("betas", [0.9, 0.999])

    # Resolution
    height = sdxl_cfg.get("height", 1024)
    width = sdxl_cfg.get("width", 1024)

    # Seed
    seed = config.get("project", {}).get("seed", 42)

    # Mixed precision
    mixed_prec = training_cfg.get("mixed_precision", {})
    dtype = mixed_prec.get("dtype", "bf16")

    # Build ai-toolkit config
    aitk_config = {
        "job": "extension",
        "config": {
            "name": "campus_ai_poster_sdxl",
            "process": [
                {
                    "type": "sd_trainer",
                    "training_folder": output_dir,
                    "device": "cuda:0",
                    "trigger_word": "campus_ai_poster",
                    "network": {
                        "type": "lora",
                        "linear": rank,
                        "linear_alpha": alpha,
                        "dropout": dropout,
                        "network_kwargs": {
                            "lora_plus_lr_ratio": lora_cfg.get("lora_plus_ratio", 1.0),
                        },
                    },
                    "save": {
                        "dtype": dtype,
                        "save_every": sdxl_lora_cfg.get("checkpointing", {}).get("save_steps", 500),
                        "max_step_saves_to_keep": sdxl_lora_cfg.get("checkpointing", {}).get("save_total_limit", 5),
                    },
                    "datasets": [
                        {
                            "folder_path": data_dir,
                            "caption_ext": "txt",
                            "caption_dropout_rate": 0.1,
                            "shuffle_tokens": True,
                            "cache_latents_to_disk": True,
                            "num_workers": 8,
                            "resolution": [width, height],
                        }
                    ],
                    "train": {
                        "batch_size": batch_size,
                        "steps": max_steps if max_steps > 0 else 12800,
                        "gradient_accumulation_steps": grad_accum,
                        "train_unet": True,
                        "train_text_encoder": False,
                        "disable_sampling": True,
                        "gradient_checkpointing": True,
                        "noise_scheduler": "ddpm",
                        "optimizer": optim_cfg.get("type", "adamw8bit"),
                        "lr": lr,
                        "lr_warmup_steps": warmup_steps,
                        "min_snr_gamma": snr_cfg.get("gamma", 5.0) if snr_cfg.get("enabled", True) else None,
                        "optimizer_params": {
                             "weight_decay": weight_decay,
                             "betas": betas,
                        },
                        "ema_config": {
                            "use_ema": True,
                            "ema_decay": 0.999,
                        },
                        "dtype": dtype,
                        "lr_scheduler": sched_cfg.get("type", "cosine_with_restarts"),
                        "lr_scheduler_params": {
                            "T_0": max(1, (max_steps if max_steps > 0 else 12800) // sched_cfg.get("num_cycles", 3)),
                            "T_mult": 1,
                            "eta_min": lr / 10,
                        },
                    },
                    "model": {
                        "name_or_path": base_model,
                        "is_xl": True,
                    },
                    "sample": {
                        "sampler": "euler_a",
                        "sample_every": 999999,
                        "width": width,
                        "height": height,
                        "prompts": [
                            "campus_ai_poster a vibrant technology fest poster with neon colors and bold typography",
                            "campus_ai_poster a colorful Diwali celebration poster with golden diyas and rangoli",
                            "campus_ai_poster a professional workshop seminar poster with modern minimalist design",
                            "campus_ai_poster a dynamic sports tournament poster with action silhouettes",
                        ],
                        "neg": "",
                        "seed": seed,
                        "walk_seed": True,
                        "guidance_scale": 5,
                        "sample_steps": 28,
                    },
                    "logging": {
                        "log_every": sdxl_lora_cfg.get("logging", {}).get("steps", 10),
                        "use_wandb": config.get("monitoring", {}).get("wandb", {}).get("enabled", False),
                        "verbose": True,
                    },
                }
            ],
            "meta": {
                "name": "campus_ai_v1",
                "version": "1.0",
            },
        },
    }

    # Write output
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        yaml.dump(aitk_config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    logger.info(f"ai-toolkit training config written to: {output_file}")
    logger.info(f"  Base model: {base_model}")
    logger.info(f"  Dataset dir: {data_dir}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  LoRA rank: {rank}, alpha: {alpha}")
    logger.info(f"  Batch size: {batch_size}, Grad accum: {grad_accum}")
    logger.info(f"  Learning rate: {lr}")
    logger.info(f"  Resolution: {width}x{height}")
    logger.info(f"  Mixed precision: {dtype}")

    return aitk_config


def main():
    parser = argparse.ArgumentParser(description="Generate ai-toolkit Training Config")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to master config.yaml")
    parser.add_argument("--output", default="configs/train_sdxl_lora.yaml", help="Output path for ai-toolkit config")
    args = parser.parse_args()

    config = load_config(args.config)
    generate_ai_toolkit_config(config, args.output)


if __name__ == "__main__":
    main()
