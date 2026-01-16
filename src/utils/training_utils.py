"""
Training utilities for fine-tuning language models with LoRA
"""

import torch
from dataclasses import dataclass
from typing import Optional
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType


@dataclass
class LoRATrainingConfig:
    """Configuration for LoRA (Low-Rank Adaptation)"""

    r: int = 8
    lora_alpha: int = 16
    target_modules: list = None
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    """Training configuration"""

    output_dir: str
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    logging_steps: int = 50
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    max_grad_norm: float = 1.0
    fp16: bool = False
    eval_strategy: str = "no"
    weight_decay: float = 0.01
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "linear"


def get_optimal_device():
    """Get the best available device (CUDA > MPS > CPU)"""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"  ✓ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("  ✓ Using Apple Silicon GPU (MPS)")
    else:
        device = "cpu"
        print("  ⚠ Using CPU (training will be slow)")

    return device


def setup_model_with_lora(
    model_name: str,
    lora_config: LoRATrainingConfig,
    device: str,
    resize_embeddings: Optional[int] = None,
):
    """
    Load model and apply LoRA configuration

    Args:
        model_name: HuggingFace model name
        lora_config: LoRA configuration
        device: Device to load model on
        resize_embeddings: New vocab size if adding special tokens
    """
    print(f"  Loading base model: {model_name}")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for stability
        device_map=None,  # We'll move to device manually
        trust_remote_code=True,
    )

    # Resize embeddings if new tokens were added
    if resize_embeddings:
        print(f"  Resizing embeddings to {resize_embeddings}")
        model.resize_token_embeddings(resize_embeddings)

    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.bias,
        target_modules=lora_config.target_modules,
    )

    print(f"  Applying LoRA with r={lora_config.r}, alpha={lora_config.lora_alpha}")
    model = get_peft_model(model, peft_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"  Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
    )
    print(f"  Total params: {total_params:,}")

    return model


def create_trainer(
    model,
    tokenizer,
    train_dataset,
    training_config: TrainingConfig,
):
    """Create HuggingFace Trainer"""

    # Convert dataclass to TrainingArguments
    training_args = TrainingArguments(
        output_dir=training_config.output_dir,
        num_train_epochs=training_config.num_train_epochs,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        warmup_steps=training_config.warmup_steps,
        logging_steps=training_config.logging_steps,
        save_strategy=training_config.save_strategy,
        save_total_limit=training_config.save_total_limit,
        max_grad_norm=training_config.max_grad_norm,
        fp16=training_config.fp16,
        evaluation_strategy=training_config.eval_strategy,
        weight_decay=training_config.weight_decay,
        optim=training_config.optim,
        lr_scheduler_type=training_config.lr_scheduler_type,
        report_to="none",  # Disable wandb/tensorboard
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    return trainer


def train_and_save(trainer, tokenizer, output_dir: str):
    """Train model and save results"""

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    # Train
    trainer.train()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - Saving model")
    print("=" * 60)

    # Save model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"✓ Model saved to: {output_dir}")


def print_training_info(
    model_name: str, dataset_size: int, max_length: int, config: TrainingConfig
):
    """Print training configuration summary"""

    print("\n" + "-" * 60)
    print("TRAINING CONFIGURATION")
    print("-" * 60)
    print(f"Model: {model_name}")
    print(f"Dataset size: {dataset_size:,} examples")
    print(f"Max sequence length: {max_length}")
    print(f"Epochs: {config.num_train_epochs}")
    print(f"Batch size: {config.per_device_train_batch_size}")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    print(
        f"Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}"
    )
    print(f"Learning rate: {config.learning_rate}")
    print(f"Warmup steps: {config.warmup_steps}")
    print(f"Weight decay: {config.weight_decay}")
    print(f"Output directory: {config.output_dir}")
    print("-" * 60 + "\n")
