"""
Training utility functions for fine-tuning language models
"""

import torch
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from typing import Optional, Dict, Any


class LoRATrainingConfig:
    """Configuration for LoRA training"""

    def __init__(
        self,
        r: int = 16,
        lora_alpha: int = 32,
        target_modules: list = None,
        lora_dropout: float = 0.05,
        bias: str = "none",
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        self.lora_dropout = lora_dropout
        self.bias = bias

    def to_lora_config(self) -> LoraConfig:
        """Convert to PEFT LoraConfig"""
        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            task_type="CAUSAL_LM",
        )


class TrainingConfig:
    """Configuration for training arguments"""

    def __init__(
        self,
        output_dir: str,
        checkpoint_dir: Optional[str] = None,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        learning_rate: float = 2e-4,
        warmup_steps: int = 100,
        logging_steps: int = 10,
        save_strategy: str = "no",
        save_total_limit: int = 2,
        max_grad_norm: float = 1.0,
        fp16: bool = False,
        eval_strategy: str = "no",
    ):
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir or output_dir
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.logging_steps = logging_steps
        self.save_strategy = save_strategy
        self.save_total_limit = save_total_limit
        self.max_grad_norm = max_grad_norm
        self.fp16 = fp16
        self.eval_strategy = eval_strategy

    def to_training_arguments(self) -> TrainingArguments:
        """Convert to Hugging Face TrainingArguments"""
        return TrainingArguments(
            output_dir=self.checkpoint_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            logging_steps=self.logging_steps,
            save_strategy=self.save_strategy,
            save_total_limit=self.save_total_limit,
            eval_strategy=self.eval_strategy,
            fp16=self.fp16,
            remove_unused_columns=False,
            max_grad_norm=self.max_grad_norm,
            report_to="none",
        )


def setup_model_with_lora(
    model_name: str,
    lora_config: LoRATrainingConfig,
    device: Optional[torch.device] = None,
) -> AutoModelForCausalLM:
    """
    Load model and apply LoRA configuration

    Args:
        model_name: Name or path of pretrained model
        lora_config: LoRA configuration
        device: Device to move model to (auto-detected if None)

    Returns:
        Model with LoRA applied
    """
    print(f"Loading base model: {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)

    print("Applying LoRA configuration...")
    peft_config = lora_config.to_lora_config()
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Setup device
    if device is None:
        device = get_optimal_device()

    print(f"Moving model to device: {device}")
    model.to(device)

    return model


def get_optimal_device() -> torch.device:
    """
    Get the optimal device for training

    Returns:
        torch.device: Best available device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def create_trainer(
    model,
    tokenizer,
    train_dataset,
    training_config: TrainingConfig,
    eval_dataset=None,
) -> Trainer:
    """
    Create Hugging Face Trainer with configuration

    Args:
        model: Model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset
        training_config: Training configuration
        eval_dataset: Optional evaluation dataset

    Returns:
        Configured Trainer
    """
    training_args = training_config.to_training_arguments()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    return trainer


def train_and_save(
    trainer: Trainer,
    tokenizer,
    output_dir: str,
    resume_from_checkpoint: Optional[str] = None,
) -> None:
    """
    Train model and save results

    Args:
        trainer: Configured trainer
        tokenizer: Tokenizer to save
        output_dir: Directory to save final model
        resume_from_checkpoint: Optional checkpoint path to resume from
    """
    print("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    print(f"Saving final model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Training complete!")


def print_training_info(
    model_name: str,
    dataset_size: int,
    max_length: int,
    training_config: TrainingConfig,
) -> None:
    """Print training configuration summary"""
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Dataset size: {dataset_size:,} examples")
    print(f"Max sequence length: {max_length}")
    print(f"Epochs: {training_config.num_train_epochs}")
    print(f"Batch size: {training_config.per_device_train_batch_size}")
    print(f"Gradient accumulation: {training_config.gradient_accumulation_steps}")
    print(
        f"Effective batch size: {training_config.per_device_train_batch_size * training_config.gradient_accumulation_steps}"
    )
    print(f"Learning rate: {training_config.learning_rate}")
    print(f"Warmup steps: {training_config.warmup_steps}")
    print(f"Output directory: {training_config.output_dir}")
    print("=" * 60 + "\n")
