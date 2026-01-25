#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_single_task_endstop.py

Single-task finetuning (LoRA) with "###" separator + "END" stop token,
mimicking the classic prompt/completion fine-tune style.

Supports:
- --task conan     : sentence1 -> sentence2..n punchline continuation
- --task keywords  : (k1, k2) -> generate joke including both keywords

Key design choices:
- Prompt ends with "\n\n###\n\n"
- Target ends with " END"
- Conan preprocessing similar to the internet example: NFKD normalization, emdash -> "--",
  spaCy sentence splitting via doc.sents
- Output length caps to reduce noisy long targets
- No multitask mixing (avoids interference)

Example:
  python train_single_task_endstop.py --task conan --data_path conan_jokes.json
  python train_single_task_endstop.py --task keywords --data_path full_jokes_general_reddit.json --max_samples 40000

Quantization:
  --load_in_4bit (default) or --load_in_8bit or --full_precision
"""

import os
import re
import json
import math
import random
import unicodedata
from datetime import datetime
from collections import Counter

import torch
import numpy as np
import spacy
from datasets import Dataset

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments


# -----------------------------
# Run bookkeeping
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def safe_symlink(target: str, link_name: str):
    link_dir = os.path.dirname(link_name) or "."
    ensure_dir(link_dir)
    tmp = link_name + ".tmp"
    try:
        if os.path.islink(tmp) or os.path.exists(tmp):
            os.remove(tmp)
        os.symlink(os.path.abspath(target), tmp)
        os.replace(tmp, link_name)
    except OSError:
        # fallback pointer file
        try:
            if os.path.islink(link_name) or os.path.exists(link_name):
                os.remove(link_name)
            with open(link_name + ".txt", "w", encoding="utf-8") as f:
                f.write(os.path.abspath(target))
        except Exception as e:
            print(f"[WARN] Could not update symlink/pointer for {link_name}: {e}")

def read_eval_loss(trainer_state_json_path: str):
    try:
        with open(trainer_state_json_path, "r", encoding="utf-8") as f:
            st = json.load(f)
        if "best_metric" in st and st["best_metric"] is not None:
            return float(st["best_metric"])
        best = None
        for row in st.get("log_history", []):
            if "eval_loss" in row:
                v = float(row["eval_loss"])
                if best is None or v < best:
                    best = v
        return best
    except Exception:
        return None

def read_best_eval_loss_from_runs(best_json_path: str):
    try:
        with open(best_json_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return float(obj.get("best_eval_loss"))
    except Exception:
        return None

def write_best_record(best_json_path: str, run_id: str, run_dir: str, eval_loss: float):
    obj = {
        "best_eval_loss": eval_loss,
        "run_id": run_id,
        "run_dir": os.path.abspath(run_dir),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(best_json_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# -----------------------------
# Prompt/completion format (like the internet example)
# -----------------------------
SEP = "\n\n###\n\n"
END = " END"

def make_pc(prompt_text: str, completion_text: str) -> str:
    # We store as one "text" field for SFTTrainer: prompt + completion
    # and ensure completions begin with a space and end with END.
    prompt = prompt_text.rstrip() + SEP
    completion = " " + completion_text.strip() + END
    return prompt + completion


# -----------------------------
# Conan preprocessing (similar to example)
# -----------------------------
def clean_conan_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = s.replace("—", "--")
    s = s.replace("  ", " ").strip()
    return s

def split_sentences_spacy(nlp, text: str):
    doc = nlp(text)
    sents = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sents


# -----------------------------
# Keywords extraction (simpler but solid)
# -----------------------------
STOP_RE = re.compile(r"^[^a-zA-Z]+$")

GENERIC = {
    "thing", "stuff", "something", "someone", "anything", "everything",
    "place", "way", "lot", "kind"
}

def normalize_for_dedup(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def contains_word(text_lower: str, w: str) -> bool:
    return re.search(rf"\b{re.escape(w)}\b", text_lower) is not None

def extract_two_keywords_from_doc(doc, text_lower: str, min_len=3):
    candidates = []
    for tok in doc:
        if tok.is_stop or tok.is_punct or tok.is_space:
            continue
        if len(tok.text) < min_len:
            continue
        if STOP_RE.match(tok.text):
            continue
        if not re.search(r"[A-Za-z]", tok.text):
            continue

        if tok.pos_ in ("NOUN", "PROPN", "ADJ"):
            lemma = tok.lemma_.lower().strip()
            if len(lemma) < min_len:
                continue
            if lemma in GENERIC:
                continue
            if lemma.isdigit():
                continue
            candidates.append((lemma, tok.pos_))

    if not candidates:
        return None

    counts = Counter([w for w, _ in candidates])

    def score(word):
        freq = counts[word]
        pos = None
        for w, p in candidates:
            if w == word:
                pos = p
                break
        pos_w = 3 if pos == "PROPN" else 2 if pos == "NOUN" else 1
        return freq * 10 + pos_w

    ranked = sorted(counts.keys(), key=score, reverse=True)

    picked = []
    for w in ranked:
        if w in picked:
            continue
        if not contains_word(text_lower, w):
            continue
        picked.append(w)
        if len(picked) == 2:
            return tuple(picked)
    return None


# -----------------------------
# Main
# -----------------------------
def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["conan", "keywords"], required=True)

    # Data paths:
    ap.add_argument("--data_path", required=True, help="JSON file with list of {body: ...} items")

    # Model:
    ap.add_argument("--base_model", default="unsloth/mistral-7b-v0.3")
    ap.add_argument("--max_seq_length", type=int, default=768)

    # Quantization modes:
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--load_in_8bit", action="store_true")
    ap.add_argument("--full_precision", action="store_true")

    # Sampling:
    ap.add_argument("--max_samples", type=int, default=None)

    # Conan controls:
    ap.add_argument("--exclude_if_contains_conan", action="store_true", help='Skip jokes containing "Conan" (like example)')
    ap.add_argument("--min_sentences", type=int, default=2)
    ap.add_argument("--max_completion_chars", type=int, default=300, help="Cap punchline length to keep it snappy")

    # Keywords controls:
    ap.add_argument("--min_joke_chars", type=int, default=20)
    ap.add_argument("--max_joke_chars", type=int, default=300, help="Cap jokes to keep them short")
    ap.add_argument("--spacy_model", default="en_core_web_sm")
    ap.add_argument("--spacy_batch_size", type=int, default=512)
    ap.add_argument("--min_keyword_len", type=int, default=3)

    # Split:
    ap.add_argument("--val_ratio", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=3407)

    # Training hyperparams:
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--per_device_batch", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--lr_sched", default="cosine")
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--eval_steps", type=int, default=7000)
    ap.add_argument("--save_steps", type=int, default=7000)
    ap.add_argument("--weight_decay", type=float, default=0.0)

    # LoRA:
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_dropout", type=float, default=0.0)

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Quantization selection:
    # Default: 4bit if nothing specified (matches your current workflow)
    if not (args.load_in_4bit or args.load_in_8bit or args.full_precision):
        args.load_in_4bit = True
    # mutual sanity:
    chosen = sum([args.load_in_4bit, args.load_in_8bit, args.full_precision])
    if chosen != 1:
        raise SystemExit("Choose exactly one: --load_in_4bit OR --load_in_8bit OR --full_precision")

    # Runs
    RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUNS_ROOT = "runs"
    RUN_DIR = os.path.join(RUNS_ROOT, f"{RUN_ID}_{args.task}")
    OUTPUT_DIR = os.path.join(RUN_DIR, "trainer_outputs")
    SAVE_LORA_DIR = os.path.join(RUN_DIR, "lora")
    ensure_dir(RUN_DIR)
    ensure_dir(OUTPUT_DIR)
    ensure_dir(SAVE_LORA_DIR)

    # Save config
    config_path = os.path.join(RUN_DIR, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\n=== RUN_ID: {RUN_ID} (task={args.task}) ===")
    print(f"Run dir: {RUN_DIR}")
    print(f"Config saved: {config_path}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    bodies = []
    for item in data:
        b = item.get("body", "")
        if isinstance(b, str):
            b = b.strip()
            if b:
                bodies.append(b)

    random.shuffle(bodies)
    if args.max_samples is not None:
        bodies = bodies[: args.max_samples]

    print("Loaded bodies:", len(bodies))

    # Build prompt-completion samples
    samples = []

    if args.task == "conan":
        # Use spaCy sentence splitting like the internet example.
        # Need the parser for doc.sents to work reliably (default pipeline has parser on).
        nlp = spacy.load(args.spacy_model)

        for b in bodies:
            if args.exclude_if_contains_conan and "Conan" in b:
                continue

            clean = clean_conan_text(b)
            sents = split_sentences_spacy(nlp, clean)
            if len(sents) < args.min_sentences:
                continue

            prompt = sents[0]
            completion = " ".join(sents[1:]).strip()

            # cap completion length (snappier punchlines, less noise)
            if len(completion) > args.max_completion_chars:
                completion = completion[: args.max_completion_chars].rstrip()

            # basic sanity
            if len(prompt) < 10 or len(completion) < 10:
                continue

            text = make_pc(prompt, completion)
            samples.append(text)

        print("CONAN samples:", len(samples))
        if samples:
            print("Example (raw text):")
            print(samples[0][:400], "...\n")

    else:  # keywords
        # For keywords we use spaCy but can disable ner/parser for speed.
        nlp = spacy.load(args.spacy_model, disable=["ner", "parser"])

        # Dedup + length cap to keep jokes shortish (helps conditioning)
        seen = set()
        filtered = []
        for b in bodies:
            b2 = b.strip()
            if len(b2) < args.min_joke_chars:
                continue
            # cap super long jokes (you can tune)
            if len(b2) > args.max_joke_chars:
                continue

            key = normalize_for_dedup(b2)
            if key in seen:
                continue
            seen.add(key)
            filtered.append(b2)

        print("Keywords: after filter+dedup:", len(filtered))

        # Batch keyword extraction
        lowers = [t.lower() for t in filtered]
        docs = list(nlp.pipe(filtered, batch_size=args.spacy_batch_size))
        for body, low, doc in zip(filtered, lowers, docs):
            kws = extract_two_keywords_from_doc(doc, low, min_len=args.min_keyword_len)
            if not kws:
                continue
            k1, k2 = kws
            if not (contains_word(low, k1) and contains_word(low, k2)):
                continue

            # Prompt is short and stable, like the internet-style approach.
            # (Task token helps when you later reuse the same base, but still single-task training here.)
            prompt = f"<TASK_KEYWORDS>\n{k1} | {k2}"
            completion = body.strip()
            text = make_pc(prompt, completion)
            samples.append(text)

        print("KEYWORDS samples:", len(samples))
        if samples:
            print("Example (raw text):")
            print(samples[0][:400], "...\n")

    if len(samples) < 50:
        raise SystemExit("Too few training samples after preprocessing. Check filters/splitting.")

    # Train/val split
    random.shuffle(samples)
    val_size = max(1, int(len(samples) * args.val_ratio))
    val_texts = samples[:val_size]
    train_texts = samples[val_size:]

    print(f"Train/val: {len(train_texts)} / {len(val_texts)}")

    train_ds = Dataset.from_dict({"text": train_texts})
    val_ds = Dataset.from_dict({"text": val_texts})

    # Load base model
    # Note: Unsloth naming differs between "-bnb-4bit" variants and plain.
    # If you want the "bnb-4bit" HF repo, pass it explicitly via --base_model.
    load_in_4bit = bool(args.load_in_4bit)
    dtype = None  # let Unsloth choose
    print(f"Loading base model: {args.base_model}")
    print(f"Quantization: {'4bit' if args.load_in_4bit else '8bit' if args.load_in_8bit else 'full'}")

    # Unsloth supports load_in_4bit; for 8bit/full you can set False and rely on dtype/torch.
    # Many users do:
    # - 4bit: load_in_4bit=True
    # - full: load_in_4bit=False + fp16/bf16
    # For 8bit: depending on your stack, you may need bitsandbytes 8bit support.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    # Training args
    # Lower LR + a couple epochs tends to be nicer for style/humor.
    targs = TrainingArguments(
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_sched,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        seed=args.seed,
        output_dir=OUTPUT_DIR,
        report_to="none",
        dataloader_pin_memory=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=8,
        args=targs,
    )

    trainer.train()

    # Save LoRA + tokenizer
    model.save_pretrained(SAVE_LORA_DIR)
    tokenizer.save_pretrained(SAVE_LORA_DIR)
    print(f"\nSaved LoRA adapter to: {SAVE_LORA_DIR}")

    # Update "latest"
    safe_symlink(RUN_DIR, os.path.join(RUNS_ROOT, f"latest_{args.task}"))

    # Best tracking
    trainer_state_path = os.path.join(OUTPUT_DIR, "trainer_state.json")
    eval_loss = read_eval_loss(trainer_state_path)

    best_record_path = os.path.join(RUNS_ROOT, f"best_{args.task}.json")
    prev_best = read_best_eval_loss_from_runs(best_record_path)

    if eval_loss is not None:
        print(f"[Run eval_loss] {eval_loss}")
        if prev_best is None or eval_loss < prev_best:
            print(f"[BEST UPDATE] New best (prev={prev_best}) -> {eval_loss}")
            write_best_record(best_record_path, RUN_ID, RUN_DIR, eval_loss)
            safe_symlink(RUN_DIR, os.path.join(RUNS_ROOT, f"best_{args.task}"))
    else:
        print("[WARN] No eval_loss found; best not updated.")

    print(f"\nRun complete.")
    print(f"Latest: runs/latest_{args.task}")
    print(f"Best record: {best_record_path}")


if __name__ == "__main__":
    main()
