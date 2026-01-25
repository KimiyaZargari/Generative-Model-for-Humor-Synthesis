#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_single_task_endstop_multilang.py

Single-task finetuning (LoRA) with "###" separator + "END" stop token,
same prompt/completion style as your current script, but extended for:
- English (en) / Spanish (es) / Chinese (zh)
- Loading from Hugging Face datasets OR your existing JSON {body: ...}
- Optional "humor-ish" filtering for OSCAR-style corpora
- Optional translation-bootstrapping: translate EN jokes into ES/ZH and train on them

Tasks:
- --task conan     : sentence1 -> sentence2..n continuation (setup -> punchline)
- --task keywords  : (k1, k2) -> generate joke including both keywords (language-neutral token picking)

Prompt format:
- Prompt ends with "\n\n###\n\n"
- Target ends with " END"

Examples:
  # Spanish / Chinese from HF OSCAR with humor-ish filter:
  python train_single_task_endstop_multilang.py --task conan --lang es --source hf --hf_dataset oscar-corpus/OSCAR-2301 --hf_config es --max_samples 80000
  python train_single_task_endstop_multilang.py --task conan --lang zh --source hf --hf_dataset oscar-corpus/OSCAR-2301 --hf_config zh --max_samples 80000

  # Spanish / Chinese from your own JSON:
  python train_single_task_endstop_multilang.py --task conan --lang es --source json --data_path my_spanish_jokes.json
  python train_single_task_endstop_multilang.py --task conan --lang zh --source json --data_path my_chinese_jokes.json

  # Bootstrap ES/ZH by translating your EN jokes JSON:
  python train_single_task_endstop_multilang.py --task conan --lang es --source json --data_path conan_jokes.json --translate_from_en --translation_limit 30000
  python train_single_task_endstop_multilang.py --task conan --lang zh --source json --data_path conan_jokes.json --translate_from_en --translation_limit 30000

Notes:
- For ES/ZH, we do sentence splitting with simple regex (no spaCy dependency needed).
- For keywords task, spaCy POS-based extraction is EN-only; here we do language-neutral token picking.
- Best practice: train separate LoRAs per language (en/es/zh). Don’t mix in one run initially.

Deps:
  pip install unsloth trl transformers datasets accelerate bitsandbytes sentencepiece
  (Optional translation) pip install sacremoses

"""

import os
import re
import json
import random
import unicodedata
from datetime import datetime
from collections import Counter

import torch
import numpy as np
from datasets import Dataset, load_dataset

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
# Prompt/completion format
# -----------------------------
SEP = "\n\n###\n\n"
END = " END"

def make_pc(prompt_text: str, completion_text: str) -> str:
    prompt = prompt_text.rstrip() + SEP
    completion = " " + completion_text.strip() + END
    return prompt + completion


# -----------------------------
# Cleaning & sentence splitting
# -----------------------------
def clean_text_basic(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s

def clean_conan_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = s.replace("—", "--")
    s = s.replace("  ", " ").strip()
    return s

def split_sentences_simple(text: str, lang: str):
    text = text.strip()
    if not text:
        return []
    if lang in ("en", "es"):
        # split on . ! ? with whitespace after
        sents = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sents if s and s.strip()]
    elif lang == "zh":
        # keep Chinese sentence punctuation
        parts = re.split(r"(?<=[。！？])", text)
        return [p.strip() for p in parts if p and p.strip()]
    else:
        # fallback
        sents = re.split(r"(?<=[.!?。！？])\s*", text)
        return [s.strip() for s in sents if s and s.strip()]


# -----------------------------
# Humor-ish filtering (optional)
# -----------------------------
HUMOR_PATTERNS = {
    "en": re.compile(r"\b(joke|funny|haha|hahaha|laugh|pun|comed(y|ic))\b", re.IGNORECASE),
    "es": re.compile(r"\b(chiste|broma|gracioso|humor|jajaja|ja ja|me re[ií]o)\b", re.IGNORECASE),
    "zh": re.compile(r"(笑|哈哈|段子|搞笑|幽默|逗)", re.IGNORECASE),
}

def looks_humorous(text: str, lang: str) -> bool:
    pat = HUMOR_PATTERNS.get(lang)
    if not pat:
        return False
    return pat.search(text) is not None


# -----------------------------
# Keywords helpers (language-neutral)
# -----------------------------
STOP_RE = re.compile(r"^[^a-zA-Z]+$")  # legacy; not used for zh

GENERIC = {
    "thing", "stuff", "something", "someone", "anything", "everything",
    "place", "way", "lot", "kind"
}

def normalize_for_dedup(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def contains_word_en_es(text_lower: str, w: str) -> bool:
    return re.search(rf"\b{re.escape(w)}\b", text_lower) is not None

def pick_two_tokens(text: str, lang: str, min_len=3):
    if lang == "zh":
        # choose 2 distinct CJK chars as "keywords" (simple but works)
        toks = list(set(re.findall(r"[\u4e00-\u9fff]", text)))
        if len(toks) < 2:
            return None
        return tuple(random.sample(toks, 2))
    else:
        # words length >= min_len, avoid ultra generic list
        words = re.findall(r"\b\w+\b", text.lower())
        words = [w for w in words if len(w) >= min_len and w not in GENERIC]
        if len(set(words)) < 2:
            return None
        return tuple(random.sample(list(set(words)), 2))


# -----------------------------
# Data loading
# -----------------------------
def load_bodies_from_json(data_path: str):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    bodies = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                b = item.get("body", "")
                if isinstance(b, str) and b.strip():
                    bodies.append(b.strip())
            elif isinstance(item, str) and item.strip():
                bodies.append(item.strip())
    elif isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        for item in data["data"]:
            if isinstance(item, dict):
                b = item.get("body", "")
                if isinstance(b, str) and b.strip():
                    bodies.append(b.strip())
    return bodies

def load_texts_from_hf(
    hf_dataset: str,
    hf_config: str | None,
    hf_split: str,
    hf_text_field: str,
    max_samples: int | None,
    seed: int,
):
    ds = load_dataset(hf_dataset, hf_config, split=hf_split)
    # try to be robust: if text_field missing, find a likely one
    if hf_text_field not in ds.column_names:
        candidates = ["text", "content", "body", "article", "prompt", "completion"]
        found = None
        for c in candidates:
            if c in ds.column_names:
                found = c
                break
        if not found:
            raise ValueError(f"Could not find text field '{hf_text_field}' in {ds.column_names}")
        hf_text_field = found
        print(f"[INFO] Using detected text field: {hf_text_field}")

    texts = [x for x in ds[hf_text_field] if isinstance(x, str) and x.strip()]
    random.Random(seed).shuffle(texts)
    if max_samples is not None:
        texts = texts[:max_samples]
    return texts


# -----------------------------
# Optional translation bootstrap (EN -> ES/ZH)
# -----------------------------
def make_translator(src: str, tgt: str):
    from transformers import pipeline
    model = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    # device=0 -> GPU, falls vorhanden
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("translation", model=model, device=device)

def translate_batch(texts, translator, batch_size=32):
    # pipeline kann direkt Listen mit batch_size effizienter abarbeiten
    res = translator(texts, batch_size=batch_size, truncation=True)
    return [x["translation_text"] for x in res]



def prefilter_for_translation(texts, max_chars=512):
    cleaned = []
    for t in texts:
        if not isinstance(t, str):
            continue
        t = t.strip()
        if not t:
            continue
        # remove weird control chars (optional, aber hilft)
        t = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        if len(t) > max_chars:
            t = t[:max_chars]
        cleaned.append(t)
    return cleaned




# -----------------------------
# Main
# -----------------------------
def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["conan", "keywords"], required=True)
    ap.add_argument("--lang", choices=["en", "es", "zh"], required=True)

    # Data source: either your JSON or HuggingFace dataset
    ap.add_argument("--source", choices=["json", "hf"], required=True)
    ap.add_argument("--data_path", default=None, help="JSON file with list of {body: ...} items (required if --source json)")

    # HF dataset options
    ap.add_argument("--hf_dataset", default="oscar-corpus/OSCAR-2301")
    ap.add_argument("--hf_config", default=None, help='e.g. "es" or "zh" for OSCAR. Some datasets have no config.')
    ap.add_argument("--hf_split", default="train")
    ap.add_argument("--hf_text_field", default="text")

    # Humor-ish filtering (useful for general web corpora like OSCAR)
    ap.add_argument("--humor_filter", action="store_true", help="Keep only humor-ish lines (regex heuristics). Recommended with OSCAR.")
    ap.add_argument("--min_text_chars", type=int, default=40)
    ap.add_argument("--max_text_chars", type=int, default=600)

    # Optional translation bootstrap: if you only have EN jokes locally
    ap.add_argument("--translate_from_en", action="store_true", help="Translate EN bodies to target --lang (es/zh). Only valid with --source json and --lang in {es, zh}.")
    ap.add_argument("--translation_limit", type=int, default=30000, help="Max EN samples to translate (cost/time control).")

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
    ap.add_argument("--max_completion_chars", type=int, default=300, help="Cap punchline length")

    # Keywords controls:
    ap.add_argument("--min_joke_chars", type=int, default=20)
    ap.add_argument("--max_joke_chars", type=int, default=300)
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

    # sanity
    if args.source == "json" and not args.data_path:
        raise SystemExit("--data_path required when --source json")
    if args.translate_from_en:
        if args.source != "json":
            raise SystemExit("--translate_from_en only supported with --source json (needs EN jokes input).")
        if args.lang not in ("es", "zh"):
            raise SystemExit("--translate_from_en requires --lang es or zh.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Quantization selection:
    if not (args.load_in_4bit or args.load_in_8bit or args.full_precision):
        args.load_in_4bit = True
    chosen = sum([args.load_in_4bit, args.load_in_8bit, args.full_precision])
    if chosen != 1:
        raise SystemExit("Choose exactly one: --load_in_4bit OR --load_in_8bit OR --full_precision")

    # Runs
    RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUNS_ROOT = "runs"
    RUN_DIR = os.path.join(RUNS_ROOT, f"{RUN_ID}_{args.task}_{args.lang}")
    OUTPUT_DIR = os.path.join(RUN_DIR, "trainer_outputs")
    SAVE_LORA_DIR = os.path.join(RUN_DIR, "lora")
    ensure_dir(RUN_DIR)
    ensure_dir(OUTPUT_DIR)
    ensure_dir(SAVE_LORA_DIR)

    # Save config
    config_path = os.path.join(RUN_DIR, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\n=== RUN_ID: {RUN_ID} (task={args.task}, lang={args.lang}) ===")
    print(f"Run dir: {RUN_DIR}")
    print(f"Config saved: {config_path}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # -----------------------------
    # Load data
    # -----------------------------
    if args.source == "json":
        bodies = load_bodies_from_json(args.data_path)
        print("Loaded JSON bodies:", len(bodies))

        # optional translation bootstrap
        if args.translate_from_en:
            limit = min(len(bodies), args.translation_limit)
            en_subset = bodies[:limit]
            en_subset = prefilter_for_translation(en_subset, max_chars=512)
            print(f"[TRANS] Translating {len(en_subset)} EN samples -> {args.lang} ...")
            translator = make_translator("en", args.lang)
            bodies = translate_batch(en_subset, translator, batch_size=32)
            print("[TRANS] Done. Translated bodies:", len(bodies))

    else:
        bodies = load_texts_from_hf(
            hf_dataset=args.hf_dataset,
            hf_config=args.hf_config,
            hf_split=args.hf_split,
            hf_text_field=args.hf_text_field,
            max_samples=args.max_samples,
            seed=args.seed,
        )
        print("Loaded HF texts:", len(bodies))

    random.shuffle(bodies)
    if args.max_samples is not None and args.source == "json":
        bodies = bodies[: args.max_samples]

    # light cleaning + length filtering
    cleaned = []
    for t in bodies:
        t2 = clean_text_basic(t)
        if len(t2) < args.min_text_chars:
            continue
        if len(t2) > args.max_text_chars:
            continue
        if args.humor_filter and not looks_humorous(t2, args.lang):
            continue
        cleaned.append(t2)

    bodies = cleaned
    print("After clean/len/filter:", len(bodies))

    if len(bodies) < 100:
        print("[WARN] Very few samples after filtering. Consider disabling --humor_filter or raising --max_samples.")

    # -----------------------------
    # Build prompt-completion samples
    # -----------------------------
    samples = []

    if args.task == "conan":
        for b in bodies:
            # keep your original conan-only skip if you want
            if args.exclude_if_contains_conan and "conan" in b.lower():
                continue

            clean = clean_conan_text(b)
            sents = split_sentences_simple(clean, args.lang)
            if len(sents) < args.min_sentences:
                continue

            prompt = sents[0].strip()
            completion = " ".join(sents[1:]).strip()

            if len(completion) > args.max_completion_chars:
                completion = completion[: args.max_completion_chars].rstrip()

            if len(prompt) < 10 or len(completion) < 10:
                continue

            samples.append(make_pc(prompt, completion))

        print("CONAN samples:", len(samples))
        if samples:
            print("Example (raw text):")
            print(samples[0][:400], "...\n")

    else:  # keywords
        seen = set()
        filtered = []
        for b in bodies:
            b2 = b.strip()
            if len(b2) < args.min_joke_chars:
                continue
            if len(b2) > args.max_joke_chars:
                continue
            key = normalize_for_dedup(b2)
            if key in seen:
                continue
            seen.add(key)
            filtered.append(b2)

        print("Keywords: after filter+dedup:", len(filtered))

        for body in filtered:
            kws = pick_two_tokens(body, args.lang, min_len=args.min_keyword_len)
            if not kws:
                continue
            k1, k2 = kws

            if args.lang == "zh":
                # for zh we don't do word-boundary checks
                if (k1 not in body) or (k2 not in body):
                    continue
            else:
                low = body.lower()
                if not (contains_word_en_es(low, k1) and contains_word_en_es(low, k2)):
                    continue

            prompt = f"<TASK_KEYWORDS_{args.lang.upper()}>\n{k1} | {k2}"
            completion = body.strip()
            samples.append(make_pc(prompt, completion))

        print("KEYWORDS samples:", len(samples))
        if samples:
            print("Example (raw text):")
            print(samples[0][:400], "...\n")

    if len(samples) < 50:
        raise SystemExit("Too few training samples after preprocessing. Adjust filters/max_samples.")

    # -----------------------------
    # Train/val split
    # -----------------------------
    random.shuffle(samples)
    val_size = max(1, int(len(samples) * args.val_ratio))
    val_texts = samples[:val_size]
    train_texts = samples[val_size:]

    print(f"Train/val: {len(train_texts)} / {len(val_texts)}")

    train_ds = Dataset.from_dict({"text": train_texts})
    val_ds = Dataset.from_dict({"text": val_texts})

    # -----------------------------
    # Load base model
    # -----------------------------
    load_in_4bit = bool(args.load_in_4bit)
    dtype = None
    print(f"Loading base model: {args.base_model}")
    print(f"Quantization: {'4bit' if args.load_in_4bit else '8bit' if args.load_in_8bit else 'full'}")

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
    # For zh, a slightly lower LR often helps stability (optional: you can set via CLI).
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
    safe_symlink(RUN_DIR, os.path.join("runs", f"latest_{args.task}_{args.lang}"))

    # Best tracking
    trainer_state_path = os.path.join(OUTPUT_DIR, "trainer_state.json")
    eval_loss = read_eval_loss(trainer_state_path)

    best_record_path = os.path.join("runs", f"best_{args.task}_{args.lang}.json")
    prev_best = read_best_eval_loss_from_runs(best_record_path)

    if eval_loss is not None:
        print(f"[Run eval_loss] {eval_loss}")
        if prev_best is None or eval_loss < prev_best:
            print(f"[BEST UPDATE] New best (prev={prev_best}) -> {eval_loss}")
            write_best_record(best_record_path, RUN_ID, RUN_DIR, eval_loss)
            safe_symlink(RUN_DIR, os.path.join("runs", f"best_{args.task}_{args.lang}"))
    else:
        print("[WARN] No eval_loss found; best not updated.")

    print(f"\nRun complete.")
    print(f"Latest: runs/latest_{args.task}_{args.lang}")
    print(f"Best record: {best_record_path}")


if __name__ == "__main__":
    main()
