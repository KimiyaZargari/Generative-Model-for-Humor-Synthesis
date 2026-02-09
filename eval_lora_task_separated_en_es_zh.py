#!/usr/bin/env python3

import argparse
import csv
import os
import sys
from typing import Dict, Optional, Tuple

import torch
from unsloth import FastLanguageModel
from peft import PeftModel


SEP = "\n\n###\n\n"
END = " END"

SENT_END_ENES = (".", "?", "!")
SENT_END_ZH = ("。", "？", "！")


# fix punctuation
def ensure_conan_sentence_end(headline: str, lang: str) -> str:
    h = (headline or "").strip()
    if not h:
        return h

    if lang == "zh":
        if not h.endswith(SENT_END_ZH):
            h = h + "。"
    else:
        if not h.endswith(SENT_END_ENES):
            h = h + "."
    return h


# prompts for finetuning to match trainiing
def build_prompt_conan_ft(headline: str) -> str:
    return headline.strip() + SEP


def build_prompt_keywords_ft(k1: str, k2: str, lang: str) -> str:
    k1 = (k1 or "").strip()
    k2 = (k2 or "").strip()

    if lang == "en":
        tag = "<TASK_KEYWORDS>"
    elif lang == "es":
        tag = "<TASK_KEYWORDS_ES>"
    elif lang == "zh":
        tag = "<TASK_KEYWORDS_ZH>"
    else:
        tag = "<TASK_KEYWORDS>"

    return f"{tag}\n{k1} | {k2}".strip() + SEP


# instruct prompts
def make_chat_or_fallback(system_msg: str, user_msg: str) -> str:
    return f"<<SYS>>\n{system_msg}\n<</SYS>>\n\n<<USER>>\n{user_msg}\n<</USER>>"


def build_prompt_conan_instruct(headline: str, lang: str) -> str:
    headline = headline.strip()

    if lang == "es":
        system = (
            "Eres un asistente de escritura cómica. "
            "Sigue las instrucciones del usuario exactamente. "
            "Devuelve solo la continuación del chiste, sin prefacio, sin comillas, sin etiquetas."
        )
        user = (
            "Tarea: Escribe una continuación cómica corta (remate) para el siguiente planteamiento.\n"
            "Restricciones:\n"
            "- Continúa directamente después del planteamiento.\n"
            "- 1–3 frases.\n"
            "- Sin explicaciones ni comentarios meta.\n"
            "- Evita añadir un nuevo planteamiento; entrega el remate.\n\n"
            f"Planteamiento:\n{headline}"
        )
    elif lang == "zh":
        system = (
            "你是一个喜剧写作助手。"
            "严格按照用户要求输出。"
            "只输出笑话的续写/包袱，不要前言、不要引号、不要标签。"
        )
        user = (
            "任务：为下面的铺垫写一个简短的搞笑续写（包袱）。\n"
            "要求：\n"
            "- 直接接着铺垫继续写。\n"
            "- 1–3 句。\n"
            "- 不要解释，不要元评论。\n"
            "- 不要再加新的铺垫，尽快给出包袱。\n\n"
            f"铺垫：\n{headline}"
        )
    else:  # en
        system = (
            "You are a comedy writing assistant. "
            "Follow the user's instructions exactly. "
            "Return only the joke continuation, with no preface, no quotes, no labels."
        )
        user = (
            "Task: Write a short comedic punchline continuation for the given setup.\n"
            "Constraints:\n"
            "- Continue directly after the setup.\n"
            "- 1–3 sentences.\n"
            "- No explanations, no meta-commentary.\n"
            "- Avoid adding your own new setup; deliver the punchline.\n\n"
            f"Setup:\n{headline}"
        )

    return make_chat_or_fallback(system, user)


def build_prompt_keywords_instruct(k1: str, k2: str, lang: str) -> str:
    k1 = (k1 or "").strip()
    k2 = (k2 or "").strip()

    if lang == "es":
        system = (
            "Eres un asistente de escritura cómica. "
            "Sigue las instrucciones del usuario exactamente. "
            "Devuelve solo el texto del chiste, sin prefacio, sin comillas, sin etiquetas."
        )
        user = (
            "Tarea: Escribe un chiste corto que incluya AMBAS palabras clave.\n"
            "Restricciones:\n"
            "- Debe contener exactamente las palabras clave tal como se escriben.\n"
            "- 1–2 frases.\n"
            "- Sin explicaciones ni comentarios meta.\n\n"
            f"Palabras clave: {k1}, {k2}"
        )
    elif lang == "zh":
        system = (
            "你是一个喜剧写作助手。"
            "严格按照用户要求输出。"
            "只输出笑话正文，不要前言、不要引号、不要标签。"
        )
        user = (
            "任务：写一个简短的笑话，必须同时包含下面两个关键词。\n"
            "要求：\n"
            "- 必须包含与给定完全一致的关键词。\n"
            "- 1–2 句。\n"
            "- 不要解释，不要元评论。\n\n"
            f"关键词：{k1}，{k2}"
        )
    else:  # en
        system = (
            "You are a comedy writing assistant. "
            "Follow the user's instructions exactly. "
            "Return only the joke text, with no preface, no quotes, no labels."
        )
        user = (
            "Task: Write a short joke that includes BOTH keywords.\n"
            "Constraints:\n"
            "- Must contain the exact keywords as written.\n"
            "- 1–2 sentences.\n"
            "- No explanations, no meta-commentary.\n\n"
            f"Keywords: {k1}, {k2}"
        )

    return make_chat_or_fallback(system, user)


def build_instruct_prompt(task: str, tokenizer, lang: str, k1=None, k2=None, headline=None) -> str:
    """
    Builds a prompt for the INSTRUCT comparison model.
    Uses tokenizer.apply_chat_template if available, otherwise falls back to a simple format.
    """
    if task == "conan":
        raw = build_prompt_conan_instruct(headline or "", lang)
    else:
        raw = build_prompt_keywords_instruct(k1 or "", k2 or "", lang)

    def extract(tag_open, tag_close, text):
        if tag_open in text and tag_close in text:
            return text.split(tag_open, 1)[1].split(tag_close, 1)[0].strip()
        return ""

    system_msg = extract("<<SYS>>", "<</SYS>>", raw)
    user_msg = extract("<<USER>>", "<</USER>>", raw)

    if hasattr(tokenizer, "apply_chat_template") and callable(getattr(tokenizer, "apply_chat_template")):
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception:
            pass

    return f"{system_msg}\n\n{user_msg}\n\nAnswer:\n"


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_tokenizer(model_name: str, max_seq_length: int, load_in_4bit: bool):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def apply_lora(model, lora_dir: str, merge: bool = False):
    if not os.path.isdir(lora_dir):
        raise FileNotFoundError(f"LoRA directory not found: {lora_dir}")
    model = PeftModel.from_pretrained(model, lora_dir)
    if merge:
        model = model.merge_and_unload()
    FastLanguageModel.for_inference(model)
    return model


@torch.inference_mode()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    seed: int,
    cut_at_end: bool = False,
) -> str:
    device = pick_device()

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(device)

    if tokenizer.pad_token_id is None:
        attention_mask = torch.ones_like(input_ids, device=device)
    else:
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

    if do_sample:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        use_cache=True,
    )

    new_tokens = out[0, input_ids.shape[-1] :]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)

    if cut_at_end and END in decoded:
        decoded = decoded.split(END, 1)[0]

    return decoded.strip()


def norm_cell(x: Optional[str]) -> str:
    return "" if x is None else x.strip()


def is_missing(x: str) -> bool:
    return (x == "") or (x == "-")


def detect_task(word1: str, word2: str, headline: str) -> Tuple[Optional[str], Optional[Dict[str, str]]]:
    w1 = norm_cell(word1)
    w2 = norm_cell(word2)
    hl = norm_cell(headline)

    conan_ok = not is_missing(hl)
    keywords_ok = (not is_missing(w1)) and (not is_missing(w2))

    if conan_ok and (not keywords_ok):
        return "conan", {"headline": hl}
    if keywords_ok and (not conan_ok):
        return "keywords", {"k1": w1, "k2": w2}
    if conan_ok and keywords_ok:
        return "both", {"headline": hl, "k1": w1, "k2": w2}
    return None, None


def build_ft_prompt(task: str, payload: Dict[str, str], lang: str) -> str:
    if task == "conan":
        headline = ensure_conan_sentence_end(payload["headline"], lang)
        return build_prompt_conan_ft(headline)
    if task == "keywords":
        return build_prompt_keywords_ft(payload["k1"], payload["k2"], lang)
    raise ValueError(f"Unknown task: {task}")


def build_cmp_prompt(task: str, payload: Dict[str, str], lang: str, cmp_tokenizer) -> str:
    if task == "conan":
        headline = ensure_conan_sentence_end(payload["headline"], lang)
        return build_instruct_prompt("conan", cmp_tokenizer, lang, headline=headline)
    return build_instruct_prompt("keywords", cmp_tokenizer, lang, k1=payload["k1"], k2=payload["k2"])



def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--lang", choices=["en", "es", "zh"], required=True)

    ap.add_argument("--input_tsv", required=True)
    ap.add_argument("--output_tsv", required=True)

    ap.add_argument("--ft_base_model", default="unsloth/mistral-7b-v0.3")

    ap.add_argument("--lora_conan_dir", required=True)
    ap.add_argument("--lora_keywords_dir", required=True)
    ap.add_argument("--merge", action="store_true", help="Merge LoRA into model weights in RAM for this run.")

    ap.add_argument("--compare_model", default="mistralai/Mistral-7B-Instruct-v0.3")

    ap.add_argument("--max_seq_length", type=int, default=768)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--load_in_4bit", action="store_true")

    ap.add_argument(
        "--conflict_policy",
        choices=["skip", "prefer_conan", "prefer_keywords"],
        default="skip",
        help="What to do if a row looks like BOTH conan and keywords.",
    )

    ap.add_argument("--limit", type=int, default=0, help="If >0, only process first N data rows.")
    ap.add_argument("--print_every", type=int, default=100)

    args = ap.parse_args()

    load_in_4bit = bool(args.load_in_4bit)

    model_conan, tok_conan = load_model_and_tokenizer(args.ft_base_model, args.max_seq_length, load_in_4bit)
    model_conan = apply_lora(model_conan, args.lora_conan_dir, merge=args.merge)

    model_keywords, tok_keywords = load_model_and_tokenizer(args.ft_base_model, args.max_seq_length, load_in_4bit)
    model_keywords = apply_lora(model_keywords, args.lora_keywords_dir, merge=args.merge)

    model_cmp, tok_cmp = load_model_and_tokenizer(args.compare_model, args.max_seq_length, load_in_4bit)

    with open(args.input_tsv, "r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in, delimiter="\t")
        if reader.fieldnames is None:
            raise SystemExit("Input TSV has no header.")

        required = {"id", "word1", "word2", "headline"}
        missing_cols = required - set(reader.fieldnames)
        if missing_cols:
            raise SystemExit(f"Missing required columns in TSV header: {sorted(missing_cols)}")

        out_fieldnames = list(reader.fieldnames) + [
            "task",
            "lora_used",
            "output_finetuned",
            "output_compare",
        ]

        with open(args.output_tsv, "w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, delimiter="\t", fieldnames=out_fieldnames, extrasaction="ignore")
            writer.writeheader()

            n_total = 0
            n_done = 0
            n_skipped = 0

            for row in reader:
                n_total += 1
                if args.limit > 0 and n_total > args.limit:
                    break

                w1 = row.get("word1", "")
                w2 = row.get("word2", "")
                hl = row.get("headline", "")

                task, payload = detect_task(w1, w2, hl)

                if task is None:
                    n_skipped += 1
                    continue

                if task == "both":
                    if args.conflict_policy == "skip":
                        n_skipped += 1
                        continue
                    elif args.conflict_policy == "prefer_conan":
                        task = "conan"
                        payload = {"headline": payload["headline"]}
                    else:
                        task = "keywords"
                        payload = {"k1": payload["k1"], "k2": payload["k2"]}

                if task == "conan":
                    ft_model = model_conan
                    ft_tok = tok_conan
                    lora_used = "conan"
                else:
                    ft_model = model_keywords
                    ft_tok = tok_keywords
                    lora_used = "keywords"

                prompt_ft = build_ft_prompt(task, payload, args.lang)
                prompt_cmp = build_cmp_prompt(task, payload, args.lang, tok_cmp)

                out_ft = generate_text(
                    model=ft_model,
                    tokenizer=ft_tok,
                    prompt=prompt_ft,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=args.seed,
                    cut_at_end=True,
                )

                out_cmp = generate_text(
                    model=model_cmp,
                    tokenizer=tok_cmp,
                    prompt=prompt_cmp,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=args.seed,
                    cut_at_end=False,
                )

                out_row = dict(row)
                out_row.update(
                    {
                        "task": task,
                        "lora_used": lora_used,
                        "output_finetuned": out_ft,
                        "output_compare": out_cmp,
                    }
                )
                writer.writerow(out_row)

                n_done += 1
                if args.print_every > 0 and (n_done % args.print_every == 0):
                    print(f"Processed {n_done} rows (skipped {n_skipped})...", file=sys.stderr)

    print(f"Done. Wrote: {args.output_tsv} | processed={n_done} skipped={n_skipped}", file=sys.stderr)


if __name__ == "__main__":
    main()
