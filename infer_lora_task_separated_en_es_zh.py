#!/usr/bin/env python3

import argparse
import os
import sys

import torch
from unsloth import FastLanguageModel
from peft import PeftModel


SEP = "\n\n###\n\n"
END = " END"


# prompts to match training
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


# instruction prompts for comparison model
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


def make_chat_or_fallback(system_msg: str, user_msg: str) -> str:
    return f"<<SYS>>\n{system_msg}\n<</SYS>>\n\n<<USER>>\n{user_msg}\n<</USER>>"


def build_instruct_prompt(task: str, tokenizer, lang: str, k1=None, k2=None, headline=None) -> str:
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


def pick_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_tokenizer(base_model: str, max_seq_length: int, load_in_4bit: bool):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
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

    new_tokens = out[0, input_ids.shape[-1]:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)

    if cut_at_end and END in decoded:
        decoded = decoded.split(END, 1)[0]

    return decoded.strip()


def run_once_finetuned(task: str, lang: str, model, tokenizer, args, k1=None, k2=None, headline=None) -> str:
    if task == "conan":
        if not headline:
            raise ValueError("headline required for conan")
        prompt = build_prompt_conan_ft(headline)
    else:
        if not (k1 and k2):
            raise ValueError("k1 and k2 required for keywords")
        prompt = build_prompt_keywords_ft(k1, k2, lang)

    return generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        cut_at_end=True,
    )


def run_once_instruct(task: str, lang: str, model, tokenizer, args, k1=None, k2=None, headline=None) -> str:
    prompt = build_instruct_prompt(task, tokenizer, lang, k1=k1, k2=k2, headline=headline)
    return generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        cut_at_end=False,
    )


def interactive_loop(args, model_ft, tok_ft, model_cmp=None, tok_cmp=None):
    print("\nInteractive mode. Type 'exit' to quit.")
    print("Commands:")
    print("  k <kw1> <kw2>                 -> keyword joke")
    print("  c <headline text...>          -> conan punchline")
    if args.compare and model_cmp is not None:
        print("  (compare mode prints INSTRUCT then FINETUNED)\n")
    else:
        print()

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not line:
            continue
        if line.lower() in {"exit", "quit"}:
            return

        if line.startswith("k "):
            mode = "keywords"
            parts = line.split()
            if len(parts) < 3:
                print("Usage: k <kw1> <kw2>\n")
                continue
            k1, k2 = parts[1], parts[2]
            headline = None
        elif line.startswith("c "):
            mode = "conan"
            headline = line[2:].strip()
            if not headline:
                print("Usage: c <headline text...>\n")
                continue
            k1 = k2 = None
        else:
            mode = args.task
            if mode == "keywords":
                parts = line.split()
                if len(parts) >= 2:
                    k1, k2 = parts[0], parts[1]
                else:
                    print("Usage: <kw1> <kw2>  (or: k <kw1> <kw2>)\n")
                    continue
                headline = None
            else:
                headline = line
                k1 = k2 = None

        if args.compare and model_cmp is not None:
            inst_out = run_once_instruct(
                task=mode, lang=args.lang,
                model=model_cmp, tokenizer=tok_cmp,
                args=args, k1=k1, k2=k2, headline=headline,
            )
            ft_out = run_once_finetuned(
                task=mode, lang=args.lang,
                model=model_ft, tokenizer=tok_ft,
                args=args, k1=k1, k2=k2, headline=headline,
            )
            print("\n[INSTRUCT]")
            print(inst_out)
            print("\n[FINETUNED]")
            print(ft_out)
            print()
        else:
            out = run_once_finetuned(
                task=mode, lang=args.lang,
                model=model_ft, tokenizer=tok_ft,
                args=args, k1=k1, k2=k2, headline=headline,
            )
            print(out)
            print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["conan", "keywords"], required=True)
    ap.add_argument("--lang", choices=["en", "es", "zh"], required=True)

    # base + lora dirs
    ap.add_argument("--base_model", default="unsloth/mistral-7b-v0.3")
    ap.add_argument("--lora_dir", default=None, help="e.g. runs/latest_conan_es/lora or runs/latest_keywords_zh/lora")
    ap.add_argument("--max_seq_length", type=int, default=768)

    # compare model (baseline instruct model)
    ap.add_argument(
        "--instruct_model",
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Model to use for comparison.",
    )

    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=3407)

    # modes
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--no_lora", action="store_true", help="Use base model only")
    ap.add_argument("--compare", action="store_true", help="Print INSTRUCT then FINETUNED")
    ap.add_argument("--merge", action="store_true", help="Merge LoRA into base (optional)")

    # quantization
    ap.add_argument("--load_in_4bit", action="store_true", help="Load models in 4bit for inference")
    args = ap.parse_args()

    if args.no_lora and args.compare:
        print("Error: --compare requires finetuned model (LoRA). Remove --no_lora.", file=sys.stderr)
        sys.exit(2)

    if (not args.no_lora) and (not args.lora_dir):
        print("Error: Please provide --lora_dir unless you use --no_lora.", file=sys.stderr)
        sys.exit(2)

    load_in_4bit = bool(args.load_in_4bit)

    model_ft, tok_ft = load_model_and_tokenizer(args.base_model, args.max_seq_length, load_in_4bit=load_in_4bit)
    if not args.no_lora:
        model_ft = apply_lora(model_ft, args.lora_dir, merge=args.merge)

    model_cmp = tok_cmp = None
    if args.compare:
        model_cmp, tok_cmp = load_model_and_tokenizer(args.instruct_model, args.max_seq_length, load_in_4bit=load_in_4bit)

    if args.interactive:
        interactive_loop(args, model_ft, tok_ft, model_cmp=model_cmp, tok_cmp=tok_cmp)
        return

    if args.task == "conan":
        headline = input("Setup/sentence1: ").strip()
        if not headline:
            raise SystemExit("Empty setup.")

        if args.compare and model_cmp is not None:
            inst_out = run_once_instruct(args.task, args.lang, model_cmp, tok_cmp, args, headline=headline)
            ft_out = run_once_finetuned(args.task, args.lang, model_ft, tok_ft, args, headline=headline)
            print("\n[INSTRUCT]\n" + inst_out + "\n\n[FINETUNED]\n" + ft_out)
        else:
            out = run_once_finetuned(args.task, args.lang, model_ft, tok_ft, args, headline=headline)
            print(out)

    else:
        k1 = input("Keyword 1: ").strip()
        k2 = input("Keyword 2: ").strip()
        if not (k1 and k2):
            raise SystemExit("Need two keywords.")

        if args.compare and model_cmp is not None:
            inst_out = run_once_instruct(args.task, args.lang, model_cmp, tok_cmp, args, k1=k1, k2=k2)
            ft_out = run_once_finetuned(args.task, args.lang, model_ft, tok_ft, args, k1=k1, k2=k2)
            print("\n[INSTRUCT]\n" + inst_out + "\n\n[FINETUNED]\n" + ft_out)
        else:
            out = run_once_finetuned(args.task, args.lang, model_ft, tok_ft, args, k1=k1, k2=k2)
            print(out)


if __name__ == "__main__":
    main()
