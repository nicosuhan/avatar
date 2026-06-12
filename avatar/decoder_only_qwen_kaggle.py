import json
import math
import os
import random
import re
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["TRANSFORMERS_NO_TORCHAUDIO"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import DataLoader, Dataset
import transformers.utils.import_utils as hf_import_utils
import transformers.utils as hf_utils

if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = getattr(torch, "float8_e4m3fn", getattr(torch, "float8_e5m2", torch.float16))

hf_import_utils._tf_available = False
hf_import_utils._flax_available = False
hf_import_utils._jax_available = False
hf_import_utils._torchvision_available = False
hf_import_utils._pil_available = False
hf_import_utils._sklearn_available = False
hf_import_utils._scipy_available = False
hf_import_utils._torchao_available = False
hf_import_utils.is_torchvision_available = lambda: False
hf_import_utils.is_vision_available = lambda: False
hf_import_utils.is_sklearn_available = lambda: False
hf_import_utils.is_scipy_available = lambda: False
hf_utils.is_torchvision_available = lambda: False
hf_utils.is_vision_available = lambda: False
hf_utils.is_sklearn_available = lambda: False
hf_utils.is_scipy_available = lambda: False

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)


KAGGLE_DATA_DIRS = [
    Path("/kaggle/input/datasets/nicoletacorinasuhan/avatar-dataset/data/avatar"),
    Path("/kaggle/input/avatar-dataset/data/avatar"),
]
LOCAL_DATA_DIR = (Path(__file__).resolve().parent / "data" / "avatar") if "__file__" in globals() else (Path.cwd() / "avatar" / "data" / "avatar")
IS_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None or (os.name != "nt" and Path("/kaggle/working").exists())
DATA_DIR = next((path for path in KAGGLE_DATA_DIRS if path.exists()), LOCAL_DATA_DIR)
OUT = Path("/kaggle/working/outputs") if IS_KAGGLE else (Path.cwd() / "outputs")

MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
TASKS = [("python2java", "python", "java"), ("java2python", "java", "python")]
RUN_NAME = "qwen25_coder_05b_decoder_only_ultrafast"

TRAIN = True
EVAL_ONLY = False
GENERATE = True

LIMIT_TRAIN = 12000
LIMIT_VALID = 128
LIMIT_TEST = 0
START_INDEX = None
ALLOW_CPU_FALLBACK = False

EPOCHS = 1
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
GENERATE_BATCH_SIZE = 24
MAX_SOURCE_LENGTH = 320
MAX_TARGET_LENGTH = 320
MAX_TOTAL_LENGTH = 640
VALID_GENERATION_LIMIT = 0
VALID_BLEU_EVERY_N_EPOCHS = 1

LR = 3e-4
WEIGHT_DECAY = 0.0
WARMUP_RATIO = 0.06
MAX_GRAD_NORM = 1.0

LORA_R = 4
LORA_ALPHA = 8
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
GRADIENT_CHECKPOINTING = False

SEED = 1234
NUM_WORKERS = 2
PREFETCH_FACTOR = 2
LOG_EVERY_STEPS = 100
SAVE_TEMP_PROGRESS_EVERY_STEPS = 100
REPETITION_PENALTY = 1.0

P100_TORCH_INSTALL_HINT = """
This Kaggle runtime has a CUDA GPU, but the installed PyTorch wheel does not include kernels for it.
For Tesla P100, either switch the Kaggle accelerator to T4/L4/A100, or run this in a separate first cell and restart the session:

!pip uninstall -y torch torchvision torchaudio torchtext xformers transformers peft accelerate bitsandbytes scikit-learn scipy pillow
!pip install --no-cache-dir --force-reinstall torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
!pip install --no-cache-dir transformers==4.46.3 peft==0.13.2 accelerate==1.1.1 bitsandbytes==0.45.5

After restart, run:
import torch
print(torch.__version__)
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print(torch.cuda.get_arch_list())

The arch list must contain the current GPU architecture, for example sm_60 for Tesla P100.
"""


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cuda_build_supports_current_gpu():
    if not torch.cuda.is_available():
        return False, None, None, []
    gpu = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    arch = f"sm_{capability[0]}{capability[1]}"
    compute = f"compute_{capability[0]}{capability[1]}"
    arch_list = torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else []
    if not arch_list:
        return True, gpu, capability, arch_list
    return arch in arch_list or compute in arch_list, gpu, capability, arch_list


def select_device():
    if not torch.cuda.is_available():
        if ALLOW_CPU_FALLBACK:
            return torch.device("cpu"), False, "cpu", None, []
        raise RuntimeError("CUDA is not available. Enable a Kaggle GPU accelerator before running this training script.")

    supported, gpu, capability, arch_list = cuda_build_supports_current_gpu()
    if not supported:
        message = (
            f"PyTorch {torch.__version__} cannot run CUDA kernels on {gpu} with capability {capability}. "
            f"This build supports: {arch_list}.\n{P100_TORCH_INSTALL_HINT}"
        )
        if ALLOW_CPU_FALLBACK:
            print("warning:", message, flush=True)
            return torch.device("cpu"), False, gpu, capability, arch_list
        raise RuntimeError(message)

    return torch.device("cuda"), True, gpu, capability, arch_list


def read_lines(path):
    return Path(path).read_text(encoding="utf-8").splitlines()


def load_parallel(data_dir, split, source, target, limit=0):
    src = read_lines(Path(data_dir) / f"{split}.java-python.{source}")
    tgt = read_lines(Path(data_dir) / f"{split}.java-python.{target}")
    if limit:
        src = src[:limit]
        tgt = tgt[:limit]
    return src, tgt


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def append_jsonl(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def batch_file(out_dir, prefix, start, end):
    return out_dir / f"{prefix}_batch_{start:04d}_{end - 1:04d}.pred"


def batch_lines(path):
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def next_batch(out_dir, prefix, total):
    for start in range(0, total, GENERATE_BATCH_SIZE):
        end = min(start + GENERATE_BATCH_SIZE, total)
        path = batch_file(out_dir, prefix, start, end)
        if len(batch_lines(path)) != end - start:
            return start
    return total


def save_batch(out_dir, prefix, start, predictions):
    end = start + len(predictions)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_file(out_dir, prefix, start, end).write_text("\n".join(predictions) + "\n", encoding="utf-8")


def merge_batches(out_dir, prefix, total):
    predictions = []
    missing = []
    for start in range(0, total, GENERATE_BATCH_SIZE):
        end = min(start + GENERATE_BATCH_SIZE, total)
        path = batch_file(out_dir, prefix, start, end)
        lines = batch_lines(path)
        if len(lines) != end - start:
            missing.append(path.name)
            continue
        predictions.extend(lines)
    if missing:
        print(prefix, "merge skipped; missing or incomplete batches:", ", ".join(missing[:10]), flush=True)
        return False
    (out_dir / "test.pred").write_text("\n".join(predictions) + "\n", encoding="utf-8")
    print(prefix, "merged", len(predictions), "predictions into", out_dir / "test.pred", flush=True)
    return True


def normalize_prediction(text):
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", text).strip()


def truncate_source_for_prompt(tokenizer, code):
    ids = tokenizer(code, add_special_tokens=False)["input_ids"]
    if len(ids) <= MAX_SOURCE_LENGTH:
        return code
    ids = ids[:MAX_SOURCE_LENGTH]
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def tokenizer_with_left_truncation(tokenizer, *args, **kwargs):
    previous_truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = "left"
    try:
        return tokenizer(*args, **kwargs)
    finally:
        tokenizer.truncation_side = previous_truncation_side


class TranslationDataset(Dataset):
    def __init__(self, sources, targets):
        self.sources = sources
        self.targets = targets

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, index):
        return self.sources[index], self.targets[index]


def make_prompt(tokenizer, src_lang, tgt_lang, code):
    code = truncate_source_for_prompt(tokenizer, code)
    content = (
        f"Translate this token-spaced {src_lang} program into token-spaced {tgt_lang}.\n"
        "Return only the translated token stream.\n\n"
        f"Source:\n{code}"
    )
    messages = [
        {
            "role": "system",
            "content": f"You translate token-spaced {src_lang} programs into token-spaced {tgt_lang}.",
        },
        {"role": "user", "content": content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def encode_example(tokenizer, src_lang, tgt_lang, source, target):
    target_ids = tokenizer(
        target,
        add_special_tokens=False,
        truncation=True,
        max_length=max(MAX_TARGET_LENGTH - 1, 1),
    )["input_ids"]
    target_ids = target_ids + [tokenizer.eos_token_id]

    prompt_budget = max(MAX_TOTAL_LENGTH - len(target_ids), 1)
    prompt = make_prompt(tokenizer, src_lang, tgt_lang, source)
    prompt_ids = tokenizer_with_left_truncation(
        tokenizer,
        prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=prompt_budget,
    )["input_ids"]

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    if len(input_ids) > MAX_TOTAL_LENGTH:
        input_ids = input_ids[-MAX_TOTAL_LENGTH:]
        labels = labels[-MAX_TOTAL_LENGTH:]
    return input_ids, labels


def left_pad(sequences, pad_value):
    max_len = max(len(sequence) for sequence in sequences)
    padded = []
    for sequence in sequences:
        pad_len = max_len - len(sequence)
        padded.append([pad_value] * pad_len + sequence)
    return torch.tensor(padded, dtype=torch.long)


def make_collate_fn(tokenizer, src_lang, tgt_lang):
    pad_id = tokenizer.pad_token_id

    def collate(batch):
        sources, targets = zip(*batch)
        encoded = [encode_example(tokenizer, src_lang, tgt_lang, source, target) for source, target in zip(sources, targets)]
        input_ids = [item[0] for item in encoded]
        labels = [item[1] for item in encoded]
        padded_input_ids = left_pad(input_ids, pad_id)
        padded_labels = left_pad(labels, -100)
        attention_mask = left_pad([[1] * len(sequence) for sequence in input_ids], 0)
        return {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "labels": padded_labels,
        }

    return collate


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def autocast_context(enabled):
    if enabled:
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def prepare_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = tokenizer.pad_token
    return tokenizer


def build_model(device):
    if device.type != "cuda":
        raise RuntimeError("This QLoRA script requires CUDA. Enable a Kaggle GPU accelerator before running it.")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
    )
    model.config.use_cache = False
    try:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=GRADIENT_CHECKPOINTING)
    except TypeError:
        model = prepare_model_for_kbit_training(model)
        if GRADIENT_CHECKPOINTING and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def build_optimizer(model):
    return torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def build_linear_warmup_decay_scheduler(optimizer, warmup_steps, total_steps):
    warmup_steps = max(int(warmup_steps), 0)
    total_steps = max(int(total_steps), 1)

    def lr_lambda(current_step):
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        remaining_steps = total_steps - current_step
        decay_steps = max(1, total_steps - warmup_steps)
        return max(0.0, float(remaining_steps) / float(decay_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def data_loader_worker_kwargs():
    if NUM_WORKERS <= 0:
        return {}
    return {
        "persistent_workers": True,
        "prefetch_factor": PREFETCH_FACTOR,
    }


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_valid_loss, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "adapter": get_peft_model_state_dict(model),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "best_valid_loss": best_valid_loss,
            "history": history,
        },
        path,
    )


def load_checkpoint(path, model, device, optimizer=None, scheduler=None, scaler=None):
    checkpoint = torch.load(path, map_location=device)
    if "adapter" in checkpoint:
        set_peft_model_state_dict(model, checkpoint["adapter"])
    else:
        model.load_state_dict(checkpoint["model"], strict=False)
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


def _get_ngrams(segment, max_order):
    counts = Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            counts[tuple(segment[i : i + order])] += 1
    return counts


def corpus_bleu(reference_corpus, translation_corpus, max_order=4, smooth=True):
    matches = [0] * max_order
    possible = [0] * max_order
    ref_len = 0
    hyp_len = 0
    for refs, hyp in zip(reference_corpus, translation_corpus):
        ref_len += min(len(ref) for ref in refs)
        hyp_len += len(hyp)
        ref_counts = Counter()
        for ref in refs:
            ref_counts |= _get_ngrams(ref, max_order)
        hyp_counts = _get_ngrams(hyp, max_order)
        overlap = hyp_counts & ref_counts
        for ngram, count in overlap.items():
            matches[len(ngram) - 1] += count
        for order in range(1, max_order + 1):
            count = len(hyp) - order + 1
            if count > 0:
                possible[order - 1] += count
    precisions = []
    for i in range(max_order):
        if smooth:
            precisions.append((matches[i] + 1.0) / (possible[i] + 1.0))
        elif possible[i]:
            precisions.append(matches[i] / possible[i])
        else:
            precisions.append(0.0)
    geo_mean = math.exp(sum(math.log(p) / max_order for p in precisions)) if min(precisions) > 0 else 0.0
    ratio = hyp_len / max(ref_len, 1)
    bp = 1.0 if ratio > 1.0 else math.exp(1.0 - 1.0 / max(ratio, 1e-9))
    return 100.0 * geo_mean * bp


def generation_prompt_inputs(tokenizer, src_lang, tgt_lang, sources, device):
    prompts = [make_prompt(tokenizer, src_lang, tgt_lang, source) for source in sources]
    return tokenizer_with_left_truncation(
        tokenizer,
        prompts,
        padding=True,
        truncation=True,
        max_length=max(MAX_TOTAL_LENGTH - MAX_TARGET_LENGTH, 1),
        pad_to_multiple_of=8,
        return_tensors="pt",
    ).to(device)


def generate_prompt_batch(model, tokenizer, src_lang, tgt_lang, sources, device):
    inputs = output_ids = None
    try:
        inputs = generation_prompt_inputs(tokenizer, src_lang, tgt_lang, sources, device)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_TARGET_LENGTH,
                do_sample=False,
                num_beams=1,
                repetition_penalty=REPETITION_PENALTY,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return [normalize_prediction(text) for text in texts]
    finally:
        if inputs is not None:
            del inputs
        if output_ids is not None:
            del output_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def generate_batch(model, tokenizer, src_lang, tgt_lang, sources, device):
    try:
        return generate_prompt_batch(model, tokenizer, src_lang, tgt_lang, sources, device)
    except torch.cuda.OutOfMemoryError:
        if len(sources) == 1:
            raise
        mid = len(sources) // 2
        return generate_batch(model, tokenizer, src_lang, tgt_lang, sources[:mid], device) + generate_batch(
            model,
            tokenizer,
            src_lang,
            tgt_lang,
            sources[mid:],
            device,
        )


def generate_texts(model, tokenizer, src_lang, tgt_lang, sources, batch_size, device):
    predictions = []
    model.eval()
    model.config.use_cache = True
    for start in range(0, len(sources), batch_size):
        predictions.extend(generate_batch(model, tokenizer, src_lang, tgt_lang, sources[start : start + batch_size], device))
    return predictions


def validate_loss(model, data_loader, device, use_amp):
    model.eval()
    model.config.use_cache = False
    losses = []
    with torch.no_grad():
        for batch in data_loader:
            batch = to_device(batch, device)
            with autocast_context(use_amp):
                loss = model(**batch).loss
            losses.append(float(loss.detach().cpu()))
    return round(sum(losses) / max(len(losses), 1), 4)


def validate_bleu(model, tokenizer, src_lang, tgt_lang, valid_src, valid_tgt, device):
    limit = min(VALID_GENERATION_LIMIT, len(valid_src))
    if limit <= 0:
        return 0.0
    predictions = generate_texts(model, tokenizer, src_lang, tgt_lang, valid_src[:limit], GENERATE_BATCH_SIZE, device)
    refs = [[target.split()] for target in valid_tgt[:limit]]
    hyps = [prediction.split() for prediction in predictions]
    return round(corpus_bleu(refs, hyps), 2)


def train_model(task_name, source, target, model, tokenizer, train_src, train_tgt, valid_src, valid_tgt, out_dir, device, use_amp):
    collate_fn = make_collate_fn(tokenizer, source, target)
    worker_kwargs = data_loader_worker_kwargs()
    train_loader = DataLoader(
        TranslationDataset(train_src, train_tgt),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
        **worker_kwargs,
    )
    valid_loader = DataLoader(
        TranslationDataset(valid_src, valid_tgt),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
        **worker_kwargs,
    )
    optimizer = build_optimizer(model)
    updates_per_epoch = max(math.ceil(len(train_loader) / max(GRAD_ACCUM_STEPS, 1)), 1)
    total_updates = updates_per_epoch * max(EPOCHS, 1)
    warmup_steps = int(total_updates * WARMUP_RATIO)
    scheduler = build_linear_warmup_decay_scheduler(optimizer, warmup_steps, total_updates)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    latest_path = out_dir / "latest.pt"
    best_path = out_dir / "best.pt"
    history_path = out_dir / "history.json"
    temp_dir = out_dir / "tmp"
    step_progress_path = temp_dir / "training_progress.jsonl"
    latest_progress_path = temp_dir / "latest_training_progress.json"
    epoch_progress_path = temp_dir / "epoch_summaries.jsonl"
    start_epoch = 0
    best_valid_loss = float("inf")
    history = []
    if latest_path.exists():
        checkpoint = load_checkpoint(latest_path, model, device, optimizer=optimizer, scheduler=scheduler, scaler=scaler)
        start_epoch = int(checkpoint.get("epoch", 0))
        best_valid_loss = float(checkpoint.get("best_valid_loss", float("inf")))
        history = list(checkpoint.get("history", []))
        print(task_name, "restored", latest_path, "start_epoch", start_epoch, flush=True)

    if start_epoch >= EPOCHS:
        print(task_name, "training already complete through epoch", start_epoch, flush=True)
        return

    model.config.use_cache = False
    for epoch in range(start_epoch + 1, EPOCHS + 1):
        model.train()
        model.config.use_cache = False
        optimizer.zero_grad(set_to_none=True)
        losses = []
        t0 = time.time()
        for step, batch in enumerate(train_loader, start=1):
            batch = to_device(batch, device)
            try:
                with autocast_context(use_amp):
                    loss = model(**batch).loss
                    scaled_loss = loss / max(GRAD_ACCUM_STEPS, 1)
                scaler.scale(scaled_loss).backward()
            except torch.cuda.OutOfMemoryError as exc:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    "CUDA OOM during training. For this Kaggle GPU, first set BATCH_SIZE = 2 and "
                    "GRAD_ACCUM_STEPS = 8. If it still OOMs, set BATCH_SIZE = 1, "
                    "GRAD_ACCUM_STEPS = 16, and GRADIENT_CHECKPOINTING = True."
                ) from exc
            losses.append(float(loss.detach().cpu()))

            if step % GRAD_ACCUM_STEPS == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if step % LOG_EVERY_STEPS == 0 or step == len(train_loader):
                recent_loss = round(sum(losses[-LOG_EVERY_STEPS:]) / max(min(len(losses), LOG_EVERY_STEPS), 1), 4)
                running_loss = round(sum(losses) / max(len(losses), 1), 4)
                progress_row = {
                    "task": task_name,
                    "type": "train_step",
                    "epoch": epoch,
                    "step": step,
                    "total_steps": len(train_loader),
                    "recent_loss": recent_loss,
                    "running_loss": running_loss,
                    "seconds": round(time.time() - t0, 2),
                    "learning_rates": [round(group["lr"], 10) for group in optimizer.param_groups],
                }
                if step % SAVE_TEMP_PROGRESS_EVERY_STEPS == 0 or step == len(train_loader):
                    append_jsonl(step_progress_path, progress_row)
                    write_json(latest_progress_path, progress_row)
                print(
                    task_name,
                    "epoch",
                    epoch,
                    "step",
                    step,
                    "/",
                    len(train_loader),
                    "loss",
                    recent_loss,
                    flush=True,
                )

        train_loss = round(sum(losses) / max(len(losses), 1), 4)
        valid_loss = validate_loss(model, valid_loader, device, use_amp)
        should_validate_bleu = (
            VALID_GENERATION_LIMIT > 0
            and (epoch == EPOCHS or epoch % max(VALID_BLEU_EVERY_N_EPOCHS, 1) == 0)
        )
        valid_bleu = validate_bleu(model, tokenizer, source, target, valid_src, valid_tgt, device) if should_validate_bleu else None
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_bleu": valid_bleu,
            "valid_bleu_evaluated": should_validate_bleu,
            "seconds": round(time.time() - t0, 2),
        }
        history.append(row)
        write_json(history_path, history)
        append_jsonl(epoch_progress_path, {"task": task_name, "type": "epoch_summary", **row})
        write_json(latest_progress_path, {"task": task_name, "type": "epoch_summary", **row})
        print(task_name, json.dumps(row), flush=True)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_checkpoint(best_path, model, optimizer, scheduler, scaler, epoch, best_valid_loss, history)
            print(task_name, "saved best checkpoint", best_path, flush=True)
        save_checkpoint(latest_path, model, optimizer, scheduler, scaler, epoch, best_valid_loss, history)


def write_run_config(out_dir, task_name, source, target):
    write_json(
        out_dir / "config.json",
        {
            "task": task_name,
            "source": source,
            "target": target,
            "model": MODEL,
            "run_name": RUN_NAME,
            "data_dir": str(DATA_DIR),
            "output_dir": str(out_dir),
            "limit_train": LIMIT_TRAIN,
            "limit_valid": LIMIT_VALID,
            "limit_test": LIMIT_TEST,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "generate_batch_size": GENERATE_BATCH_SIZE,
            "max_source_length": MAX_SOURCE_LENGTH,
            "max_target_length": MAX_TARGET_LENGTH,
            "max_total_length": MAX_TOTAL_LENGTH,
            "valid_generation_limit": VALID_GENERATION_LIMIT,
            "valid_bleu_every_n_epochs": VALID_BLEU_EVERY_N_EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "lora_target_modules": LORA_TARGET_MODULES,
            "gradient_checkpointing": GRADIENT_CHECKPOINTING,
            "num_workers": NUM_WORKERS,
            "prefetch_factor": PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
            "seed": SEED,
            "save_temp_progress_every_steps": SAVE_TEMP_PROGRESS_EVERY_STEPS,
            "temp_progress_files": {
                "training_progress": str(out_dir / "tmp" / "training_progress.jsonl"),
                "latest_training_progress": str(out_dir / "tmp" / "latest_training_progress.json"),
                "epoch_summaries": str(out_dir / "tmp" / "epoch_summaries.jsonl"),
            },
        },
    )


def run_task(task_name, source, target, tokenizer, device, use_amp):
    out_dir = OUT / f"{RUN_NAME}_{task_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(out_dir, task_name, source, target)
    print(task_name, "output_dir", out_dir, flush=True)

    train_src, train_tgt = load_parallel(DATA_DIR, "train", source, target, LIMIT_TRAIN)
    valid_src, valid_tgt = load_parallel(DATA_DIR, "valid", source, target, LIMIT_VALID)
    test_src, _ = load_parallel(DATA_DIR, "test", source, target, LIMIT_TEST)
    print(
        task_name,
        "sizes",
        {"train": len(train_src), "valid": len(valid_src), "test": len(test_src)},
        flush=True,
    )

    model = build_model(device)
    if TRAIN and not EVAL_ONLY:
        train_model(task_name, source, target, model, tokenizer, train_src, train_tgt, valid_src, valid_tgt, out_dir, device, use_amp)

    if not GENERATE:
        return

    best_path = out_dir / "best.pt"
    latest_path = out_dir / "latest.pt"
    checkpoint_path = latest_path if latest_path.exists() else best_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{task_name} has no checkpoint to generate from: expected {best_path} or {latest_path}")
    load_checkpoint(checkpoint_path, model, device)
    model.eval()
    model.config.use_cache = True
    print(task_name, "loaded generation checkpoint", checkpoint_path, flush=True)

    total = len(test_src)
    start = START_INDEX if START_INDEX is not None else next_batch(out_dir, task_name, total)
    print(task_name, "generate start", start, "/", total, flush=True)
    while start < total:
        end = min(start + GENERATE_BATCH_SIZE, total)
        t0 = time.time()
        predictions = generate_texts(model, tokenizer, source, target, test_src[start:end], GENERATE_BATCH_SIZE, device)
        save_batch(out_dir, task_name, start, predictions)
        print(task_name, start, "-", end - 1, "sec", round(time.time() - t0, 3), flush=True)
        start = end if START_INDEX is not None else next_batch(out_dir, task_name, total)
    merge_batches(out_dir, task_name, total)


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"AVATAR data directory was not found: {DATA_DIR}")
    device, use_amp, gpu, capability, arch_list = select_device()
    set_seed(SEED)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print("torch", torch.__version__, "device", device, "gpu", gpu, "capability", capability, flush=True)
    print("torch_arch_list", arch_list, flush=True)
    print("data_dir", DATA_DIR, flush=True)
    print("out", OUT, flush=True)
    print(
        "run_config",
        {
            "epochs": EPOCHS,
            "limit_train": LIMIT_TRAIN,
            "limit_valid": LIMIT_VALID,
            "limit_test": LIMIT_TEST,
            "run_name": RUN_NAME,
            "batch_size": BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "generate_batch_size": GENERATE_BATCH_SIZE,
            "max_source_length": MAX_SOURCE_LENGTH,
            "max_target_length": MAX_TARGET_LENGTH,
            "max_total_length": MAX_TOTAL_LENGTH,
            "valid_generation_limit": VALID_GENERATION_LIMIT,
            "valid_bleu_every_n_epochs": VALID_BLEU_EVERY_N_EPOCHS,
            "lora_r": LORA_R,
            "lora_target_modules": LORA_TARGET_MODULES,
            "gradient_checkpointing": GRADIENT_CHECKPOINTING,
            "num_workers": NUM_WORKERS,
        },
        flush=True,
    )

    tokenizer = prepare_tokenizer()
    for task_name, source, target in TASKS:
        run_task(task_name, source, target, tokenizer, device, use_amp)
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
