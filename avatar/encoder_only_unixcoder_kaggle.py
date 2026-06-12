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
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import transformers.utils.import_utils as hf_import_utils

hf_import_utils._tf_available = False
hf_import_utils._flax_available = False
hf_import_utils._jax_available = False
hf_import_utils._torchvision_available = False
hf_import_utils._sklearn_available = False
hf_import_utils._scipy_available = False
hf_import_utils._torchao_available = False


KAGGLE_DATA_DIRS = [
    Path("/kaggle/input/datasets/nicoletacorinasuhan/avatar-dataset/data/avatar"),
    Path("/kaggle/input/avatar-dataset/data/avatar"),
]
LOCAL_DATA_DIR = (Path(__file__).resolve().parent / "data" / "avatar") if "__file__" in globals() else (Path.cwd() / "avatar" / "data" / "avatar")
IS_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None or (os.name != "nt" and Path("/kaggle/working").exists())
DATA_DIR = next((path for path in KAGGLE_DATA_DIRS if path.exists()), LOCAL_DATA_DIR)
OUT = Path("/kaggle/working/outputs") if IS_KAGGLE else (Path.cwd() / "outputs")

MODEL = "microsoft/unixcoder-base"
TASKS = [("python2java", "python", "java"), ("java2python", "java", "python")]

LIMIT_TRAIN = 0
LIMIT_VALID = 0
LIMIT_TEST = 0

TRAIN = True
EVAL_ONLY = False
GENERATE = True
START_INDEX = None
ALLOW_CPU_FALLBACK = False

EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
GENERATE_BATCH_SIZE = 8
MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 512
VALID_GENERATION_LIMIT = 64

ENCODER_LR = 2e-5
DECODER_LR = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
LABEL_SMOOTHING = 0.1
MAX_GRAD_NORM = 1.0

DECODER_LAYERS = 4
DECODER_DROPOUT = 0.1
FREEZE_ENCODER = False
GRADIENT_CHECKPOINTING = True

SEED = 1234
NUM_WORKERS = 0
LOG_EVERY_STEPS = 100
SAVE_TEMP_PROGRESS_EVERY_STEPS = 100

P100_TORCH_INSTALL_HINT = """
This Kaggle runtime has a CUDA GPU, but the installed PyTorch wheel does not include kernels for it.
For Tesla P100, either switch the Kaggle accelerator to T4/L4/A100, or run this in a separate first cell and restart the session:

!pip uninstall -y torch torchvision torchaudio
!pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

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


class TranslationDataset(Dataset):
    def __init__(self, sources, targets):
        self.sources = sources
        self.targets = targets

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, index):
        return self.sources[index], self.targets[index]


def encode_target(tokenizer, text):
    ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max(MAX_TARGET_LENGTH - 2, 1),
    )["input_ids"]
    return torch.tensor([tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id], dtype=torch.long)


def make_collate_fn(tokenizer):
    pad_id = tokenizer.pad_token_id

    def collate(batch):
        sources, targets = zip(*batch)
        source_batch = tokenizer(
            list(sources),
            padding=True,
            truncation=True,
            max_length=MAX_SOURCE_LENGTH,
            return_tensors="pt",
        )
        target_ids = [encode_target(tokenizer, target) for target in targets]
        padded = pad_sequence(target_ids, batch_first=True, padding_value=pad_id)
        decoder_input_ids = padded[:, :-1].contiguous()
        labels = padded[:, 1:].contiguous()
        labels = labels.masked_fill(labels.eq(pad_id), -100)
        return {
            "source_input_ids": source_batch["input_ids"],
            "source_attention_mask": source_batch["attention_mask"],
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
        }

    return collate


class EncoderOnlyTranslator(nn.Module):
    def __init__(self, model_name, tokenizer):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.hidden_size = self.encoder.config.hidden_size
        self.num_heads = self.encoder.config.num_attention_heads
        self.max_target_length = MAX_TARGET_LENGTH + 1

        if GRADIENT_CHECKPOINTING and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        if FREEZE_ENCODER:
            for param in self.encoder.parameters():
                param.requires_grad = False

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=self.encoder.config.intermediate_size,
            dropout=DECODER_DROPOUT,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=DECODER_LAYERS)
        self.target_positions = nn.Embedding(MAX_TARGET_LENGTH + 1, self.hidden_size)
        self.output_bias = nn.Parameter(torch.zeros(self.encoder.config.vocab_size))

    def target_embeddings(self, decoder_input_ids):
        seq_len = decoder_input_ids.shape[1]
        if seq_len > self.max_target_length:
            raise ValueError(f"decoder sequence length {seq_len} exceeds MAX_TARGET_LENGTH={self.max_target_length}")
        position_ids = torch.arange(seq_len, device=decoder_input_ids.device).unsqueeze(0)
        token_embeddings = self.encoder.get_input_embeddings()(decoder_input_ids)
        return token_embeddings + self.target_positions(position_ids)

    @staticmethod
    def causal_mask(size, device):
        return torch.triu(torch.full((size, size), float("-inf"), device=device), diagonal=1)

    def decode_logits(self, decoder_input_ids, encoder_outputs, source_attention_mask):
        target_hidden = self.target_embeddings(decoder_input_ids)
        target_key_padding_mask = decoder_input_ids.eq(self.pad_token_id)
        memory_key_padding_mask = source_attention_mask.eq(0)
        decoded = self.decoder(
            tgt=target_hidden,
            memory=encoder_outputs,
            tgt_mask=self.causal_mask(decoder_input_ids.shape[1], decoder_input_ids.device),
            tgt_key_padding_mask=target_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        embedding_weight = self.encoder.get_input_embeddings().weight
        return F.linear(decoded, embedding_weight, self.output_bias)

    def forward(self, source_input_ids, source_attention_mask, decoder_input_ids, labels=None):
        encoder_outputs = self.encoder(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask,
            return_dict=True,
        ).last_hidden_state
        logits = self.decode_logits(decoder_input_ids, encoder_outputs, source_attention_mask)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                labels.reshape(-1),
                ignore_index=-100,
                label_smoothing=LABEL_SMOOTHING,
            )
        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def greedy_decode(self, source_input_ids, source_attention_mask):
        encoder_outputs = self.encoder(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask,
            return_dict=True,
        ).last_hidden_state
        batch_size = source_input_ids.shape[0]
        tokens = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=source_input_ids.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=source_input_ids.device)
        for _ in range(MAX_TARGET_LENGTH):
            logits = self.decode_logits(tokens, encoder_outputs, source_attention_mask)
            next_token = logits[:, -1, :].argmax(dim=-1)
            next_token = torch.where(finished, torch.full_like(next_token, self.eos_token_id), next_token)
            tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)
            finished = finished | next_token.eq(self.eos_token_id)
            if bool(finished.all()):
                break
        return tokens[:, 1:]


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def autocast_context(enabled):
    if enabled:
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def build_optimizer(model):
    encoder_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_params.append(param)
        else:
            decoder_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": ENCODER_LR, "weight_decay": WEIGHT_DECAY},
            {"params": decoder_params, "lr": DECODER_LR, "weight_decay": WEIGHT_DECAY},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_valid_loss, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
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
    model.load_state_dict(checkpoint["model"])
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


def generate_texts(model, tokenizer, sources, batch_size, device, use_amp):
    predictions = []
    model.eval()
    for start in range(0, len(sources), batch_size):
        batch_sources = sources[start : start + batch_size]
        source_batch = tokenizer(
            batch_sources,
            padding=True,
            truncation=True,
            max_length=MAX_SOURCE_LENGTH,
            return_tensors="pt",
        )
        source_input_ids = source_batch["input_ids"].to(device)
        source_attention_mask = source_batch["attention_mask"].to(device)
        with autocast_context(use_amp):
            output_ids = model.greedy_decode(source_input_ids, source_attention_mask)
        texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        predictions.extend(normalize_prediction(text) for text in texts)
    return predictions


def validate_loss(model, data_loader, device, use_amp):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in data_loader:
            batch = to_device(batch, device)
            with autocast_context(use_amp):
                loss = model(**batch)["loss"]
            losses.append(float(loss.detach().cpu()))
    return round(sum(losses) / max(len(losses), 1), 4)


def validate_bleu(model, tokenizer, valid_src, valid_tgt, device, use_amp):
    limit = min(VALID_GENERATION_LIMIT, len(valid_src))
    if limit <= 0:
        return 0.0
    predictions = generate_texts(model, tokenizer, valid_src[:limit], GENERATE_BATCH_SIZE, device, use_amp)
    refs = [[target.split()] for target in valid_tgt[:limit]]
    hyps = [prediction.split() for prediction in predictions]
    return round(corpus_bleu(refs, hyps), 2)


def train_model(task_name, model, tokenizer, train_src, train_tgt, valid_src, valid_tgt, out_dir, device, use_amp):
    collate_fn = make_collate_fn(tokenizer)
    train_loader = DataLoader(
        TranslationDataset(train_src, train_tgt),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        TranslationDataset(valid_src, valid_tgt),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )
    optimizer = build_optimizer(model)
    updates_per_epoch = max(math.ceil(len(train_loader) / max(GRAD_ACCUM_STEPS, 1)), 1)
    total_updates = updates_per_epoch * max(EPOCHS, 1)
    warmup_steps = int(total_updates * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_updates)
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

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        t0 = time.time()
        for step, batch in enumerate(train_loader, start=1):
            batch = to_device(batch, device)
            with autocast_context(use_amp):
                loss = model(**batch)["loss"]
                scaled_loss = loss / max(GRAD_ACCUM_STEPS, 1)
            scaler.scale(scaled_loss).backward()
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
        valid_bleu = validate_bleu(model, tokenizer, valid_src, valid_tgt, device, use_amp)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_bleu": valid_bleu,
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
            "encoder_lr": ENCODER_LR,
            "decoder_lr": DECODER_LR,
            "decoder_layers": DECODER_LAYERS,
            "freeze_encoder": FREEZE_ENCODER,
            "gradient_checkpointing": GRADIENT_CHECKPOINTING,
            "seed": SEED,
            "save_temp_progress_every_steps": SAVE_TEMP_PROGRESS_EVERY_STEPS,
            "temp_progress_files": {
                "training_progress": str(out_dir / "tmp" / "training_progress.jsonl"),
                "latest_training_progress": str(out_dir / "tmp" / "latest_training_progress.json"),
                "epoch_summaries": str(out_dir / "tmp" / "epoch_summaries.jsonl"),
            },
        },
    )


def prepare_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token = tokenizer.cls_token or tokenizer.eos_token
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = tokenizer.sep_token or tokenizer.bos_token
    return tokenizer


def run_task(task_name, source, target, tokenizer, device, use_amp):
    out_dir = OUT / f"unixcoder_encoder_only_{task_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(out_dir, task_name, source, target)
    print(task_name, "output_dir", out_dir, flush=True)

    train_src, train_tgt = load_parallel(DATA_DIR, "train", source, target, LIMIT_TRAIN)
    valid_src, valid_tgt = load_parallel(DATA_DIR, "valid", source, target, LIMIT_VALID)
    test_src, _ = load_parallel(DATA_DIR, "test", source, target, LIMIT_TEST)

    model = EncoderOnlyTranslator(MODEL, tokenizer).to(device)
    if TRAIN and not EVAL_ONLY:
        train_model(task_name, model, tokenizer, train_src, train_tgt, valid_src, valid_tgt, out_dir, device, use_amp)

    if not GENERATE:
        return

    best_path = out_dir / "best.pt"
    latest_path = out_dir / "latest.pt"
    checkpoint_path = best_path if best_path.exists() else latest_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{task_name} has no checkpoint to generate from: expected {best_path} or {latest_path}")
    load_checkpoint(checkpoint_path, model, device)
    print(task_name, "loaded generation checkpoint", checkpoint_path, flush=True)

    total = len(test_src)
    start = START_INDEX if START_INDEX is not None else next_batch(out_dir, task_name, total)
    print(task_name, "generate start", start, "/", total, flush=True)
    while start < total:
        end = min(start + GENERATE_BATCH_SIZE, total)
        t0 = time.time()
        predictions = generate_texts(model, tokenizer, test_src[start:end], GENERATE_BATCH_SIZE, device, use_amp)
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

    tokenizer = prepare_tokenizer()
    for task_name, source, target in TASKS:
        run_task(task_name, source, target, tokenizer, device, use_amp)
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
