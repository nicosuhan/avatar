import json
import math
import os
import random
import re
import time
from contextlib import nullcontext
from pathlib import Path

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["TRANSFORMERS_NO_TORCHAUDIO"] = "1"
os.environ["TRANSFORMERS_NO_VISION"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from torch.utils.data import DataLoader, Dataset
import transformers.utils.import_utils as hf_import_utils
from huggingface_hub import hf_hub_download

hf_import_utils._scipy_available = False
hf_import_utils._sklearn_available = False
hf_import_utils._torchvision_available = False
hf_import_utils._vision_available = False

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup


KAGGLE_DATA_DIRS = [
    Path("/kaggle/input/datasets/nicoletacorinasuhan/avatar-dataset/data/avatar"),
    Path("/kaggle/input/avatar-dataset/data/avatar"),
]
LOCAL_DATA_DIR = Path(__file__).resolve().parent / "data" / "avatar" if "__file__" in globals() else Path.cwd() / "avatar" / "data" / "avatar"
DATA_DIR = next((path for path in KAGGLE_DATA_DIRS if path.exists()), LOCAL_DATA_DIR)
OUT = Path("/kaggle/working/outputs") if Path("/kaggle/working").exists() else Path.cwd() / "outputs"

MODEL = "Salesforce/codet5p-220m"
RUN_NAME = "codet5p_220m_seq2seq"
TASKS = [("java2python", "java", "python"), ("python2java", "python", "java")]

TRAIN = True
GENERATE = True
LIMIT_TRAIN = 0
LIMIT_VALID = 0
LIMIT_TEST = 0

EPOCHS = 2
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
GENERATE_BATCH_SIZE = 8
MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 512
NUM_BEAMS = 4
SAVE_EPOCH_CHECKPOINTS = True
USE_AMP = True
CHECKPOINT_VERSION = 3
PREFER_FAST_TOKENIZER = True

LR = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
MAX_GRAD_NORM = 1.0
GRADIENT_CHECKPOINTING = False

SEED = 1234
NUM_WORKERS = 2
LOG_EVERY_STEPS = 100

class TokenizedTranslationDataset(Dataset):
    def __init__(self, source_ids, target_ids):
        self.source_ids = source_ids
        self.target_ids = target_ids

    def __len__(self):
        return len(self.source_ids)

    def __getitem__(self, index):
        return self.source_ids[index], self.target_ids[index]


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def get_pairs(word):
    pairs = set()
    previous = word[0]
    for char in word[1:]:
        pairs.add((previous, char))
        previous = char
    return pairs


class SimpleByteLevelBPETokenizer:
    def __init__(self, vocab_file, merges_file):
        self.encoder = json.loads(Path(vocab_file).read_text(encoding="utf-8"))
        self.decoder = {value: key for key, value in self.encoder.items()}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {value: key for key, value in self.byte_encoder.items()}
        merges = Path(merges_file).read_text(encoding="utf-8").splitlines()
        merges = [tuple(line.split()) for line in merges if line and not line.startswith("#")]
        self.bpe_ranks = {merge: index for index, merge in enumerate(merges)}
        self.cache = {}
        self.pat = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+")

        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        self.mask_token = "<mask>"
        self.bos_token_id = self.encoder[self.bos_token]
        self.eos_token_id = self.encoder[self.eos_token]
        self.unk_token_id = self.encoder[self.unk_token]
        self.pad_token_id = self.encoder[self.pad_token]
        self.mask_token_id = self.encoder[self.mask_token]
        self.all_special_ids = {
            self.bos_token_id,
            self.eos_token_id,
            self.unk_token_id,
            self.pad_token_id,
            self.mask_token_id,
            *[token_id for token, token_id in self.encoder.items() if token.startswith("<extra_id_")],
        }

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        if not word:
            return ""
        pairs = get_pairs(word)
        if not pairs:
            return token
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = get_pairs(word)
        word_text = " ".join(word)
        self.cache[token] = word_text
        return word_text

    def encode(self, text, max_length=None, truncation=False, add_special_tokens=True):
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            for bpe_token in self.bpe(token).split(" "):
                bpe_tokens.append(self.encoder.get(bpe_token, self.unk_token_id))
        if add_special_tokens:
            bpe_tokens = [self.bos_token_id, *bpe_tokens, self.eos_token_id]
        if truncation and max_length is not None and len(bpe_tokens) > max_length:
            if add_special_tokens and max_length >= 2:
                bpe_tokens = bpe_tokens[:max_length]
                bpe_tokens[-1] = self.eos_token_id
            else:
                bpe_tokens = bpe_tokens[:max_length]
        return bpe_tokens

    def __call__(self, texts, padding=False, truncation=False, max_length=None, return_tensors=None):
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        input_ids = [self.encode(text, max_length=max_length, truncation=truncation) for text in texts]
        attention_mask = [[1] * len(ids) for ids in input_ids]
        if padding:
            pad_to = max(len(ids) for ids in input_ids) if input_ids else 0
            if max_length is not None:
                pad_to = min(max_length, pad_to)
            padded_ids = []
            padded_masks = []
            for ids, mask in zip(input_ids, attention_mask):
                ids = ids[:pad_to]
                mask = mask[:pad_to]
                pad_len = pad_to - len(ids)
                padded_ids.append(ids + [self.pad_token_id] * pad_len)
                padded_masks.append(mask + [0] * pad_len)
            input_ids = padded_ids
            attention_mask = padded_masks
        result = {"input_ids": input_ids, "attention_mask": attention_mask}
        if return_tensors == "pt":
            result = {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}
        if single:
            return {key: value[0] if return_tensors != "pt" else value for key, value in result.items()}
        return result

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        tokens = []
        for token_id in ids:
            token_id = int(token_id)
            if skip_special_tokens and token_id in self.all_special_ids:
                continue
            token = self.decoder.get(token_id, self.unk_token)
            if skip_special_tokens and token.startswith("<extra_id_"):
                continue
            tokens.append(token)
        text = "".join(tokens)
        byte_array = bytearray()
        for char in text:
            if char in self.byte_decoder:
                byte_array.append(self.byte_decoder[char])
        return byte_array.decode("utf-8", errors="replace")

    def batch_decode(self, sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return [
            self.decode(
                sequence,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )
            for sequence in sequences
        ]

    def __len__(self):
        return len(self.encoder)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Enable a Kaggle GPU before running this script.")
    return torch.device("cuda")


def read_lines(path):
    return Path(path).read_text(encoding="utf-8").splitlines()


def load_parallel(split, source, target, limit):
    src = read_lines(DATA_DIR / f"{split}.java-python.{source}")
    tgt = read_lines(DATA_DIR / f"{split}.java-python.{target}")
    if limit:
        src = src[:limit]
        tgt = tgt[:limit]
    return src, tgt


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def normalize_prediction(text):
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", text).strip()


def non_pad_length(tokenizer, text, max_length):
    encoded = tokenizer(
        text,
        padding=False,
        truncation=True,
        max_length=max_length,
    )
    return sum(1 for token_id in encoded["input_ids"] if token_id != tokenizer.pad_token_id)


def tokenizer_has_content(tokenizer, texts, max_length):
    lengths = [non_pad_length(tokenizer, text, max_length) for text in texts if text and text.strip()]
    return bool(lengths) and max(lengths) > 2, lengths


def validate_tokenizer_or_raise(tokenizer, name):
    samples = [
        "public class Main { public static void main ( String [ ] args ) { System . out . println ( 1 ) ; } }",
        "print ( 1 ) NEW_LINE",
    ]
    ok, lengths = tokenizer_has_content(tokenizer, samples, MAX_TARGET_LENGTH)
    if not ok:
        raise RuntimeError(f"{name} tokenizer is broken: sample token lengths are {lengths}")
    print(name, "tokenizer sample lengths", lengths, flush=True)


def checked_loss_value(loss, where):
    value = float(loss.detach().cpu())
    if not math.isfinite(value):
        raise RuntimeError(f"{where} produced non-finite loss {value}")
    return value


def archive_prediction_file(path, reason):
    if not path.exists():
        return
    stamp = time.strftime("%Y%m%d-%H%M%S")
    archive = path.with_name(f"{path.name}.{reason}.{stamp}.bak")
    path.replace(archive)
    print("archived bad prediction file", path, "->", archive, flush=True)


def archive_checkpoint_file(path, reason):
    if not path.exists():
        return
    safe_reason = re.sub(r"[^A-Za-z0-9_.-]+", "-", reason).strip("-") or "bad"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    archive = path.with_name(f"{path.name}.{safe_reason}.{stamp}.bak")
    path.replace(archive)
    print("archived bad checkpoint", path, "->", archive, flush=True)


def read_existing_predictions(path, expected_count):
    if not path.exists():
        return []
    predictions = read_lines(path)
    if len(predictions) > expected_count:
        raise RuntimeError(f"{path} has {len(predictions)} lines but only {expected_count} test examples.")
    empty = sum(1 for prediction in predictions if not prediction.strip())
    if empty:
        archive_prediction_file(path, f"{empty}-empty-lines")
        return []
    return predictions


def tokenize_texts(tokenizer, texts, max_length, label):
    started = time.time()
    tokenized = []
    for index, text in enumerate(texts, start=1):
        tokenized.append(tokenizer.encode(text, max_length=max_length, truncation=True))
        if index % 10000 == 0:
            print(label, "tokenized", index, "/", len(texts), flush=True)
    lengths = [len(ids) for ids in tokenized]
    average = round(sum(lengths) / max(len(lengths), 1), 2)
    print(
        label,
        "tokenized",
        len(tokenized),
        "examples",
        "avg_len",
        average,
        "max_len",
        max(lengths, default=0),
        "sec",
        round(time.time() - started, 2),
        flush=True,
    )
    return tokenized


def pad_id_lists(sequences, pad_value):
    max_len = max(len(sequence) for sequence in sequences)
    return torch.tensor(
        [sequence + [pad_value] * (max_len - len(sequence)) for sequence in sequences],
        dtype=torch.long,
    )


def make_collate_fn(tokenizer):
    pad_id = tokenizer.pad_token_id

    def collate(batch):
        source_ids, target_ids = zip(*batch)
        input_ids = pad_id_lists(source_ids, pad_id)
        labels = pad_id_lists(target_ids, -100)
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(pad_id).long(),
            "labels": labels,
        }

    return collate


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def autocast_context():
    if USE_AMP:
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def load_tokenizer():
    if PREFER_FAST_TOKENIZER:
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
            validate_tokenizer_or_raise(tokenizer, "AutoTokenizer fast")
            print("tokenizer AutoTokenizer fast", flush=True)
            return tokenizer
        except Exception as exc:
            print("fast tokenizer unavailable, using SimpleByteLevelBPETokenizer:", repr(exc), flush=True)

    tokenizer = SimpleByteLevelBPETokenizer(
        vocab_file=hf_hub_download(MODEL, "vocab.json"),
        merges_file=hf_hub_download(MODEL, "merges.txt"),
    )
    validate_tokenizer_or_raise(tokenizer, "SimpleByteLevelBPETokenizer")
    print("tokenizer SimpleByteLevelBPETokenizer vocab.json + merges.txt", flush=True)
    return tokenizer


def validate_training_texts(task_name, tokenizer, train_src, train_tgt, valid_src, valid_tgt):
    checks = [
        ("train source", train_src[:16], MAX_SOURCE_LENGTH),
        ("train target", train_tgt[:16], MAX_TARGET_LENGTH),
        ("valid source", valid_src[:16], MAX_SOURCE_LENGTH),
        ("valid target", valid_tgt[:16], MAX_TARGET_LENGTH),
    ]
    for label, texts, max_length in checks:
        ok, lengths = tokenizer_has_content(tokenizer, texts, max_length)
        print(task_name, label, "tokenized lengths", lengths[:8], flush=True)
        if not ok:
            preview = [text[:120] for text in texts[:3]]
            raise RuntimeError(
                f"{task_name} {label} tokenization is broken: lengths={lengths[:8]} preview={preview}"
            )


def load_model_and_tokenizer(device):
    tokenizer = load_tokenizer()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).float()
    if int(model.config.vocab_size) != len(tokenizer):
        raise RuntimeError(f"model vocab size {model.config.vocab_size} != tokenizer vocab size {len(tokenizer)}")
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = tokenizer.pad_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if model.config.eos_token_id is None:
        model.config.eos_token_id = tokenizer.eos_token_id
    if GRADIENT_CHECKPOINTING and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model = model.to(device)
    print("model parameter dtype", next(model.parameters()).dtype, flush=True)
    return model, tokenizer


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_valid_loss, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "best_valid_loss": best_valid_loss,
            "history": history,
            "checkpoint_version": CHECKPOINT_VERSION,
        },
        path,
    )


def checkpoint_history_is_sane(checkpoint):
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        return False, "old checkpoint version"
    history = checkpoint.get("history") or []
    if not history:
        return False, "missing history"
    last = history[-1]
    for key in ("train_loss", "valid_loss"):
        try:
            value = float(last[key])
        except Exception:
            return False, f"missing {key}"
        if not math.isfinite(value) or value <= 0.0:
            return False, f"{key}={value}"
    return True, "ok"


def load_checkpoint(path, model, device, optimizer=None, scheduler=None, scaler=None):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer_loaded = False
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
            optimizer_loaded = True
        except Exception as exc:
            print("warning: optimizer state does not match current script; resetting optimizer:", exc, flush=True)
    if scheduler is not None and checkpoint.get("scheduler") is not None and optimizer_loaded:
        try:
            scheduler.load_state_dict(checkpoint["scheduler"])
        except Exception as exc:
            print("warning: scheduler state does not match current script; resetting scheduler:", exc, flush=True)
    elif scheduler is not None and checkpoint.get("scheduler") is not None and not optimizer_loaded:
        print("warning: scheduler reset because optimizer state was reset", flush=True)
    if scaler is not None and checkpoint.get("scaler") is not None:
        try:
            scaler.load_state_dict(checkpoint["scaler"])
        except Exception as exc:
            print("warning: scaler state does not match current script; resetting scaler:", exc, flush=True)
    return checkpoint


def find_resume_checkpoint(out_dir):
    candidates = [out_dir / "latest.pt", out_dir / "best.pt"]
    candidates.extend(sorted(out_dir.glob("epoch_*.pt"), reverse=True))
    seen = set()
    for path in candidates:
        if not path.exists() or path in seen:
            continue
        seen.add(path)
        checkpoint = torch.load(path, map_location="cpu")
        sane, reason = checkpoint_history_is_sane(checkpoint)
        del checkpoint
        if sane:
            return path
        archive_checkpoint_file(path, reason)
    return None


def validate_loss(model, data_loader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for step, batch in enumerate(data_loader, start=1):
            batch = to_device(batch, device)
            with autocast_context():
                loss = model(**batch).loss
            losses.append(checked_loss_value(loss, f"validation step {step}"))
    if not losses:
        raise RuntimeError("validation loader is empty")
    valid_loss = sum(losses) / len(losses)
    if not math.isfinite(valid_loss) or valid_loss <= 0.0:
        raise RuntimeError(f"validation average loss is invalid: {valid_loss}")
    return round(valid_loss, 6)


def train_model(task_name, model, tokenizer, train_src, train_tgt, valid_src, valid_tgt, out_dir, device):
    validate_training_texts(task_name, tokenizer, train_src, train_tgt, valid_src, valid_tgt)
    train_source_ids = tokenize_texts(tokenizer, train_src, MAX_SOURCE_LENGTH, f"{task_name} train source")
    train_target_ids = tokenize_texts(tokenizer, train_tgt, MAX_TARGET_LENGTH, f"{task_name} train target")
    valid_source_ids = tokenize_texts(tokenizer, valid_src, MAX_SOURCE_LENGTH, f"{task_name} valid source")
    valid_target_ids = tokenize_texts(tokenizer, valid_tgt, MAX_TARGET_LENGTH, f"{task_name} valid target")
    collate_fn = make_collate_fn(tokenizer)
    train_loader = DataLoader(
        TokenizedTranslationDataset(train_source_ids, train_target_ids),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        TokenizedTranslationDataset(valid_source_ids, valid_target_ids),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    updates_per_epoch = max(math.ceil(len(train_loader) / GRAD_ACCUM_STEPS), 1)
    total_updates = updates_per_epoch * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * WARMUP_RATIO),
        num_training_steps=total_updates,
    )
    scaler = torch.amp.GradScaler("cuda") if USE_AMP else None

    latest_path = out_dir / "latest.pt"
    best_path = out_dir / "best.pt"
    history_path = out_dir / "history.json"
    start_epoch = 0
    best_valid_loss = float("inf")
    history = []
    resume_path = find_resume_checkpoint(out_dir)
    if resume_path is not None:
        checkpoint = load_checkpoint(resume_path, model, device, optimizer, scheduler, scaler)
        start_epoch = int(checkpoint.get("epoch", 0))
        best_valid_loss = float(checkpoint.get("best_valid_loss", float("inf")))
        history = list(checkpoint.get("history", []))
        print(task_name, "restored", resume_path, "start_epoch", start_epoch, flush=True)

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        started = time.time()
        for step, batch in enumerate(train_loader, start=1):
            batch = to_device(batch, device)
            try:
                with autocast_context():
                    loss = model(**batch).loss
                    loss_value = checked_loss_value(loss, f"{task_name} epoch {epoch} train step {step}")
                    scaled_loss = loss / GRAD_ACCUM_STEPS
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                raise RuntimeError(
                    "CUDA OOM. Set BATCH_SIZE=2 and GRAD_ACCUM_STEPS=8. "
                    "If it still OOMs, set BATCH_SIZE=1 and GRAD_ACCUM_STEPS=16."
                ) from exc

            losses.append(loss_value)
            if step % GRAD_ACCUM_STEPS == 0 or step == len(train_loader):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if step % LOG_EVERY_STEPS == 0 or step == len(train_loader):
                print(task_name, "epoch", epoch, "step", step, "/", len(train_loader), "loss", round(sum(losses) / len(losses), 6), flush=True)

        if not losses:
            raise RuntimeError(f"{task_name} epoch {epoch} had no training batches")
        train_loss_raw = sum(losses) / len(losses)
        if not math.isfinite(train_loss_raw) or train_loss_raw <= 0.0:
            raise RuntimeError(f"{task_name} epoch {epoch} average train loss is invalid: {train_loss_raw}")
        train_loss = round(train_loss_raw, 6)
        valid_loss = validate_loss(model, valid_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss, "seconds": round(time.time() - started, 2)}
        history.append(row)
        write_json(history_path, history)
        print(task_name, json.dumps(row), flush=True)

        if SAVE_EPOCH_CHECKPOINTS:
            save_checkpoint(out_dir / f"epoch_{epoch}.pt", model, optimizer, scheduler, scaler, epoch, best_valid_loss, history)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_checkpoint(best_path, model, optimizer, scheduler, scaler, epoch, best_valid_loss, history)
        save_checkpoint(latest_path, model, optimizer, scheduler, scaler, epoch, best_valid_loss, history)


def load_generation_checkpoint(out_dir, model):
    candidates = [out_dir / "latest.pt", out_dir / "best.pt"]
    candidates.extend(sorted(out_dir.glob("epoch_*.pt"), reverse=True))
    seen = set()
    for path in candidates:
        if not path.exists() or path in seen:
            continue
        seen.add(path)
        checkpoint = torch.load(path, map_location="cpu")
        sane, reason = checkpoint_history_is_sane(checkpoint)
        if not sane:
            print("skip generation checkpoint", path, reason, flush=True)
            continue
        model.load_state_dict(checkpoint["model"])
        print("loaded generation checkpoint", path, "epoch", checkpoint.get("epoch"), flush=True)
        return path, checkpoint
    raise RuntimeError(
        f"No sane checkpoint found in {out_dir}. If epoch 2 reported zero loss, remove the broken "
        "CodeT5+ output folder and retrain; the previous checkpoint was probably overwritten."
    )


@torch.no_grad()
def generate_batch(model, tokenizer, source_ids, device):
    input_ids = pad_id_lists(source_ids, tokenizer.pad_token_id)
    inputs = {
        "input_ids": input_ids.to(device),
        "attention_mask": input_ids.ne(tokenizer.pad_token_id).long().to(device),
    }
    output_ids = model.generate(
        **inputs,
        max_length=MAX_TARGET_LENGTH,
        num_beams=NUM_BEAMS,
        do_sample=False,
        early_stopping=True,
        decoder_start_token_id=model.config.decoder_start_token_id,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return [normalize_prediction(text) for text in texts]


def generate_predictions(task_name, model, tokenizer, test_src, out_dir, device):
    pred_path = out_dir / "test.pred"
    existing = read_existing_predictions(pred_path, len(test_src))
    start = len(existing)
    if start == 0:
        pred_path.write_text("", encoding="utf-8")
    print(task_name, "generate start", start, "/", len(test_src), "output", pred_path, flush=True)

    model.eval()
    test_source_ids = tokenize_texts(tokenizer, test_src, MAX_SOURCE_LENGTH, f"{task_name} test source")
    for index in range(start, len(test_src), GENERATE_BATCH_SIZE):
        end = min(index + GENERATE_BATCH_SIZE, len(test_src))
        started = time.time()
        try:
            predictions = generate_batch(model, tokenizer, test_source_ids[index:end], device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            predictions = []
            for source_ids in test_source_ids[index:end]:
                predictions.extend(generate_batch(model, tokenizer, [source_ids], device))
        empty = [offset for offset, prediction in enumerate(predictions) if not prediction.strip()]
        if empty:
            absolute = [index + offset for offset in empty]
            raise RuntimeError(
                f"{task_name} generated empty predictions at test indexes {absolute}. "
                "Generation stopped before writing this bad batch."
            )
        with open(pred_path, "a", encoding="utf-8") as f:
            f.write("\n".join(predictions) + "\n")
        print(task_name, index, "-", end - 1, "sec", round(time.time() - started, 3), flush=True)


def write_config(out_dir, task_name, source, target):
    write_json(
        out_dir / "config.json",
        {
            "task": task_name,
            "source": source,
            "target": target,
            "model": MODEL,
            "tokenizer": "AutoTokenizer fast if validation passes, otherwise SimpleByteLevelBPETokenizer",
            "prefer_fast_tokenizer": PREFER_FAST_TOKENIZER,
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
            "num_beams": NUM_BEAMS,
            "save_epoch_checkpoints": SAVE_EPOCH_CHECKPOINTS,
            "use_amp": USE_AMP,
            "checkpoint_version": CHECKPOINT_VERSION,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "gradient_checkpointing": GRADIENT_CHECKPOINTING,
            "seed": SEED,
        },
    )


def run_task(task_name, source, target, device):
    out_dir = OUT / f"{RUN_NAME}_{task_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_config(out_dir, task_name, source, target)

    train_src, train_tgt = load_parallel("train", source, target, LIMIT_TRAIN)
    valid_src, valid_tgt = load_parallel("valid", source, target, LIMIT_VALID)
    test_src, _ = load_parallel("test", source, target, LIMIT_TEST)
    print(task_name, {"train": len(train_src), "valid": len(valid_src), "test": len(test_src), "out": str(out_dir)}, flush=True)

    model, tokenizer = load_model_and_tokenizer(device)
    if TRAIN:
        train_model(task_name, model, tokenizer, train_src, train_tgt, valid_src, valid_tgt, out_dir, device)

    if GENERATE:
        checkpoint_path, _ = load_generation_checkpoint(out_dir, model)
        print(task_name, "using checkpoint for generation", checkpoint_path, flush=True)
        generate_predictions(task_name, model, tokenizer, test_src, out_dir, device)


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"AVATAR data directory was not found: {DATA_DIR}")
    device = select_device()
    set_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(
        "torch",
        torch.__version__,
        "gpu",
        torch.cuda.get_device_name(0),
        "capability",
        torch.cuda.get_device_capability(0),
        flush=True,
    )
    print("data_dir", DATA_DIR, "out", OUT, flush=True)
    for task_name, source, target in TASKS:
        run_task(task_name, source, target, device)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
