from constants import BASE_TRAIN_CONFIG, OUTPUT_ROOT, TRANSFORMER_MODEL
from transformer import Transformer, TransformerConfig

import json
import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from keras import config as keras_config
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm


SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if tf.is_tensor(value):
        value = value.numpy()
        return value.item() if np.ndim(value) == 0 else value.tolist()
    return value


def build_args():
    config = dict(BASE_TRAIN_CONFIG)
    config["runtime"] = dict(config["runtime"])
    config.update(TRANSFORMER_MODEL)
    config["runtime"]["mixed_precision"] = True
    config["runtime"]["flash_attention"] = True
    config["bucket_boundaries"] = [64, 128, 256, 384]
    config["bucket_batch_sizes"] = [32, 24, 16, 12, config["batch_size"]]
    config["eval_batch_size"] = 16
    config["validate_every"] = 2
    config["save_every"] = 2
    config["output_dir"] = str(OUTPUT_ROOT / f"{config['model']}_{config['source']}2{config['target']}")
    return SimpleNamespace(**config)


def configure_runtime(runtime):
    runtime = runtime or {}
    set_seed(runtime.get("seed", 1234))
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        if runtime.get("require_gpu", False):
            raise RuntimeError(f'GPU required but not found. Expected: {runtime.get("expected_gpu", "GPU")}')
        print(json.dumps(to_jsonable({"device": "cpu", "expected_gpu": runtime.get("expected_gpu", "GPU")})))
        return False
    gpu_index = min(max(int(runtime.get("gpu_index", 0)), 0), len(gpus) - 1)
    gpu = gpus[gpu_index]
    try:
        tf.config.set_visible_devices(gpu, "GPU")
        if runtime.get("memory_growth", True):
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass
    if runtime.get("flash_attention", False):
        keras_config.enable_flash_attention()
    if runtime.get("mixed_precision", False):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        details = tf.config.experimental.get_device_details(gpu)
        device_name = details.get("device_name", gpu.name)
    except Exception:
        device_name = gpu.name
    print(json.dumps(to_jsonable({"device": device_name, "expected_gpu": runtime.get("expected_gpu", "GPU"), "mixed_precision": runtime.get("mixed_precision", False)})))
    return True


def load_parallel(data_dir, split, source, target, limit=0):
    src = read_lines(Path(data_dir) / f"{split}.java-python.{source}")
    tgt = read_lines(Path(data_dir) / f"{split}.java-python.{target}")
    ids = read_lines(Path(data_dir) / f"{split}.java-python.id")
    if limit:
        src = src[:limit]
        tgt = tgt[:limit]
        ids = ids[:limit]
    return src, tgt, ids


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def load_reference_sets(ids, jsonl_path, lang):
    ref_map = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ref_map[row["id"]] = row[lang]
    return [ref_map[i] for i in ids]


def _get_ngrams(segment, max_order):
    counts = Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            counts[tuple(segment[i:i + order])] += 1
    return counts


def corpus_bleu(reference_corpus, translation_corpus, max_order=4, smooth=True):
    matches = [0] * max_order
    possible = [0] * max_order
    ref_len = 0
    hyp_len = 0
    for refs, hyp in zip(reference_corpus, translation_corpus):
        ref_len += min(len(r) for r in refs)
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
    geo_mean = tf.math.exp(sum(tf.math.log(p) / max_order for p in precisions)).numpy() if min(precisions) > 0 else 0.0
    ratio = hyp_len / max(ref_len, 1)
    bp = 1.0 if ratio > 1.0 else np.exp(1.0 - 1.0 / max(ratio, 1e-9))
    return float(100.0 * geo_mean * bp)


def build_vocab(texts, max_vocab_size):
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    vocab = SPECIAL_TOKENS + [token for token, _ in counter.most_common(max_vocab_size - len(SPECIAL_TOKENS))]
    token_to_id = {token: i for i, token in enumerate(vocab)}
    return vocab, token_to_id


def encode_text(text, token_to_id, max_length, add_bos=False, add_eos=False):
    pieces = text.split()
    reserve = int(add_bos) + int(add_eos)
    if max_length:
        pieces = pieces[: max(max_length - reserve, 1)]
    ids = [token_to_id.get(piece, token_to_id["<unk>"]) for piece in pieces]
    if add_bos:
        ids = [token_to_id["<s>"]] + ids
    if add_eos:
        ids = ids + [token_to_id["</s>"]]
    return ids


def decode_ids(ids, vocab, eos_id, pad_id):
    pieces = []
    for idx in ids:
        idx = int(idx)
        if idx in {eos_id, pad_id}:
            break
        if idx == 1:
            continue
        pieces.append(vocab[idx])
    return " ".join(pieces)


def build_dataset(
    src_texts,
    tgt_texts,
    token_to_id,
    max_source_length,
    max_target_length,
    batch_size,
    shuffle,
    bucket_boundaries=None,
    bucket_batch_sizes=None,
):
    pad_id = token_to_id["<pad>"]

    def generator():
        for src, tgt in zip(src_texts, tgt_texts):
            yield (
                np.asarray(encode_text(src, token_to_id, max_source_length, add_eos=True), dtype=np.int32),
                np.asarray(encode_text(tgt, token_to_id, max_target_length, add_bos=True), dtype=np.int32),
                np.asarray(encode_text(tgt, token_to_id, max_target_length, add_eos=True), dtype=np.int32),
            )

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        ),
    )
    if shuffle:
        dataset = dataset.shuffle(min(len(src_texts), 10000), reshuffle_each_iteration=True)
    if bucket_boundaries and bucket_batch_sizes:
        dataset = dataset.bucket_by_sequence_length(
            element_length_func=lambda src, dec_in, dec_tgt: tf.maximum(tf.shape(src)[0], tf.shape(dec_tgt)[0]),
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
            padded_shapes=([None], [None], [None]),
            padding_values=(pad_id, pad_id, pad_id),
            pad_to_bucket_boundary=False,
            drop_remainder=False,
        )
    else:
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=([None], [None], [None]),
            padding_values=(pad_id, pad_id, pad_id),
        )
    return dataset.prefetch(tf.data.AUTOTUNE)


def build_source_dataset(src_texts, token_to_id, max_source_length, batch_size):
    pad_id = token_to_id["<pad>"]

    def generator():
        for src in src_texts:
            yield np.asarray(encode_text(src, token_to_id, max_source_length, add_eos=True), dtype=np.int32)

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=tf.TensorSpec(shape=(None,), dtype=tf.int32),
    )
    dataset = dataset.padded_batch(batch_size, padded_shapes=[None], padding_values=pad_id)
    return dataset.prefetch(tf.data.AUTOTUNE)


def create_model(args, vocab_size, token_to_id):
    return Transformer(
        TransformerConfig(
            vocab_size=vocab_size,
            pad_id=token_to_id["<pad>"],
            bos_id=token_to_id["<s>"],
            eos_id=token_to_id["</s>"],
            model_dim=args.model_dim,
            ffn_dim=args.ffn_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
        )
    )


def label_smoothed_loss(labels, logits, pad_id, label_smoothing):
    logits = tf.cast(logits, tf.float32)
    vocab_size = tf.shape(logits)[-1]
    flat_labels = tf.reshape(labels, [-1])
    flat_logits = tf.reshape(logits, [-1, vocab_size])
    log_probs = tf.nn.log_softmax(flat_logits, axis=-1)
    nll = -tf.gather(log_probs, flat_labels, axis=1, batch_dims=1)
    nll = tf.reshape(nll, tf.shape(labels))
    smooth = -tf.reduce_mean(tf.reshape(log_probs, tf.concat([tf.shape(labels), [vocab_size]], axis=0)), axis=-1)
    losses = (1.0 - label_smoothing) * nll + label_smoothing * smooth
    mask = tf.cast(tf.not_equal(labels, pad_id), tf.float32)
    return tf.reduce_sum(losses * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)


def make_train_step(model, optimizer, pad_id, label_smoothing):
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None], dtype=tf.int32),
            tf.TensorSpec(shape=[None, None], dtype=tf.int32),
            tf.TensorSpec(shape=[None, None], dtype=tf.int32),
        ],
        reduce_retracing=True,
    )
    def train_step(encoder_input, decoder_input, decoder_target):
        with tf.GradientTape() as tape:
            logits = model((encoder_input, decoder_input), training=True)
            loss = label_smoothed_loss(decoder_target, logits, pad_id, label_smoothing)
            scaled_loss = optimizer.scale_loss(loss) if hasattr(optimizer, "scale_loss") else loss
        grads = tape.gradient(scaled_loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    return train_step


def generate_predictions(model, src_texts, vocab, token_to_id, max_source_length, max_target_length, batch_size, beam_size):
    eos_id = token_to_id["</s>"]
    pad_id = token_to_id["<pad>"]
    if beam_size == 1:
        predictions = []
        for batch in build_source_dataset(src_texts, token_to_id, max_source_length, batch_size):
            outputs = model.greedy_decode(batch, max_target_length)
            predictions.extend(decode_ids(row.numpy(), vocab, eos_id, pad_id) for row in outputs)
        return predictions
    predictions = []
    for src in src_texts:
        src_ids = np.asarray(encode_text(src, token_to_id, max_source_length, add_eos=True), dtype=np.int32)
        outputs = model.beam_decode(tf.constant(src_ids), max_target_length, beam_size=beam_size)
        predictions.append(decode_ids(outputs, vocab, eos_id, pad_id))
    return predictions


def evaluate_bleu(model, src_texts, ids, split_jsonl, vocab, token_to_id, target_lang, max_source_length, max_target_length, batch_size):
    predictions = generate_predictions(
        model,
        src_texts,
        vocab,
        token_to_id,
        max_source_length,
        max_target_length,
        batch_size,
        beam_size=1,
    )
    refs = load_reference_sets(ids, split_jsonl, target_lang)
    token_refs = [[r.split() for r in ref_set] for ref_set in refs]
    token_preds = [prediction.split() for prediction in predictions]
    return float(round(corpus_bleu(token_refs, token_preds), 2))


def token_length(text, max_length, reserve):
    return min(len(text.split()) + reserve, max_length)


def estimate_steps(src_texts, tgt_texts, max_source_length, max_target_length, bucket_boundaries, bucket_batch_sizes):
    counts = [0] * len(bucket_batch_sizes)
    for src, tgt in zip(src_texts, tgt_texts):
        length = max(
            token_length(src, max_source_length, 1),
            token_length(tgt, max_target_length, 1),
        )
        bucket = 0
        while bucket < len(bucket_boundaries) and length > bucket_boundaries[bucket]:
            bucket += 1
        counts[bucket] += 1
    return sum((count + batch_size - 1) // batch_size for count, batch_size in zip(counts, bucket_batch_sizes))


def main():
    args = build_args()

    if args.source == args.target:
        raise ValueError("source and target must differ")

    has_gpu = configure_runtime(getattr(args, "runtime", {}))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_src, train_tgt, _ = load_parallel(args.data_dir, "train", args.source, args.target, args.limit_train)
    valid_src, _, valid_ids = load_parallel(args.data_dir, "valid", args.source, args.target, args.limit_valid)
    vocab_path = output_dir / "vocab.json"
    if vocab_path.exists():
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        token_to_id = {token: i for i, token in enumerate(vocab)}
    else:
        vocab, token_to_id = build_vocab(train_src + train_tgt, args.max_vocab_size)
        vocab_path.write_text(json.dumps(vocab), encoding="utf-8")

    if not has_gpu:
        print(json.dumps(to_jsonable({"device": "cpu", "batch_size": args.batch_size})))

    model = create_model(args, len(vocab), token_to_id)
    model.warmup()
    base_optimizer = tf.keras.optimizers.AdamW(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
    )
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_optimizer) if args.runtime.get("mixed_precision", False) else base_optimizer
    optimizer.build(model.trainable_variables)

    ckpt_path = output_dir / "best.weights.h5"
    history_path = output_dir / "history.json"
    periodic_dir = output_dir / "epoch_checkpoints"
    periodic_dir.mkdir(parents=True, exist_ok=True)
    completed_epochs = tf.Variable(0, dtype=tf.int32, trainable=False)
    periodic_ckpt = tf.train.Checkpoint(
        model=model,
        optimizer=optimizer,
        completed_epochs=completed_epochs,
    )
    periodic_manager = tf.train.CheckpointManager(periodic_ckpt, str(periodic_dir), max_to_keep=10)

    start_epoch = 0
    if periodic_manager.latest_checkpoint:
        periodic_ckpt.restore(periodic_manager.latest_checkpoint)
        start_epoch = int(completed_epochs.numpy())
        print(json.dumps(to_jsonable({"restored_checkpoint": periodic_manager.latest_checkpoint, "start_epoch": start_epoch})))
    elif ckpt_path.exists():
        model.load_weights(str(ckpt_path))

    if not args.eval_only:
        train_dataset = build_dataset(
            train_src,
            train_tgt,
            token_to_id,
            args.max_source_length,
            args.max_target_length,
            args.batch_size,
            shuffle=True,
            bucket_boundaries=args.bucket_boundaries,
            bucket_batch_sizes=args.bucket_batch_sizes,
        )
        train_step = make_train_step(model, optimizer, token_to_id["<pad>"], args.label_smoothing)
        history = []
        if history_path.exists():
            history = [item for item in json.loads(history_path.read_text(encoding="utf-8")) if item.get("epoch", 0) <= start_epoch]
        best_bleu = max([item.get("valid_bleu", -1.0) for item in history if "valid_bleu" in item], default=-1.0)
        patience = 0
        steps_per_epoch = estimate_steps(
            train_src,
            train_tgt,
            args.max_source_length,
            args.max_target_length,
            args.bucket_boundaries,
            args.bucket_batch_sizes,
        )
        for epoch in range(start_epoch + 1, args.epochs + 1):
            losses = []
            progress = tqdm(train_dataset, total=steps_per_epoch, desc=f"epoch {epoch}/{args.epochs}", unit="batch", dynamic_ncols=True)
            for encoder_input, decoder_input, decoder_target in progress:
                loss = float(train_step(encoder_input, decoder_input, decoder_target).numpy())
                losses.append(loss)
                progress.set_postfix(loss=round(sum(losses) / len(losses), 4))
            valid_bleu = None
            if epoch % args.validate_every == 0 or epoch == args.epochs:
                valid_bleu = evaluate_bleu(
                    model,
                    valid_src,
                    valid_ids,
                    Path(args.data_dir) / "valid.jsonl",
                    vocab,
                    token_to_id,
                    args.target,
                    args.max_source_length,
                    args.max_target_length,
                    args.eval_batch_size,
                )
            epoch_result = {"epoch": epoch, "train_loss": round(sum(losses) / max(len(losses), 1), 4)}
            if valid_bleu is not None:
                epoch_result["valid_bleu"] = round(valid_bleu, 2)
            epoch_result = to_jsonable(epoch_result)
            history.append(epoch_result)
            history_path.write_text(json.dumps(to_jsonable(history), indent=2), encoding="utf-8")
            print(json.dumps(epoch_result))
            if epoch % args.save_every == 0:
                completed_epochs.assign(epoch)
                periodic_manager.save(checkpoint_number=epoch)
            if valid_bleu is not None:
                if valid_bleu > best_bleu:
                    best_bleu = valid_bleu
                    patience = 0
                    model.save_weights(str(ckpt_path))
                else:
                    patience += 1
                    if patience >= args.patience:
                        break
        if history and completed_epochs.numpy() != history[-1]["epoch"]:
            completed_epochs.assign(history[-1]["epoch"])
            periodic_manager.save(checkpoint_number=history[-1]["epoch"])
        model.load_weights(str(ckpt_path))

    print(json.dumps(to_jsonable({"checkpoint": str(ckpt_path), "output_dir": str(output_dir)})))


main()
