from constants import BASE_TRAIN_CONFIG, OUTPUT_ROOT, SEQ2SEQ_MODEL, TRANSFORMER_MODEL
from metrics import evaluate, load_reference_sets, read_lines
from seq2seq import Seq2Seq, Seq2SeqConfig
from transformer import Transformer, TransformerConfig

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tensorflow as tf


MODEL_WEIGHTS = ""
MODEL_TYPE = "transformer"

SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>"]


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
    config.update(TRANSFORMER_MODEL if MODEL_TYPE == "transformer" else SEQ2SEQ_MODEL)
    config["output_dir"] = str(OUTPUT_ROOT / f"{config['model']}_{config['source']}2{config['target']}")
    return SimpleNamespace(**config)


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
    if args.model == "transformer":
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
    return Seq2Seq(
        Seq2SeqConfig(
            vocab_size=vocab_size,
            pad_id=token_to_id["<pad>"],
            bos_id=token_to_id["<s>"],
            eos_id=token_to_id["</s>"],
            embed_size=args.embed_size,
            encoder_hidden_size=args.encoder_hidden_size,
            decoder_hidden_size=args.decoder_hidden_size,
            encoder_layers=args.encoder_layers,
            decoder_layers=args.decoder_layers,
            dropout=args.dropout,
        )
    )


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


def main():
    if not MODEL_WEIGHTS:
        raise ValueError("set MODEL_WEIGHTS to the checkpoint path")

    args = build_args()
    weights_path = Path(MODEL_WEIGHTS)
    output_dir = weights_path.parent
    vocab = json.loads((output_dir / "vocab.json").read_text(encoding="utf-8"))
    token_to_id = {token: i for i, token in enumerate(vocab)}
    model = create_model(args, len(vocab), token_to_id)
    model.warmup()
    model.load_weights(str(weights_path))

    test_src = read_lines(Path(args.data_dir) / f"test.java-python.{args.source}")
    test_tgt = read_lines(Path(args.data_dir) / f"test.java-python.{args.target}")
    test_ids = read_lines(Path(args.data_dir) / "test.java-python.id")
    predictions = generate_predictions(
        model,
        test_src,
        vocab,
        token_to_id,
        args.max_source_length,
        args.max_target_length,
        args.batch_size,
        args.beam_size,
    )
    refs = load_reference_sets(test_ids, Path(args.data_dir) / "test.jsonl", args.target)
    result = evaluate(
        predictions,
        test_ids,
        refs,
        args.target,
        args.data_dir,
        args.ext_lib_dir,
        workers=args.workers,
        timeout=10,
        run_compile=not args.skip_compile,
        run_exec=not args.skip_exec,
    )
    result.update(
        {
            "model": args.model,
            "source": args.source,
            "target": args.target,
            "test_examples": len(test_src),
            "test_reference_pairs": len(test_tgt),
            "beam_size": args.beam_size,
            "checkpoint": str(weights_path),
        }
    )
    (output_dir / "test.pred").write_text("\n".join(predictions) + "\n", encoding="utf-8")
    result = to_jsonable(result)
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


main()
