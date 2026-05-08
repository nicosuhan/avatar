from pathlib import Path
import os

WORK_ROOT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
OUTPUT_ROOT = WORK_ROOT / "outputs"
RUNS_DIR = WORK_ROOT / "runs"
DEFAULT_WORKERS = min(max(os.cpu_count() or 1, 1), 8)
DATA_ROOT = "/kaggle/input/datasets/nicoletacorinasuhan/avatar-dataset/data/"

TRAIN_RUNTIME = {
    "gpu_index": 0,
    "expected_gpu": "GPU",
    "memory_growth": True,
    "mixed_precision": False,
    "seed": 1234,
}

METRICS_RUNTIME = {
    "gpu_index": 0,
    "expected_gpu": "GPU",
    "memory_growth": False,
    "mixed_precision": False,
    "seed": 1234,
}

SEQ2SEQ_MODEL = {
    "name": "seq2seq",
    "embed_size": 512,
    "encoder_hidden_size": 512,
    "decoder_hidden_size": 512,
    "encoder_layers": 1,
    "decoder_layers": 1,
    "dropout": 0.2,
    "label_smoothing": 0.1,
    "weight_decay": 0.01,
    "learning_rate": 1e-3,
    "batch_size": 16,
    "beam_size": 5,
    "patience": 5,
}

TRANSFORMER_MODEL = {
    "name": "transformer",
    "model_dim": 768,
    "ffn_dim": 3072,
    "num_heads": 12,
    "num_layers": 6,
    "dropout": 0.2,
    "attention_dropout": 0.2,
    "activation_dropout": 0.2,
    "label_smoothing": 0.1,
    "weight_decay": 0.01,
    "learning_rate": 1e-4,
    "batch_size": 8,
    "beam_size": 5,
    "patience": 10,
}

BASE_TRAIN_CONFIG = {
    "model": "seq2seq",
    "data_dir": str(Path(DATA_ROOT) / "avatar"),
    "ext_lib_dir": str(Path(DATA_ROOT) / "avatar_extLibraries"),
    "source": "java",
    "target": "python",
    "epochs": 10,
    "max_vocab_size": 50000,
    "max_source_length": 512,
    "max_target_length": 512,
    "workers": DEFAULT_WORKERS,
    "limit_train": 0,
    "limit_valid": 0,
    "limit_test": 0,
    "eval_only": False,
    "skip_compile": False,
    "skip_exec": False,
    "runtime": TRAIN_RUNTIME,
}

TRAIN_CONFIG = {
    "model": "seq2seq",
    **BASE_TRAIN_CONFIG,
    **SEQ2SEQ_MODEL,
    "output_dir": str(OUTPUT_ROOT / f"seq2seq_java2python"),
}

METRICS_CONFIG = {
    "predictions": str(OUTPUT_ROOT / f"{TRAIN_CONFIG['model']}_{TRAIN_CONFIG['source']}2{TRAIN_CONFIG['target']}" / "test.pred"),
    "ids": str(Path(TRAIN_CONFIG["data_dir"]) / "test.java-python.id"),
    "references_jsonl": str(Path(TRAIN_CONFIG["data_dir"]) / "test.jsonl"),
    "target_lang": TRAIN_CONFIG["target"],
    "data_dir": TRAIN_CONFIG["data_dir"],
    "ext_lib_dir": TRAIN_CONFIG["ext_lib_dir"],
    "workers": DEFAULT_WORKERS,
    "timeout": 10,
    "skip_compile": False,
    "skip_exec": False,
    "output": str(OUTPUT_ROOT / f"{TRAIN_CONFIG['model']}_{TRAIN_CONFIG['source']}2{TRAIN_CONFIG['target']}" / "metrics_from_script.json"),
    "runtime": METRICS_RUNTIME,
}
