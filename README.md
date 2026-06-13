# AVATAR Java-Python Program Translation

This repository contains the code and thesis material for a bachelor's thesis on
automatic program translation between Java and Python using the AVATAR dataset.
The project studies whether modern pretrained code models can translate programs
while preserving executable behavior, not only lexical similarity.

The experiments compare four model categories:

- **Zero-shot**: Qwen2.5-Coder-7B-Instruct used directly through prompting.
- **Encoder-only**: UniXcoder as the source encoder with a learned Transformer
  decoder.
- **Decoder-only**: Qwen2.5-Coder-0.5B-Instruct fine-tuned with QLoRA.
- **Seq2Seq**: CodeT5+ 220M fine-tuned as an encoder-decoder model.

The task is evaluated in both directions: Java-to-Python and Python-to-Java.
The evaluation uses BLEU, CodeBLEU when available, syntax or compilation checks,
Execution Accuracy, and Computational Accuracy.

## Repository Layout

```text
avatar/
  constants.py                         Shared paths and baseline configuration
  metrics.py                           BLEU, CodeBLEU, compilation, and execution metrics
  tokenize_generated_code.py           Helpers for converting generated code to AVATAR token format

  zeroshot_qwen_kaggle.py              Zero-shot Qwen2.5-Coder-7B experiment
  zeroshot_qwen_metrics.py             Metrics for zero-shot Qwen outputs

  encoder_only_unixcoder_kaggle.py     Encoder-only UniXcoder experiment
  evaluate_encoder_only_unixcoder_metrics.py

  decoder_only_qwen_kaggle.py          Decoder-only Qwen2.5-Coder-0.5B QLoRA experiment
  evaluate_decoder_only_qwen_metrics.py

  seq2seq_codet5p_kaggle.py            Seq2Seq CodeT5+ 220M experiment
  evaluate_seq2seq_codet5p_metrics.py

  transformer.py                       Baseline Transformer model
  train_transformer.py                 Baseline Transformer training script

  data/avatar_extLibraries/            Java/Python helper libraries used by AVATAR evaluation

latex/
  tex/                                 Thesis source files
  bibliography.bib                     Thesis bibliography
  out/main.pdf                         Compiled thesis PDF
```

Generated predictions, model checkpoints, metric runs, local datasets, and other
large runtime artifacts are intentionally ignored by Git.

## Dataset

The code expects the AVATAR data split files under:

```text
avatar/data/avatar/
```

The local dataset copy is not committed to GitHub. On Kaggle, the scripts first
look for the dataset in the configured Kaggle input locations. Locally, place the
AVATAR files in `avatar/data/avatar/` with names such as:

```text
train.java-python.java
train.java-python.python
valid.java-python.java
valid.java-python.python
test.java-python.java
test.java-python.python
test.java-python.id
test.jsonl
```

The evaluation scripts also use `avatar/data/avatar_extLibraries/` for helper
libraries needed by some Java programs. Java source helper files may be tracked,
but generated `.class` files are ignored.

## Environment

The training scripts were written for Kaggle GPU notebooks, but they can also be
run locally if the same dependencies and dataset layout are available.

A typical local Python environment is:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch transformers peft accelerate bitsandbytes huggingface-hub tqdm numpy
```

For the baseline Transformer scripts, install TensorFlow/Keras as well:

```powershell
pip install tensorflow keras
```

For CodeBLEU, install the `codebleu` package if available in your environment.
Without it, the custom BLEU and execution-based metrics still run.

For execution-based Java evaluation, a JDK must be available on `PATH`:

```powershell
javac -version
java -version
```

## Running Experiments

Each main experiment script is self-contained and defines its model, data paths,
training configuration, generation configuration, and output directory inside the
file. The scripts are intended to be copied into or run inside a Kaggle notebook.

```powershell
python avatar/zeroshot_qwen_kaggle.py
python avatar/encoder_only_unixcoder_kaggle.py
python avatar/decoder_only_qwen_kaggle.py
python avatar/seq2seq_codet5p_kaggle.py
```

The generated artifacts are written under `outputs/`. This directory is ignored
because it can contain predictions, checkpoints, and large intermediate files.

## Evaluating Models

The metric scripts expect the generated outputs to already exist in the expected
`outputs/` subdirectories. They write resumable metric logs and summaries under
`results/`, which is also ignored by Git.

Zero-shot Qwen:

```powershell
python avatar/zeroshot_qwen_metrics.py
```

Encoder-only UniXcoder:

```powershell
python avatar/evaluate_encoder_only_unixcoder_metrics.py --task python2java
python avatar/evaluate_encoder_only_unixcoder_metrics.py --task java2python
```

Decoder-only Qwen:

```powershell
python avatar/evaluate_decoder_only_qwen_metrics.py --task python2java
python avatar/evaluate_decoder_only_qwen_metrics.py --task java2python
```

Seq2Seq CodeT5+:

```powershell
python avatar/evaluate_seq2seq_codet5p_metrics.py --task python2java
python avatar/evaluate_seq2seq_codet5p_metrics.py --task java2python
```

Useful evaluator options:

```powershell
--workers 8
--timeout 10
--skip-compile
--skip-exec
```

For CodeT5+, a custom prediction file can be supplied with:

```powershell
python avatar/evaluate_seq2seq_codet5p_metrics.py --task python2java --pred-path path\to\test.pred
```