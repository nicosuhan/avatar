from constants import METRICS_CONFIG, RUNS_DIR

import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tokenize
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

try:
    from codebleu import calc_codebleu
except ImportError:
    calc_codebleu = None


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
    geo_mean = math.exp(sum(math.log(p) / max_order for p in precisions)) if min(precisions) > 0 else 0.0
    ratio = hyp_len / max(ref_len, 1)
    bp = 1.0 if ratio > 1.0 else math.exp(1.0 - 1.0 / max(ratio, 1e-9))
    return 100.0 * geo_mean * bp


def text_metrics(reference_sets, predictions, lang):
    token_refs = [[r.split() for r in refs] for refs in reference_sets]
    token_preds = [p.split() for p in predictions]
    exact = sum(any(ref == pred for ref in refs) for refs, pred in zip(token_refs, token_preds))
    bleu = round(corpus_bleu(token_refs, token_preds), 2)
    result = {
        "bleu": bleu,
        "exact_match": round(100.0 * exact / max(len(predictions), 1), 2),
    }
    if calc_codebleu is None:
        result.update(
            {
                "ngram_match": bleu,
                "weighted_ngram_match": bleu,
                "syntax_match": 0.0,
                "dataflow_match": 0.0,
                "codebleu": round((bleu + bleu) / 4.0, 2),
                "codebleu_available": False,
            }
        )
        return result
    codebleu = calc_codebleu(reference_sets, predictions, lang)
    result.update(
        {
            "ngram_match": round(100.0 * codebleu["ngram_match_score"], 2),
            "weighted_ngram_match": round(100.0 * codebleu["weighted_ngram_match_score"], 2),
            "syntax_match": round(100.0 * codebleu["syntax_match_score"], 2),
            "dataflow_match": round(100.0 * codebleu["dataflow_match_score"], 2),
            "codebleu": round(100.0 * codebleu["codebleu"], 2),
            "codebleu_available": True,
        }
    )
    return result


def _fix_literal(match):
    quote = match.group(0)[0]
    body = match.group(0)[1:-1]
    body = body.replace("STRNEWLINE", "\n").replace("TABSYMBOL", "\t")
    body = body.replace("â–", "▁").replace(" ", "").replace("▁", " ")
    return f"{quote}{body}{quote}"


def detokenize_java(code):
    code = " ".join(code) if isinstance(code, list) else code
    code = code.replace("â–", "▁")
    code = re.sub(r'"(?:\\.|[^"\\])*"', _fix_literal, code)
    code = re.sub(r"'(?:\\.|[^'\\])*'", _fix_literal, code)
    return code.replace("> > >", ">>>").replace("< < <", "<<<").replace("> >", ">>").replace("< <", "<<")


def detokenize_python(code):
    code = " ".join(code) if isinstance(code, list) else code
    code = code.replace("ENDCOM", "NEW_LINE").replace("â–", "▁")
    lines = []
    indent = 0
    for raw in code.split("NEW_LINE"):
        raw = raw.strip()
        while raw.startswith("DEDENT"):
            indent = max(indent - 1, 0)
            raw = raw[6:].strip()
        while raw.startswith("INDENT"):
            indent += 1
            raw = raw[6:].strip()
        lines.append(("    " * indent + raw) if raw else "")
    code = "\n".join(lines)
    try:
        for tok in list(tokenize.tokenize(io.BytesIO(code.encode("utf-8")).readline)):
            if tok.type in {tokenize.STRING, tokenize.COMMENT}:
                fixed = tok.string.replace("STRNEWLINE", "\n").replace("TABSYMBOL", "\t")
                fixed = fixed.replace("▁", "SPACETOKEN").replace(" ", "").replace("SPACETOKEN", " ")
                code = code.replace(tok.string, fixed)
    except Exception:
        pass
    for modifier in ["r", "u", "f", "rf", "fr", "b", "rb", "br"]:
        code = code.replace(f" {modifier} '", f" {modifier}'").replace(f' {modifier} "', f' {modifier}"')
        code = code.replace(f" {modifier.upper()} '", f" {modifier.upper()}'").replace(
            f' {modifier.upper()} "', f' {modifier.upper()}"'
        )
    return (
        code.replace("import.", "import .")
        .replace("from.", "from .")
        .replace("> >", ">>")
        .replace("< <", "<<")
        .replace("▁", " ")
    )


def detokenize(code, lang):
    return detokenize_python(code) if lang == "python" else detokenize_java(code)


def outputs_match(predicted, expected):
    pred_lines = predicted.strip().splitlines()
    exp_lines = expected.strip().splitlines()
    if len(pred_lines) != len(exp_lines):
        return False
    for pred_line, exp_line in zip(pred_lines, exp_lines):
        pred_tokens = pred_line.strip().split()
        exp_tokens = exp_line.strip().split()
        if len(pred_tokens) != len(exp_tokens):
            return False
        for pred_token, exp_token in zip(pred_tokens, exp_tokens):
            if pred_token == exp_token:
                continue
            try:
                if abs(float(pred_token) - float(exp_token)) < 1e-6:
                    continue
            except ValueError:
                pass
            return False
    return True


def java_class_name(code):
    match = re.search(r"\bpublic\s+(?:final\s+)?class\s+([A-Za-z_]\w*)", code)
    if match:
        return match.group(1)
    match = re.search(r"\bclass\s+([A-Za-z_]\w*)", code)
    return match.group(1) if match else "Main"


def python_compile(code, work_dir, ext_lib_dir):
    path = Path(work_dir) / "main.py"
    path.write_text(code, encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ext_lib_dir), env.get("PYTHONPATH", "")]).strip(os.pathsep)
    proc = subprocess.run(
        [sys.executable, "-m", "py_compile", str(path)],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    return proc, path, env


def java_compile(code, work_dir, ext_lib_dir):
    if shutil.which("javac") is None:
        raise FileNotFoundError("javac was not found. Install a JDK or set skip_compile/skip_exec for Java evaluation.")
    class_name = java_class_name(code)
    path = Path(work_dir) / f"{class_name}.java"
    path.write_text(code, encoding="utf-8")
    classpath = os.pathsep.join([str(work_dir), str(ext_lib_dir)])
    proc = subprocess.run(
        ["javac", "-cp", classpath, str(path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return proc, class_name, classpath


def compile_worker(item):
    code, lang, ext_lib_dir = item
    code = detokenize(code, lang)
    with tempfile.TemporaryDirectory(prefix="avatar_compile_", dir=str(RUNS_DIR)) as work_dir:
        if lang == "python":
            proc, _, _ = python_compile(code, work_dir, ext_lib_dir)
        else:
            proc, _, _ = java_compile(code, work_dir, ext_lib_dir)
        return proc.returncode == 0


def execution_worker(item):
    code, problem_id, lang, testcases, ext_lib_dir, timeout = item
    code = detokenize(code, lang)
    with tempfile.TemporaryDirectory(prefix="avatar_eval_", dir=str(RUNS_DIR)) as work_dir:
        if lang == "python":
            compile_proc, script_path, env = python_compile(code, work_dir, ext_lib_dir)
            if compile_proc.returncode != 0:
                return {"id": problem_id, "status": "error", "error_type": "compile", "stderr": compile_proc.stderr.strip()}
            for input_text, expected_output in testcases:
                proc = subprocess.run(
                    [sys.executable, str(script_path)],
                    input=input_text,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=env,
                )
                if proc.returncode != 0 or proc.stderr.strip():
                    return {
                        "id": problem_id,
                        "status": "error",
                        "error_type": "runtime",
                        "stderr": (proc.stderr or proc.stdout).strip(),
                    }
                if not outputs_match(proc.stdout, expected_output):
                    return {"id": problem_id, "status": "failure", "error_type": None, "stderr": ""}
            return {"id": problem_id, "status": "success", "error_type": None, "stderr": ""}
        compile_proc, class_name, classpath = java_compile(code, work_dir, ext_lib_dir)
        if compile_proc.returncode != 0:
            return {"id": problem_id, "status": "error", "error_type": "compile", "stderr": compile_proc.stderr.strip()}
        for input_text, expected_output in testcases:
            proc = subprocess.run(
                ["java", "-cp", classpath, class_name],
                input=input_text,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if proc.returncode != 0 or proc.stderr.strip():
                return {
                    "id": problem_id,
                    "status": "error",
                    "error_type": "runtime",
                    "stderr": (proc.stderr or proc.stdout).strip(),
                }
            if not outputs_match(proc.stdout, expected_output):
                return {"id": problem_id, "status": "failure", "error_type": None, "stderr": ""}
        return {"id": problem_id, "status": "success", "error_type": None, "stderr": ""}


def load_testcases(data_dir):
    merged = {}
    for path in sorted(Path(data_dir).glob("io_testcases_*.json")):
        if path.name.endswith("ORIG.json"):
            continue
        with open(path, encoding="utf-8") as f:
            merged.update(json.load(f))
    return merged


def compile_metrics(predictions, lang, ext_lib_dir, workers):
    if lang == "java" and shutil.which("javac") is None:
        return {
            "compilation_accuracy": 0.0,
            "compiled": 0,
            "compile_total": len(predictions),
            "javac_available": False,
        }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        success = list(pool.map(compile_worker, [(p, lang, ext_lib_dir) for p in predictions]))
    compiled = sum(success)
    return {
        "compilation_accuracy": round(100.0 * compiled / max(len(predictions), 1), 2),
        "compiled": compiled,
        "compile_total": len(predictions),
        "javac_available": True,
    }


def execution_metrics(predictions, ids, lang, data_dir, ext_lib_dir, workers, timeout):
    if lang == "java" and shutil.which("javac") is None:
        return {
            "testcase_total": 0,
            "execution_accuracy": 0.0,
            "computational_accuracy": 0.0,
            "success": 0,
            "failure": 0,
            "error": 0,
            "compile_error": 0,
            "runtime_error": 0,
            "compile_error_rate": 0.0,
            "runtime_error_rate": 0.0,
            "logs": [],
            "javac_available": False,
        }
    testcase_map = load_testcases(data_dir)
    items = [
        (prediction, problem_id, lang, testcase_map[problem_id], ext_lib_dir, timeout)
        for prediction, problem_id in zip(predictions, ids)
        if problem_id in testcase_map
    ]
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(execution_worker, items))
    total = len(results)
    success = sum(r["status"] == "success" for r in results)
    failure = sum(r["status"] == "failure" for r in results)
    error = sum(r["status"] == "error" for r in results)
    compile_error = sum(r["error_type"] == "compile" for r in results)
    runtime_error = sum(r["error_type"] == "runtime" for r in results)
    return {
        "testcase_total": total,
        "execution_accuracy": round(100.0 * (success + failure) / max(total, 1), 2),
        "computational_accuracy": round(100.0 * success / max(total, 1), 2),
        "success": success,
        "failure": failure,
        "error": error,
        "compile_error": compile_error,
        "runtime_error": runtime_error,
        "compile_error_rate": round(100.0 * compile_error / max(total, 1), 2),
        "runtime_error_rate": round(100.0 * runtime_error / max(total, 1), 2),
        "logs": results,
        "javac_available": True,
    }


def evaluate(predictions, ids, reference_sets, lang, data_dir, ext_lib_dir, workers=4, timeout=10, run_compile=True, run_exec=True):
    result = text_metrics(reference_sets, predictions, lang)
    if run_compile:
        result.update(compile_metrics(predictions, lang, ext_lib_dir, workers))
    if run_exec:
        result.update(execution_metrics(predictions, ids, lang, data_dir, ext_lib_dir, workers, timeout))
    return result


def main():
    args = SimpleNamespace(**METRICS_CONFIG)
    predictions = read_lines(args.predictions)
    ids = read_lines(args.ids)
    reference_sets = load_reference_sets(ids, args.references_jsonl, args.target_lang)
    result = evaluate(
        predictions,
        ids,
        reference_sets,
        args.target_lang,
        args.data_dir,
        args.ext_lib_dir,
        workers=args.workers,
        timeout=args.timeout,
        run_compile=not args.skip_compile,
        run_exec=not args.skip_exec,
    )
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
