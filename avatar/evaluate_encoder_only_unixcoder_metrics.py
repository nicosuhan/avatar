import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from metrics import (
    detokenize,
    java_compile,
    load_reference_sets,
    load_testcases,
    outputs_match,
    python_compile,
    read_lines,
    text_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "avatar" / "data" / "avatar"
EXT_LIB_DIR = ROOT / "avatar" / "data" / "avatar_extLibraries"
OUTPUT_ROOT = ROOT / "outputs"
RESULTS_ROOT = ROOT / "results" / "encoder_only_unixcoder_metrics"
TMP_ROOT = RESULTS_ROOT / "tmp"
DEFAULT_WORKERS = min(max(os.cpu_count() or 1, 1), 8)
PRINT_EVERY_STAGE_ROWS = 25

TASKS = {
    "python2java": {
        "source": "python",
        "target": "java",
        "output_dir": OUTPUT_ROOT / "unixcoder_encoder_only_python2java",
        "batch_prefix": "python2java",
    },
    "java2python": {
        "source": "java",
        "target": "python",
        "output_dir": OUTPUT_ROOT / "unixcoder_encoder_only_java2python",
        "batch_prefix": "java2python",
    },
}


def now():
    return datetime.now(timezone.utc).isoformat()


def short(text, limit=4000):
    return (text or "").strip()[:limit]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(data, indent=2)
    last_error = None
    for _ in range(10):
        try:
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.2)
    try:
        path.write_text(payload, encoding="utf-8")
    except PermissionError:
        print(f"warning: could not update {path}: {last_error}", flush=True)


def append_jsonl(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_jsonl(path):
    rows = {}
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                rows[int(row["index"])] = row
            except Exception:
                pass
    return rows


def batch_path(output_dir, prefix, start, end):
    return output_dir / f"{prefix}_batch_{start:04d}_{end - 1:04d}.pred"


def verify_and_merge_batches(output_dir, prefix, total, batch_size):
    missing_or_bad = []
    predictions = []
    expected_batches = 0
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        expected_batches += 1
        path = batch_path(output_dir, prefix, start, end)
        if not path.exists():
            missing_or_bad.append({"file": path.name, "reason": "missing"})
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        expected_lines = end - start
        if len(lines) != expected_lines:
            missing_or_bad.append(
                {
                    "file": path.name,
                    "reason": "bad_line_count",
                    "expected": expected_lines,
                    "actual": len(lines),
                }
            )
            continue
        predictions.extend(lines)

    if missing_or_bad:
        raise RuntimeError(f"{prefix} has missing or incomplete batch files: {missing_or_bad[:10]}")

    test_pred = output_dir / "test.pred"
    test_pred.write_text("\n".join(predictions) + "\n", encoding="utf-8")
    return {
        "expected_batches": expected_batches,
        "merged_predictions": len(predictions),
        "test_pred": str(test_pred),
    }


def compile_one(index, problem_id, prediction, lang, ext_lib_dir):
    if lang == "java" and shutil.which("javac") is None:
        return {"index": index, "id": problem_id, "status": "error", "ok": False, "error_type": "missing_javac"}
    code = detokenize(prediction, lang)
    try:
        with tempfile.TemporaryDirectory(prefix="compile_", dir=str(TMP_ROOT)) as work_dir:
            proc, _, _ = python_compile(code, work_dir, ext_lib_dir) if lang == "python" else java_compile(code, work_dir, ext_lib_dir)
        ok = proc.returncode == 0
        return {
            "index": index,
            "id": problem_id,
            "status": "compiled" if ok else "compile_error",
            "ok": ok,
            "stderr": short(proc.stderr),
        }
    except subprocess.TimeoutExpired as exc:
        return {"index": index, "id": problem_id, "status": "error", "ok": False, "error_type": "compile_timeout", "stderr": short(str(exc))}
    except Exception as exc:
        return {"index": index, "id": problem_id, "status": "error", "ok": False, "error_type": type(exc).__name__, "stderr": short(str(exc))}


def execute_one(index, problem_id, prediction, lang, testcases, ext_lib_dir, timeout):
    if lang == "java" and shutil.which("javac") is None:
        return {"index": index, "id": problem_id, "status": "error", "error_type": "missing_javac", "stderr": ""}
    code = detokenize(prediction, lang)
    try:
        with tempfile.TemporaryDirectory(prefix="exec_", dir=str(TMP_ROOT)) as work_dir:
            if lang == "python":
                compile_proc, script_path, env = python_compile(code, work_dir, ext_lib_dir)
                if compile_proc.returncode != 0:
                    return {"index": index, "id": problem_id, "status": "error", "error_type": "compile", "stderr": short(compile_proc.stderr)}
                command = [sys.executable, str(script_path)]
                run_env = env
            else:
                compile_proc, class_name, classpath = java_compile(code, work_dir, ext_lib_dir)
                if compile_proc.returncode != 0:
                    return {"index": index, "id": problem_id, "status": "error", "error_type": "compile", "stderr": short(compile_proc.stderr)}
                command = ["java", "-cp", classpath, class_name]
                run_env = None

            for case_index, (input_text, expected_output) in enumerate(testcases):
                try:
                    proc = subprocess.run(command, input=input_text, capture_output=True, text=True, timeout=timeout, env=run_env)
                except subprocess.TimeoutExpired as exc:
                    return {
                        "index": index,
                        "id": problem_id,
                        "status": "error",
                        "error_type": "runtime",
                        "case": case_index,
                        "stderr": short(str(exc)),
                    }
                if proc.returncode != 0 or proc.stderr.strip():
                    return {
                        "index": index,
                        "id": problem_id,
                        "status": "error",
                        "error_type": "runtime",
                        "case": case_index,
                        "stderr": short(proc.stderr or proc.stdout),
                    }
                if not outputs_match(proc.stdout, expected_output):
                    return {"index": index, "id": problem_id, "status": "failure", "error_type": None, "case": case_index, "stderr": ""}
        return {"index": index, "id": problem_id, "status": "success", "error_type": None, "stderr": ""}
    except Exception as exc:
        return {"index": index, "id": problem_id, "status": "error", "error_type": type(exc).__name__, "stderr": short(str(exc))}


def summarize(task_name, total_predictions, testcase_total, text, compile_records, exec_records, merge):
    compiled = sum(1 for row in compile_records.values() if row.get("ok"))
    compile_done = len(compile_records)
    success = sum(1 for row in exec_records.values() if row.get("status") == "success")
    failure = sum(1 for row in exec_records.values() if row.get("status") == "failure")
    error = sum(1 for row in exec_records.values() if row.get("status") == "error")
    compile_error = sum(1 for row in exec_records.values() if row.get("error_type") == "compile")
    runtime_error = sum(1 for row in exec_records.values() if row.get("error_type") == "runtime")
    exec_done = len(exec_records)
    return {
        "task": task_name,
        "updated_at": now(),
        "total_predictions": total_predictions,
        "merge": merge,
        **text,
        "compile_done": compile_done,
        "compile_total": total_predictions,
        "compilation_accuracy": round(100.0 * compiled / max(compile_done, 1), 2),
        "compiled": compiled,
        "execution_done": exec_done,
        "testcase_total": testcase_total,
        "execution_accuracy": round(100.0 * (success + failure) / max(exec_done, 1), 2),
        "computational_accuracy": round(100.0 * success / max(exec_done, 1), 2),
        "success": success,
        "failure": failure,
        "error": error,
        "compile_error": compile_error,
        "runtime_error": runtime_error,
        "compile_error_rate": round(100.0 * compile_error / max(exec_done, 1), 2),
        "runtime_error_rate": round(100.0 * runtime_error / max(exec_done, 1), 2),
        "complete": compile_done == total_predictions and exec_done == testcase_total,
    }


def save_progress(result_dir, summary, stage, done, expected):
    write_json(result_dir / "summary.json", summary)
    append_jsonl(result_dir / "progress.jsonl", {"time": now(), "stage": stage, "done": done, "expected": expected, "summary": summary})
    if done == expected or done % PRINT_EVERY_STAGE_ROWS == 0:
        print(
            f"{summary['task']} {stage} {done}/{expected} "
            f"compile={summary['compilation_accuracy']} exec={summary['execution_accuracy']} ca={summary['computational_accuracy']}",
            flush=True,
        )


def run_stage(result_dir, stage, records, items, worker, summary_args, workers):
    task_name, total_predictions, testcase_total, text, compile_records, exec_records, merge = summary_args
    pending = [item for item in items if item[0] not in records]
    expected = len(items)
    print(f"{task_name} {stage} resume: {len(records)}/{expected} already saved, {len(pending)} pending", flush=True)
    current_compile = records if stage == "compile" else compile_records
    current_exec = records if stage == "execution" else exec_records
    save_progress(result_dir, summarize(task_name, total_predictions, testcase_total, text, current_compile, current_exec, merge), stage, expected - len(pending), expected)
    if not pending:
        return records
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, *item): item[0] for item in pending}
        for future in as_completed(futures):
            row = future.result()
            records[int(row["index"])] = row
            append_jsonl(result_dir / f"{stage}.jsonl", row)
            current_compile = records if stage == "compile" else compile_records
            current_exec = records if stage == "execution" else exec_records
            summary = summarize(task_name, total_predictions, testcase_total, text, current_compile, current_exec, merge)
            save_progress(result_dir, summary, stage, len(records), expected)
    return records


def evaluate_task(name, batch_size, workers, timeout, skip_compile, skip_exec):
    task = TASKS[name]
    output_dir = task["output_dir"]
    if not output_dir.exists():
        raise FileNotFoundError(f"Missing output directory for {name}: {output_dir}")

    result_dir = RESULTS_ROOT / name
    result_dir.mkdir(parents=True, exist_ok=True)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)

    ids = read_lines(DATA_DIR / "test.java-python.id")
    merge = verify_and_merge_batches(output_dir, task["batch_prefix"], len(ids), batch_size)
    predictions = read_lines(output_dir / "test.pred")
    if len(predictions) != len(ids):
        raise RuntimeError(f"{name} has {len(predictions)} predictions, expected {len(ids)}")

    refs = load_reference_sets(ids, DATA_DIR / "test.jsonl", task["target"])
    text_path = result_dir / "text_metrics.json"
    if text_path.exists():
        text = json.loads(text_path.read_text(encoding="utf-8"))
    else:
        print(f"{name} text metrics start", flush=True)
        text = text_metrics(refs, predictions, task["target"])
        write_json(text_path, text)
        print(f"{name} text metrics saved: bleu={text['bleu']} codebleu={text['codebleu']}", flush=True)

    history_path = output_dir / "history.json"
    config_path = output_dir / "config.json"
    run_meta = {
        "task": name,
        "source": task["source"],
        "target": task["target"],
        "model": "microsoft/unixcoder-base encoder-only",
        "output_dir": str(output_dir),
        "training_history": json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else None,
        "run_config": json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else None,
    }
    write_json(result_dir / "run_meta.json", run_meta)

    testcase_map = load_testcases(DATA_DIR)
    execution_items = [
        (i, problem_id, predictions[i], task["target"], testcase_map[problem_id], str(EXT_LIB_DIR), timeout)
        for i, problem_id in enumerate(ids)
        if problem_id in testcase_map
    ]
    compile_records = load_jsonl(result_dir / "compile.jsonl")
    exec_records = load_jsonl(result_dir / "execution.jsonl")
    summary_args = (name, len(predictions), len(execution_items), text, compile_records, exec_records, merge)

    if not skip_compile:
        compile_items = [(i, problem_id, predictions[i], task["target"], str(EXT_LIB_DIR)) for i, problem_id in enumerate(ids)]
        compile_records = run_stage(result_dir, "compile", compile_records, compile_items, compile_one, summary_args, workers)
        summary_args = (name, len(predictions), len(execution_items), text, compile_records, exec_records, merge)

    if not skip_exec:
        exec_records = run_stage(result_dir, "execution", exec_records, execution_items, execute_one, summary_args, workers)

    summary = summarize(name, len(predictions), len(execution_items), text, compile_records, exec_records, merge)
    write_json(result_dir / "summary.json", summary)
    write_json(result_dir / "metrics.json", {**run_meta, **summary})
    return {**run_meta, **summary}


def main():
    parser = argparse.ArgumentParser(description="Evaluate encoder-only UniXcoder AVATAR predictions with resumable logs.")
    parser.add_argument("--task", choices=sorted(TASKS), default="python2java")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-exec", action="store_true")
    args = parser.parse_args()

    result = evaluate_task(
        args.task,
        batch_size=args.batch_size,
        workers=args.workers,
        timeout=args.timeout,
        skip_compile=args.skip_compile,
        skip_exec=args.skip_exec,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
