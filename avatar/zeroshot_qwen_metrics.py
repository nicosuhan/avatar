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
OUT = ROOT / "outputs"
RESULTS = ROOT / "results" / "zeroshot_qwen_metrics"
TMP = RESULTS / "tmp"
BATCH_SIZE = 12
WORKERS = min(max(os.cpu_count() or 1, 1), 8)
TIMEOUT = 10

TASKS = [
    ("java2python", "python", OUT / "qwen25_coder_7b_hf_zeroshot_java2python"),
    ("python2java", "java", OUT / "qwen25_coder_7b_hf_zeroshot_python2java"),
]


def now():
    return datetime.now(timezone.utc).isoformat()


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
    records = {}
    if not path.exists():
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                records[int(row["index"])] = row
            except Exception:
                pass
    return records


def batch_path(batch_dir, name, start, total):
    end = min(start + BATCH_SIZE, total)
    return batch_dir / f"{name}_batch_{start:04d}_{end - 1:04d}.pred"


def load_predictions(name, batch_dir, total):
    missing = []
    predictions = []
    for start in range(0, total, BATCH_SIZE):
        path = batch_path(batch_dir, name, start, total)
        if not path.exists():
            missing.append(path.name)
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        expected = min(BATCH_SIZE, total - start)
        if len(lines) != expected:
            raise RuntimeError(f"{path} has {len(lines)} lines, expected {expected}")
        predictions.extend(lines)
    if missing:
        raise RuntimeError(f"{name} is missing {len(missing)} batch files: {', '.join(missing[:10])}")
    return predictions


def short(text, limit=4000):
    text = (text or "").strip()
    return text[:limit]


def compile_one(index, problem_id, prediction, lang):
    if lang == "java" and shutil.which("javac") is None:
        return {"index": index, "id": problem_id, "status": "error", "ok": False, "error_type": "missing_javac"}
    code = detokenize(prediction, lang)
    try:
        with tempfile.TemporaryDirectory(prefix="compile_", dir=str(TMP)) as work_dir:
            proc, _, _ = python_compile(code, work_dir, EXT_LIB_DIR) if lang == "python" else java_compile(code, work_dir, EXT_LIB_DIR)
        ok = proc.returncode == 0
        return {
            "index": index,
            "id": problem_id,
            "status": "compiled" if ok else "compile_error",
            "ok": ok,
            "stderr": short(proc.stderr),
        }
    except Exception as exc:
        return {"index": index, "id": problem_id, "status": "error", "ok": False, "error_type": type(exc).__name__, "stderr": short(str(exc))}


def execute_one(index, problem_id, prediction, lang, testcases):
    if lang == "java" and shutil.which("javac") is None:
        return {"index": index, "id": problem_id, "status": "error", "error_type": "missing_javac", "stderr": ""}
    code = detokenize(prediction, lang)
    try:
        with tempfile.TemporaryDirectory(prefix="exec_", dir=str(TMP)) as work_dir:
            if lang == "python":
                compile_proc, script_path, env = python_compile(code, work_dir, EXT_LIB_DIR)
                if compile_proc.returncode != 0:
                    return {"index": index, "id": problem_id, "status": "error", "error_type": "compile", "stderr": short(compile_proc.stderr)}
                command = [sys.executable, str(script_path)]
                run_env = env
                classpath = None
            else:
                compile_proc, class_name, classpath = java_compile(code, work_dir, EXT_LIB_DIR)
                if compile_proc.returncode != 0:
                    return {"index": index, "id": problem_id, "status": "error", "error_type": "compile", "stderr": short(compile_proc.stderr)}
                command = ["java", "-cp", classpath, class_name]
                run_env = None
            for case_index, (input_text, expected_output) in enumerate(testcases):
                proc = subprocess.run(command, input=input_text, capture_output=True, text=True, timeout=TIMEOUT, env=run_env)
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
    except subprocess.TimeoutExpired as exc:
        return {"index": index, "id": problem_id, "status": "error", "error_type": "runtime", "stderr": short(str(exc))}
    except Exception as exc:
        return {"index": index, "id": problem_id, "status": "error", "error_type": type(exc).__name__, "stderr": short(str(exc))}


def summarize(direction, total, testcase_total, text, compile_records, exec_records):
    compiled = sum(1 for row in compile_records.values() if row.get("ok"))
    compile_done = len(compile_records)
    success = sum(1 for row in exec_records.values() if row.get("status") == "success")
    failure = sum(1 for row in exec_records.values() if row.get("status") == "failure")
    error = sum(1 for row in exec_records.values() if row.get("status") == "error")
    compile_error = sum(1 for row in exec_records.values() if row.get("error_type") == "compile")
    runtime_error = sum(1 for row in exec_records.values() if row.get("error_type") == "runtime")
    exec_done = len(exec_records)
    summary = {
        "direction": direction,
        "updated_at": now(),
        "total_predictions": total,
        "expected_testcase_total": testcase_total,
        **text,
        "compile_done": compile_done,
        "compile_total": total,
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
        "complete": compile_done == total and exec_done == testcase_total,
    }
    return summary


def save_progress(out_dir, summary, stage, done, expected):
    write_json(out_dir / "summary.json", summary)
    append_jsonl(out_dir / "progress.jsonl", {"time": now(), "stage": stage, "done": done, "expected": expected, "summary": summary})
    print(f"{summary['direction']} {stage} {done}/{expected} compile={summary['compilation_accuracy']} exec={summary['execution_accuracy']} ca={summary['computational_accuracy']}", flush=True)


def run_stage(out_dir, stage, records, items, worker, summary_args):
    direction, total, testcase_total, text, saved_compile, saved_exec = summary_args
    pending = [item for item in items if item[0] not in records]
    expected = len(items)
    print(f"{direction} {stage} resume: {len(records)}/{expected} already saved, {len(pending)} pending", flush=True)
    compile_records = records if stage == "compile" else saved_compile
    exec_records = records if stage == "execution" else saved_exec
    save_progress(out_dir, summarize(direction, total, testcase_total, text, compile_records, exec_records), stage, expected - len(pending), expected)
    if not pending:
        return records
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(worker, *item): item[0] for item in pending}
        for future in as_completed(futures):
            row = future.result()
            records[int(row["index"])] = row
            append_jsonl(out_dir / f"{stage}.jsonl", row)
            compile_records = records if stage == "compile" else saved_compile
            exec_records = records if stage == "execution" else saved_exec
            summary = summarize(direction, total, testcase_total, text, compile_records, exec_records)
            save_progress(out_dir, summary, stage, len(records), expected)
    return records


def run_direction(name, lang, batch_dir, ids, testcase_map):
    out_dir = RESULTS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    predictions = load_predictions(name, batch_dir, len(ids))
    (out_dir / "test.pred").write_text("\n".join(predictions) + "\n", encoding="utf-8")
    reference_sets = load_reference_sets(ids, DATA_DIR / "test.jsonl", lang)
    text_path = out_dir / "text_metrics.json"
    if text_path.exists():
        text = json.loads(text_path.read_text(encoding="utf-8"))
    else:
        print(f"{name} text metrics start", flush=True)
        text = text_metrics(reference_sets, predictions, lang)
        write_json(text_path, text)
        print(f"{name} text metrics saved: bleu={text['bleu']} codebleu={text['codebleu']}", flush=True)

    compile_records = load_jsonl(out_dir / "compile.jsonl")
    exec_records = load_jsonl(out_dir / "execution.jsonl")
    execution_items = [
        (i, problem_id, predictions[i], lang, testcase_map[problem_id])
        for i, problem_id in enumerate(ids)
        if problem_id in testcase_map
    ]
    summary_args = (name, len(predictions), len(execution_items), text, compile_records, exec_records)
    compile_items = [(i, problem_id, predictions[i], lang) for i, problem_id in enumerate(ids)]
    compile_records = run_stage(out_dir, "compile", compile_records, compile_items, compile_one, summary_args)
    summary_args = (name, len(predictions), len(execution_items), text, compile_records, exec_records)
    exec_records = run_stage(out_dir, "execution", exec_records, execution_items, execute_one, summary_args)
    summary = summarize(name, len(predictions), len(execution_items), text, compile_records, exec_records)
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    ids = read_lines(DATA_DIR / "test.java-python.id")
    testcase_map = load_testcases(DATA_DIR)
    summaries = {}
    for name, lang, batch_dir in TASKS:
        summaries[name] = run_direction(name, lang, batch_dir, ids, testcase_map)
        write_json(RESULTS / "all_summaries.json", summaries)
    if TMP.exists():
        shutil.rmtree(TMP, ignore_errors=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
