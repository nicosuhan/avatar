from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
from urllib import error, request


# Edit these settings, then run:
# python avatar\zeroshot_qwen_api.py
DATA_DIR = Path("avatar/data/avatar")
OUTPUT_ROOT = Path("outputs")
LIMIT = 0  # 0 means full test set.
CONCURRENCY = 2
DRY_RUN = False

# Recommended key: OpenRouter API key, stored in AVATAR_API_KEY or OPENROUTER_API_KEY.
API_BASE_URL = os.environ.get("AVATAR_API_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY = os.environ.get("AVATAR_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
MODEL = os.environ.get("AVATAR_API_MODEL", "qwen/qwen2.5-coder-7b-instruct")

TEMPERATURE = 0.0
MAX_TOKENS = 4096
TIMEOUT = 120
RETRIES = 5
BACKOFF = 2.0

SYSTEM_PROMPT = (
    "You translate competitive-programming programs between Java and Python. "
    "Return only the translated program. Do not explain. Do not use Markdown."
)

USER_PROMPT = """Translate this AVATAR-tokenized {source_language} program to AVATAR-tokenized {target_language}.

Rules:
- Output one single AVATAR-tokenized program.
- Keep spaces between tokens.
- Use NEW_LINE, INDENT, and DEDENT tokens instead of real indentation.
- Preserve string placeholder tokens such as STRNEWLINE, TABSYMBOL, and \u2581 when needed.
- Do not wrap the answer in code fences.
- Do not add comments or explanations.

Source program:
{source}
"""

DIRECTIONS = (
    ("java2python", "java", "python", "qwen25_coder_7b_zeroshot_java2python"),
    ("python2java", "python", "java", "qwen25_coder_7b_zeroshot_python2java"),
)


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def endpoint_from_base(base_url):
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def is_dry_run_row(row):
    return bool(row.get("dry_run")) or row.get("finish_reason") == "dry_run" or str(row.get("prediction", "")).startswith("DRY_RUN_")


def load_completed(path, dry_run):
    completed = {}
    if not path.exists():
        return completed
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if is_dry_run_row(row) == dry_run:
                completed[int(row["index"])] = row
    return completed


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def clean_prediction(text):
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return " ".join(text.splitlines()).strip()


def make_messages(source_lang, target_lang, source):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT.format(
                source_language=source_lang,
                target_language=target_lang,
                source=source,
            ),
        },
    ]


def post_chat(endpoint, api_key, model, messages, temperature, max_tokens, timeout):
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    ).encode("utf-8")
    req = request.Request(
        endpoint,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def response_content(response):
    choice = response.get("choices", [{}])[0]
    message = choice.get("message") or {}
    if message.get("content") is not None:
        return message.get("content"), choice.get("finish_reason")
    return choice.get("text", ""), choice.get("finish_reason")


def request_with_retries(task, endpoint):
    direction, source_lang, target_lang, index, problem_id, source = task
    messages = make_messages(source_lang, target_lang, source)
    started = time.time()
    if DRY_RUN:
        prediction = f"DRY_RUN_{target_lang}_{index}"
        return {
            "ok": True,
            "direction": direction,
            "index": index,
            "id": problem_id,
            "source": source_lang,
            "target": target_lang,
            "prediction": prediction,
            "finish_reason": "dry_run",
            "usage": {},
            "dry_run": True,
            "elapsed_sec": round(time.time() - started, 3),
        }

    last_error = ""
    for attempt in range(1, RETRIES + 1):
        try:
            response = post_chat(
                endpoint,
                API_KEY,
                MODEL,
                messages,
                TEMPERATURE,
                MAX_TOKENS,
                TIMEOUT,
            )
            content, finish_reason = response_content(response)
            return {
                "ok": True,
                "direction": direction,
                "index": index,
                "id": problem_id,
                "source": source_lang,
                "target": target_lang,
                "prediction": clean_prediction(content),
                "finish_reason": finish_reason,
                "usage": response.get("usage", {}),
                "dry_run": False,
                "elapsed_sec": round(time.time() - started, 3),
            }
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {body[:1000]}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < RETRIES:
            time.sleep(BACKOFF * (2 ** (attempt - 1)))

    return {
        "ok": False,
        "direction": direction,
        "index": index,
        "id": problem_id,
        "source": source_lang,
        "target": target_lang,
        "error": last_error,
        "attempts": RETRIES,
        "dry_run": DRY_RUN,
        "elapsed_sec": round(time.time() - started, 3),
    }


def write_predictions(output_dir, total, completed):
    predictions = []
    for index in range(total):
        row = completed.get(index)
        predictions.append(row["prediction"] if row else "")
    (output_dir / "test.pred").write_text("\n".join(predictions) + "\n", encoding="utf-8")


def write_config(output_dir, endpoint, source, target, total):
    config = {
        "model": MODEL,
        "api_base_url": None if DRY_RUN else API_BASE_URL,
        "endpoint": None if DRY_RUN else endpoint,
        "source": source,
        "target": target,
        "limit": LIMIT,
        "examples": total,
        "concurrency": CONCURRENCY,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "timeout": TIMEOUT,
        "retries": RETRIES,
        "dry_run": DRY_RUN,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def build_direction_state(key, source_lang, target_lang, out_name):
    output_dir = OUTPUT_ROOT / out_name
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = DATA_DIR / f"test.java-python.{source_lang}"
    ids_path = DATA_DIR / "test.java-python.id"
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if not ids_path.exists():
        raise FileNotFoundError(ids_path)
    sources = read_lines(source_path)
    ids = read_lines(ids_path)
    total = min(LIMIT or len(sources), len(sources), len(ids))
    responses_path = output_dir / "responses.jsonl"
    errors_path = output_dir / "errors.jsonl"
    completed = load_completed(responses_path, DRY_RUN)
    return {
        "key": key,
        "source": source_lang,
        "target": target_lang,
        "output_dir": output_dir,
        "responses_path": responses_path,
        "errors_path": errors_path,
        "sources": sources[:total],
        "ids": ids[:total],
        "total": total,
        "completed": {i: row for i, row in completed.items() if i < total},
    }


def main():
    if LIMIT < 0:
        raise ValueError("LIMIT must be >= 0")
    if CONCURRENCY < 1:
        raise ValueError("CONCURRENCY must be >= 1")
    if RETRIES < 1:
        raise ValueError("RETRIES must be >= 1")
    if not DRY_RUN and (not API_BASE_URL or not API_KEY):
        raise ValueError("Set AVATAR_API_KEY or OPENROUTER_API_KEY, or set DRY_RUN = True.")

    endpoint = "" if DRY_RUN else endpoint_from_base(API_BASE_URL)
    states = [build_direction_state(*direction) for direction in DIRECTIONS]
    for state in states:
        write_config(state["output_dir"], endpoint, state["source"], state["target"], state["total"])

    tasks = []
    state_by_key = {state["key"]: state for state in states}
    for state in states:
        for index, source in enumerate(state["sources"]):
            if index in state["completed"]:
                continue
            tasks.append((state["key"], state["source"], state["target"], index, state["ids"][index], source))

    print(json.dumps({"pending": len(tasks), "directions": {s["key"]: s["total"] for s in states}}, indent=2))
    if tasks:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            futures = [pool.submit(request_with_retries, task, endpoint) for task in tasks]
            for future in as_completed(futures):
                row = future.result()
                state = state_by_key[row["direction"]]
                if row["ok"]:
                    row.pop("ok")
                    state["completed"][row["index"]] = row
                    append_jsonl(state["responses_path"], row)
                    done = len(state["completed"])
                    print(f'{row["direction"]} {done}/{state["total"]} index={row["index"]}')
                else:
                    row.pop("ok")
                    append_jsonl(state["errors_path"], row)
                    print(f'{row["direction"]} error index={row["index"]}: {row["error"][:200]}')

    for state in states:
        write_predictions(state["output_dir"], state["total"], state["completed"])
        missing = state["total"] - len(state["completed"])
        print(json.dumps({"direction": state["key"], "predictions": state["total"], "missing": missing}))


if __name__ == "__main__":
    main()
