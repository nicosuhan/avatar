import os, time
from pathlib import Path

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["TRANSFORMERS_NO_TORCHAUDIO"] = "1"
import torch
import transformers.utils.import_utils as hf_import_utils
hf_import_utils._tf_available = False
hf_import_utils._flax_available = False
hf_import_utils._jax_available = False
hf_import_utils._torchvision_available = False
hf_import_utils._sklearn_available = False
hf_import_utils._scipy_available = False
hf_import_utils._torchao_available = False
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DATA_DIR = Path("/kaggle/input/datasets/nicoletacorinasuhan/avatar-dataset/data/avatar")
if not DATA_DIR.exists():
    DATA_DIR = Path("/kaggle/input/avatar-dataset/data/avatar")
OUT = Path("/kaggle/working/outputs")
MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
LIMIT = 0
START_INDEX = 1746
BATCH_SIZE = 12
MAX_NEW_TOKENS = 1024
REPETITION_PENALTY = 1.0
SEED = 0

BASE_PROMPT = """Translate this token-spaced {src_lang} program into normal {tgt_lang} source code.

Important:
- Reconstruct the source program before translating it.
- NEW_LINE means a line break. INDENT and DEDENT describe Python-style block nesting in the source.
- Preserve the algorithm, constants, input/output behavior, and edge cases.
- Write complete runnable {tgt_lang} code.
- Do not copy {src_lang} libraries, methods, wrappers, or syntax into the answer.
- Output normal {tgt_lang} source code, not a token stream.
- Keep hardcoded values hardcoded. Only read input when the source program reads input.
- Mentally check the final code for valid {tgt_lang} syntax before returning it.

Target language: {tgt_lang}
{target_examples}

Return only the translated {tgt_lang} source code. Do not explain. Do not use Markdown.

Source tokenized {src_lang} program:
{code}
"""

PYTHON_EXAMPLES = """Python output requirements:
- Output Python code only.
- Do not use Java syntax such as public, static, void, class Main, Scanner, String[] args, System.out.println, System.out.printf, charAt, toCharArray, or semicolons as statement endings.
- Use Python input(), sys.stdin, print(), sys.stdout.write(), lists, strings, math, and normal Python functions/classes where appropriate.
- Translate Java string methods to Python: s.charAt(i) becomes s[i], s.length() becomes len(s), toCharArray() becomes the same string or list(s).
- Translate Java Math methods to Python math functions or operators.
- If Java uses geometry/library objects, reimplement the math with Python functions, tuples, or complex numbers. Do not call methods like intersects(), distance(), p1, or p2 on tuples.
- Avoid shadowing Python builtins such as len, str, input, sum, max, min, list, dict, and set."""

JAVA_EXAMPLES = """Java output requirements:
- Output Java code only.
- Do not use Python syntax such as def, print(), range(), while True, import sys, list methods, string zfill, or colon-based blocks.
- Use Java Scanner/BufferedReader/StringTokenizer only when the source reads input. Keep hardcoded test values hardcoded.
- Use System.out.print/println/printf, arrays, ArrayList, StringBuilder, Math, braces, and semicolons where appropriate.
- Translate Python ** to Java Math.pow(...) or direct multiplication; never output * *.
- Translate Python zfill(n) to zero padding with String.format(...).replace(' ', '0') or equivalent Java code.
- Initialize boolean sieve arrays correctly, usually with Arrays.fill(prime, true), before marking composites.
- Methods called from main should be static unless they are called through an object."""

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False

def lines(path):
    return path.read_text(encoding="utf-8").splitlines()


def batch_file(out_dir, prefix, start, end):
    return out_dir / f"{prefix}_batch_{start:04d}_{end - 1:04d}.pred"


def next_batch(out_dir, prefix, total):
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        path = batch_file(out_dir, prefix, start, end)
        if not path.exists() or len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]) < end - start:
            return start
    return total


def save_batch(out_dir, prefix, start, preds):
    end = start + len(preds)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_file(out_dir, prefix, start, end).write_text("\n".join(preds) + "\n", encoding="utf-8")


def make_prompt(tokenizer, src_lang, tgt_lang, code):
    target_examples = PYTHON_EXAMPLES if tgt_lang == "python" else JAVA_EXAMPLES
    content = BASE_PROMPT.replace("{src_lang}", src_lang).replace("{tgt_lang}", tgt_lang).replace("{target_examples}", target_examples).replace("{code}", code)
    messages = [
        {"role": "system", "content": f"You translate competitive-programming programs from token-spaced {src_lang} into valid {tgt_lang}. Return only {tgt_lang} source code."},
        {"role": "user", "content": content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_prompts(tokenizer, model, tgt_lang, prompts):
    inputs = output_ids = None
    try:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, pad_to_multiple_of=8).to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=0.0,
                num_beams=1,
                repetition_penalty=REPETITION_PENALTY,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        texts = tokenizer.batch_decode(output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return [postprocess(text, tgt_lang) for text in texts]
    finally:
        if inputs is not None:
            del inputs
        if output_ids is not None:
            del output_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def generate_batch(tokenizer, model, src_lang, tgt_lang, codes):
    prompts = [make_prompt(tokenizer, src_lang, tgt_lang, code) for code in codes]
    try:
        return generate_prompts(tokenizer, model, tgt_lang, prompts)
    except torch.cuda.OutOfMemoryError:
        if len(codes) == 1:
            raise
        mid = len(codes) // 2
        return generate_batch(tokenizer, model, src_lang, tgt_lang, codes[:mid]) + generate_batch(tokenizer, model, src_lang, tgt_lang, codes[mid:])


print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map={"": 0},
    trust_remote_code=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16),
)
model.config.use_cache = True
model.eval()

tasks = []
for name, src_lang, tgt_lang in [("java2python", "java", "python"), ("python2java", "python", "java")]:
    src = lines(DATA_DIR / f"test.java-python.{src_lang}")
    total = min(LIMIT or len(src), len(src))
    out_dir = OUT / f"qwen25_coder_7b_hf_zeroshot_{name}"
    start = START_INDEX if START_INDEX is not None else next_batch(out_dir, name, total)
    print(name, "start", start, "/", total, flush=True)
    tasks.append({"name": name, "src_lang": src_lang, "tgt_lang": tgt_lang, "src": src, "total": total, "out_dir": out_dir, "next": start})

while any(task["next"] < task["total"] for task in tasks):
    for task in tasks:
        start = task["next"]
        if start >= task["total"]:
            continue
        end = min(start + BATCH_SIZE, task["total"])
        t0 = time.time()
        preds = generate_batch(tokenizer, model, task["src_lang"], task["tgt_lang"], task["src"][start:end])
        save_batch(task["out_dir"], task["name"], start, preds)
        task["next"] = end if START_INDEX is not None else next_batch(task["out_dir"], task["name"], task["total"])
        print(task["name"], start, "-", end - 1, "sec", round(time.time() - t0, 3), flush=True)
