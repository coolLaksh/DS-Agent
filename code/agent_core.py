#!/usr/bin/env python3
"""Shared core for the document-discovery agent arms.

Holds everything the ReAct control (`react_tools.py`) and the Reflexion variant
(`reflexion_tools.py`) have in common, so the two arms differ *only* by the
Reflexion loop. That is what makes attributing any score change valid.

Design rule for every prompt/description in this file: domain-neutral only.
No dataset column names, no dtypes, no domain rules. The agent is meant to
discover those itself from the reference documents.
"""

import ast
import io
import json
import logging
import os
import re
import threading
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI, RateLimitError
from tqdm import tqdm

import fee_core as fc
from build_context import build_context

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "DataSet"
CONTEXT_DIR = (DATASET_DIR / "context").resolve()
TASKS_DIR = DATASET_DIR / "tasks"

ENDPOINT = os.environ["ENDPOINT"]
SUBSCRIPTION_KEY = os.environ["SUBSCRIPTION_KEY"]
API_VERSION = os.environ.get("API_VERSION", "2024-12-01-preview")
DEPLOYMENT = os.environ.get("DEPLOYMENT", "gpt-4o-mini")

client = AzureOpenAI(
    api_version=API_VERSION,
    azure_endpoint=ENDPOINT,
    api_key=SUBSCRIPTION_KEY,
)


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        tqdm.write(self.format(record))


logging.basicConfig(level=logging.WARNING, handlers=[TqdmLoggingHandler()])
logger = logging.getLogger(__name__)

WRITE_LOCK = threading.Lock()

MAX_TOOL_OUTPUT_CHARS = 4000
MAX_DOC_CHARS = 20000
LARGE_FILE_BYTES = 100_000
STRUCTURED_SUFFIXES = {".csv", ".json", ".parquet", ".tsv"}
MAX_RATE_LIMIT_RETRIES = 6

EMPTY_RESULT_WARNING = (
    "\n\n[!] This result is empty. An empty result can mean the data genuinely has no "
    "matching rows, or that the filter never matched - e.g. a dtype or value-encoding "
    "mismatch. Verify the column's dtype and its distinct values before concluding the "
    "data is absent."
)

SYSTEM_PROMPT = f"""You are a data analyst agent answering questions about a dataset.

The working directory is {CONTEXT_DIR}. It contains data files and reference documents;
use `list_documents` to see what is there.

Use the `run_python_code` tool to explore and compute (pandas is preloaded as `pd`, the
working directory path is preloaded as `CTX_DIR`). Work step by step: inspect first, then
compute. Print intermediate results with print() so you can see them in the tool output.

When you have the final answer, call the `final_answer` tool with the answer formatted
EXACTLY as the guidelines specify. Never call `final_answer` with an empty string. Only
answer "Not Applicable" after you've genuinely exhausted the ways to answer.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python_code",
            "description": "Execute Python code for data analysis. pandas is preloaded as `pd`, "
                            "the working directory path is preloaded as `CTX_DIR`. Use print() to see output.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python code to execute"}},
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "List the files available in the working directory, with a one-line size summary for each.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": "Read a reference document by filename. Use it to check definitions, "
                            "conventions, or rules before relying on an assumption about the data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Name of the file to read"},
                    "offset": {"type": "integer", "description": "Line number to start from (default 0)"},
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Submit the final answer to the question. Must be non-empty.",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string", "description": "The final answer, formatted per the guidelines"}},
                "required": ["answer"],
            },
        },
    },
]

DOC_TOOLS = {"list_documents", "read_document"}


# --- Tool: run_python_code ---

def _value_is_empty(value) -> bool:
    """Check the live object, before it is repr()'d, for emptiness.

    Deliberately does NOT treat a scalar 0 as empty - that is a legitimate answer.
    """
    if value is None:
        return False
    try:
        if hasattr(value, "empty"):          # DataFrame / Series
            return bool(value.empty)
        if isinstance(value, (list, dict, set, tuple, frozenset)):
            return len(value) == 0
        if hasattr(value, "__len__") and not isinstance(value, str):
            return len(value) == 0
    except Exception:
        pass
    return False


def _output_looks_empty(output: str) -> bool:
    if "Empty DataFrame" in output or "Series([], " in output:
        return True
    if re.search(r"\[0 rows x \d+ columns\]", output):
        return True
    return output.strip() in ("[]", "{}", "set()")


_CARD_SCHEMES = sorted(fc.IN_RULES["card_scheme"], key=str)
_MERCHANTS = sorted(fc.merchants, key=str)
_MERCHANT_COL_CMP = re.compile(
    r"""\[\s*['"]merchant['"]\s*\]\s*==\s*['"](\w+)['"]     # payments['merchant'] == 'X'
      | \.merchant\s*==\s*['"](\w+)['"]                     # payments.merchant == 'X'
    """, re.VERBOSE)


def _entity_confusion_notes(code: str) -> list:
    """Catch `payments['merchant'] == '<card scheme>'` — a raw pandas filter that bypasses
    rule_matches' merchant= guard entirely and always returns empty, since payments.csv's
    merchant column only ever holds the 5 real merchant names."""
    notes = []
    for m in _MERCHANT_COL_CMP.finditer(code):
        value = m.group(1) or m.group(2)
        hit = next((s for s in _CARD_SCHEMES if s.lower() == value.lower()), None)
        if hit and hit not in _MERCHANTS:
            notes.append(
                f"payments['merchant'] == '{value}' will always be empty: '{value}' is a "
                f"card_scheme value, not a merchant. The 5 real merchants are {_MERCHANTS}. "
                f"If the question names a card scheme (\"on {value}\"), it is describing "
                f"the card_scheme column, not filtering by merchant at all — most such "
                f"questions (e.g. \"most expensive ACI on {value}\") do not involve "
                f"payments.csv or a merchant in the first place.")
    return notes


def run_code(code: str, namespace: dict) -> str:
    buf = io.StringIO()
    saw_empty_value = False
    fc.drain_notices()          # discard anything left over from a prior execution

    def local_print(*args, **kwargs):
        kwargs["file"] = buf
        print(*args, **kwargs)

    namespace["print"] = local_print

    try:
        tree = ast.parse(code, mode="exec")
        trailing_expr = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            trailing_expr = ast.Expression(tree.body.pop().value)

        exec(compile(tree, "<agent_code>", "exec"), namespace)
        if trailing_expr is not None:
            value = eval(compile(trailing_expr, "<agent_code>", "eval"), namespace)
            saw_empty_value = _value_is_empty(value)
            if value is not None:
                local_print(repr(value))
    except Exception as e:
        output = f"{buf.getvalue()}\nError: {e}"
    else:
        output = buf.getvalue()

    output = output.strip() or "(no output - use print() to see values)"
    if len(output) > MAX_TOOL_OUTPUT_CHARS:
        output = output[:MAX_TOOL_OUTPUT_CHARS] + "\n... (truncated)"

    if "Error:" not in output and (saw_empty_value or _output_looks_empty(output)):
        output += EMPTY_RESULT_WARNING

    notices = fc.drain_notices() + _entity_confusion_notes(code)
    if notices:
        output += "\n\nNote:\n" + "\n".join(f"- {n}" for n in notices)

    return output


# --- Tools: list_documents / read_document ---

def list_documents_text() -> str:
    """Plain listing, reusable outside a tool span (e.g. by an evaluator)."""
    rows = []
    for p in sorted(CONTEXT_DIR.iterdir()):
        if not p.is_file():
            continue
        size = p.stat().st_size
        if size < 2_000_000:
            try:
                with p.open(encoding="utf-8", errors="replace") as f:
                    n_lines = sum(1 for _ in f)
                rows.append(f"{p.name}  ({size:,} bytes, {n_lines:,} lines)")
            except Exception:
                rows.append(f"{p.name}  ({size:,} bytes)")
        else:
            rows.append(f"{p.name}  ({size:,} bytes)")
    return "\n".join(rows)


def list_documents() -> str:
    output = list_documents_text()
    return output


def read_document(filename: str, offset: int = 0) -> str:

    try:
        target = (CONTEXT_DIR / filename).resolve()
        target.relative_to(CONTEXT_DIR)          # raises if outside the working directory
    except (ValueError, OSError):
        output = f"Error: '{filename}' is outside the working directory."
        return output

    if not target.is_file():
        output = f"Error: '{filename}' not found. Use list_documents to see what is available."
        return output

    with target.open(encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    total = len(lines)

    # Large structured files: paging through as text would burn the step budget - preview + redirect instead.
    if target.suffix.lower() in STRUCTURED_SUFFIXES and target.stat().st_size > LARGE_FILE_BYTES:
        preview = "".join(lines[:40])[:MAX_DOC_CHARS]
        output = (
            f"'{filename}' is a large structured data file ({total:,} lines, "
            f"{target.stat().st_size:,} bytes). Reading it as text is usually the wrong "
            f"approach - query it with run_python_code instead. First 40 lines as a preview:"
            f"\n\n{preview}"
        )
        return output
    offset = max(0, min(offset, total))
    chunk, chars = [], 0
    for i in range(offset, total):
        chars += len(lines[i])
        if chars > MAX_DOC_CHARS and chunk:
            break
        chunk.append(lines[i])

    end = offset + len(chunk)
    body = "".join(chunk)
    if end < total:
        footer = f"\n\n--- showing lines {offset}-{end} of {total}. Call read_document(filename='{filename}', offset={end}) for more. ---"
    else:
        footer = f"\n\n--- end of file (lines {offset}-{end} of {total}) ---"

    output = body + footer
    return output


# --- LLM call with backoff ---

def create_completion_with_backoff(**kwargs):
    for attempt in range(MAX_RATE_LIMIT_RETRIES):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError:
            if attempt == MAX_RATE_LIMIT_RETRIES - 1:
                raise
            wait = min(2 ** attempt, 60)
            logger.warning(f"Rate limited, retrying in {wait}s (attempt {attempt + 1}/{MAX_RATE_LIMIT_RETRIES})")
            time.sleep(wait)


# --- ReAct episode - shared verbatim by both arms ---

def react_attempt(task: dict, max_steps: int, temperature: float, notes: list[str] | None = None,
                  use_context: bool = True) -> dict:
    """One ReAct episode: think -> act -> observe.

    `notes` carries Reflexion memory; the ReAct control arm passes None, which makes
    the two arms byte-identical up to that point.

    `use_context` toggles the deterministic context injection from build_context.py -
    keep it as a flag so a with/without comparison stays attributable.
    """
    namespace = {"pd": pd, "os": os, "json": json, "CTX_DIR": str(CONTEXT_DIR)}

    context_text = ""
    if use_context:
        context_text, ns_extra = build_context(task)
        namespace.update(ns_extra)

    user_content = f"Question: {task['question']}\nGuidelines: {task['guidelines']}"
    if context_text:
        user_content = f"{context_text}\n\n---\n\n{user_content}"
    if notes:
        memory = "\n\n".join(f"--- Attempt {i} ---\n{n}" for i, n in enumerate(notes, start=1))
        user_content += (
            f"\n\nYou have attempted this task before. Notes from previous attempts:\n{memory}\n\n"
            "Reuse anything marked VERIFIED above instead of recomputing it from scratch. "
            "Do not repeat any mistake marked MISTAKE - that reasoning path was already tried and failed."
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    trace = []
    documents_read = []
    computations = []          # fee_core._record_computation output - lets a check verify the
                                # answer against the helper's own result, not by re-parsing stdout
    n_empty_results = 0
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0}

    def finish(answer, reason):
        return {
            "answer": answer,
            "finish_reason": reason,
            "trace": trace,
            "documents_read": documents_read,
            "computations": computations,
            "n_empty_results": n_empty_results,
            "n_code_calls": sum(1 for t in trace if t["tool"] == "run_python_code"),
            "usage": usage,
            "used_context": bool(context_text),
        }

    for _ in range(max_steps):
        response = create_completion_with_backoff(
            model=DEPLOYMENT,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=4096,
            temperature=temperature,
            top_p=1.0,
        )
        if getattr(response, "usage", None):
            usage["prompt_tokens"] += response.usage.prompt_tokens or 0
            usage["completion_tokens"] += response.usage.completion_tokens or 0
        usage["llm_calls"] += 1

        msg = response.choices[0].message

        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or ""})
            return finish(msg.content or "", "plain_content")

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        })

        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            raw_args = tool_call.function.arguments or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError as e:
                # Recover instead of crashing the whole attempt: feed the error back as a tool
                # result and let the agent retry - costs one step, not the entire task.
                result = (f"Error: your arguments for {name!r} were not valid JSON ({e}). This "
                          f"usually happens when a long `code` string contains an unescaped quote, "
                          f"backslash, or literal newline inside the JSON string value. Re-issue "
                          f"the call with valid JSON.")
                trace.append({"tool": name, "input": raw_args[:500], "output": result})
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
                continue

            if name == "final_answer":
                answer = str(args.get("answer", ""))
                return finish(answer, "final_answer_empty" if not answer.strip() else "final_answer")

            if name == "run_python_code":
                code = args.get("code", "")
                result = run_code(code, namespace)
                if EMPTY_RESULT_WARNING.strip()[:20] in result:
                    n_empty_results += 1
                trace.append({"tool": name, "input": code, "output": result})
                computations.extend(fc.drain_computations())

            elif name == "list_documents":
                result = list_documents()
                trace.append({"tool": name, "input": "", "output": result})

            elif name == "read_document":
                fname = args.get("filename", "")
                result = read_document(fname, int(args.get("offset", 0) or 0))
                if not result.startswith("Error:") and fname not in documents_read:
                    documents_read.append(fname)
                trace.append({"tool": name, "input": fname, "output": result})

            else:
                result = f"Unknown tool: {name}"
                trace.append({"tool": name, "input": "", "output": result})

            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

    return finish("", "max_steps_exhausted")


def condense_trace(trace: list[dict], max_steps: int = 6, max_chars: int = 700) -> str:
    """Condense a trace for the judge.

    Document reads are ALWAYS retained even if they fall outside the recency window -
    otherwise the judge cannot tell whether the agent consulted anything.
    """
    if not trace:
        return "(no tool calls)"

    keep_idx = {i for i, s in enumerate(trace) if s["tool"] in DOC_TOOLS}
    keep_idx |= set(range(max(0, len(trace) - max_steps), len(trace)))

    parts = []
    for i in sorted(keep_idx):
        s = trace[i]
        head = f"[step {i + 1}] {s['tool']}"
        if s["input"]:
            head += f"\nINPUT:\n{str(s['input'])[:max_chars]}"
        parts.append(f"{head}\nOUTPUT:\n{s['output'][:max_chars]}")
    return "\n\n".join(parts)


# --- Runner helpers ---

def score_answer(predicted: str, expected: str) -> bool:
    return str(predicted).strip().lower() == str(expected).strip().lower()


def append_jsonl(entry: dict, path: Path):
    with WRITE_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def load_tasks(split: str) -> list[dict]:
    path = TASKS_DIR / f"{split}.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def already_done_ids(answers_file: Path) -> set[str]:
    if not answers_file.exists():
        return set()
    with open(answers_file, encoding="utf-8") as f:
        return {json.loads(line)["task_id"] for line in f if line.strip()}


def select_tasks(tasks: list[dict], tasks_ids, max_tasks: int) -> list[dict]:
    if tasks_ids is not None:
        # tolerate a single space-separated string (zsh does not word-split unquoted vars)
        wanted = {i for chunk in tasks_ids for i in str(chunk).split()}
        return [t for t in tasks if t["task_id"] in wanted]
    if max_tasks >= 0:
        return tasks[:max_tasks]
    return tasks


def prepare_run_dir(agent_name: str, split: str, timestamp: str | None) -> Path:
    runs_dir = ROOT / "runs" / f"{agent_name}_{DEPLOYMENT.replace('/', '_')}" / split
    base = runs_dir / (timestamp or str(int(time.time())))
    base.mkdir(parents=True, exist_ok=True)
    return base
