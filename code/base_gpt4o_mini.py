#!/usr/bin/env python3
"""DABstep baseline agent using Azure OpenAI gpt-4o-mini.

Runs a simple ReAct-style code-execution agent (LLM writes pandas/python code
via tool calls, we execute it and feed back the output) against the local
DABstep dataset in ../DataSet.

Usage:
    python3 base_gpt4o_mini.py --split dev
    python3 base_gpt4o_mini.py --split all --max-tasks 20 --concurrency 4
"""

import argparse
import ast
import io
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langfuse import Langfuse, observe
from langfuse.openai import AzureOpenAI
from openai import RateLimitError
from tqdm import tqdm

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "DataSet"
CONTEXT_DIR = DATASET_DIR / "context"
TASKS_DIR = DATASET_DIR / "tasks"

ENDPOINT = os.environ["ENDPOINT"]
SUBSCRIPTION_KEY = os.environ["SUBSCRIPTION_KEY"]
API_VERSION = os.environ.get("API_VERSION", "2024-12-01-preview")
DEPLOYMENT = os.environ.get("DEPLOYMENT", "gpt-4o-mini")

# Reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL from the environment.
langfuse_client = Langfuse()

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

SYSTEM_PROMPT = f"""You are a data analyst agent answering questions about payment transaction data.

All the data you need lives in this directory: {CONTEXT_DIR}
Files available there:
- payments.csv           (transaction-level data, 138,236 rows)
- payments-readme.md     (column definitions for payments.csv)
- merchant_data.json     (merchant metadata)
- merchant_category_codes.csv
- acquirer_countries.csv
- fees.json              (fee rule definitions)
- manual.md              (fee-calculation rules handbook)

Use the `run_python_code` tool to explore and compute (pandas is preloaded as `pd`,
the context directory path is preloaded as `CTX_DIR`). Always read the relevant
files before assuming something is unavailable or answering "Not Applicable".
Work step by step: inspect first, then compute. Print intermediate results with
print() so you can see them in the tool output.

When you have the final answer, call the `final_answer` tool with the answer
formatted EXACTLY as the guidelines specify. Only answer "Not Applicable" after
you've genuinely exhausted the ways to answer from the available data.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python_code",
            "description": "Execute Python code for data analysis. pandas is preloaded as `pd`, "
                            "the context directory path is preloaded as `CTX_DIR`. Use print() to see output.",
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
            "name": "final_answer",
            "description": "Submit the final answer to the question.",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string", "description": "The final answer, formatted per the guidelines"}},
                "required": ["answer"],
            },
        },
    },
]

MAX_TOOL_OUTPUT_CHARS = 4000


@observe(name="run_python_code", as_type="tool", capture_input=False, capture_output=False)
def run_code(code: str, namespace: dict) -> str:
    langfuse_client.update_current_span(input=code)
    buf = io.StringIO()

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
            if value is not None:
                local_print(repr(value))
    except Exception as e:
        output = f"{buf.getvalue()}\nError: {e}"
    else:
        output = buf.getvalue()
    output = output.strip() or "(no output — use print() to see values)"
    if len(output) > MAX_TOOL_OUTPUT_CHARS:
        output = output[:MAX_TOOL_OUTPUT_CHARS] + "\n... (truncated)"
    langfuse_client.update_current_span(output=output)
    return output


MAX_RATE_LIMIT_RETRIES = 6


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


@observe(name="dabstep_task", as_type="agent")
def run_single_task(task: dict, max_steps: int, temperature: float) -> str:
    langfuse_client.update_current_span(
        name=f"task_{task['task_id']}",
        input={"question": task["question"], "guidelines": task["guidelines"]},
        metadata={"task_id": task["task_id"], "level": task.get("level")},
    )
    namespace = {"pd": pd, "os": os, "json": json, "CTX_DIR": str(CONTEXT_DIR)}
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {task['question']}\nGuidelines: {task['guidelines']}"},
    ]

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
        msg = response.choices[0].message

        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or ""})
            return msg.content or "Not Applicable"

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        })

        for tool_call in msg.tool_calls:
            args = json.loads(tool_call.function.arguments or "{}")
            if tool_call.function.name == "final_answer":
                return str(args.get("answer", "Not Applicable"))

            if tool_call.function.name == "run_python_code":
                result = run_code(args.get("code", ""), namespace)
            else:
                result = f"Unknown tool: {tool_call.function.name}"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    return "Not Applicable"


def score_answer(predicted: str, expected: str) -> bool:
    return str(predicted).strip().lower() == str(expected).strip().lower()


def append_jsonl(entry: dict, path: Path):
    with WRITE_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def process_task(task: dict, is_dev_data: bool, max_steps: int, temperature: float, answers_file: Path):
    try:
        answer = run_single_task(task, max_steps, temperature)
    except Exception as e:
        logger.warning(f"Task id: {task['task_id']} FAILED: {e}")
        entry = {"task_id": str(task["task_id"]), "agent_answer": "", "error": str(e)}
        if is_dev_data:
            entry["answer"] = task["answer"]
            entry["correct"] = False
            entry["level"] = task.get("level")
        append_jsonl(entry, answers_file)
        return

    logger.warning(f"Task id: {task['task_id']}\tQuestion: {task['question']}\tAnswer: {answer}\n{'=' * 50}")

    entry = {"task_id": str(task["task_id"]), "agent_answer": str(answer)}
    if is_dev_data:
        entry["answer"] = task["answer"]
        entry["correct"] = score_answer(answer, task["answer"])
        entry["level"] = task.get("level")
    append_jsonl(entry, answers_file)


def load_tasks(split: str) -> list[dict]:
    path = TASKS_DIR / f"{split}.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def already_done_ids(answers_file: Path) -> set[str]:
    if not answers_file.exists():
        return set()
    with open(answers_file, encoding="utf-8") as f:
        return {json.loads(line)["task_id"] for line in f if line.strip()}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="dev", choices=["all", "dev"])
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tasks", type=int, default=-1)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--tasks-ids", type=str, nargs="+", default=None)
    parser.add_argument("--timestamp", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    logger.warning(f"Starting run with arguments: {args}")

    tasks = load_tasks(args.split)
    if args.tasks_ids is not None:
        wanted = set(args.tasks_ids)
        tasks = [t for t in tasks if t["task_id"] in wanted]
    elif args.max_tasks >= 0:
        tasks = tasks[: args.max_tasks]

    runs_dir = ROOT / "Agents" / "runs" / DEPLOYMENT.replace("/", "_") / args.split
    timestamp = args.timestamp or str(int(time.time()))
    base_filename = runs_dir / timestamp
    base_filename.mkdir(parents=True, exist_ok=True)
    answers_file = base_filename / "answers.jsonl"

    done = already_done_ids(answers_file)
    tasks_to_run = [t for t in tasks if t["task_id"] not in done]
    logger.warning(f"Running {len(tasks_to_run)}/{len(tasks)} tasks (skipping {len(done)} already done)")

    with ThreadPoolExecutor(max_workers=args.concurrency) as exe:
        futures = [
            exe.submit(process_task, task, args.split == "dev", args.max_steps, args.temperature, answers_file)
            for task in tasks_to_run
        ]
        for f in tqdm(as_completed(futures), total=len(tasks_to_run), desc="Processing tasks"):
            f.result()

    if args.split == "dev":
        rows = [json.loads(l) for l in open(answers_file, encoding="utf-8")]
        correct = sum(r["correct"] for r in rows)
        logger.warning(f"Dev accuracy: {correct}/{len(rows)} = {correct / len(rows):.1%}")

    logger.warning(f"All tasks processed. Results in {answers_file}")

    langfuse_client.flush()
    logger.warning(f"Traces flushed to Langfuse. View them at {os.environ.get('LANGFUSE_BASE_URL', 'https://cloud.langfuse.com')}")


if __name__ == "__main__":
    main()
