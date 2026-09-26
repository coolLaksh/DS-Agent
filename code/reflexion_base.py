#!/usr/bin/env python3
"""DABstep Reflexion agent using Azure OpenAI gpt-4o-mini.

Reflexion = ReAct + per-task episodic memory:

    Task i -> ReAct attempt (think -> act -> observe) -> Evaluator (pass/fail)
                   ^                                          |
                   |                                   pass -> next task
             retry with note <- Reflect + add note <- fail          (memory cleared)

The Evaluator combines two signals:
  1. Finish signal  - how the ReAct loop terminated (programmatic, no LLM).
  2. LLM-as-Judge   - scores the answer/trace against the documented failure
                      modes (see Docs/react-agent-failure-analysis.md).

Memory is per-task: reflection notes accumulate across attempts on the SAME
task and are wiped before the next task.

Usage:
    python3 reflexion_base.py --split dev
    python3 reflexion_base.py --split all --max-attempts 3 --concurrency 4
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
formatted EXACTLY as the guidelines specify. Never call `final_answer` with an
empty string. Only answer "Not Applicable" after you've genuinely exhausted the
ways to answer from the available data.
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
            "description": "Submit the final answer to the question. Must be non-empty.",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string", "description": "The final answer, formatted per the guidelines"}},
                "required": ["answer"],
            },
        },
    },
]

JUDGE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "submit_evaluation",
            "description": "Submit the evaluation verdict for the agent's attempt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "verdict": {"type": "string", "enum": ["pass", "fail"]},
                    "failure_mode": {
                        "type": "string",
                        "enum": [
                            "none",
                            "blank_answer",
                            "unjustified_not_applicable",
                            "format_violation",
                            "unsupported_by_evidence",
                            "silent_empty_result",
                            "incomplete_analysis",
                        ],
                    },
                    "reason": {"type": "string", "description": "One or two sentences explaining the verdict."},
                },
                "required": ["verdict", "failure_mode", "reason"],
            },
        },
    },
]

MAX_TOOL_OUTPUT_CHARS = 4000
MAX_RATE_LIMIT_RETRIES = 6


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


# --- 1. ReAct attempt (think -> act -> observe) ---

@observe(name="react_attempt", as_type="chain")
def react_attempt(task: dict, max_steps: int, temperature: float, notes: list[str]) -> dict:
    """One ReAct episode. Returns the answer, how it finished, and its trace."""
    namespace = {"pd": pd, "os": os, "json": json, "CTX_DIR": str(CONTEXT_DIR)}

    user_content = f"Question: {task['question']}\nGuidelines: {task['guidelines']}"
    if notes:
        memory = "\n".join(f"- {n}" for n in notes)
        user_content += (
            f"\n\nYou have attempted this task before and failed. Lessons from your previous attempts:\n{memory}\n"
            "Apply these lessons. Do not repeat the same mistakes."
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    trace = []
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
            return {"answer": msg.content or "", "finish_reason": "plain_content", "trace": trace}

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        })

        for tool_call in msg.tool_calls:
            args = json.loads(tool_call.function.arguments or "{}")

            if tool_call.function.name == "final_answer":
                answer = str(args.get("answer", ""))
                finish = "final_answer_empty" if not answer.strip() else "final_answer"
                return {"answer": answer, "finish_reason": finish, "trace": trace}

            if tool_call.function.name == "run_python_code":
                code = args.get("code", "")
                result = run_code(code, namespace)
                trace.append({"code": code, "output": result})
            else:
                result = f"Unknown tool: {tool_call.function.name}"

            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

    return {"answer": "", "finish_reason": "max_steps_exhausted", "trace": trace}


# --- 2. Evaluator (signal 1: finish reason, signal 2: LLM-as-judge) ---

def finish_signal(attempt: dict) -> dict:
    """Signal 1 — purely programmatic, derived from how the ReAct loop ended."""
    reason = attempt["finish_reason"]
    answer = attempt["answer"].strip()

    if reason == "final_answer_empty" or not answer:
        return {"status": "fail", "detail": "blank_answer: final_answer called with an empty string"}
    if reason == "max_steps_exhausted":
        return {"status": "fail", "detail": "max_steps_exhausted: ran out of steps without calling final_answer"}
    if answer.lower() == "not applicable":
        return {"status": "suspect", "detail": "answered 'Not Applicable' — needs the judge to confirm it is justified"}
    if reason == "plain_content":
        return {"status": "suspect", "detail": "ended with plain text instead of calling final_answer"}
    return {"status": "pass", "detail": "final_answer called with a non-empty answer"}


def condense_trace(trace: list[dict], max_steps: int = 6, max_chars: int = 700) -> str:
    if not trace:
        return "(no code was executed)"
    shown = trace[-max_steps:]
    parts = []
    for i, step in enumerate(shown, start=len(trace) - len(shown) + 1):
        parts.append(f"[step {i}] CODE:\n{step['code'][:max_chars]}\nOUTPUT:\n{step['output'][:max_chars]}")
    return "\n\n".join(parts)


JUDGE_SYSTEM_PROMPT = """You are a strict evaluator of a data-analysis agent's attempt at a question.

You are given the question, its formatting guidelines, the agent's final answer, and a
condensed trace of the code it ran. Decide whether the attempt should PASS or FAIL.

FAIL the attempt if any of these failure modes apply:
- blank_answer: the answer is empty or whitespace.
- unjustified_not_applicable: the answer is "Not Applicable" but the trace shows the agent
  never genuinely exhausted the data (e.g. it stopped early, or its filter returned nothing
  because of a bug rather than because the data truly has no such rows).
- format_violation: the answer does not follow the guidelines exactly (e.g. guidelines say
  "just the country code" but the answer is "A. NL"; a required rounding/precision is missing;
  extra prose or labels are included).
- unsupported_by_evidence: the trace does not actually compute what the answer claims, or the
  answer does not follow from the printed output.
- silent_empty_result: the agent drew a conclusion from an empty DataFrame / empty result that
  was most likely caused by its own buggy filter. Watch for type mismatches such as comparing
  the int column `year` to the string '2023', or comparing the bool columns `is_credit` /
  `has_fraudulent_dispute` / `is_refused_by_adyen` to strings like 'T', 'Y' or 'True'.
- incomplete_analysis: the agent clearly stopped mid-investigation.

PASS only if the answer is well-formed, follows the guidelines, and is supported by the trace.
You are judging the process and the format, not recomputing the ground truth yourself.

Call `submit_evaluation` with your verdict."""


@observe(name="llm_judge", as_type="evaluator")
def llm_judge(task: dict, attempt: dict, temperature: float) -> dict:
    """Signal 2 — LLM-as-judge scored against the documented failure modes."""
    user_content = (
        f"QUESTION:\n{task['question']}\n\n"
        f"GUIDELINES:\n{task['guidelines']}\n\n"
        f"AGENT'S FINAL ANSWER:\n{attempt['answer']!r}\n\n"
        f"HOW THE ATTEMPT ENDED: {attempt['finish_reason']}\n\n"
        f"CONDENSED TRACE:\n{condense_trace(attempt['trace'])}"
    )

    response = create_completion_with_backoff(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        tools=JUDGE_TOOLS,
        tool_choice={"type": "function", "function": {"name": "submit_evaluation"}},
        max_tokens=1024,
        temperature=temperature,
    )
    msg = response.choices[0].message
    if not msg.tool_calls:
        return {"verdict": "pass", "failure_mode": "none", "reason": "judge returned no verdict; defaulting to pass"}

    args = json.loads(msg.tool_calls[0].function.arguments or "{}")
    return {
        "verdict": args.get("verdict", "pass"),
        "failure_mode": args.get("failure_mode", "none"),
        "reason": args.get("reason", ""),
    }


@observe(name="evaluator", as_type="evaluator")
def evaluate_attempt(task: dict, attempt: dict, temperature: float) -> dict:
    """Combines both signals. Passes only if neither signal objects."""
    signal_1 = finish_signal(attempt)

    # A hard finish-level failure (blank / exhausted) needs no judge call.
    if signal_1["status"] == "fail":
        evaluation = {
            "passed": False,
            "finish_signal": signal_1,
            "judge": {"verdict": "skipped", "failure_mode": "none", "reason": "short-circuited by finish signal"},
        }
    else:
        judge = llm_judge(task, attempt, temperature)
        evaluation = {
            "passed": judge["verdict"] == "pass",
            "finish_signal": signal_1,
            "judge": judge,
        }

    langfuse_client.update_current_span(output=evaluation)
    return evaluation


# --- 3. Reflect (why did it fail? -> note appended to per-task memory) ---

REFLECT_SYSTEM_PROMPT = """You are the reflection step of a Reflexion agent.

The agent just failed an attempt at a data-analysis task. Given the question, the agent's
answer, the evaluator's verdict, and the code it ran, write ONE short, concrete, actionable
lesson (max 3 sentences) that will be given to the agent on its next attempt.

Be specific and technical. Good lessons name the exact mistake and the exact fix, e.g.:
"`year` is int64, not a string — filter with `df['year'] == 2023`, not `'2023'`, which
silently returns an empty DataFrame."

Bad lessons are vague, e.g. "be more careful" or "check the data".

Write only the lesson text, nothing else."""


@observe(name="reflect", as_type="chain")
def reflect(task: dict, attempt: dict, evaluation: dict, temperature: float) -> str:
    user_content = (
        f"QUESTION:\n{task['question']}\n\n"
        f"GUIDELINES:\n{task['guidelines']}\n\n"
        f"AGENT'S ANSWER:\n{attempt['answer']!r}\n\n"
        f"HOW IT ENDED: {attempt['finish_reason']}\n\n"
        f"EVALUATOR - finish signal: {evaluation['finish_signal']['detail']}\n"
        f"EVALUATOR - judge: {evaluation['judge']['failure_mode']} — {evaluation['judge']['reason']}\n\n"
        f"CODE IT RAN:\n{condense_trace(attempt['trace'])}"
    )

    response = create_completion_with_backoff(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": REFLECT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=300,
        temperature=temperature,
    )
    return (response.choices[0].message.content or "").strip()


# --- Reflexion outer loop ---

def pick_best_answer(attempts: list[dict]) -> str:
    """Never return a blank answer — prefer a real answer, then NA, then a safe default."""
    for a in attempts:
        ans = a["answer"].strip()
        if ans and ans.lower() != "not applicable":
            return ans
    for a in attempts:
        if a["answer"].strip():
            return a["answer"].strip()
    return "Not Applicable"


@observe(name="dabstep_task", as_type="agent")
def run_single_task(task: dict, max_steps: int, temperature: float, max_attempts: int) -> dict:
    langfuse_client.update_current_span(
        name=f"task_{task['task_id']}",
        input={"question": task["question"], "guidelines": task["guidelines"]},
        metadata={"task_id": task["task_id"], "level": task.get("level")},
    )

    notes: list[str] = []        # per-task episodic memory, cleared when this function returns
    attempts: list[dict] = []
    history: list[dict] = []

    for attempt_idx in range(1, max_attempts + 1):
        attempt = react_attempt(task, max_steps, temperature, notes)
        attempts.append(attempt)
        evaluation = evaluate_attempt(task, attempt, temperature)

        history.append({
            "attempt": attempt_idx,
            "answer": attempt["answer"],
            "finish_reason": attempt["finish_reason"],
            "passed": evaluation["passed"],
            "finish_signal": evaluation["finish_signal"]["status"],
            "judge_failure_mode": evaluation["judge"]["failure_mode"],
            "judge_reason": evaluation["judge"]["reason"],
            "n_code_calls": len(attempt["trace"]),
        })

        if evaluation["passed"]:
            return {"answer": attempt["answer"], "attempts": attempt_idx, "history": history, "notes": notes}

        if attempt_idx < max_attempts:
            note = reflect(task, attempt, evaluation, temperature)
            if note:
                notes.append(note)
                history[-1]["reflection"] = note

    return {"answer": pick_best_answer(attempts), "attempts": max_attempts, "history": history, "notes": notes}


# --- Runner ---

def score_answer(predicted: str, expected: str) -> bool:
    return str(predicted).strip().lower() == str(expected).strip().lower()


def append_jsonl(entry: dict, path: Path):
    with WRITE_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def process_task(task: dict, is_dev_data: bool, max_steps: int, temperature: float,
                 max_attempts: int, answers_file: Path, attempts_file: Path):
    try:
        result = run_single_task(task, max_steps, temperature, max_attempts)
    except Exception as e:
        logger.warning(f"Task id: {task['task_id']} FAILED: {e}")
        entry = {"task_id": str(task["task_id"]), "agent_answer": "Not Applicable"}
        if is_dev_data:
            entry["answer"] = task["answer"]
            entry["correct"] = False
            entry["level"] = task.get("level")
        append_jsonl(entry, answers_file)
        append_jsonl({"task_id": str(task["task_id"]), "error": str(e)}, attempts_file)
        return

    answer = result["answer"]
    logger.warning(
        f"Task id: {task['task_id']}\tattempts: {result['attempts']}\t"
        f"Question: {task['question']}\tAnswer: {answer}\n{'=' * 50}"
    )

    # answers.jsonl stays leaderboard-clean: only task_id + agent_answer (+ dev scoring).
    entry = {"task_id": str(task["task_id"]), "agent_answer": str(answer)}
    if is_dev_data:
        entry["answer"] = task["answer"]
        entry["correct"] = score_answer(answer, task["answer"])
        entry["level"] = task.get("level")
    append_jsonl(entry, answers_file)

    # diagnostics go in a separate file
    append_jsonl({
        "task_id": str(task["task_id"]),
        "level": task.get("level"),
        "final_answer": answer,
        "n_attempts": result["attempts"],
        "history": result["history"],
    }, attempts_file)


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
    parser.add_argument("--max-attempts", type=int, default=3, help="Reflexion retries per task")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--tasks-ids", type=str, nargs="+", default=None)
    parser.add_argument("--timestamp", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    logger.warning(f"Starting Reflexion run with arguments: {args}")

    tasks = load_tasks(args.split)
    if args.tasks_ids is not None:
        wanted = set(args.tasks_ids)
        tasks = [t for t in tasks if t["task_id"] in wanted]
    elif args.max_tasks >= 0:
        tasks = tasks[: args.max_tasks]

    runs_dir = ROOT / "Agents" / "runs" / f"reflexion_{DEPLOYMENT.replace('/', '_')}" / args.split
    timestamp = args.timestamp or str(int(time.time()))
    base_filename = runs_dir / timestamp
    base_filename.mkdir(parents=True, exist_ok=True)
    answers_file = base_filename / "answers.jsonl"
    attempts_file = base_filename / "attempts.jsonl"

    done = already_done_ids(answers_file)
    tasks_to_run = [t for t in tasks if t["task_id"] not in done]
    logger.warning(f"Running {len(tasks_to_run)}/{len(tasks)} tasks (skipping {len(done)} already done)")

    with ThreadPoolExecutor(max_workers=args.concurrency) as exe:
        futures = [
            exe.submit(process_task, task, args.split == "dev", args.max_steps,
                       args.temperature, args.max_attempts, answers_file, attempts_file)
            for task in tasks_to_run
        ]
        for f in tqdm(as_completed(futures), total=len(tasks_to_run), desc="Processing tasks"):
            f.result()

    if args.split == "dev":
        rows = [json.loads(l) for l in open(answers_file, encoding="utf-8")]
        correct = sum(r["correct"] for r in rows)
        logger.warning(f"Dev accuracy: {correct}/{len(rows)} = {correct / len(rows):.1%}")

    attempt_rows = [json.loads(l) for l in open(attempts_file, encoding="utf-8")]
    retried = sum(1 for r in attempt_rows if r.get("n_attempts", 1) > 1)
    logger.warning(f"Tasks that needed >1 attempt: {retried}/{len(attempt_rows)}")
    logger.warning(f"All tasks processed. Results in {answers_file}, diagnostics in {attempts_file}")

    langfuse_client.flush()
    logger.warning(f"Traces flushed to Langfuse. View them at {os.environ.get('LANGFUSE_BASE_URL', 'https://cloud.langfuse.com')}")


if __name__ == "__main__":
    main()
