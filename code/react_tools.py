#!/usr/bin/env python3
"""ReAct control arm: plain ReAct + document tools + loud empty results.

This is the control for the Reflexion experiment. It shares `react_attempt`
verbatim with `reflexion_tools.py` via `agent_core`, so the only difference
between the two arms is the Reflexion loop itself.

Usage:
    python3 react_tools.py --split dev
    python3 react_tools.py --split all --concurrency 2
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from agent_core import (
    logger, react_attempt, score_answer, append_jsonl,
    load_tasks, already_done_ids, select_tasks, prepare_run_dir,
)

AGENT_NAME = "react_tools"


def run_single_task(task: dict, max_steps: int, temperature: float, use_context: bool) -> dict:
    attempt = react_attempt(task, max_steps, temperature, notes=None, use_context=use_context)
    # A blank answer is never valid on the leaderboard; both arms avoid submitting one.
    answer = attempt["answer"].strip() or "Not Applicable"
    return {"answer": answer, "attempt": attempt}


def process_task(task: dict, is_dev_data: bool, max_steps: int, temperature: float,
                 use_context: bool, answers_file: Path, attempts_file: Path, trace_file: Path):
    try:
        result = run_single_task(task, max_steps, temperature, use_context)
    except Exception as e:
        logger.warning(f"Task id: {task['task_id']} FAILED: {e}")
        entry = {"task_id": str(task["task_id"]), "agent_answer": "Not Applicable"}
        if is_dev_data:
            entry.update({"answer": task["answer"], "correct": False, "level": task.get("level")})
        append_jsonl(entry, answers_file)
        append_jsonl({"task_id": str(task["task_id"]), "error": str(e)}, attempts_file)
        return

    answer, a = result["answer"], result["attempt"]
    logger.warning(
        f"Task id: {task['task_id']}\tdocs: {a['documents_read']}\t"
        f"Question: {task['question']}\tAnswer: {answer}\n{'=' * 50}"
    )

    entry = {"task_id": str(task["task_id"]), "agent_answer": str(answer)}
    if is_dev_data:
        entry.update({
            "answer": task["answer"],
            "correct": score_answer(answer, task["answer"]),
            "level": task.get("level"),
        })
    append_jsonl(entry, answers_file)

    append_jsonl({
        "task_id": str(task["task_id"]),
        "level": task.get("level"),
        "final_answer": answer,
        "finish_reason": a["finish_reason"],
        "documents_read": a["documents_read"],
        "n_code_calls": a["n_code_calls"],
        "n_empty_results": a["n_empty_results"],
        "usage": a["usage"],
        "used_context": a["used_context"],
    }, attempts_file)

    append_jsonl({"task_id": str(task["task_id"]), "level": task.get("level"),
                  "final_answer": answer, "trace": a["trace"]}, trace_file)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", type=str, default="dev", choices=["all", "dev"])
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--max-tasks", type=int, default=-1)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--tasks-ids", type=str, nargs="+", default=None)
    p.add_argument("--timestamp", type=str, default=None)
    p.add_argument("--no-context", action="store_true", help="disable build_context injection")
    return p.parse_args()


def main():
    args = parse_args()
    logger.warning(f"Starting {AGENT_NAME} run with arguments: {args}")

    tasks = select_tasks(load_tasks(args.split), args.tasks_ids, args.max_tasks)
    base = prepare_run_dir(AGENT_NAME, args.split, args.timestamp)
    answers_file, attempts_file = base / "answers.jsonl", base / "attempts.jsonl"
    trace_file = base / "trace.jsonl"

    done = already_done_ids(answers_file)
    todo = [t for t in tasks if t["task_id"] not in done]
    logger.warning(f"Running {len(todo)}/{len(tasks)} tasks (skipping {len(done)} already done)")

    with ThreadPoolExecutor(max_workers=args.concurrency) as exe:
        futures = [
            exe.submit(process_task, t, args.split == "dev", args.max_steps,
                       args.temperature, not args.no_context, answers_file, attempts_file, trace_file)
            for t in todo
        ]
        for f in tqdm(as_completed(futures), total=len(todo), desc="Processing tasks"):
            f.result()

    if args.split == "dev" and answers_file.exists():
        rows = [json.loads(l) for l in open(answers_file, encoding="utf-8")]
        correct = sum(r["correct"] for r in rows)
        logger.warning(f"Dev accuracy: {correct}/{len(rows)} = {correct / len(rows):.1%}")

    if not attempts_file.exists():
        logger.warning("No tasks ran — nothing to summarise.")
        return
    rows = [json.loads(l) for l in open(attempts_file, encoding="utf-8")]
    read_any = sum(1 for r in rows if r.get("documents_read"))
    logger.warning(f"Tasks that read >=1 document: {read_any}/{len(rows)}")
    tok = sum(r.get("usage", {}).get("prompt_tokens", 0) + r.get("usage", {}).get("completion_tokens", 0) for r in rows)
    logger.warning(f"Total tokens: {tok:,}  |  context injection: {'OFF' if args.no_context else 'ON'}")
    logger.warning(f"Results in {answers_file}, diagnostics in {attempts_file}, traces in {trace_file}")


if __name__ == "__main__":
    main()
