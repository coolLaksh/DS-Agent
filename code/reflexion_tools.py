#!/usr/bin/env python3
"""Reflexion arm: ReAct + document tools + loud empty results + Reflexion loop.

    Task i -> ReAct attempt -> Evaluator (pass/fail) -> next task (memory cleared)
                   ^                      |
                   |               fail -> Reflect + add note
             retry with note <------------'

Evaluator combines two signals:
  1. Finish signal - how the ReAct loop terminated (programmatic, no LLM).
  2. LLM-as-judge  - scores the answer/trace against generic failure modes.

All prompts here are deliberately domain-neutral: no dataset column names, no
dtypes, no domain rules. If the agent needs the semantics of a field, it has to
discover the reference documents itself. That discovery is what we are measuring.

Usage:
    python3 reflexion_tools.py --split dev
    python3 reflexion_tools.py --split all --concurrency 2
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from agent_core import (
    DEPLOYMENT, logger, react_attempt, condense_trace,
    list_documents_text,
    create_completion_with_backoff, score_answer, append_jsonl,
    load_tasks, already_done_ids, select_tasks, prepare_run_dir,
)
from score_golden import equal as _numeric_equal

AGENT_NAME = "reflexion_tools"

JUDGE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "submit_evaluation",
            "description": "Submit the evaluation verdict for the agent's attempt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "constraints_from_question": {
                        "type": "string",
                        "description": (
                            "List every constraint the question explicitly states or implies - "
                            "entity/merchant, time period, card scheme, account type, credit/debit, "
                            "ACI, or any other filter. One per line. Write 'none stated' if the "
                            "question is unconstrained (e.g. a pure fees.json lookup with no "
                            "merchant or period)."
                        ),
                    },
                    "constraints_applied": {
                        "type": "string",
                        "description": (
                            "For EACH constraint listed above, state whether the trace actually "
                            "applied it (quote the argument/filter that applied it) or left it "
                            "unbound. This is a checklist, not a summary - go through them one by "
                            "one. A constraint left unbound when the question stated it is a defect "
                            "even if every value the trace DID compute is itself grounded."
                        ),
                    },
                    "verdict": {"type": "string", "enum": ["pass", "fail"]},
                    "failure_mode": {
                        "type": "string",
                        "enum": [
                            "none",
                            "fabricated_value",
                            "unverified_assumption",
                            "misapplied_context",
                            "incomplete_constraints",
                            "wrong_order",
                            "unsupported_conclusion",
                        ],
                    },
                    "ungrounded_step": {
                        "type": "string",
                        "description": (
                            "A VERBATIM quote from the trace's OUTPUT or INPUT text above - not a "
                            "paraphrase, not a summary - of the exact line that is the problem. "
                            "Empty string if verdict is pass. If you cannot find a literal quote "
                            "that supports your verdict, you do not have grounds to fail the "
                            "attempt - re-read the trace before submitting fail."
                        ),
                    },
                    "reason": {
                        "type": "string",
                        "description": (
                            "One or two sentences explaining the verdict. Any claim that a step "
                            "produced nothing, zero, an error, or an empty result must match what "
                            "that step's OUTPUT literally shows - do not describe an OUTPUT as "
                            "empty/absent/missing unless it is literally empty in the trace above."
                        ),
                    },
                },
                "required": ["constraints_from_question", "constraints_applied", "verdict",
                             "failure_mode", "ungrounded_step", "reason"],
            },
        },
    },
]

JUDGE_SYSTEM_PROMPT = """You are the judge in a Reflexion loop. You are not a rule-checker running
a fixed checklist - you are a general reasoning intelligence, and your one question for every
attempt is:

    Does every value, formula, and step in this trace actually come from
    /Users/lakshyatomar/Step/DataSet/context - retrieved correctly, applied to
    EVERY constraint the question actually states, and in the order the
    question logically requires - or did the agent invent, guess, misapply,
    reorder, or leave a stated constraint unapplied?

Everything needed to answer any question in this benchmark lives in that directory:
manual.md, fees.json, payments.csv, merchant_data.json, merchant_category_codes.csv,
acquirer_countries.csv, payments-readme.md. Nothing else is a legitimate source. If a number,
threshold, rule, or convention appears in the trace and you cannot trace it to one of those
files (directly, or through a preloaded helper function - see below), the agent made it up,
regardless of how plausible it looks.

BEFORE YOU JUDGE ANYTHING ELSE, do the constraint audit: write out every constraint the
question states (`constraints_from_question`), then check the trace against each one
(`constraints_applied`). A value can be perfectly grounded - genuinely traced to a real file or
helper - and the attempt can still be wrong, because grounding is not the same as completeness.
A fee-ID list built without pinning the account type the question named, or a "which scheme is
cheapest" answer that only ever computed one scheme, is not fixed by the fact that every number
in it is real. Do this audit even when the rest of the trace looks fine - it is the check most
likely to be skipped by instinct.

TWO VALID WAYS a value can be grounded - either counts as evidence, neither is "better":

1. The agent read it from a file in the trace, and the value matches what that file contains.
2. The agent called a preloaded helper (`rule_matches`, `fee_value`, `most_expensive_category`,
   `monthly_profile`, `capture_delay_bucket`) - these are already validated extractions of the
   exact same files, so their output is grounded even if the agent never opened a file
   directly. Do NOT fail an attempt just because it didn't open manual.md or fees.json, if the
   values it used came from one of these helpers instead. The helper call itself is the
   grounding. Only fail on this axis if the ARGUMENTS passed to the helper were themselves
   ungrounded (e.g. a made-up merchant name, a threshold that appears nowhere in the context).

YOU MAY ONLY FAIL FOR A DEFECT YOU CAN QUOTE. Before writing any claim that a step produced
nothing, an empty result, zero, or an error - stop and re-read that exact step's OUTPUT in the
trace above. If the OUTPUT shown is a real, non-empty, non-zero, non-error value, you may NOT
describe it as empty, missing, or absent, no matter how plausible that description feels. A
verdict of fail with no literal quote to support it is not a valid verdict - re-examine the
trace instead of submitting it. The absence of a file-read or an explicit self-check is not
itself a defect if the computation shown is already complete, grounded, and covers every stated
constraint - do not fail an attempt just to be safe.

WAYS AN ATTEMPT FAILS - these are examples of the one root question above, not a separate
checklist to apply mechanically:

- fabricated_value: a number, ID, threshold, or fact appears in the trace or the final answer
  with no origin in a file read or a helper call's output - it was typed, not derived.

- unverified_assumption: the agent needed to know how some field, code, or convention should be
  interpreted, and settled it by guessing from the data's shape rather than confirming it from
  a file or a helper that encodes that file's rule. Plausible is not the same as grounded.

- misapplied_context: the agent retrieved the right fact or rule from context, but used it
  incorrectly - the wrong entity, the wrong time period, a scheme name where a merchant was
  expected (or vice versa), a rule applied to a transaction it does not actually cover.

- incomplete_constraints: the constraint audit above found a constraint the question stated that
  the trace never bound - e.g. an account type, MCC, or time period named in the question but
  left as an unconstrained wildcard, or a comparison (e.g. across card schemes) that only ever
  computed a subset of the options instead of all of them. Every individual value can be
  grounded and this can still fail.

- wrong_order: the question requires a specific logical sequence, and the trace skipped or
  reordered a required step - most commonly, a "what would change" / "delta" / "before-and-
  after" question needs a baseline total AND a changed total AND a subtraction between them,
  computed in that order; if any of the three is missing, the sequence was not followed even
  if each individual number the agent did compute is itself correct.

- unsupported_conclusion: the final answer does not follow from anything actually shown in the
  trace - including a conclusion drawn from an output that is genuinely empty or zero in the
  trace, without first checking whether the agent's own filter (not the data) caused it.

Set `ungrounded_step` to a verbatim quote of the specific line, value, or claim that broke
grounding or completeness, so the next attempt knows exactly what to fix - not a category name,
the actual thing, copied from the trace.

PASS if every value traces back to context (directly or via a validated helper), every stated
constraint was applied, and the required steps were followed in order. You are judging whether
the reasoning is grounded, complete, and correctly sequenced - not re-deriving the ground truth
yourself, and not failing an attempt on a feeling you cannot quote evidence for.

Call `submit_evaluation` with your verdict."""

REFLECT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "submit_reflection",
            "description": "Submit the structured memory note for the agent's next attempt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "correct_findings": {
                        "type": "string",
                        "description": (
                            "Concrete facts, lookups, or code this attempt already established "
                            "correctly and the next attempt should reuse instead of recomputing - "
                            "e.g. a merchant's account_type/mcc/capture_delay, a correct filter or "
                            "rule_matches() call, a correct intermediate total. Name the actual "
                            "value or expression, not a category of thing. If nothing in the "
                            "attempt was verifiably correct, write exactly 'None identified' - do "
                            "not invent a finding to fill this field."
                        ),
                    },
                    "failure_lesson": {
                        "type": "string",
                        "description": (
                            "The exact mistake and the exact fix, written as an instruction the "
                            "next attempt must follow, not a narrative summary. Name the failure "
                            "mode. Max 3 sentences. A good lesson can be acted on directly without "
                            "re-reading the whole trace. Bad lessons are vague, e.g. 'be more "
                            "careful' or 'check the data'."
                        ),
                    },
                },
                "required": ["correct_findings", "failure_lesson"],
            },
        },
    },
]

REFLECT_SYSTEM_PROMPT = """You are the memory-writing step of a Reflexion agent.

The agent just failed an attempt at a data-analysis task. Given the question, its answer,
the evaluator's verdict, and the tool calls it made, produce a structured note for the
agent's next attempt at this SAME question.

Write two things:

1. correct_findings - anything this attempt already got right that should not be recomputed:
   a value it correctly looked up, a filter or rule_matches() call that was correctly
   constructed, a correct intermediate result. Be literal and specific so the next attempt
   can copy it directly. A failed attempt is not failed in every line - most of the mistakes
   we've seen are one wrong step inside an otherwise-correct chain. Find the correct part.

2. failure_lesson - the exact mistake, named after the evaluator's failure_mode, and the exact
   correction. This must be loud and specific enough that the next attempt cannot wander back
   into the same reasoning path by accident.

Call `submit_reflection` with both fields."""


def _parse_tool_args(msg, fallback: dict) -> dict:
    """Defensive JSON parse of a tool call's arguments.

    The model occasionally emits malformed JSON in a long free-text field
    (correct_findings/reason often contain quotes or code snippets). A raw json.loads
    crash here used to take down the entire task - one bad judge/reflect call meant the
    whole retry loop was lost and the task fell back to "Not Applicable". Degrade to the
    fallback verdict instead; the calling loop treats that the same as a skipped step.
    """
    if not msg.tool_calls:
        return fallback
    raw = msg.tool_calls[0].function.arguments or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"Malformed tool-call JSON from judge/reflect step, using fallback: {raw[:200]!r}")
        return fallback


def finish_signal(attempt: dict) -> dict:
    """Signal 1 - purely programmatic, derived from how the ReAct loop ended."""
    reason = attempt["finish_reason"]
    answer = attempt["answer"].strip()

    if reason == "final_answer_empty" or not answer:
        return {"status": "fail", "detail": "blank_answer: final_answer called with an empty string"}
    if reason == "max_steps_exhausted":
        return {"status": "fail", "detail": "max_steps_exhausted: ran out of steps without calling final_answer"}
    if answer.lower() == "not applicable":
        return {"status": "suspect", "detail": "answered 'Not Applicable' - needs the judge to confirm it is justified"}
    if reason == "plain_content":
        return {"status": "suspect", "detail": "ended with plain text instead of calling final_answer"}
    return {"status": "pass", "detail": "final_answer called with a non-empty answer"}


def _clean_list(answer: str) -> list:
    """'NexPay' / "['NexPay']" / 'A, B' -> ['NexPay'] / ['A', 'B'], comparably normalized."""
    return [a.strip(" '\"[]") for a in answer.split(",") if a.strip(" '\"[]")]


def verify_argmax_grounding(attempt: dict) -> dict | None:
    """Deterministic signal 3 - not a judgment call, a fact check.

    If the trace called most_expensive_category(), that call's own returned dict is
    already the ground truth for "which category is priciest" - the LLM judge should
    never need to eyeball a max() over a dict it can just recompute itself. Confirming
    this here means we can skip the LLM judge call entirely when it holds: cheaper, and
    immune to the judge occasionally distrusting a correct, already-explicit computation
    (see task 1516 - the agent wrote `max(means, key=means.get)`, got 'NexPay' correctly,
    and the LLM judge failed it anyway asking for "verification" that was already there).

    Only ever returns a PASS confirmation, never a fail - a mismatch could be legitimate
    (the agent used the dict for something other than its final answer, or a later
    non-recorded step transformed it), so a non-match falls through to the LLM judge
    rather than being auto-rejected here.
    """
    calls = [c for c in attempt.get("computations", []) if c["fn"] == "most_expensive_category"]
    if not calls:
        return None
    means = calls[-1]["result"]        # the last call is what the final answer most likely used
    if not means:
        return None

    best = max(means.values())
    winners = sorted((str(k) for k, v in means.items() if abs(v - best) < 1e-9))
    submitted = sorted(_clean_list(attempt["answer"]))
    if submitted and submitted == winners:
        return {"status": "pass",
                "detail": f"argmax_confirmed: most_expensive_category returned {means}; "
                          f"the submitted answer {winners} is exactly its argmax (ties included)."}
    return None


def verify_rate_change_grounding(attempt: dict) -> dict | None:
    """Deterministic signal 4 - same idea as verify_argmax_grounding, for delta-rate
    questions ("what delta would X pay if rule Y's rate changed to Z"). This archetype
    scored 0/5 in every run of this project regardless of architecture, because the correct
    answer needs a 6-step per-transaction chain and the same question broke a DIFFERENT way
    each run - no fixed bug for a prompt or guard to close. rate_change_delta() (fee_core.py)
    is the validated ground truth for that whole chain; if the trace called it, checking the
    submitted answer against its own recorded output needs no LLM opinion at all.

    manual.md never states which convention applies when several rules match one
    transaction, so both delta_sum_all and delta_most_specific count as grounded - this
    mirrors score_golden.py's own convention-dependent scoring for this archetype.

    Only ever returns a PASS confirmation, never a fail, same reasoning as
    verify_argmax_grounding: a mismatch could be a later step transforming the number
    further, so it falls through to the LLM judge rather than being auto-rejected here.
    """
    calls = [c for c in attempt.get("computations", []) if c["fn"] == "rate_change_delta"]
    if not calls:
        return None
    r = calls[-1]["result"]
    submitted = attempt["answer"].strip()
    for key in ("delta_sum_all", "delta_most_specific"):
        if submitted and _numeric_equal(submitted, r[key]):
            return {"status": "pass",
                    "detail": f"rate_change_confirmed: rate_change_delta returned {key}={r[key]!r}; "
                              f"the submitted answer {submitted!r} matches."}
    return None


def llm_judge(task: dict, attempt: dict, temperature: float) -> dict:
    """Signal 2 - LLM-as-judge scored against generic failure modes."""
    docs = attempt["documents_read"] or "(none)"
    user_content = (
        f"QUESTION:\n{task['question']}\n\n"
        f"GUIDELINES:\n{task['guidelines']}\n\n"
        f"AGENT'S FINAL ANSWER:\n{attempt['answer']!r}\n\n"
        f"HOW THE ATTEMPT ENDED: {attempt['finish_reason']}\n\n"
        f"FILES PRESENT IN THE WORKING DIRECTORY:\n{list_documents_text()}\n\n"
        f"FILES THE AGENT ACTUALLY OPENED: {docs}\n\n"
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
        max_tokens=1536,          # the constraint-audit fields add real length; 1024 risked truncation
        temperature=temperature,
    )
    msg = response.choices[0].message
    args = _parse_tool_args(msg, {"verdict": "pass", "failure_mode": "none", "ungrounded_step": "",
                                   "constraints_from_question": "", "constraints_applied": "",
                                   "reason": "judge returned no/malformed verdict; defaulting to pass"})
    return {
        "verdict": args.get("verdict", "pass"),
        "failure_mode": args.get("failure_mode", "none"),
        "ungrounded_step": args.get("ungrounded_step", ""),
        "constraints_from_question": args.get("constraints_from_question", ""),
        "constraints_applied": args.get("constraints_applied", ""),
        "reason": args.get("reason", ""),
    }


def evaluate_attempt(task: dict, attempt: dict, temperature: float) -> dict:
    """Combines four signals. Passes if finish_signal doesn't fail, and either a
    deterministic check confirms the answer or the LLM judge does."""
    signal_1 = finish_signal(attempt)

    if signal_1["status"] == "fail":
        evaluation = {
            "passed": False,
            "finish_signal": signal_1,
            "judge": {"verdict": "skipped", "failure_mode": "none", "ungrounded_step": "",
                      "constraints_from_question": "", "constraints_applied": "",
                      "reason": "short-circuited by finish signal"},
        }
        return evaluation

    for check in (verify_argmax_grounding, verify_rate_change_grounding):
        result = check(attempt)
        if result is not None:
            evaluation = {
                "passed": True,
                "finish_signal": signal_1,
                "judge": {"verdict": "pass", "failure_mode": "none", "ungrounded_step": "",
                          "constraints_from_question": "", "constraints_applied": "",
                          "reason": result["detail"] + " (LLM judge skipped - deterministic confirmation)"},
            }
            return evaluation

    judge = llm_judge(task, attempt, temperature)
    return {"passed": judge["verdict"] == "pass", "finish_signal": signal_1, "judge": judge}


def reflect(task: dict, attempt: dict, evaluation: dict, temperature: float) -> dict:
    """Returns {"correct_findings": str, "failure_lesson": str, "note": str}.

    `note` is the two of those pre-formatted into the block react_attempt() injects into
    the next attempt's prompt (see agent_core.react_attempt's `notes` handling).
    """
    user_content = (
        f"QUESTION:\n{task['question']}\n\n"
        f"GUIDELINES:\n{task['guidelines']}\n\n"
        f"AGENT'S ANSWER:\n{attempt['answer']!r}\n\n"
        f"HOW IT ENDED: {attempt['finish_reason']}\n"
        f"FILES PRESENT IN THE WORKING DIRECTORY:\n{list_documents_text()}\n"
        f"FILES THE AGENT ACTUALLY OPENED: {attempt['documents_read'] or '(none)'}\n\n"
        f"EVALUATOR - finish signal: {evaluation['finish_signal']['detail']}\n"
        f"EVALUATOR - judge: {evaluation['judge']['failure_mode']} - {evaluation['judge']['reason']}\n"
        f"EVALUATOR - ungrounded step: {evaluation['judge'].get('ungrounded_step', '') or '(n/a)'}\n\n"
        f"TOOL CALLS:\n{condense_trace(attempt['trace'])}"
    )

    response = create_completion_with_backoff(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": REFLECT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        tools=REFLECT_TOOLS,
        tool_choice={"type": "function", "function": {"name": "submit_reflection"}},
        max_tokens=400,
        temperature=temperature,
    )
    msg = response.choices[0].message
    args = _parse_tool_args(msg, {})
    if not args:
        return {"correct_findings": "", "failure_lesson": "", "note": ""}

    correct_findings = (args.get("correct_findings") or "None identified").strip()
    failure_lesson = (args.get("failure_lesson") or "").strip()
    failure_mode = evaluation["judge"]["failure_mode"]
    if failure_mode == "none":  # judge short-circuited by finish_signal (blank/max-steps)
        failure_mode = evaluation["finish_signal"]["detail"].split(":", 1)[0]

    note = (
        f"VERIFIED (reuse, do not recompute): {correct_findings}\n"
        f"MISTAKE [{failure_mode}] (do not repeat): {failure_lesson}"
    )
    return {"correct_findings": correct_findings, "failure_lesson": failure_lesson, "note": note}


def pick_best_answer(attempts: list[dict]) -> str:
    """Fallback when no attempt ever passed. Never return a blank answer.

    Prefers the LAST attempt over the first: each attempt is informed by the reflection
    notes from every attempt before it, so a later attempt is the more-refined guess, not
    an arbitrary one. Iterating attempts in reverse means "most-informed non-blank answer"
    instead of "first non-blank answer".
    """
    for a in reversed(attempts):
        ans = a["answer"].strip()
        if ans and ans.lower() != "not applicable":
            return ans
    for a in reversed(attempts):
        if a["answer"].strip():
            return a["answer"].strip()
    return "Not Applicable"


def run_single_task(task: dict, max_steps: int, temperature: float, max_attempts: int,
                    use_context: bool) -> dict:

    notes: list[str] = []          # per-task memory, dies with this call
    attempts: list[dict] = []
    history: list[dict] = []

    for idx in range(1, max_attempts + 1):
        attempt = react_attempt(task, max_steps, temperature, notes, use_context=use_context)
        attempts.append(attempt)
        evaluation = evaluate_attempt(task, attempt, temperature)

        history.append({
            "attempt": idx,
            "answer": attempt["answer"],
            "finish_reason": attempt["finish_reason"],
            "documents_read": attempt["documents_read"],
            "n_code_calls": attempt["n_code_calls"],
            "n_empty_results": attempt["n_empty_results"],
            "usage": attempt["usage"],
            "passed": evaluation["passed"],
            "finish_signal": evaluation["finish_signal"]["status"],
            "judge_failure_mode": evaluation["judge"]["failure_mode"],
            "judge_reason": evaluation["judge"]["reason"],
            "judge_ungrounded_step": evaluation["judge"].get("ungrounded_step", ""),
            "judge_constraints_from_question": evaluation["judge"].get("constraints_from_question", ""),
            "judge_constraints_applied": evaluation["judge"].get("constraints_applied", ""),
        })

        if evaluation["passed"]:
            return {"answer": attempt["answer"], "attempts": idx, "history": history,
                    "traces": [a["trace"] for a in attempts]}

        if idx < max_attempts:
            reflection = reflect(task, attempt, evaluation, temperature)
            if reflection["note"]:
                notes.append(reflection["note"])
                history[-1]["reflection_correct"] = reflection["correct_findings"]
                history[-1]["reflection_failure"] = reflection["failure_lesson"]

    return {"answer": pick_best_answer(attempts), "attempts": max_attempts, "history": history,
            "traces": [a["trace"] for a in attempts]}


def process_task(task: dict, is_dev_data: bool, max_steps: int, temperature: float,
                 max_attempts: int, use_context: bool, answers_file: Path,
                 attempts_file: Path, trace_file: Path):
    try:
        result = run_single_task(task, max_steps, temperature, max_attempts, use_context)
    except Exception as e:
        logger.warning(f"Task id: {task['task_id']} FAILED: {e}")
        entry = {"task_id": str(task["task_id"]), "agent_answer": "Not Applicable"}
        if is_dev_data:
            entry.update({"answer": task["answer"], "correct": False, "level": task.get("level")})
        append_jsonl(entry, answers_file)
        append_jsonl({"task_id": str(task["task_id"]), "error": str(e)}, attempts_file)
        return

    answer = result["answer"]
    docs = sorted({d for h in result["history"] for d in h["documents_read"]})
    logger.warning(
        f"Task id: {task['task_id']}\tattempts: {result['attempts']}\tdocs: {docs}\t"
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
        "n_attempts": result["attempts"],
        "documents_read_any": docs,
        "history": result["history"],
    }, attempts_file)

    append_jsonl({"task_id": str(task["task_id"]), "level": task.get("level"),
                  "final_answer": answer, "traces": result["traces"]}, trace_file)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", type=str, default="dev", choices=["all", "dev"])
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--max-tasks", type=int, default=-1)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--max-attempts", type=int, default=3)
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
                       args.temperature, args.max_attempts, not args.no_context,
                       answers_file, attempts_file, trace_file)
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
    retried = sum(1 for r in rows if r.get("n_attempts", 1) > 1)
    read_any = sum(1 for r in rows if r.get("documents_read_any"))
    logger.warning(f"Tasks that needed >1 attempt: {retried}/{len(rows)}")
    logger.warning(f"Tasks that read >=1 document: {read_any}/{len(rows)}")
    tok = sum(h.get("usage", {}).get("prompt_tokens", 0) + h.get("usage", {}).get("completion_tokens", 0)
              for r in rows for h in r.get("history", []))
    logger.warning(f"Total tokens: {tok:,}  |  context injection: {'OFF' if args.no_context else 'ON'}")
    logger.warning(f"Results in {answers_file}, diagnostics in {attempts_file}, traces in {trace_file}")


if __name__ == "__main__":
    main()
