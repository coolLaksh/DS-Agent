#!/usr/bin/env python3
"""Score a run against the golden set — local eval, no leaderboard needed.

Usage:
    python3 score_golden.py runs/react_tools_gpt-4o-mini/all/<timestamp>
    python3 score_golden.py <dir_a> <dir_b>        # compare two runs side by side

Matching follows the benchmark's own scorer: case-insensitive, punctuation-stripped,
numeric-tolerant, order-independent for comma lists.

Convention-dependent tasks count as correct if the answer matches EITHER convention
(sum-all or most-specific), since the handbook does not specify which is intended.
"""

import json
import math
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GOLDEN = json.load(open(ROOT / "golden" / "golden_set.json"))["tasks"]


def _num(s):
    try:
        return float(str(s).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _norm(s):
    return re.sub(r"[^\w]", "", str(s).strip().lower())


def equal(a, b) -> bool:
    """Benchmark-style lenient comparison."""
    a, b = str(a).strip(), str(b).strip()
    if not a or not b:
        return False
    na, nb = _num(a), _num(b)
    if na is not None and nb is not None:
        if na == nb:
            return True
        da = len(a.split(".")[-1]) if "." in a else 0
        db = len(b.split(".")[-1]) if "." in b else 0
        d = min(da, db)
        if round(na, d) == round(nb, d):
            return True
        return math.isclose(na, nb, rel_tol=1e-4, abs_tol=1e-4)
    # "key: value" pairs — compare the label exactly and the number numerically
    if ":" in a and ":" in b and a.count(":") == 1 and b.count(":") == 1:
        ka, va = (x.strip() for x in a.split(":"))
        kb, vb = (x.strip() for x in b.split(":"))
        if _norm(ka) == _norm(kb):
            return equal(va, vb)
        return False

    if "," in a or "," in b:
        stripped = lambda s: s.strip().lstrip("[").rstrip("]").strip()
        la = sorted(stripped(x) for x in stripped(a).split(",") if stripped(x))
        lb = sorted(stripped(x) for x in stripped(b).split(",") if stripped(x))
        return len(la) == len(lb) and all(equal(x, y) for x, y in zip(la, lb))
    if _norm(a) == _norm(b):
        return True
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio() > 0.95


def check(tid: str, submitted: str) -> tuple[bool, str]:
    """Return (correct, note). Handles the three golden answer shapes."""
    g = GOLDEN[tid]
    if g.get("confidence") == "unsolved":
        return False, "unsolved — excluded"

    if "answer_count" in g:
        n = len([x for x in submitted.split(",") if x.strip()])
        if "answer" in g and equal(submitted, g["answer"]):
            return True, "exact list"
        return (n == g["answer_count"]), f"{n} ids vs {g['answer_count']}"

    for key in ("answer", "answer_sum_all", "answer_most_specific"):
        if key in g and g[key] is not None and equal(submitted, g[key]):
            return True, key
    return False, "no match"


def score(run_dir: Path) -> dict:
    ans_path = run_dir / "answers.jsonl"
    if not ans_path.exists():
        sys.exit(f"no answers.jsonl in {run_dir}")
    answers = {json.loads(l)["task_id"]: json.loads(l)["agent_answer"]
               for l in open(ans_path) if l.strip()}

    scored, correct, per_arch, misses = 0, 0, {}, []
    for tid, g in GOLDEN.items():
        if g.get("confidence") == "unsolved" or tid not in answers:
            continue
        scored += 1
        ok, note = check(tid, answers[tid])
        arch = g.get("archetype", "other")
        per_arch.setdefault(arch, [0, 0])
        per_arch[arch][1] += 1
        if ok:
            correct += 1
            per_arch[arch][0] += 1
        else:
            misses.append((tid, arch, answers[tid][:46], note))

    usage = {"prompt": 0, "completion": 0}
    att = run_dir / "attempts.jsonl"
    if att.exists():
        for l in open(att):
            r = json.loads(l)
            for u in ([r["usage"]] if "usage" in r else
                      [h["usage"] for h in r.get("history", []) if "usage" in h]):
                usage["prompt"] += u.get("prompt_tokens", 0)
                usage["completion"] += u.get("completion_tokens", 0)

    return {"dir": run_dir, "scored": scored, "correct": correct,
            "per_arch": per_arch, "misses": misses, "usage": usage}


def report(r: dict, verbose=True):
    pct = 100 * r["correct"] / r["scored"] if r["scored"] else 0
    print(f"\n=== {r['dir'].name} ===")
    print(f"  {r['correct']}/{r['scored']} correct  ({pct:.1f}%)")
    tok = r["usage"]["prompt"] + r["usage"]["completion"]
    if tok:
        print(f"  tokens: {tok:,}  (prompt {r['usage']['prompt']:,} / completion {r['usage']['completion']:,})")
    print("  by archetype:")
    for a, (c, n) in sorted(r["per_arch"].items(), key=lambda x: -x[1][1]):
        bar = "#" * c + "." * (n - c)
        print(f"    {a:24s} {c:2d}/{n:2d}  {bar}")
    if verbose and r["misses"]:
        print(f"  misses ({len(r['misses'])}):")
        for tid, arch, got, note in r["misses"][:60]:
            print(f"    {tid:6s} {arch:22s} got={got!r:50s} {note}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    results = [score(Path(d)) for d in sys.argv[1:]]
    for r in results:
        report(r, verbose=len(results) == 1)
    if len(results) > 1:
        print("\n=== side by side ===")
        archs = sorted({a for r in results for a in r["per_arch"]})
        w = max(len(a) for a in archs) + 2
        print(f"  {'archetype':{w}s} " + "  ".join(f"{r['dir'].name[:18]:>18s}" for r in results))
        for a in archs:
            cells = []
            for r in results:
                c, n = r["per_arch"].get(a, (0, 0))
                cells.append(f"{c}/{n}".rjust(18))
            print(f"  {a:{w}s} " + "  ".join(cells))
        print(f"  {'TOTAL':{w}s} " + "  ".join(f"{r['correct']}/{r['scored']}".rjust(18) for r in results))
