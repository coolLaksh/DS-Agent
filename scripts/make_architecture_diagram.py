"""Render the final reflexion_V3 architecture as a clean node-and-arrow diagram.

Style: pill-shaped start/end nodes, boxes colour-coded by role (green = deterministic /
mechanical, purple = LLM-involving, light blue = an edge condition label), thin arrows,
one curved retry loop that hugs the left margin (never crosses a box). No enclosing group
boxes. Vertical spacing is computed from fixed constants, not hand-picked per box, so a
label can't end up flush against a border again. Regenerate with:
    python3 scripts/make_architecture_diagram.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "results" / "figures" / "architecture.png"

DETERMINISTIC = dict(facecolor="#DFF3EA", edgecolor="#3F8F6C")
LLM = dict(facecolor="#E7E4FB", edgecolor="#5B4FCB")
ENDPOINT = dict(facecolor="#F2EEE4", edgecolor="#8A8272")
LABEL_BG = "#DDF1FB"

X = 5.0
BOX_W = 5.8
BOX_H = 1.75
LABEL_H = 0.55
PLAIN_GAP = 1.0     # vertical space for a bare arrow, no label
LABEL_GAP = 1.35     # vertical space when a condition label sits in the middle

fig, ax = plt.subplots(figsize=(8.6, 16.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 21.5)
ax.axis("off")


def pill(cx, cy, w, h, text, **style):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.05,rounding_size=0.5",
                                linewidth=1.4, **style))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=11)


def box(cx, cy, w, h, title, subtitle=None, path=None, title_size=11.5, **style):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.05,rounding_size=0.14",
                                linewidth=1.4, **style))
    lines = [title] + ([subtitle] if subtitle else []) + ([path] if path else [])
    n = len(lines)
    gap = 0.38
    top = cy + gap * (n - 1) / 2
    for i, line in enumerate(lines):
        fs = title_size if i == 0 else 9.7
        ax.text(cx, top - i * gap, line, ha="center", va="center", fontsize=fs,
                family="monospace" if (path and i == n - 1) else None)


def arrow(x1, y1, x2, y2, **kw):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15,
                                 linewidth=1.3, color="#555555", **kw))


def edge_label(cx, cy, text, w=2.9, h=LABEL_H):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.04,rounding_size=0.08",
                                linewidth=0, facecolor=LABEL_BG))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=8.6)


# --- vertical layout, computed top to bottom so gaps can never collide ---
y = 20.6
pill_y = y
y -= 0.45 + 0.55  # pill half-height + arrow room
arrow(X, y + 0.15, X, y - PLAIN_GAP + 0.85)
y -= PLAIN_GAP

context_y = y - BOX_H / 2
y = context_y - BOX_H / 2
arrow(X, y, X, y - PLAIN_GAP)
y -= PLAIN_GAP

react_y = y - BOX_H / 2
y = react_y - BOX_H / 2
arrow(X, y, X, y - PLAIN_GAP)
y -= PLAIN_GAP

finish_y = y - BOX_H / 2
y = finish_y - BOX_H / 2
label1_y = y - LABEL_GAP / 2
arrow(X, y, X, label1_y + LABEL_H / 2 + 0.05)
arrow(X, label1_y - LABEL_H / 2 - 0.05, X, y - LABEL_GAP + 0.15)
y -= LABEL_GAP

det_y = y - BOX_H / 2
y = det_y - BOX_H / 2
label2_y = y - LABEL_GAP / 2
arrow(X, y, X, label2_y + LABEL_H / 2 + 0.05)
arrow(X, label2_y - LABEL_H / 2 - 0.05, X, y - LABEL_GAP + 0.15)
y -= LABEL_GAP

judge_y = y - BOX_H / 2
y = judge_y - BOX_H / 2 - 1.1  # room for the diagonal split

split_y = y - BOX_H / 2
fail_x, pass_x = 2.3, 7.7

final_y = split_y - BOX_H / 2 - 1.3

# --- draw the nodes ---
pill(X, pill_y, 3.4, 0.9, "task", **ENDPOINT)
box(X, context_y, BOX_W, BOX_H, "Context layer", "keyword match, no LLM",
    "build_context.py", **DETERMINISTIC)
box(X, react_y, BOX_W, BOX_H, "ReAct loop", "gpt-4o-mini, up to 10 steps",
    "agent_core.py", **LLM)
box(X, finish_y, BOX_W, BOX_H, "finish_signal", "blank / max-steps → fail",
    "reflexion_tools.py", **DETERMINISTIC)
edge_label(X, label1_y, "if not blank, not exhausted")
box(X, det_y, BOX_W, BOX_H, "deterministic checks", "verify_argmax / rate_change_grounding",
    "reflexion_tools.py", **DETERMINISTIC)
edge_label(X, label2_y, "if not confirmed")
box(X, judge_y, BOX_W, BOX_H, "LLM judge", "grounding + constraint audit, cites a step",
    "reflexion_tools.py", **LLM)

arrow(X - 0.5, judge_y - BOX_H / 2, fail_x + 0.3, split_y + BOX_H / 2 + 0.05,
     connectionstyle="arc3,rad=-0.12")
ax.text((X + fail_x) / 2 - 0.15, (judge_y + split_y) / 2 + 0.55, "fail", fontsize=8.6, style="italic")
arrow(X + 0.5, judge_y - BOX_H / 2, pass_x - 0.3, split_y + BOX_H / 2 + 0.05,
     connectionstyle="arc3,rad=0.12")
ax.text((X + pass_x) / 2 + 0.15, (judge_y + split_y) / 2 + 0.55, "pass / exhausted", fontsize=8.6, style="italic")

box(fail_x, split_y, 3.9, BOX_H, "reflect()", "writes VERIFIED / MISTAKE note",
    "reflexion_tools.py", title_size=11, **LLM)
box(pass_x, split_y, 3.9, BOX_H, "pick_best_answer()", "prefers the LAST non-blank attempt",
    "reflexion_tools.py", title_size=10.5, **DETERMINISTIC)

# retry loop: a single vertical run up the left margin, then a short stub into the
# ReAct loop's left edge - stays well clear of every box (boxes start at x=2.1)
retry_x = fail_x - 1.95  # = reflect()'s own left edge, so the line starts flush with it
arrow(retry_x, split_y + BOX_H / 2 + 0.1, retry_x, react_y)
arrow(retry_x, react_y, X - BOX_W / 2, react_y)
ax.text(retry_x - 0.35, (split_y + react_y) / 2, "retry with note",
        ha="center", va="center", fontsize=8.2, rotation=90)

arrow(pass_x, split_y - BOX_H / 2, pass_x, final_y + 0.55)
pill(pass_x, final_y, 3.6, 0.9, "final answer", **ENDPOINT)

fig.tight_layout()
fig.savefig(OUT, dpi=170, bbox_inches="tight")
print(f"wrote {OUT}  ({OUT.stat().st_size:,} bytes)")
