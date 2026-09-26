#!/usr/bin/env python3

import re

import fee_core as fc


CORE_RULES = """## Domain rules (from the merchant handbook)

**Fee formula.** `fee = fixed_amount + rate * transaction_value / 10000`
The divisor is 10000, not 100. A helper `fee_value(rule, amount)` is preloaded — prefer it.

**Wildcards.** In a fee rule, a field set to `null` or `[]` is a WILDCARD: it applies to
ALL values of that field, not to none. This holds for every field in a rule, with no
exception. A helper `rule_matches(rule, **constraints)` is preloaded and implements this
correctly for every field — prefer it over writing your own filter.

**Intracountry.** `intracountry` is True when `issuing_country == acquirer_country`.
Both columns are in payments.csv.

**Natural months.** Monthly volume and fraud-level tiers are computed per merchant over
natural calendar months (1st to last day). payments.csv has no `month` column — derive it
from `day_of_year` (2023 is not a leap year).

**Tier boundaries.** monthly_volume: `<100k`, `100k-1m`, `1m-5m`, `>5m` (EUR).
monthly_fraud_level: `<7.2%`, `7.2%-7.7%`, `7.7%-8.3%`, `>8.3%` (fraudulent volume / total volume).
capture_delay in merchant_data.json is a number of days or `immediate`/`manual`; fee rules
bucket it as `<3`, `3-5`, `>5`, `immediate`, `manual`."""

PRELOADED_HELPERS = """## Preloaded helpers

These are already in your Python namespace — do not redefine them:

- `matching_rules(merchant=..., month=..., card_scheme=..., is_credit=..., aci=..., ...)`
  → the list of fee rules that apply. **This is how you get a filtered set of rules — do
  not write `[r for r in fees if <your own condition>]`.** A hand-written condition almost
  always compares a rule's field with `==`, which silently drops every rule where that
  field is a wildcard (`null` or `[]`) instead of a specific value - and wildcards are not
  rare: `is_credit` is null in 100/1000 rules, `intracountry` in 561/1000, `capture_delay`
  in 500/1000, `monthly_fraud_level` in 900/1000, `monthly_volume` in 800/1000. Excluding
  them is usually most of the real answer, not an edge case. `matching_rules(...)` costs
  the same one line as writing the filter yourself and cannot make that mistake:

  ```python
  rules = matching_rules(card_scheme='SwiftCharge', is_credit=True)
  avg_fee = sum(fee_value(r, 5000) for r in rules) / len(rules)
  ```

  **Pass `merchant=` and `month=` whenever the question names them.** This fills in the
  five fields those determine — `account_type`, `mcc` and `capture_delay` from
  merchant_data.json, and `monthly_volume` / `monthly_fraud_level` from that merchant's
  natural calendar-month totals. Do not compute those yourself and do not leave them
  unconstrained: a question that names a merchant and a month determines all five, and
  leaving one out silently lets non-applicable rules through. Any argument you pass
  explicitly overrides the derived one, so counterfactual questions still work.

  Each constraint is a SINGLE value describing one transaction — `aci='B'`, `mcc=5812`.
  Values are validated against the vocabulary in fees.json: an impossible value raises
  with the legal list rather than silently matching nothing.

- `rule_matches(rule, **same constraints as above)` → True/False for one rule at a time.
  Use this only when constraints vary per row inside a loop (e.g. checking one rule per
  transaction, where `card_scheme=t['card_scheme']` differs each iteration) - `matching_rules`
  cannot help there since there is no single fixed filter to compute once:

  ```python
  for t in transactions:
      applicable = [r for r in fees if rule_matches(r, card_scheme=t['card_scheme'],
                    is_credit=t['is_credit'], aci=t['aci'],
                    intracountry=t['issuing_country'] == t['acquirer_country'])]
  ```

  For any other case - a single fixed set of constraints, not varying per row - use
  `matching_rules(...)` instead of this pattern.

- `fee_value(rule, amount)` → the fee for one rule at one transaction value.

- `fees` (list of 1000 rule dicts, if you need to inspect one directly, e.g. by ID),
  `merchants` (dict keyed by merchant name), `MCC_BY_DESC` (lowercased MCC description →
  code), `monthly_profile(merchant, month)` → `(volume_tier, fraud_tier)`,
  `capture_delay_bucket(days)` → the fee-rule bucket."""

PAYMENTS_SCHEMA = """## payments.csv — already loaded as `payments`

A pandas DataFrame named `payments` is already in your namespace: all 138,236 rows, plus two
columns computed for you that are NOT in the CSV on disk:

- `month` — the natural calendar month, 1-12, from `day_of_year`. Use this for any
  "in July" / "in January" question. Do not derive a month yourself: `day_of_year // 30`
  is not a month, it drifts and produces a 13th month.
- `intracountry` — `issuing_country == acquirer_country`, as a bool.

Re-reading the CSV with `pd.read_csv` gives you a frame WITHOUT those two columns. Use
`payments` as given.

Sample rows:

```
{sample}
```

Full column list: `psp_reference` (transaction id — NOT a fee id), `merchant`, `card_scheme`,
`year` (int64, always 2023), `hour_of_day`, `minute_of_hour`, `day_of_year` (1-365),
`is_credit` (bool), `eur_amount` (float), `ip_country`, `issuing_country`, `device_type`,
`ip_address`, `email_address`, `card_number`, `shopper_interaction`, `card_bin`,
`has_fraudulent_dispute` (bool), `is_refused_by_adyen` (bool), `aci` (single letter A-G),
`acquirer_country`, `month` (added), `intracountry` (added).

Note the dtypes: `year` is an integer and `is_credit` / `has_fraudulent_dispute` /
`is_refused_by_adyen` are booleans. Comparing them to strings silently returns an empty result."""

_SAMPLE_COLS = ['psp_reference', 'merchant', 'card_scheme', 'day_of_year', 'month',
                'eur_amount', 'is_credit', 'aci', 'issuing_country', 'acquirer_country',
                'intracountry']
PAYMENTS_SCHEMA = PAYMENTS_SCHEMA.replace(
    "{sample}", fc.pay[_SAMPLE_COLS].head(3).to_string(index=False))

FEE_SCHEMA = """## fees.json rule fields

`ID`, `card_scheme`, `account_type` (list), `capture_delay`, `monthly_fraud_level`,
`monthly_volume`, `merchant_category_code` (list of ints), `is_credit` (bool or null),
`aci` (list of letters), `fixed_amount` (EUR), `rate` (int), `intracountry` (1.0/0.0/null).

Only 49 distinct MCCs appear across all rules, and 127 rules have an empty (wildcard) MCC list."""


MOST_EXPENSIVE_NOTE = """## Note: "most expensive X" questions are about fees.json alone

This question has no merchant and no transaction history in it — do not touch `payments`
or `merchant_data.json`. "TransactPlus", "NexPay", "GlobalCard", "SwiftCharge" are
`card_scheme` VALUES, not merchants{scheme_hint}. There are only 5 real merchants
(`merchants` dict); a card scheme is never one of them, and filtering `payments['merchant']`
against a scheme name returns nothing.

A helper called `most_expensive_category` is ALREADY in your Python namespace and does this
end to end — call it directly, do not `import` it (there is no module to import from) and
do not write your own version. A hand-written version of this is easy to get subtly wrong:
a rule's `aci`/`mcc` field is a LIST even for one value, so grouping by `rule['aci']`
directly crashes or silently drops rows — the preloaded version already handles this.

```python
means = most_expensive_category(amount, group_by='aci', card_scheme='NexPay', is_credit=True)
# {{'A': 0.585, 'B': 0.64, ..., 'F': 0.767}}
winner = max(means, key=means.get)          # ties: sorted(means, key=means.get)[-1] isn't
                                             # stable — for a tie, take min() of the tied keys
```

`group_by` is whichever field the question is ranking — `'aci'`, `'mcc'`, or `'card_scheme'`.
Pass every other constraint the question states (`card_scheme=`, `is_credit=`, ...) as
keyword arguments, same names as `rule_matches`. **The cost of a category is the MEAN fee
across every rule that matches it — never the sum, and never a single rule's fee.** A
category can win with zero rules that name it explicitly, if wildcard rules still cover it
(e.g. `aci='G'` appears in no rule's aci list, yet rules that leave aci as a wildcard still
apply to it and can make it the answer)."""

DELTA_RATE_NOTE = """## Note: "delta ... if the fee with ID=X changed to Y" questions

This asks: if fee rule X's `rate` were Y instead of its current value, how much MORE or LESS
would the merchant pay over the period? This needs six correct steps in order (find the rule,
find which transactions it actually applies to, sum fees at the old rate, sum at the new rate,
subtract) done per-transaction - hand-writing this loop is where every past attempt at this
question shape went wrong, and it went wrong a DIFFERENT way each time (skipped the matching
step, matched only some of the transaction's fields, or got the right numbers but flipped the
sign). A preloaded helper does the whole chain correctly:

```python
r = rate_change_delta(merchant='Rafa_AI', rule_id=141, new_rate=1, month=12)   # month=None for the whole year
r['delta_sum_all']        # every transaction rule 141 matches contributes its fee
r['delta_most_specific']  # only transactions where rule 141 is the SINGLE best-matching rule
```

manual.md never states which convention is correct when several rules match one transaction,
so both are computed - if the guidelines don't say which to use, sum_all is usually intended
unless the question specifically asks about "the applicable rule" for each transaction. Do not
write your own version of this: getting the per-transaction month, tier, and match logic right
is exactly the six-step chain this helper exists to replace."""

DELTA_MCC_NOTE = """## Note: "delta ... if merchant had changed its MCC code" questions

```python
mcc_change_delta(merchant='Rafa_AI', new_mcc=5911, month=None) -> {'delta_most_specific': ..., ...}
```
Only `delta_most_specific` is meaningful here - summing every matching rule explodes into an
implausible number once MCC changes how many wildcard-MCC rules apply. Do not write your own
version: this changes which rules apply to EVERY transaction, not just one rule's rate."""

FEE_ID_NOTE = """## Note: "applicable Fee IDs" questions

A *fee ID* is a rule's `ID` field in fees.json - not a `psp_reference`, not an ACI letter.

If the question names a merchant and a period, call `applicable_fee_ids` - NOT `matching_rules`:

```python
applicable_fee_ids(merchant='Crossfit_Hanna', month=1)          # or day=200 for one day_of_year
```

`matching_rules(merchant=..., month=...)` is the WRONG tool for this question and will
overcount: it returns every rule the merchant's PROFILE (account_type/mcc/capture_delay/tier)
could match, without checking whether any real transaction actually has the specific
card_scheme/aci/is_credit/intracountry that rule also requires. `applicable_fee_ids` checks
every real transaction individually and only counts a rule if at least one transaction
actually triggers it - that is the one and only correct method for this question shape."""

TOTAL_FEES_NOTE = """## Note: "total fees ... paid" questions

```python
total_fees(merchant='Belles_cookbook_store', month=7)   # or day=N for one day_of_year
```
Returns `{'sum_all', 'most_specific'}` - manual.md never states which convention applies when
several rules match one transaction, so both are given; use whichever the guidelines imply, or
sum_all if they don't say. Do not sum `eur_amount` (that's transaction volume, not fees) and do
not hand-write the per-transaction matching loop - getting the wildcard and per-transaction
tier logic right for every row is exactly what this replaces."""

STEER_NOTE = """## Note: "steer traffic to which card scheme" questions

This asks what the merchant's REAL transactions would cost under each of the 4 card schemes -
not the merchant's current scheme mix. Format the answer as `{{card_scheme}}:{{fee}}`.

```python
totals = reroute_totals(merchant='Martinis_Fine_Steakhouse', month={month_arg})   # {{'GlobalCard': ..., ...}}
best = min(totals, key=totals.get)     # cheapest - use max() for "most expensive"/"maximum"
f"{{best}}:{{totals[best]}}"
```
{scope_note}
Do not hand-write this: it recomputes fees for every transaction under all 4 schemes, and
getting the wildcard/per-transaction matching right for that many combinations by hand is
exactly the six-step-chain mistake this replaces."""


def _merchant_block(name: str) -> str:
    m = fc.merchants[name]
    bucket = fc.capture_delay_bucket(m["capture_delay"])
    line = (f"- **{name}** — account_type `{m['account_type']}`, "
            f"merchant_category_code `{m['merchant_category_code']}`, "
            f"capture_delay `{m['capture_delay']}` (fee-rule bucket `{bucket}`), "
            f"acquirers {', '.join(m['acquirer'])}")
    if m["merchant_category_code"] not in _MCCS_IN_FEES:
        line += ("\n  Note: this MCC appears in no fee rule, so only the 127 "
                 "wildcard-MCC rules can apply to it.")
    return line


_MCCS_IN_FEES = {c for f in fc.fees for c in f["merchant_category_code"]}


# Fee/merchant-domain vocabulary. If none of these appear, the question never touches
# fees.json, so injecting the fee formula/wildcard rule/helper docs is pure dilution.
# Getting this wrong once buried a one-line guideline under ~1,300 tokens of irrelevant
# domain rules and cost a real leaderboard regression (79.17 -> 75.00 easy accuracy, task 14).
DOMAIN_KEYWORDS = (
    "fee", "aci", "mcc", "merchant categor", "card scheme", "card_scheme",
    "account type", "account_type", "capture delay", "capture_delay",
    "intracountry", "wildcard", "rule id", "delta", "expensive", "cheap",
    "steer", "authorization characteristics",
)


def build_context(task: dict) -> tuple[str, dict]:
    """Return (context_text, namespace_additions) for one task. No LLM call."""
    q = task.get("question", "")
    ql = q.lower()

    named = [n for n in fc.merchants if n.lower() in ql]
    # Naming a merchant doesn't imply fees.json is needed (task 625 is a pure payments.csv
    # aggregation) - so the merchant block below is shown regardless, gated separately.
    needs_domain = any(k in ql for k in DOMAIN_KEYWORDS)

    parts = [CORE_RULES, PRELOADED_HELPERS, PAYMENTS_SCHEMA] if needs_domain else [PAYMENTS_SCHEMA]

    if named:
        parts.append("## Merchant profiles referenced by this question\n\n"
                     + "\n".join(_merchant_block(n) for n in named))

    if "fee" in ql or "delta" in ql:
        parts.append(FEE_SCHEMA)

    # Resolve any MCC description quoted in the question to its numeric code.
    if "mcc description" in ql:
        hits = [(code, desc) for desc, code in fc.MCC_BY_DESC.items() if desc and desc in ql]
        if hits:
            parts.append("## MCC resolution\n\n"
                         + "\n".join(f"- `{d}` → `{c}`" for c, d in hits))

    # "the Nth of the year" is day_of_year=N, not a month - testing showed month= won out
    # over a brief parenthetical day= mention, so hand back the exact call instead.
    day_match = re.search(r"\bfor the (\d+)\w*\s+of the year", ql)
    if day_match and named:
        day_n = int(day_match.group(1))
        parts.append(
            f"## Note: this question's period is day_of_year={day_n} - ONE SPECIFIC DAY, "
            f"not a month\n\n\"The {day_match.group(1)} of the year\" means day_of_year "
            f"== {day_n}. Do not convert this to a month and do not use month= - that "
            f"answers for the whole month, a different (much larger) period. Use:\n\n"
            f"```python\ntotal_fees(merchant={named[0]!r}, day={day_n})\n"
            f"applicable_fee_ids(merchant={named[0]!r}, day={day_n})\n```")

    if re.search(r"\bfee id", ql):
        parts.append(FEE_ID_NOTE)

    if "total fee" in ql:
        parts.append(TOTAL_FEES_NOTE)

    if "most expensive" in ql or "least expensive" in ql:
        scheme = next((s for s in fc.IN_RULES['card_scheme'] if s.lower() in ql), None)
        parts.append(MOST_EXPENSIVE_NOTE.format(
            scheme_hint=f" (this question names `{scheme}`)" if scheme else ""))

    if "delta" in ql and re.search(r"\bid\s*=?\s*\d+", ql):
        parts.append(DELTA_RATE_NOTE)

    if "delta" in ql and "mcc" in ql:
        parts.append(DELTA_MCC_NOTE)

    if "steer" in ql:
        named_month = next((n for n in fc.MONTH_NAMES if n in ql and len(n) > 3), None)
        if named_month:
            month_arg = fc.MONTH_NAMES[named_month]
            scope_note = f"This question names a specific month ({named_month}) - pass month={month_arg}."
        else:
            month_arg = "None"
            scope_note = ('"the year 2023" means the WHOLE year - pass month=None. Do not pick '
                         'one month, and do not loop over months and combine results.')
        parts.append(STEER_NOTE.format(month_arg=month_arg, scope_note=scope_note))

    text = ("# Reference context for this task\n\n"
            "The following is established, verified information about this dataset. "
            "Use it directly rather than re-deriving it.\n\n" + "\n\n".join(parts))

    namespace = {
        "payments": fc.pay.copy(),      # per-task copy: threads must not share a mutable frame
        "fee_value": fc.fee_value,
        "rule_matches": fc.rule_matches,
        "matching_rules": fc.matching_rules,
        "most_expensive_category": fc.most_expensive_category,
        "rate_change_delta": fc.rate_change_delta,
        "mcc_change_delta": fc.mcc_change_delta,
        "applicable_fee_ids": fc.applicable_fee_ids,
        "total_fees": fc.total_fees,
        "reroute_totals": fc.reroute_totals,
        "fees": fc.fees,
        "merchants": fc.merchants,
        "MCC_BY_DESC": fc.MCC_BY_DESC,
        "monthly_profile": fc.monthly_profile,
        "capture_delay_bucket": fc.capture_delay_bucket,
    }
    return text, namespace


if __name__ == "__main__":
    import json
    from pathlib import Path

    tasks_path = Path(__file__).resolve().parent.parent / "DataSet" / "tasks" / "all.jsonl"
    tasks = {json.loads(l)["task_id"]: json.loads(l) for l in open(tasks_path)}
    for tid in ["1286", "1817", "1485", "1312"]:
        text, ns = build_context(tasks[tid])
        print(f"--- task {tid} --- {len(text)} chars, ~{len(text)//4} tokens, "
              f"{len(ns)} namespace additions")
        print(f"    sections: {[s.splitlines()[0].lstrip('# ') for s in text.split(chr(10)+chr(10)) if s.startswith('##')]}")
