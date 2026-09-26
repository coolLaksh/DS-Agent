"""Correct fee machinery, derived strictly from manual.md.

Rules used (manual.md section 5):
  fee = fixed_amount + rate * transaction_value / 10000
  A null / [] field is a WILDCARD - the rule applies to all values of that field.
  intracountry = (issuing_country == acquirer_country)
  Monthly volume / fraud level are natural-month aggregates for the merchant.

Two agent-facing safeguards, both measured against the 450-task trace:

  (A) DERIVATION. `rule_matches(rule, merchant=..., month=...)` fills in the five
      fields that are *determined* once a merchant and month are known - account_type,
      mcc, capture_delay bucket, monthly_volume, monthly_fraud_level. The agent reads
      the merchant and the month out of the question (it does that reliably); the
      lookup chain it does NOT do reliably now belongs to the function. In the traced
      run, 88 of 214 rule_matches tasks passed monthly_fraud_level=None / monthly_volume=None
      for questions that pinned both a merchant and a month, silently over-matching.

  (B) VOCABULARY GUARD. Every constraint field has a small closed set of legal values,
      read out of fees.json itself. A value outside it can never match a rule, so it is
      never a legitimate call. Unambiguous shapes are coerced (capture_delay=2 -> '<3',
      aci='b' -> 'B'); genuinely unknown values raise with the legal list. Values that
      are real in the data but absent from every rule (aci='G', 25463 transactions)
      emit a notice rather than raising - a true zero-match is a finding, not a bug.
"""
import json
import threading
from functools import lru_cache
from pathlib import Path

import pandas as pd

CTX = str((Path(__file__).resolve().parent.parent / 'DataSet' / 'context'))

fees      = json.load(open(f'{CTX}/fees.json'))
merchants = {m['merchant']: m for m in json.load(open(f'{CTX}/merchant_data.json'))}
mcc_df    = pd.read_csv(f'{CTX}/merchant_category_codes.csv')
MCC_BY_DESC = {d.strip().lower(): int(c) for c, d in zip(mcc_df['mcc'], mcc_df['description'])}

pay = pd.read_csv(f'{CTX}/payments.csv')
pay['month'] = pd.to_datetime(pay['day_of_year'], format='%j').dt.month     # 2023 is not a leap year
pay['intracountry'] = pay['issuing_country'] == pay['acquirer_country']

# card_scheme is never a wildcard, so bucketing by it cuts mcc_change_delta()'s
# per-transaction full-fees scan to ~1/4 size.
_FEES_BY_SCHEME = {}
for _r in fees:
    _FEES_BY_SCHEME.setdefault(_r['card_scheme'], []).append(_r)


# --- Notices: drained by run_code() and appended to its output (thread-local: concurrency > 1) ---

_tl = threading.local()


def _notice(msg: str) -> None:
    lst = getattr(_tl, "notices", None)
    if lst is None:
        lst = _tl.notices = []
    if msg not in lst:
        lst.append(msg)


def drain_notices() -> list:
    lst = getattr(_tl, "notices", None) or []
    _tl.notices = []
    return lst


def _record_computation(fn_name: str, result) -> None:
    """Record a helper's own computed result, so a downstream verifier can check the
    agent's final answer against it directly - not by re-parsing stdout text, which
    varies with print style, comments, and formatting. See most_expensive_category()."""
    lst = getattr(_tl, "computations", None)
    if lst is None:
        lst = _tl.computations = []
    lst.append({"fn": fn_name, "result": result})


def drain_computations() -> list:
    lst = getattr(_tl, "computations", None) or []
    _tl.computations = []
    return lst


def capture_delay_bucket(v):
    """merchant_data stores '1','2','7','immediate','manual'; fees.json uses buckets."""
    if v in ('immediate', 'manual'):
        return v
    n = float(v)
    if n < 3:   return '<3'
    if n <= 5:  return '3-5'
    return '>5'


def volume_tier(eur):
    if eur < 100_000:     return '<100k'
    if eur < 1_000_000:   return '100k-1m'
    if eur <= 5_000_000:  return '1m-5m'
    return '>5m'


def fraud_tier(ratio_pct):
    if ratio_pct < 7.2:   return '<7.2%'
    if ratio_pct <= 7.7:  return '7.2%-7.7%'
    if ratio_pct <= 8.3:  return '7.7%-8.3%'
    return '>8.3%'


@lru_cache(maxsize=None)
def monthly_profile(merchant, month):
    """Natural-month volume and fraud tier for a merchant (manual.md section 5 notes).

    Cached: rule_matches(merchant=..., month=...) is called once per rule inside
    1000-rule loops, and this does a full dataframe scan.
    """
    m = pay[(pay['merchant'] == merchant) & (pay['month'] == month)]
    if len(m) == 0:
        return None, None
    vol = m['eur_amount'].sum()
    fraud = m.loc[m['has_fraudulent_dispute'], 'eur_amount'].sum()
    return volume_tier(vol), fraud_tier(100 * fraud / vol)


def fee_value(f, amount):
    return f['fixed_amount'] + f['rate'] * amount / 10000


def most_expensive_category(amount, group_by, **constraints):
    """Rank a fee dimension by cost, for questions shaped 'what is the most expensive
    <ACI/MCC/card_scheme/...> for a transaction of N euros'.

    These questions have NO merchant and NO transaction history in them - they ask about
    fees.json alone. group_by is the field to rank ('aci', 'mcc', 'card_scheme', ...);
    **constraints pins the rest (card_scheme=, is_credit=, ...), same keywords as
    rule_matches. Each candidate value's cost is the MEAN fee across every rule that
    matches it (not the sum, and not a single rule's fee) - that is the convention this
    benchmark's own withheld answers use.

    Returns a dict {value: mean_fee, ...} for every value with at least one matching rule
    (including a value with zero explicit rules for it, if wildcard rules still cover it -
    e.g. aci='G' never appears in a rule's aci list but 112 rules leave aci as a wildcard
    and so still apply to it). Use max(result, key=result.get) for the winner, or sort
    result.items() by value descending for a full ranking; break ties alphabetically.
    """
    candidates = sorted(IN_DATA[group_by], key=str)
    means = {}
    for v in candidates:
        matching = [r for r in fees if rule_matches(r, **{**constraints, group_by: v})]
        if matching:
            means[v] = sum(fee_value(r, amount) for r in matching) / len(matching)
    _record_computation("most_expensive_category", means)
    return means


def _specificity(rule):
    """How many fields on this rule are pinned (non-wildcard). Used to pick a single
    winning rule per transaction under the 'most-specific-wins' convention - manual.md
    never states which rule wins when several match, so both conventions are computed."""
    fields = ('card_scheme', 'account_type', 'capture_delay', 'monthly_fraud_level',
              'monthly_volume', 'merchant_category_code', 'is_credit', 'aci', 'intracountry')
    return sum(1 for f in fields if rule[f] not in (None, []))


@lru_cache(maxsize=None)
def _winning_rule_id(merchant, month, card_scheme, is_credit, aci, intracountry):
    """The single rule that applies to one transaction under most-specific-wins: most
    pinned fields, lowest ID on ties. Cached - called once per (merchant, month, aci,
    card_scheme, is_credit, intracountry) combination, which repeats heavily across a
    merchant's transactions in one month."""
    candidates = [r for r in fees if rule_matches(r, merchant=merchant, month=month,
                                                  card_scheme=card_scheme, is_credit=is_credit,
                                                  aci=aci, intracountry=intracountry)]
    if not candidates:
        return None
    return min(candidates, key=lambda r: (-_specificity(r), r['ID']))['ID']


def rate_change_delta(merchant, rule_id, new_rate, month=None):
    """Deterministic answer key for 'what delta would <merchant> pay in <period> if the
    relative fee of the fee with ID=<rule_id> changed to <new_rate>' - the delta-rate
    archetype, which scored 0/5 in every run of this project regardless of architecture.

    Root cause diagnosed from traces: the correct answer needs a 6-step chain (find the
    rule, determine which of the merchant's transactions it actually applies to, sum fees
    at the old rate, sum at the new rate, subtract) done per-transaction, and the same
    question produced 5 different wrong mechanisms across 5 runs - not one fixed bug, so
    no prompt or guard closes it. This function does the whole chain in one call.

    manual.md never states which rule wins when several match one transaction (see
    fee_core README), so both documented conventions are returned:
      - sum_all: every transaction rule_id itself matches contributes its fee, regardless
        of whether a more specific rule would also apply to that transaction.
      - most_specific: only transactions where rule_id is the SINGLE most-specific
        matching rule are affected - if a more specific rule already wins that
        transaction, changing rule_id's rate has no effect on it at all.
    A rate change does not alter which rule wins by specificity (specificity depends on
    which fields are pinned, not on the rate value), so the winning-rule set is identical
    before and after the change - only rule_id's own contribution moves.

    month=None means the whole year (every month the merchant has transactions).
    """
    merchant = _resolve_merchant(merchant)
    rule = next((r for r in fees if r['ID'] == rule_id), None)
    if rule is None:
        raise RuleMatchArgError(f"rule_id={rule_id} is not a fee rule ID in fees.json "
                                f"(valid range is whatever IDs actually appear there - "
                                f"check with [r['ID'] for r in fees]).")
    mutated = {**rule, 'rate': new_rate}

    df = pay[pay['merchant'] == merchant]
    if month is not None:
        month = _resolve_month(month)
        df = df[df['month'] == month]

    sum_all_baseline = sum_all_changed = 0.0
    ms_baseline = ms_changed = 0.0
    n_matched = n_winning = 0

    for t in df.itertuples():
        txn_month = int(t.month)
        matches = rule_matches(rule, merchant=merchant, month=txn_month, card_scheme=t.card_scheme,
                               is_credit=t.is_credit, aci=t.aci, intracountry=t.intracountry)
        if matches:
            n_matched += 1
            sum_all_baseline += fee_value(rule, t.eur_amount)
            sum_all_changed += fee_value(mutated, t.eur_amount)

            winner = _winning_rule_id(merchant, txn_month, t.card_scheme, t.is_credit,
                                      t.aci, bool(t.intracountry))
            if winner == rule_id:
                n_winning += 1
                ms_baseline += fee_value(rule, t.eur_amount)
                ms_changed += fee_value(mutated, t.eur_amount)

    result = {
        "rule_id": rule_id, "n_transactions": len(df), "n_matched_by_rule": n_matched,
        "n_where_rule_wins": n_winning,
        "delta_sum_all": sum_all_changed - sum_all_baseline,
        "delta_most_specific": ms_changed - ms_baseline,
        "baseline_sum_all": sum_all_baseline, "changed_sum_all": sum_all_changed,
        "baseline_most_specific": ms_baseline, "changed_most_specific": ms_changed,
    }
    _record_computation("rate_change_delta", result)
    return result


@lru_cache(maxsize=None)
def _winner_for_mcc(merchant, month, card_scheme, is_credit, aci, intracountry, mcc):
    """Same as _winning_rule_id but with mcc passed explicitly, so it can be a
    counterfactual value instead of the merchant's real one - see mcc_change_delta."""
    candidates = [r for r in _FEES_BY_SCHEME.get(card_scheme, [])
                  if rule_matches(r, merchant=merchant, month=month, card_scheme=card_scheme,
                                  is_credit=is_credit, aci=aci, intracountry=intracountry, mcc=mcc)]
    if not candidates:
        return None
    return min(candidates, key=lambda r: (-_specificity(r), r['ID']))['ID']


def mcc_change_delta(merchant, new_mcc, month=None):
    """Delta in total fees `merchant` would pay for the given period if its MCC had been
    `new_mcc` all along, instead of its real one. month=None means the full year 2023.

    Only the most-specific-wins convention is meaningful here (unlike rate_change_delta,
    which reports both) - changing MCC changes how many wildcard-MCC rules apply to every
    transaction, and summing all of them explodes into an implausible number (validated:
    the benchmark's own 2 known delta-MCC answers match most_specific exactly and sum_all
    by a factor of ~10-15x in the wrong direction). Reproduces both to the cent.
    """
    m = _resolve_merchant(merchant)
    real_mcc = merchants[m]['merchant_category_code']
    df = pay[pay['merchant'] == m]
    if month is not None:
        month = _resolve_month(month)
        df = df[df['month'] == month]

    baseline = changed = 0.0
    for t in df.itertuples():
        key = (m, int(t.month), t.card_scheme, t.is_credit, t.aci, bool(t.intracountry))
        w_base = _winner_for_mcc(*key, real_mcc)
        w_new = _winner_for_mcc(*key, new_mcc)
        if w_base is not None:
            baseline += fee_value(next(r for r in fees if r['ID'] == w_base), t.eur_amount)
        if w_new is not None:
            changed += fee_value(next(r for r in fees if r['ID'] == w_new), t.eur_amount)

    result = {"real_mcc": real_mcc, "new_mcc": new_mcc, "n_transactions": len(df),
              "delta_most_specific": changed - baseline,
              "baseline_most_specific": baseline, "changed_most_specific": changed}
    _record_computation("mcc_change_delta", result)
    return result


def _scoped_transactions(merchant, month=None, day=None):
    """merchant's real transactions in a period. day=N (day_of_year, 1-365) takes
    precedence over month=N (1-12); neither given means the whole year."""
    m = _resolve_merchant(merchant)
    df = pay[pay['merchant'] == m]
    if day is not None:
        df = df[df['day_of_year'] == day]
    elif month is not None:
        df = df[df['month'] == _resolve_month(month)]
    return m, df


def applicable_fee_ids(merchant, month=None, day=None):
    """Sorted list of fee rule IDs that apply to at least one of `merchant`'s real
    transactions in the period (day=N takes precedence over month=N; neither = whole
    year). Reproduces this benchmark's own answer counts exactly (33, 23, 32 IDs on the
    3 cases checked).

    Not the same as "every rule the merchant's profile could match" - that overcounts,
    because it ignores each transaction's actual card_scheme/aci/is_credit/intracountry.
    Only rules that a REAL transaction actually triggers belong in the answer.
    """
    m, df = _scoped_transactions(merchant, month, day)
    ids = set()
    for t in df.itertuples():
        for r in _FEES_BY_SCHEME.get(t.card_scheme, []):
            if rule_matches(r, merchant=m, month=int(t.month), card_scheme=t.card_scheme,
                            is_credit=t.is_credit, aci=t.aci, intracountry=bool(t.intracountry)):
                ids.add(r['ID'])
    result = sorted(ids)
    _record_computation("applicable_fee_ids", {"merchant": m, "month": month, "day": day,
                                                 "n_ids": len(result), "ids": result})
    return result


def total_fees(merchant, month=None, day=None):
    """Total fees `merchant` paid in the period (day=N takes precedence over month=N;
    neither = whole year). Returns {'sum_all', 'most_specific'} - manual.md never states
    which convention applies when several rules match one transaction, so both are given.
    Reproduces all 5 of this benchmark's own total-fees answers to the cent, both
    conventions.
    """
    m, df = _scoped_transactions(merchant, month, day)
    sum_all = most_spec = 0.0
    for t in df.itertuples():
        key = (m, int(t.month), t.card_scheme, t.is_credit, t.aci, bool(t.intracountry))
        for r in _FEES_BY_SCHEME.get(t.card_scheme, []):
            if rule_matches(r, merchant=m, month=key[1], card_scheme=t.card_scheme,
                            is_credit=t.is_credit, aci=t.aci, intracountry=key[5]):
                sum_all += fee_value(r, t.eur_amount)
        winner = _winning_rule_id(*key)
        if winner is not None:
            most_spec += fee_value(next(r for r in fees if r['ID'] == winner), t.eur_amount)

    result = {"sum_all": round(sum_all, 2), "most_specific": round(most_spec, 2),
              "n_transactions": len(df)}
    _record_computation("total_fees", {"merchant": m, "month": month, "day": day, **result})
    return result


def reroute_totals(merchant, month=None):
    """Total fees `merchant` would pay in the period if ALL its real transactions were
    processed through each card scheme instead of their real one - everything else about
    each transaction (is_credit, aci, intracountry, month) stays as it actually was.

    Returns {scheme: total} using the sum_all convention (validated exact-to-the-cent
    against all 3 of this benchmark's own scheme-steering answers). Use with min()/max()
    to answer "which scheme should X steer traffic to for the cheapest/most expensive fees":

        totals = reroute_totals('Martinis_Fine_Steakhouse', month=6)
        best = min(totals, key=totals.get)   # -> 'NexPay', totals['NexPay'] -> 382.85
    """
    m = _resolve_merchant(merchant)
    df = pay[pay['merchant'] == m]
    if month is not None:
        df = df[df['month'] == _resolve_month(month)]

    totals = {s: 0.0 for s in IN_RULES['card_scheme']}
    for t in df.itertuples():
        for s in totals:
            for r in _FEES_BY_SCHEME.get(s, []):
                if rule_matches(r, merchant=m, month=int(t.month), card_scheme=s,
                                is_credit=t.is_credit, aci=t.aci, intracountry=bool(t.intracountry)):
                    totals[s] += fee_value(r, t.eur_amount)

    totals = {s: round(v, 2) for s, v in totals.items()}
    _record_computation("reroute_totals", {"merchant": m, "month": month, **totals})
    return totals


# --- (B) Vocabulary, read out of the data rather than hand-authored ---

def _rule_vocab(field):
    vals = set()
    for r in fees:
        v = r[field]
        if isinstance(v, list):
            vals.update(v)
        elif v is not None:
            vals.add(v)
    return vals


# Values that CAN match at least one fee rule.
IN_RULES = {
    'card_scheme':         _rule_vocab('card_scheme'),
    'account_type':        _rule_vocab('account_type'),
    'capture_delay':       _rule_vocab('capture_delay'),
    'monthly_fraud_level': _rule_vocab('monthly_fraud_level'),
    'monthly_volume':      _rule_vocab('monthly_volume'),
    'mcc':                 _rule_vocab('merchant_category_code'),
    'is_credit':           {True, False},
    'aci':                 _rule_vocab('aci'),
    'intracountry':        {True, False},
}

# Values that are real in the dataset, whether or not any rule covers them.
# aci='G' lives here and not in IN_RULES: 25463 genuine transactions, zero rules.
IN_DATA = {
    'card_scheme':         set(pay['card_scheme'].dropna().unique()),
    'account_type':        IN_RULES['account_type'] | {m['account_type'] for m in merchants.values()},
    'capture_delay':       IN_RULES['capture_delay'],
    'monthly_fraud_level': IN_RULES['monthly_fraud_level'],
    'monthly_volume':      IN_RULES['monthly_volume'],
    'mcc':                 set(int(c) for c in mcc_df['mcc']),
    'is_credit':           {True, False},
    'aci':                 set(pay['aci'].dropna().unique()),
    'intracountry':        {True, False},
}

_RULE_FIELD = {f: f for f in IN_RULES}
_RULE_FIELD['mcc'] = 'merchant_category_code'

_LEGAL_HINT = {
    'capture_delay': ("merchant_data.json stores a number of days; fee rules use buckets. "
                      "capture_delay_bucket(2) -> '<3'."),
    'mcc':           "Pass the numeric code, e.g. mcc=5812, not the description.",
    'aci':           "A single letter A-G, uppercase.",
    'account_type':  "A single uppercase letter: D, F, H, R or S.",
}

MONTH_NAMES = {n.lower(): i for i, n in enumerate(
    ['January', 'February', 'March', 'April', 'May', 'June', 'July',
     'August', 'September', 'October', 'November', 'December'], start=1)}
MONTH_NAMES.update({n[:3]: i for n, i in list(MONTH_NAMES.items())})


class RuleMatchArgError(TypeError):
    """Raised when a constraint is passed in a shape rule_matches cannot interpret."""


def _scalar(name, v):
    """Constraints describe ONE transaction, so each is a single value.

    Agents routinely pass ['R'] instead of 'R' because the *rule* field is a list -
    that silently matched nothing before. Unwrap the unambiguous case, raise loudly
    on the ambiguous one.
    """
    if v is None or not isinstance(v, (list, tuple, set)):
        return v
    v = list(v)
    if len(v) == 1:
        return v[0]
    raise RuleMatchArgError(
        f"{name}= expects a single value describing one transaction, e.g. {name}='R', "
        f"not a list of {len(v)}. You passed {v!r}. To test several values, call "
        f"rule_matches once per value.")


@lru_cache(maxsize=None)
def _canon(field, v):
    """Coerce v into the canonical form for `field`. Returns (value, notice_or_None).

    Cached because rule_matches runs inside million-iteration loops and the vocabulary
    is tiny. Raises RuleMatchArgError for a value no rule could ever match.
    """
    in_rules, in_data = IN_RULES[field], IN_DATA[field]
    orig = v

    # --- unambiguous coercions -------------------------------------------------
    if field in ('aci', 'account_type') and isinstance(v, str):
        v = v.strip().upper()
    elif field == 'card_scheme' and isinstance(v, str):
        lookup = {s.lower(): s for s in in_rules}
        v = lookup.get(v.strip().lower(), v.strip())
    elif field in ('monthly_fraud_level', 'monthly_volume') and isinstance(v, str):
        v = v.strip()
    elif field == 'mcc':
        if isinstance(v, str) and v.strip().isdigit():
            v = int(v.strip())
        elif isinstance(v, float) and float(v).is_integer():
            v = int(v)
        else:
            try:
                v = int(v)
            except (TypeError, ValueError):
                pass
    elif field == 'is_credit':
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ('true', 'false'):
                v = (s == 'true')
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and v in (0, 1):
            v = bool(v)
        elif not isinstance(v, bool):
            v = bool(v)
    elif field == 'intracountry':
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ('true', 'false'):
                v = (s == 'true')
        else:
            v = bool(v)
    elif field == 'capture_delay':
        if isinstance(v, str) and v.strip().lower() in ('immediate', 'manual'):
            v = v.strip().lower()
        elif v not in in_rules:
            try:
                bucketed = capture_delay_bucket(v)
            except (TypeError, ValueError):
                bucketed = None
            if bucketed is not None:
                return bucketed, (
                    f"capture_delay={orig!r} is a number of days, but fee rules store "
                    f"buckets - read as {bucketed!r}. Use capture_delay_bucket() to be explicit.")

    # --- three-tier verdict ----------------------------------------------------
    if v in in_rules:
        return v, None
    if v in in_data:
        n_wild = sum(1 for r in fees if not r[_RULE_FIELD[field]])
        return v, (f"{field}={v!r} is real in the dataset but no fee rule names it, so only "
                   f"the {n_wild} rules that leave {field} as a wildcard can match. That is a "
                   f"property of the dataset, not an error - do not substitute another value "
                   f"to force a match.")
    legal = sorted(in_rules, key=str)
    shown = legal if len(legal) <= 8 else legal[:8] + ['...']
    raise RuleMatchArgError(
        f"{field}={orig!r} matches no fee rule and is not a value that appears in the "
        f"dataset. Legal values: {shown}. " + _LEGAL_HINT.get(field, ""))


def _check(field, v):
    v = _scalar(field, v)
    if v is None:
        return None
    try:
        v, note = _canon(field, v)
    except TypeError as e:
        if isinstance(e, RuleMatchArgError):
            raise
        raise RuleMatchArgError(f"{field}={v!r} could not be interpreted: {e}") from e
    if note:
        _notice(note)
    return v


def _resolve_merchant(name):
    if name in merchants:
        return name
    hit = [k for k in merchants if k.lower() == str(name).strip().lower()]
    if hit:
        return hit[0]
    scheme_hit = [s for s in IN_RULES['card_scheme'] if s.lower() == str(name).strip().lower()]
    if scheme_hit:
        raise RuleMatchArgError(
            f"merchant={name!r} is not a merchant - it is a card scheme (one of "
            f"{sorted(IN_RULES['card_scheme'])}). merchant_data.json's 5 merchants "
            f"(who processes the payment) are a different thing from card_scheme "
            f"(which network issued the card): {sorted(merchants)}. "
            f"Pass card_scheme={scheme_hit[0]!r} instead, and drop merchant= entirely "
            f"if the question never names one of the 5 merchants.")
    raise RuleMatchArgError(
        f"merchant={name!r} is not in merchant_data.json. Known merchants: "
        f"{sorted(merchants)}")


def _resolve_month(month):
    if isinstance(month, str):
        s = month.strip().lower()
        if s in MONTH_NAMES:
            return MONTH_NAMES[s]
        if s.isdigit():
            month = int(s)
    try:
        month = int(month)
    except (TypeError, ValueError):
        raise RuleMatchArgError(
            f"month={month!r} is not a month. Pass 1-12 or a name like 'July'.")
    if not 1 <= month <= 12:
        raise RuleMatchArgError(
            f"month={month} is not a calendar month. Months are 1-12. If this came from "
            f"day_of_year, note that (day_of_year - 1) // 30 + 1 is NOT the month - it "
            f"produces a 13th month and misdates 25 days a year. manual.md requires "
            f"natural months: pd.to_datetime(day_of_year, format='%j').dt.month.")
    return month


def rule_matches(f, *, merchant=None, month=None, card_scheme=None, account_type=None,
                 capture_delay=None, monthly_fraud_level=None, monthly_volume=None,
                 mcc=None, is_credit=None, aci=None, intracountry=None):
    """True if fee rule f applies. A wildcard field on the RULE matches anything.
    A None argument means 'do not constrain on this field'.

    Pass merchant= (and month=) and the fields those determine are filled in for you:
    account_type, mcc and capture_delay from merchant_data.json; monthly_volume and
    monthly_fraud_level from that merchant's natural-month totals. An argument you pass
    explicitly always wins, so counterfactuals still work.
    """
    if merchant is not None:
        merchant = _resolve_merchant(merchant)
        m = merchants[merchant]
        if account_type is None:
            account_type = m['account_type']
        if mcc is None:
            mcc = m['merchant_category_code']
        if capture_delay is None:
            capture_delay = capture_delay_bucket(m['capture_delay'])
        if month is None:
            _notice(f"merchant={merchant!r} given without month=, so monthly_volume and "
                    f"monthly_fraud_level stay unconstrained. If the question names a "
                    f"month, pass it - both tiers are determined by it.")

    if month is not None:
        month = _resolve_month(month)
        if merchant is None:
            _notice("month= given without merchant=, so it was used only for validation. "
                    "The volume and fraud tiers are per-merchant; pass merchant= too.")
        else:
            vol, fraud = monthly_profile(merchant, month)
            if monthly_volume is None:
                monthly_volume = vol
            if monthly_fraud_level is None:
                monthly_fraud_level = fraud

    card_scheme         = _check('card_scheme', card_scheme)
    account_type        = _check('account_type', account_type)
    capture_delay       = _check('capture_delay', capture_delay)
    monthly_fraud_level = _check('monthly_fraud_level', monthly_fraud_level)
    monthly_volume      = _check('monthly_volume', monthly_volume)
    mcc                 = _check('mcc', mcc)
    is_credit           = _check('is_credit', is_credit)
    aci                 = _check('aci', aci)
    intracountry        = _check('intracountry', intracountry)

    if card_scheme is not None and f['card_scheme'] != card_scheme: return False
    if account_type is not None and f['account_type'] and account_type not in f['account_type']: return False
    if mcc is not None and f['merchant_category_code'] and mcc not in f['merchant_category_code']: return False
    if capture_delay is not None and f['capture_delay'] is not None and f['capture_delay'] != capture_delay: return False
    if monthly_fraud_level is not None and f['monthly_fraud_level'] is not None and f['monthly_fraud_level'] != monthly_fraud_level: return False
    if monthly_volume is not None and f['monthly_volume'] is not None and f['monthly_volume'] != monthly_volume: return False
    if is_credit is not None and f['is_credit'] is not None and f['is_credit'] != is_credit: return False
    if aci is not None and f['aci'] and aci not in f['aci']: return False
    if intracountry is not None and f['intracountry'] is not None and bool(f['intracountry']) != bool(intracountry): return False
    return True


def matching_rules(**constraints):
    """All fee rules that apply under these constraints - same keywords as rule_matches
    (merchant=, month=, card_scheme=, is_credit=, aci=, ...).

    This is the safe replacement for `[r for r in fees if <hand-written condition>]`. A
    hand-written condition almost always compares a rule's field with `==`, which silently
    drops every rule where that field is a wildcard (`null` or `[]`) - is_credit is null in
    100/1000 rules, intracountry in 561/1000, capture_delay in 500/1000, monthly_fraud_level
    in 900/1000, monthly_volume in 800/1000. None of those are edge cases; excluding them is
    usually the majority of the real answer. Call this instead of writing your own filter -
    it costs one line, the same as writing the filter would, and cannot make that mistake.
    """
    result = [r for r in fees if rule_matches(r, **constraints)]
    _record_computation("matching_rules", {"constraints": constraints, "n_matched": len(result),
                                            "ids": [r["ID"] for r in result]})
    return result

