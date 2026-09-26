"""Shim — the canonical module now lives at Agents/fee_core.py.

Loaded by path to avoid this file shadowing its own import target.
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_fee_core_canonical", Path(__file__).resolve().parent.parent / "fee_core.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

fees = _mod.fees
merchants = _mod.merchants
MCC_BY_DESC = _mod.MCC_BY_DESC
pay = _mod.pay
rule_matches = _mod.rule_matches
fee_value = _mod.fee_value
capture_delay_bucket = _mod.capture_delay_bucket
volume_tier = _mod.volume_tier
fraud_tier = _mod.fraud_tier
monthly_profile = _mod.monthly_profile
