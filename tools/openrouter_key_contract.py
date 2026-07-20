#!/usr/bin/env python3
"""Frozen OpenRouter key-contract modes for the E12 live experiment."""
from __future__ import annotations

from decimal import Decimal


STRICT_KEY_CONTRACT = "strict_server_cap_v1"
E12_MARKETPLACE_KEY_CONTRACT = "e12_marketplace_deepinfra_no_byok_v1"
KEY_CONTRACT_MODES = frozenset({
    STRICT_KEY_CONTRACT,
    E12_MARKETPLACE_KEY_CONTRACT,
})
AUTHORIZED_BUDGET_USD = Decimal("3")
E12_MARKETPLACE_KEY_LIMIT_USD = Decimal("5")


def key_limit_max_usd(mode: str) -> Decimal:
    """Return the only accepted server limit for a frozen contract mode."""
    if mode == STRICT_KEY_CONTRACT:
        return AUTHORIZED_BUDGET_USD
    if mode == E12_MARKETPLACE_KEY_CONTRACT:
        return E12_MARKETPLACE_KEY_LIMIT_USD
    raise ValueError("unknown OpenRouter key contract mode")


def include_byok_in_limit_required(mode: str) -> bool:
    """Return the exact key-metadata bit required by ``mode``."""
    if mode == STRICT_KEY_CONTRACT:
        return True
    if mode == E12_MARKETPLACE_KEY_CONTRACT:
        # This exception is safe only because the E12 request path is separately
        # frozen to OpenRouter marketplace DeepInfra with no BYOK and because
        # the independent $3 stage-delta gates remain authoritative.
        return False
    raise ValueError("unknown OpenRouter key contract mode")


def validate_key_contract_mode(mode: str) -> str:
    if mode not in KEY_CONTRACT_MODES:
        raise ValueError("unknown OpenRouter key contract mode")
    return mode
