"""Small display helpers shared by the pages and scripts (rupee amounts with Indian digit grouping)."""
from __future__ import annotations

RUPEE = chr(0x20B9)          # the rupee sign, built from its code number so this file stays plain ASCII


def group_indian(number: int) -> str:
    """1234567 -> '12,34,567'  (the last three digits, then groups of two)."""
    digits = str(abs(int(number)))
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        digits = ",".join(parts + [tail])
    return ("-" if number < 0 else "") + digits


def inr(amount: float, symbol: str = RUPEE) -> str:
    """inr(1234567) -> '<rupee sign>12,34,567'.  Use symbol='Rs ' for a terminal that cannot show the rupee sign."""
    return symbol + group_indian(round(amount))
