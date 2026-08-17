from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from .paths import default_hla_lookup_path

CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


def looks_like_sequence(value: object, min_len: int = 20) -> bool:
    text = str(value).strip().upper()
    return len(text) >= min_len and all(char in CANONICAL_AA for char in text)


def hla_aliases(value: object) -> list[str]:
    raw = str(value).strip()
    if not raw:
        return []
    upper = raw.upper().replace(" ", "")
    rest = upper[4:] if upper.startswith("HLA-") else upper
    rest = rest.replace("*", "")
    aliases = {upper, rest, f"HLA-{rest}"}

    # Map extended HLA fields to their two-field protein allele.
    if len(rest) >= 3 and rest[0].isalpha():
        locus, body = rest[0], rest[1:]
        fields = body.split(":")
        if len(fields) >= 2:
            two_field = f"{locus}{fields[0][:2]}:{fields[1][:2]}"
            aliases.update({two_field, f"HLA-{two_field}"})
        elif len(body) == 2:
            two_field = f"{locus}{body}:01"
            aliases.update({two_field, f"HLA-{two_field}"})
        elif len(body) >= 4:
            two_field = f"{locus}{body[:2]}:{body[2:4]}"
            aliases.update({two_field, f"HLA-{two_field}"})

    compact = rest.replace(":", "")
    aliases.add(compact)
    return sorted(aliases)


def load_hla_lookup(path: str | Path | None = None) -> dict[str, str]:
    lookup_path = Path(path or default_hla_lookup_path()).expanduser().resolve()
    if not lookup_path.exists():
        return {}
    table = pd.read_csv(lookup_path)
    hla_col = "HLA_type" if "HLA_type" in table.columns else table.columns[0]
    seq_col = "HLA_sequence" if "HLA_sequence" in table.columns else table.columns[1]
    lookup: dict[str, str] = {}
    for _, row in table.iterrows():
        seq = str(row[seq_col]).strip().upper()
        for alias in hla_aliases(row[hla_col]):
            lookup[alias] = seq
    return lookup


def resolve_hla_sequence(
    row: Mapping[str, object],
    hla_lookup: Mapping[str, str] | None = None,
) -> str:
    for col in ("hla_sequence", "HLA_sequence", "mhc_sequence", "MHC_sequence"):
        value = row.get(col)
        if value is not None and looks_like_sequence(value):
            return str(value).strip().upper()

    hla_value = None
    for col in ("hla", "HLA", "mhc", "MHC", "hla_type", "HLA_type"):
        value = row.get(col)
        if value is not None and str(value).strip():
            hla_value = value
            break

    if hla_value is None:
        raise ValueError("Row must contain hla_sequence or an HLA/MHC allele column.")
    if looks_like_sequence(hla_value):
        return str(hla_value).strip().upper()

    lookup = dict(hla_lookup or load_hla_lookup())
    for alias in hla_aliases(hla_value):
        if alias in lookup:
            return lookup[alias]
    raise KeyError(f"Could not resolve HLA pseudo-sequence for {hla_value!r}")
