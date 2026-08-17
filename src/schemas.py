from __future__ import annotations

from pathlib import Path

import pandas as pd


ALIASES = {
    "sample_id": ("sample_id", "id", "name"),
    "hla": ("hla", "HLA", "mhc", "MHC", "hla_type", "HLA_type"),
    "hla_sequence": ("hla_sequence", "HLA_sequence", "mhc_sequence", "MHC_sequence"),
    "peptide": ("peptide", "epitope", "epi", "antigen"),
    "cdr3": ("cdr3", "CDR3", "tcr", "TCR"),
    "target_a": ("target_a", "mut_a", "peptide_a", "antigen_a"),
    "target_b": ("target_b", "mut_b", "peptide_b", "antigen_b"),
    "wt_a": ("wt_a", "wildtype_a", "wild_type_a"),
    "wt_b": ("wt_b", "wildtype_b", "wild_type_b"),
    "wt": ("wt", "wildtype", "wild_type"),
    "variant": ("variant", "mutant", "escape_variant", "peptide"),
}


def read_csv(path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    path = Path(path)
    separator = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    table = pd.read_csv(path, sep=separator).fillna("")
    if max_rows is not None and int(max_rows) > 0:
        table = table.head(int(max_rows)).copy()
    return table


def first_existing_column(
    table: pd.DataFrame,
    canonical: str,
    required: bool = True,
) -> str | None:
    for name in ALIASES.get(canonical, (canonical,)):
        if name in table.columns:
            return name
    if required:
        raise ValueError(
            f"Missing required column {canonical!r}. Accepted aliases: "
            f"{', '.join(ALIASES.get(canonical, (canonical,)))}"
        )
    return None
