from __future__ import annotations

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
SEP_TOKEN = "<sep>"
TAG_CDR3 = "<tag_cdr3>"
TAG_EPI = "<tag_epitope>"
TAG_HLA = "<tag_hla>"

PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3


class AminoAcidTokenizer:
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    special_tokens = [
        PAD_TOKEN,
        UNK_TOKEN,
        SOS_TOKEN,
        EOS_TOKEN,
        SEP_TOKEN,
        TAG_CDR3,
        TAG_EPI,
        TAG_HLA,
    ]

    def __init__(self) -> None:
        self.vocab = self.special_tokens + self.amino_acids
        self.token2id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id2token = {idx: token for idx, token in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    def encode(self, sequence: str, add_special_tokens: bool = True) -> list[int]:
        text = "" if sequence is None else str(sequence).strip().upper()
        if not text or text == "NAN":
            return []
        ids = [self.token2id.get(char, UNK_IDX) for char in text]
        if add_special_tokens:
            return [SOS_IDX, *ids, EOS_IDX]
        return ids

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        tokens: list[str] = []
        for value in token_ids:
            idx = int(value.item()) if hasattr(value, "item") else int(value)
            token = self.id2token.get(idx, UNK_TOKEN)
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)
        return "".join(tokens)

    def get_token_id(self, token: str) -> int:
        return self.token2id.get(token, UNK_IDX)
