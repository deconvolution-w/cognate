import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        scale = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale)
        self.register_buffer("pe", encoding.unsqueeze(0))

    def forward(self, hidden):
        hidden = hidden + self.pe[:, : hidden.size(1), :]
        return self.dropout(hidden)


class CognateEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        nhead,
        num_layers,
        dropout=0.1,
        esm_model=None,
        esm_trainable=False,
        esm_scalar_mix_k=4,
        use_segment_embedding=True,
    ):
        super().__init__()
        self.esm_model = esm_model
        self.esm_trainable = bool(esm_trainable)
        self.segment_embedding = (
            nn.Embedding(3, d_model) if use_segment_embedding else None
        )
        self.esm_mix_k = max(1, int(esm_scalar_mix_k))
        if self.esm_model is not None:
            esm_output_dim = self.esm_model.config.hidden_size
            self.esm_layer_weights = nn.Parameter(
                torch.zeros(self.esm_mix_k, dtype=torch.float32)
            )
            self.esm_layer_gamma = nn.Parameter(torch.ones(1, dtype=torch.float32))
            self.esm_projection = nn.Linear(esm_output_dim, d_model)
            self.esm_gate = nn.Linear(d_model * 2, 1)

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.d_model = d_model

    def _esm_scalar_mix(self, esm_tokens, esm_mask):
        attention_mask = None
        if esm_mask is not None:
            attention_mask = (
                (~esm_mask).long() if esm_mask.dtype == torch.bool else esm_mask.long()
            )

        # Avoid retaining all ESM hidden states when only the final layer is used.
        if self.esm_mix_k == 1:
            if self.esm_trainable:
                outputs = self.esm_model(
                    esm_tokens,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    return_dict=True,
                )
            else:
                self.esm_model.eval()
                with torch.no_grad():
                    outputs = self.esm_model(
                        esm_tokens,
                        attention_mask=attention_mask,
                        output_hidden_states=False,
                        return_dict=True,
                    )
            mixed = self.esm_layer_gamma * outputs.last_hidden_state
            return mixed[:, 1:-1, :]

        if self.esm_trainable:
            outputs = self.esm_model(
                esm_tokens,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            self.esm_model.eval()
            with torch.no_grad():
                outputs = self.esm_model(
                    esm_tokens,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
        hidden_states = outputs.hidden_states[-self.esm_mix_k :]
        layer_weights = torch.softmax(self.esm_layer_weights, dim=0)
        mixed = 0.0
        for weight, hidden_state in zip(layer_weights, hidden_states):
            mixed = mixed + weight * hidden_state
        mixed = self.esm_layer_gamma * mixed
        return mixed[:, 1:-1, :]

    def forward(
        self,
        src,
        src_key_padding_mask=None,
        segment_ids=None,
        esm_hla_tokens=None,
        esm_hla_mask=None,
        esm_pep_tokens=None,
        esm_pep_mask=None,
        hla_lens=None,
        pep_lens=None,
    ):
        hidden = self.embedding(src) * math.sqrt(self.d_model)
        if segment_ids is not None and self.segment_embedding is not None:
            hidden = hidden + self.segment_embedding(segment_ids)
        hidden = self.pos_encoder(hidden)

        if (
            self.esm_model is not None
            and esm_hla_tokens is not None
            and esm_pep_tokens is not None
            and hla_lens is not None
            and pep_lens is not None
        ):
            hla_features = self._esm_scalar_mix(esm_hla_tokens, esm_hla_mask)
            peptide_features = self._esm_scalar_mix(esm_pep_tokens, esm_pep_mask)
            batch_size, sequence_length = src.shape
            aligned_features = torch.zeros(
                (batch_size, sequence_length, hla_features.size(-1)),
                device=src.device,
                dtype=hla_features.dtype,
            )
            for row, (hla_length, peptide_length) in enumerate(
                zip(hla_lens.tolist(), pep_lens.tolist())
            ):
                if hla_length > 0:
                    aligned_features[row, 1 : 1 + hla_length, :] = hla_features[
                        row, :hla_length, :
                    ]
                if peptide_length > 0:
                    peptide_start = 2 + hla_length
                    aligned_features[
                        row,
                        peptide_start : peptide_start + peptide_length,
                        :,
                    ] = peptide_features[row, :peptide_length, :]
            projected_features = self.esm_projection(aligned_features)
            gate = torch.sigmoid(
                self.esm_gate(torch.cat([hidden, projected_features], dim=-1))
            )
            hidden = hidden + gate * projected_features

        return self.transformer_encoder(
            hidden,
            src_key_padding_mask=src_key_padding_mask,
        )


class CognateDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.linear = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        tgt_pad_mask=None,
        memory_key_padding_mask=None,
    ):
        hidden = self.embedding(tgt) * math.sqrt(self.d_model)
        hidden = self.pos_encoder(hidden)
        decoded = self.transformer_decoder(
            hidden,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.linear(decoded)
