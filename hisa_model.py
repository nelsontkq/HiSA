import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class SparseMultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, sparsity=0.9):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = dropout
        self.sparsity = sparsity

    def forward(self, query, key, value):
        batch_size, seq_len, _ = query.shape

        Q = self.W_q(query).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)

        mask = (
            torch.rand((batch_size, self.nhead, seq_len, seq_len), device=query.device)
            > self.sparsity
        )

        attn_output = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
        )

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )
        return self.W_o(attn_output)


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, 4 * d_model)
        self.linear2 = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class HiSABlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = SparseMultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout)

    def forward(self, x):
        x = self.norm1(x + self.self_attn(x, x, x))
        x = self.norm2(x + self.ff(x))
        return x


class HiSAForPTB(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer_blocks = nn.ModuleList(
            [HiSABlock(d_model, nhead, dropout) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        for block in self.transformer_blocks:
            x = block(x)
        return self.fc_out(x)


class HiSAGPT(nn.Module):
    def __init__(
        self, vocab_size, d_model, nhead, num_layers, dropout, use_checkpoint=False
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.d_model = d_model
        self.transformer_blocks = nn.ModuleList(
            [HiSABlock(d_model, nhead, dropout) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size
        self.use_checkpoint = use_checkpoint

    def forward(self, src):
        if src.max() >= self.vocab_size or src.min() < 0:
            raise ValueError(
                f"Input contains token IDs outside the valid range [0, {self.vocab_size-1}]"
            )

        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        for block in self.transformer_blocks:
            if self.use_checkpoint:
                x = checkpoint(block, x)
            else:
                x = block(x)
        return self.fc_out(x)

    def generate(self, input_ids, max_length):
        for _ in range(max_length):
            output = self(input_ids)
            next_token = output[:, -1, :].argmax(dim=-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
        return input_ids
