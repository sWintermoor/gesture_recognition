import torch
import torch.nn as nn
import math
from typing import Tuple

class MultiHeadAttention(nn.Module):
    def __init__(self, nheads: int, d_model:int):
        if(d_model % nheads != 0):
            raise ValueError(f"Can't divide {d_model} (d_model) by {nheads} (nheads).")

        super().__init__()

        self.nheads = nheads
        self.d_head = d_model // nheads

        self.w_q = nn.Linear(d_model, d_model, bias=True)
        self.w_k = nn.Linear(d_model, d_model, bias=True)
        self.w_v = nn.Linear(d_model, d_model, bias=True)
        self.output = nn.Linear(d_model, d_model, bias=True)


    def split(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_split = q.reshape(q.size(0), q.size(1), self.nheads, self.d_head).transpose(1, 2)
        k_split = k.reshape(k.size(0), k.size(1), self.nheads, self.d_head).transpose(1, 2)
        v_split = v.reshape(v.size(0), v.size(1), self.nheads, self.d_head).transpose(1, 2)

        return q_split, k_split, v_split


    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None) -> torch.Tensor:
        score_input = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        if mask is not None:
            score_input = score_input.masked_fill(mask[:, None, None, :], float('-inf'))

        score = torch.softmax(score_input, dim=-1) 
        score = score @ v
        return score


    def forward(self, x_input, mask=None) -> torch.Tensor:
        Q = self.w_q(x_input)
        K = self.w_k(x_input)
        V = self.w_v(x_input)

        Q_split, K_split, V_split = self.split(Q, K, V)

        head_query = []

        for i in range(self.nheads):
            Q_i, K_i, V_i = Q_split[:, i, :, :], K_split[:, i, :, :], V_split[:, i, :, :]
            head = self.attention(Q_i, K_i, V_i, mask=mask)
            head_query.append(head)

        head_query = torch.stack(head_query, dim=1)

        result = head_query.transpose(1, 2).contiguous().view(x_input.size(0), x_input.size(1), -1)

        return self.output(result)


class EncodingLayer(nn.Module):
    def __init__(self, nheads:int, d_model: int, dim_ff: int, dropout_p: float=0.1):
        super().__init__()

        self.attention = MultiHeadAttention(nheads=nheads, d_model=d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff, bias=True),
            nn.GELU(),
            nn.Linear(dim_ff, d_model, bias=True)
        )

    def forward(self, x_input, mask=None):
        x = self.norm1(x_input)
        x = self.attention(x, mask)
        x = self.dropout1(x)
        x_res = x + x_input

        x = self.norm2(x_res)
        x = self.ffn(x)
        x = self.dropout2(x)
        x = x + x_res

        return x
