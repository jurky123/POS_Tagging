"""
定义用于词性标注的Transformer模型架构，使用RoPE位置编码。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                mask: torch.Tensor | None = None):
        bsz, seq_len, _ = x.shape
        cos = cos[:seq_len].unsqueeze(0).unsqueeze(2)  # (1, seq, 1, head_dim/2)
        sin = sin[:seq_len].unsqueeze(0).unsqueeze(2)

        # 先添加位置编码（RoPE：对q和k做旋转）
        q = self.q_proj(x).view(bsz, seq_len, self.nhead, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.nhead, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.nhead, self.head_dim)

        def rotate(t):
            t_even, t_odd = t[..., 0::2], t[..., 1::2]
            t_rot_even = t_even * cos - t_odd * sin
            t_rot_odd = t_even * sin + t_odd * cos
            return torch.stack([t_rot_even, t_rot_odd], dim=-1).flatten(-2)

        q = rotate(q).transpose(1, 2)  # (bsz, nhead, seq, head_dim)
        k = rotate(k).transpose(1, 2)
        v = v.transpose(1, 2)

        # 进行多头自注意力
        scale = self.head_dim ** 0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        attn_out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        attn_out = self.out_proj(attn_out)

        # 归一化和残差连接
        x = self.norm1(x + attn_out)

        # 线性层
        x = self.norm2(x + self.dropout(self.linear2(F.gelu(self.linear1(x)))))
        return x
        


class BERT_POS_Tagging_Model(nn.Module):
    """基于Transformer Encoder的词性标注模型，使用RoPE位置编码。"""

    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_labels)

        # 预计算RoPE频率
        head_dim = d_model // nhead
        theta = 10000.0
        i = torch.arange(0, head_dim, 2).float()
        freqs = 1.0 / (theta ** (i / head_dim))
        positions = torch.arange(max_len).float()
        angles = torch.outer(positions, freqs)  # (max_len, head_dim/2)
        self.register_buffer('rope_cos', torch.cos(angles))
        self.register_buffer('rope_sin', torch.sin(angles))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        _, seq_len = input_ids.shape

        x = self.token_embedding(input_ids)
        x = self.dropout(x)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)

        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin, key_padding_mask)

        x = self.final_norm(x)
        logits = self.classifier(x)
        return logits

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_model(self, path: str):
        self.load_state_dict(torch.load(path))
