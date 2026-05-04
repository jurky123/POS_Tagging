"""
定义用于词性标注的BERT模型架构。
"""
import math

import torch
import torch.nn as nn


class BERT_POS_Tagging_Model(nn.Module):
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

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        # input_ids: (batch_size, seq_len)
        _, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        # 先线性层把token_id转换为embedding后的表示
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        # 再加上位置编码
        x = x + self.position_embedding(positions)
        x = self.dropout(x)

        # 经过多层Transformer Encoder，得到每个token的表示
        # 注意残差连接和层归一化还有Dropout（TransformerEncoderLayer 内部已包含）
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # 再经过线性层，输出每个token的POS标签得分
        logits = self.classifier(x)

        return logits

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_model(self, path: str):
        self.load_state_dict(torch.load(path))
