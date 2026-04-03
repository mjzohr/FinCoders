from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import Dataset


class TorchSequenceDataset(Dataset):
    def __init__(self, price_seq, news_seq, targets) -> None:
        self.price_seq = torch.as_tensor(price_seq, dtype=torch.float32)
        self.news_seq = torch.as_tensor(news_seq, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "price_seq": self.price_seq[index],
            "news_seq": self.news_seq[index],
            "target": self.targets[index],
        }


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class LSTMFusionModel(nn.Module):
    def __init__(
        self,
        price_dim: int,
        news_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        use_news: bool,
    ) -> None:
        super().__init__()
        self.use_news = use_news
        self.price_encoder = nn.LSTM(
            input_size=price_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        if use_news:
            self.news_encoder = nn.LSTM(
                input_size=news_dim,
                hidden_size=hidden_dim // 2,
                num_layers=1,
                batch_first=True,
            )
            fusion_dim = hidden_dim + hidden_dim // 2
        else:
            self.news_encoder = None
            fusion_dim = hidden_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = MLPHead(hidden_dim, dropout=dropout)

    def forward(self, price_seq: torch.Tensor, news_seq: torch.Tensor) -> torch.Tensor:
        _, (price_hidden, _) = self.price_encoder(price_seq)
        fused = price_hidden[-1]
        if self.use_news and self.news_encoder is not None:
            _, (news_hidden, _) = self.news_encoder(news_seq)
            fused = torch.cat([fused, news_hidden[-1]], dim=-1)
        fused = self.fusion(fused)
        return self.head(fused)


class PatchEmbedding(nn.Module):
    def __init__(self, input_dim: int, patch_len: int, d_model: int) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.projection = nn.Linear(input_dim * patch_len, d_model)

    def forward(self, series: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, feature_dim = series.shape
        remainder = seq_len % self.patch_len
        if remainder != 0:
            pad = self.patch_len - remainder
            padding = torch.zeros(batch_size, pad, feature_dim, device=series.device, dtype=series.dtype)
            series = torch.cat([padding, series], dim=1)
            seq_len = series.shape[1]

        patch_count = seq_len // self.patch_len
        patches = series.reshape(batch_size, patch_count, self.patch_len * feature_dim)
        return self.projection(patches)


class TimeXerFusionModel(nn.Module):
    """
    Practical TimeXer-style forecaster:
    patched price tokens + optional exogenous news tokens + cross attention.
    """

    def __init__(
        self,
        price_dim: int,
        news_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        patch_len: int,
        dropout: float,
        use_news: bool,
    ) -> None:
        super().__init__()
        self.use_news = use_news
        self.price_embedding = PatchEmbedding(price_dim, patch_len, d_model)
        self.news_embedding = PatchEmbedding(max(news_dim, 1), patch_len, d_model)
        self.position = nn.Parameter(torch.zeros(1, 256, d_model))

        def build_encoder(layer_count: int) -> nn.TransformerEncoder:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            return nn.TransformerEncoder(layer, num_layers=layer_count)

        self.price_encoder = build_encoder(num_layers)
        self.news_encoder = build_encoder(max(1, num_layers - 1))
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.post_norm = nn.LayerNorm(d_model)
        self.head = MLPHead(d_model, dropout=dropout)

    def forward(self, price_seq: torch.Tensor, news_seq: torch.Tensor) -> torch.Tensor:
        price_tokens = self.price_embedding(price_seq)
        price_tokens = price_tokens + self.position[:, : price_tokens.shape[1], :]
        price_tokens = self.price_encoder(price_tokens)

        if self.use_news:
            news_tokens = self.news_embedding(news_seq)
            news_tokens = news_tokens + self.position[:, : news_tokens.shape[1], :]
            news_tokens = self.news_encoder(news_tokens)
            attended, _ = self.cross_attention(price_tokens, news_tokens, news_tokens)
            price_tokens = self.post_norm(price_tokens + attended)

        pooled = price_tokens.mean(dim=1)
        return self.head(pooled)


class HuggingFacePatchTSTModel(nn.Module):
    """
    Wrapper around Hugging Face PatchTST for sequence-level regression/classification.
    Price and news features are concatenated along the channel axis.
    """

    def __init__(
        self,
        price_dim: int,
        news_dim: int,
        context_length: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        patch_len: int,
        dropout: float,
        use_news: bool,
        task: str,
    ) -> None:
        super().__init__()
        try:
            from transformers import PatchTSTConfig, PatchTSTForClassification, PatchTSTForRegression
        except ImportError as exc:
            raise ImportError(
                "Hugging Face PatchTST requires `transformers` to be installed. "
                "Run `pip install -r requirements-training.txt`."
            ) from exc

        self.use_news = use_news
        self.task = task
        num_input_channels = price_dim + (news_dim if use_news else 0)
        patch_len = max(1, min(patch_len, context_length))
        patch_stride = max(1, patch_len // 2)

        config = PatchTSTConfig(
            num_input_channels=num_input_channels,
            context_length=context_length,
            prediction_length=1,
            patch_length=patch_len,
            patch_stride=patch_stride,
            num_hidden_layers=num_layers,
            d_model=d_model,
            num_attention_heads=nhead,
            ffn_dim=d_model * 4,
            attention_dropout=dropout,
            positional_dropout=dropout,
            ff_dropout=dropout,
            head_dropout=dropout,
            use_cls_token=True,
            pooling_type="mean",
            norm_type="layernorm",
            scaling="std",
            num_targets=1,
            loss="mse",
            problem_type="single_label_classification" if task == "classification" else "regression",
        )

        if task == "classification":
            self.model = PatchTSTForClassification(config)
        else:
            self.model = PatchTSTForRegression(config)

    def forward(self, price_seq: torch.Tensor, news_seq: torch.Tensor) -> torch.Tensor:
        past_values = torch.cat([price_seq, news_seq], dim=-1) if self.use_news else price_seq
        past_values = torch.nan_to_num(past_values, nan=0.0, posinf=0.0, neginf=0.0)
        observed_mask = torch.ones_like(past_values, dtype=torch.bool)
        outputs = self.model(past_values=past_values, past_observed_mask=observed_mask)

        if self.task == "classification":
            logits = outputs.prediction_logits
        else:
            logits = outputs.regression_outputs

        if logits.ndim == 2 and logits.shape[-1] == 1:
            return logits.squeeze(-1)
        return logits.squeeze()


@dataclass(slots=True)
class NeuralBatch:
    price_seq: torch.Tensor
    news_seq: torch.Tensor
    target: torch.Tensor
