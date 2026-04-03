from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

ModelName = Literal["lightgbm", "lstm", "timexer", "hf_patchtst"]
TaskName = Literal["regression", "classification"]
ModalityName = Literal["price", "price_news"]
EvalModeName = Literal["holdout", "walkforward"]

DEFAULT_HORIZONS = (1, 5, 21)


@dataclass(slots=True)
class ExperimentConfig:
    price_csv: Path = Path("data/dates_on_left_stock_data.csv")
    news_csv: Path | None = Path("data/news_all_sentiment.csv")
    output_dir: Path = Path("artifacts")
    model_name: ModelName = "lightgbm"
    task: TaskName = "regression"
    modalities: ModalityName = "price_news"
    horizon: int = 1
    lookback: int = 30
    news_lag_days: int = 1
    eval_mode: EvalModeName = "walkforward"
    min_train_days: int = 126
    val_days: int = 21
    test_days: int = 21
    step_days: int = 21
    batch_size: int = 128
    epochs: int = 40
    patience: int = 7
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.10
    d_model: int = 128
    nhead: int = 4
    transformer_layers: int = 3
    patch_len: int = 5
    num_boost_round: int = 500
    early_stopping_rounds: int = 50
    flat_threshold: float = 0.0
    target_clip: float = 0.30
    seed: int = 7
    device: str = "cuda"
    num_workers: int = 0

    def run_name(self) -> str:
        return (
            f"{self.model_name}_h{self.horizon}_{self.modalities}_{self.task}"
            f"_lb{self.lookback}_lag{self.news_lag_days}_{self.eval_mode}_s{self.seed}"
        )

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, Path):
                data[key] = str(value)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ExperimentConfig":
        payload = dict(data)
        for key in ("price_csv", "news_csv", "output_dir"):
            if payload.get(key) is not None:
                payload[key] = Path(str(payload[key]))
        return cls(**payload)
