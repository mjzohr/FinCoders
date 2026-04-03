from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(slots=True)
class PanelBundle:
    panel: pd.DataFrame
    price_feature_cols: list[str]
    news_feature_cols: list[str]


@dataclass(slots=True)
class SequenceSamples:
    price_seq: np.ndarray
    news_seq: np.ndarray
    targets: np.ndarray
    dates: np.ndarray
    tickers: np.ndarray

    def subset(self, mask: np.ndarray) -> "SequenceSamples":
        return SequenceSamples(
            price_seq=self.price_seq[mask],
            news_seq=self.news_seq[mask],
            targets=self.targets[mask],
            dates=self.dates[mask],
            tickers=self.tickers[mask],
        )

    @property
    def size(self) -> int:
        return int(self.targets.shape[0])


@dataclass(slots=True)
class SplitMasks:
    name: str
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray


@dataclass(slots=True)
class SequenceStandardizer:
    price_mean: np.ndarray
    price_std: np.ndarray
    news_mean: np.ndarray
    news_std: np.ndarray

    def transform(self, samples: SequenceSamples) -> SequenceSamples:
        price_seq = (samples.price_seq - self.price_mean) / self.price_std
        news_seq = (samples.news_seq - self.news_mean) / self.news_std
        return SequenceSamples(
            price_seq=price_seq.astype(np.float32),
            news_seq=news_seq.astype(np.float32),
            targets=samples.targets.astype(np.float32),
            dates=samples.dates,
            tickers=samples.tickers,
        )


class InsufficientDataError(ValueError):
    pass


def load_price_panel(path: Path) -> pd.DataFrame:
    price_wide = pd.read_csv(path)
    price_wide = price_wide.loc[:, ~price_wide.columns.duplicated()].copy()
    price_wide["Date"] = pd.to_datetime(price_wide["Date"], format="%m/%d/%Y", errors="coerce")
    price_wide = price_wide.dropna(subset=["Date"]).sort_values("Date")

    price_columns = [column for column in price_wide.columns if column.endswith("_Price")]
    records: list[pd.DataFrame] = []

    for price_column in price_columns:
        ticker = price_column[:-6]
        pct_column = f"{ticker}_PctChange"
        columns = ["Date", price_column]
        rename_map = {"Date": "date", price_column: "price"}
        if pct_column in price_wide.columns:
            columns.append(pct_column)
            rename_map[pct_column] = "pct_change"

        ticker_frame = price_wide[columns].rename(columns=rename_map).copy()
        ticker_frame["ticker"] = ticker
        records.append(ticker_frame)

    panel = pd.concat(records, ignore_index=True)
    panel["price"] = pd.to_numeric(panel["price"], errors="coerce")
    if "pct_change" in panel.columns:
        panel["pct_change"] = pd.to_numeric(panel["pct_change"], errors="coerce") / 100.0
    else:
        panel["pct_change"] = np.nan
    panel = panel.dropna(subset=["date", "price"])
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    grouped = panel.groupby("ticker", group_keys=False)
    panel["return_1d"] = grouped["price"].pct_change()
    panel["pct_change"] = panel["pct_change"].fillna(panel["return_1d"])
    panel["log_return_1d"] = np.log1p(panel["return_1d"].clip(lower=-0.95))

    for window in (3, 5, 10, 21):
        panel[f"ret_mean_{window}"] = grouped["return_1d"].transform(
            lambda series, w=window: series.rolling(w).mean()
        )
        panel[f"ret_std_{window}"] = grouped["return_1d"].transform(
            lambda series, w=window: series.rolling(w).std()
        )
        panel[f"momentum_{window}"] = grouped["price"].transform(
            lambda series, w=window: series / series.rolling(w).mean() - 1.0
        )

    rolling_mean_21 = grouped["price"].transform(lambda series: series.rolling(21).mean())
    rolling_std_21 = grouped["price"].transform(lambda series: series.rolling(21).std())
    rolling_max_21 = grouped["price"].transform(lambda series: series.rolling(21).max())
    panel["price_zscore_21"] = (panel["price"] - rolling_mean_21) / rolling_std_21.replace(0.0, np.nan)
    panel["drawdown_21"] = panel["price"] / rolling_max_21 - 1.0
    return panel.reset_index(drop=True)


def load_news_panel(path: Path | None) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=["ticker", "date"])

    news = pd.read_csv(path, on_bad_lines="skip")
    lowered = {column.lower(): column for column in news.columns}

    ticker_col = lowered.get("ticker")
    date_col = lowered.get("date")
    title_col = lowered.get("title")
    excerpt_col = lowered.get("excerpt")
    summary_col = lowered.get("summary")
    title_sent_col = lowered.get("title sentiment score")
    excerpt_sent_col = lowered.get("excerpt sentiment score")
    summary_sent_col = lowered.get("summary sentiment score")

    if ticker_col is None or date_col is None:
        return pd.DataFrame(columns=["ticker", "date"])

    base = pd.DataFrame(
        {
            "ticker": news[ticker_col].astype(str).str.strip(),
            "date": pd.to_datetime(news[date_col], format="%m/%d/%Y", errors="coerce"),
            "title": news[title_col].astype(str) if title_col else "",
            "excerpt": news[excerpt_col].astype(str) if excerpt_col else "",
            "summary": news[summary_col].astype(str) if summary_col else "",
        }
    )
    base = base.dropna(subset=["date"])
    base = base[base["ticker"].ne("")]

    for column_name, source in (
        ("title_sentiment", title_sent_col),
        ("excerpt_sentiment", excerpt_sent_col),
        ("summary_sentiment", summary_sent_col),
    ):
        if source:
            base[column_name] = pd.to_numeric(news.loc[base.index, source], errors="coerce")
        else:
            base[column_name] = np.nan

    base["title_len"] = base["title"].str.len().astype(float)
    base["excerpt_len"] = base["excerpt"].str.len().astype(float)
    base["summary_len"] = base["summary"].str.len().astype(float)
    base["article_sentiment"] = base[
        ["title_sentiment", "excerpt_sentiment", "summary_sentiment"]
    ].mean(axis=1, skipna=True)
    base["positive_article"] = (base["article_sentiment"] > 0.10).astype(float)
    base["negative_article"] = (base["article_sentiment"] < -0.10).astype(float)
    base["article_count"] = 1.0

    aggregated = (
        base.groupby(["ticker", "date"])
        .agg(
            article_count=("article_count", "sum"),
            article_sentiment_mean=("article_sentiment", "mean"),
            article_sentiment_std=("article_sentiment", "std"),
            article_sentiment_min=("article_sentiment", "min"),
            article_sentiment_max=("article_sentiment", "max"),
            positive_article_ratio=("positive_article", "mean"),
            negative_article_ratio=("negative_article", "mean"),
            title_sentiment_mean=("title_sentiment", "mean"),
            excerpt_sentiment_mean=("excerpt_sentiment", "mean"),
            summary_sentiment_mean=("summary_sentiment", "mean"),
            title_len_mean=("title_len", "mean"),
            excerpt_len_mean=("excerpt_len", "mean"),
            summary_len_mean=("summary_len", "mean"),
        )
        .reset_index()
    )

    aggregated["has_news"] = (aggregated["article_count"] > 0).astype(float)
    numeric_cols = [column for column in aggregated.columns if column not in {"ticker", "date"}]
    aggregated[numeric_cols] = aggregated[numeric_cols].astype(float).fillna(0.0)
    return aggregated


def build_model_panel(
    price_csv: Path,
    news_csv: Path | None,
    news_lag_days: int,
    horizons: tuple[int, ...],
) -> PanelBundle:
    panel = load_price_panel(price_csv)
    news = load_news_panel(news_csv)

    news_feature_cols: list[str] = []
    if not news.empty:
        panel = panel.merge(news, on=["ticker", "date"], how="left")
        news_feature_cols = [column for column in news.columns if column not in {"ticker", "date"}]
        panel[news_feature_cols] = panel.groupby("ticker", group_keys=False)[news_feature_cols].shift(
            news_lag_days
        )
        panel[news_feature_cols] = panel[news_feature_cols].fillna(0.0)

    grouped = panel.groupby("ticker", group_keys=False)
    for horizon in horizons:
        panel[f"target_return_h{horizon}"] = grouped["price"].shift(-horizon) / panel["price"] - 1.0

    price_feature_cols = [
        "price",
        "pct_change",
        "return_1d",
        "log_return_1d",
        "ret_mean_3",
        "ret_mean_5",
        "ret_mean_10",
        "ret_mean_21",
        "ret_std_5",
        "ret_std_10",
        "ret_std_21",
        "momentum_5",
        "momentum_10",
        "momentum_21",
        "price_zscore_21",
        "drawdown_21",
    ]

    panel[price_feature_cols] = panel[price_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if news_feature_cols:
        panel[news_feature_cols] = panel[news_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return PanelBundle(panel=panel, price_feature_cols=price_feature_cols, news_feature_cols=news_feature_cols)


def create_sequence_samples(
    bundle: PanelBundle,
    horizon: int,
    lookback: int,
    task: str,
    flat_threshold: float = 0.0,
    use_news: bool = True,
    target_clip: float | None = None,
) -> SequenceSamples:
    target_column = f"target_return_h{horizon}"
    price_sequences: list[np.ndarray] = []
    news_sequences: list[np.ndarray] = []
    targets: list[float] = []
    dates: list[np.datetime64] = []
    tickers: list[str] = []

    news_columns = bundle.news_feature_cols if bundle.news_feature_cols and use_news else []
    zero_news_width = max(len(news_columns), 1)

    for ticker, group in bundle.panel.groupby("ticker"):
        group = group.sort_values("date").reset_index(drop=True)
        price_values = group[bundle.price_feature_cols].to_numpy(dtype=np.float32)
        if news_columns:
            news_values = group[news_columns].to_numpy(dtype=np.float32)
        else:
            news_values = np.zeros((len(group), zero_news_width), dtype=np.float32)
        target_values = group[target_column].to_numpy(dtype=np.float32)
        date_values = group["date"].to_numpy()

        for end_idx in range(lookback - 1, len(group)):
            target_value = target_values[end_idx]
            if np.isnan(target_value):
                continue
            if target_clip is not None:
                target_value = float(np.clip(target_value, -target_clip, target_clip))

            start_idx = end_idx - lookback + 1
            price_sequences.append(price_values[start_idx : end_idx + 1])
            news_sequences.append(news_values[start_idx : end_idx + 1])
            if task == "classification":
                targets.append(1.0 if target_value > flat_threshold else 0.0)
            else:
                targets.append(float(target_value))
            dates.append(date_values[end_idx])
            tickers.append(ticker)

    return SequenceSamples(
        price_seq=np.asarray(price_sequences, dtype=np.float32),
        news_seq=np.asarray(news_sequences, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.float32),
        dates=np.asarray(dates),
        tickers=np.asarray(tickers),
    )


def create_latest_samples(
    bundle: PanelBundle,
    lookback: int,
    use_news: bool = True,
    as_of_date: pd.Timestamp | None = None,
) -> SequenceSamples:
    price_sequences: list[np.ndarray] = []
    news_sequences: list[np.ndarray] = []
    targets: list[float] = []
    dates: list[np.datetime64] = []
    tickers: list[str] = []

    news_columns = bundle.news_feature_cols if bundle.news_feature_cols and use_news else []
    zero_news_width = max(len(news_columns), 1)
    cutoff = pd.Timestamp(as_of_date) if as_of_date is not None else None

    for ticker, group in bundle.panel.groupby("ticker"):
        group = group.sort_values("date").reset_index(drop=True)
        if cutoff is not None:
            group = group.loc[group["date"] <= cutoff].reset_index(drop=True)
        if len(group) < lookback:
            continue

        end_idx = len(group) - 1
        start_idx = end_idx - lookback + 1
        price_values = group[bundle.price_feature_cols].to_numpy(dtype=np.float32)
        if news_columns:
            news_values = group[news_columns].to_numpy(dtype=np.float32)
        else:
            news_values = np.zeros((len(group), zero_news_width), dtype=np.float32)

        price_sequences.append(price_values[start_idx : end_idx + 1])
        news_sequences.append(news_values[start_idx : end_idx + 1])
        targets.append(np.nan)
        dates.append(group["date"].iloc[end_idx].to_datetime64())
        tickers.append(ticker)

    return SequenceSamples(
        price_seq=np.asarray(price_sequences, dtype=np.float32),
        news_seq=np.asarray(news_sequences, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.float32),
        dates=np.asarray(dates),
        tickers=np.asarray(tickers),
    )


def build_tabular_matrix(samples: SequenceSamples, include_news: bool) -> np.ndarray:
    price_flat = samples.price_seq.reshape(samples.price_seq.shape[0], -1)
    price_summary = np.concatenate(
        [
            samples.price_seq[:, -1, :],
            samples.price_seq.mean(axis=1),
            samples.price_seq.std(axis=1),
            samples.price_seq.min(axis=1),
            samples.price_seq.max(axis=1),
        ],
        axis=1,
    )

    matrices = [price_flat, price_summary]
    if include_news:
        news_flat = samples.news_seq.reshape(samples.news_seq.shape[0], -1)
        news_summary = np.concatenate(
            [
                samples.news_seq[:, -1, :],
                samples.news_seq.mean(axis=1),
                samples.news_seq.std(axis=1),
                samples.news_seq.min(axis=1),
                samples.news_seq.max(axis=1),
            ],
            axis=1,
        )
        matrices.extend([news_flat, news_summary])

    return np.concatenate(matrices, axis=1).astype(np.float32)


def make_splits(
    sample_dates: np.ndarray,
    eval_mode: str,
    min_train_days: int,
    val_days: int,
    test_days: int,
    step_days: int,
) -> list[SplitMasks]:
    unique_dates = np.sort(np.unique(sample_dates))
    required_dates = min_train_days + val_days + test_days
    if len(unique_dates) < required_dates:
        raise InsufficientDataError(
            "Not enough dated samples for the requested split sizes. "
            f"available={len(unique_dates)} required={required_dates}. "
            "Try a shorter lookback, a shorter horizon, or smaller split windows."
        )

    def build_split(name: str, train_dates: np.ndarray, val_dates: np.ndarray, test_dates: np.ndarray) -> SplitMasks:
        return SplitMasks(
            name=name,
            train_mask=np.isin(sample_dates, train_dates),
            val_mask=np.isin(sample_dates, val_dates),
            test_mask=np.isin(sample_dates, test_dates),
        )

    if eval_mode == "holdout":
        test_start = len(unique_dates) - test_days
        val_start = test_start - val_days
        return [
            build_split(
                name="holdout",
                train_dates=unique_dates[:val_start],
                val_dates=unique_dates[val_start:test_start],
                test_dates=unique_dates[test_start:],
            )
        ]

    splits: list[SplitMasks] = []
    split_index = 0
    for test_start in range(min_train_days + val_days, len(unique_dates) - test_days + 1, step_days):
        val_start = test_start - val_days
        splits.append(
            build_split(
                name=f"fold_{split_index:02d}",
                train_dates=unique_dates[:val_start],
                val_dates=unique_dates[val_start:test_start],
                test_dates=unique_dates[test_start : test_start + test_days],
            )
        )
        split_index += 1

    return splits


def fit_standardizer(samples: SequenceSamples) -> SequenceStandardizer:
    price_mean = samples.price_seq.mean(axis=(0, 1), keepdims=True)
    price_std = samples.price_seq.std(axis=(0, 1), keepdims=True)
    news_mean = samples.news_seq.mean(axis=(0, 1), keepdims=True)
    news_std = samples.news_seq.std(axis=(0, 1), keepdims=True)

    price_std = np.where(price_std < 1e-6, 1.0, price_std)
    news_std = np.where(news_std < 1e-6, 1.0, news_std)
    return SequenceStandardizer(
        price_mean=price_mean.astype(np.float32),
        price_std=price_std.astype(np.float32),
        news_mean=news_mean.astype(np.float32),
        news_std=news_std.astype(np.float32),
    )
