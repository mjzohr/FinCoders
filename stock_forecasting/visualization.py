from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_price_panel


def choose_random_tickers(
    predictions_df: pd.DataFrame,
    n_tickers: int = 5,
    seed: int = 7,
    min_points: int = 8,
) -> list[str]:
    if predictions_df.empty:
        return []
    counts = predictions_df.groupby("ticker").size()
    eligible = counts.loc[counts >= min_points].index.to_list()
    if not eligible:
        eligible = counts.index.to_list()
    if not eligible:
        return []

    rng = np.random.default_rng(seed)
    sample_size = min(n_tickers, len(eligible))
    selected = rng.choice(np.array(sorted(eligible), dtype=object), size=sample_size, replace=False)
    return selected.tolist()


def history_label_for_horizon(horizon: int) -> str:
    return {1: "Daily", 5: "Weekly", 21: "Monthly"}.get(horizon, f"Horizon {horizon}")


def resample_rule_for_horizon(horizon: int) -> str | None:
    if horizon <= 1:
        return None
    if horizon <= 5:
        return "W-FRI"
    return "ME"


def _resample_history(frame: pd.DataFrame, value_columns: list[str], horizon: int) -> pd.DataFrame:
    resample_rule = resample_rule_for_horizon(horizon)
    if resample_rule is None:
        return frame.sort_values("date").reset_index(drop=True)

    tmp = frame.sort_values("date").set_index("date")[value_columns]
    resampled = tmp.resample(resample_rule).last().dropna(how="all").reset_index()
    return resampled


def build_close_derived_ohlc(history_frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    frame = history_frame.sort_values("date").copy()
    resample_rule = resample_rule_for_horizon(horizon)

    if resample_rule is None:
        frame["open"] = frame["price"].shift(1).fillna(frame["price"])
        frame["high"] = frame[["open", "price"]].max(axis=1)
        frame["low"] = frame[["open", "price"]].min(axis=1)
        frame["close"] = frame["price"]
        return frame[["date", "open", "high", "low", "close"]].reset_index(drop=True)

    series = frame.set_index("date")["price"]
    aggregated = series.resample(resample_rule).agg(["first", "max", "min", "last"]).dropna()
    aggregated = aggregated.rename(
        columns={
            "first": "open",
            "max": "high",
            "min": "low",
            "last": "close",
        }
    ).reset_index()
    return aggregated


def prepare_stock_history_forecast_frame(
    predictions_df: pd.DataFrame,
    price_panel: pd.DataFrame,
    ticker: str,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    history = price_panel.loc[price_panel["ticker"] == ticker, ["date", "price"]].copy()
    history = history.sort_values("date").reset_index(drop=True)
    if history.empty:
        return pd.DataFrame(), pd.DataFrame()

    history["target_date"] = history["date"].shift(-horizon)
    history["actual_future_price"] = history["price"].shift(-horizon)

    prediction_slice = predictions_df.loc[predictions_df["ticker"] == ticker, ["date", "prediction", "target"]].copy()
    prediction_slice["date"] = pd.to_datetime(prediction_slice["date"])
    merged = prediction_slice.merge(
        history.rename(columns={"price": "source_price"}),
        on="date",
        how="left",
    )
    merged = merged.dropna(subset=["source_price", "target_date", "actual_future_price"])
    if merged.empty:
        return history[["date", "price"]].copy(), pd.DataFrame()

    merged["prediction_clipped"] = merged["prediction"].clip(lower=-0.95)
    merged["predicted_future_price"] = merged["source_price"] * (1.0 + merged["prediction_clipped"])
    merged = merged[["target_date", "actual_future_price", "predicted_future_price"]].rename(
        columns={"target_date": "date"}
    )

    history_plot = _resample_history(history[["date", "price"]].copy(), ["price"], horizon=horizon)
    test_plot = _resample_history(
        merged,
        ["actual_future_price", "predicted_future_price"],
        horizon=horizon,
    )
    return history_plot, test_plot


def plot_random_stock_history_forecasts(
    predictions_df: pd.DataFrame,
    price_csv: str | Path,
    horizon: int,
    title: str,
    n_tickers: int = 5,
    seed: int = 7,
):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError(
            "Plotly is required for these stock-history charts. "
            "Install it with `pip install plotly` or `pip install -r requirements-training.txt`."
        ) from exc

    price_panel = load_price_panel(Path(price_csv))
    selected_tickers = choose_random_tickers(predictions_df, n_tickers=n_tickers, seed=seed)
    if not selected_tickers:
        raise ValueError("No ticker predictions were available for stock-history plotting.")

    cols = 2
    rows = math.ceil(len(selected_tickers) / cols)
    subplot_titles = [f"{ticker} ({history_label_for_horizon(horizon)})" for ticker in selected_tickers]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

    show_legend = True
    for idx, ticker in enumerate(selected_tickers):
        row = idx // cols + 1
        col = idx % cols + 1
        history_frame, test_frame = prepare_stock_history_forecast_frame(
            predictions_df=predictions_df,
            price_panel=price_panel,
            ticker=ticker,
            horizon=horizon,
        )
        if history_frame.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=history_frame["date"],
                y=history_frame["price"],
                mode="lines",
                name="Historical price",
                line=dict(color="#4C78A8", width=2),
                opacity=0.50,
                showlegend=show_legend,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=test_frame["date"],
                y=test_frame["actual_future_price"],
                mode="lines",
                name="Actual test price",
                line=dict(color="#2CA02C", width=3),
                showlegend=show_legend,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=test_frame["date"],
                y=test_frame["predicted_future_price"],
                mode="lines",
                name="Predicted price",
                line=dict(color="#F58518", width=3, dash="dash"),
                showlegend=show_legend,
            ),
            row=row,
            col=col,
        )
        show_legend = False

    fig.update_layout(
        template="plotly_white",
        title=title,
        height=max(420, 320 * rows),
        width=1250,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(l=40, r=40, t=90, b=40),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", title_text="Date")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", title_text="Price")
    return fig, selected_tickers


def plot_random_stock_candlestick_forecasts(
    predictions_df: pd.DataFrame,
    price_csv: str | Path,
    horizon: int,
    title: str,
    n_tickers: int = 5,
    seed: int = 7,
):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError(
            "Plotly is required for candlestick forecast charts. "
            "Install it with `pip install plotly` or `pip install -r requirements-training.txt`."
        ) from exc

    price_panel = load_price_panel(Path(price_csv))
    selected_tickers = choose_random_tickers(predictions_df, n_tickers=n_tickers, seed=seed)
    if not selected_tickers:
        raise ValueError("No ticker predictions were available for candlestick plotting.")

    cols = 2
    rows = math.ceil(len(selected_tickers) / cols)
    subplot_titles = [f"{ticker} ({history_label_for_horizon(horizon)})" for ticker in selected_tickers]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

    show_legend = True
    for idx, ticker in enumerate(selected_tickers):
        row = idx // cols + 1
        col = idx % cols + 1

        history = price_panel.loc[price_panel["ticker"] == ticker, ["date", "price"]].copy()
        history = history.sort_values("date").reset_index(drop=True)
        history_ohlc = build_close_derived_ohlc(history, horizon=horizon)
        _, test_frame = prepare_stock_history_forecast_frame(
            predictions_df=predictions_df,
            price_panel=price_panel,
            ticker=ticker,
            horizon=horizon,
        )
        if history_ohlc.empty:
            continue

        history_display = history_ohlc.copy()
        if not test_frame.empty:
            test_start = pd.to_datetime(test_frame["date"]).min()
            test_end = pd.to_datetime(test_frame["date"]).max()
            history_display = history_ohlc.loc[history_ohlc["date"] < test_start].copy()
            fig.add_vrect(
                x0=test_start,
                x1=test_end,
                fillcolor="rgba(31, 119, 180, 0.08)",
                line_width=0,
                row=row,
                col=col,
            )
        else:
            test_start = None
            test_end = None

        fig.add_trace(
            go.Candlestick(
                x=history_display["date"],
                open=history_display["open"],
                high=history_display["high"],
                low=history_display["low"],
                close=history_display["close"],
                name="Historical candles",
                increasing_line_color="#2CA02C",
                decreasing_line_color="#D62728",
                showlegend=show_legend,
                opacity=0.50,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=test_frame["date"],
                y=test_frame["actual_future_price"],
                mode="lines+markers",
                name="Actual test price",
                line=dict(color="#1F77B4", width=3),
                marker=dict(color="#1F77B4", size=5),
                showlegend=show_legend,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=test_frame["date"],
                y=test_frame["predicted_future_price"],
                mode="lines+markers",
                name="Predicted price",
                line=dict(color="#FF7F0E", width=3, dash="dash"),
                marker=dict(color="#FF7F0E", size=5, symbol="diamond"),
                showlegend=show_legend,
            ),
            row=row,
            col=col,
        )
        if test_start is not None and test_end is not None:
            fig.add_annotation(
                x=test_start,
                y=float(test_frame[["actual_future_price", "predicted_future_price"]].max().max()),
                text="Test window",
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                font=dict(size=11, color="#1F3B64"),
                bgcolor="rgba(255,255,255,0.75)",
                row=row,
                col=col,
            )
        show_legend = False

    fig.update_layout(
        template="plotly_white",
        title=title,
        height=max(420, 340 * rows),
        width=1250,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(l=40, r=40, t=90, b=40),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", title_text="Date", rangeslider_visible=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", title_text="Price")
    note = (
        "Candles show pre-test historical price action only. "
        "The shaded region is the test window, where the blue line is the actual test price "
        "and the orange dashed line is the predicted price. "
        "Candles are close-derived because the dataset does not include true OHLC bars."
    )
    return fig, selected_tickers, note
