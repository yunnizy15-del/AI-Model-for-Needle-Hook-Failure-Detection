from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FEATURE_NAMES = [
    "len",
    "mu_mean",
    "mu_std",
    "mu_min",
    "mu_max",
    "mu_range",
    "mu_median",
    "mu_q10",
    "mu_q25",
    "mu_q75",
    "mu_q90",
    "mu_iqr",
    "mu_start",
    "mu_end",
    "mu_delta",
    "slope",
    "dmu_mean",
    "dmu_std",
    "dmu_abs_mean",
    "dmu_abs_max",
    "autocorr_lag1",
    "fft_total_power",
    "fft_dom_freq",
    "fft_dom_power_ratio",
    "fft_high_power_ratio",
]


def iter_csv_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*.csv") if p.is_file())


def _safe_float_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return arr


def read_signal(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    if "mu_true" not in df.columns:
        raise ValueError(f"Missing column mu_true in {csv_path}")

    if "t_s" in df.columns:
        t = _safe_float_array(df["t_s"].to_numpy())
    else:
        t = np.arange(len(df), dtype=float)

    mu = _safe_float_array(df["mu_true"].to_numpy())
    n = min(len(t), len(mu))
    if n == 0:
        raise ValueError(f"No valid numeric rows in {csv_path}")
    return t[:n], mu[:n]


def extract_feature_vector(t: np.ndarray, mu: np.ndarray) -> np.ndarray:
    n = len(mu)
    if n == 0:
        raise ValueError("Signal length is zero")

    q10, q25, q75, q90 = np.percentile(mu, [10, 25, 75, 90])
    mu_mean = float(np.mean(mu))
    mu_std = float(np.std(mu))
    mu_min = float(np.min(mu))
    mu_max = float(np.max(mu))
    mu_median = float(np.median(mu))
    mu_start = float(mu[0])
    mu_end = float(mu[-1])
    mu_delta = mu_end - mu_start
    mu_range = mu_max - mu_min
    mu_iqr = float(q75 - q25)

    if n >= 2:
        x = t
        if np.ptp(x) <= 0:
            x = np.arange(n, dtype=float)
        slope = float(np.polyfit(x, mu, 1)[0])
        dmu = np.diff(mu)
        dmu_mean = float(np.mean(dmu))
        dmu_std = float(np.std(dmu))
        dmu_abs_mean = float(np.mean(np.abs(dmu)))
        dmu_abs_max = float(np.max(np.abs(dmu)))

        x0 = mu[:-1] - np.mean(mu[:-1])
        x1 = mu[1:] - np.mean(mu[1:])
        denom = float(np.sqrt(np.sum(x0 * x0) * np.sum(x1 * x1)))
        autocorr_lag1 = float(np.sum(x0 * x1) / denom) if denom > 0 else 0.0
    else:
        slope = 0.0
        dmu_mean = 0.0
        dmu_std = 0.0
        dmu_abs_mean = 0.0
        dmu_abs_max = 0.0
        autocorr_lag1 = 0.0

    if n >= 4:
        dt = np.diff(t)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        median_dt = float(np.median(dt)) if len(dt) > 0 else 1.0

        centered = mu - mu_mean
        fft = np.fft.rfft(centered)
        power = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(n, d=median_dt)

        total_power = float(np.sum(power[1:])) if len(power) > 1 else 0.0
        if len(power) > 1 and total_power > 0:
            idx = int(np.argmax(power[1:])) + 1
            dom_freq = float(freqs[idx])
            dom_power_ratio = float(power[idx] / total_power)

            high_start = max(1, int(0.75 * len(power)))
            high_power_ratio = float(np.sum(power[high_start:]) / total_power)
        else:
            dom_freq = 0.0
            dom_power_ratio = 0.0
            high_power_ratio = 0.0
    else:
        total_power = 0.0
        dom_freq = 0.0
        dom_power_ratio = 0.0
        high_power_ratio = 0.0

    features = np.array(
        [
            float(n),
            mu_mean,
            mu_std,
            mu_min,
            mu_max,
            mu_range,
            mu_median,
            float(q10),
            float(q25),
            float(q75),
            float(q90),
            mu_iqr,
            mu_start,
            mu_end,
            mu_delta,
            slope,
            dmu_mean,
            dmu_std,
            dmu_abs_mean,
            dmu_abs_max,
            autocorr_lag1,
            total_power,
            dom_freq,
            dom_power_ratio,
            high_power_ratio,
        ],
        dtype=float,
    )
    return features


def extract_features_from_csv(csv_path: Path) -> np.ndarray:
    t, mu = read_signal(csv_path)
    return extract_feature_vector(t, mu)


def build_dataset(
    valid_dir: Path, invalid_dir: Path
) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    paths: list[Path] = []

    for p in iter_csv_files(valid_dir):
        xs.append(extract_features_from_csv(p))
        ys.append(0)
        paths.append(p)

    for p in iter_csv_files(invalid_dir):
        xs.append(extract_features_from_csv(p))
        ys.append(1)
        paths.append(p)

    if not xs:
        raise ValueError("No CSV files found in input directories")

    x = np.vstack(xs)
    y = np.asarray(ys, dtype=int)
    return x, y, paths

