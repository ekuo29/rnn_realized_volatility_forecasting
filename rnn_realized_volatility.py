"""Forecast next-day NVDA realized volatility with recurrent neural networks.

This standalone workflow contains only the RNN contribution from a larger
course project. It compares single- and two-layer SimpleRNN architectures,
supports univariate and multivariate feature sets, and keeps the final test
period untouched until model selection is complete.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import GlorotUniform, Orthogonal
from tensorflow.keras.layers import Dense, Dropout, Input, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop, SGD


RANDOM_SEED = 42
DEFAULT_SEQUENCE_LENGTH = 21

FEATURE_GROUPS: dict[str, list[str]] = {
    "univariate_rv": ["log_rv"],
    "multivariate_rv_ma10": ["log_rv", "log_rv_ma10"],
    "multivariate_rv_vix": ["log_rv", "log_vix"],
    "multivariate_rv_ma10_vix": ["log_rv", "log_rv_ma10", "log_vix"],
}


@dataclass(frozen=True)
class RNNConfig:
    """Model and training settings for one experiment."""

    name: str
    features: tuple[str, ...]
    layers: int
    units: int
    dropout: float
    learning_rate: float
    optimizer: str
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH


@dataclass
class RunResult:
    """Metrics, fitted model, and diagnostic data from one experiment."""

    config: RNNConfig
    validation_rmse: float
    validation_mae: float
    history: dict[str, list[float]]
    validation_dates: pd.DatetimeIndex
    validation_actual: np.ndarray
    validation_predicted: np.ndarray
    model: tf.keras.Model
    test_rmse: float | None = None
    test_mae: float | None = None
    test_dates: pd.DatetimeIndex | None = None
    test_actual: np.ndarray | None = None
    test_predicted: np.ndarray | None = None

    def metrics_record(self) -> dict[str, object]:
        record: dict[str, object] = asdict(self.config)
        record.update(
            {
                "features": list(self.config.features),
                "validation_rmse": self.validation_rmse,
                "validation_mae": self.validation_mae,
                "test_rmse": self.test_rmse,
                "test_mae": self.test_mae,
            }
        )
        return record


def set_reproducible_seed(seed: int = RANDOM_SEED) -> None:
    """Reset random state so model comparisons use consistent initialization."""

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def _validate_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def load_vix(path: Path) -> pd.DataFrame:
    """Load a FRED VIX workbook or a two-column CSV file."""

    if path.suffix.lower() in {".xlsx", ".xls"}:
        workbook = pd.ExcelFile(path)
        sheet = "Daily, Close" if "Daily, Close" in workbook.sheet_names else workbook.sheet_names[0]
        raw = pd.read_excel(path, sheet_name=sheet)
    else:
        raw = pd.read_csv(path)

    date_candidates = ["observation_date", "DATE", "date"]
    value_candidates = ["VIXCLS", "vix", "Close"]
    date_col = next((col for col in date_candidates if col in raw.columns), None)
    value_col = next((col for col in value_candidates if col in raw.columns), None)
    if date_col is None or value_col is None:
        raise ValueError(
            "VIX data must contain a date column (observation_date, DATE, or date) "
            "and a value column (VIXCLS, vix, or Close)."
        )

    vix = raw[[date_col, value_col]].rename(columns={date_col: "date", value_col: "vix"})
    vix["date"] = pd.to_datetime(vix["date"], errors="coerce")
    vix["vix"] = pd.to_numeric(vix["vix"], errors="coerce")
    return vix.dropna(subset=["date"]).sort_values("date").set_index("date")


def prepare_model_frame(
    realized_volatility_path: Path,
    vix_path: Path | None,
    symbol: str,
) -> pd.DataFrame:
    """Load the target asset and construct leakage-safe model features."""

    raw = pd.read_pickle(realized_volatility_path)
    _validate_columns(raw, ["sym_root", "date", "realized_volatility"], "RV dataset")

    asset = raw.loc[
        raw["sym_root"].astype(str).str.upper().eq(symbol.upper()),
        ["date", "realized_volatility"],
    ].copy()
    if asset.empty:
        raise ValueError(f"No observations found for symbol {symbol!r}.")

    asset["date"] = pd.to_datetime(asset["date"], errors="coerce")
    asset["rv"] = pd.to_numeric(asset["realized_volatility"], errors="coerce")
    asset = asset.dropna(subset=["date", "rv"]).sort_values("date").set_index("date")
    asset = asset.loc[~asset.index.duplicated(keep="last"), ["rv"]]

    asset["log_rv"] = np.log(asset["rv"].clip(lower=1e-8))
    asset["rv_ma10"] = asset["rv"].rolling(window=10, min_periods=10).mean()
    asset["log_rv_ma10"] = np.log(asset["rv_ma10"].clip(lower=1e-8))

    if vix_path is not None:
        vix = load_vix(vix_path)
        asset = asset.join(vix, how="left")
        asset["vix"] = asset["vix"].ffill()
        asset["log_vix"] = np.log(asset["vix"].clip(lower=1e-8))

    return asset.replace([np.inf, -np.inf], np.nan)


def split_by_target_date(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create chronological train, validation, and untouched test periods."""

    return {
        "train": frame.loc["2018":"2022"].copy(),
        "validation": frame.loc["2023":"2024"].copy(),
        "test": frame.loc["2025":].copy(),
    }


def make_next_day_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    dates: pd.DatetimeIndex,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Use the previous `sequence_length` days to predict the following day."""

    if len(features) <= sequence_length:
        raise ValueError(
            f"Need more than {sequence_length} observations in every data split; "
            f"received {len(features)}."
        )

    X_seq = np.asarray(
        [features[target_pos - sequence_length : target_pos] for target_pos in range(sequence_length, len(features))]
    )
    y_seq = targets[sequence_length:]
    target_dates = pd.DatetimeIndex(dates[sequence_length:])
    return X_seq, y_seq, target_dates


def build_rnn(config: RNNConfig) -> Sequential:
    """Construct and compile a one- or two-layer SimpleRNN model."""

    set_reproducible_seed()
    model = Sequential(name=config.name.replace(" ", "_"))
    model.add(Input(shape=(config.sequence_length, len(config.features))))

    common = {
        "units": config.units,
        "activation": "tanh",
        "kernel_initializer": GlorotUniform(seed=RANDOM_SEED),
        "recurrent_initializer": Orthogonal(seed=RANDOM_SEED),
    }

    if config.layers == 1:
        model.add(SimpleRNN(**common))
        model.add(Dropout(config.dropout, seed=RANDOM_SEED))
    elif config.layers == 2:
        model.add(SimpleRNN(**common, return_sequences=True))
        model.add(Dropout(config.dropout, seed=RANDOM_SEED))
        model.add(SimpleRNN(**common))
        model.add(Dropout(config.dropout, seed=RANDOM_SEED))
    else:
        raise ValueError("RNN layers must be 1 or 2.")

    model.add(Dense(1))
    optimizers = {
        "adam": Adam(learning_rate=config.learning_rate),
        "rmsprop": RMSprop(learning_rate=config.learning_rate),
        "sgd": SGD(learning_rate=config.learning_rate, momentum=0.9),
    }
    if config.optimizer not in optimizers:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")

    model.compile(optimizer=optimizers[config.optimizer], loss=tf.keras.losses.Huber())
    return model


def run_experiment(
    frame: pd.DataFrame,
    config: RNNConfig,
    epochs: int,
    patience: int,
    batch_size: int,
    evaluate_test: bool = False,
    verbose: int = 0,
) -> RunResult:
    """Fit one configuration and evaluate it on validation and optionally test data."""

    required = list(dict.fromkeys([*config.features, "log_rv"]))
    _validate_columns(frame, required, "Model frame")
    clean = frame[required].dropna().copy()
    periods = split_by_target_date(clean)

    scaler_X = StandardScaler().fit(periods["train"][list(config.features)])
    scaler_y = StandardScaler().fit(periods["train"][["log_rv"]])

    prepared: dict[str, tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]] = {}
    for period_name, period in periods.items():
        X = scaler_X.transform(period[list(config.features)])
        y = scaler_y.transform(period[["log_rv"]])
        prepared[period_name] = make_next_day_sequences(
            X,
            y,
            period.index,
            config.sequence_length,
        )

    X_train, y_train, _ = prepared["train"]
    X_validation, y_validation, validation_dates = prepared["validation"]

    model = build_rnn(config)
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=verbose,
        shuffle=False,
    )

    validation_predicted_log = scaler_y.inverse_transform(
        model.predict(X_validation, verbose=0)
    ).reshape(-1)
    validation_actual_log = scaler_y.inverse_transform(y_validation).reshape(-1)
    validation_predicted = np.exp(validation_predicted_log)
    validation_actual = np.exp(validation_actual_log)

    result = RunResult(
        config=config,
        validation_rmse=float(np.sqrt(mean_squared_error(validation_actual, validation_predicted))),
        validation_mae=float(mean_absolute_error(validation_actual, validation_predicted)),
        history={key: [float(value) for value in values] for key, values in history.history.items()},
        validation_dates=validation_dates,
        validation_actual=validation_actual,
        validation_predicted=validation_predicted,
        model=model,
    )

    if evaluate_test:
        X_test, y_test, test_dates = prepared["test"]
        test_predicted_log = scaler_y.inverse_transform(model.predict(X_test, verbose=0)).reshape(-1)
        test_actual_log = scaler_y.inverse_transform(y_test).reshape(-1)
        result.test_predicted = np.exp(test_predicted_log)
        result.test_actual = np.exp(test_actual_log)
        result.test_dates = test_dates
        result.test_rmse = float(np.sqrt(mean_squared_error(result.test_actual, result.test_predicted)))
        result.test_mae = float(mean_absolute_error(result.test_actual, result.test_predicted))

    return result


def plot_loss(result: RunResult, output_path: Path) -> None:
    plt.figure(figsize=(9, 4.5))
    plt.plot(result.history["loss"], label="Training loss")
    plt.plot(result.history["val_loss"], label="Validation loss")
    plt.title("Final RNN: training and validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Huber loss")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_forecast(
    dates: pd.DatetimeIndex,
    actual: np.ndarray,
    predicted: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(11, 5.5))
    plt.plot(dates, actual, label="Actual realized volatility", color="#555555", linewidth=1.1)
    plt.plot(dates, predicted, label="RNN forecast", color="#1464F4", linewidth=1.5)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Realized volatility")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_final_outputs(result: RunResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = result.metrics_record()
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    plot_loss(result, output_dir / "training_history.png")
    plot_forecast(
        result.validation_dates,
        result.validation_actual,
        result.validation_predicted,
        "NVDA validation-period realized volatility forecast",
        output_dir / "validation_forecast.png",
    )
    if result.test_dates is not None and result.test_actual is not None and result.test_predicted is not None:
        plot_forecast(
            result.test_dates,
            result.test_actual,
            result.test_predicted,
            "NVDA out-of-sample realized volatility forecast (2025)",
            output_dir / "test_forecast.png",
        )


def default_final_config() -> RNNConfig:
    """Return the final configuration selected in the original RNN analysis."""

    return RNNConfig(
        name="two_layer_rnn_rv_ma10",
        features=tuple(FEATURE_GROUPS["multivariate_rv_ma10"]),
        layers=2,
        units=32,
        dropout=0.2,
        learning_rate=0.0005,
        optimizer="adam",
    )


def tune_models(
    frame: pd.DataFrame,
    epochs: int,
    patience: int,
    batch_size: int,
    verbose: int,
) -> tuple[RunResult, pd.DataFrame]:
    """Reproduce staged architecture, feature, and hyperparameter selection."""

    all_results: list[RunResult] = []
    tuned_winners: list[RunResult] = []

    for layer_count in (1, 2):
        development: list[RunResult] = []
        for group_name, features in FEATURE_GROUPS.items():
            config = RNNConfig(
                name=f"{layer_count}_layer_{group_name}",
                features=tuple(features),
                layers=layer_count,
                units=32,
                dropout=0.1,
                learning_rate=0.001,
                optimizer="adam",
            )
            result = run_experiment(frame, config, epochs, patience, batch_size, verbose=verbose)
            development.append(result)
            all_results.append(result)

        best_features = min(development, key=lambda item: item.validation_rmse).config.features
        tuning: list[RunResult] = []
        for dropout in (0.1, 0.2):
            for learning_rate in (0.0005, 0.001, 0.002):
                for units in (32, 64):
                    for optimizer in ("adam", "rmsprop", "sgd"):
                        config = RNNConfig(
                            name=f"tuned_{layer_count}_layer_{'_'.join(best_features)}",
                            features=best_features,
                            layers=layer_count,
                            units=units,
                            dropout=dropout,
                            learning_rate=learning_rate,
                            optimizer=optimizer,
                        )
                        result = run_experiment(
                            frame,
                            config,
                            epochs,
                            patience,
                            batch_size,
                            verbose=verbose,
                        )
                        tuning.append(result)
                        all_results.append(result)
        tuned_winners.append(min(tuning, key=lambda item: item.validation_rmse))

    selected = min(tuned_winners, key=lambda item: item.validation_rmse)
    final_result = run_experiment(
        frame,
        selected.config,
        epochs,
        patience,
        batch_size,
        evaluate_test=True,
        verbose=verbose,
    )
    all_results.append(final_result)
    table = pd.DataFrame([result.metrics_record() for result in all_results])
    return final_result, table.sort_values("validation_rmse")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rv-data", type=Path, required=True, help="Path to daily_metrics.pkl")
    parser.add_argument("--vix-data", type=Path, help="Optional path to the FRED VIX workbook or CSV")
    parser.add_argument("--symbol", default="NVDA", help="Target ticker symbol (default: NVDA)")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run the full staged grid search instead of the selected final model only",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.tune and args.vix_data is None:
        raise SystemExit("--vix-data is required for the full feature and hyperparameter search.")

    frame = prepare_model_frame(args.rv_data, args.vix_data, args.symbol)
    if args.tune:
        result, tuning_table = tune_models(
            frame,
            args.epochs,
            args.patience,
            args.batch_size,
            args.verbose,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        tuning_table.to_csv(args.output_dir / "tuning_results.csv", index=False)
    else:
        result = run_experiment(
            frame,
            default_final_config(),
            args.epochs,
            args.patience,
            args.batch_size,
            evaluate_test=True,
            verbose=args.verbose,
        )

    save_final_outputs(result, args.output_dir)
    print(json.dumps(result.metrics_record(), indent=2))


if __name__ == "__main__":
    main()
