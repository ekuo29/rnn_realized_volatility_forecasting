# RNN Realized Volatility Forecasting

This repository contains my Recurrent Neural Network (RNN) contribution to a FINANCE 704 group project on next-day realized-volatility forecasting. It keeps the original modeling choices and excludes the other group members' models.

## Project objective

The analysis tests whether recurrent neural networks can capture the persistence and clustering of NVIDIA's realized volatility. It compares:

- single-layer and multi-layer SimpleRNN architectures;
- univariate and multivariate feature specifications;
- 32 and 64 recurrent units;
- dropout rates of 0.1 and 0.2;
- learning rates of 0.0005, 0.001, and 0.002; and
- Adam, RMSprop, and SGD optimizers.

## Modeling workflow

| Stage | Implementation |
|---|---|
| Target | Shifted next-day log realized volatility |
| Input window | 21 observations |
| Features | Log realized volatility, log VIX, and a log 10-day volatility moving average |
| Time split | 2018-2022 training, 2023-2024 validation, and 2025 test |
| Scaling | `StandardScaler` fitted on the training period |
| Architectures | One-layer and two-layer `SimpleRNN` models with dropout and a dense output layer |
| Loss | Huber loss |
| Selection | Lowest validation RMSE before final test evaluation |

## Original results

The submitted analysis selected a tuned two-layer multivariate RNN using `log_rv` and `log_rv_ma10`.

| Model | Validation RMSE |
|---|---:|
| Best tuned single-layer RNN | 0.006432 |
| Best tuned multi-layer RNN | 0.006427 |

The final RNN recorded a test RMSE of **0.006548** in the submitted group report.

## Code

[`rnn_realized_volatility.py`](rnn_realized_volatility.py) contains the original RNN analysis with light cleanup of redundant code. The feature comparisons, model architectures, hyperparameter grid, validation-based selection, and final test workflow are unchanged.

The script now defines the `nvda` DataFrame directly from `daily_metrics.pkl`. Keep that file in the same folder when running the analysis. The dataset must contain:

- a `sym_root` column containing the ticker symbol;
- a `date` column; and
- a `realized_volatility` column.

It downloads VIX observations with `yfinance`, constructs the RNN features, trains the candidate models, selects the final specification using validation RMSE, and evaluates the selected model on the test period.

Run the analysis with:

```bash
python rnn_realized_volatility.py
```

## Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Repository structure

```text
.
├── rnn_realized_volatility.py   # Cleaned original RNN analysis
├── requirements.txt             # Python dependencies
└── README.md
```

## Publication note

The assignment instructions, full group report, and classmates' code are intentionally excluded. This repository is intended to focus on my RNN contribution. The course dataset may have separate redistribution restrictions, so permission should be confirmed before keeping `daily_metrics.pkl` in a public repository.

No open-source license has been applied. The repository does not grant redistribution rights for the underlying course data.

## Reference

Bucci, A. (2020). *Realized Volatility Forecasting with Neural Networks*. Journal of Financial Econometrics, 18(3), 502-531.
