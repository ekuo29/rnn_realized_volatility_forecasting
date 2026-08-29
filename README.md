# RNN Realized Volatility Forecasting

This repository contains my original Recurrent Neural Network (RNN) contribution to a FINANCE 704 group project on next-day realized-volatility forecasting. The code is preserved as submitted and excludes the other group members' models.

## Project objective

The analysis tests whether recurrent neural networks can capture the persistence and clustering of NVIDIA's realized volatility. It compares:

- single-layer and multi-layer SimpleRNN architectures;
- univariate and multivariate feature specifications;
- 32 and 64 recurrent units;
- dropout rates of 0.1 and 0.2;
- learning rates of 0.0005, 0.001, and 0.002; and
- Adam, RMSprop, and SGD optimizers.

## Original modeling workflow

| Stage | Original implementation |
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

[`rnn_realized_volatility.py`](rnn_realized_volatility.py) is the exact RNN section from the submitted group Python export. Its model logic, sequence construction, tuning loops, variable names, comments, and plotting code have not been refactored or rewritten.

The original section begins after the group's shared preprocessing and therefore expects an existing `nvda` pandas DataFrame with:

- a datetime index; and
- an `rv_unit` column containing NVIDIA realized volatility.

It downloads VIX observations with `yfinance`, constructs the RNN features, trains the candidate models, selects the final specification using validation RMSE, and evaluates the selected model on the test period.

## Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Repository structure

```text
.
├── rnn_realized_volatility.py   # Original submitted RNN code
├── requirements.txt             # Python dependencies
└── README.md
```

## Data and publication note

The course dataset, assignment instructions, full group report, and classmates' code are intentionally excluded because they are course or third-party materials. This repository contains only my original RNN contribution and a project description.

No open-source license has been applied. The repository does not grant redistribution rights for the underlying course data.

## Reference

Bucci, A. (2020). *Realized Volatility Forecasting with Neural Networks*. Journal of Financial Econometrics, 18(3), 502-531.
