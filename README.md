# RNN Realized Volatility Forecasting

We develop Recurrent Neural Network (RNN) models to forecast NVIDIA's next-day realized volatility. The analysis examines whether recurrent architectures can capture volatility persistence and long-memory behavior, and whether moving-average volatility and the CBOE Volatility Index (VIX) improve predictions beyond a univariate specification.

## Project objective

The workflow is designed to:

- forecast next-day realized volatility for NVIDIA (NVDA);
- compare single-layer and multi-layer SimpleRNN architectures;
- compare univariate and multivariate feature specifications;
- tune the number of recurrent units, dropout rate, learning rate, and optimizer; and
- select the final model using validation performance before evaluating the untouched test period.

## Data

The analysis uses daily realized-volatility observations from 2018 through 2025. NVIDIA is selected as the target asset, and its reported realized volatility is used directly as `rv_unit`. The script downloads daily VIX observations with `yfinance` and joins them to the NVIDIA series by date.

The input file, `daily_metrics.pkl`, must contain at least the following columns:

- `sym_root`: asset ticker;
- `date`: observation date; and
- `realized_volatility`: daily realized volatility.

Realized volatility measures the variation observed from intraday price movements, while VIX represents the market's forward-looking implied-volatility expectations. VIX is evaluated as an external feature because broader market uncertainty may help explain changes in NVIDIA's volatility.

## Modeling approach

| Stage | Implementation |
|---|---|
| Data preparation | Filter the daily dataset for NVDA, create a chronological datetime index, download VIX, and join the series by date. |
| Feature engineering | Create log realized volatility, log VIX, and a log 10-day moving average of realized volatility. |
| Target | Forecast next-day log realized volatility using a one-day-ahead shifted target. |
| Sequence construction | Use the previous 21 observations as each RNN input window. |
| Time split | Train on 2018–2022, validate on 2023–2024, and test on 2025. |
| Architectures | Compare one-layer and two-layer `SimpleRNN` models with dropout and a dense output layer. |
| Model development | Evaluate four univariate and multivariate feature groups for each architecture. |
| Hyperparameter tuning | Test 32 or 64 units, dropout of 0.1 or 0.2, three learning rates, and Adam, RMSprop, or SGD. |
| Training | Use Huber loss and early stopping with restoration of the best validation weights. |
| Model selection | Select the lowest validation RMSE before performing a single final test evaluation. |

The multivariate specifications test a volatility moving average, VIX, and their combination. This design separates model development from final evaluation and prevents the 2025 test period from influencing model selection.

## Results

| Model | Validation RMSE |
|---|---:|
| Best tuned single-layer RNN | 0.006432 |
| Best tuned multi-layer RNN | 0.006427 |

- The multi-layer RNN performed best, but improved validation RMSE by only about 0.08% over the single-layer model.
- The selected features were `log_rv` and `log_rv_ma10`; adding VIX did not improve the final specification.
- Test RMSE was **0.006548**, approximately 1.9% higher than validation RMSE.
- RMSE values are reported on the original realized-volatility scale.

## Reproduce the analysis

Create a Python environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Keep `daily_metrics.pkl` in the repository folder, then run:

```bash
python rnn_realized_volatility.py
```

The script runs the complete development and tuning workflow, so execution may take time depending on the available hardware.

## Repository structure

```text
.
├── rnn_realized_volatility.py   # Reproducible RNN modeling workflow
├── daily_metrics.pkl            # Required realized-volatility data
├── requirements.txt             # Python dependencies
└── README.md
```

## Data use note

The dataset may have separate redistribution restrictions. Confirm that public distribution is permitted before keeping `daily_metrics.pkl` in a public repository.

## Reference

Bucci, A. (2020). *Realized Volatility Forecasting with Neural Networks*. Journal of Financial Econometrics, 18(3), 502-531.
