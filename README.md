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

## How the RNN works

An RNN reads a time series one observation at a time. At each step, it combines the current input with a hidden state carrying information from earlier observations. This memory allows the model to learn temporal patterns such as volatility persistence and clustering.

The code implements this process as follows:

1. It transforms realized volatility, its 10-day moving average, and VIX into log features and creates a shifted next-day target.
2. It splits the observations chronologically and fits the feature and target scalers using training data only.
3. It converts each dataset into 21-day sequences, so the RNN receives recent volatility history as a three-dimensional input: samples, time steps, and features.
4. Each `SimpleRNN` layer updates its hidden state as it moves through the sequence. In the two-layer model, the first layer returns the full sequence of hidden states to the second layer.
5. Dropout randomly removes part of the hidden representation during training to reduce overfitting, and the final dense layer produces one scaled log-volatility forecast.
6. The model minimizes Huber loss with early stopping. Predictions are transformed back to the original realized-volatility scale before RMSE and MAE are calculated.

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

The tuned multi-layer RNN produced the lowest validation RMSE. Its RMSE was only 0.000005 lower than that of the tuned single-layer model, an improvement of approximately 0.08%. The additional recurrent layer therefore provided only a marginal validation benefit.

The selected specification used `log_rv` and `log_rv_ma10`. This suggests that NVIDIA's recent volatility level and smoothed volatility history contained the strongest predictive information among the tested feature sets. VIX was evaluated but was not part of the winning specification, so it did not provide enough incremental validation improvement in this sample. This does not imply that VIX is uninformative in other assets or market periods.

The selected RNN achieved a test RMSE of **0.006548** on the reserved 2025 observations. This was approximately 1.9% higher than its validation RMSE, indicating a modest deterioration on unseen data rather than a large validation-to-test gap. RMSE was calculated after reversing the scaling and log transformation, so the reported values are on the original realized-volatility scale.

These results compare the RNN specifications tested in this repository. A direct claim that the RNN outperforms traditional forecasting methods would require comparison with a benchmark such as a historical-average, autoregressive, or HAR model.

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
