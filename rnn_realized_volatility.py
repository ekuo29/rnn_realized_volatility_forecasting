"""# **Single Layer & Multi-Layer RNN (Eileen)**"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import yfinance as yf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import GlorotUniform, Orthogonal
from tensorflow.keras.optimizers import Adam, RMSprop, SGD



#1. SETTINGS

RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 8
SEQ_LENGTH = 21

UNITS_CANDIDATES = [32, 64]
DROPOUT_CANDIDATES = [0.1, 0.2]
LEARNING_RATE_CANDIDATES = [0.001, 0.002, 0.0005]
OPTIMIZER_CANDIDATES = ["adam", "rmsprop", "sgd"]

#2. DATA PREPARATION
#Load the original daily realized-volatility dataset and define NVDA.
df_raw = pd.read_pickle("daily_metrics.pkl")
nvda = df_raw[df_raw["sym_root"] == "NVDA"].copy()
nvda["date"] = pd.to_datetime(nvda["date"])
nvda.set_index("date", inplace=True)
nvda.sort_index(inplace=True)
nvda["rv_unit"] = pd.to_numeric(nvda["realized_volatility"], errors="coerce")

#Download VIX data from Yahoo Finance.
vix_data = yf.download("^VIX", start="2017-01-01", end="2026-01-01", auto_adjust=False)


#Flatten it into a simple one-level column format if needed.
if isinstance(vix_data.columns, pd.MultiIndex):
    vix_data.columns = ['_'.join(map(str, col)).strip('_') for col in vix_data.columns]

#Extract the VIX close column and rename it to "vix"
close_col = [c for c in vix_data.columns if "Close" in c][0]
vix_data = vix_data[[close_col]].rename(columns={close_col: "vix"})

#Start from the existing NVDA dataframe.
model_df = nvda.copy()

#Ensure the index is datetime and sorted in chronological order
model_df.index = pd.to_datetime(model_df.index)
model_df = model_df.sort_index()

#Ensure VIX index is also datetime
vix_data.index = pd.to_datetime(vix_data.index)

#If a vix column already exists in nvda, remove it before merging
if "vix" in model_df.columns:
    model_df = model_df.drop(columns=["vix"])

#Merge VIX into the NVDA dataset by date
model_df = model_df.join(vix_data, how="left")

model_df["vix"] = model_df["vix"].ffill().bfill()

#3. FEATURE ENGINEERING

#Ensure rv_unit is numeric
model_df["rv_unit"] = pd.to_numeric(model_df["rv_unit"], errors="coerce")

#Ensure vix is numeric
model_df["vix"] = pd.to_numeric(model_df["vix"], errors="coerce")

#Core transformed features
#Log transform helps reduce skewness and scale effects.
model_df["log_rv"] = np.log(model_df["rv_unit"].clip(lower=1e-8))
model_df["log_vix"] = np.log(model_df["vix"].clip(lower=1e-8))

#Rolling volatility features
#These summarize recent short-term volatility conditions.
model_df["rv_ma10"] = model_df["rv_unit"].rolling(10).mean()

#Log versions of rolling averages
model_df["log_rv_ma10"] = np.log(model_df["rv_ma10"].clip(lower=1e-8))

#Target is next-day realized volatility.
model_df["target"] = model_df["log_rv"].shift(-1)

#4. SPLIT DATA
def split_data(df, feature_cols):
    temp = df[feature_cols + ["target"]].dropna().copy()

    train_df = temp.loc["2018":"2022"].copy()
    valid_df = temp.loc["2023":"2024"].copy()
    test_df = temp.loc["2025":].copy()

    return train_df, valid_df, test_df

#5. CREATE SEQUENCES
def make_sequences(X, y, dates, seq_length):
    X_seq, y_seq, seq_dates = [], [], []

    for i in range(seq_length, len(X)):
        #Previous 21 days become one input window
        X_seq.append(X[i - seq_length:i])

        #The target aligned with that window
        y_seq.append(y[i])

        #Save the corresponding date
        seq_dates.append(dates[i])

    return np.array(X_seq), np.array(y_seq), pd.Index(seq_dates)

#6. BUILD MODEL
def build_rnn(
    n_features,
    seq_length,
    layers=1,
    units=32,
    dropout_rate=0.1,
    learning_rate=0.001,
    optimizer_name="adam"):

    #Reset state so every configuration is reproducible and independent.
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    model = Sequential()
    model.add(Input(shape=(seq_length, n_features))) # Explicitly define Input layer

    if layers == 1:
        #Single-layer RNN: takes a sequence and outputs one hidden representation
        model.add(SimpleRNN(
            units=units,
            activation="tanh",
            kernel_initializer=GlorotUniform(seed=RANDOM_SEED),
            recurrent_initializer=Orthogonal(seed=RANDOM_SEED)))
        model.add(Dropout(dropout_rate))

    elif layers == 2:
        model.add(SimpleRNN(
            units=units,
            activation="tanh",
            return_sequences=True,
            kernel_initializer=GlorotUniform(seed=RANDOM_SEED),
            recurrent_initializer=Orthogonal(seed=RANDOM_SEED)))
        model.add(Dropout(dropout_rate))

        model.add(SimpleRNN(
            units=units,
            activation="tanh",
            kernel_initializer=GlorotUniform(seed=RANDOM_SEED),
            recurrent_initializer=Orthogonal(seed=RANDOM_SEED)))
        model.add(Dropout(dropout_rate))
    else:
        raise ValueError("layers must be 1 or 2")

    #Final dense layer outputs one forecasted value
    model.add(Dense(1))

    #try out different optimizer
    if optimizer_name == "adam":
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == "rmsprop":
        optimizer = RMSprop(learning_rate=learning_rate)
    elif optimizer_name == "sgd":
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        raise ValueError("optimizer_name must be 'adam', 'rmsprop', or 'sgd'")

    #Huber loss:less sensitive to large outliers than MSE
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber())

    return model


#7. RUN MODEL
def run_rnn(
    model_name,
    feature_cols,
    layers,
    seq_length,
    units=32,
    dropout_rate=0.1,
    learning_rate=0.001,
    optimizer_name="adam",
    use_test=False):
    train_df, valid_df, test_df = split_data(model_df, feature_cols)

    #Standardize features and target
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    #Fit scalers only on training data to avoid leakage
    X_train = scaler_X.fit_transform(train_df[feature_cols])
    X_valid = scaler_X.transform(valid_df[feature_cols])

    y_train = scaler_y.fit_transform(train_df[["target"]])
    y_valid = scaler_y.transform(valid_df[["target"]])

    #Create rolling sequences
    X_train_seq, y_train_seq, _ = make_sequences(X_train, y_train, train_df.index, seq_length)
    X_valid_seq, y_valid_seq, valid_dates = make_sequences(X_valid, y_valid, valid_df.index, seq_length)

    #Build the RNN
    model = build_rnn(
        n_features=len(feature_cols),
        seq_length=seq_length,
        layers=layers,
        units=units,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,optimizer_name=optimizer_name)

    #Early stopping prevents overfitting
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True)

    #Fit the model
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_valid_seq, y_valid_seq),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=0)

    #Validation predictions in log scale
    valid_pred_log = scaler_y.inverse_transform(model.predict(X_valid_seq, verbose=0))
    y_valid_actual_log = scaler_y.inverse_transform(y_valid_seq)

    #Convert back to RV level
    valid_pred = np.exp(valid_pred_log)
    y_valid_actual = np.exp(y_valid_actual_log)

    #Store validation results
    result = {
        "Model": model_name,
        "Features": feature_cols,
        "Layers": layers,
        "Seq_Length": seq_length,
        "Units": units,
        "Dropout": dropout_rate,
        "Learning_Rate": learning_rate,
        "Optimizer": optimizer_name,
        "Validation_RMSE": np.sqrt(mean_squared_error(y_valid_actual, valid_pred)),
        "Validation_MAE": mean_absolute_error(y_valid_actual, valid_pred),
        "history": history,
        "valid_actual": y_valid_actual.flatten(),
        "valid_pred": valid_pred.flatten(),
        "valid_dates": valid_dates}

    # Test evaluation (if requested)
    if use_test:
        X_test = scaler_X.transform(test_df[feature_cols])
        y_test = scaler_y.transform(test_df[["target"]])
        X_test_seq, y_test_seq, test_dates = make_sequences(
            X_test, y_test, test_df.index, seq_length)

        test_pred_log = scaler_y.inverse_transform(model.predict(X_test_seq, verbose=0))
        y_test_actual_log = scaler_y.inverse_transform(y_test_seq)
        test_pred = np.exp(test_pred_log)
        y_test_actual = np.exp(y_test_actual_log)
        result["Test_RMSE"] = np.sqrt(mean_squared_error(y_test_actual, test_pred))
        result["Test_MAE"] = mean_absolute_error(y_test_actual, test_pred)
        result["test_actual"] = y_test_actual.flatten()
        result["test_pred"] = test_pred.flatten()
        result["test_dates"] = test_dates

    return result

#8. PRINT / PLOT HELPERS
def plot_validation_result(result, figure_title, forecast_label):
    plt.figure(figsize=(10, 5))
    plt.plot(result["history"].history["loss"], label="Training Loss")
    plt.plot(result["history"].history["val_loss"], label="Validation Loss")
    plt.title(f"{figure_title} Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(result["valid_dates"], result["valid_actual"], label="Actual RV", color="gray")
    plt.plot(result["valid_dates"], result["valid_pred"], label=forecast_label, color="blue")
    plt.title(f"{figure_title} Validation Performance")
    plt.xlabel("Date")
    plt.ylabel("Realized Volatility")
    plt.legend()
    plt.grid(True)
    plt.show()


def tune_rnn(best_stage, layers):
    tuning_rows = []
    best_result = None

    for dropout_rate in DROPOUT_CANDIDATES:
        for learning_rate in LEARNING_RATE_CANDIDATES:
            for units in UNITS_CANDIDATES:
                for optimizer_name in OPTIMIZER_CANDIDATES:
                    result = run_rnn(
                        model_name=f"Tuned {best_stage['Model']}",
                        feature_cols=best_stage["Features"],
                        layers=layers,
                        seq_length=SEQ_LENGTH,
                        units=units,
                        dropout_rate=dropout_rate,
                        learning_rate=learning_rate,
                        optimizer_name=optimizer_name,
                        use_test=False)
                    tuning_rows.append({
                        "Model": result["Model"],
                        "Optimizer": result["Optimizer"],
                        "Dropout": result["Dropout"],
                        "Learning_Rate": result["Learning_Rate"],
                        "Units": result["Units"],
                        "Validation_RMSE": result["Validation_RMSE"]})

                    if (best_result is None or
                            result["Validation_RMSE"] < best_result["Validation_RMSE"]):
                        best_result = result

    tuning_df = pd.DataFrame(tuning_rows).sort_values("Validation_RMSE")
    return tuning_df, best_result

#9. FEATURE SETS

feature_groups = {
    "Univariate_RV": ["log_rv"],
    "Multivariate_1_RV_MA10": ["log_rv", "log_rv_ma10"],
    "Multivariate_2_RV_VIX": ["log_rv", "log_vix"],
    "Multivariate_3_RV_MA10_VIX": ["log_rv", "log_rv_ma10", "log_vix"]}

#test each category separately for multivariate

# 10. STAGE B: SINGLE-LAYER DEVELOPMENT
single_stage_results = []

for group_name, feature_cols in feature_groups.items():
    res = run_rnn(
        model_name=f"Single-Layer {group_name}",
        feature_cols=feature_cols,
        layers=1,
        seq_length=SEQ_LENGTH,
        units=32,
        dropout_rate=0.1,
        learning_rate=0.001,
        optimizer_name="adam",
        use_test=False)
    single_stage_results.append(res)

    plot_validation_result(
        res,
        figure_title=f"Single-Layer {group_name}",
        forecast_label=f"Validation Forecast ({' + '.join(feature_cols)})")

print("\n================ SINGLE-LAYER DEVELOPMENT ================")
for r in single_stage_results:
    print(f"{r['Model']}")
    print(f"  Features: {', '.join(r['Features'])}")
    print(f"  Optimizer: {r['Optimizer']}")
    print(f"  Validation RMSE: {r['Validation_RMSE']:.6f}")
    print(f"  Validation MAE : {r['Validation_MAE']:.6f}")
    print("--------------------------------------------------")


#Select better single-layer structure based on validation RMSE
best_single_stage = min(single_stage_results, key=lambda x: x["Validation_RMSE"])
print(f"\n>>> Best SINGLE-LAYER structure: {best_single_stage['Model']}")


#11. STAGE C: SINGLE-LAYER HYPERPARAMETER TUNING
single_tuning_df, best_single_final = tune_rnn(
    best_single_stage,
    layers=1)

print("\n================ SINGLE-LAYER TUNING ======================")
print(single_tuning_df.head(10))
print(f"\n>>> Best tuned SINGLE-LAYER model: {best_single_final['Model']}")
print(f"    Optimizer: {best_single_final['Optimizer']}")
print(f"    Validation RMSE: {best_single_final['Validation_RMSE']:.6f}")


#12. STAGE D: MULTI-LAYER DEVELOPMENT
multi_stage_results = []

for group_name, feature_cols in feature_groups.items():
    res = run_rnn(
        model_name=f"Multi-Layer {group_name}",
        feature_cols=feature_cols,
        layers=2,
        seq_length=SEQ_LENGTH,
        units=32,
        dropout_rate=0.1,
        learning_rate=0.001,
        optimizer_name="adam",
        use_test=False)
    multi_stage_results.append(res)

    plot_validation_result(
        res,
        figure_title=f"Multi-Layer {group_name}",
        forecast_label=f"Validation Forecast ({' + '.join(feature_cols)})")

print("\n================ MULTI-LAYER DEVELOPMENT ==================")
for r in multi_stage_results:
    print(f"{r['Model']}")
    print(f"  Features: {r['Features']}")
    print(f"  Optimizer: {r['Optimizer']}")
    print(f"  Validation RMSE: {r['Validation_RMSE']:.6f}")
    print(f"  Validation MAE : {r['Validation_MAE']:.6f}")
    print("--------------------------------------------------")

#Select better multi-layer structure based on validation RMSE
best_multi_stage = min(multi_stage_results, key=lambda x: x["Validation_RMSE"])
print(f"\n>>> Better MULTI-LAYER model: {best_multi_stage['Model']}")


#13. STAGE E: MULTI-LAYER HYPERPARAMETER TUNING
multi_tuning_df, best_multi_final = tune_rnn(
    best_multi_stage,
    layers=2)

print("\n================ MULTI-LAYER TUNING ======================")
print(multi_tuning_df.head(10))
print(f"\n>>> Best tuned MULTI-LAYER model: {best_multi_final['Model']}")
print(f"    Optimizer: {best_multi_final['Optimizer']}")
print(f"    Validation RMSE: {best_multi_final['Validation_RMSE']:.6f}")


#14. FINAL VALIDATION SELECTION
#Compare best tuned single-layer vs best tuned multi-layer
final_candidates = [best_single_final, best_multi_final]

final_candidates_df = pd.DataFrame([{
    "Model": r["Model"],
    "Features": r["Features"],
    "Layers": r["Layers"],
    "Seq_Length": r["Seq_Length"],
    "Units": r["Units"],
    "Dropout": r["Dropout"],
    "Learning_Rate": r["Learning_Rate"],
    "Optimizer": r["Optimizer"],
    "Validation_RMSE": r["Validation_RMSE"],
    "Validation_MAE": r["Validation_MAE"]
} for r in final_candidates]).sort_values("Validation_RMSE")

print("\n================ FINAL MODEL COMPARISON ===================")
for r in final_candidates:
    print(f"{r['Model']}")
    print(f"  Layers: {r['Layers']}")
    print(f"  Features: {r['Features']}")
    print(f"  Optimizer: {r['Optimizer']}")
    print(f"  Validation RMSE: {r['Validation_RMSE']:.6f}")
    print("--------------------------------------------------")

#Final model is chosen only from validation performance
final_model = min(final_candidates, key=lambda x: x["Validation_RMSE"])

print("\nFINAL CANDIDATES (VALIDATION ONLY)")
print(final_candidates_df)

print(f"\n>>> FINAL SELECTED MODEL (based on validation): {final_model['Model']}")
print(f"    Optimizer: {final_model['Optimizer']}")
print(f"    Validation RMSE: {final_model['Validation_RMSE']:.6f}")
print(f"    Validation MAE : {final_model['Validation_MAE']:.6f}")


#15. FINAL TEST EVALUATION
#Now evaluate the selected model on the untouched test period
final_result = run_rnn(
    model_name=final_model["Model"],
    feature_cols=final_model["Features"],
    layers=final_model["Layers"],
    seq_length=final_model["Seq_Length"],
    units=final_model["Units"],
    dropout_rate=final_model["Dropout"],
    learning_rate=final_model["Learning_Rate"],
    optimizer_name=final_model["Optimizer"],
    use_test=True)

print("\n================ FINAL TEST PERFORMANCE ===================")
print(f"Test RMSE: {final_result['Test_RMSE']:.6f}")
print(f"Test MAE : {final_result['Test_MAE']:.6f}")

#16. FINAL TEST PLOT
plt.figure(figsize=(12, 6))
plt.plot(final_result["test_dates"], final_result["test_actual"], label="Actual RV", color="gray")
plt.plot(final_result["test_dates"], final_result["test_pred"], label="Test Forecast", color="blue")
plt.title(f"Final Test Performance: {final_result['Model']}")
plt.xlabel("Date")
plt.ylabel("Realized Volatility")
plt.legend()
plt.grid(True)
plt.show()
