#!/usr/bin/env python3
"""
Multistep (k-step) forecasting test that MUST use:
- process_data from stock_prediction.py
- create_model from train_test_function.py

It prints shapes, horizon-wise metrics (vs naive), and a sample forecast with dates
in original price units to prove multistep behavior.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# --- mandatory imports ---
from stock_prediction import process_data
from train_test_function import create_model

# -------- config --------
TICKER = "AAPL"               # change if you like
START = "2020-01-01"
END   = "2024-01-01"
TEST_RATIO = 0.2
N_STEPS = 50
H = 5                         # horizon (k steps)
EPOCHS = 100
BATCH = 64
# ------------------------


# ---- small helpers for proof/logging ----
def inverse_close(arr, close_scaler):
    """Inverse-transform Close (1D or 2D)."""
    arr2d = arr.reshape(-1, 1)
    inv = close_scaler.inverse_transform(arr2d).reshape(arr.shape)
    return inv

def horizon_metrics(y_true, y_pred):
    """MAE/RMSE per horizon + overall."""
    assert y_true.shape == y_pred.shape, f"Shape mismatch: {y_true.shape} vs {y_pred.shape}"
    Hh = y_true.shape[1]
    rows = []
    for h in range(Hh):
        err = (y_pred[:, h] - y_true[:, h]).astype(np.float32)
        rows.append({"horizon": h+1,
                     "MAE": float(np.mean(np.abs(err))),
                     "RMSE": float(np.sqrt(np.mean(err**2)))})
    overall = (y_pred - y_true).astype(np.float32)
    rows.append({"horizon": "overall",
                 "MAE": float(np.mean(np.abs(overall))),
                 "RMSE": float(np.sqrt(np.mean(overall**2)))})
    return pd.DataFrame(rows)

def evaluate_on_test(data, model, horizon):
    """Compute scaled + original unit metrics vs naive baseline."""
    X_test = data["X_test"]; y_test = data["y_test"]
    close_scaler = data["column_scaler"]["Close"]
    Hh = horizon

    y_pred = model.predict(X_test, verbose=0)
    last = X_test[:, -1, 0]
    y_naive = np.tile(last.reshape(-1,1), (1, Hh))

    df_scaled = horizon_metrics(y_test, y_pred); df_scaled["units"] = "scaled"
    df_scaled_naive = horizon_metrics(y_test, y_naive); df_scaled_naive["units"] = "scaled (naive)"

    y_true_inv = inverse_close(y_test, close_scaler)
    y_pred_inv = inverse_close(y_pred, close_scaler)
    y_naive_inv = inverse_close(y_naive, close_scaler)

    df_price = horizon_metrics(y_true_inv, y_pred_inv); df_price["units"] = "price"
    df_price_naive = horizon_metrics(y_true_inv, y_naive_inv); df_price_naive["units"] = "price (naive)"

    return pd.concat([df_scaled, df_scaled_naive, df_price, df_price_naive], ignore_index=True)

def print_sample_forecast(data, model, horizon, sample_idx=0):
    """Print one sample's forecast in scaled and price units with aligned dates."""
    X_test = data["X_test"]; y_test = data["y_test"]
    df_all = data["df"]; test_df = data["test_df"]
    close_scaler = data["column_scaler"]["Close"]
    Hh = horizon

    x = X_test[sample_idx:sample_idx+1]
    y_true = y_test[sample_idx:sample_idx+1]
    y_pred = model.predict(x, verbose=0)
    naive = np.tile(x[:, -1, 0].reshape(1,1), (1, Hh))

    last_input_date = test_df.index[sample_idx]
    all_dates = df_all.index
    pos = np.where(all_dates == last_input_date)[0][0]
    future_dates = all_dates[pos+1:pos+1+Hh]

    def ser(a): return pd.Series(a.reshape(-1), index=future_dates)

    tbl_scaled = pd.DataFrame({
        "y_true_scaled": ser(y_true),
        "y_pred_scaled": ser(y_pred),
        "naive_scaled":  ser(naive)
    })

    y_true_inv = inverse_close(y_true, close_scaler).reshape(-1)
    y_pred_inv = inverse_close(y_pred, close_scaler).reshape(-1)
    naive_inv  = inverse_close(naive, close_scaler).reshape(-1)
    tbl_price = pd.DataFrame({
        "y_true": ser(y_true_inv),
        "y_pred": ser(y_pred_inv),
        "naive":  ser(naive_inv),
    })

    print("Sample", sample_idx, "| Last input date:", last_input_date)
    print("\n--- SCALED ---")
    print(tbl_scaled.round(4))
    print("\n--- ORIGINAL UNITS (inverse-transformed Close) ---")
    print(tbl_price.round(4))

# ------------------------------------------


def main():
    # reproducibility
    np.random.seed(42); tf.random.set_seed(42)

    print("=== Building dataset via process_data (mandatory) ===")
    data = process_data(
        TICKER, START, END,
        test_ratio=TEST_RATIO,
        n_steps=N_STEPS,
        lookup_step=1,      # kept for compatibility; not used for multi-step targets
        horizon=H           # <-- REQUIRED for multistep
    )

    # Validate shapes
    X_train = data["X_train"]; y_train = data["y_train"]
    X_test  = data["X_test"];  y_test  = data["y_test"]
    print("X_train:", X_train.shape, " y_train:", y_train.shape)
    print("X_test :", X_test.shape,  " y_test :",  y_test.shape)
    assert y_train.ndim == 2 and y_train.shape[1] == H, "process_data must return y with shape [N, H] for multistep"
    assert X_train.ndim == 3, "X must be [N, timesteps, features]"

    # ----------------- build model using your create_model (MANDATORY) -----------------
    N_FEATURES = X_train.shape[2]
    print("N_FEATURES:", N_FEATURES)
    # Build your base model
    base = create_model(
        sequence_length=N_STEPS,
        n_features=N_FEATURES,
        # add your own kwargs here if your create_model supports them
        # e.g. units=256, dropout=0.2, loss="mse", optimizer="adam", etc.
    )

    # IMPORTANT: "call" the base model on a Keras Input to define outputs
    inp = tf.keras.Input(shape=(N_STEPS, N_FEATURES))
    base_out = base(inp)  # now the graph is built and we can attach heads

    # If your base outputs a sequence [batch, time, features], reduce it to a vector
    if len(base_out.shape) == 3:
        # You can use either pooling or flatten; flatten is simple and reliable:
        base_out = tf.keras.layers.Flatten()(base_out)

    # Ensure multi-output of size H (k-step)
    # Ensure multi-output of size H (k-step) bounded to [0,1]
    out = tf.keras.layers.Dense(H, activation="sigmoid", name="multistep_head")(base_out)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    print("Model output shape:", model.output_shape)
# -------------------------------------------------------------------------------

    print("=== Training ===")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS, batch_size=BATCH, verbose=1
    )
    trn = history.history.get("loss", [])
    if trn:
        print("Train loss (first 3):", [round(v, 6) for v in trn[:3]])
        print("Train loss (last 3) :", [round(v, 6) for v in trn[-3:]])

    print("\n=== Evaluating vs naive baseline ===")
    metrics = evaluate_on_test(data, model, horizon=H)
    print(metrics.round(4).to_string(index=False))

    overall_price = metrics[(metrics["horizon"] == "overall") & (metrics["units"] == "price")]["RMSE"].values
    overall_naive = metrics[(metrics["horizon"] == "overall") & (metrics["units"] == "price (naive)")]["RMSE"].values
    if overall_price.size and overall_naive.size:
        print(f"\nOverall RMSE (price): model={overall_price[0]:.4f}, naive={overall_naive[0]:.4f}")

    print("\n=== Sample forecast with dates ===")
    print_sample_forecast(data, model, horizon=H, sample_idx=0)


if __name__ == "__main__":
    main()
