
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def inverse_close(arr, close_scaler):
    """Inverse-transform a 1D or 2D array of scaled Close values using the MinMaxScaler for Close."""
    arr2d = arr.reshape(-1, 1)
    inv = close_scaler.inverse_transform(arr2d).reshape(arr.shape)
    return inv

def horizon_metrics(y_true, y_pred):
    """
    Compute MAE/RMSE per horizon step and overall.
    y_true, y_pred: arrays of shape [N, H]
    Returns a pandas DataFrame.
    """
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have same shape"
    H = y_true.shape[1]
    rows = []
    for h in range(H):
        err = y_pred[:, h] - y_true[:, h]
        mae = np.mean(np.abs(err))
        rmse = np.sqrt(np.mean(err**2))
        rows.append({"horizon": h+1, "MAE": mae, "RMSE": rmse})
    # overall (macro-average across horizons)
    overall_err = y_pred - y_true
    rows.append({
        "horizon": "overall",
        "MAE": np.mean(np.abs(overall_err)),
        "RMSE": np.sqrt(np.mean(overall_err**2))
    })
    return pd.DataFrame(rows)

def naive_baseline_from_last(X, close_idx):
    """
    Naive baseline for multi-step: repeat the last observed close in input window across all H steps.
    X: [N, W, F]
    close_idx: index of 'Close' within feature_columns (0 if it's the first)
    Returns an array of shape [N, H] AFTER you pass H (you need to know H to tile).
    """
    last_close = X[:, -1, close_idx]  # [N]
    return last_close

def print_sample_forecast(data, model, feature_columns, horizon, sample_idx=0, unscale=True):
    """
    Log a single test sample's forecast vs. ground truth and a naive baseline.
    - Shows dates: the last input date and the next H dates.
    - Prints true, pred, baseline in both scaled and (optionally) original price units.
    """
    X_test = data["X_test"]                               # [N, W, F]
    y_test = data["y_test"]                               # [N, H]
    test_df = data["test_df"]                             # rows indexed by last input date per sample
    df_all = data["df"]                                   # full original dataframe (pre-scaled copy)
    close_idx = feature_columns.index("Close")
    H = horizon

    # Grab the sample
    x = X_test[sample_idx:sample_idx+1]                   # [1, W, F]
    y_true = y_test[sample_idx:sample_idx+1]              # [1, H]
    last_input_date = test_df.index[sample_idx]

    # Predict
    y_pred = model.predict(x, verbose=0)                  # [1, H]

    # Build naive baseline (scaled domain), then reshape to [1, H]
    baseline_last = naive_baseline_from_last(x, close_idx) # [1]
    y_naive = np.tile(baseline_last.reshape(1,1), (1,H))   # repeat across horizon

    # Map dates: find the next H trading dates after the last input date
    all_dates = df_all.index
    # find position of last_input_date in all_dates
    pos = np.where(all_dates == last_input_date)[0]
    if len(pos) == 0:
        raise ValueError("Last input date not found in df_all index; check alignment.")
    pos = pos[0]
    future_dates = all_dates[pos+1:pos+1+H]

    def to_series(arr):
        return pd.Series(arr.reshape(-1), index=future_dates)

    # Prepare scaled view
    tbl_scaled = pd.DataFrame({
        "y_true_scaled": to_series(y_true),
        "y_pred_scaled": to_series(y_pred),
        "naive_scaled":  to_series(y_naive)
    })

    if unscale:
        close_scaler = data["column_scaler"]["Close"]
        y_true_inv = inverse_close(y_true.reshape(-1), close_scaler)
        y_pred_inv = inverse_close(y_pred.reshape(-1), close_scaler)
        y_naive_inv = inverse_close(y_naive.reshape(-1), close_scaler)
        tbl_unscaled = pd.DataFrame({
            "y_true": pd.Series(y_true_inv, index=future_dates),
            "y_pred": pd.Series(y_pred_inv, index=future_dates),
            "naive":  pd.Series(y_naive_inv, index=future_dates)
        })
    else:
        tbl_unscaled = None

    print("Sample", sample_idx, "| Last input date:", last_input_date)
    print("\n--- SCALED ---")
    print(tbl_scaled.round(4))
    if tbl_unscaled is not None:
        print("\n--- ORIGINAL UNITS (inverse-transformed Close) ---")
        print(tbl_unscaled.round(4))

    return tbl_scaled, tbl_unscaled

def plot_sample_forecast(data, model, feature_columns, horizon, sample_idx=0):
    """
    Simple line plot of true vs predicted vs naive for one sample (inverse-transformed).
    """
    _, tbl_unscaled = print_sample_forecast(
        data, model, feature_columns, horizon, sample_idx=sample_idx, unscale=True
    )
    if tbl_unscaled is None:
        raise ValueError("Need inverse scaler for plotting in original units.")
    plt.figure()
    tbl_unscaled[["y_true", "y_pred", "naive"]].plot(marker="o")
    plt.title(f"Multi-step forecast (sample {sample_idx})")
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.grid(True)
    plt.show()

def evaluate_on_test(data, model, feature_columns, horizon):
    """
    Compute horizon-wise MAE/RMSE on the whole test set.
    Returns a DataFrame with metrics in both scaled and original units.
    """
    X_test = data["X_test"]
    y_test = data["y_test"]
    H = horizon
    close_idx = feature_columns.index("Close")

    # Predictions
    y_pred = model.predict(X_test, verbose=0)   # [N, H]

    # Scaled metrics
    df_scaled = horizon_metrics(y_test, y_pred)
    df_scaled["units"] = "scaled"

    # Naive baseline (scaled)
    last = naive_baseline_from_last(X_test, close_idx)  # [N]
    y_naive = np.tile(last.reshape(-1, 1), (1, H))
    df_scaled_naive = horizon_metrics(y_test, y_naive)
    df_scaled_naive["units"] = "scaled (naive)"

    # Unscaled metrics
    close_scaler = data["column_scaler"]["Close"]
    y_true_inv = inverse_close(y_test, close_scaler)
    y_pred_inv = inverse_close(y_pred, close_scaler)
    y_naive_inv = inverse_close(y_naive, close_scaler)

    df_unscaled = horizon_metrics(y_true_inv, y_pred_inv)
    df_unscaled["units"] = "price"
    df_unscaled_naive = horizon_metrics(y_true_inv, y_naive_inv)
    df_unscaled_naive["units"] = "price (naive)"

    # Combine
    out = pd.concat([df_scaled, df_scaled_naive, df_unscaled, df_unscaled_naive], axis=0, ignore_index=True)
    return out
