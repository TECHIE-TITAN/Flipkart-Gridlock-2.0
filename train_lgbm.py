from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score
import pygeohash as pgh
import xgboost as xgb

def minute_of_day(ts: str) -> int:
    h, m = ts.split(":")
    return int(h) * 60 + int(m)

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["minute_of_day"] = df["timestamp"].map(minute_of_day)
    df["hour"] = (df["minute_of_day"] // 60).astype(int)
    df["quarter"] = (df["minute_of_day"] // 15).astype(int)
    df["part_of_day"] = (df["hour"] // 4).astype(int)
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["geohash3"] = df["geohash"].str[:3]
    df["geohash4"] = df["geohash"].str[:4]
    df["geohash5"] = df["geohash"].str[:5]
    return df

def decode_geohash(df: pd.DataFrame) -> pd.DataFrame:
    lat = []
    lon = []
    for gh in df["geohash"].astype(str):
        try:
            la, lo = pgh.decode(gh)
        except Exception:
            la, lo = (np.nan, np.nan)
        lat.append(la)
        lon.append(lo)
    df["gh_lat"] = lat
    df["gh_lon"] = lon
    return df

def prepare_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    df_all = pd.concat([train.drop(columns=["demand"]), test], sort=False).reset_index(drop=True)
    df_all = add_time_features(df_all)
    df_all = decode_geohash(df_all)

    for col in ["Temperature", "NumberofLanes"]:
        if col in df_all.columns:
            df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

    cat_cols = [c for c in ["geohash", "geohash3", "geohash4", "geohash5", "RoadType", "LargeVehicles", "Landmarks", "Weather"] if c in df_all.columns]
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df_all[cat_cols] = enc.fit_transform(df_all[cat_cols].fillna("NA").astype(str))

    n_train = len(train)
    X_all = df_all
    X_train = X_all.iloc[:n_train].copy()
    X_test = X_all.iloc[n_train:].copy()

    features = [
        "minute_of_day",
        "hour",
        "quarter",
        "part_of_day",
        "sin_hour",
        "cos_hour",
        "gh_lat",
        "gh_lon",
    ]
    
    for c in ["NumberofLanes", "Temperature"] + cat_cols:
        if c in X_train.columns and c not in features:
            features.append(c)

    X_train[features] = X_train[features].fillna(-999)
    X_test[features] = X_test[features].fillna(-999)

    return X_train, X_test, features

def time_split_index(df: pd.DataFrame, val_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    df2 = df.copy()
    if "day" in df2.columns and "minute_of_day" in df2.columns:
        df2 = df2.assign(_day=df2["day"].astype(int), _minute=df2["minute_of_day"].astype(int))
        df2 = df2.sort_values(["_day", "_minute"]).reset_index()
    else:
        df2 = df2.reset_index()
    n = len(df2)
    split = int(n * (1.0 - val_frac))
    train_idx = df2.loc[: split - 1, "index"].values
    val_idx = df2.loc[split:, "index"].values
    return train_idx, val_idx

def train_model(X_tr: pd.DataFrame, y_tr: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray,) -> tuple[object, np.ndarray, str, int | None]:
    if xgb is None:
        raise ModuleNotFoundError("xgboost is required. Install it via requirements.txt")

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    params = {
        "objective": "reg:squarederror",
        "eta": 0.03,
        "max_depth": 8,
        "min_child_weight": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "alpha": 0.0,
        "lambda": 1.0,
        "tree_method": "hist",
        "seed": 42,
        "eval_metric": "rmse",
    }
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=4000,
        evals=[(dval, "validation")],
        early_stopping_rounds=150,
        verbose_eval=200,
    )

    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None:
        val_pred = model.predict(dval, iteration_range=(0, best_iteration + 1))
    else:
        val_pred = model.predict(dval)
    return model, val_pred, "xgboost", best_iteration

def train_and_predict(train_path: Path, test_path: Path, output_path: Path) -> None:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    X_train_df, X_test_df, features = prepare_features(train, test)
    y = train["demand"].values

    train_idx, val_idx = time_split_index(X_train_df, val_frac=0.2)
    X_tr = X_train_df.loc[train_idx, features]
    X_val = X_train_df.loc[val_idx, features]
    y_tr = y[train_idx]
    y_val = y[val_idx]

    model, val_pred, backend, best_iteration = train_model(X_tr, y_tr, X_val, y_val)

    if backend == "xgboost" and best_iteration is not None:
        preds = model.predict(xgb.DMatrix(X_test_df[features]), iteration_range=(0, best_iteration + 1))
    else:
        preds = model.predict(xgb.DMatrix(X_test_df[features])) if backend == "xgboost" else model.predict(X_test_df[features])
    preds = np.clip(preds, a_min=0.0, a_max=None)

    out = pd.DataFrame({"Index": test["Index"].values, "demand": preds})
    out.to_csv(output_path, index=False)

    r2 = r2_score(y_val, val_pred)
    print(f"Validation rows: {len(y_val)}")
    print(f"Model backend: {backend}")
    if best_iteration is not None:
        print(f"Best iteration: {best_iteration}")
    print(f"Validation R2: {r2:.6f}")
    print(f"Validation score: {max(0.0, 100.0 * r2):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("dataset/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("dataset/test.csv"))
    parser.add_argument("--output", type=Path, default=Path("submissions/submission_0.csv"))
    args = parser.parse_args()
    train_and_predict(args.train, args.test, args.output)
