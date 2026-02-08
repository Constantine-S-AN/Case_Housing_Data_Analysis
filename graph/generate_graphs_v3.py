#!/usr/bin/env python3
"""Generate presentation-focused plots 07-12 for housing analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
OUT_DIR = Path(__file__).resolve().parent
ROOT_DIR = OUT_DIR.parent


def info(msg: str) -> None:
    print(f"[INFO] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def find_cleaned_train_csv() -> Path | None:
    candidates = [
        Path.cwd() / "train_cleaned_v2.csv",
        Path.cwd() / "data" / "train_cleaned_v2.csv",
        ROOT_DIR / "train_cleaned_v2.csv",
        ROOT_DIR / "data" / "train_cleaned_v2.csv",
    ]
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.exists():
            return path

    # Fallback search if expected paths are missing.
    for path in Path.cwd().rglob("train_cleaned_v2.csv"):
        return path
    for path in ROOT_DIR.rglob("train_cleaned_v2.csv"):
        return path
    return None


def as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def make_one_hot_encoder() -> OneHotEncoder:
    # sklearn compatibility across versions.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def save_fig(fig: plt.Figure, filename: str) -> None:
    out_path = OUT_DIR / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    info(f"Saved {out_path}")


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "SalePrice" in df.columns:
        saleprice = as_numeric(df["SalePrice"]).fillna(0)
        df["logSalePrice"] = np.log1p(saleprice)
    else:
        warn("SalePrice column is missing; dependent plots will be skipped.")

    gr_liv = as_numeric(df["GrLivArea"]) if "GrLivArea" in df.columns else pd.Series(0.0, index=df.index)
    bsmt = as_numeric(df["TotalBsmtSF"]) if "TotalBsmtSF" in df.columns else pd.Series(0.0, index=df.index)
    df["TotalSF"] = gr_liv.fillna(0) + bsmt.fillna(0)

    if "YrSold" in df.columns:
        yrsold = as_numeric(df["YrSold"])
        fallback_year = int(np.nanmax(yrsold.values)) if np.isfinite(np.nanmax(yrsold.values)) else 2010
        yrsold = yrsold.fillna(fallback_year)
    else:
        fallback_year = 2010
        yrsold = pd.Series(fallback_year, index=df.index, dtype=float)

    if "YearBuilt" in df.columns:
        year_built = as_numeric(df["YearBuilt"])
        df["HouseAge"] = (yrsold - year_built).clip(lower=0)
    else:
        warn("YearBuilt missing; HouseAge cannot be engineered.")

    if "YearRemodAdd" in df.columns:
        remod = as_numeric(df["YearRemodAdd"])
        df["RemodAge"] = (yrsold - remod).clip(lower=0)
    else:
        warn("YearRemodAdd missing; RemodAge cannot be engineered.")

    return df


def choose_id_values(df: pd.DataFrame, idx: Iterable[int]) -> list[str]:
    if "Id" in df.columns:
        ids = as_numeric(df.loc[list(idx), "Id"]).fillna(-1).astype(int).astype(str)
        return ids.tolist()
    return [str(i) for i in idx]


def plot_07_outlier_candidates(df: pd.DataFrame) -> pd.Index:
    required = ["GrLivArea", "logSalePrice"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        warn(f"Skipping 07_outlier_candidates_labeled.png; missing columns: {missing}")
        return pd.Index([])

    data = df[required].copy()
    data["GrLivArea"] = as_numeric(data["GrLivArea"])
    data["logSalePrice"] = as_numeric(data["logSalePrice"])
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if data.shape[0] < 20:
        warn("Skipping 07_outlier_candidates_labeled.png; not enough valid rows.")
        return pd.Index([])

    x = data["GrLivArea"]
    y = data["logSalePrice"]
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    resid = y - pred
    score = ((x - x.mean()) / (x.std(ddof=0) + 1e-12)) - ((resid - resid.mean()) / (resid.std(ddof=0) + 1e-12))

    top_n = min(10, len(score))
    top_idx = score.nlargest(top_n).index

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, alpha=0.35, color="gray", s=20, label="All homes")
    ax.scatter(x.loc[top_idx], y.loc[top_idx], color="crimson", s=45, label="Suspicious candidates")
    ax.plot(np.sort(x.values), slope * np.sort(x.values) + intercept, color="black", linewidth=1, alpha=0.7)

    labels = choose_id_values(df, top_idx)
    for idx, label in zip(top_idx, labels):
        ax.annotate(label, (x.loc[idx], y.loc[idx]), xytext=(5, 5), textcoords="offset points", fontsize=8)

    ax.set_title("Outlier Candidates: High GrLivArea with Relatively Low Price")
    ax.set_xlabel("GrLivArea")
    ax.set_ylabel("log1p(SalePrice)")
    ax.legend(frameon=False)
    save_fig(fig, "07_outlier_candidates_labeled.png")

    return top_idx


def compute_ols_diagnostics(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "logSalePrice",
) -> dict[str, np.ndarray | pd.Index | float] | None:
    existing = [c for c in feature_cols if c in df.columns]
    if not existing:
        warn("Cook's distance setup failed; no baseline features are available.")
        return None
    if target_col not in df.columns:
        warn(f"Cook's distance setup failed; missing target {target_col}.")
        return None

    work = df[existing + [target_col]].copy()
    work[target_col] = as_numeric(work[target_col])
    work = work.dropna(subset=[target_col])
    if work.shape[0] < 25:
        warn("Cook's distance setup failed; not enough valid rows.")
        return None

    X = work[existing].copy()
    for col in X.columns:
        if X[col].dtype == object or str(X[col].dtype).startswith("category"):
            X[col] = X[col].fillna("Missing").astype(str)
        else:
            X[col] = as_numeric(X[col])
            fill = X[col].median()
            if pd.isna(fill):
                fill = 0.0
            X[col] = X[col].fillna(fill)

    X_enc = pd.get_dummies(X, drop_first=True)
    if X_enc.shape[1] == 0:
        warn("Cook's distance setup failed; encoded feature matrix is empty.")
        return None

    y = work[target_col].to_numpy(dtype=float)
    X_mat = X_enc.to_numpy(dtype=float)
    n = X_mat.shape[0]
    X_design = np.column_stack([np.ones(n), X_mat])

    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    y_hat = X_design @ beta
    resid = y - y_hat

    p = X_design.shape[1]
    dof = max(n - p, 1)
    mse = np.sum(resid**2) / dof
    xtx_inv = np.linalg.pinv(X_design.T @ X_design)
    leverage = np.einsum("ij,jk,ik->i", X_design, xtx_inv, X_design)

    eps = 1e-12
    cooks = (resid**2 / max(p * mse, eps)) * (leverage / np.clip((1.0 - leverage) ** 2, eps, None))
    threshold = 4.0 / n

    return {
        "index": work.index,
        "cooks": cooks,
        "threshold": threshold,
    }


def plot_08_cooks_distance(df: pd.DataFrame, diagnostics: dict[str, np.ndarray | pd.Index | float] | None) -> pd.Index:
    if diagnostics is None:
        warn("Skipping 08_cooks_distance.png; diagnostics are unavailable.")
        return pd.Index([])

    idx = diagnostics["index"]
    cooks = diagnostics["cooks"]
    threshold = float(diagnostics["threshold"])

    x_axis = np.arange(len(cooks))
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(x_axis, cooks, s=12, alpha=0.75, color="steelblue")
    ax.axhline(threshold, color="crimson", linestyle="--", linewidth=1.2, label=f"4/n threshold = {threshold:.4f}")

    top_n = min(8, len(cooks))
    top_positions = np.argsort(cooks)[-top_n:]
    top_index_labels = [idx[i] for i in top_positions]
    top_ids = choose_id_values(df, top_index_labels)
    for pos, house_id in zip(top_positions, top_ids):
        ax.annotate(house_id, (x_axis[pos], cooks[pos]), xytext=(4, 4), textcoords="offset points", fontsize=8)

    ax.set_title("Cook's Distance for Baseline OLS-Style Model")
    ax.set_xlabel("Observation index")
    ax.set_ylabel("Cook's distance")
    ax.legend(frameon=False)
    save_fig(fig, "08_cooks_distance.png")

    influential_mask = cooks > threshold
    influential_index = pd.Index([idx[i] for i, m in enumerate(influential_mask) if m])
    if influential_index.empty:
        fallback_k = min(5, len(cooks))
        fallback_positions = np.argsort(cooks)[-fallback_k:]
        influential_index = pd.Index([idx[i] for i in fallback_positions])
        warn("No points exceeded 4/n threshold; using top 5 Cook's distance points as influential fallback.")
    info(f"Identified {len(influential_index)} influential points for impact test.")
    return influential_index


def make_linear_pipeline(X: pd.DataFrame) -> Pipeline:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    transformers = []

    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                num_cols,
            )
        )
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                cat_cols,
            )
        )

    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline([("preprocessor", pre), ("model", LinearRegression())])


def evaluate_rmse(df: pd.DataFrame, feature_cols: list[str], target_col: str = "logSalePrice") -> tuple[float, float] | None:
    existing = [c for c in feature_cols if c in df.columns]
    if not existing or target_col not in df.columns:
        return None

    work = df[existing + [target_col]].copy()
    work[target_col] = as_numeric(work[target_col])
    work = work.dropna(subset=[target_col])
    if work.shape[0] < 30:
        return None

    X = work[existing]
    y = work[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    pipe = make_linear_pipeline(X)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_val)
    holdout_rmse = float(np.sqrt(mean_squared_error(y_val, pred)))

    n_splits = 5 if len(work) >= 100 else max(3, min(5, len(work) // 10))
    if n_splits < 2:
        cv_rmse = np.nan
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = -cross_val_score(pipe, X, y, scoring="neg_root_mean_squared_error", cv=cv)
        cv_rmse = float(np.mean(cv_scores))

    return holdout_rmse, cv_rmse


def plot_09_outlier_impact_rmse(
    df: pd.DataFrame,
    influential_index: pd.Index,
    feature_cols: list[str],
) -> None:
    before = evaluate_rmse(df, feature_cols)
    if before is None:
        warn("Skipping 09_outlier_impact_rmse.png; unable to compute baseline RMSE.")
        return

    cleaned = df.drop(index=influential_index, errors="ignore")
    after = evaluate_rmse(cleaned, feature_cols)
    if after is None:
        warn("Skipping 09_outlier_impact_rmse.png; unable to compute post-removal RMSE.")
        return

    before_holdout, before_cv = before
    after_holdout, after_cv = after

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(2)
    width = 0.35
    holdout_vals = [before_holdout, after_holdout]
    cv_vals = [before_cv, after_cv]
    labels = ["Before", "After"]

    bars1 = ax.bar(x - width / 2, holdout_vals, width, label="Holdout RMSE", color="steelblue")
    bars2 = ax.bar(x + width / 2, cv_vals, width, label="KFold CV RMSE", color="darkorange")
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if np.isfinite(h):
            ax.annotate(f"{h:.3f}", (bar.get_x() + bar.get_width() / 2, h), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("RMSE on log1p(SalePrice)")
    ax.set_title("RMSE Impact of Removing Influential Points")
    ax.legend(frameon=False)
    save_fig(fig, "09_outlier_impact_rmse.png")

    info(f"RMSE before removal (holdout/CV): {before_holdout:.4f}/{before_cv:.4f}")
    info(f"RMSE after  removal (holdout/CV): {after_holdout:.4f}/{after_cv:.4f}")


def plot_10_totalsf_vs_logprice(df: pd.DataFrame) -> None:
    required = ["TotalSF", "logSalePrice"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        warn(f"Skipping 10_totalsf_vs_logprice.png; missing columns: {missing}")
        return

    work = df[required].copy()
    work["TotalSF"] = as_numeric(work["TotalSF"])
    work["logSalePrice"] = as_numeric(work["logSalePrice"])
    work = work.replace([np.inf, -np.inf], np.nan).dropna()

    if work.shape[0] < 20 or work["TotalSF"].nunique() < 2:
        warn("Skipping 10_totalsf_vs_logprice.png; not enough variation in TotalSF.")
        return

    corr = float(work["TotalSF"].corr(work["logSalePrice"]))
    x = work["TotalSF"].to_numpy()
    y = work["logSalePrice"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(x, y, s=18, alpha=0.35, color="teal")
    line_x = np.linspace(x.min(), x.max(), 300)
    ax.plot(line_x, slope * line_x + intercept, color="black", linewidth=1.2, alpha=0.8)
    ax.set_title(f"Engineered TotalSF vs log1p(SalePrice) (corr={corr:.3f})")
    ax.set_xlabel("TotalSF = GrLivArea + TotalBsmtSF")
    ax.set_ylabel("log1p(SalePrice)")
    save_fig(fig, "10_totalsf_vs_logprice.png")


def binned_median(x: pd.Series, y: pd.Series, max_bins: int = 10) -> tuple[np.ndarray, np.ndarray] | None:
    data = pd.DataFrame({"x": as_numeric(x), "y": as_numeric(y)}).dropna()
    if data.shape[0] < 20 or data["x"].nunique() < 2:
        return None

    q = min(max_bins, int(data["x"].nunique()))
    if q < 2:
        return None

    try:
        groups = pd.qcut(data["x"], q=q, duplicates="drop")
    except ValueError:
        return None

    medians = data.groupby(groups, observed=True)["y"].median()
    centers = np.array([interval.mid for interval in medians.index], dtype=float)
    values = medians.to_numpy(dtype=float)
    order = np.argsort(centers)
    return centers[order], values[order]


def plot_11_houseage_remodage_trends(df: pd.DataFrame) -> None:
    if "logSalePrice" not in df.columns:
        warn("Skipping 11_houseage_remodage_trends.png; missing logSalePrice.")
        return

    series_to_plot = []
    if "HouseAge" in df.columns:
        binned = binned_median(df["HouseAge"], df["logSalePrice"])
        if binned is not None:
            series_to_plot.append(("HouseAge", binned, "royalblue"))
    if "RemodAge" in df.columns:
        binned = binned_median(df["RemodAge"], df["logSalePrice"])
        if binned is not None:
            series_to_plot.append(("RemodAge", binned, "darkorange"))

    if not series_to_plot:
        warn("Skipping 11_houseage_remodage_trends.png; age features are unavailable or invalid.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, (xv, yv), color in series_to_plot:
        ax.plot(xv, yv, marker="o", linewidth=1.8, markersize=4, label=label, color=color)

    ax.set_title("Binned Median log1p(SalePrice) vs HouseAge / RemodAge")
    ax.set_xlabel("Age (years, binned midpoints)")
    ax.set_ylabel("Median log1p(SalePrice)")
    ax.legend(frameon=False)
    save_fig(fig, "11_houseage_remodage_trends.png")


def plot_12_lasso_coefficients(df: pd.DataFrame) -> None:
    if "logSalePrice" not in df.columns:
        warn("Skipping 12_lasso_coefficients.png; missing logSalePrice.")
        return

    drop_cols = {"SalePrice", "logSalePrice", "Id"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    if not feature_cols:
        warn("Skipping 12_lasso_coefficients.png; no modeling features are available.")
        return

    work = df[feature_cols + ["logSalePrice"]].copy()
    work["logSalePrice"] = as_numeric(work["logSalePrice"])
    work = work.dropna(subset=["logSalePrice"])
    if work.shape[0] < 30:
        warn("Skipping 12_lasso_coefficients.png; not enough rows.")
        return

    X = work[feature_cols]
    y = work["logSalePrice"].to_numpy(dtype=float)

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    transformers = []

    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            )
        )
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                cat_cols,
            )
        )

    if not transformers:
        warn("Skipping 12_lasso_coefficients.png; no usable feature transformers.")
        return

    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    X_mat = pre.fit_transform(X)

    lasso = LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=20000)
    lasso.fit(X_mat, y)

    feature_names = pre.get_feature_names_out()
    coef = lasso.coef_
    if coef.size == 0:
        warn("Skipping 12_lasso_coefficients.png; model returned empty coefficients.")
        return

    abs_coef = np.abs(coef)
    nonzero = np.where(abs_coef > 1e-8)[0]
    pool = nonzero if nonzero.size else np.arange(coef.size)
    top_k = min(15, pool.size)
    top_idx = pool[np.argsort(abs_coef[pool])[-top_k:]]
    top_idx = top_idx[np.argsort(abs_coef[top_idx])]

    names = [feature_names[i] for i in top_idx]
    vals = coef[top_idx]
    colors = ["seagreen" if v >= 0 else "indianred" for v in vals]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, vals, color=colors, alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Top 15 Lasso Coefficients by Absolute Magnitude")
    ax.set_xlabel("Coefficient value (signed)")
    save_fig(fig, "12_lasso_coefficients.png")


def main() -> None:
    data_path = find_cleaned_train_csv()
    if data_path is None:
        raise FileNotFoundError("Could not locate train_cleaned_v2.csv in root or data/ directories.")

    info(f"Loading data from: {data_path.resolve()}")
    df = pd.read_csv(data_path)
    df = add_engineered_features(df)

    if "logSalePrice" not in df.columns:
        raise ValueError("SalePrice is required to create logSalePrice and run these plots.")

    baseline_features = [
        "OverallQual",
        "GrLivArea",
        "TotalBsmtSF",
        "GarageCars",
        "YearBuilt",
    ]
    if "Neighborhood" in df.columns:
        baseline_features.append("Neighborhood")

    plot_07_outlier_candidates(df)
    diagnostics = compute_ols_diagnostics(df, baseline_features, target_col="logSalePrice")
    influential_index = plot_08_cooks_distance(df, diagnostics)
    plot_09_outlier_impact_rmse(df, influential_index, baseline_features)
    plot_10_totalsf_vs_logprice(df)
    plot_11_houseage_remodage_trends(df)
    plot_12_lasso_coefficients(df)
    info("Graph generation complete.")


if __name__ == "__main__":
    main()
