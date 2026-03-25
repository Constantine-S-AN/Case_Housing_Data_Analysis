#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
HOLDOUT_SIZE = 0.2
TOP_COEF_EXPORT = 30
TOP_EQUATION = 10

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = ROOT_DIR / "outputs"
GRAPH_SCRIPT = ROOT_DIR / "graph" / "generate_graphs_v3.py"


def info(message: str) -> None:
    print(f"[INFO] {message}")


def find_cleaned_train_csv() -> Path:
    """Locate cleaned housing training data.

    Priority:
    1) train_cleaned_v2.csv
    2) train_cleaned.csv
    Then fallback recursive search under cwd and project root.
    """
    filenames = ["train_cleaned_v2.csv", "train_cleaned.csv"]
    candidates: list[Path] = []

    for name in filenames:
        candidates.extend(
            [
                Path.cwd() / name,
                Path.cwd() / "data" / name,
                ROOT_DIR / name,
                ROOT_DIR / "data" / name,
            ]
        )

    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.exists():
            return path

    for name in filenames:
        for path in Path.cwd().rglob(name):
            return path
        for path in ROOT_DIR.rglob(name):
            return path

    raise FileNotFoundError("Could not locate cleaned data file train_cleaned_v2.csv or train_cleaned.csv")


def make_onehot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def run_graph_generator() -> None:
    if not GRAPH_SCRIPT.exists():
        raise FileNotFoundError(f"Missing graph generator script: {GRAPH_SCRIPT}")

    info(f"Generating graph/07-16 via {GRAPH_SCRIPT}")
    subprocess.run(
        [sys.executable, str(GRAPH_SCRIPT)],
        check=True,
        cwd=ROOT_DIR,
    )


def select_top_coefficients(nonzero_coef_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    positive = nonzero_coef_df[nonzero_coef_df["coefficient"] > 0].sort_values(
        "abs_coefficient", ascending=False
    )
    negative = nonzero_coef_df[nonzero_coef_df["coefficient"] < 0].sort_values(
        "abs_coefficient", ascending=False
    )

    selected = pd.concat([positive.head(top_n // 2), negative.head(top_n // 2)])

    if len(selected) < top_n:
        remaining = nonzero_coef_df.drop(index=selected.index, errors="ignore")
        selected = pd.concat([selected, remaining.head(top_n - len(selected))])

    selected = (
        selected.sort_values("abs_coefficient", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    selected.insert(0, "rank", np.arange(1, len(selected) + 1))
    return selected


def inline_markdown_to_html(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    return escaped


def markdown_table_to_html(lines: list[str]) -> str:
    rows = []
    for line in lines:
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)

    if len(rows) < 2:
        return ""

    headers = rows[0]
    body_rows = rows[2:]

    parts = ["<table>", "<thead><tr>"]
    for cell in headers:
        parts.append(f"<th>{inline_markdown_to_html(cell)}</th>")
    parts.append("</tr></thead><tbody>")

    for row in body_rows:
        parts.append("<tr>")
        for cell in row:
            parts.append(f"<td>{inline_markdown_to_html(cell)}</td>")
        parts.append("</tr>")

    parts.append("</tbody></table>")
    return "".join(parts)


def simple_markdown_to_html(markdown_text: str, title: str = "Model Report") -> str:
    lines = markdown_text.splitlines()
    html_parts = [
        "<!DOCTYPE html>",
        "<html lang=\"en\">",
        "<head>",
        "<meta charset=\"UTF-8\" />",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />",
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;line-height:1.6;color:#111;max-width:980px;margin:2rem auto;padding:0 1rem;}",
        "pre{background:#f5f5f5;border:1px solid #ddd;border-radius:6px;padding:.9rem;overflow-x:auto;}",
        "code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;}",
        "table{border-collapse:collapse;width:100%;margin:.7rem 0 1.2rem;}",
        "th,td{border:1px solid #ddd;padding:.5rem .6rem;text-align:left;font-size:.95rem;}",
        "th{background:#f3f4f6;}",
        "img{max-width:100%;border:1px solid #ddd;border-radius:4px;}",
        "</style>",
        "</head>",
        "<body>",
    ]

    in_code = False
    code_lines: list[str] = []
    in_list = False
    table_buffer: list[str] = []

    def flush_code() -> None:
        nonlocal code_lines
        if code_lines:
            code_text = "\n".join(code_lines)
            html_parts.append(f"<pre><code>{html.escape(code_text)}</code></pre>")
            code_lines = []

    def flush_list() -> None:
        nonlocal in_list
        if in_list:
            html_parts.append("</ul>")
            in_list = False

    def flush_table() -> None:
        nonlocal table_buffer
        if table_buffer:
            html_parts.append(markdown_table_to_html(table_buffer))
            table_buffer = []

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_table()
            flush_list()
            if not in_code:
                in_code = True
                code_lines = []
            else:
                in_code = False
                flush_code()
            continue

        if in_code:
            code_lines.append(line)
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            flush_list()
            table_buffer.append(stripped)
            continue
        else:
            flush_table()

        if not stripped:
            flush_list()
            continue

        if stripped.startswith("### "):
            flush_list()
            html_parts.append(f"<h3>{inline_markdown_to_html(stripped[4:])}</h3>")
            continue

        if stripped.startswith("## "):
            flush_list()
            html_parts.append(f"<h2>{inline_markdown_to_html(stripped[3:])}</h2>")
            continue

        if stripped.startswith("# "):
            flush_list()
            html_parts.append(f"<h1>{inline_markdown_to_html(stripped[2:])}</h1>")
            continue

        image_match = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
        if image_match:
            flush_list()
            alt, src = image_match.groups()
            html_parts.append(
                f"<p><img src=\"{html.escape(src)}\" alt=\"{html.escape(alt)}\" /></p>"
            )
            continue

        if stripped.startswith("- "):
            flush_table()
            if not in_list:
                html_parts.append("<ul>")
                in_list = True
            html_parts.append(f"<li>{inline_markdown_to_html(stripped[2:])}</li>")
            continue

        flush_list()
        html_parts.append(f"<p>{inline_markdown_to_html(stripped)}</p>")

    flush_table()
    flush_list()
    if in_code:
        flush_code()

    html_parts.extend(["</body>", "</html>"])
    return "\n".join(html_parts) + "\n"


def write_fallback_pdf(markdown_text: str, pdf_path: Path, image_paths: list[Path]) -> None:
    from matplotlib.backends.backend_pdf import PdfPages

    lines = markdown_text.splitlines()
    lines_per_page = 42

    with PdfPages(pdf_path) as pdf:
        # Text pages
        for start in range(0, len(lines), lines_per_page):
            chunk = lines[start : start + lines_per_page]
            fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")  # A4 portrait
            fig.patch.set_facecolor("white")
            fig.text(
                0.06,
                0.965,
                "\n".join(chunk),
                va="top",
                ha="left",
                fontsize=8.5,
                family="monospace",
            )
            # Avoid implicit axes and tight-bbox cropping artifacts.
            pdf.savefig(fig)
            plt.close(fig)

        # Figure pages
        for image_path in image_paths:
            if not image_path.exists():
                continue
            try:
                img = plt.imread(str(image_path))
            except Exception:
                continue

            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.imshow(img)
            ax.set_title(image_path.name, fontsize=12)
            ax.axis("off")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def compute_cooks_influential_indices(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "logSalePrice",
) -> tuple[pd.Index, float]:
    existing = [c for c in feature_cols if c in df.columns]
    if not existing or target_col not in df.columns:
        return pd.Index([]), float("nan")

    work = df[existing + [target_col]].copy()
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.dropna(subset=[target_col])
    if work.shape[0] < 25:
        return pd.Index([]), float("nan")

    X = work[existing].copy()
    for col in X.columns:
        if X[col].dtype == object or str(X[col].dtype).startswith("category"):
            X[col] = X[col].fillna("Missing").astype(str)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            fill = X[col].median()
            if pd.isna(fill):
                fill = 0.0
            X[col] = X[col].fillna(fill)

    X_enc = pd.get_dummies(X, drop_first=True)
    if X_enc.shape[1] == 0:
        return pd.Index([]), float("nan")

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
    threshold = float(4.0 / n)
    influential_mask = cooks > threshold
    influential_idx = pd.Index([work.index[i] for i, hit in enumerate(influential_mask) if hit])
    return influential_idx, threshold


def get_model_configs() -> dict[str, object]:
    """Return dictionary of model configurations to compare."""
    return {
        "lasso": LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=20000),
        "random_forest": RandomForestRegressor(
            n_estimators=50,  # Reduced for faster testing
            max_depth=10,     # Reduced depth
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=1  # Use single job to avoid threading issues
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=20,  # Reduced from 50
            learning_rate=0.1,
            max_depth=2,      # Reduced from 3
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE
        ),
    }


def fit_model_pipeline_result(df_input: pd.DataFrame, model_name: str, model_estimator) -> dict[str, object]:
    excluded_cols = {"SalePrice", "logSalePrice", "Id"}
    feature_cols = [c for c in df_input.columns if c not in excluded_cols]
    if not feature_cols:
        raise ValueError("No usable feature columns after excluding SalePrice/logSalePrice/Id")

    X = df_input[feature_cols].copy()
    y = df_input["logSalePrice"].copy()

    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_onehot()),
                    ]
                ),
                categorical_cols,
            )
        )
    if not transformers:
        raise ValueError("No valid transformers could be created for model pipeline")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (model_name, model_estimator),
        ]
    )

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=HOLDOUT_SIZE,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)
    y_hat_holdout = model.predict(X_holdout)

    holdout_rmse_log = rmse(y_holdout.to_numpy(), y_hat_holdout)
    kfold = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)  # Reduced from 5 to 3
    cv_scores = -cross_val_score(
        model,
        X,
        y,
        cv=kfold,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,  # Use single job to avoid issues
    )
    cv_rmse_log_mean = float(np.mean(cv_scores))
    cv_rmse_log_std = float(np.std(cv_scores, ddof=1))

    saleprice_true = np.expm1(y_holdout.to_numpy())
    saleprice_hat = np.expm1(y_hat_holdout)
    holdout_rmse_dollar = rmse(saleprice_true, saleprice_hat)
    holdout_mae_dollar = float(mean_absolute_error(saleprice_true, saleprice_hat))
    typical_relative_error = float(np.expm1(holdout_rmse_log))

    fitted_model = model.named_steps[model_name]
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()

    # Handle coefficients/feature importances based on model type
    if hasattr(fitted_model, 'coef_'):
        # Linear models like Lasso
        coef = fitted_model.coef_
        coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coef})
        coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
        coef_df["sign"] = np.where(
            coef_df["coefficient"] > 0,
            "positive",
            np.where(coef_df["coefficient"] < 0, "negative", "zero"),
        )
        nonzero_coef_df = coef_df[coef_df["abs_coefficient"] > 1e-12].copy()
        nonzero_coef_df = nonzero_coef_df.sort_values("abs_coefficient", ascending=False)
    elif hasattr(fitted_model, 'feature_importances_'):
        # Tree-based models
        importances = fitted_model.feature_importances_
        coef_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        coef_df["abs_coefficient"] = coef_df["importance"].abs()
        coef_df["coefficient"] = coef_df["importance"]  # For compatibility
        coef_df["sign"] = "importance"
        nonzero_coef_df = coef_df[coef_df["abs_coefficient"] > 1e-12].copy()
        nonzero_coef_df = nonzero_coef_df.sort_values("abs_coefficient", ascending=False)
    else:
        # Fallback for models without coefficients or importances
        coef_df = pd.DataFrame({"feature": feature_names, "coefficient": [0.0] * len(feature_names)})
        coef_df["abs_coefficient"] = 0.0
        coef_df["sign"] = "unknown"
        nonzero_coef_df = coef_df.copy()

    return {
        "X": X,
        "y": y,
        "model": model,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "feature_names": feature_names,
        "coef_df": coef_df,
        "nonzero_coef_df": nonzero_coef_df,
        "nonzero_count": int(nonzero_coef_df.shape[0]),
        "best_alpha": float(getattr(fitted_model, "alpha_", getattr(fitted_model, "alpha", np.nan))),
        "intercept": float(getattr(fitted_model, "intercept_", 0.0)),
        "model_name": model_name,
        "y_holdout_log": y_holdout.to_numpy(),
        "y_hat_holdout_log": y_hat_holdout,
        "saleprice_true_holdout": saleprice_true,
        "saleprice_hat_holdout": saleprice_hat,
        "holdout_rmse_log": holdout_rmse_log,
        "cv_rmse_log_mean": cv_rmse_log_mean,
        "cv_rmse_log_std": cv_rmse_log_std,
        "cv_rmse_log_folds": cv_scores.tolist(),
        "holdout_rmse_dollar": holdout_rmse_dollar,
        "holdout_mae_dollar": holdout_mae_dollar,
        "typical_relative_error": typical_relative_error,
        "n_rows": int(df_input.shape[0]),
        "n_features_before_encoding": int(X.shape[1]),
        "n_features_after_encoding": int(len(feature_names)),
    }


def fit_lasso_pipeline_result(df_input: pd.DataFrame) -> dict[str, object]:
    """Legacy function for backward compatibility."""
    return fit_model_pipeline_result(df_input, "lasso", get_model_configs()["lasso"])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_graph_generator()

    data_path = find_cleaned_train_csv()
    info(f"Using cleaned data: {data_path.resolve()}")
    df = pd.read_csv(data_path)

    if "SalePrice" not in df.columns:
        raise ValueError("Input data must contain SalePrice")

    df = df.copy()
    df["SalePrice"] = pd.to_numeric(df["SalePrice"], errors="coerce")
    df = df.dropna(subset=["SalePrice"])
    df["logSalePrice"] = np.log1p(df["SalePrice"])

    # Fit and compare multiple models
    model_configs = get_model_configs()
    model_results = {}
    
    info("Fitting and comparing multiple models...")
    for model_name, model_estimator in model_configs.items():
        info(f"Fitting {model_name}...")
        try:
            result = fit_model_pipeline_result(df, model_name, model_estimator)
            model_results[model_name] = result
            cv_rmse = result["cv_rmse_log_mean"]
            holdout_rmse = result["holdout_rmse_log"]
            info(f"  {model_name}: CV RMSE = {cv_rmse:.4f}, Holdout RMSE = {holdout_rmse:.4f}")
        except Exception as exc:
            info(f"  {model_name} failed: {exc}")
            continue

    # Select best model based on CV RMSE
    if not model_results:
        raise ValueError("No models could be fitted successfully")
    
    best_model_name = min(model_results.keys(), key=lambda k: model_results[k]["cv_rmse_log_mean"])
    best_result = model_results[best_model_name]
    info(f"Selected best model: {best_model_name} (CV RMSE: {best_result['cv_rmse_log_mean']:.4f})")

    # For backward compatibility, set baseline_result to the best model
    baseline_result = best_result

    # Skip influential points filtering for model comparison
    influential_count = 0
    filtered_result = None
    selected_label = best_model_name
    selected_reason = f"Selected {best_model_name} as it had the lowest CV RMSE among compared models"
    selected_result = best_result

    active = selected_result
    nonzero_coef_df = active["nonzero_coef_df"]  # type: ignore[assignment]
    top_coefficients = select_top_coefficients(nonzero_coef_df, top_n=TOP_COEF_EXPORT)
    top_coefficients.to_csv(OUTPUT_DIR / "top_coefficients.csv", index=False)

    # Generate equation for Lasso only
    equation_expanded = ""
    if best_model_name == "lasso":
        top10 = nonzero_coef_df.head(TOP_EQUATION).copy()
        intercept = float(active["intercept"])
        equation_terms = []
        for _, row in top10.iterrows():
            sign = "+" if row["coefficient"] >= 0 else "-"
            equation_terms.append(f" {sign} {abs(row['coefficient']):.6f} * {row['feature']}")
        equation_expanded = f"y_hat = {intercept:.6f}" + "".join(equation_terms)
    else:
        top10 = nonzero_coef_df.head(TOP_EQUATION).copy()

    y_holdout_log = np.asarray(active["y_holdout_log"], dtype=float)
    y_hat_holdout_log = np.asarray(active["y_hat_holdout_log"], dtype=float)
    holdout_rmse_log = float(active["holdout_rmse_log"])
    cv_rmse_log_mean = float(active["cv_rmse_log_mean"])
    cv_rmse_log_std = float(active["cv_rmse_log_std"])
    cv_rmse_log_folds = [float(v) for v in active["cv_rmse_log_folds"]]  # type: ignore[arg-type]
    holdout_rmse_dollar = float(active["holdout_rmse_dollar"])
    holdout_mae_dollar = float(active["holdout_mae_dollar"])
    typical_relative_error = float(active["typical_relative_error"])
    best_alpha = float(active["best_alpha"])
    nonzero_count = int(active["nonzero_count"])

    # Plot holdout predicted vs actual in log space for selected model.
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_holdout_log, y_hat_holdout_log, alpha=0.72, s=34, edgecolor="white", linewidth=0.3)

    lower = float(min(y_holdout_log.min(), y_hat_holdout_log.min()))
    upper = float(max(y_holdout_log.max(), y_hat_holdout_log.max()))
    ax.plot([lower, upper], [lower, upper], "r--", linewidth=2, label="y = x")

    ax.set_xlabel("Actual logSalePrice")
    ax.set_ylabel("Predicted logSalePrice")
    ax.set_title(f"Holdout: Predicted vs Actual (logSalePrice, {selected_label})")
    ax.text(
        0.03,
        0.97,
        f"Holdout RMSE(log) = {holdout_rmse_log:.6f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#999"},
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pred_vs_actual.png", dpi=180)
    plt.close(fig)

    # Add model comparison results
    model_comparison = {}
    for model_name, result in model_results.items():
        model_comparison[model_name] = {
            "cv_rmse_log_mean": float(result["cv_rmse_log_mean"]),
            "holdout_rmse_log": float(result["holdout_rmse_log"]),
            "holdout_rmse_dollar": float(result["holdout_rmse_dollar"]),
            "holdout_mae_dollar": float(result["holdout_mae_dollar"]),
            "nonzero_count": int(result["nonzero_count"]),
        }

    metrics = {
        "data_path": str(data_path.resolve()),
        "target": "logSalePrice = log1p(SalePrice)",
        "selected_model": best_model_name,
        "selected_model_reason": f"Selected {best_model_name} as it had the lowest CV RMSE among compared models",
        "model_comparison": model_comparison,
        "holdout_rmse": holdout_rmse_log,
        "holdout_rmse_log": holdout_rmse_log,
        "cv_rmse": cv_rmse_log_mean,
        "cv_rmse_log_mean": cv_rmse_log_mean,
        "cv_rmse_log_std": cv_rmse_log_std,
        "cv_rmse_log_folds": cv_rmse_log_folds,
        "holdout_rmse_dollar": holdout_rmse_dollar,
        "holdout_mae_dollar": holdout_mae_dollar,
        "typical_relative_error": typical_relative_error,
        "best_alpha": best_alpha,
        "nonzero_count": nonzero_count,
        "cooks_distance_threshold_4_over_n": None,
        "influential_points_count": 0,
        "n_rows_total": int(df.shape[0]),
        "n_rows_model": int(active["n_rows"]),
        "n_features_before_encoding": int(active["n_features_before_encoding"]),
        "n_features_after_encoding": int(active["n_features_after_encoding"]),
        "baseline_holdout_rmse_log": float(baseline_result["holdout_rmse_log"]),
        "baseline_cv_rmse_log_mean": float(baseline_result["cv_rmse_log_mean"]),
        "filtered_holdout_rmse_log": None,
        "filtered_cv_rmse_log_mean": None,
        "filtered_improvement_ratio_vs_baseline_cv": None,
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Define positive and negative coefficient/feature tables
    positive_top10 = nonzero_coef_df[nonzero_coef_df["coefficient"] > 0].head(10)
    negative_top10 = nonzero_coef_df[nonzero_coef_df["coefficient"] < 0].head(10)

    pos_lines = ["| Feature | Coefficient |", "| --- | --- |"]
    for _, row in positive_top10.iterrows():
        pos_lines.append(f"| {row['feature']} | {row['coefficient']:.6f} |")

    neg_lines = ["| Feature | Coefficient |", "| --- | --- |"]
    for _, row in negative_top10.iterrows():
        neg_lines.append(f"| {row['feature']} | {row['coefficient']:.6f} |")

    top10_table = [
        "| Rank | Feature | Coefficient |",
        "| --- | --- | --- |",
    ]
    for i, (_, row) in enumerate(top10.iterrows(), start=1):
        top10_table.append(f"| {i} | {row['feature']} | {row['coefficient']:.6f} |")

    rmse_summary_table = [
        "| Metric | Value |",
        "| --- | --- |",
        f"| holdout_rmse_log | {holdout_rmse_log:.6f} |",
        f"| cv_rmse_log_mean | {cv_rmse_log_mean:.6f} |",
        f"| holdout_rmse_dollar | {holdout_rmse_dollar:.2f} |",
        f"| holdout_mae_dollar | {holdout_mae_dollar:.2f} |",
        (
            "| typical_relative_error = exp(holdout_rmse_log)-1 | "
            f"{typical_relative_error:.6f} ({typical_relative_error * 100:.2f}%) |"
        ),
    ]

    sensitivity_table = []
    if filtered_result is not None:
        baseline_selected = "Yes" if selected_label == "baseline" else "No"
        filtered_selected = "Yes" if selected_label == "filtered" else "No"
        sensitivity_table = [
            "| Model | n_rows | holdout_rmse_log | cv_rmse_log_mean | selected |",
            "| --- | --- | --- | --- | --- |",
            (
                f"| Baseline | {int(baseline_result['n_rows'])} | "
                f"{float(baseline_result['holdout_rmse_log']):.6f} | "
                f"{float(baseline_result['cv_rmse_log_mean']):.6f} | {baseline_selected} |"
            ),
            (
                f"| Filtered (Cook's D > 4/n) | {int(filtered_result['n_rows'])} | "
                f"{float(filtered_result['holdout_rmse_log']):.6f} | "
                f"{float(filtered_result['cv_rmse_log_mean']):.6f} | {filtered_selected} |"
            ),
        ]

    pandoc_available = shutil.which("pandoc") is not None
    if pandoc_available:
        note_line = "**Pandoc detected. PDF export is enabled.**"
    else:
        note_line = "**Pandoc not found. HTML is generated, and Python fallback also exports PDF.**"

    # Get string representation of the selected model
    model_estimator = model_configs[best_model_name]
    if best_model_name == "lasso":
        model_repr = "LassoCV(cv=5, random_state=42)"
        model_import = "from sklearn.linear_model import LassoCV"
    elif best_model_name == "random_forest":
        model_repr = "RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1)"
        model_import = "from sklearn.ensemble import RandomForestRegressor"
    elif best_model_name == "gradient_boosting":
        model_repr = "GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, min_samples_split=5, min_samples_leaf=2, random_state=42)"
        model_import = "from sklearn.ensemble import GradientBoostingRegressor"
    else:
        model_repr = str(type(model_estimator).__name__) + "(...)"
        model_import = f"from {type(model_estimator).__module__} import {type(model_estimator).__name__}"

    pipeline_code = f"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
{model_import}
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_cols),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("{best_model_name}", {model_repr}),
])
""".strip()

    report_md = "\n".join(
        [
            "# Model Report",
            "",
            note_line,
            "",
            "## Model Choice",
            "",
            f"- Selected model: **{best_model_name}**",
            f"- Selection rationale: {selected_reason}",
            "",
            "## Model Comparison",
            "",
            "Performance comparison across different models:",
            "",
            "| Model | CV RMSE (log) | Holdout RMSE (log) | Holdout RMSE ($) | Holdout MAE ($) |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    
    for model_name in sorted(model_results.keys()):
        result = model_results[model_name]
        report_md += f"| {model_name} | {result['cv_rmse_log_mean']:.4f} | {result['holdout_rmse_log']:.4f} | {result['holdout_rmse_dollar']:.0f} | {result['holdout_mae_dollar']:.0f} |\n"
    
    if best_model_name == "lasso":
        section2 = [
            "## 2) Regression Equation",
            "",
            "General form:",
            "- y = log1p(SalePrice)",
            "- y_hat = beta0 + sum(beta_j * x_j_tilde)",
            "",
            "Expanded equation (Top10 coefficients only):",
            "",
            "```text",
            equation_expanded,
            "```",
            "",
            "Top10 absolute coefficients:",
        ]
        section3_extra = [
            "",
            "Top positive coefficients:",
            "",
            *pos_lines,
            "",
            "Top negative coefficients:",
            "",
            *neg_lines,
            "",
            "### Coefficient interpretation",
            "- Numeric features use StandardScaler, so each numeric coefficient means expected change in log1p(SalePrice) for a +1 standard deviation change.",
            "- Categorical features are one-hot encoded, so each category coefficient is interpreted relative to the omitted baseline category.",
            "- Rare-category coefficients can be unstable; treat them as predictive signals rather than causal effects.",
        ]
    else:
        section2 = [
            "## 2) Feature Importances",
            "",
            f"Top10 feature importances for {best_model_name}:",
        ]
        section3_extra = [
            "",
            "### Feature importance interpretation",
            "- Feature importances show the relative contribution of each feature to the model's predictions.",
            "- Higher values indicate features that are more important for prediction.",
            "- Importances are calculated based on how much each feature contributes to reducing impurity in the trees.",
        ]
    
    report_md += "\n" + "\n".join([
            "## 1) Coding (Core Pipeline)",
            "",
            "```python",
            pipeline_code,
            "```",
            "",
            *section2,
            "",
            *top10_table,
            "",
            "## 3) Model Result",
            "",
            f"- Model type: **{best_model_name}**",
            f"- Best alpha: **{best_alpha:.8f}**" if best_model_name == "lasso" else f"- Model parameters: {model_repr}",
            f"- Non-zero features: **{nonzero_count}**",
            "- Top coefficients/importances file: `outputs/top_coefficients.csv`",
            "- CV protocol: `KFold(n_splits=5, shuffle=True, random_state=42)` with `neg_root_mean_squared_error` on log1p scale.",
            *section3_extra,
            "",
            "## 4) Graph: Predicted vs Actual",
            "",
            "![Holdout Predicted vs Actual](pred_vs_actual.png)",
            "",
            "## 5) RMSE",
            "",
            "RMSE (log1p scale) formula:",
            "- RMSE_log = sqrt((1/n) * sum((y_i - y_hat_i)^2)), where y = log1p(SalePrice).",
            "- This log-scale error emphasizes multiplicative/relative discrepancy.",
            "",
            "Back-transform and dollar-scale metrics:",
            "- SalePrice_hat = exp(y_hat) - 1",
            "- SalePrice_true = exp(y_true) - 1",
            "- typical_relative_error = exp(holdout_rmse_log) - 1",
            "",
            *rmse_summary_table,
            "",
            f"- 5-fold CV RMSE (folds): {', '.join(f'{v:.6f}' for v in cv_rmse_log_folds)}",
            "",
            "### Baseline vs Filtered (Sensitivity Analysis)",
            *(sensitivity_table if sensitivity_table else ["- Filtered sensitivity run was unavailable in this execution."]),
            "- This is a sensitivity analysis for influential points, not arbitrary deletion of data.",
            "- A better filtered score indicates influence sensitivity; otherwise baseline is already robust.",
            "",
            "## Appendix: Outlier Diagnostics",
            "",
            "Note: Outlier filtering was not applied in this model comparison run.",
            "",
            "Outlier impact figure:",
            "- See `../graph/09_outlier_impact_rmse.png` for RMSE before and after removing high-influence points.",
            "- The chart is a sensitivity check: a noticeable RMSE drop indicates a small set of influential points drives error disproportionately.",
            "- If the change is small, model performance is relatively robust to those candidate outliers.",
            "",
            "## Appendix: Model Comparison Visualizations", 
            "",
            "Feature importance comparison:",
            "- See `../graph/12_lasso_coefficients.png` for top Lasso coefficients by absolute magnitude.", 
            "- See `../graph/13_random_forest_importances.png` for top Random Forest feature importances.", 
            "- See `../graph/14_gradient_boosting_importances.png` for top Gradient Boosting feature importances.", 
            "- See `../graph/15_model_comparison_rmse.png` for RMSE comparison across all models.", 
            "- See `../graph/16_residual_distributions.png` for residual distribution comparison across models.",
        ]
    ) + "\n"

    report_md_path = OUTPUT_DIR / "Model_Report.md"
    report_html_path = OUTPUT_DIR / "Model_Report.html"
    report_pdf_path = OUTPUT_DIR / "Model_Report.pdf"
    
    info(f"Writing markdown report to {report_md_path}")
    report_md_path.write_text(report_md, encoding="utf-8")
    info(f"Markdown report written successfully")

    pdf_generated = False
    if pandoc_available:
        try:
            subprocess.run(
                ["pandoc", str(report_md_path), "-o", str(report_pdf_path)],
                check=True,
                cwd=OUTPUT_DIR,
            )
            pdf_generated = True
        except subprocess.CalledProcessError:
            pdf_generated = False

    if not pdf_generated:
        report_html = simple_markdown_to_html(report_md, title="Model Report")
        report_html_path.write_text(report_html, encoding="utf-8")
        fallback_images = [
            OUTPUT_DIR / "pred_vs_actual.png",
            ROOT_DIR / "graph" / "09_outlier_impact_rmse.png",
            ROOT_DIR / "graph" / "08_cooks_distance.png",
        ]
        try:
            write_fallback_pdf(report_md, report_pdf_path, fallback_images)
            pdf_generated = True
        except Exception as exc:
            info(f"Fallback PDF generation failed: {exc}")

    info(f"Wrote {OUTPUT_DIR / 'top_coefficients.csv'}")
    info(f"Wrote {OUTPUT_DIR / 'pred_vs_actual.png'}")
    info(f"Wrote {OUTPUT_DIR / 'metrics.json'}")
    info(f"Wrote {OUTPUT_DIR / 'Model_Report.md'}")
    if report_html_path.exists():
        info(f"Wrote {report_html_path}")
    if pdf_generated and report_pdf_path.exists():
        info(f"Wrote {report_pdf_path}")

    info("All reports generated successfully")


if __name__ == "__main__":
    main()
