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

    info(f"Generating graph/07-12 via {GRAPH_SCRIPT}")
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


def fit_lasso_pipeline_result(df_input: pd.DataFrame) -> dict[str, object]:
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
            ("lasso", LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=20000)),
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
    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = -cross_val_score(
        model,
        X,
        y,
        cv=kfold,
        scoring="neg_root_mean_squared_error",
        n_jobs=None,
    )
    cv_rmse_log_mean = float(np.mean(cv_scores))
    cv_rmse_log_std = float(np.std(cv_scores, ddof=1))

    saleprice_true = np.expm1(y_holdout.to_numpy())
    saleprice_hat = np.expm1(y_hat_holdout)
    holdout_rmse_dollar = rmse(saleprice_true, saleprice_hat)
    holdout_mae_dollar = float(mean_absolute_error(saleprice_true, saleprice_hat))
    typical_relative_error = float(np.expm1(holdout_rmse_log))

    lasso = model.named_steps["lasso"]
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    coef = lasso.coef_

    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coef})
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df["sign"] = np.where(
        coef_df["coefficient"] > 0,
        "positive",
        np.where(coef_df["coefficient"] < 0, "negative", "zero"),
    )
    nonzero_coef_df = coef_df[coef_df["abs_coefficient"] > 1e-12].copy()
    nonzero_coef_df = nonzero_coef_df.sort_values("abs_coefficient", ascending=False)

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
        "best_alpha": float(getattr(lasso, "alpha_", getattr(lasso, "alpha", np.nan))),
        "intercept": float(lasso.intercept_),
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

    baseline_result = fit_lasso_pipeline_result(df)

    baseline_features = ["OverallQual", "GrLivArea", "TotalBsmtSF", "GarageCars", "YearBuilt"]
    if "Neighborhood" in df.columns:
        baseline_features.append("Neighborhood")
    influential_index, cooks_threshold = compute_cooks_influential_indices(
        df,
        feature_cols=baseline_features,
        target_col="logSalePrice",
    )
    influential_count = int(len(influential_index))

    filtered_result: dict[str, object] | None = None
    if influential_count > 0:
        filtered_df = df.drop(index=influential_index, errors="ignore")
        if filtered_df.shape[0] >= 120:
            try:
                filtered_result = fit_lasso_pipeline_result(filtered_df)
            except Exception as exc:
                info(f"Filtered model skipped due to fitting error: {exc}")

    selected_label = "baseline"
    selected_reason = "Baseline is used as default model."
    selected_result = baseline_result
    improvement_ratio = 0.0
    if filtered_result is not None:
        baseline_cv = float(baseline_result["cv_rmse_log_mean"])
        filtered_cv = float(filtered_result["cv_rmse_log_mean"])
        if baseline_cv > 0:
            improvement_ratio = (baseline_cv - filtered_cv) / baseline_cv
        if improvement_ratio >= 0.03:
            selected_label = "filtered"
            selected_result = filtered_result
            selected_reason = (
                f"Filtered CV RMSE improved by {improvement_ratio:.2%}; choose filtered model."
            )
        else:
            selected_reason = (
                f"Filtered CV RMSE improvement is {improvement_ratio:.2%}, "
                "so baseline is considered sufficiently robust."
            )
    else:
        selected_reason = (
            "Influence-filtered variant was unavailable or not stable; baseline is retained."
        )

    active = selected_result
    nonzero_coef_df = active["nonzero_coef_df"]  # type: ignore[assignment]
    top_coefficients = select_top_coefficients(nonzero_coef_df, top_n=TOP_COEF_EXPORT)
    top_coefficients.to_csv(OUTPUT_DIR / "top_coefficients.csv", index=False)

    top10 = nonzero_coef_df.head(TOP_EQUATION).copy()
    intercept = float(active["intercept"])

    equation_terms = []
    for _, row in top10.iterrows():
        sign = "+" if row["coefficient"] >= 0 else "-"
        equation_terms.append(f" {sign} {abs(row['coefficient']):.6f} * {row['feature']}")
    equation_expanded = f"y_hat = {intercept:.6f}" + "".join(equation_terms)

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

    baseline_cv_text = float(baseline_result["cv_rmse_log_mean"])
    filtered_cv_text = float(filtered_result["cv_rmse_log_mean"]) if filtered_result is not None else None
    filtered_holdout_text = (
        float(filtered_result["holdout_rmse_log"]) if filtered_result is not None else None
    )

    metrics = {
        "data_path": str(data_path.resolve()),
        "target": "logSalePrice = log1p(SalePrice)",
        "selected_model": selected_label,
        "selected_model_reason": selected_reason,
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
        "cooks_distance_threshold_4_over_n": (
            float(cooks_threshold) if np.isfinite(cooks_threshold) else None
        ),
        "influential_points_count": influential_count,
        "n_rows_total": int(df.shape[0]),
        "n_rows_model": int(active["n_rows"]),
        "n_features_before_encoding": int(active["n_features_before_encoding"]),
        "n_features_after_encoding": int(active["n_features_after_encoding"]),
        "baseline_holdout_rmse_log": float(baseline_result["holdout_rmse_log"]),
        "baseline_cv_rmse_log_mean": baseline_cv_text,
        "filtered_holdout_rmse_log": filtered_holdout_text,
        "filtered_cv_rmse_log_mean": filtered_cv_text,
        "filtered_improvement_ratio_vs_baseline_cv": (
            float(improvement_ratio) if filtered_result is not None else None
        ),
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    top10_table = [
        "| Rank | Feature | Coefficient |",
        "| --- | --- | --- |",
    ]
    for i, (_, row) in enumerate(top10.iterrows(), start=1):
        top10_table.append(f"| {i} | {row['feature']} | {row['coefficient']:.6f} |")

    positive_top10 = nonzero_coef_df[nonzero_coef_df["coefficient"] > 0].head(10)
    negative_top10 = nonzero_coef_df[nonzero_coef_df["coefficient"] < 0].head(10)

    pos_lines = ["| Feature | Coefficient |", "| --- | --- |"]
    for _, row in positive_top10.iterrows():
        pos_lines.append(f"| {row['feature']} | {row['coefficient']:.6f} |")

    neg_lines = ["| Feature | Coefficient |", "| --- | --- |"]
    for _, row in negative_top10.iterrows():
        neg_lines.append(f"| {row['feature']} | {row['coefficient']:.6f} |")

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

    pipeline_code = """
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
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
    ("lasso", LassoCV(cv=5, random_state=42)),
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
            f"- Selected model: **{selected_label}**",
            f"- Selection rationale: {selected_reason}",
            "",
            "## 1) Coding (Core Pipeline)",
            "",
            "```python",
            pipeline_code,
            "```",
            "",
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
            "",
            *top10_table,
            "",
            "## 3) Regression Result",
            "",
            f"- Best alpha: **{best_alpha:.8f}**",
            f"- Non-zero coefficients: **{nonzero_count}**",
            "- Top coefficients file: `outputs/top_coefficients.csv`",
            "- CV protocol: `KFold(n_splits=5, shuffle=True, random_state=42)` with `neg_root_mean_squared_error` on log1p scale.",
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
            "Cook's Distance threshold (4/n rule):",
            (
                f"- D_i > 4/n, where n = {len(df)} and 4/n = **{cooks_threshold:.6f}**."
                if np.isfinite(cooks_threshold)
                else "- Cook's distance threshold unavailable for this run."
            ),
            "",
            "Outlier impact figure:",
            "- See `../graph/09_outlier_impact_rmse.png` for RMSE before and after removing high-influence points.",
            "- The chart is a sensitivity check: a noticeable RMSE drop indicates a small set of influential points drives error disproportionately.",
            "- If the change is small, model performance is relatively robust to those candidate outliers.",
        ]
    ) + "\n"

    report_md_path = OUTPUT_DIR / "Model_Report.md"
    report_html_path = OUTPUT_DIR / "Model_Report.html"
    report_pdf_path = OUTPUT_DIR / "Model_Report.pdf"
    report_md_path.write_text(report_md, encoding="utf-8")

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


if __name__ == "__main__":
    main()
