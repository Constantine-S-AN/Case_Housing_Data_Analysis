#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from build_model_report import (
    HOLDOUT_SIZE,
    OUTPUT_DIR,
    RANDOM_STATE,
    find_cleaned_train_csv,
    info,
    rmse,
    simple_markdown_to_html,
    write_fallback_pdf,
)


REPORT_MD = OUTPUT_DIR / "Basic_Model_Report.md"
REPORT_HTML = OUTPUT_DIR / "Basic_Model_Report.html"
REPORT_PDF = OUTPUT_DIR / "Basic_Model_Report.pdf"
METRICS_JSON = OUTPUT_DIR / "basic_model_metrics.json"
COEFFICIENTS_CSV = OUTPUT_DIR / "basic_model_coefficients.csv"
PLOT_PATH = OUTPUT_DIR / "basic_pred_vs_actual.png"

warnings.filterwarnings(
    "ignore",
    message="Found unknown categories in columns .* during transform. These unknown categories will be encoded as all zeros",
    category=UserWarning,
)


def make_onehot_first() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", drop="first", sparse=True)


def build_basic_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_onehot_first()),
                    ]
                ),
                categorical_cols,
            )
        )

    if not transformers:
        raise ValueError("No usable predictors were found for the basic linear model")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("linear", LinearRegression()),
        ]
    )


def markdown_table_from_frame(df: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values: list[str] = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data_path = find_cleaned_train_csv()
    info(f"Using cleaned data for basic linear model: {data_path.resolve()}")
    df = pd.read_csv(data_path)

    if "SalePrice" not in df.columns:
        raise ValueError("Input data must contain SalePrice")

    df = df.copy()
    df["SalePrice"] = pd.to_numeric(df["SalePrice"], errors="coerce")
    df = df.dropna(subset=["SalePrice"])

    excluded_cols = {"SalePrice", "logSalePrice", "Id"}
    feature_cols = [col for col in df.columns if col not in excluded_cols]
    if not feature_cols:
        raise ValueError("No explanatory variables available after exclusions")

    X = df[feature_cols].copy()
    y = df["SalePrice"].copy()

    numeric_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    model = build_basic_pipeline(numeric_cols, categorical_cols)

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=HOLDOUT_SIZE,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)
    y_hat_holdout = model.predict(X_holdout)

    holdout_rmse_dollar = rmse(y_holdout.to_numpy(), y_hat_holdout)
    holdout_mae_dollar = float(mean_absolute_error(y_holdout.to_numpy(), y_hat_holdout))

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = -cross_val_score(
        build_basic_pipeline(numeric_cols, categorical_cols),
        X,
        y,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=None,
    )
    cv_rmse_dollar_mean = float(np.mean(cv_scores))
    cv_rmse_dollar_std = float(np.std(cv_scores, ddof=1))

    linear = model.named_steps["linear"]
    intercept = float(linear.intercept_)
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()

    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": linear.coef_,
        }
    )
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df["sign"] = np.where(
        coef_df["coefficient"] > 0,
        "positive",
        np.where(coef_df["coefficient"] < 0, "negative", "zero"),
    )
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    coef_df.insert(0, "rank", np.arange(1, len(coef_df) + 1))
    coef_df.to_csv(COEFFICIENTS_CSV, index=False)

    top10 = coef_df.head(10).copy()
    equation_terms = []
    for _, row in top10.iterrows():
        sign = "+" if row["coefficient"] >= 0 else "-"
        equation_terms.append(f" {sign} {abs(float(row['coefficient'])):.3f} * {row['feature']}")
    equation_expanded = f"SalePrice_hat = {intercept:.3f}" + "".join(equation_terms)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_holdout, y_hat_holdout, alpha=0.72, s=34, edgecolor="white", linewidth=0.3)
    lower = float(min(y_holdout.min(), y_hat_holdout.min()))
    upper = float(max(y_holdout.max(), y_hat_holdout.max()))
    ax.plot([lower, upper], [lower, upper], "r--", linewidth=2, label="y = x")
    ax.set_xlabel("Actual SalePrice")
    ax.set_ylabel("Predicted SalePrice")
    ax.set_title("Basic Linear Model: Predicted vs Actual")
    ax.text(
        0.03,
        0.97,
        f"Holdout RMSE = {holdout_rmse_dollar:,.0f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#999"},
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=180)
    plt.close(fig)

    top_coeff_table = markdown_table_from_frame(
        top10[["rank", "feature", "coefficient"]],
        ["rank", "feature", "coefficient"],
    )

    summary_table = [
        "| Metric | Value |",
        "| --- | --- |",
        f"| holdout_rmse_dollar | {holdout_rmse_dollar:.2f} |",
        f"| holdout_mae_dollar | {holdout_mae_dollar:.2f} |",
        f"| cv_rmse_dollar_mean | {cv_rmse_dollar_mean:.2f} |",
        f"| cv_rmse_dollar_std | {cv_rmse_dollar_std:.2f} |",
        f"| number_of_predictors_after_encoding | {len(feature_names)} |",
    ]

    metrics = {
        "data_path": str(data_path.resolve()),
        "model_family": "basic_multiple_linear_regression",
        "target": "SalePrice",
        "formula": "y_i = beta_0 + beta_x X_i + beta_D D_i + U_i",
        "all_available_variables_used": True,
        "n_rows": int(df.shape[0]),
        "n_features_before_encoding": int(len(feature_cols)),
        "n_features_after_encoding": int(len(feature_names)),
        "holdout_rmse_dollar": holdout_rmse_dollar,
        "holdout_mae_dollar": holdout_mae_dollar,
        "cv_rmse_dollar_mean": cv_rmse_dollar_mean,
        "cv_rmse_dollar_std": cv_rmse_dollar_std,
        "cv_rmse_dollar_folds": cv_scores.tolist(),
        "intercept": intercept,
        "coefficients_csv": str(COEFFICIENTS_CSV),
    }
    METRICS_JSON.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    pipeline_code = """
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ]), numeric_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
    ]), categorical_cols),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("linear", LinearRegression()),
])
""".strip()

    report_md = "\n".join(
        [
            "# Basic Model Report",
            "",
            "**This basic model follows the course specification: use all available explanatory variables in a single multiple linear regression.**",
            "",
            "## Model Specification",
            "",
            "The housing group estimates:",
            "",
            "```text",
            "y_i = beta_0 + beta_X X_i + beta_D D_i + U_i",
            "```",
            "",
            "- Dependent variable: `SalePrice`.",
            "- `X_i`: matrix of numerical house characteristics.",
            "- `D_i`: matrix of dummy variables for categorical house characteristics.",
            "- Sample split: 80% in-sample / 20% out-of-sample.",
            f"- All available variables are used: **{len(feature_cols)}** raw predictors before encoding.",
            "",
            "## 1) Coding (Core Basic Pipeline)",
            "",
            "```python",
            pipeline_code,
            "```",
            "",
            "## 2) Regression Equation",
            "",
            "General form:",
            "- SalePrice_i = beta_0 + beta_X X_i + beta_D D_i + U_i",
            "",
            "Expanded with top 10 coefficients by absolute magnitude:",
            "",
            "```text",
            equation_expanded,
            "```",
            "",
            "## 3) Regression Result",
            "",
            f"- Intercept: **{intercept:.3f}**",
            f"- Predictors before encoding: **{len(feature_cols)}**",
            f"- Predictors after encoding: **{len(feature_names)}**",
            "- Full coefficient table: `outputs/basic_model_coefficients.csv`",
            "",
            "Top 10 coefficients by absolute magnitude:",
            "",
            *top_coeff_table,
            "",
            "### Coefficient interpretation",
            "- Numeric coefficients are in raw dollar units per one-unit increase in the corresponding numeric feature, holding the other included regressors fixed.",
            "- Dummy coefficients are measured relative to the omitted baseline category because one-hot encoding uses `drop='first'`.",
            "- These coefficients are descriptive associations in a high-dimensional linear model and should not be interpreted causally.",
            "",
            "## 4) Graph: Predicted vs Actual",
            "",
            "![Basic Linear Model Predicted vs Actual](basic_pred_vs_actual.png)",
            "",
            "## 5) RMSE",
            "",
            "Dollar-scale error formulas:",
            "- RMSE = sqrt((1/n) * sum((y_i - y_hat_i)^2))",
            "- MAE = (1/n) * sum(abs(y_i - y_hat_i))",
            "",
            "Summary metrics:",
            "",
            *summary_table,
            "",
            f"- 5-fold CV RMSE (dollar, folds): {', '.join(f'{float(v):.2f}' for v in cv_scores)}",
            "",
            "## Notes",
            "",
            "- This is the basic all-variables linear specification required before any feature selection or regularization improvement.",
            "- The Lasso report remains the improved model; this report is the full-variable baseline.",
        ]
    ) + "\n"

    REPORT_MD.write_text(report_md, encoding="utf-8")

    pandoc_available = shutil.which("pandoc") is not None
    pdf_generated = False
    if pandoc_available:
        try:
            subprocess.run(
                ["pandoc", str(REPORT_MD), "-o", str(REPORT_PDF)],
                check=True,
                cwd=OUTPUT_DIR,
            )
            pdf_generated = True
        except subprocess.CalledProcessError:
            pdf_generated = False

    report_html = simple_markdown_to_html(report_md, title="Basic Model Report")
    REPORT_HTML.write_text(report_html, encoding="utf-8")

    if not pdf_generated:
        write_fallback_pdf(report_md, REPORT_PDF, [PLOT_PATH])

    info(f"Wrote {COEFFICIENTS_CSV}")
    info(f"Wrote {METRICS_JSON}")
    info(f"Wrote {PLOT_PATH}")
    info(f"Wrote {REPORT_MD}")
    info(f"Wrote {REPORT_HTML}")
    info(f"Wrote {REPORT_PDF}")


if __name__ == "__main__":
    main()
