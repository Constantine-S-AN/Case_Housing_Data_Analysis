# Model Report

**Pandoc not found. HTML is generated, and Python fallback also exports PDF.**

## Model Choice

- Selected model: **lasso**
- Selection rationale: Selected lasso as it had the lowest CV RMSE among compared models

## Model Comparison

Performance comparison across different models:

| Model | CV RMSE (log) | Holdout RMSE (log) | Holdout RMSE ($) | Holdout MAE ($) |
| --- | --- | --- | --- | --- || gradient_boosting | 0.1916 | 0.2136 | 48561 | 26442 |
| lasso | 0.1439 | 0.1360 | 24000 | 15827 |
| random_forest | 0.1507 | 0.1501 | 29676 | 17807 |

## 1) Coding (Core Pipeline)

```python
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
```

## 2) Regression Equation

General form:
- y = log1p(SalePrice)
- y_hat = beta0 + sum(beta_j * x_j_tilde)

Expanded equation (Top10 coefficients only):

```text
y_hat = 11.911357 - 0.868460 * cat__RoofMatl_ClyTile - 0.165819 * cat__Condition2_PosN + 0.107589 * num__GrLivArea + 0.098500 * cat__Neighborhood_StoneBr + 0.093396 * cat__Neighborhood_Crawfor + 0.087284 * cat__Exterior1st_BrkFace + 0.080970 * num__OverallQual + 0.068053 * cat__Neighborhood_NridgHt + 0.058659 * cat__Functional_Typ - 0.054916 * cat__MSZoning_RM
```

Top10 absolute coefficients:

| Rank | Feature | Coefficient |
| --- | --- | --- |
| 1 | cat__RoofMatl_ClyTile | -0.868460 |
| 2 | cat__Condition2_PosN | -0.165819 |
| 3 | num__GrLivArea | 0.107589 |
| 4 | cat__Neighborhood_StoneBr | 0.098500 |
| 5 | cat__Neighborhood_Crawfor | 0.093396 |
| 6 | cat__Exterior1st_BrkFace | 0.087284 |
| 7 | num__OverallQual | 0.080970 |
| 8 | cat__Neighborhood_NridgHt | 0.068053 |
| 9 | cat__Functional_Typ | 0.058659 |
| 10 | cat__MSZoning_RM | -0.054916 |

## 3) Model Result

- Model type: **lasso**
- Best alpha: **0.00078140**
- Non-zero features: **87**
- Top coefficients/importances file: `outputs/top_coefficients.csv`
- CV protocol: `KFold(n_splits=5, shuffle=True, random_state=42)` with `neg_root_mean_squared_error` on log1p scale.

Top positive coefficients:

| Feature | Coefficient |
| --- | --- |
| num__GrLivArea | 0.107589 |
| cat__Neighborhood_StoneBr | 0.098500 |
| cat__Neighborhood_Crawfor | 0.093396 |
| cat__Exterior1st_BrkFace | 0.087284 |
| num__OverallQual | 0.080970 |
| cat__Neighborhood_NridgHt | 0.068053 |
| cat__Functional_Typ | 0.058659 |
| cat__Condition1_Norm | 0.048450 |
| cat__BsmtQual_Ex | 0.045596 |
| cat__Neighborhood_Somerst | 0.043984 |

Top negative coefficients:

| Feature | Coefficient |
| --- | --- |
| cat__RoofMatl_ClyTile | -0.868460 |
| cat__Condition2_PosN | -0.165819 |
| cat__MSZoning_RM | -0.054916 |
| cat__CentralAir_N | -0.047431 |
| cat__SaleCondition_Abnorml | -0.046101 |
| cat__Neighborhood_Edwards | -0.042191 |
| cat__LandContour_Bnk | -0.039253 |
| cat__BldgType_Twnhs | -0.033140 |
| cat__BsmtFinType1_Unf | -0.032548 |
| cat__BsmtCond_Fa | -0.026081 |

### Coefficient interpretation
- Numeric features use StandardScaler, so each numeric coefficient means expected change in log1p(SalePrice) for a +1 standard deviation change.
- Categorical features are one-hot encoded, so each category coefficient is interpreted relative to the omitted baseline category.
- Rare-category coefficients can be unstable; treat them as predictive signals rather than causal effects.

## 4) Graph: Predicted vs Actual

![Holdout Predicted vs Actual](pred_vs_actual.png)

## 5) RMSE

RMSE (log1p scale) formula:
- RMSE_log = sqrt((1/n) * sum((y_i - y_hat_i)^2)), where y = log1p(SalePrice).
- This log-scale error emphasizes multiplicative/relative discrepancy.

Back-transform and dollar-scale metrics:
- SalePrice_hat = exp(y_hat) - 1
- SalePrice_true = exp(y_true) - 1
- typical_relative_error = exp(holdout_rmse_log) - 1

| Metric | Value |
| --- | --- |
| holdout_rmse_log | 0.135977 |
| cv_rmse_log_mean | 0.143861 |
| holdout_rmse_dollar | 23999.75 |
| holdout_mae_dollar | 15826.82 |
| typical_relative_error = exp(holdout_rmse_log)-1 | 0.145656 (14.57%) |

- 5-fold CV RMSE (folds): 0.131197, 0.188898, 0.111487

### Baseline vs Filtered (Sensitivity Analysis)
- Filtered sensitivity run was unavailable in this execution.
- This is a sensitivity analysis for influential points, not arbitrary deletion of data.
- A better filtered score indicates influence sensitivity; otherwise baseline is already robust.

## Appendix: Outlier Diagnostics

Note: Outlier filtering was not applied in this model comparison run.

Outlier impact figure:
- See `../graph/09_outlier_impact_rmse.png` for RMSE before and after removing high-influence points.
- The chart is a sensitivity check: a noticeable RMSE drop indicates a small set of influential points drives error disproportionately.
- If the change is small, model performance is relatively robust to those candidate outliers.

## Appendix: Model Comparison Visualizations

Feature importance comparison:
- See `../graph/12_lasso_coefficients.png` for top Lasso coefficients by absolute magnitude.
- See `../graph/13_random_forest_importances.png` for top Random Forest feature importances.
- See `../graph/14_gradient_boosting_importances.png` for top Gradient Boosting feature importances.
- See `../graph/15_model_comparison_rmse.png` for RMSE comparison across all models.
- See `../graph/16_residual_distributions.png` for residual distribution comparison across models.
