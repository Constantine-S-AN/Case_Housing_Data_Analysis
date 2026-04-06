# Model Report


## Model Choice

- Selected model: **lasso**
- Lasso is a linear regression model with L1 regularization that helps reduce overfitting by shrinking less useful feature coefficients to zero.
- Selection rationale: Selected Lasso as it had the best validation performance and the cleanest balance of accuracy, sparsity, and interpretability.

## Model Journey

- Started with a full OLS benchmark to establish a baseline and understand the general signal in the data.
- Moved to Lasso because the dataset has many predictors and likely noisy categories, so regularization and feature selection were needed.
- Tested Random Forest and Gradient Boosting as nonlinear alternatives to check whether tree-based models could improve predictions.
- Selected Lasso because it had the best validation performance and produced a simpler, more stable final model.

The starting OLS benchmark recorded CV RMSE 0.1667 and holdout RMSE 0.1597, so Lasso improved the core linear fit before even comparing the nonlinear alternatives.

## Model Comparison

Performance comparison across different models:

| Model | CV RMSE | Holdout RMSE | Holdout RMSE ($) | Holdout MAE ($) |
| --- | --- | --- | --- | --- |
| OLS benchmark | 0.1667 | 0.1597 | 30,098 | 19,538 |
| Lasso | 0.1439 | 0.1360 | 24,000 | 15,827 |
| Random Forest | 0.1507 | 0.1501 | 29,676 | 17,807 |
| Gradient Boosting | 0.1916 | 0.2136 | 48,561 | 26,442 |

Lasso had the lowest CV RMSE and the lowest holdout RMSE and MAE. Random Forest was the second-best comparison model, while Gradient Boosting performed worst under the tested setup. The initial OLS benchmark was the starting point, and Lasso improved on it by shrinking noisy coefficients and selecting a smaller active set; the final Lasso retained **87 non-zero features**. This reduced the effective model from the full encoded feature space to a smaller active set, improving interpretability and helping control overfitting. Relative to the OLS benchmark, Lasso reduced holdout RMSE from 0.1597 to 0.1360 and dollar RMSE from about $30.1k to $24.0k.

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

Predicting log1p(SalePrice) helps with the strong right-skew in house prices by compressing large values and making model errors more stable.

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
- The most intuitive recurring predictors are `GrLivArea`, `OverallQual`, key `Neighborhood` indicators, `CentralAir_N`, and basement/quality signals.
- Categorical features are one-hot encoded, so each category coefficient is interpreted relative to the omitted baseline category.
- Rare-category coefficients can be unstable; treat them as predictive signals rather than causal effects.

## 4) Graph: Predicted vs Actual

![Holdout Predicted vs Actual](pred_vs_actual.png)

## 5) RMSE

RMSE (log1p scale) formula:
- RMSE_log = sqrt((1/n) * sum((y_i - y_hat_i)^2)), where y = log1p(SalePrice).
- Predicting log1p(SalePrice) helps handle the strong right-skew in house prices by stabilizing large values.
- This log-scale error emphasizes multiplicative / relative discrepancy.

Back-transform and dollar-scale metrics:
- SalePrice_hat = exp(y_hat) - 1
- SalePrice_true = exp(y_true) - 1
- typical_relative_error = exp(holdout_rmse_log) - 1

| Metric | Value |
| --- | --- |
| holdout_rmse_log | 0.135977 |
| cv_rmse_log_mean | 0.143861 |
| holdout_rmse_dollar | 23,999.75 |
| holdout_mae_dollar | 15,826.82 |
| typical_relative_error = exp(holdout_rmse_log)-1 | 0.145656 (14.57%) |

- A holdout RMSE of about 0.136 on the log scale corresponds to roughly 14.6% typical relative prediction error.

### Outlier Sensitivity
- Outlier influence was checked, but the main results are reported on the full consistent dataset.

## Why Lasso Won
- Best CV RMSE
- Best holdout RMSE
- Best holdout MAE
- Cleaner residual distribution under the tested setup
- More interpretable and sparser model, with only 87 non-zero features

Lasso gave the best balance of predictive accuracy, stability, and interpretability for this housing dataset, therefore it was selected as the final model over the OLS benchmark, Random Forest, and Gradient Boosting.

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
