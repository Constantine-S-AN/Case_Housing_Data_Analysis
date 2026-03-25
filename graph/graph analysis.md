## Graphs and Analysis (from `graph/`)
Below are the plots generated from `train_cleaned_v2.csv` and the key insights for each.

1) **SalePrice Distribution** (`graph/01_saleprice_distribution.png`)
   - The raw SalePrice is strongly right-skewed, indicating a long tail of expensive homes.
   - The log-transformed distribution is closer to symmetric, suggesting log(SalePrice) is more suitable for linear modeling.

2) **GrLivArea vs SalePrice** (`graph/02_grlivarea_vs_saleprice.png`)
   - Larger above-ground living area generally increases SalePrice with a clear positive trend.
   - A few large homes are priced unexpectedly low, which may be outliers or data quality issues worth checking.

3) **Neighborhood Price Spread** (`graph/03_neighborhood_boxplot.png`)
   - Neighborhood strongly differentiates price; medians vary widely across areas.
   - Price dispersion within neighborhoods suggests location alone is not sufficient; size/quality still matter.

4) **Top Numeric Correlations** (`graph/04_top_correlations.png`)
   - OverallQual, GrLivArea, GarageCars, and TotalBsmtSF are among the strongest numeric predictors.
   - This helps prioritize features for modeling and feature engineering.

5) **YearBuilt vs Median SalePrice** (`graph/05_yearbuilt_median_trend.png`)
   - Newer homes tend to have higher median prices, with a noticeable upward trend after the 1980s.
   - The pattern supports including age or build era in any pricing analysis.

6) **Missing Values by Column** (`graph/06_missing_values.png`)
   - Missingness is concentrated in features like PoolQC, Alley, Fence, and MiscFeature.
   - These are likely “absence-as-missing” features; treating missing as “None” can be informative.

7) **Outlier Candidates (Labeled)** (`graph/07_outlier_candidates_labeled.png`)
   - Insight: A small set of homes has unusually high `GrLivArea` but unexpectedly low `log1p(SalePrice)` relative to the fitted trend.
   - Decision: Keep these points in a flagged-review set first, then compare model behavior with and without them.
   - Validation: Cross-check flagged IDs against raw records and confirm overlap with the high-influence points from Cook's distance.

8) **Cook's Distance** (`graph/08_cooks_distance.png`)
   - Insight: `78` observations exceed the standard influence threshold `4/n` (`~0.00274`), so influential cases are material for linear fits.
   - Decision: Use Cook's D screening as the default outlier/influence gate for interpretable linear-model experiments.
   - Validation: Refit after filtering and confirm lower residual spread plus more stable coefficient signs/magnitudes.

9) **Outlier Impact on RMSE** (`graph/09_outlier_impact_rmse.png`)
   - Insight: Removing influential points reduced RMSE on `log1p(SalePrice)` from `0.1597 -> 0.1196` (holdout) and `0.1667 -> 0.1169` (KFold CV).
   - Decision: Treat this as an OLS-style sensitivity diagnostic, not the final model scorecard. Keep both full-data and filtered variants visible so the final Lasso choice is evidence-based rather than hidden.
   - Validation: Promote the filtered version only when both holdout RMSE and CV RMSE improve together in the final Lasso pipeline as well.

10) **Engineered TotalSF vs Price** (`graph/10_totalsf_vs_logprice.png`)
   - Insight: `TotalSF = GrLivArea + TotalBsmtSF` shows a strong positive association with `log1p(SalePrice)` (corr `~0.773`).
   - Decision: Treat `TotalSF` as a candidate engineered predictor worth testing, not an automatic inclusion.
   - Validation: Retain `TotalSF` only if the final Lasso cross-validation actually improves; otherwise keep the original variables for a cleaner and more defensible specification.

11) **HouseAge & RemodAge Trends** (`graph/11_houseage_remodage_trends.png`)
   - Insight: Binned median `log1p(SalePrice)` declines with both `HouseAge` and `RemodAge`, consistent with aging/depreciation effects.
   - Decision: Use these age variables as motivated candidates for feature engineering, then keep them only if they add validation value beyond the original housing attributes already in the model.
   - Validation: Compare the engineered-feature variant against the original Lasso feature set and only retain the extra variables when cross-validated RMSE improves materially.

12) **Lasso Coefficients (Top 15)** (`graph/12_lasso_coefficients.png`)
   - Insight: Largest-magnitude coefficients combine structural and location effects (for example `GrLivArea`, `OverallQual`, and neighborhood indicators), with clear positive/negative signs.
   - Decision: Use signed top Lasso coefficients as the interpretable feature story for the final presentation.
   - Validation: Re-run LassoCV on repeated folds/seeds and keep only features whose sign/rank remains stable.
