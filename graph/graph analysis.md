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

13) **Random Forest Feature Importances (Top 15)** (`graph/13_random_forest_importances.png`)
   - Insight: Tree-based importance ranks reinforce some overlap with Lasso (e.g., `OverallQual`, `GrLivArea`) but also highlight different non-linear interactions and complex spreads.
   - Decision: Recognize Random Forest is less sparse, and use this chart to identify candidates where nonlinearity could matter even if Lasso shrinks them.
   - Validation: Cross-compare the top features with the Lasso non-zero list; most consistent predictors support primary model confidence, while divergent predictors get explored in ablation runs.

14) **Gradient Boosting Feature Importances (Top 15)** (`graph/14_gradient_boosting_importances.png`)
   - Insight: Gradient Boosting importance tends to highlight stronger non-linear-chain features and occasional deep splits; a smaller effective predictor set may indicate terminal performance differences in dataset complexity.
   - Decision: Evaluate where GB’s highly ranked features differ from Lasso, then test whether including those as engineered interactions improves linear model robustness.
   - Validation: Use this with `plot_15` and `plot_16` to validate whether GB rank differences actually correspond to predictive gain or overfit noise.

15) **Model Comparison RMSE (CV vs Holdout)** (`graph/15_model_comparison_rmse.png`)
   - Insight: Lasso obtains the lowest CV RMSE and holdout RMSE, confirming the best bias-variance tradeoff on this dataset versus RF/GB under current hyperparameters.
   - Decision: Prefer Lasso as main candidate in final report but still keep RF/GB as a sensitivity check for model risk and nonlinearity.
   - Validation: Re-run with alternate hyperparameters and >5 CV folds to ensure the ranking is stable; if RF/GB beats Lasso then update model choice and narrative.

16) **Residual Distributions (Lasso, RF, GB)** (`graph/16_residual_distributions.png`)
   - Insight: Lasso residuals are more tightly centered with fewer extreme tails, while RF/GB show heavier tails/variance from overfitting small clusters.
   - Decision: Use Lasso resid distribution as evidence for robust and interpretable errors; include tree-based residuals in appendix for risk assessment.
   - Validation: Quantify with distribution metrics (skew, kurtosis, quantile spread), then use a threshold-based deployment plan that favors model stability and interpretability.

17) **Random Forest Predictions (Predicted vs Actual)** (`graph/17_random_forest_predictions.png`)
   - Insight: Random Forest model achieves RMSE=0.1496, MAE=0.1007, R²=0.8800 on the holdout test set.
   - Pattern: Predictions cluster tightly around the perfect-prediction diagonal (red dashed line), with some scatter above/below indicating unavoidable variance in individual house valuations.
   - Decision: RF performs well (R²=0.88) but with slightly higher error than Lasso, suggesting some overfitting or sensitivity to specific neighborhoods and features.
   - Validation: This visualization supports the conclusion that Lasso's sparsity provides better generalization despite RF's nonlinear flexibility on this dataset.

18) **Gradient Boosting Predictions (Predicted vs Actual)** (`graph/18_gradient_boosting_predictions.png`)
   - Insight: Gradient Boosting model achieves RMSE=0.1829, MAE=0.1226, R²=0.8208 on the holdout test set.
   - Pattern: Predictions show wider scatter from the diagonal than RF, with more frequent under-predictions for high-priced homes (upper right), indicating GB's constraint hyperparameters may be too conservative (small n_estimators, shallow trees).
   - Decision: GB underperforms both Lasso and RF on this dataset---likely because the shallow trees and limited boosting iterations restrict learning capacity for complex price interactions.
   - Validation: Results validate Lasso as the best bias-variance choice; tree-based models suggest feature interactions exist but are not captured well without deeper trees (which would increase overfit risk).
