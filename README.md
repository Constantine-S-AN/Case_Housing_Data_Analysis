## Case Housing — Data Cleaning

This repo contains a simple data-cleaning script for the Ames/Case Housing dataset.

### Files
- **`train.csv`**: original training data
- **`train_cleaned.csv`**: cleaned output produced by the script
- **`clean_data.py`**: cleaning script
- **`data_description.txt`**: feature descriptions

### How to run

```bash
python clean_data.py
```

This will write `train_cleaned.csv` in the same folder.

### Graphs and Analysis (from `graph/`)
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
