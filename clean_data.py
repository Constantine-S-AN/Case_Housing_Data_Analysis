import pandas as pd
import numpy as np

# Load the data
print("Loading data...")
df = pd.read_csv('train.csv')

print(f"Original data shape: {df.shape}")
print(f"Original missing values:\n{df.isnull().sum().sum()} total missing values")

# Replace 'NA' strings with actual NaN (pandas will recognize this)
# Some columns use 'NA' to mean "not applicable" which should be NaN
df = df.replace('NA', np.nan)

# Check data types and convert where necessary
print("\nChecking data types...")

# Convert numeric columns that might be strings
numeric_columns = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt', 'BsmtFinSF1', 
                   'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 
                   'BsmtHalfBath', 'GarageCars', 'GarageArea', 'SalePrice']

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Handle missing values based on data description
print("\nHandling missing values...")

# LotFrontage: Linear feet of street - can be imputed with median by neighborhood
if 'LotFrontage' in df.columns:
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )
    # If still missing, fill with overall median
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())

# MasVnrType and MasVnrArea: Masonry veneer - if type is missing, area should be 0
if 'MasVnrType' in df.columns:
    df['MasVnrType'] = df['MasVnrType'].fillna('None')
if 'MasVnrArea' in df.columns:
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

# Basement-related columns: If no basement, these should be filled appropriately
basement_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                 'BsmtFinType2', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 
                 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']

for col in basement_cols:
    if col in df.columns:
        if col in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']:
            df[col] = df[col].fillna('None')
        else:  # numeric columns
            df[col] = df[col].fillna(0)

# Garage-related columns: If no garage, fill appropriately
garage_cols = ['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 
               'GarageArea', 'GarageQual', 'GarageCond']

for col in garage_cols:
    if col in df.columns:
        if col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
            df[col] = df[col].fillna('None')
        elif col == 'GarageYrBlt':
            # Fill with year built if garage exists but year is missing
            df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
        else:  # numeric columns
            df[col] = df[col].fillna(0)

# Fireplace: If no fireplace, quality should be None
if 'FireplaceQu' in df.columns:
    df['FireplaceQu'] = df['FireplaceQu'].fillna('None')

# Pool: If no pool, quality should be None
if 'PoolQC' in df.columns:
    df['PoolQC'] = df['PoolQC'].fillna('None')

# Fence: If no fence, should be None
if 'Fence' in df.columns:
    df['Fence'] = df['Fence'].fillna('None')

# MiscFeature: If no misc feature, should be None
if 'MiscFeature' in df.columns:
    df['MiscFeature'] = df['MiscFeature'].fillna('None')

# Alley: If no alley access, should be None
if 'Alley' in df.columns:
    df['Alley'] = df['Alley'].fillna('None')

# Electrical: Fill with most common value
if 'Electrical' in df.columns:
    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

# Utilities: Fill with most common value (likely AllPub)
if 'Utilities' in df.columns:
    df['Utilities'] = df['Utilities'].fillna(df['Utilities'].mode()[0])

# Exterior1st and Exterior2nd: Fill with most common value
if 'Exterior1st' in df.columns:
    df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])
if 'Exterior2nd' in df.columns:
    df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])

# KitchenQual: Fill with most common value
if 'KitchenQual' in df.columns:
    df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])

# Functional: Fill with most common value (likely Typ)
if 'Functional' in df.columns:
    df['Functional'] = df['Functional'].fillna(df['Functional'].mode()[0])

# SaleType: Fill with most common value
if 'SaleType' in df.columns:
    df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])

# MSZoning: Fill with most common value
if 'MSZoning' in df.columns:
    df['MSZoning'] = df['MSZoning'].fillna(df['MSZoning'].mode()[0])

# Check for any remaining missing values
remaining_missing = df.isnull().sum()
if remaining_missing.sum() > 0:
    print("\nRemaining missing values:")
    print(remaining_missing[remaining_missing > 0])
    # Fill any remaining numeric columns with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    # Fill any remaining categorical columns with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')

# Check for duplicates
print(f"\nDuplicate rows: {df.duplicated().sum()}")

# Check for outliers in key numeric columns (optional - just report)
print("\nChecking for potential outliers...")
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if col != 'Id':  # Skip ID column
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
        if len(outliers) > 0:
            print(f"{col}: {len(outliers)} potential outliers")

# Ensure SalePrice is numeric and has no missing values (it's the target)
if 'SalePrice' in df.columns:
    df['SalePrice'] = pd.to_numeric(df['SalePrice'], errors='coerce')
    if df['SalePrice'].isnull().sum() > 0:
        print(f"Warning: {df['SalePrice'].isnull().sum()} missing values in SalePrice")

# Final check
print(f"\nFinal data shape: {df.shape}")
print(f"Final missing values: {df.isnull().sum().sum()}")

# Save cleaned data
output_file = 'train_cleaned.csv'
df.to_csv(output_file, index=False)
print(f"\nCleaned data saved to {output_file}")

# Display summary statistics
print("\nSummary statistics:")
print(df.describe())
