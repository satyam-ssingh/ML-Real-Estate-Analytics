# ==========================================================
# PROJECT: Machine Learning based Buyer Segmentation and
#          Investment Profiling for Real Estate Market Intelligence
# FILE: eda_analysis.py
# Purpose: Exploratory Data Analysis (EDA) for research paper
# ==========================================================

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("EDA: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

try:
    df = pd.read_csv("final_segmented_data.csv")
    print(f"Loaded final_segmented_data.csv -> {df.shape[0]} rows, {df.shape[1]} columns")
except Exception as e:
    print("Error loading final_segmented_data.csv. Run the pipeline script first.", e)
    sys.exit()

sns.set_style("whitegrid")

print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Numeric Summary ---")
print(df.describe())

# 1. Age Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['Age'], bins=20, kde=True, color='steelblue')
plt.title('Age Distribution of Buyers')
plt.xlabel('Age')
plt.ylabel('Count')
plt.savefig('eda_age_distribution.png')
plt.close()
print("Saved: eda_age_distribution.png")

# 2. Satisfaction Score Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['satisfaction_score'], bins=10, kde=True, color='seagreen')
plt.title('Satisfaction Score Distribution')
plt.xlabel('Satisfaction Score')
plt.ylabel('Count')
plt.savefig('eda_satisfaction_distribution.png')
plt.close()
print("Saved: eda_satisfaction_distribution.png")

# 3. Buyer count by Region
if 'region' in df.columns:
    plt.figure(figsize=(9, 5))
    region_counts = df['region'].value_counts()
    sns.barplot(x=region_counts.index, y=region_counts.values, color='coral')
    plt.title('Buyer Count by Region')
    plt.xlabel('Region')
    plt.ylabel('Number of Buyers')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('eda_buyers_by_region.png')
    plt.close()
    print("Saved: eda_buyers_by_region.png")

# 4. Acquisition Purpose breakdown
if 'acquisition_purpose' in df.columns:
    plt.figure(figsize=(7, 5))
    purpose_counts = df['acquisition_purpose'].value_counts()
    plt.pie(purpose_counts.values, labels=purpose_counts.index, autopct='%1.1f%%',
            colors=sns.color_palette('pastel'))
    plt.title('Acquisition Purpose Breakdown')
    plt.savefig('eda_acquisition_purpose.png')
    plt.close()
    print("Saved: eda_acquisition_purpose.png")

# 5. Loan Applied vs Client Type
if 'loan_applied' in df.columns and 'client_type' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='client_type', hue='loan_applied', palette='Set2')
    plt.title('Loan Applied Status by Client Type')
    plt.xlabel('Client Type')
    plt.ylabel('Count')
    plt.savefig('eda_loan_by_clienttype.png')
    plt.close()
    print("Saved: eda_loan_by_clienttype.png")

# 6. Sale Price distribution by Segment
if 'sale_price' in df.columns and 'Segment_Name' in df.columns:
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x='Segment_Name', y='sale_price', palette='Set3')
    plt.title('Sale Price Distribution by Buyer Segment')
    plt.xlabel('Segment')
    plt.ylabel('Sale Price')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig('eda_price_by_segment.png')
    plt.close()
    print("Saved: eda_price_by_segment.png")

# 7. Segment distribution (count of clients per segment)
if 'Segment_Name' in df.columns:
    plt.figure(figsize=(8, 5))
    segment_counts = df['Segment_Name'].value_counts()
    sns.barplot(x=segment_counts.index, y=segment_counts.values, palette='viridis')
    plt.title('Number of Clients per Buyer Segment')
    plt.xlabel('Segment')
    plt.ylabel('Client Count')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig('eda_segment_distribution.png')
    plt.close()
    print("Saved: eda_segment_distribution.png")

# 8. Correlation heatmap (numeric columns only)
numeric_df = df.select_dtypes(include=['int64', 'float64'])
if numeric_df.shape[1] > 1:
    plt.figure(figsize=(9, 7))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap (Numeric Features)')
    plt.tight_layout()
    plt.savefig('eda_correlation_heatmap.png')
    plt.close()
    print("Saved: eda_correlation_heatmap.png")

print("\n" + "=" * 60)
print("EDA COMPLETE - All charts saved in working directory")
print("=" * 60)