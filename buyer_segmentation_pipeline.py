# ==========================================================
# PROJECT: Machine Learning based Buyer Segmentation and
#          Investment Profiling for Real Estate Market Intelligence
# FILE: buyer_segmentation_pipeline.py
# ==========================================================

import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

print("=" * 60)
print("STEP 1: DATA LOADING & CLEANING")
print("=" * 60)

try:
    clients_df = pd.read_csv('clients.csv')
    properties_df = pd.read_csv('properties.csv')
    print("Files loaded successfully!")
except Exception as e:
    print("Critical Error loading files:", e)
    sys.exit()

clients_df.columns = clients_df.columns.str.strip()
properties_df.columns = properties_df.columns.str.strip()
clients_df['date_of_birth'] = clients_df['date_of_birth'].astype(str).str.strip()

categorical_text_cols_clients = ['client_type', 'gender', 'country', 'region',
                                  'acquisition_purpose', 'referral_channel']
for col in categorical_text_cols_clients:
    if col in clients_df.columns:
        clients_df[col] = clients_df[col].astype(str).str.strip().str.title()

if 'listing_status' in properties_df.columns:
    properties_df['listing_status'] = properties_df['listing_status'].astype(str).str.strip().str.title()

clients_df = clients_df.drop_duplicates(subset=['client_id'], keep='first')

sold_properties = properties_df[properties_df['listing_status'] == 'Sold'].copy()

try:
    merged_df = pd.merge(
    clients_df,
    sold_properties,
    left_on="client_id",
    right_on="client_ref",
    how="inner",
    suffixes=('_client', '_prop')  
)
    print(f"Merge successful. Rows: {merged_df.shape[0]}")
except Exception as e:
    print("Merge failed:", e)
    sys.exit()

raw_dob = merged_df['date_of_birth'].replace(['nan', 'None', ''], np.nan)
parsed_dates = pd.to_datetime(raw_dob, errors='coerce', dayfirst=False)

still_missing = parsed_dates.isnull() & raw_dob.notnull()
if still_missing.any():
    retry_dates = pd.to_datetime(raw_dob[still_missing], errors='coerce', dayfirst=True)
    parsed_dates.loc[still_missing] = retry_dates

still_missing = parsed_dates.isnull() & raw_dob.notnull()
if still_missing.any():
    common_formats = ['%d-%m-%Y', '%m-%d-%Y', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d', '%d %b %Y', '%b %d, %Y']
    for fmt in common_formats:
        still_missing = parsed_dates.isnull() & raw_dob.notnull()
        if not still_missing.any():
            break
        attempt = pd.to_datetime(raw_dob[still_missing], format=fmt, errors='coerce')
        parsed_dates.loc[still_missing] = attempt

merged_df['date_of_birth'] = parsed_dates

current_year = pd.Timestamp.now().year
merged_df['Age'] = current_year - merged_df['date_of_birth'].dt.year

if merged_df['Age'].isnull().any():
    median_age = merged_df['Age'].median()
    merged_df['Age'] = merged_df['Age'].fillna(median_age)
    print(f"Missing ages filled with median: {median_age}")

merged_df['Age'] = merged_df['Age'].astype(int)
merged_df['date_of_birth'] = merged_df['date_of_birth'].dt.strftime('%Y-%m-%d')
merged_df['date_of_birth'] = merged_df['date_of_birth'].fillna('Unknown')

# ----------------------------------------------------------
# Clean sale_price column - remove $ and , then convert to numeric
# ----------------------------------------------------------
if 'sale_price' in merged_df.columns:
    merged_df['sale_price'] = (
        merged_df['sale_price']
        .astype(str)
        .str.replace('$', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.strip()
    )
    merged_df['sale_price'] = pd.to_numeric(merged_df['sale_price'], errors='coerce')

    if merged_df['sale_price'].isnull().any():
        median_price = merged_df['sale_price'].median()
        merged_df['sale_price'] = merged_df['sale_price'].fillna(median_price)
        print(f"Missing sale_price filled with median: {median_price}")

# ----------------------------------------------------------
# Clean floor_area_sqft column too, in case it also has units/commas
# ----------------------------------------------------------
if 'floor_area_sqft' in merged_df.columns:
    merged_df['floor_area_sqft'] = (
        merged_df['floor_area_sqft']
        .astype(str)
        .str.replace(',', '', regex=False)
        .str.replace('sqft', '', case=False, regex=False)
        .str.replace('sq.ft', '', case=False, regex=False)
        .str.strip()
    )
    merged_df['floor_area_sqft'] = pd.to_numeric(merged_df['floor_area_sqft'], errors='coerce')

    if merged_df['floor_area_sqft'].isnull().any():
        median_area = merged_df['floor_area_sqft'].median()
        merged_df['floor_area_sqft'] = merged_df['floor_area_sqft'].fillna(median_area)
        print(f"Missing floor_area_sqft filled with median: {median_area}")

numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if merged_df[col].isnull().any():
        merged_df[col] = merged_df[col].fillna(merged_df[col].median())

categorical_cols = merged_df.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    if merged_df[col].isnull().any():
        mode_val = merged_df[col].mode()
        fill_val = mode_val[0] if not mode_val.empty else "Unknown"
        merged_df[col] = merged_df[col].fillna(fill_val)

print("Step 1 complete. Cleaned data shape:", merged_df.shape)
merged_df.to_csv("cleaned_merged_data.csv", index=False)
print("Saved checkpoint file: cleaned_merged_data.csv")


print("\n" + "=" * 60)
print("STEP 2: FEATURE ENCODING")
print("=" * 60)

ml_df = merged_df.copy()
label_enc_cols = ['client_type', 'loan_applied']
label_encoders = {}

for col in label_enc_cols:
    if col in ml_df.columns:
        le = LabelEncoder()
        ml_df[col + '_encoded'] = le.fit_transform(ml_df[col].astype(str))
        label_encoders[col] = le

one_hot_cols = ['region', 'acquisition_purpose', 'referral_channel', 'country']
one_hot_cols_present = [c for c in one_hot_cols if c in ml_df.columns]
ml_df = pd.get_dummies(ml_df, columns=one_hot_cols_present, prefix=one_hot_cols_present)

print("Step 2 complete. Encoded data shape:", ml_df.shape)


print("\n" + "=" * 60)
print("STEP 3: FEATURE SCALING")
print("=" * 60)

scale_cols = ['Age', 'satisfaction_score']
scale_cols_present = [c for c in scale_cols if c in ml_df.columns]
scaler = StandardScaler()
ml_df[scale_cols_present] = scaler.fit_transform(ml_df[scale_cols_present])

print("Step 3 complete. Scaled columns:", scale_cols_present)


print("\n" + "=" * 60)
print("STEP 4 & 5: CLUSTERING MODEL + OPTIMAL CLUSTER SELECTION")
print("=" * 60)

feature_cols = scale_cols_present + [c + '_encoded' for c in label_enc_cols if c in ml_df.columns] + \
               [c for c in ml_df.columns if any(c.startswith(prefix + '_') for prefix in one_hot_cols_present)]

X = ml_df[feature_cols].copy()
X = X.fillna(0)

inertia_values = []
k_range = range(2, 11)

for k in k_range:
    kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_test.fit(X)
    inertia_values.append(kmeans_test.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(list(k_range), inertia_values, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.savefig('elbow_method.png')
plt.close()
print("Elbow method chart saved as elbow_method.png")

silhouette_scores = []
for k in k_range:
    kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans_test.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)
    print(f"K={k} -> Silhouette Score: {score:.4f}")

best_k_silhouette = list(k_range)[np.argmax(silhouette_scores)]
print(f"\nBest K based on Silhouette Score: {best_k_silhouette}")

best_k = 4
print(f"Using forced K = {best_k} (to match PRD's 4 buyer segments: C1-C4)")

kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
ml_df['Cluster_KMeans'] = kmeans_final.fit_predict(X)

hierarchical = AgglomerativeClustering(n_clusters=best_k)
ml_df['Cluster_Hierarchical'] = hierarchical.fit_predict(X)

print("Step 4 & 5 complete. Clusters assigned using K-Means and Hierarchical Clustering.")


print("\n" + "=" * 60)
print("STEP 6: CLUSTER INTERPRETATION")
print("=" * 60)

merged_df['Cluster_KMeans'] = ml_df['Cluster_KMeans']
merged_df['Cluster_Hierarchical'] = ml_df['Cluster_Hierarchical']

cluster_summary = merged_df.groupby('Cluster_KMeans').agg(
    Avg_Age=('Age', 'mean'),
    Avg_Satisfaction=('satisfaction_score', 'mean'),
    Total_Clients=('client_id', 'count'),
    Loan_Applied_Rate=('loan_applied', lambda x: (x.astype(str).str.lower() == 'yes').mean())
).reset_index()

print("\nCluster Summary:")
print(cluster_summary)

merged_df.to_csv("final_segmented_data.csv", index=False)
cluster_summary.to_csv("cluster_summary.csv", index=False)


print("\n" + "=" * 60)
print("STEP 6B: AUTOMATIC CLUSTER NAMING (Logical Condition-Based)")
print("=" * 60)

naming_profile = merged_df.groupby('Cluster_KMeans').agg(
    Avg_Age=('Age', 'mean'),
    Avg_Satisfaction=('satisfaction_score', 'mean'),
    Loan_Applied_Rate=('loan_applied', lambda x: (x.astype(str).str.lower() == 'yes').mean()),
    Corporate_Rate=('client_type', lambda x: (x.astype(str).str.lower() == 'corporate').mean()),
    Investment_Rate=('acquisition_purpose', lambda x: (x.astype(str).str.lower() == 'investment').mean()),
    Avg_Sale_Price=('sale_price', 'mean') if 'sale_price' in merged_df.columns else ('Age', 'mean'),
    Total_Clients=('client_id', 'count')
).reset_index()

naming_profile['Segment_Name'] = "Unassigned"

n_clusters_total = len(naming_profile)
assigned = [False] * n_clusters_total

avg_age_all = naming_profile['Avg_Age'].mean()
avg_satisfaction_all = naming_profile['Avg_Satisfaction'].mean()
avg_sale_price_all = naming_profile['Avg_Sale_Price'].mean()
avg_loan_all = naming_profile['Loan_Applied_Rate'].mean()
avg_investment_all = naming_profile['Investment_Rate'].mean()

# 1. Corporate Buyers: highest Corporate_Rate (even if 0, pick the relative max)
remaining = [i for i in range(n_clusters_total) if not assigned[i]]
idx = naming_profile.loc[remaining, 'Corporate_Rate'].idxmax()
naming_profile.loc[idx, 'Segment_Name'] = "Corporate Buyers"
assigned[idx] = True

# 2. Luxury Investors: among remaining, pick the one with highest satisfaction,
#    preferring genuine above-average candidates first
remaining = [i for i in range(n_clusters_total) if not assigned[i]]
candidates = [i for i in remaining if naming_profile.loc[i, 'Avg_Satisfaction'] >= avg_satisfaction_all]
if not candidates:
    candidates = remaining
idx = naming_profile.loc[candidates, 'Avg_Satisfaction'].idxmax()
naming_profile.loc[idx, 'Segment_Name'] = "Luxury Investors"
assigned[idx] = True

# 3. First-Time Buyers: MUST be below-average age (hard requirement, no exceptions
#    unless truly no cluster qualifies)
remaining = [i for i in range(n_clusters_total) if not assigned[i]]
candidates = [i for i in remaining if naming_profile.loc[i, 'Avg_Age'] < avg_age_all]
if candidates:
    idx = naming_profile.loc[candidates, 'Loan_Applied_Rate'].idxmax()
else:
    idx = naming_profile.loc[remaining, 'Avg_Age'].idxmin()
naming_profile.loc[idx, 'Segment_Name'] = "First-Time Buyers"
assigned[idx] = True

# 4. Global Investors: whatever cluster is left
remaining = [i for i in range(n_clusters_total) if not assigned[i]]
idx = remaining[0]
naming_profile.loc[idx, 'Segment_Name'] = "Global Investors"
assigned[idx] = True

print("\nCluster Naming Result:")
print(naming_profile[['Cluster_KMeans', 'Segment_Name', 'Avg_Age', 'Avg_Satisfaction',
                       'Loan_Applied_Rate', 'Corporate_Rate', 'Investment_Rate', 'Avg_Sale_Price']])

segment_map = dict(zip(naming_profile['Cluster_KMeans'], naming_profile['Segment_Name']))
merged_df['Segment_Name'] = merged_df['Cluster_KMeans'].map(segment_map)

merged_df.to_csv("final_segmented_data.csv", index=False)
naming_profile.to_csv("cluster_naming_profile.csv", index=False)

print("\nFinal files saved:")
print("1. final_segmented_data.csv     (includes Segment_Name column)")
print("2. cluster_naming_profile.csv   (naming logic breakdown)")
print("3. cluster_summary.csv          (cluster statistics)")
print("4. elbow_method.png             (elbow chart)")

print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)


# ==========================================================
# STEP 7: ADVANCED CLUSTER EVALUATION & VISUALIZATION
# ==========================================================
print("\n" + "=" * 60)
print("STEP 7: ADVANCED CLUSTER EVALUATION & VISUALIZATION")
print("=" * 60)

from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

try:
    db_index = davies_bouldin_score(X, ml_df['Cluster_KMeans'])
    ch_score = calinski_harabasz_score(X, ml_df['Cluster_KMeans'])
    final_silhouette = silhouette_score(X, ml_df['Cluster_KMeans'])

    print("\n--- Cluster Evaluation Metrics (K-Means, K=4) ---")
    print(f"Silhouette Score        : {final_silhouette:.4f}  (higher is better, range -1 to 1)")
    print(f"Davies-Bouldin Index     : {db_index:.4f}  (lower is better)")
    print(f"Calinski-Harabasz Score  : {ch_score:.4f}  (higher is better)")

    evaluation_metrics = pd.DataFrame({
        "Metric": ["Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Score"],
        "Value": [round(final_silhouette, 4), round(db_index, 4), round(ch_score, 2)],
        "Interpretation": [
            "Higher is better (range -1 to 1)",
            "Lower is better",
            "Higher is better"
        ]
    })
    evaluation_metrics.to_csv("cluster_evaluation_metrics.csv", index=False)
    print("\nSaved: cluster_evaluation_metrics.csv")

except Exception as e:
    print("Error computing evaluation metrics:", e)

try:
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(X)

    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(
        pca_result[:, 0], pca_result[:, 1],
        c=ml_df['Cluster_KMeans'], cmap='viridis', alpha=0.6, s=20
    )
    plt.title('PCA 2D Projection - K-Means Clusters')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    plt.savefig('pca_kmeans_clusters.png')
    plt.close()
    print("Saved: pca_kmeans_clusters.png")

    plt.figure(figsize=(9, 7))
    scatter2 = plt.scatter(
        pca_result[:, 0], pca_result[:, 1],
        c=ml_df['Cluster_Hierarchical'], cmap='plasma', alpha=0.6, s=20
    )
    plt.title('PCA 2D Projection - Hierarchical Clusters')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    plt.colorbar(scatter2, label='Cluster')
    plt.tight_layout()
    plt.savefig('pca_hierarchical_clusters.png')
    plt.close()
    print("Saved: pca_hierarchical_clusters.png")

except Exception as e:
    print("Error generating PCA scatter plots:", e)

try:
    sample_size = min(300, X.shape[0])
    X_sample = X.sample(n=sample_size, random_state=42)

    linkage_matrix = linkage(X_sample, method='ward')

    plt.figure(figsize=(12, 6))
    dendrogram(linkage_matrix, truncate_mode='lastp', p=30, leaf_rotation=90)
    plt.title(f'Hierarchical Clustering Dendrogram (sample of {sample_size} clients)')
    plt.xlabel('Client Sample Index')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig('hierarchical_dendrogram.png')
    plt.close()
    print("Saved: hierarchical_dendrogram.png")

except Exception as e:
    print("Error generating dendrogram:", e)

print("\n" + "=" * 60)
print("STEP 7 COMPLETE - Evaluation metrics and visualizations saved")
print("=" * 60)