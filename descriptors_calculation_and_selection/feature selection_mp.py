# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:22:58 2026

@author: rachid
"""



# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_validate
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =====================================================
# 1. Charger les données
# =====================================================

input_file = "DES_with_RDKit_descriptors.xlsx"
df = pd.read_excel(input_file)

target = "Melting_point_K"
mandatory_features = ["x1", "x2"]

exclude_cols = [
    "des_type", "smiles_1", "smiles_2",
    target, "x1", "x2"
]

y = df[target].copy()
X_mandatory = df[mandatory_features].copy()

descriptor_cols = [col for col in df.columns if col not in exclude_cols]

X_desc = df[descriptor_cols].copy()
X_desc = X_desc.select_dtypes(include=[np.number])

# Remplacement des valeurs manquantes
X_desc = X_desc.fillna(X_desc.median())
X_mandatory = X_mandatory.fillna(X_mandatory.median())

# Sauvegarde des valeurs originales NON normalisées
X_desc_original = X_desc.copy()

# =====================================================
# 2. Standardisation uniquement pour la sélection
# =====================================================

scaler = StandardScaler()

X_scaled = pd.DataFrame(
    scaler.fit_transform(X_desc),
    columns=X_desc.columns,
    index=X_desc.index
)

# =====================================================
# 3. Supprimer colonnes constantes
# =====================================================

selector = VarianceThreshold(threshold=0.0)

X_var = pd.DataFrame(
    selector.fit_transform(X_scaled),
    columns=X_scaled.columns[selector.get_support()],
    index=X_scaled.index
)

print("Après VarianceThreshold :", X_var.shape)

# =====================================================
# 4. Analyse de corrélation globale sur données normalisées
# =====================================================

corr_matrix = X_var.corr().abs()
corr_matrix.to_excel("Correlation_matrix_all_descriptors.xlsx")

upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

high_corr_pairs = []

for col in upper.columns:
    correlated = upper.index[upper[col] > 0.95].tolist()
    for row in correlated:
        high_corr_pairs.append({
            "feature_1": row,
            "feature_2": col,
            "correlation": upper.loc[row, col]
        })

high_corr_df = pd.DataFrame(high_corr_pairs)
high_corr_df.to_excel("Highly_correlated_pairs_above_095.xlsx", index=False)

to_drop = [
    column for column in upper.columns
    if any(upper[column] > 0.95)
]

X_uncorr = X_var.drop(columns=to_drop)

print("Après suppression corrélation :", X_uncorr.shape)

pd.DataFrame({"Removed_correlated_features": to_drop}).to_excel(
    "Removed_correlated_features.xlsx",
    index=False
)

# =====================================================
# 5. Random Forest initial pour feature importance
# =====================================================

rf_initial = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

rf_initial.fit(X_uncorr, y)

importance_df = pd.DataFrame({
    "feature": X_uncorr.columns,
    "importance": rf_initial.feature_importances_
}).sort_values(by="importance", ascending=False)

importance_df.to_excel("Feature_importance_RDKit_all_initial.xlsx", index=False)

# =====================================================
# 6. Sélection équilibrée : 31 comp1 + 31 comp2
# =====================================================

importance_comp1 = importance_df[
    importance_df["feature"].str.startswith("comp1_")
].copy()

importance_comp2 = importance_df[
    importance_df["feature"].str.startswith("comp2_")
].copy()

top_comp1 = importance_comp1["feature"].head(31).tolist()
top_comp2 = importance_comp2["feature"].head(31).tolist()

selected_features = top_comp1 + top_comp2

# IMPORTANT :
# on récupère ici les valeurs originales non normalisées
X_selected_desc = X_desc_original[selected_features]

X_final = pd.concat(
    [
        X_mandatory.reset_index(drop=True),
        X_selected_desc.reset_index(drop=True)
    ],
    axis=1
)

print("Nombre final de features :", X_final.shape[1])

selected_df = pd.concat(
    [
        df[[target]].reset_index(drop=True),
        X_final.reset_index(drop=True)
    ],
    axis=1
)

selected_df.to_excel("DES_selected_64_features_original_values.xlsx", index=False)
selected_df.to_csv("DES_selected_64_features_original_values.csv", index=False)

# =====================================================
# 7. Analyse des descripteurs communs/différents
# =====================================================

base_comp1 = [f.replace("comp1_", "") for f in top_comp1]
base_comp2 = [f.replace("comp2_", "") for f in top_comp2]

common_descriptors = sorted(list(set(base_comp1).intersection(set(base_comp2))))
unique_comp1 = sorted(list(set(base_comp1) - set(base_comp2)))
unique_comp2 = sorted(list(set(base_comp2) - set(base_comp1)))

max_len = max(len(common_descriptors), len(unique_comp1), len(unique_comp2))

comparison_df = pd.DataFrame({
    "Common_descriptors": common_descriptors + [""] * (max_len - len(common_descriptors)),
    "Unique_comp1": unique_comp1 + [""] * (max_len - len(unique_comp1)),
    "Unique_comp2": unique_comp2 + [""] * (max_len - len(unique_comp2))
})

comparison_df.to_excel("Descriptor_comparison_comp1_comp2.xlsx", index=False)

# =====================================================
# 8. Corrélation des features sélectionnées
# =====================================================

selected_corr = X_final.corr().abs()
selected_corr.to_excel("Correlation_matrix_selected_64_features_original_values.xlsx")

selected_corr_pairs = []

upper_selected = selected_corr.where(
    np.triu(np.ones(selected_corr.shape), k=1).astype(bool)
)

for col in upper_selected.columns:
    correlated = upper_selected.index[upper_selected[col] > 0.80].tolist()
    for row in correlated:
        selected_corr_pairs.append({
            "feature_1": row,
            "feature_2": col,
            "correlation": upper_selected.loc[row, col]
        })

selected_corr_pairs_df = pd.DataFrame(selected_corr_pairs)
selected_corr_pairs_df.to_excel(
    "Selected_features_correlated_pairs_above_080_original_values.xlsx",
    index=False
)

# =====================================================
# 9. Train/test split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_final,
    y,
    test_size=0.20,
    random_state=42
)

# =====================================================
# 10. GridSearchCV Random Forest
# =====================================================

param_grid = {
    "n_estimators": [300, 500, 800],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", 0.5, 1.0]
}

rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring="r2",
    cv=cv,
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)

best_rf = grid.best_estimator_

print("Meilleurs paramètres :")
print(grid.best_params_)

grid_results = pd.DataFrame(grid.cv_results_)
grid_results.to_excel("GridSearchCV_results_RF.xlsx", index=False)

# =====================================================
# 11. Validation train/test
# =====================================================

y_train_pred = best_rf.predict(X_train)
y_test_pred = best_rf.predict(X_test)

def regression_metrics(y_true, y_pred, dataset_name):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return {
        "Dataset": dataset_name,
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse
    }

metrics = []

metrics.append(regression_metrics(y_train, y_train_pred, "Train"))
metrics.append(regression_metrics(y_test, y_test_pred, "Test"))

metrics_df = pd.DataFrame(metrics)
metrics_df.to_excel("Model_metrics_train_test.xlsx", index=False)

predictions_df = pd.DataFrame({
    "y_test_true": y_test.values,
    "y_test_pred": y_test_pred,
    "residual": y_test.values - y_test_pred
})

predictions_df.to_excel("Predictions_test_RF.xlsx", index=False)

# =====================================================
# 12. Cross-validation complète
# =====================================================

cv_scores = cross_validate(
    best_rf,
    X_final,
    y,
    cv=cv,
    scoring={
        "R2": "r2",
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error"
    },
    return_train_score=True,
    n_jobs=-1
)

cv_results_summary = pd.DataFrame({
    "Metric": [
        "Train_R2",
        "Test_R2",
        "Train_MAE",
        "Test_MAE",
        "Train_RMSE",
        "Test_RMSE"
    ],
    "Mean": [
        np.mean(cv_scores["train_R2"]),
        np.mean(cv_scores["test_R2"]),
        -np.mean(cv_scores["train_MAE"]),
        -np.mean(cv_scores["test_MAE"]),
        -np.mean(cv_scores["train_RMSE"]),
        -np.mean(cv_scores["test_RMSE"])
    ],
    "Std": [
        np.std(cv_scores["train_R2"]),
        np.std(cv_scores["test_R2"]),
        np.std(-cv_scores["train_MAE"]),
        np.std(-cv_scores["test_MAE"]),
        np.std(-cv_scores["train_RMSE"]),
        np.std(-cv_scores["test_RMSE"])
    ]
})

cv_results_summary.to_excel("Cross_validation_metrics_RF.xlsx", index=False)

# =====================================================
# 13. Feature importance finale avec meilleur modèle
# =====================================================

final_importance_df = pd.DataFrame({
    "feature": X_final.columns,
    "importance": best_rf.feature_importances_
}).sort_values(by="importance", ascending=False)

final_importance_df.to_excel("Feature_importance_final_selected_64_RF.xlsx", index=False)

final_importance_comp1 = final_importance_df[
    final_importance_df["feature"].str.startswith("comp1_")
]

final_importance_comp2 = final_importance_df[
    final_importance_df["feature"].str.startswith("comp2_")
]

final_importance_comp1.to_excel("Feature_importance_final_comp1.xlsx", index=False)
final_importance_comp2.to_excel("Feature_importance_final_comp2.xlsx", index=False)

# =====================================================
# 14. Importance par famille de descripteurs
# =====================================================

def descriptor_family(feature):
    if feature in ["x1", "x2"]:
        return "composition"

    base = feature.replace("comp1_", "").replace("comp2_", "")

    if "BCUT" in base:
        return "BCUT"
    elif "VSA" in base:
        return "VSA"
    elif "EState" in base:
        return "EState"
    elif "PEOE" in base:
        return "PEOE"
    elif "SlogP" in base or "MolLogP" in base or "LOGP" in base:
        return "hydrophobicity"
    elif "TPSA" in base:
        return "polarity_TPSA"
    elif "Kappa" in base or "Chi" in base:
        return "shape_topology"
    elif "FpDensity" in base:
        return "fingerprint_density"
    elif "MolWt" in base or "MW" in base:
        return "molecular_weight"
    elif "qed" in base:
        return "qed_composite"
    else:
        return "other"

final_importance_df["family"] = final_importance_df["feature"].apply(descriptor_family)

family_importance = (
    final_importance_df
    .groupby("family")["importance"]
    .sum()
    .reset_index()
    .sort_values(by="importance", ascending=False)
)

family_importance.to_excel("Feature_importance_by_descriptor_family.xlsx", index=False)

# =====================================================
# 15. Sauvegarder liste finale
# =====================================================

selected_features_df = pd.DataFrame({
    "order": range(1, len(X_final.columns) + 1),
    "feature": X_final.columns
})

selected_features_df.to_excel("Selected_64_features_final.xlsx", index=False)

# =====================================================
# 16. Résumé final
# =====================================================

print("\n================ RÉSUMÉ ================")
print("Nombre total de lignes :", df.shape[0])
print("Nombre de features finales :", X_final.shape[1])
print("Meilleurs paramètres RF :", grid.best_params_)

print("\nMetrics train/test :")
print(metrics_df)

print("\nCross-validation :")
print(cv_results_summary)

print("\nFichiers créés :")
print("DES_selected_64_features_original_values.xlsx")
print("DES_selected_64_features_original_values.csv")
print("Descriptor_comparison_comp1_comp2.xlsx")
print("Correlation_matrix_all_descriptors.xlsx")
print("Highly_correlated_pairs_above_095.xlsx")
print("Correlation_matrix_selected_64_features_original_values.xlsx")
print("Selected_features_correlated_pairs_above_080_original_values.xlsx")
print("GridSearchCV_results_RF.xlsx")
print("Model_metrics_train_test.xlsx")
print("Cross_validation_metrics_RF.xlsx")
print("Predictions_test_RF.xlsx")
print("Feature_importance_final_selected_64_RF.xlsx")
print("Feature_importance_by_descriptor_family.xlsx")
print("Selected_64_features_final.xlsx")