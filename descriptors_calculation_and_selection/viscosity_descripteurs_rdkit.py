# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_validate
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =====================================================
# 1. PARAMÈTRES
# =====================================================

input_file = "Viscosity_selected_ind_descriptors____.xlsx"
output_rdkit_file = "DES_viscosity_with_RDKit_descriptors.xlsx"

target = "Ln_Viscosity_cP"
temperature_col = "Temperature_K"

N_COMP1 = 30
N_COMP2 = 30

USE_LOG10_TARGET = False

possible_target_cols = [
    "Ln_Viscosity_cP"
]

possible_temperature_cols = [
    "Temperature_K"
]

# =====================================================
# 2. CHARGER LE FICHIER
# =====================================================

if input_file.endswith(".xlsx") or input_file.endswith(".xls"):
    df_raw = pd.read_excel(input_file)
else:
    df_raw = pd.read_csv(input_file)

print("Colonnes disponibles :")
print(df_raw.columns.tolist())

def find_column(df, possible_names):
    for c in possible_names:
        if c in df.columns:
            return c
    raise ValueError(
        "Aucune colonne trouvée parmi : "
        + str(possible_names)
        + "\nColonnes disponibles : "
        + str(df.columns.tolist())
    )

target_col = find_column(df_raw, possible_target_cols)
temp_col = find_column(df_raw, possible_temperature_cols)

col_des_type = "DES_type"
col_smiles1 = "SMILES_1"
col_smiles2 = "SMILES_2"
col_x1 = "X_1"
col_x2 = "X_2"

df = df_raw[
    [
        col_des_type,
        target_col,
        temp_col,
        col_smiles1,
        col_smiles2,
        col_x1,
        col_x2
    ]
].copy()

df.rename(
    columns={
        col_des_type: "des_type",
        target_col: "Viscosity",
        temp_col: "Temperature",
        col_smiles1: "smiles_1",
        col_smiles2: "smiles_2",
        col_x1: "x1",
        col_x2: "x2"
    },
    inplace=True
)

df["smiles_1"] = df["smiles_1"].astype(str).str.strip()
df["smiles_2"] = df["smiles_2"].astype(str).str.strip()

df = df.dropna(
    subset=[
        "Viscosity",
        "Temperature",
        "x1",
        "x2",
        "smiles_1",
        "smiles_2"
    ]
)

# =====================================================
# 3. CALCUL DES DESCRIPTEURS RDKIT
# =====================================================

descriptor_names = [x[0] for x in Descriptors._descList]
descriptor_functions = [x[1] for x in Descriptors._descList]

print("Nombre de descripteurs RDKit :", len(descriptor_names))

def calculate_rdkit_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        print("SMILES invalide :", smiles)
        return {}

    results = {}

    for name, func in zip(descriptor_names, descriptor_functions):
        try:
            results[name] = func(mol)
        except Exception:
            results[name] = np.nan

    return results

unique_smiles = pd.unique(
    pd.concat(
        [df["smiles_1"], df["smiles_2"]],
        ignore_index=True
    )
)

descriptor_dict = {}

print("Nombre de SMILES uniques :", len(unique_smiles))

for i, smi in enumerate(unique_smiles, start=1):
    print(f"Calcul RDKit {i}/{len(unique_smiles)} : {smi}")
    descriptor_dict[smi] = calculate_rdkit_descriptors(smi)

comp1_list = []
comp2_list = []

for _, row in df.iterrows():
    d1 = descriptor_dict.get(row["smiles_1"], {})
    d2 = descriptor_dict.get(row["smiles_2"], {})

    d1 = {f"comp1_{k}": v for k, v in d1.items()}
    d2 = {f"comp2_{k}": v for k, v in d2.items()}

    comp1_list.append(d1)
    comp2_list.append(d2)

comp1_df = pd.DataFrame(comp1_list).apply(pd.to_numeric, errors="coerce")
comp2_df = pd.DataFrame(comp2_list).apply(pd.to_numeric, errors="coerce")

df_rdkit = pd.concat(
    [
        df.reset_index(drop=True),
        comp1_df.reset_index(drop=True),
        comp2_df.reset_index(drop=True)
    ],
    axis=1
)

df_rdkit.to_excel(output_rdkit_file, index=False)

print("Fichier RDKit créé :", output_rdkit_file)
print("Dimensions RDKit :", df_rdkit.shape)

# =====================================================
# 4. PRÉPARATION X / y
# =====================================================

y_raw = df_rdkit["Viscosity"].copy()

if USE_LOG10_TARGET:
    y = np.log10(y_raw)
else:
    y = y_raw.copy()

mandatory_features = ["Temperature", "x1", "x2"]
X_mandatory = df_rdkit[mandatory_features].copy()

exclude_cols = [
    "des_type",
    "smiles_1",
    "smiles_2",
    "Viscosity",
    "Temperature",
    "x1",
    "x2"
]

descriptor_cols = [
    col for col in df_rdkit.columns
    if col not in exclude_cols
]

X_desc = df_rdkit[descriptor_cols].copy()
X_desc = X_desc.select_dtypes(include=[np.number])

X_desc = X_desc.fillna(X_desc.median())
X_mandatory = X_mandatory.fillna(X_mandatory.median())

X_desc_original = X_desc.copy()

# =====================================================
# 5. STANDARDISATION DES DESCRIPTEURS RDKIT
# =====================================================

scaler = StandardScaler()

X_scaled = pd.DataFrame(
    scaler.fit_transform(X_desc),
    columns=X_desc.columns,
    index=X_desc.index
)

# =====================================================
# 6. SUPPRIMER COLONNES CONSTANTES
# =====================================================

selector = VarianceThreshold(threshold=0.0)

X_var = pd.DataFrame(
    selector.fit_transform(X_scaled),
    columns=X_scaled.columns[selector.get_support()],
    index=X_scaled.index
)

print("Après VarianceThreshold :", X_var.shape)

# =====================================================
# 7. ANALYSE DE CORRÉLATION DES DESCRIPTEURS RDKIT
# =====================================================

corr_matrix = X_var.corr().abs()
corr_matrix.to_excel("Viscosity_correlation_matrix_all_descriptors.xlsx")

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

pd.DataFrame(high_corr_pairs).to_excel(
    "Viscosity_highly_correlated_pairs_above_095.xlsx",
    index=False
)

to_drop = [
    column for column in upper.columns
    if any(upper[column] > 0.95)
]

X_uncorr = X_var.drop(columns=to_drop)

pd.DataFrame({"Removed_correlated_features": to_drop}).to_excel(
    "Viscosity_removed_correlated_features.xlsx",
    index=False
)

print("Après suppression corrélation :", X_uncorr.shape)

# =====================================================
# 8. RANDOM FOREST INITIAL AVEC TEMPERATURE, x1, x2
# =====================================================

X_importance = pd.concat(
    [
        X_mandatory.reset_index(drop=True),
        X_uncorr.reset_index(drop=True)
    ],
    axis=1
)

rf_initial = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

rf_initial.fit(X_importance, y)

importance_df = pd.DataFrame({
    "feature": X_importance.columns,
    "importance": rf_initial.feature_importances_
}).sort_values(by="importance", ascending=False)

importance_df.to_excel(
    "Viscosity_feature_importance_all_initial_with_temperature.xlsx",
    index=False
)

importance_df[
    importance_df["feature"].isin(["Temperature", "x1", "x2"])
].to_excel(
    "Viscosity_feature_importance_experimental_variables.xlsx",
    index=False
)

# =====================================================
# 9. SÉLECTION ÉQUILIBRÉE COMP1 / COMP2
# =====================================================

importance_comp1 = importance_df[
    importance_df["feature"].str.startswith("comp1_")
].copy()

importance_comp2 = importance_df[
    importance_df["feature"].str.startswith("comp2_")
].copy()

top_comp1 = importance_comp1["feature"].head(N_COMP1).tolist()
top_comp2 = importance_comp2["feature"].head(N_COMP2).tolist()

selected_features = top_comp1 + top_comp2

X_selected_scaled = X_uncorr[selected_features]
X_selected_original = X_desc_original[selected_features]

X_final_model = pd.concat(
    [
        X_mandatory.reset_index(drop=True),
        X_selected_scaled.reset_index(drop=True)
    ],
    axis=1
)

X_final_export = pd.concat(
    [
        X_mandatory.reset_index(drop=True),
        X_selected_original.reset_index(drop=True)
    ],
    axis=1
)

print("Nombre comp1 sélectionnés :", len(top_comp1))
print("Nombre comp2 sélectionnés :", len(top_comp2))
print("Nombre total de features :", X_final_model.shape[1])

selected_df = pd.concat(
    [
        df_rdkit[["Viscosity"]].reset_index(drop=True),
        X_final_export.reset_index(drop=True)
    ],
    axis=1
)

selected_df.to_excel(
    "Viscosity_selected_features_balanced_original_values.xlsx",
    index=False
)

selected_df.to_csv(
    "Viscosity_selected_features_balanced_original_values.csv",
    index=False
)

# =====================================================
# 10. DESCRIPTEURS COMMUNS ET DIFFÉRENTS
# =====================================================

base_comp1 = [f.replace("comp1_", "") for f in top_comp1]
base_comp2 = [f.replace("comp2_", "") for f in top_comp2]

common_descriptors = sorted(list(set(base_comp1).intersection(set(base_comp2))))
unique_comp1 = sorted(list(set(base_comp1) - set(base_comp2)))
unique_comp2 = sorted(list(set(base_comp2) - set(base_comp1)))

max_len = max(
    len(common_descriptors),
    len(unique_comp1),
    len(unique_comp2)
)

comparison_df = pd.DataFrame({
    "Common_descriptors": common_descriptors + [""] * (max_len - len(common_descriptors)),
    "Unique_comp1": unique_comp1 + [""] * (max_len - len(unique_comp1)),
    "Unique_comp2": unique_comp2 + [""] * (max_len - len(unique_comp2))
})

comparison_df.to_excel(
    "Viscosity_descriptor_comparison_comp1_comp2.xlsx",
    index=False
)

# =====================================================
# 11. CORRÉLATION DES FEATURES SÉLECTIONNÉES
# =====================================================

selected_corr = X_final_export.corr().abs()
selected_corr.to_excel(
    "Viscosity_correlation_matrix_selected_features_original_values.xlsx"
)

upper_selected = selected_corr.where(
    np.triu(np.ones(selected_corr.shape), k=1).astype(bool)
)

selected_corr_pairs = []

for col in upper_selected.columns:
    correlated = upper_selected.index[upper_selected[col] > 0.80].tolist()
    for row in correlated:
        selected_corr_pairs.append({
            "feature_1": row,
            "feature_2": col,
            "correlation": upper_selected.loc[row, col]
        })

pd.DataFrame(selected_corr_pairs).to_excel(
    "Viscosity_selected_features_correlated_pairs_above_080_original_values.xlsx",
    index=False
)

# =====================================================
# 12. TRAIN / TEST
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_final_model,
    y,
    test_size=0.20,
    random_state=42
)

# =====================================================
# 13. GRID SEARCH CV RANDOM FOREST
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

print("Meilleurs paramètres RF :")
print(grid.best_params_)

pd.DataFrame(grid.cv_results_).to_excel(
    "Viscosity_gridsearch_RF_results.xlsx",
    index=False
)

# =====================================================
# 14. MÉTRIQUES TRAIN / TEST
# =====================================================

y_train_pred = best_rf.predict(X_train)
y_test_pred = best_rf.predict(X_test)

if USE_LOG10_TARGET:
    y_train_true_original = 10 ** y_train
    y_test_true_original = 10 ** y_test
    y_train_pred_original = 10 ** y_train_pred
    y_test_pred_original = 10 ** y_test_pred
else:
    y_train_true_original = y_train
    y_test_true_original = y_test
    y_train_pred_original = y_train_pred
    y_test_pred_original = y_test_pred

def regression_metrics(y_true, y_pred, dataset_name):
    return {
        "Dataset": dataset_name,
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred))
    }

metrics_df = pd.DataFrame([
    regression_metrics(y_train_true_original, y_train_pred_original, "Train"),
    regression_metrics(y_test_true_original, y_test_pred_original, "Test")
])

metrics_df.to_excel(
    "Viscosity_model_metrics_train_test.xlsx",
    index=False
)

predictions_df = pd.DataFrame({
    "y_test_true": y_test_true_original.values,
    "y_test_pred": y_test_pred_original,
    "residual": y_test_true_original.values - y_test_pred_original
})

predictions_df.to_excel(
    "Viscosity_predictions_test_RF.xlsx",
    index=False
)

# =====================================================
# 15. CROSS-VALIDATION FINALE
# =====================================================

cv_scores = cross_validate(
    best_rf,
    X_final_model,
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

cv_results_summary.to_excel(
    "Viscosity_cross_validation_metrics_RF.xlsx",
    index=False
)

# =====================================================
# 16. FEATURE IMPORTANCE FINALE
# =====================================================

final_importance_df = pd.DataFrame({
    "feature": X_final_model.columns,
    "importance": best_rf.feature_importances_
}).sort_values(by="importance", ascending=False)

final_importance_df.to_excel(
    "Viscosity_feature_importance_final_selected_RF.xlsx",
    index=False
)

final_importance_df[
    final_importance_df["feature"].isin(["Temperature", "x1", "x2"])
].to_excel(
    "Viscosity_feature_importance_final_experimental_variables.xlsx",
    index=False
)

final_importance_df[
    final_importance_df["feature"].str.startswith("comp1_")
].to_excel(
    "Viscosity_feature_importance_final_comp1.xlsx",
    index=False
)

final_importance_df[
    final_importance_df["feature"].str.startswith("comp2_")
].to_excel(
    "Viscosity_feature_importance_final_comp2.xlsx",
    index=False
)

# =====================================================
# 17. IMPORTANCE PAR FAMILLE CHIMIQUE
# =====================================================

def descriptor_family(feature):
    if feature == "Temperature":
        return "temperature"

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

final_importance_df["family"] = final_importance_df["feature"].apply(
    descriptor_family
)

family_importance = (
    final_importance_df
    .groupby("family")["importance"]
    .sum()
    .reset_index()
    .sort_values(by="importance", ascending=False)
)

family_importance.to_excel(
    "Viscosity_feature_importance_by_descriptor_family.xlsx",
    index=False
)

# =====================================================
# 18. SAUVEGARDE LISTE FINALE DES FEATURES
# =====================================================

selected_features_df = pd.DataFrame({
    "order": range(1, len(X_final_model.columns) + 1),
    "feature": X_final_model.columns
})

selected_features_df.to_excel(
    "Viscosity_selected_features_final_ordered.xlsx",
    index=False
)

# =====================================================
# 19. RÉSUMÉ FINAL
# =====================================================

print("\n================ RÉSUMÉ VISCOSITÉ ================")
print("Nombre de lignes :", df_rdkit.shape[0])
print("Nombre de features finales :", X_final_model.shape[1])
print("Features expérimentales incluses : Temperature, x1, x2")
print("Nombre comp1 :", len(top_comp1))
print("Nombre comp2 :", len(top_comp2))
print("Meilleurs paramètres RF :", grid.best_params_)

print("\nMetrics train/test :")
print(metrics_df)

print("\nCross-validation :")
print(cv_results_summary)

print("\nFichiers principaux créés :")
print("DES_viscosity_with_RDKit_descriptors.xlsx")
print("Viscosity_feature_importance_all_initial_with_temperature.xlsx")
print("Viscosity_feature_importance_experimental_variables.xlsx")
print("Viscosity_selected_features_balanced_original_values.xlsx")
print("Viscosity_selected_features_balanced_original_values.csv")
print("Viscosity_descriptor_comparison_comp1_comp2.xlsx")
print("Viscosity_correlation_matrix_selected_features_original_values.xlsx")
print("Viscosity_gridsearch_RF_results.xlsx")
print("Viscosity_model_metrics_train_test.xlsx")
print("Viscosity_cross_validation_metrics_RF.xlsx")
print("Viscosity_feature_importance_final_selected_RF.xlsx")
print("Viscosity_feature_importance_final_experimental_variables.xlsx")
print("Viscosity_feature_importance_by_descriptor_family.xlsx")