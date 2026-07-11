# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 10:53:51 2026

@author: rachid
"""

# -*- coding: utf-8 -*-

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# ============================================================
# 1. Fichier d'entrée
# ============================================================

input_file = r"C:\Users\rachid\Desktop\données Mahdi\Viscosity-20250113T100030Z-001\Melting_point_selected_ind_descriptors____.xlsx"

output_excel = "DES_with_RDKit_descriptors.xlsx"
output_csv = "DES_with_RDKit_descriptors.csv"

# ============================================================
# 2. Colonnes
# ============================================================

col_des_type = "DES_type"
col_mp = "Melting_point_K"
col_smiles1 = "SMILES_1"
col_smiles2 = "SMILES_2"
col_x1 = "X_1"
col_x2 = "X_2"

# ============================================================
# 3. Chargement
# ============================================================

if input_file.endswith(".xlsx"):
    df = pd.read_excel(input_file)
else:
    df = pd.read_csv(input_file)

df = df[
    [
        col_des_type,
        col_mp,
        col_smiles1,
        col_smiles2,
        col_x1,
        col_x2
    ]
].copy()

df.rename(
    columns={
        col_des_type: "des_type",
        col_smiles1: "smiles_1",
        col_smiles2: "smiles_2",
        col_x1: "x1",
        col_x2: "x2"
    },
    inplace=True
)

# ============================================================
# 4. Liste des descripteurs RDKit
# ============================================================

descriptor_names = [x[0] for x in Descriptors._descList]
descriptor_functions = [x[1] for x in Descriptors._descList]

print("Nombre de descripteurs RDKit :", len(descriptor_names))

# ============================================================
# 5. Fonction calcul descripteurs
# ============================================================

def calculate_rdkit_descriptors(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return {}

    results = {}

    for name, func in zip(descriptor_names, descriptor_functions):

        try:
            results[name] = func(mol)

        except:
            results[name] = None

    return results

# ============================================================
# 6. Calcul unique pour chaque molécule
# ============================================================

unique_smiles = pd.unique(
    pd.concat(
        [df["smiles_1"], df["smiles_2"]],
        ignore_index=True
    )
)

descriptor_dict = {}

print("SMILES uniques :", len(unique_smiles))

for i, smi in enumerate(unique_smiles):

    print(f"{i+1}/{len(unique_smiles)}")

    descriptor_dict[smi] = calculate_rdkit_descriptors(smi)

# ============================================================
# 7. Création comp1 et comp2
# ============================================================

comp1_list = []
comp2_list = []

for _, row in df.iterrows():

    d1 = descriptor_dict[row["smiles_1"]]
    d2 = descriptor_dict[row["smiles_2"]]

    d1 = {f"comp1_{k}": v for k, v in d1.items()}
    d2 = {f"comp2_{k}": v for k, v in d2.items()}

    comp1_list.append(d1)
    comp2_list.append(d2)

comp1_df = pd.DataFrame(comp1_list)
comp2_df = pd.DataFrame(comp2_list)

# ============================================================
# 8. Fusion
# ============================================================

final_df = pd.concat(
    [
        df.reset_index(drop=True),
        comp1_df.reset_index(drop=True),
        comp2_df.reset_index(drop=True)
    ],
    axis=1
)

# ============================================================
# 9. Suppression colonnes constantes
# ============================================================

descriptor_cols = [
    c for c in final_df.columns
    if c.startswith("comp1_") or c.startswith("comp2_")
]

constant_cols = [
    c for c in descriptor_cols
    if final_df[c].nunique(dropna=True) <= 1
]

final_df.drop(columns=constant_cols, inplace=True)

# ============================================================
# 10. Sauvegarde
# ============================================================

final_df.to_excel(output_excel, index=False)
final_df.to_csv(output_csv, index=False)

print()
print("Dimensions finales :", final_df.shape)
print("Fichier Excel :", output_excel)
print("Fichier CSV :", output_csv)
