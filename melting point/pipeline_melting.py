# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 23:57:26 2026

@author: rachid
"""

# -*- coding: utf-8 -*-

"""
Supervised VAE pipeline for melting point prediction and COCONUT matching.


"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from scipy.stats import loguniform, randint, uniform


DEFAULTS = {
    "train_xlsx": "DES_selected_64_features_original_values.xlsx",
    
    "train_smiles_xlsx": "Melting_point_selected_ind_descriptors____.xlsx",
    "coconut_xlsx": "coconut_selected_descriptors.xlsx",
    "outdir": "orchestrator_outputs_melting",
    "coconut_sheet": "melting_point",
}



# ============================================================
# Reproducibility utilities
# ============================================================
def seed_everything(seed: int = 2026, deterministic: bool = True):
    """
    Fixes the main sources of randomness used by numpy, Python, PyTorch,
    scikit-learn and XGBoost wrappers as far as possible.

    Notes:
    - Exact bitwise reproducibility is easiest on CPU.
    - On GPU, some CUDA operations can still differ slightly depending on
      PyTorch/CUDA versions and hardware.
    """
    seed = int(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)

    return seed


def make_torch_generator(seed: int, device: str = "cpu"):
    """Creates a seeded torch.Generator for deterministic DataLoader shuffling."""
    # DataLoader expects a CPU generator for index shuffling.
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return g


def seed_worker(worker_id):
    """Initializes DataLoader workers deterministically."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def ensure_outdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


POSSIBLE_TARGETS_MP = [
    "Melting_point_K", "Melting_point", "melting_point",
    "melting_point_k", "Tmelt, K", "Tmelt (K)",
    "Tmelt", "T_melt", "T_melt_K", "MP", "mp",
]


def _find_mp_col(cols):
    cols = list(cols)
    low = [str(c).lower() for c in cols]

    for name in POSSIBLE_TARGETS_MP:
        if name in cols:
            return name

    for name in POSSIBLE_TARGETS_MP:
        if name.lower() in low:
            return cols[low.index(name.lower())]

    for i, c in enumerate(low):
        if "melt" in c or "fusion" in c:
            return cols[i]

    return None


def detect_columns(df: pd.DataFrame):
    target = _find_mp_col(df.columns)

    if target is None:
        raise ValueError("Unable to detect the melting point target column.")

    numeric = df.select_dtypes(include=[np.number]).copy()

    if target in numeric.columns:
        X = numeric.drop(columns=[target])
    else:
        X = numeric.copy()

    y = pd.to_numeric(df[target], errors="coerce")

    valid = y.notna()
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)

    if X.shape[1] == 0:
        raise ValueError("No numeric descriptor columns detected.")

    return X, y, target


def detect_smiles_col(df: pd.DataFrame):
    for c in df.columns:
        if str(c).lower() == "smiles":
            return c
    for c in df.columns:
        if "smiles" in str(c).lower():
            return c
    return None


def choose_coconut_sheet(coconut_xlsx: Path, preferred_sheet: str):
    xl = pd.ExcelFile(coconut_xlsx)
    sheets = xl.sheet_names

    if preferred_sheet in sheets:
        return preferred_sheet

    if "melting_point" in sheets:
        print("[WARN] Requested sheet not found. Using 'melting_point'.")
        return "melting_point"

    if "all_selected_descriptors" in sheets:
        print("[WARN] Requested sheet not found. Using 'all_selected_descriptors'.")
        return "all_selected_descriptors"

    print(f"[WARN] Requested sheet not found. Using: {sheets[0]}")
    return sheets[0]


def safe_numeric(df: pd.DataFrame, columns, clip_value=1e12):
    out = df[list(columns)].copy()
    out = out.apply(pd.to_numeric, errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.mask(out.abs() > clip_value, np.nan)

    med = out.median(numeric_only=True)
    med = med.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out = out.fillna(med).fillna(0.0)
    out = out.replace([np.inf, -np.inf], 0.0)
    out = out.mask(out.abs() > clip_value, 0.0)

    return out.astype(np.float64), med


def clean_numeric_matrix(df: pd.DataFrame, med=None, clip_value=1e12):
    out = df.copy()
    out = out.apply(pd.to_numeric, errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.mask(out.abs() > clip_value, np.nan)

    if med is not None:
        out = out.fillna(med)

    out = out.fillna(0.0)
    out = out.replace([np.inf, -np.inf], 0.0)
    out = out.mask(out.abs() > clip_value, 0.0)

    return out.astype(np.float64)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def canonical_pair(s1, s2):
    a = "" if pd.isna(s1) else str(s1).strip()
    b = "" if pd.isna(s2) else str(s2).strip()
    first, second = sorted([a, b])
    return first + " || " + second
class SurrogateMLP(nn.Module):
    def __init__(self, in_dim, hidden=(256, 128, 64)):
        super().__init__()
        layers = []
        last = in_dim

        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h

        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class VAE(nn.Module):
    def __init__(self, in_dim, latent_dim=16, hidden=(256, 128, 64)):
        super().__init__()

        enc = []
        last = in_dim
        for h in hidden:
            enc += [nn.Linear(last, h), nn.ReLU()]
            last = h

        self.encoder = nn.Sequential(*enc)
        self.mu = nn.Linear(last, latent_dim)
        self.logvar = nn.Linear(last, latent_dim)

        dec = []
        last = latent_dim
        for h in reversed(hidden):
            dec += [nn.Linear(last, h), nn.ReLU()]
            last = h

        dec += [nn.Linear(last, in_dim)]
        self.decoder = nn.Sequential(*dec)

        self.t_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        xr = self.decode(z)
        t_pred = self.t_head(z).squeeze(-1)
        return xr, mu, logvar, z, t_pred


def vae_loss(x, xr, mu, logvar, t_true, t_pred, beta=1e-3, gamma=2.5, feat_w=None):
    if feat_w is None:
        rec = F.mse_loss(xr, x, reduction="mean")
    else:
        w = feat_w.to(x.device)
        rec = torch.mean(((xr - x) ** 2) * w)

    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    tl = F.mse_loss(t_pred, t_true, reduction="mean")

    return rec + beta * kl + gamma * tl, rec.detach(), kl.detach(), tl.detach()


def train_surrogate(Xs, y_rf, device="cpu", epochs=200, batch=256, lr=1e-3, seed=2026):
    ds = TensorDataset(
        torch.from_numpy(Xs.astype(np.float32)),
        torch.from_numpy(y_rf.astype(np.float32))
    )

    dl = DataLoader(
        ds,
        batch_size=batch,
        shuffle=True,
        drop_last=False,
        generator=make_torch_generator(seed),
        worker_init_fn=seed_worker,
        num_workers=0,
    )

    model = SurrogateMLP(Xs.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(1, epochs + 1):
        losses = []
        model.train()

        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)

            yp = model(xb)
            loss = loss_fn(yp, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())

        if ep == 1 or ep % 50 == 0 or ep == epochs:
            print(f"[Surrogate] {ep:03d} MSE={np.mean(losses):.6f}")

    model.eval()
    return model


def train_vae(
    Xs,
    Ts_std,
    feat_w=None,
    latent_dim=16,
    epochs=300,
    batch=256,
    lr=1e-3,
    beta_final=1e-3,
    gamma_final=2.5,
    device="cpu",
    seed=2026
):
    ds = TensorDataset(
        torch.from_numpy(Xs.astype(np.float32)),
        torch.from_numpy(Ts_std.astype(np.float32))
    )

    dl = DataLoader(
        ds,
        batch_size=batch,
        shuffle=True,
        drop_last=False,
        generator=make_torch_generator(seed),
        worker_init_fn=seed_worker,
        num_workers=0,
    )

    vae = VAE(Xs.shape[1], latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=lr)

    fw = None
    if feat_w is not None:
        fw = torch.from_numpy(feat_w.astype(np.float32))

    for ep in range(1, epochs + 1):
        r = min(1.0, ep / max(1, int(0.3 * epochs)))
        beta = r * beta_final
        gamma = 1.0 + r * (gamma_final - 1.0)

        L, R, K, T = [], [], [], []
        vae.train()

        for xb, tb in dl:
            xb = xb.to(device)
            tb = tb.to(device)

            xr, mu, logvar, z, t_pred = vae(xb)

            loss, rec, kl, tl = vae_loss(
                xb,
                xr,
                mu,
                logvar,
                tb,
                t_pred,
                beta=beta,
                gamma=gamma,
                feat_w=fw
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            L.append(loss.item())
            R.append(rec.item())
            K.append(kl.item())
            T.append(tl.item())

        if ep == 1 or ep % 25 == 0 or ep == epochs:
            print(
                f"[VAE] {ep:03d} "
                f"loss={np.mean(L):.4f} "
                f"rec={np.mean(R):.4f} "
                f"kl={np.mean(K):.4f} "
                f"T={np.mean(T):.4f}"
            )

    vae.eval()
    return vae


@torch.no_grad()
def encode_mu(vae, Xs, device="cpu", batch=512):
    mus = []
    vae.eval()

    for i in range(0, Xs.shape[0], batch):
        xb = torch.from_numpy(Xs[i:i + batch].astype(np.float32)).to(device)
        mu, _ = vae.encode(xb)
        mus.append(mu.cpu())

    return torch.cat(mus, 0)


def cond_optimize_z(
    vae,
    mu,
    T_target_std,
    lam=0.1,
    iters=60,
    lr=0.05,
    surrogate=None,
    alpha=0.7,
    sT=None,
    device="cpu"
):
    z = mu.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)

    for _ in range(iters):
        t_std = vae.t_head(z).squeeze(-1)
        loss = (t_std - T_target_std) ** 2 + lam * torch.mean((z - mu) ** 2)

        if surrogate is not None and sT is not None:
            x_dec = vae.decode(z)
            t_rf_hat = surrogate(x_dec)
            T_target_K = T_target_std * sT.scale_[0] + sT.mean_[0]
            loss = loss + alpha * torch.mean((t_rf_hat - T_target_K) ** 2)

        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        Xs_gen = vae.decode(z)

    return Xs_gen
def step1_train_and_generate(
    train_xlsx: Path,
    outdir: Path,
    n_gen=200,
    latent_dim=16,
    epochs=300,
    batch=256,
    lr=1e-3,
    beta=1e-3,
    gamma=2.5,
    device=None,
    tmin=None,
    tmax=None,
    seed=2026,
    deterministic=True,
    force_cpu=False
):
    seed_everything(seed, deterministic=deterministic)
    if force_cpu:
        device = "cpu"
    else:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("[Step1] Train file:", train_xlsx)
    print("[Step1] Device:", device)

    df = pd.read_excel(train_xlsx).dropna().reset_index(drop=True)
    X, y, target_name = detect_columns(df)

    print("[Step1] Target:", target_name)
    print("[Step1] X shape:", X.shape)

    sX = StandardScaler().fit(X.values)
    Xs = sX.transform(X.values)

    sT = StandardScaler().fit(y.values.reshape(-1, 1))
    Ts_std = sT.transform(y.values.reshape(-1, 1)).reshape(-1)

    joblib.dump(sX, outdir / "scaler_X.pkl")
    joblib.dump(sT, outdir / "scaler_T.pkl")

    rf = RandomForestRegressor(
        n_estimators=500,
        random_state=seed,
        n_jobs=-1
    )
    rf.fit(Xs, y.values)

    gb = GradientBoostingRegressor(
        n_estimators=600,
        learning_rate=0.05,
        random_state=seed
    )
    gb.fit(Xs, y.values)

    y_rf = rf.predict(Xs)

    print("[Step1] RF internal R2:", r2_score(y, y_rf))
    print("[Step1] RF internal RMSE:", rmse(y, y_rf))

    feat_w = rf.feature_importances_.astype(np.float32)
    feat_w = feat_w / (feat_w.mean() + 1e-8)
    np.save(outdir / "feature_weights.npy", feat_w)

    surrogate = train_surrogate(
        Xs,
        y_rf,
        device=device,
        epochs=200,
        batch=batch,
        lr=1e-3,
        seed=seed
    )

    torch.save(surrogate.state_dict(), outdir / "surrogate.pt")
    joblib.dump(rf, outdir / "rf_full_mp.joblib")
    joblib.dump(gb, outdir / "gb_full_mp.joblib")

    vae = train_vae(
        Xs,
        Ts_std,
        feat_w=feat_w,
        latent_dim=latent_dim,
        epochs=epochs,
        batch=batch,
        lr=lr,
        beta_final=beta,
        gamma_final=gamma,
        device=device,
        seed=seed
    )

    torch.save(vae.state_dict(), outdir / "vae.pt")

    mus_train = encode_mu(vae, Xs, device=device).cpu().numpy()

    rng = np.random.default_rng(int(seed))
    idxs = rng.choice(mus_train.shape[0], size=n_gen, replace=True)

    if tmin is not None and tmax is not None:
        if float(tmin) >= float(tmax):
            raise ValueError("tmin must be lower than tmax.")
        T_targets_K = rng.uniform(float(tmin), float(tmax), size=n_gen)
    else:
        T_targets_K = y.values[idxs]

    T_targets_std = sT.transform(T_targets_K.reshape(-1, 1)).reshape(-1)

    mu_sel = torch.from_numpy(mus_train[idxs]).to(device)
    surrogate = surrogate.to(device)

    Xs_gen_list = []

    for i in range(n_gen):
        Xs_i = cond_optimize_z(
            vae,
            mu_sel[i:i + 1],
            torch.tensor([T_targets_std[i]], dtype=torch.float32, device=device),
            lam=0.1,
            iters=60,
            lr=0.05,
            surrogate=None,#surrogate,
            alpha=0.7,
            sT=sT,
            device=device
        )
        Xs_gen_list.append(Xs_i.cpu().numpy())

    Xs_gen = np.concatenate(Xs_gen_list, axis=0)

    # Convert back to original values for export.
    X_gen = sX.inverse_transform(Xs_gen)

    out_df = pd.DataFrame(X_gen, columns=X.columns)
    out_df[target_name] = T_targets_K

    gen_path = outdir / "generated_melting_point.xlsx"
    out_df.to_excel(gen_path, index=False)

    print("[Step1] Generated file:", gen_path)

    try:
        pca = PCA(n_components=2, random_state=seed)
        coords_train = pca.fit_transform(mus_train)

        mu_gen = encode_mu(vae, Xs_gen, device=device).cpu().numpy()
        coords_gen = pca.transform(mu_gen)

        pd.DataFrame({
            "PC1": coords_train[:, 0],
            "PC2": coords_train[:, 1],
            target_name: y.values
        }).to_csv(outdir / "latent_pca_coords_training_mp.csv", index=False)

        pd.DataFrame({
            "PC1": coords_gen[:, 0],
            "PC2": coords_gen[:, 1],
            target_name: T_targets_K
        }).to_csv(outdir / "latent_pca_coords_generated_mp.csv", index=False)

        plt.figure(figsize=(6.5, 5.5))
        sc = plt.scatter(coords_train[:, 0], coords_train[:, 1], c=y.values, s=14, alpha=0.7, label="Training set")
        cbar = plt.colorbar(sc)
        cbar.set_label("Melting point (K)")

        plt.scatter(coords_gen[:, 0], coords_gen[:, 1], s=28, facecolors="none", edgecolors="k", linewidths=1.0, label="Generated candidates")

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("SVAE latent space for generated candidates")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(outdir / "latent_pca_combined_mp.png", dpi=160)
        plt.close()

    except Exception as e:
        print("[WARN] PCA skipped:", e)

    return gen_path, target_name


def step2_match_to_coconut(
    generated_xlsx: Path,
    coconut_xlsx: Path,
    outdir: Path,
    train_xlsx: Path = None,
    train_smiles_xlsx: Path = None,
    coconut_sheet="melting_point",
    n_neighbors=1,
    n_keep=400,
    exclude_existing_pairs=True
):
    print("[Step2] Generated:", generated_xlsx)
    print("[Step2] COCONUT:", coconut_xlsx)

    gen = pd.read_excel(generated_xlsx)
    sheet = choose_coconut_sheet(coconut_xlsx, coconut_sheet)

    print("[Step2] Sheet used:", sheet)

    target_col = _find_mp_col(gen.columns)

    exclude = {target_col, "x1", "x2", "X#1", "X#2", "X_1", "X_2"}
    exclude = {c for c in exclude if c is not None}
    exclude |= {c for c in gen.columns if "smiles" in str(c).lower()}

    feature_cols = [
        c for c in gen.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]

    if len(feature_cols) == 0:
        raise RuntimeError("No numeric descriptors detected in the generated file.")

    comp1_cols = [c for c in feature_cols if str(c).startswith("comp1_")]
    comp2_cols = [c for c in feature_cols if str(c).startswith("comp2_")]

    if len(comp1_cols) == 0 or len(comp2_cols) == 0:
        raise RuntimeError("comp1_ / comp2_ columns not found.")

    comp1_base = [str(c).replace("comp1_", "", 1) for c in comp1_cols]
    comp2_base = [str(c).replace("comp2_", "", 1) for c in comp2_cols]

    needed = set(feature_cols) | set(comp1_base) | set(comp2_base)

    def use_col(c):
        s = str(c)
        sl = s.lower()
        return (
            s in needed
            or "smiles" in sl
            or "id" in sl
            or "name" in sl
            or sl in {"identifier", "coconut_id"}
        )

    coco = pd.read_excel(coconut_xlsx, sheet_name=sheet, usecols=use_col)

    if coco.empty:
        raise RuntimeError(f"The sheet '{sheet}' is empty or no useful column was read.")

    smiles_col = detect_smiles_col(coco)

    id_cols = [
        c for c in coco.columns
        if c != smiles_col and (
            "id" in str(c).lower()
            or "name" in str(c).lower()
            or str(c).lower() in {"identifier", "coconut_id"}
        )
    ]

    pair_common = [c for c in feature_cols if c in coco.columns]

    if len(pair_common) >= max(5, int(0.70 * len(feature_cols))):
        print(f"[Step2] DES-pair mode: {len(pair_common)} common columns.")

        coco_X, med = safe_numeric(coco, pair_common)
        gen_X = clean_numeric_matrix(gen[pair_common], med=med)

        scaler = StandardScaler().fit(coco_X.values)
        Xc = scaler.transform(coco_X.values)
        Xg = scaler.transform(gen_X.values)

        Xc = np.nan_to_num(Xc, nan=0.0, posinf=0.0, neginf=0.0)
        Xg = np.nan_to_num(Xg, nan=0.0, posinf=0.0, neginf=0.0)

        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        nn.fit(Xc)

        dist, idx = nn.kneighbors(Xg)
        best = idx[:, 0]

        out = gen.copy()

        for c in pair_common:
            out[c] = coco.iloc[best][c].to_numpy()

        out["coconut_match_index"] = best
        out["coconut_match_distance"] = dist[:, 0]

        if smiles_col is not None:
            out["COCONUT_SMILES"] = coco.iloc[best][smiles_col].to_numpy()

        for c in id_cols[:5]:
            out[f"COCONUT_{c}"] = coco.iloc[best][c].to_numpy()

        # Exclude SMILES pairs already present in the training set,
        # then keep the n_keep best candidates based on minimum distance.
        out = filter_existing_training_pairs(
            out,
            train_xlsx=train_xlsx,
            train_smiles_xlsx=train_smiles_xlsx,
            exclude_existing_pairs=exclude_existing_pairs
        )
        out = keep_best_by_distance(
            out,
            distance_col="coconut_match_distance",
            n_keep=n_keep
        )

        matched_path = outdir / "matched_real_from_coconut_mp.xlsx"
        out.to_excel(matched_path, index=False)

        print("[Step2] Matched file:", matched_path)
        return matched_path

    common1 = [(gc, bc) for gc, bc in zip(comp1_cols, comp1_base) if bc in coco.columns]
    common2 = [(gc, bc) for gc, bc in zip(comp2_cols, comp2_base) if bc in coco.columns]

    if len(common1) < 3 or len(common2) < 3:
        raise RuntimeError(
            "Pas assez de colonnes communes entre generated et COCONUT.\n"
            f"Commun comp1 : {len(common1)} / {len(comp1_cols)}\n"
            f"Commun comp2 : {len(common2)} / {len(comp2_cols)}"
        )

    print(f"[Step2] Single-molecule mode: comp1={len(common1)}, comp2={len(common2)}")

    gen1_cols = [x[0] for x in common1]
    coco1_cols = [x[1] for x in common1]

    gen2_cols = [x[0] for x in common2]
    coco2_cols = [x[1] for x in common2]

    coco1, med1 = safe_numeric(coco, coco1_cols)
    coco2, med2 = safe_numeric(coco, coco2_cols)

    gen1 = gen[gen1_cols].copy()
    gen2 = gen[gen2_cols].copy()

    gen1.columns = coco1_cols
    gen2.columns = coco2_cols

    gen1 = clean_numeric_matrix(gen1, med=med1)
    gen2 = clean_numeric_matrix(gen2, med=med2)

    scaler1 = StandardScaler().fit(coco1.values)
    scaler2 = StandardScaler().fit(coco2.values)

    Xc1 = scaler1.transform(coco1.values)
    Xg1 = scaler1.transform(gen1.values)

    Xc2 = scaler2.transform(coco2.values)
    Xg2 = scaler2.transform(gen2.values)

    Xc1 = np.nan_to_num(Xc1, nan=0.0, posinf=0.0, neginf=0.0)
    Xg1 = np.nan_to_num(Xg1, nan=0.0, posinf=0.0, neginf=0.0)
    Xc2 = np.nan_to_num(Xc2, nan=0.0, posinf=0.0, neginf=0.0)
    Xg2 = np.nan_to_num(Xg2, nan=0.0, posinf=0.0, neginf=0.0)

    nn1 = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(Xc1)
    nn2 = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(Xc2)

    dist1, idx1 = nn1.kneighbors(Xg1)
    dist2, idx2 = nn2.kneighbors(Xg2)

    best1 = idx1[:, 0]
    best2 = idx2[:, 0]

    out = gen.copy()

    for gc, bc in common1:
        out[gc] = coco.iloc[best1][bc].to_numpy()

    for gc, bc in common2:
        out[gc] = coco.iloc[best2][bc].to_numpy()

    out["coconut_index_comp1"] = best1
    out["coconut_index_comp2"] = best2
    out["coconut_distance_comp1"] = dist1[:, 0]
    out["coconut_distance_comp2"] = dist2[:, 0]
    out["coconut_distance_sum"] = dist1[:, 0] + dist2[:, 0]

    if smiles_col is not None:
        out.insert(0, "Smiles#1", coco.iloc[best1][smiles_col].to_numpy())
        out.insert(1, "Smiles#2", coco.iloc[best2][smiles_col].to_numpy())

    for c in id_cols[:5]:
        out[f"COCONUT_comp1_{c}"] = coco.iloc[best1][c].to_numpy()
        out[f"COCONUT_comp2_{c}"] = coco.iloc[best2][c].to_numpy()

    # Exclude SMILES pairs already present in the training set,
    # then keep the n_keep best pairs based on total distance.
    out = filter_existing_training_pairs(
        out,
        train_xlsx=train_xlsx,
        train_smiles_xlsx=train_smiles_xlsx,
        exclude_existing_pairs=exclude_existing_pairs
    )
    out = keep_best_by_distance(
        out,
        distance_col="coconut_distance_sum",
        n_keep=n_keep
    )

    matched_path = outdir / "matched_real_from_coconut_mp.xlsx"
    out.to_excel(matched_path, index=False)

    print("[Step2] Matched file:", matched_path)
    return matched_path
def step3_evaluate_models_train_real_validate_external(
    train_xlsx: Path,
    external_xlsx: Path,
    outdir: Path,
    test_size=0.2,
    cv_splits=5,
    n_iter=30,
    random_state=2026
):
    seed_everything(random_state, deterministic=True)
    print("[Step3] Real train/test split + external COCONUT validation")

    real_df = pd.read_excel(train_xlsx)
    ext_df = pd.read_excel(external_xlsx)

    X_real_raw, y_real_raw, tgt_name = detect_columns(real_df)
    X_ext_raw, y_ext_raw, _ = detect_columns(ext_df)

    common = X_real_raw.columns.intersection(X_ext_raw.columns)

    if len(common) == 0:
        raise RuntimeError("No common column between real and external data.")

    X_real_raw = X_real_raw[common].copy()
    X_ext_raw = X_ext_raw[common].copy()

    y_real = y_real_raw.copy()
    y_ext = y_ext_raw.copy()

    X_real_imp, med = safe_numeric(X_real_raw, X_real_raw.columns)
    X_ext_imp = clean_numeric_matrix(X_ext_raw, med=med)

    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X_real_imp,
        y_real,
        test_size=test_size,
        random_state=random_state
    )

    # Common standardization for all models.
    # The scaler is fitted only on the real training set.
    sX = StandardScaler().fit(X_tr_raw.values)

    X_tr_s = sX.transform(X_tr_raw.values)
    X_te_s = sX.transform(X_te_raw.values)
    X_ext_s = sX.transform(X_ext_imp.values)

    models = {}

    try:
        from xgboost import XGBRegressor

        xgb = XGBRegressor(
            objective="reg:squarederror",
            random_state=random_state,
            n_estimators=500,
            n_jobs=-1,
            tree_method="hist"
        )

        xgb_dist = {
            "n_estimators": randint(300, 1200),
            "max_depth": randint(3, 10),
            "learning_rate": loguniform(1e-3, 2e-1),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "reg_lambda": loguniform(1e-2, 1e2),
        }

        models["XGBoost"] = (xgb, xgb_dist)

    except Exception as e:
        print("[WARN] xgboost non disponible:", e)

    rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)

    rf_dist = {
        "n_estimators": randint(300, 1200),
        "max_depth": randint(5, 40),
        "min_samples_leaf": randint(1, 6),
        "max_features": ["sqrt", "log2", None],
    }

    models["RandomForest"] = (rf, rf_dist)

    svr = SVR()

    svr_dist = {
        "C": loguniform(1e-1, 1e3),
        "epsilon": loguniform(1e-3, 1e0),
        "gamma": ["scale", "auto"],
        "kernel": ["rbf"],
    }

    models["SVR"] = (svr, svr_dist)

    mlp = MLPRegressor(
        max_iter=1000,
        early_stopping=True,
        random_state=random_state
    )

    mlp_dist = {
        "hidden_layer_sizes": [
            (256, 128, 64),
            (256, 128),
            (128, 64),
            (512, 256, 128),
        ],
        "alpha": loguniform(1e-6, 1e-2),
        "learning_rate_init": loguniform(1e-4, 5e-2),
        "activation": ["relu", "tanh"],
    }

    models["MLP"] = (mlp, mlp_dist)

    kf = KFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state
    )

    results = []

    for name, (estimator, dist) in models.items():
        print(f"[Step3] Tuning {name}")

        Xtr_fit = X_tr_s
        Xte_fit = X_te_s
        Xva_fit = X_ext_s

        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=dist,
            n_iter=n_iter,
            cv=kf,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            random_state=random_state,
            verbose=0
        )

        search.fit(Xtr_fit, y_tr.values)
        best = search.best_estimator_

        y_tr_pred = best.predict(Xtr_fit)
        y_te_pred = best.predict(Xte_fit)
        y_va_pred = best.predict(Xva_fit)

        metrics = {
            "model": name,
            "best_params": str(search.best_params_),
            "Train_R2": float(r2_score(y_tr, y_tr_pred)),
            "Train_RMSE": rmse(y_tr, y_tr_pred),
            "Test_R2": float(r2_score(y_te, y_te_pred)),
            "Test_RMSE": rmse(y_te, y_te_pred),
            "Val_R2": float(r2_score(y_ext, y_va_pred)),
            "Val_RMSE": rmse(y_ext, y_va_pred),
        }

        results.append(metrics)

        plt.figure(figsize=(5.8, 5.8))

        plt.scatter(y_tr, y_tr_pred, s=12, alpha=0.45, label="Training set")
        plt.scatter(y_te, y_te_pred, s=18, alpha=0.65, marker="D", label="Test set")
        plt.scatter(y_ext, y_va_pred, s=24, alpha=0.85, marker="^", label="COCONUT validation")

        lim_min = min(y_tr.min(), y_te.min(), y_ext.min())
        lim_max = max(y_tr.max(), y_te.max(), y_ext.max())

        plt.plot([lim_min, lim_max], [lim_min, lim_max], "--", linewidth=1.2)

        plt.xlabel("Experimental melting point (K)")
        plt.ylabel("Predicted melting point (K)")
        plt.title(rf"COCONUT validation: $R^2$={metrics['Val_R2']:.3f}, RMSE={metrics['Val_RMSE']:.3f} K")
        plt.legend(frameon=False)
        plt.tight_layout()

        fname = outdir / f"external_scatter_{name}_mp.png"
        plt.savefig(fname, dpi=160)
        plt.close()

        joblib.dump(best, outdir / f"best_{name}_mp.joblib")

    rep = pd.DataFrame(results).sort_values("Val_RMSE")
    rep_path = outdir / "external_eval_best_models_mp.csv"
    rep.to_csv(rep_path, index=False)

    joblib.dump(sX, outdir / "scaler_features_real_train_mp.pkl")

    print("[Step3] Metrics saved:", rep_path)
    return rep_path


def detect_smiles_pair_cols(df: pd.DataFrame):
    cols = list(df.columns)

    def norm(s):
        return str(s).lower().replace(" ", "").replace("-", "")

    norm_map = {norm(c): c for c in cols}

    candidates_1 = [
        "smiles#1",
        "smiles_1",
        "smiles1",
        "smilescomp1",
        "smiles_comp1",
    ]

    candidates_2 = [
        "smiles#2",
        "smiles_2",
        "smiles2",
        "smilescomp2",
        "smiles_comp2",
    ]

    c1 = None
    c2 = None

    for x in candidates_1:
        if x in norm_map:
            c1 = norm_map[x]
            break

    for x in candidates_2:
        if x in norm_map:
            c2 = norm_map[x]
            break

    if c1 is None or c2 is None:
        smiles_like = [c for c in cols if "smiles" in str(c).lower()]
        if len(smiles_like) >= 2:
            smiles_like = sorted(smiles_like, key=lambda z: str(z).lower())
            c1 = smiles_like[0]
            c2 = smiles_like[1]

    if c1 is None or c2 is None:
        return None, None

    return c1, c2


def load_training_with_pair_keys(
    train_xlsx: Path,
    train_smiles_xlsx: Path = None,
    dropna=True
):
    """
    Loads the descriptor file used for training and adds SMILES pairs
    from a second file aligned row-by-row.

    Expected case here:
    - train_xlsx = DES_selected_64_features_original_values.xlsx
      contains the descriptors + Melting_point_K, but not the SMILES;
    - train_smiles_xlsx = Melting_point_selected_ind_descriptors____.xlsx
      contains SMILES_1 / SMILES_2 in the same order as train_xlsx.

    If train_xlsx already contains the two SMILES columns, train_smiles_xlsx is optional.
    """
    train_df = pd.read_excel(train_xlsx).reset_index(drop=True)

    c1_train, c2_train = detect_smiles_pair_cols(train_df)

    if c1_train is not None and c2_train is not None:
        df = train_df.copy()
        c1, c2 = c1_train, c2_train
    else:
        if train_smiles_xlsx is None:
            raise RuntimeError(
                "train_xlsx does not contain SMILES pairs. "
                "You must provide --train_smiles_xlsx with a row-aligned file."
            )

        smiles_df = pd.read_excel(train_smiles_xlsx).reset_index(drop=True)
        c1, c2 = detect_smiles_pair_cols(smiles_df)
        if c1 is None or c2 is None:
            raise RuntimeError(
                "Unable to detect SMILES_1 / SMILES_2 in train_smiles_xlsx."
            )

        if len(train_df) != len(smiles_df):
            raise RuntimeError(
                f"The two files cannot be aligned: "
                f"train_xlsx contains {len(train_df)} rows, "
                f"train_smiles_xlsx contains {len(smiles_df)} rows."
            )

        df = train_df.copy()
        # Only add the SMILES columns to the descriptor file.
        # The row order is assumed to be identical between the two files.
        df["SMILES_1"] = smiles_df[c1].values
        df["SMILES_2"] = smiles_df[c2].values
        c1, c2 = "SMILES_1", "SMILES_2"

    df["pair_key"] = [canonical_pair(a, b) for a, b in zip(df[c1], df[c2])]

    if dropna:
        df = df.dropna().reset_index(drop=True)

    return df, c1, c2



def keep_best_by_distance(df: pd.DataFrame, distance_col: str, n_keep=100):
    """Sorts candidates by increasing distance and keeps the n_keep best ones."""
    if n_keep is None:
        return df.reset_index(drop=True)

    if distance_col not in df.columns:
        print(f"[WARN] Distance column missing: {distance_col}. No n_keep filtering.")
        return df.reset_index(drop=True)

    n_keep = int(n_keep)
    if n_keep <= 0:
        raise ValueError("n_keep must be strictly positive.")

    out = df.copy()
    out[distance_col] = pd.to_numeric(out[distance_col], errors="coerce")
    out = out.sort_values(distance_col, ascending=True, na_position="last")
    out = out.head(min(n_keep, len(out))).reset_index(drop=True)
    return out


def filter_existing_training_pairs(
    matched_df: pd.DataFrame,
    train_xlsx: Path = None,
    train_smiles_xlsx: Path = None,
    exclude_existing_pairs=True
):
    """
    Removes rows where the pair (Smiles#1, Smiles#2) already exists
    in the VAE training set.

    The comparison is symmetric: A-B is identical to B-A.
    """
    if not exclude_existing_pairs:
        return matched_df.reset_index(drop=True)

    if train_xlsx is None:
        print("[WARN] train_xlsx is missing: existing-pair exclusion skipped.")
        return matched_df.reset_index(drop=True)

    c1_m, c2_m = detect_smiles_pair_cols(matched_df)
    if c1_m is None or c2_m is None:
        print("[WARN] No SMILES pair detected in matched: exclusion skipped.")
        return matched_df.reset_index(drop=True)

    try:
        train_df, c1_t, c2_t = load_training_with_pair_keys(
            train_xlsx=train_xlsx,
            train_smiles_xlsx=train_smiles_xlsx,
            dropna=False
        )
    except Exception as e:
        print(f"[WARN] Unable to load training pairs: {e}. Exclusion skipped.")
        return matched_df.reset_index(drop=True)

    train_pairs = {
        canonical_pair(a, b)
        for a, b in zip(train_df[c1_t], train_df[c2_t])
        if not (pd.isna(a) or pd.isna(b))
    }

    out = matched_df.copy()
    out["pair_key"] = [canonical_pair(a, b) for a, b in zip(out[c1_m], out[c2_m])]
    out["pair_already_in_train"] = out["pair_key"].isin(train_pairs)

    n_before = len(out)
    out = out.loc[~out["pair_already_in_train"]].reset_index(drop=True)
    n_removed = n_before - len(out)

    print(f"[Step2] Pairs already present in training removed: {n_removed} / {n_before}")

    return out


def step4_check_existing_pairs(matched_xlsx: Path, train_xlsx: Path, outdir: Path, train_smiles_xlsx: Path = None):
    print("[Step4] Check existing SMILES pairs")

    df1 = pd.read_excel(matched_xlsx)
    try:
        df2, _, _ = load_training_with_pair_keys(
            train_xlsx=train_xlsx,
            train_smiles_xlsx=train_smiles_xlsx,
            dropna=False
        )
    except Exception as e:
        print(f"[WARN] Unable to load training pairs: {e}. Step 4 skipped.")
        return None

    c1_f1, c2_f1 = detect_smiles_pair_cols(df1)
    c1_f2, c2_f2 = detect_smiles_pair_cols(df2)

    if c1_f1 is None or c2_f1 is None:
        print("[WARN] No SMILES pairs detected in matched. Step 4 skipped.")
        return None

    if c1_f2 is None or c2_f2 is None:
        print("[WARN] No SMILES pairs detected in training. Step 4 skipped.")
        return None

    def prep(df, tag, c1, c2):
        out = df.copy()
        out[f"row_index_{tag}"] = out.index + 2
        out["pair_key"] = [canonical_pair(a, b) for a, b in zip(out[c1], out[c2])]
        out["Smiles#1_std"] = out[c1]
        out["Smiles#2_std"] = out[c2]
        return out

    f1 = prep(df1, "matched", c1_f1, c2_f1)
    f2 = prep(df2, "train", c1_f2, c2_f2)

    matches = f1.merge(
        f2,
        on="pair_key",
        how="inner",
        suffixes=("_matched", "_train")
    )

    stats = pd.DataFrame([
        ("Rows matched", len(f1)),
        ("Rows train", len(f2)),
        ("Unique pairs matched", f1["pair_key"].nunique()),
        ("Unique pairs train", f2["pair_key"].nunique()),
        ("Unique pairs in both", matches["pair_key"].nunique() if not matches.empty else 0),
    ], columns=["Metric", "Value"])

    s1 = f1.groupby("pair_key", as_index=False).size().rename(columns={"size": "count_matched"})
    s2 = f2.groupby("pair_key", as_index=False).size().rename(columns={"size": "count_train"})

    pair_summary = s1.merge(s2, on="pair_key", how="outer").fillna(0)
    pair_summary["count_matched"] = pair_summary["count_matched"].astype(int)
    pair_summary["count_train"] = pair_summary["count_train"].astype(int)
    pair_summary["is_in_both"] = (
        (pair_summary["count_matched"] > 0)
        & (pair_summary["count_train"] > 0)
    )

    out_xlsx = outdir / "mp_results_smiles_pairs.xlsx"

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        matches.to_excel(writer, sheet_name="matches_locations", index=False)
        stats.to_excel(writer, sheet_name="general_stats", index=False)
        pair_summary.to_excel(writer, sheet_name="pair_summary", index=False)
        f1.to_excel(writer, sheet_name="matched_with_keys", index=False)
        f2.to_excel(writer, sheet_name="training_with_keys", index=False)

    print("[Step4] Report saved:", out_xlsx)
    return out_xlsx




def make_masked_des_split(
    train_xlsx: Path,
    outdir: Path,
    train_smiles_xlsx: Path = None,
    hidden_fraction=0.10,
    random_state=2026,
    stratify_target=True
):
    """
    Creates a hidden-DES validation split by UNIQUE PAIR, even if the file
    used for training does not directly contain SMILES.

    Expected usage:
    - train_xlsx: descriptor file used by the SVAE
      ex. DES_selected_64_features_original_values.xlsx
    - train_smiles_xlsx: file containing SMILES_1 / SMILES_2 in the same order
      ex. Melting_point_selected_ind_descriptors____.xlsx

    All rows from the same SMILES pair are assigned either to the training set,
    or to the hidden DES set. This prevents a pair from appearing in training with
    one X_1 / X_2 composition and in validation with another composition.
    """
    print("[HiddenValidation] Creating the hidden-DES split by unique pairs")

    df, c1, c2 = load_training_with_pair_keys(
        train_xlsx=train_xlsx,
        train_smiles_xlsx=train_smiles_xlsx,
        dropna=True
    )

    X, y, target_name = detect_columns(df)

    pair_table = (
        df.groupby("pair_key")
        .agg(
            n_rows=("pair_key", "size"),
            target_mean=(target_name, "mean"),
            target_min=(target_name, "min"),
            target_max=(target_name, "max"),
            smiles_1=(c1, "first"),
            smiles_2=(c2, "first"),
        )
        .reset_index()
    )

    strat = None
    if stratify_target:
        try:
            n_pairs = len(pair_table)
            q = min(10, max(2, int(n_pairs * hidden_fraction)))
            strat = pd.qcut(
                pair_table["target_mean"],
                q=q,
                labels=False,
                duplicates="drop"
            )
            if pd.Series(strat).nunique() < 2:
                strat = None
        except Exception as e:
            print("[WARN] Target-based stratification skipped:", e)
            strat = None

    train_pairs, hidden_pairs = train_test_split(
        pair_table["pair_key"].values,
        test_size=float(hidden_fraction),
        random_state=random_state,
        shuffle=True,
        stratify=strat
    )

    train_pairs = set(train_pairs)
    hidden_pairs = set(hidden_pairs)

    masked_train = df.loc[df["pair_key"].isin(train_pairs)].copy()
    hidden = df.loc[df["pair_key"].isin(hidden_pairs)].copy()

    overlap = set(masked_train["pair_key"]) & set(hidden["pair_key"])
    if len(overlap) > 0:
        raise RuntimeError(
            f"Split error: {len(overlap)} pairs are present "
            "both in the masked training set and in the hidden DES set."
        )

    hidden_in_train = hidden["pair_key"].isin(set(masked_train["pair_key"])).sum()
    if hidden_in_train > 0:
        raise RuntimeError(
            f"Leakage detected: {hidden_in_train} hidden rows have a pair present in the training set."
        )

    split_dir = outdir / "hidden_des_validation"
    ensure_outdir(split_dir)

    masked_train_path = split_dir / "masked_train_DES.xlsx"
    hidden_path = split_dir / "hidden_DES_test.xlsx"
    pair_split_report_path = split_dir / "hidden_pair_split_report.xlsx"

    # These files contain descriptors + SMILES_1 / SMILES_2 + pair_key.
    # detect_columns() will automatically ignore non-numeric columns.
    masked_train.to_excel(masked_train_path, index=False)
    hidden.to_excel(hidden_path, index=False)

    pair_table["split"] = np.where(
        pair_table["pair_key"].isin(hidden_pairs),
        "hidden_DES",
        "masked_train"
    )
    pair_table.to_excel(pair_split_report_path, index=False)

    print(f"[HiddenValidation] Total pairs: {len(pair_table)}")
    print(f"[HiddenValidation] Training pairs: {len(train_pairs)}")
    print(f"[HiddenValidation] Hidden pairs: {len(hidden_pairs)}")
    print(f"[HiddenValidation] Masked training rows: {masked_train.shape}")
    print(f"[HiddenValidation] Hidden DES rows: {hidden.shape}")
    print("[HiddenValidation] Masked training file:", masked_train_path)
    print("[HiddenValidation] Hidden DES file:", hidden_path)
    print("[HiddenValidation] Pair split report:", pair_split_report_path)

    return masked_train_path, hidden_path, target_name

def step_hidden_des_recovery_validation(
    generated_xlsx: Path,
    hidden_xlsx: Path,
    masked_train_xlsx: Path,
    outdir: Path,
    topk_values=(1, 5, 10, 20, 50)
):
    """
    Validation 1: hidden-DES recovery.

    COCONUT matching is NOT used here.
    Generated candidates are directly compared with the hidden DES.
    For each hidden DES, the nearest generated candidate is searched
    in the standardized descriptor space using the scaler fitted on the masked training set.
    """
    print("[HiddenValidation] Direct matching: generated candidates ↔ hidden DES")

    gen_df = pd.read_excel(generated_xlsx)
    hidden_df = pd.read_excel(hidden_xlsx)
    train_df = pd.read_excel(masked_train_xlsx)

    X_train, y_train, target_name = detect_columns(train_df)
    X_hidden, y_hidden, _ = detect_columns(hidden_df)
    X_gen, y_gen, _ = detect_columns(gen_df)

    common = X_train.columns.intersection(X_hidden.columns).intersection(X_gen.columns)
    if len(common) == 0:
        raise RuntimeError("No common column among masked training, hidden DES, and generated data.")

    X_train = X_train[common].copy()
    X_hidden = X_hidden[common].copy()
    X_gen = X_gen[common].copy()

    X_train_imp, med = safe_numeric(X_train, common)
    X_hidden_imp = clean_numeric_matrix(X_hidden, med=med)
    X_gen_imp = clean_numeric_matrix(X_gen, med=med)

    scaler = StandardScaler().fit(X_train_imp.values)
    Xt = scaler.transform(X_train_imp.values)
    Xh = scaler.transform(X_hidden_imp.values)
    Xg = scaler.transform(X_gen_imp.values)

    Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)
    Xh = np.nan_to_num(Xh, nan=0.0, posinf=0.0, neginf=0.0)
    Xg = np.nan_to_num(Xg, nan=0.0, posinf=0.0, neginf=0.0)

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(Xg)
    dist, idx = nn.kneighbors(Xh)
    nearest_gen_idx = idx[:, 0]
    nearest_dist = dist[:, 0]

    rows = []
    c1_h, c2_h = detect_smiles_pair_cols(hidden_df)
    c1_g, c2_g = detect_smiles_pair_cols(gen_df)

    for i in range(len(hidden_df)):
        j = int(nearest_gen_idx[i])
        t_real = float(y_hidden.iloc[i])
        t_gen = float(y_gen.iloc[j])
        row = {
            "hidden_row_index": i,
            "nearest_generated_row_index": j,
            "descriptor_distance": float(nearest_dist[i]),
            f"hidden_real_{target_name}": t_real,
            f"generated_target_{target_name}": t_gen,
            "absolute_error_K": abs(t_gen - t_real),
            "signed_error_K": t_gen - t_real,
        }
        if c1_h is not None and c2_h is not None:
            row["hidden_Smiles#1"] = hidden_df.iloc[i][c1_h]
            row["hidden_Smiles#2"] = hidden_df.iloc[i][c2_h]
        if c1_g is not None and c2_g is not None:
            row["generated_Smiles#1"] = gen_df.iloc[j][c1_g]
            row["generated_Smiles#2"] = gen_df.iloc[j][c2_g]
        rows.append(row)

    recovery = pd.DataFrame(rows).sort_values("descriptor_distance").reset_index(drop=True)

    # Global summary.
    y_true = recovery[f"hidden_real_{target_name}"].values
    y_pred = recovery[f"generated_target_{target_name}"].values

    hidden_R2 = float(r2_score(y_true, y_pred))

    global_summary = pd.DataFrame([{
        "n_hidden_DES": len(recovery),
        "n_generated": len(gen_df),
        "n_common_descriptors": len(common),
        "R2": hidden_R2,
        "MAE_K": float(np.mean(np.abs(y_pred - y_true))),
        "RMSE_K": rmse(y_true, y_pred),
        "median_descriptor_distance": float(np.median(recovery["descriptor_distance"])),
        "mean_descriptor_distance": float(np.mean(recovery["descriptor_distance"])),
    }])

    # Summary of the best recovery cases: top-k best recovered hidden DES.
    top_rows = []
    for k in topk_values:
        k = int(k)
        if k <= 0:
            continue
        sub = recovery.head(min(k, len(recovery)))
        yt = sub[f"hidden_real_{target_name}"].values
        yp = sub[f"generated_target_{target_name}"].values
        topk_R2 = float(r2_score(yt, yp)) if len(yt) >= 2 else np.nan

        top_rows.append({
            "top_k_best_recovered_hidden_DES": min(k, len(recovery)),
            "mean_distance": float(sub["descriptor_distance"].mean()),
            "median_distance": float(sub["descriptor_distance"].median()),
            "R2": topk_R2,
            "MAE_K": float(np.mean(np.abs(yp - yt))),
            "RMSE_K": rmse(yt, yp),
        })
    topk_summary = pd.DataFrame(top_rows)

    recovery_dir = outdir / "hidden_des_validation"
    ensure_outdir(recovery_dir)

    recovery_xlsx = recovery_dir / "hidden_DES_recovery_report.xlsx"
    summary_csv = recovery_dir / "hidden_DES_recovery_summary.csv"
    topk_csv = recovery_dir / "hidden_DES_topk_summary.csv"

    with pd.ExcelWriter(recovery_xlsx, engine="openpyxl") as writer:
        recovery.to_excel(writer, sheet_name="nearest_generated", index=False)
        global_summary.to_excel(writer, sheet_name="global_summary", index=False)
        topk_summary.to_excel(writer, sheet_name="topk_summary", index=False)
        hidden_df.to_excel(writer, sheet_name="hidden_DES", index=False)
        gen_df.to_excel(writer, sheet_name="generated_candidates", index=False)

    global_summary.to_csv(summary_csv, index=False)
    topk_summary.to_csv(topk_csv, index=False)

    # Figure: experimental vs nearest generated value.
    try:
        plt.figure(figsize=(5.8, 5.8))
        plt.scatter(y_true, y_pred, s=22, alpha=0.75)
        lim_min = min(np.min(y_true), np.min(y_pred))
        lim_max = max(np.max(y_true), np.max(y_pred))
        plt.plot([lim_min, lim_max], [lim_min, lim_max], "--", linewidth=1.2)
        plt.xlabel("Experimental melting point (K)")
        plt.ylabel("Generated melting point (K)")
        plt.title(rf"Hidden DES recovery: $R^2$={hidden_R2:.3f}, RMSE={rmse(y_true, y_pred):.3f} K")
        plt.tight_layout()
        plt.savefig(recovery_dir / "hidden_DES_recovery_true_vs_generated.png", dpi=160)
        plt.close()
    except Exception as e:
        print("[WARN] Hidden recovery figure skipped:", e)

    print("[HiddenValidation] Report:", recovery_xlsx)
    print("[HiddenValidation] Summary:", summary_csv)

    return recovery_xlsx, summary_csv, topk_csv


def main():
    parser = argparse.ArgumentParser(
        description="Supervised VAE for melting point with hidden-DES validation and COCONUT matching"
    )

    parser.add_argument("--train_xlsx", type=str, default=DEFAULTS["train_xlsx"])
    parser.add_argument(
        "--train_smiles_xlsx",
        type=str,
        default=DEFAULTS["train_smiles_xlsx"],
        help=(
            "File containing SMILES_1 / SMILES_2, aligned row-by-row with train_xlsx. "
            "Used to create hidden DES by pair and exclude already seen pairs."
        )
    )
    parser.add_argument("--coconut_xlsx", type=str, default=DEFAULTS["coconut_xlsx"])
    parser.add_argument("--coconut_sheet", type=str, default=DEFAULTS["coconut_sheet"])
    parser.add_argument("--outdir", type=str, default=DEFAULTS["outdir"])

    parser.add_argument("--n_gen", type=int, default=200)
    parser.add_argument("--n_gen_hidden", type=int, default=5000,
                        help="Number of generated candidates for hidden-DES validation.")
    parser.add_argument("--n_gen_coconut", type=int, default=None,
                        help="Number of generated candidates for the COCONUT phase. If omitted, --n_gen is used.")
    parser.add_argument("--hidden_fraction", type=float, default=0.10,
                        help="Fraction of hidden DES for masked validation.")
    parser.add_argument("--random_state", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=2026,
                        help="Global seed used for numpy, Python, PyTorch, train/test splits and generation.")
    parser.add_argument("--force_cpu", action="store_true", default=False,
                        help="Force CPU execution. Recommended when exact reproducibility is required.")
    parser.add_argument("--non_deterministic", action="store_true", default=False,
                        help="Disable strict deterministic PyTorch settings.")

    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=2.5)
    parser.add_argument("--tmin", type=float, default=None)
    parser.add_argument("--tmax", type=float, default=None)
    parser.add_argument("--n_neighbors", type=int, default=1)
    parser.add_argument("--n_keep", type=int, default=100)

    parser.add_argument("--run_hidden_validation", action="store_true", default=True,
                        help="Run hidden-DES validation.")
    parser.add_argument("--skip_hidden_validation", action="store_false", dest="run_hidden_validation",
                        help="Disable hidden-DES validation.")
    parser.add_argument("--run_coconut_validation", action="store_true", default=True,
                        help="Run COCONUT validation/discovery.")
    parser.add_argument("--skip_coconut_validation", action="store_false", dest="run_coconut_validation",
                        help="Disable COCONUT validation/discovery.")

    parser.add_argument(
        "--exclude_existing_pairs",
        action="store_true",
        default=True,
        help="Exclude from the matched file the SMILES pairs already present in train_xlsx."
    )
    parser.add_argument(
        "--allow_existing_pairs",
        action="store_false",
        dest="exclude_existing_pairs",
        help="Disable exclusion of SMILES pairs already present in train_xlsx."
    )

    args = parser.parse_args()

    seed_everything(args.seed, deterministic=not args.non_deterministic)

    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    summary = {
        "train_xlsx": args.train_xlsx,
        "train_smiles_xlsx": args.train_smiles_xlsx,
        "coconut_xlsx": args.coconut_xlsx,
        "coconut_sheet": args.coconut_sheet,
        "seed": args.seed,
        "deterministic": not args.non_deterministic,
        "force_cpu": args.force_cpu,
        "hidden_validation": None,
        "coconut_validation": None,
    }

    # ============================================================
    # Validation 1: hidden DES
    # ============================================================
    if args.run_hidden_validation:
        hidden_outdir = outdir / "hidden_des_validation"
        ensure_outdir(hidden_outdir)

        masked_train_path, hidden_path, hidden_target_name = make_masked_des_split(
            train_xlsx=Path(args.train_xlsx),
            outdir=outdir,
            train_smiles_xlsx=Path(args.train_smiles_xlsx),
            hidden_fraction=args.hidden_fraction,
            random_state=args.random_state,
            stratify_target=True
        )

        hidden_gen_path, _ = step1_train_and_generate(
            train_xlsx=masked_train_path,
            outdir=hidden_outdir,
            n_gen=args.n_gen_hidden,
            latent_dim=args.latent_dim,
            epochs=args.epochs,
            batch=args.batch,
            lr=args.lr,
            beta=args.beta,
            gamma=args.gamma,
            tmin=args.tmin,
            tmax=args.tmax,
            seed=args.seed,
            deterministic=not args.non_deterministic,
            force_cpu=args.force_cpu
        )

        recovery_xlsx, recovery_summary_csv, recovery_topk_csv = step_hidden_des_recovery_validation(
            generated_xlsx=hidden_gen_path,
            hidden_xlsx=hidden_path,
            masked_train_xlsx=masked_train_path,
            outdir=outdir,
            topk_values=(1, 5, 10, 20, 50)
        )

        summary["hidden_validation"] = {
            "masked_train_file": str(masked_train_path),
            "hidden_test_file": str(hidden_path),
            "train_smiles_file": args.train_smiles_xlsx,
            "hidden_generated_file": str(hidden_gen_path),
            "hidden_recovery_report": str(recovery_xlsx),
            "hidden_recovery_summary_csv": str(recovery_summary_csv),
            "hidden_recovery_topk_csv": str(recovery_topk_csv),
            "hidden_fraction": args.hidden_fraction,
            "n_gen_hidden": args.n_gen_hidden,
            "target_name": hidden_target_name,
        }

    # ============================================================
    # Validation 2: COCONUT / real discovery
    # ============================================================
    if args.run_coconut_validation:
        coconut_outdir = outdir / "coconut_validation"
        ensure_outdir(coconut_outdir)

        n_gen_coconut = args.n_gen if args.n_gen_coconut is None else args.n_gen_coconut

        gen_path, target_name = step1_train_and_generate(
            train_xlsx=Path(args.train_xlsx),
            outdir=coconut_outdir,
            n_gen=n_gen_coconut,
            latent_dim=args.latent_dim,
            epochs=args.epochs,
            batch=args.batch,
            lr=args.lr,
            beta=args.beta,
            gamma=args.gamma,
            tmin=args.tmin,
            tmax=args.tmax,
            seed=args.seed,
            deterministic=not args.non_deterministic,
            force_cpu=args.force_cpu
        )

        matched_path = step2_match_to_coconut(
            generated_xlsx=gen_path,
            coconut_xlsx=Path(args.coconut_xlsx),
            outdir=coconut_outdir,
            train_xlsx=Path(args.train_xlsx),
            train_smiles_xlsx=Path(args.train_smiles_xlsx),
            coconut_sheet=args.coconut_sheet,
            n_neighbors=args.n_neighbors,
            n_keep=args.n_keep,
            exclude_existing_pairs=args.exclude_existing_pairs
        )

        metrics_csv = step3_evaluate_models_train_real_validate_external(
            train_xlsx=Path(args.train_xlsx),
            external_xlsx=matched_path,
            outdir=coconut_outdir,
            random_state=args.seed
        )

        try:
            overlap_xlsx = step4_check_existing_pairs(
                matched_xlsx=matched_path,
                train_xlsx=Path(args.train_xlsx),
                train_smiles_xlsx=Path(args.train_smiles_xlsx),
                outdir=coconut_outdir
            )
        except Exception as e:
            print("[WARN] Step 4 skipped:", e)
            overlap_xlsx = None

        summary["coconut_validation"] = {
            "generated_file": str(gen_path),
            "matched_coconut_file": str(matched_path),
            "train_smiles_file": args.train_smiles_xlsx,
            "ml_metrics_csv": str(metrics_csv),
            "smiles_pairs_report": str(overlap_xlsx),
            "target_name": target_name,
            "n_gen_coconut": n_gen_coconut,
            "n_keep": args.n_keep,
            "exclude_existing_pairs": args.exclude_existing_pairs,
        }

    summary_path = outdir / "pipeline_summary_melting_two_validations.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[DONE] Summary:", summary_path)


if __name__ == "__main__":
    main()
