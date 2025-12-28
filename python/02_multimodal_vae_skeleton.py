"""Multi-omics integration demo: linear latent baseline + optional multi-modal VAE.

What it does
- Loads `data/toy_multiomics_4patients.csv`.
- Builds 3 aligned blocks (microbiome, metabolomics, transcriptomics) + a group label.
- Applies practical preprocessing:
  - Microbiome is compositional -> uses ALR log-ratios (relative to BugD).
  - Metabolites/genes -> log1p transform.
  - Standardizes each block.
- Runs one of two models:
  1) Linear shared-latent baseline (always available; numpy only):
     - concatenate blocks -> PCA latent Z -> linear reconstruction of each block from Z
  2) Multi-modal VAE (optional; requires torch):
     - one encoder per block -> shared latent Z -> one decoder per block

Why the script simulates more samples
With only 4 patients, training any flexible model is unstable.
This script can create a small synthetic dataset by adding noise around the 4 toy patients,
so you can see the mechanics. The toy 4 patients are still used for reporting.

Run
  # linear baseline (no torch needed)
  python python/02_multimodal_vae_skeleton.py

  # train a small multi-modal VAE (requires torch installed)
  python python/02_multimodal_vae_skeleton.py --vae --epochs 300

  # label-conditional decoder (useful for representation/imputation; not a "discovery")
  python python/02_multimodal_vae_skeleton.py --vae --conditional-decoder

Notes on interpretation
- These models find *associations* / shared structure. They do not establish direction or causality.
- If you feed the group label into the model ("conditional"), separation by group can be partly
  because you gave the label as input.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

MICROBE_COLS = ["BugA", "BugB", "BugC", "BugD"]
MET_COLS = ["Met1", "Met2", "Met3"]
GENE_COLS = ["Gene1", "Gene2", "Gene3"]


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_toy_df() -> pd.DataFrame:
    path = os.path.join(repo_root(), "data", "toy_multiomics_4patients.csv")
    return pd.read_csv(path)


def alr_log_ratios(df: pd.DataFrame, denom: str = "BugD", pseudocount: float = 1e-6) -> np.ndarray:
    """ALR: log(BugX / denom) for X in {BugA, BugB, BugC}.

    Returns shape (n, 3).
    """
    denom_v = df[denom].to_numpy(dtype=float) + pseudocount
    ratios = []
    for col in ["BugA", "BugB", "BugC"]:
        num = df[col].to_numpy(dtype=float) + pseudocount
        ratios.append(np.log(num / denom_v))
    return np.column_stack(ratios)


def invert_alr_to_composition(r: np.ndarray) -> np.ndarray:
    """Inverse of ALR relative to BugD.

    Input r shape: (n, 3) for log(BugA/BugD), log(BugB/BugD), log(BugC/BugD)
    Output shape: (n, 4) for [BugA, BugB, BugC, BugD] summing to 1.
    """
    exp_r = np.exp(r)
    denom = 1.0 / (1.0 + exp_r.sum(axis=1, keepdims=True))
    bugs_abc = exp_r * denom
    bug_d = denom
    return np.column_stack([bugs_abc, bug_d])


@dataclass(frozen=True)
class BlockData:
    X_micro: np.ndarray
    X_met: np.ndarray
    X_gene: np.ndarray
    y: np.ndarray


def build_blocks(df: pd.DataFrame) -> BlockData:
    X_micro = alr_log_ratios(df)
    X_met = np.log1p(df[MET_COLS].to_numpy(dtype=float))
    X_gene = np.log1p(df[GENE_COLS].to_numpy(dtype=float))
    y = df["Group"].to_numpy(dtype=int)
    return BlockData(X_micro=X_micro, X_met=X_met, X_gene=X_gene, y=y)


@dataclass
class Standardizer:
    mean_: np.ndarray
    std_: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Standardizer":
        mean_ = X.mean(axis=0)
        std_ = X.std(axis=0, ddof=0)
        std_ = np.where(std_ == 0, 1.0, std_)
        return cls(mean_=mean_, std_=std_)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_


def simulate_from_toy(df_toy: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Creates a slightly larger dataset by jittering around the 4 toy patients.

    This is for *demonstration only* so models can be trained stably.
    """
    rng = np.random.default_rng(seed)
    base = df_toy.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)

    # Microbiome: jitter in ALR space, then map back to a valid composition.
    r = alr_log_ratios(base)
    r = r + rng.normal(0.0, 0.15, size=r.shape)
    bugs = invert_alr_to_composition(r)

    # Metabolites / genes: jitter in log1p space, then invert.
    met_log = np.log1p(base[MET_COLS].to_numpy(dtype=float))
    met_log = met_log + rng.normal(0.0, 0.10, size=met_log.shape)
    met = np.expm1(met_log)
    met = np.clip(met, 0.0, None)

    gene_log = np.log1p(base[GENE_COLS].to_numpy(dtype=float))
    gene_log = gene_log + rng.normal(0.0, 0.05, size=gene_log.shape)
    gene = np.expm1(gene_log)
    gene = np.clip(gene, 0.0, None)

    out = pd.DataFrame(
        {
            "Patient": [f"S{i+1}" for i in range(n)],
            "Group": base["Group"].to_numpy(dtype=int),
            "BugA": bugs[:, 0],
            "BugB": bugs[:, 1],
            "BugC": bugs[:, 2],
            "BugD": bugs[:, 3],
            "Met1": met[:, 0],
            "Met2": met[:, 1],
            "Met3": met[:, 2],
            "Gene1": gene[:, 0],
            "Gene2": gene[:, 1],
            "Gene3": gene[:, 2],
        }
    )
    return out


def add_intercept(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(X.shape[0]), X])


def fit_ols(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Least squares coefficients for Y ~ X.

    Returns coef with shape (X_dim, Y_dim).
    """
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return coef


def pca_latent(X: np.ndarray, latent_dim: int) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    U, s, _Vt = np.linalg.svd(Xc, full_matrices=False)
    return U[:, :latent_dim] * s[:latent_dim]


def linear_shared_latent_demo(df_train: pd.DataFrame, df_report: pd.DataFrame, latent_dim: int) -> None:
    blocks_train = build_blocks(df_train)
    blocks_report = build_blocks(df_report)

    scalers = {
        "micro": Standardizer.fit(blocks_train.X_micro),
        "met": Standardizer.fit(blocks_train.X_met),
        "gene": Standardizer.fit(blocks_train.X_gene),
    }

    X_train = np.concatenate(
        [
            scalers["micro"].transform(blocks_train.X_micro),
            scalers["met"].transform(blocks_train.X_met),
            scalers["gene"].transform(blocks_train.X_gene),
        ],
        axis=1,
    )

    Z_train = pca_latent(X_train, latent_dim=latent_dim)

    # Reconstruct each block from Z (simple linear decoder).
    Z_design = add_intercept(Z_train)

    def recon_mse(X_block_std: np.ndarray) -> float:
        coef = fit_ols(Z_design, X_block_std)
        pred = Z_design @ coef
        return float(np.mean((X_block_std - pred) ** 2))

    mse_micro = recon_mse(scalers["micro"].transform(blocks_train.X_micro))
    mse_met = recon_mse(scalers["met"].transform(blocks_train.X_met))
    mse_gene = recon_mse(scalers["gene"].transform(blocks_train.X_gene))

    print("Linear shared-latent baseline (PCA latent + linear recon):")
    print(f"- latent_dim = {latent_dim}")
    print(f"- recon MSE (microbiome ALR): {mse_micro:.4f}")
    print(f"- recon MSE (metabolomics log1p): {mse_met:.4f}")
    print(f"- recon MSE (transcriptomics log1p): {mse_gene:.4f}")
    print()

    # Report latents for the original 4 patients.
    X_report = np.concatenate(
        [
            scalers["micro"].transform(blocks_report.X_micro),
            scalers["met"].transform(blocks_report.X_met),
            scalers["gene"].transform(blocks_report.X_gene),
        ],
        axis=1,
    )

    # Project into the PCA latent space computed from training.
    Xc_train = X_train - X_train.mean(axis=0, keepdims=True)
    _U, _s, Vt = np.linalg.svd(Xc_train, full_matrices=False)
    W = Vt[:latent_dim].T
    Z_report = (X_report - X_train.mean(axis=0, keepdims=True)) @ W

    cols = {"z1": Z_report[:, 0]}
    if latent_dim >= 2:
        cols["z2"] = Z_report[:, 1]

    out = pd.DataFrame({"Patient": df_report["Patient"], "Group": blocks_report.y, **cols})
    print("Latent scores for the 4 toy patients (from the linear baseline):")
    print(out.to_string(index=False))
    print()


def try_train_torch_vae(
    df_train: pd.DataFrame,
    df_report: pd.DataFrame,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    beta: float,
    conditional_decoder: bool,
    seed: int,
) -> None:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
    except Exception:
        print("Torch not installed; running the linear baseline instead.")
        linear_shared_latent_demo(df_train=df_train, df_report=df_report, latent_dim=latent_dim)
        return

    torch.manual_seed(seed)

    blocks_train = build_blocks(df_train)
    blocks_report = build_blocks(df_report)

    scalers = {
        "micro": Standardizer.fit(blocks_train.X_micro),
        "met": Standardizer.fit(blocks_train.X_met),
        "gene": Standardizer.fit(blocks_train.X_gene),
    }

    X_micro = scalers["micro"].transform(blocks_train.X_micro)
    X_met = scalers["met"].transform(blocks_train.X_met)
    X_gene = scalers["gene"].transform(blocks_train.X_gene)
    y = blocks_train.y

    X_micro_t = torch.tensor(X_micro, dtype=torch.float32)
    X_met_t = torch.tensor(X_met, dtype=torch.float32)
    X_gene_t = torch.tensor(X_gene, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    ds = TensorDataset(X_micro_t, X_met_t, X_gene_t, y_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    group_dim = int(len(np.unique(blocks_train.y)))

    class Encoder(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int, z_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.mu = nn.Linear(hidden_dim, z_dim)
            self.logvar = nn.Linear(hidden_dim, z_dim)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            h = self.net(x)
            return self.mu(h), self.logvar(h)

    class Decoder(nn.Module):
        def __init__(self, z_dim: int, hidden_dim: int, out_dim: int, cond_dim: int = 0):
            super().__init__()
            self.cond_dim = cond_dim
            in_dim = z_dim + cond_dim
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )

        def forward(self, z: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
            if self.cond_dim:
                if cond is None:
                    raise ValueError("cond must be provided when cond_dim > 0")
                z = torch.cat([z, cond], dim=1)
            return self.net(z)

    def poe_combine(mus: list[torch.Tensor], logvars: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Product-of-experts for diagonal Gaussians + standard normal prior.

        Returns (mu, logvar).
        """
        device = mus[0].device
        z_dim = mus[0].shape[1]

        # Prior N(0, I)
        mu0 = torch.zeros((mus[0].shape[0], z_dim), device=device)
        logvar0 = torch.zeros_like(mu0)
        mus_all = [mu0] + mus
        logvars_all = [logvar0] + logvars

        precisions = [torch.exp(-lv) for lv in logvars_all]
        precision_sum = torch.stack(precisions, dim=0).sum(dim=0)
        mu = torch.stack([m * p for m, p in zip(mus_all, precisions)], dim=0).sum(dim=0) / precision_sum
        var = 1.0 / precision_sum
        logvar = torch.log(var)
        return mu, logvar

    def reparam(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    enc_micro = Encoder(in_dim=X_micro.shape[1], hidden_dim=32, z_dim=latent_dim)
    enc_met = Encoder(in_dim=X_met.shape[1], hidden_dim=32, z_dim=latent_dim)
    enc_gene = Encoder(in_dim=X_gene.shape[1], hidden_dim=32, z_dim=latent_dim)

    cond_dim = group_dim if conditional_decoder else 0
    dec_micro = Decoder(z_dim=latent_dim, hidden_dim=32, out_dim=X_micro.shape[1], cond_dim=cond_dim)
    dec_met = Decoder(z_dim=latent_dim, hidden_dim=32, out_dim=X_met.shape[1], cond_dim=cond_dim)
    dec_gene = Decoder(z_dim=latent_dim, hidden_dim=32, out_dim=X_gene.shape[1], cond_dim=cond_dim)

    params = list(enc_micro.parameters()) + list(enc_met.parameters()) + list(enc_gene.parameters())
    params += list(dec_micro.parameters()) + list(dec_met.parameters()) + list(dec_gene.parameters())

    opt = torch.optim.Adam(params, lr=lr)

    def batch_loss(xm: torch.Tensor, xmet: torch.Tensor, xg: torch.Tensor, yb: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        mu_m, lv_m = enc_micro(xm)
        mu_met, lv_met = enc_met(xmet)
        mu_g, lv_g = enc_gene(xg)

        mu, logvar = poe_combine([mu_m, mu_met, mu_g], [lv_m, lv_met, lv_g])
        z = reparam(mu, logvar)

        cond = None
        if conditional_decoder:
            cond = F.one_hot(yb, num_classes=group_dim).float()

        xm_hat = dec_micro(z, cond)
        xmet_hat = dec_met(z, cond)
        xg_hat = dec_gene(z, cond)

        recon = F.mse_loss(xm_hat, xm) + F.mse_loss(xmet_hat, xmet) + F.mse_loss(xg_hat, xg)
        kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        total = recon + beta * kl

        stats = {
            "recon": float(recon.detach().cpu().item()),
            "kl": float(kl.detach().cpu().item()),
            "total": float(total.detach().cpu().item()),
        }
        return total, stats

    print("Training multi-modal VAE:")
    print(f"- samples: {len(df_train)} (simulated from 4 toy patients)")
    print(f"- latent_dim: {latent_dim}")
    print(f"- conditional_decoder: {conditional_decoder}")

    for epoch in range(1, epochs + 1):
        losses = []
        for xm_b, xmet_b, xg_b, yb in dl:
            opt.zero_grad()
            loss, _stats = batch_loss(xm_b, xmet_b, xg_b, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))

        if epoch in {1, 10, 50, 100, epochs}:
            print(f"  epoch {epoch:>4}: loss={np.mean(losses):.4f}")

    # Report: latent means for the 4 toy patients.
    X_micro_r = torch.tensor(scalers["micro"].transform(blocks_report.X_micro), dtype=torch.float32)
    X_met_r = torch.tensor(scalers["met"].transform(blocks_report.X_met), dtype=torch.float32)
    X_gene_r = torch.tensor(scalers["gene"].transform(blocks_report.X_gene), dtype=torch.float32)

    with torch.no_grad():
        mu_m, lv_m = enc_micro(X_micro_r)
        mu_met, lv_met = enc_met(X_met_r)
        mu_g, lv_g = enc_gene(X_gene_r)
        mu, _logvar = poe_combine([mu_m, mu_met, mu_g], [lv_m, lv_met, lv_g])

    Z = mu.detach().cpu().numpy()
    out = pd.DataFrame({"Patient": df_report["Patient"], "Group": blocks_report.y})
    out["z1"] = Z[:, 0]
    if latent_dim >= 2:
        out["z2"] = Z[:, 1]

    print("\nLatent means for the 4 toy patients (VAE):")
    print(out.to_string(index=False))
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--vae", action="store_true", help="Train the multi-modal VAE (requires torch).")
    p.add_argument("--conditional-decoder", action="store_true", help="Feed Group into the decoder (conditional VAE).")

    p.add_argument("--simulate-n", type=int, default=300, help="How many simulated samples to train on.")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--latent-dim", type=int, default=2)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=0.1, help="KL weight.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    df_toy = load_toy_df()
    df_train = simulate_from_toy(df_toy, n=args.simulate_n, seed=args.seed)

    if args.vae:
        try_train_torch_vae(
            df_train=df_train,
            df_report=df_toy,
            latent_dim=args.latent_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            beta=args.beta,
            conditional_decoder=args.conditional_decoder,
            seed=args.seed,
        )
    else:
        linear_shared_latent_demo(df_train=df_train, df_report=df_toy, latent_dim=args.latent_dim)
        print("Tip: to train the multi-modal VAE, install torch and run with --vae.")


if __name__ == "__main__":
    main()
