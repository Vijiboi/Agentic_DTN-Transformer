"""
data.py
-------
Loads `pso_training_dataset_extended.xlsx` (or a cached .parquet) and turns it
into fixed-size candidate sets that a Set-Transformer can consume.

Dataset shape (confirmed by inspection):
    199,540 rows = 19,954 (scenario_id, slot, task_id) task-groups
                    x 10 candidate APs each (candidate_rank 1..10)
    exactly one row per group has pso_selected == 1
    no missing values, no "LOCAL" execution rows in this extract

Each group -> one training example:
    X : (10, F)   per-candidate feature vector
    y : scalar in [0, 9]   index of the PSO-selected candidate
    aux_y : (10,) candidate_local_cost, used for an auxiliary regression head
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ----------------------------------------------------------------------
# Feature schema
# ----------------------------------------------------------------------

# Continuous features that already live on a natural numeric scale.
NUMERIC_COLS = [
    # task-level (identical across the 10 candidates in a group, but the
    # transformer sees them per-token; harmless and lets the model learn
    # task x AP interactions rather than only AP-level effects)
    "L_u_mbit", "D_u_cycles_per_bit", "omega_u", "psi_u", "AoI",
    # candidate/AP-level
    "candidate_rank", "distance_km",
    "ap_bandwidth", "ap_cpu_capacity", "ap_communication_budget", "ap_power_budget",
    "ap_trust", "ap_intrinsic_reliability", "ap_current_task_load", "ap_current_cpu_load",
    # digital-twin state (this is what makes the outer loop "twin-aware")
    "twin_predicted_load", "twin_predicted_bandwidth", "twin_predicted_cpu_ratio",
    "twin_age", "twin_uncertainty", "twin_mismatch", "twin_fidelity",
    # regional coordination state (dual prices / consensus)
    "coord_dual_price", "coord_quantized_dual", "coord_coupling_penalty",
    "predicted_rate_mbps", "sync_active", "coord_active",
    # QUBO/inner-loop-derived candidate costs -> this is the bridge between
    # the inner-loop QUBO surrogate and the outer-loop selection policy
    "candidate_delay_s", "candidate_energy_j", "candidate_freshness_risk",
    "candidate_fidelity_penalty", "candidate_local_cost",
]

# Features available *before* the inner QUBO coefficients are calculated.
# Deliberately excludes every QUBO-derived candidate-cost component, including
# candidate_local_cost, so a surrogate trained on this schema cannot copy its
# answer from the target calculation.
QUBO_SURROGATE_NUMERIC_COLS = [
    "L_u_mbit", "D_u_cycles_per_bit", "omega_u", "psi_u", "AoI",
    "candidate_rank", "distance_km",
    "ap_bandwidth", "ap_cpu_capacity", "ap_communication_budget", "ap_power_budget",
    "ap_trust", "ap_intrinsic_reliability", "ap_current_task_load", "ap_current_cpu_load",
    "twin_predicted_load", "twin_predicted_bandwidth", "twin_predicted_cpu_ratio",
    "twin_age", "twin_uncertainty", "twin_mismatch", "twin_fidelity",
    "coord_dual_price", "coord_quantized_dual", "coord_coupling_penalty",
    "predicted_rate_mbps", "sync_active", "coord_active",
]

CATEGORICAL_COLS = {
    "xi_u": ["control", "imagery", "alert"],                 # task class
    "destination_tier": ["BS", "HAP", "LEO"],                 # AP tier
}

GROUP_KEYS = ["scenario_id", "slot", "task_id"]
LABEL_COL = "pso_selected"
AUX_TARGET_COL = "candidate_local_cost"
CANDIDATES_PER_TASK = 10  # fixed in this extract; code below still masks


@dataclasses.dataclass
class FeatureStats:
    mean: np.ndarray
    std: np.ndarray
    numeric_cols: list
    categorical_cols: dict

    @property
    def feature_dim(self) -> int:
        cat_dim = sum(len(v) for v in self.categorical_cols.values())
        return len(self.numeric_cols) + cat_dim

    def save(self, path: str | Path):
        np.savez(path, mean=self.mean, std=self.std,
                 numeric_cols=np.array(self.numeric_cols, dtype=object),
                 categorical_cols=np.array(list(self.categorical_cols.items()), dtype=object))

    @staticmethod
    def load(path: str | Path) -> "FeatureStats":
        z = np.load(path, allow_pickle=True)
        cat = {k: list(v) for k, v in z["categorical_cols"]}
        return FeatureStats(mean=z["mean"], std=z["std"],
                             numeric_cols=list(z["numeric_cols"]), categorical_cols=cat)


def load_raw(path: str | Path) -> pd.DataFrame:
    """Loads CSV, xlsx, or parquet training data transparently."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    cache = path.with_suffix(".parquet")
    if cache.exists():
        return pd.read_parquet(cache)
    df = pd.read_excel(path, engine="openpyxl")
    try:
        df.to_parquet(cache)
    except Exception:
        pass  # caching is best-effort only
    return df


def _one_hot(series: pd.Series, categories: list[str]) -> np.ndarray:
    cat = pd.Categorical(series, categories=categories)
    return np.eye(len(categories), dtype=np.float32)[cat.codes.clip(min=0)]


def build_feature_matrix(
    df: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    categorical_cols: dict[str, list[str]] | None = None,
) -> np.ndarray:
    """Returns an (N_rows, F) float32 matrix, numeric cols first, then one-hots."""
    numeric_cols = numeric_cols or NUMERIC_COLS
    categorical_cols = categorical_cols or CATEGORICAL_COLS
    numeric = df[numeric_cols].to_numpy(dtype=np.float32)
    blocks = [numeric]
    for col, cats in categorical_cols.items():
        blocks.append(_one_hot(df[col], cats))
    return np.concatenate(blocks, axis=1)


def compute_feature_stats(feat: np.ndarray, n_numeric: int) -> FeatureStats:
    """Standardize only the numeric block; one-hots are left as 0/1."""
    mean = np.zeros(feat.shape[1], dtype=np.float32)
    std = np.ones(feat.shape[1], dtype=np.float32)
    mean[:n_numeric] = feat[:, :n_numeric].mean(axis=0)
    std[:n_numeric] = feat[:, :n_numeric].std(axis=0) + 1e-6
    return FeatureStats(mean=mean, std=std, numeric_cols=NUMERIC_COLS, categorical_cols=CATEGORICAL_COLS)


class OuterLoopCandidateDataset(Dataset):
    """
    One item = one (scenario_id, slot, task_id) task-group:
        candidates : (K, F) standardized features, K = CANDIDATES_PER_TASK (padded if fewer)
        mask       : (K,) 1 for real candidates, 0 for padding
        label      : () long index of the PSO-selected candidate
        aux        : (K,) candidate_local_cost target for the auxiliary regression head
        dest_ids   : list[str] destination_id per slot, for turning predictions back
                     into an AP id at inference time
    """

    def __init__(
        self,
        df: pd.DataFrame,
        stats: FeatureStats,
        max_candidates: int = CANDIDATES_PER_TASK,
        label_col: str = LABEL_COL,
    ):
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' is not present in the dataset.")
        self.max_candidates = max_candidates
        self.stats = stats
        self.label_col = label_col

        df = df.sort_values(GROUP_KEYS + ["candidate_rank"]).reset_index(drop=True)
        feat = build_feature_matrix(df)
        feat = (feat - stats.mean) / stats.std
        self.feat = feat.astype(np.float32)
        self.aux = df[AUX_TARGET_COL].to_numpy(dtype=np.float32)
        self.label_flag = df[label_col].to_numpy(dtype=np.int64)
        self.dest_ids = df["destination_id"].to_numpy()

        self.groups = list(df.groupby(GROUP_KEYS, sort=False).indices.values())

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        row_idx = self.groups[idx]
        k = len(row_idx)
        k_use = min(k, self.max_candidates)
        row_idx = row_idx[:k_use]

        F = self.feat.shape[1]
        cand = np.zeros((self.max_candidates, F), dtype=np.float32)
        mask = np.zeros((self.max_candidates,), dtype=np.float32)
        aux = np.zeros((self.max_candidates,), dtype=np.float32)

        cand[:k_use] = self.feat[row_idx]
        mask[:k_use] = 1.0
        aux[:k_use] = self.aux[row_idx]

        sel = np.where(self.label_flag[row_idx] == 1)[0]
        label = int(sel[0]) if len(sel) else 0  # fallback, shouldn't happen in this data

        return {
            "candidates": torch.from_numpy(cand),
            "mask": torch.from_numpy(mask),
            "label": torch.tensor(label, dtype=torch.long),
            "aux": torch.from_numpy(aux),
            "dest_ids": [self.dest_ids[i] for i in row_idx] + [""] * (self.max_candidates - k_use),
        }


class QuboCostSurrogateDataset(Dataset):
    """One candidate set per task with QUBO local cost as a regression target."""

    def __init__(
        self,
        df: pd.DataFrame,
        stats: FeatureStats,
        max_candidates: int = CANDIDATES_PER_TASK,
        target_col: str = AUX_TARGET_COL,
    ):
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' is not present in the dataset.")
        self.max_candidates = max_candidates
        self.stats = stats
        df = df.sort_values(GROUP_KEYS + ["candidate_rank"]).reset_index(drop=True)
        feat = build_feature_matrix(
            df, numeric_cols=stats.numeric_cols, categorical_cols=stats.categorical_cols
        )
        self.feat = ((feat - stats.mean) / stats.std).astype(np.float32)
        self.target = df[target_col].to_numpy(dtype=np.float32)
        self.groups = list(df.groupby(GROUP_KEYS, sort=False).indices.values())

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        row_idx = self.groups[idx]
        k_use = min(len(row_idx), self.max_candidates)
        row_idx = row_idx[:k_use]
        feature_dim = self.feat.shape[1]
        candidates = np.zeros((self.max_candidates, feature_dim), dtype=np.float32)
        mask = np.zeros((self.max_candidates,), dtype=np.float32)
        target = np.zeros((self.max_candidates,), dtype=np.float32)
        candidates[:k_use] = self.feat[row_idx]
        mask[:k_use] = 1.0
        target[:k_use] = self.target[row_idx]
        return {
            "candidates": torch.from_numpy(candidates),
            "mask": torch.from_numpy(mask),
            "target": torch.from_numpy(target),
        }


def scenario_split(df: pd.DataFrame, val_frac=0.15, test_frac=0.15, seed=7):
    """
    Splits by scenario_id (stratified by scenario_profile) so that no task from
    a given scenario leaks across train/val/test -- this mirrors evaluating on
    genuinely unseen network conditions rather than unseen tasks within a
    scenario the model already trained on.
    """
    rng = np.random.default_rng(seed)
    scen = df[["scenario_id", "scenario_profile"]].drop_duplicates()
    train_ids, val_ids, test_ids = [], [], []
    for profile, group in scen.groupby("scenario_profile"):
        ids = group["scenario_id"].to_numpy().copy()
        rng.shuffle(ids)
        n = len(ids)
        n_val = max(1, int(round(n * val_frac)))
        n_test = max(1, int(round(n * test_frac)))
        val_ids.extend(ids[:n_val])
        test_ids.extend(ids[n_val:n_val + n_test])
        train_ids.extend(ids[n_val + n_test:])

    train_df = df[df.scenario_id.isin(train_ids)]
    val_df = df[df.scenario_id.isin(val_ids)]
    test_df = df[df.scenario_id.isin(test_ids)]
    return train_df, val_df, test_df
