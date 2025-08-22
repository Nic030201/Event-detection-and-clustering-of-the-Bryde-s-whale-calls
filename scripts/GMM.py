"""
GMM noise-vs-call clustering + optional evaluation against true selections.

Usage
-----
# 1) Cluster detections only (no evaluation). The output will be written to results/GMM/<detections>_with_gmm.csv
python GMM.py --detections 20230422_171301_2hr_45hr_start_events_for_gmm.csv

# 2) Cluster + evaluate with ground truth (Raven table or your selection file)
python GMM.py \
  --detections 20230422_171301_2hr_45hr_start_events_for_gmm.csv \
  --truth 20230422_171301.Detections.selections.test.txt \
  --threshold 0.5 \
  --time_tolerance 0.20
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from pathlib import Path


def find_file(filename: str, search_root: Path = Path.cwd()) -> Path:
    """
    Search recursively under search_root for a file with the given name.
    Returns the first match found.
    """
    matches = list(search_root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find file {filename} under {search_root}")
    return matches[0].resolve()


# ---- Feature presets (matching your file columns) ----
FEATURE_PRESETS = {
    # Good all‑rounder using interpretable anchors
    "tier1": [
        "dur", "nb_ratio_med", "nb_ratio_p75", "tonality_med",
        "rms_med", "td_energy_med", "flux_med",
        "td_slope_med", "td_slope_min"
    ],

    # Very compact, fast to iterate; keep at least one “anchor” so call mapping works
    "slim5": ["nb_ratio_med", "tonality_med", "dur", "rms_med", "td_slope_med"],

    # Tonality/energy/slope focused (often good for Bryde’s downsweep vs noise)
    "tonal_focus": [
        "dur", "nb_ratio_med", "nb_ratio_p75", "tonality_med",
        "td_slope_med", "td_slope_min", "flux_med", "td_energy_med"
    ],

    # ---- MFCC-only options (names that exist in your CSV) ----
    # (Add at least one anchor version too, to avoid label-flip issues.)
    "mfcc_mean_5": [
        "mfcc_00_mean","mfcc_01_mean","mfcc_02_mean","mfcc_03_mean","mfcc_04_mean"
    ],
    "mfcc_meanstd_5": [
        "mfcc_00_mean","mfcc_01_mean","mfcc_02_mean","mfcc_03_mean","mfcc_04_mean",
        "mfcc_00_std","mfcc_01_std","mfcc_02_std","mfcc_03_std","mfcc_04_std"
    ],
    "mfcc_stats_10": [
        "mfcc_00_mean","mfcc_01_mean","mfcc_02_mean","mfcc_03_mean","mfcc_04_mean",
        "mfcc_05_mean","mfcc_06_mean","mfcc_07_mean","mfcc_08_mean","mfcc_09_mean",
        "mfcc_00_std","mfcc_01_std","mfcc_02_std","mfcc_03_std","mfcc_04_std",
        "mfcc_05_std","mfcc_06_std","mfcc_07_std","mfcc_08_std","mfcc_09_std"
    ],
    "mfcc_anchor_mix": [
        "mfcc_00_mean","mfcc_01_mean","mfcc_02_mean","mfcc_03_mean",
        "nb_ratio_med","tonality_med","td_slope_med","dur"
    ],

    # ---- LFCC options (often more robust at low freqs) ----
    "lfcc_mean_5": [
        "lfcc_00_mean","lfcc_01_mean","lfcc_02_mean","lfcc_03_mean","lfcc_04_mean"
    ],
    "lfcc_meanstd_5": [
        "lfcc_00_mean","lfcc_01_mean","lfcc_02_mean","lfcc_03_mean","lfcc_04_mean",
        "lfcc_00_std","lfcc_01_std","lfcc_02_std","lfcc_03_std","lfcc_04_std"
    ],
    "lfcc_stats_10": [
        "lfcc_00_mean","lfcc_01_mean","lfcc_02_mean","lfcc_03_mean","lfcc_04_mean",
        "lfcc_05_mean","lfcc_06_mean","lfcc_07_mean","lfcc_08_mean","lfcc_09_mean",
        "lfcc_00_std","lfcc_01_std","lfcc_02_std","lfcc_03_std","lfcc_04_std",
        "lfcc_05_std","lfcc_06_std","lfcc_07_std","lfcc_08_std","lfcc_09_std"
    ],
    "lfcc_anchor_mix": [
        "lfcc_00_mean","lfcc_01_mean","lfcc_02_mean","lfcc_03_mean",
        "nb_ratio_med","tonality_med","td_slope_med","dur"
    ],

    # ---- Delta-LFCC (dynamics) ----
    "dlfcc_mean_6": [
        "dlfcc_00_mean","dlfcc_01_mean","dlfcc_02_mean","dlfcc_03_mean",
        "dlfcc_04_mean","dlfcc_05_mean"
    ],
    "dlfcc_anchor_mix": [
        "dlfcc_00_mean","dlfcc_01_mean","dlfcc_02_mean",
        "nb_ratio_med","tonality_med","td_slope_med","dur"
    ],

    # Hybrid: small cepstral block + interpretable anchors
    "hybrid_small": [
        "mfcc_00_mean","mfcc_01_mean","lfcc_00_mean","lfcc_01_mean",
        "nb_ratio_med","tonality_med","td_slope_med","dur"
    ],
}

def choose_features(df: pd.DataFrame, preset: str) -> list[str]:
    wanted = FEATURE_PRESETS[preset]
    have = [c for c in wanted if c in df.columns]
    missing = [c for c in wanted if c not in df.columns]
    if len(have) < 3:
        raise ValueError(f"Too few features available for preset '{preset}'. Missing: {missing}")
    if missing:
        print(f"[features] Missing in this file (skipped): {missing}")
    print(f"[features] Using {len(have)} features: {have}")
    return have

def load_truth(path: str) -> pd.DataFrame:
    """
    Robustly load a truth/selection table and return columns ['start_s','end_s'] as float.
    - Accepts CSV/TXT with comma, tab, or whitespace separators.
    - Accepts common Raven headers (Begin/End Time (s)), or generic names containing
      start/begin/onset and end/offset/finish (case-insensitive; spaces ignored).
    - If no header match is found, tries headerless inference by picking a numeric
      column pair where end > start most of the time.
    """
    import re

    def _normalize(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", s.strip().lower())

    # 1) Try reading with header row (auto-separator)
    try:
        df = pd.read_csv(path, engine="python", sep=None)
        header_mode = "headered"
    except Exception:
        # fallback: whitespace-delimited
        df = pd.read_csv(path, sep=r"\s+", engine="python")
        header_mode = "headered"

    # 2) If very few columns or obviously no header, try header=None
    need_headerless_try = False
    if df.shape[1] == 1:
        need_headerless_try = True
    else:
        # if all column names look like integers (0,1,2,...), probably no header
        if all(isinstance(c, (int, np.integer)) for c in df.columns):
            need_headerless_try = True

    # Try headerless load if needed
    if need_headerless_try:
        try:
            df = pd.read_csv(path, engine="python", sep=None, header=None)
            header_mode = "headerless"
        except Exception:
            df = pd.read_csv(path, sep=r"\s+", engine="python", header=None)
            header_mode = "headerless"
        # Give temporary names
        df.columns = [f"col{i}" for i in range(df.shape[1])]

    # 3) Try to find columns by name (if headered)
    if header_mode == "headered":
        cols_norm = {_normalize(c): c for c in df.columns}

        # Exact raven
        raven_start_keys = ["begintimes", "begintimes", "begintimes"]  # normalized forms
        # But safer: look for any 'begin time (s)' etc after normalization
        preferred_starts = ["begintimes", "begintimes", "begintimes"]

        # Generic patterns
        start_candidates = [
            c for norm, c in cols_norm.items()
            if any(k in norm for k in ["begintime", "starttime", "start", "begin", "onset"])
        ]
        end_candidates = [
            c for norm, c in cols_norm.items()
            if any(k in norm for k in ["endtime", "end", "offset", "finish"])
        ]

        # Also allow exact Raven names with spaces/case, normalized
        # (The generic lists above already catch most cases.)

        if start_candidates and end_candidates:
            # choose first plausible pair
            start_col = start_candidates[0]
            end_col = end_candidates[0]
            out = df[[start_col, end_col]].copy()
            out.columns = ["start_s", "end_s"]
            # coerce numeric
            out = out.apply(pd.to_numeric, errors="coerce")
            out = out.dropna(subset=["start_s", "end_s"])
            return out

    # 4) If we get here, infer start/end from numeric columns (works for headerless or odd headers)
    # Keep only numeric columns
    df_num = df.copy()
    for c in df_num.columns:
        df_num[c] = pd.to_numeric(df_num[c], errors="coerce")
    numeric_cols = [c for c in df_num.columns if pd.api.types.is_numeric_dtype(df_num[c])]
    if len(numeric_cols) < 2:
        raise ValueError("Truth file has fewer than two numeric columns; cannot infer start/end.")

    # Score all ordered pairs of distinct numeric columns: how often end > start?
    best = None
    best_pair = None
    for i, sc in enumerate(numeric_cols):
        for j, ec in enumerate(numeric_cols):
            if i == j:
                continue
            s = df_num[sc]
            e = df_num[ec]
            mask = s.notna() & e.notna()
            if mask.sum() == 0:
                continue
            # score: fraction of rows with e > s plus how many valid rows we have
            frac_valid = (e[mask] > s[mask]).mean()
            # composite score: prefer high validity and duration positivity
            score = frac_valid * (mask.mean())
            if (best is None) or (score > best):
                best = score
                best_pair = (sc, ec)

    if best_pair is None:
        raise ValueError("Could not infer start/end columns in truth file.")

    sc, ec = best_pair
    out = df_num[[sc, ec]].copy()
    out.columns = ["start_s", "end_s"]
    out = out.dropna(subset=["start_s", "end_s"])
    print(f"[load_truth] Inferred columns by numeric pattern: start='{sc}', end='{ec}' (score={best:.3f})")
    return out

def winsorize_df(df: pd.DataFrame, cols, lo=1.0, hi=99.0) -> pd.DataFrame:
    """Clip extremes to (lo, hi) percentiles for stability."""
    X = df.copy()
    for c in cols:
        if c in X.columns:
            vals = X[c].to_numpy()
            lo_v, hi_v = np.percentile(vals[~np.isnan(vals)], [lo, hi])
            X[c] = np.clip(X[c], lo_v, hi_v)
    return X

def component_means_in_space(responsibilities: np.ndarray, X_z: np.ndarray) -> np.ndarray:
    """
    Compute component means in a given z-scored feature space by responsibility-weighted averages.
    responsibilities: (N, K)
    X_z:             (N, D_anchor_z)  (already StandardScaler-transformed)
    Returns: (K, D_anchor_z) array of component means.
    """
    R = responsibilities  # (N, K)
    Nk = R.sum(axis=0) + 1e-12
    # Weighted means per component: mu_k = (R[:,k]^T X) / Nk
    means = (R.T @ X_z) / Nk[:, None]
    return means


def mark_truth_overlaps(det_df: pd.DataFrame, truth_df: pd.DataFrame, tol: float) -> pd.Series:
    """
    For each detection [s1,e1], mark 1 if it overlaps any truth [s2,e2] with tolerance:
      s1 < e2 + tol  AND  e1 > s2 - tol
    Returns 0/1 Series aligned to det_df.index.
    """
    if truth_df is None or truth_df.empty:
        return pd.Series(np.zeros(len(det_df), dtype=int), index=det_df.index)

    truth_sorted = truth_df.sort_values("start_s").reset_index(drop=True)
    s2 = truth_sorted["start_s"].to_numpy()
    e2 = truth_sorted["end_s"].to_numpy()

    y_true = np.zeros(len(det_df), dtype=int)
    s1_arr = det_df["start_s"].to_numpy()
    e1_arr = det_df["end_s"].to_numpy()

    for i, (s1, e1) in enumerate(zip(s1_arr, e1_arr)):
        mask = (s2 <= e1 + tol) & (e2 >= s1 - tol)
        if np.any(mask):
            y_true[i] = 1
    return pd.Series(y_true, index=det_df.index, dtype=int)


def pick_call_component(means: np.ndarray, features: list[str]) -> int:
    """
    Decide which GMM component is 'call-like' using anchor features that exist in `features`.

    Inputs
    ------
    means : (K, D) array of component means in z-space (after StandardScaler)
    features : list[str] of feature names used to fit the GMM (columns of X)

    Heuristics (z-space):
      + Higher is call-like:  nb_ratio_med, nb_ratio_p75, tonality_med, rms_med, td_energy_med
      + Lower is call-like:   flux_med, bw, zcr
      + Downsweep slope:      td_slope_med (more negative is more call-like)

    Notes
    -----
    - We only score anchors that are actually present in `features`.
    - We normalise by total |weight| applied so scores are comparable across presets.
    - If no anchors are found, we fall back to a generic rule (component with larger
      overall mean in z-space).
    """

    # helpers
    def has(f: str) -> bool:
        return f in features

    def get(f: str):
        # returns (K,) z-mean column for feature f
        return means[:, features.index(f)]

    K = means.shape[0]
    score = np.zeros(K, dtype=float)
    total_weight = np.zeros(K, dtype=float)  # track applied |weights| to normalise

    # --------- anchors (only if present) ----------
    # Positive anchors (higher → more call-like)
    pos_anchors = {
        "nb_ratio_med": 1.0,
        "nb_ratio_p75": 0.6,
        "tonality_med": 1.0,
        "rms_med": 0.7,
        "td_energy_med": 0.8,
    }

    # Negative anchors (lower → more call-like)
    neg_anchors = {
        "flux_med": 0.8,
        "bw": 0.5,
        "zcr": 0.5,
    }

    # Apply positive anchors
    for f, w in pos_anchors.items():
        if has(f):
            v = get(f)                 # z-mean
            score += w * v
            total_weight += w

    # Apply negative anchors
    for f, w in neg_anchors.items():
        if has(f):
            v = get(f)
            score += w * (-v)
            total_weight += w

    # Downsweep slope: more negative is more call-like.
    # We reward negativity and penalise positive slopes, with a moderate weight.
    if has("td_slope_med"):
        v = get("td_slope_med")
        w = 0.8
        score += w * (-v)     # negative v → positive contribution
        total_weight += w

    # Optional: centre frequency proximity (if you later add 'cfreq')
    # Example (Bryde’s in-band ~30–80 Hz, mid ~55 Hz). Small, gentle weight.
    if has("cfreq"):
        v = get("cfreq")
        w = 0.3
        # Convert to a proximity score: closer to 55 → higher score
        # Use a robust scale so it doesn't dominate.
        prox = -np.abs(v - 55.0) / 55.0
        score += w * prox
        total_weight += w

    # --------- normalise if we used any anchors ----------
    used_any_anchors = np.any(total_weight > 0)
    if used_any_anchors:
        # normalise by the scalar total weight (same for all K because we add same w to all comps)
        denom = np.max(total_weight)  # same for all components here
        if denom > 0:
            score = score / denom
        # Pick the component with the largest call-likeness score
        return int(np.argmax(score))

    # --------- fallbacks (no anchors in preset) ----------
    # 1) If MFCC/LFCC energy-like stats are present, prefer higher overall mean energy-ish stats.
    #    (mfcc_00_mean often correlates with log-energy; lfcc_00_mean similarly.)
    for energyish in ["mfcc_00_mean", "lfcc_00_mean"]:
        if has(energyish):
            return int(np.argmax(get(energyish)))

    # 2) Generic: choose the component with the larger overall mean in z-space
    # (This is crude, but avoids random label flips.)
    generic = means.mean(axis=1)
    return int(np.argmax(generic))


def anchor_scores(means_z: np.ndarray, features: list[str]) -> np.ndarray:
    """
    Return per-component anchor scores in z-space, higher = more call-like.
    Mirrors the logic in pick_call_component, but returns the full score vector.
    """
    def has(f): return f in features
    def get(f): return means_z[:, features.index(f)]

    K = means_z.shape[0]
    score = np.zeros(K, dtype=float)
    total_w = 0.0

    pos = {"nb_ratio_med":1.0, "nb_ratio_p75":0.6, "tonality_med":1.0, "rms_med":0.7, "td_energy_med":0.8}
    neg = {"flux_med":0.8, "bw":0.5, "zcr":0.5}

    for f,w in pos.items():
        if has(f): score += w * get(f); total_w += w
    for f,w in neg.items():
        if has(f): score += w * (-get(f)); total_w += w

    if has("td_slope_med"):
        w = 0.8
        score += w * (-get("td_slope_med"))
        total_w += w

    if has("cfreq"):
        w = 0.3
        score += w * (-np.abs(get("cfreq") - 55.0) / 55.0)
        total_w += w

    if total_w > 0:
        score = score / total_w
    else:
        # Fallback: energy-ish first, else overall mean
        if has("mfcc_00_mean"): return get("mfcc_00_mean")
        if has("lfcc_00_mean"): return get("lfcc_00_mean")
        return means_z.mean(axis=1)
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", required=True, help="Detections CSV filename (no path needed; script will search recursively)")
    ap.add_argument("--truth", default=None, help="Optional truth/selection filename (no path needed)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--time_tolerance", type=float, default=0.20)
    ap.add_argument("--features", default="tier1", choices=list(FEATURE_PRESETS.keys()),  help="Feature preset for CLUSTERING (Xc)")
    ap.add_argument("--anchor_features", default="tier1", choices=list(FEATURE_PRESETS.keys()), help="Feature preset for ORIENTATION (Xa)")
    ap.add_argument("--k", type=int, default=2, help="Number of GMM components (try 3 to split noise modes)")
    ap.add_argument("--collapse", default="top1",
                choices=["top1", "top2", "top_delta", "positive"], help="How to select call components when k>2.")
    ap.add_argument("--top_delta", type=float, default=0.15,
                    help="For collapse=top_delta: include any comp within delta of the top score.")
    ap.add_argument("--debug_orient", action="store_true",
                    help="Print anchor scores per component.")
    args = ap.parse_args()

    # ---------- Find detections (and truth if given) ----------
    detections_path = find_file(args.detections, Path.cwd())
    truth_path = find_file(args.truth, Path.cwd()) if args.truth else None

    # ---------- Build output path under results/GMM ----------
    project_root = Path.cwd()
    out_dir = project_root / "results" / "GMM"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_name = detections_path.stem + "_with_Clustered_labels.csv"
    out_path = out_dir / out_name

    print(f"Using detections: {detections_path}")
    if truth_path:
        print(f"Using truth: {truth_path}")
    print(f"Results will be written to: {out_path}")

    


    # ---------- Load detections ----------
    det = pd.read_csv(detections_path)
    for col in ["start_s", "end_s"]:
        if col not in det.columns:
            raise ValueError(f"Detections file must include '{col}'.")


    # ---------- Prepare features ----------
    # CLUSTER space (what the GMM learns on)
    cluster_feats = FEATURE_PRESETS[args.features]
    for col in cluster_feats:
        if col not in det.columns:
            raise ValueError(f"Missing required feature '{col}' in detections for clustering.")
    Xc = det[cluster_feats].copy()
    Xc = winsorize_df(Xc, cluster_feats, 1.0, 99.0)
    scaler_c = StandardScaler()
    Xc_z = scaler_c.fit_transform(Xc)

    # ANCHOR space (what we use to decide which component = call)
    anchor_feats = FEATURE_PRESETS[args.anchor_features]
    for col in anchor_feats:
        if col not in det.columns:
            raise ValueError(f"Missing required feature '{col}' in detections for anchor/orientation.")
    Xa = det[anchor_feats].copy()
    Xa = winsorize_df(Xa, anchor_feats, 1.0, 99.0)
    scaler_a = StandardScaler()
    Xa_z = scaler_a.fit_transform(Xa)

    # ---------- Fit GMM in CLUSTER space ----------
    gmm = GaussianMixture(
        n_components=args.k,
        covariance_type="diag",
        n_init=10,
        reg_covar=1e-4,
        random_state=42
    )
    gmm.fit(Xc_z)
    proba = gmm.predict_proba(Xc_z)  # (N, K)

    # ---------- Decide which component is "call" using ANCHOR space ----------
    # Get component means IN ANCHOR Z-SPACE via responsibility-weighted averages:
    means_anchor_z = component_means_in_space(proba, Xa_z)  # (K, D_anchor)

    scores = anchor_scores(means_anchor_z, anchor_feats)
    if args.debug_orient:
        print(f"[orient] anchor scores per comp: {np.round(scores, 3)}")

    # Decide which comps are "call"
    if args.collapse == "top1" or args.k == 2:
        call_idxs = [int(np.argmax(scores))]
    elif args.collapse == "top2":
        call_idxs = np.argsort(scores)[-2:].tolist()
    elif args.collapse == "top_delta":
        m = float(scores.max())
        call_idxs = np.where(scores >= m - args.top_delta)[0].tolist()
    elif args.collapse == "positive":
        # mark any component that scores above 0 (call-like side of the anchor scale)
        call_idxs = np.where(scores > 0.0)[0].tolist()
    else:
        call_idxs = [int(np.argmax(scores))]

    # p_call = sum of probabilities of all call-like components
    det["p_call"] = proba[:, call_idxs].sum(axis=1) if len(call_idxs) > 1 else proba[:, call_idxs[0]]
    det["gmm_label"] = (det["p_call"] >= args.threshold).astype(int)


    # ---------- Model-fit metric (BIC) ----------
    bic = gmm.bic(Xc_z)  # lower is better; use CLUSTER space
    print(f"BIC (k={args.k}): {bic:.2f}")

    # Save BIC to a small sidecar text file in results/GMM
    metrics_path = out_dir / (detections_path.stem + "_gmm_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"BIC_k{args.k}={bic:.6f}\n")
        f.write(f"threshold={args.threshold}\n")
        f.write(f"time_tolerance={args.time_tolerance}\n")
    print(f"\nMetrics saved: {metrics_path}")


    # ---------- Optional evaluation ----------
    if truth_path is not None:
        truth = load_truth(str(truth_path))  # use resolved path
        y_true = mark_truth_overlaps(det[["start_s", "end_s"]], truth, tol=args.time_tolerance)
        det["y_true"] = y_true

        y_pred = det["gmm_label"].to_numpy()
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = (2*precision*recall)/(precision+recall) if (precision+recall) else 0.0
        accuracy  = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

        print("\n=== Evaluation (GMM vs Truth) ===")
        print(f"Confusion matrix [[TN, FP],[FN, TP]] = [[{tn}, {fp}], [{fn}, {tp}]]")
        print(f"Accuracy : {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall   : {recall:.3f}")
        print(f"F1-score : {f1:.3f}")
        print(f"(threshold={args.threshold}, time_tolerance={args.time_tolerance}s)")


    # ---------- Save ----------
    metrics_path = out_dir / (detections_path.stem + "_gmm_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"BIC_k2={bic:.6f}\n")
        f.write(f"threshold={args.threshold}\n")
        f.write(f"time_tolerance={args.time_tolerance}\n")
    print(f"\nMetrics saved: {metrics_path}")
    # ---------- Save Raven-compatible table ----------
    raven_out = out_dir / (detections_path.stem + "_raven.txt")
    raven = pd.DataFrame({
        "Selection": np.arange(1, len(det) + 1),
        "Begin Time (s)": det["start_s"],
        "End Time (s)": det["end_s"],
        "Channel": 1,
        "p_call": det["p_call"],
        "gmm_label": det["gmm_label"]
    })
    raven.to_csv(raven_out, sep="\t", index=False)
    print(f"\nRaven table saved: {raven_out}")
    det.to_csv(out_path, index=False)  # <-- save to results/GMM/<name>_with_gmm.csv
    print(f"\nSaved: {out_path}   rows={len(det)}  cols={det.shape[1]}")

if __name__ == "__main__":
    main()
