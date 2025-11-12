
"""
GMM noise-vs-call clustering + optional evaluation against true selections.
BEST VERSION
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler



def find_file(filename: str, search_root: Path = Path.cwd()) -> Path:
    matches = list(search_root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find file {filename} under {search_root}")
    return matches[0].resolve()


FEATURE_PRESETS = {

    "core" : [ "lfcc_01_p90", "lfcc_00_mean", "lfcc_01_std",  "nb_ratio_med", "lfcc_02_p90", "gtcc_01_p90" ,"snr_local_db", "ber_20_100__100_300", "zcr_mean" ],



    "anchor_core" : ["tonality_med", "nb_ratio_med", "crest_med", "sfm_med", "snr_local_db"],

}


def anchor_scores(means_z: np.ndarray, features: list[str], style: str = "normal") -> np.ndarray:
    """
    Compute 'call-likeness' per GMM component in anchor space.

    - style='normal'  : short, tonal, narrow-band calls (higher tonality/NB/SNR; lower SFM better)
    - style='dt'      : Downtown downsweeps (steep negative slope, longer duration, lower SFM/centroid better)
    - style='hybrid'  : max(normal, dt)

    Uses ONLY the features actually present in `features`. Missing features are ignored (weight=0).
    Signed weights: positive means "higher is better", negative means "lower is better".
    """

    K = means_z.shape[0]
    feat_index = {f: i for i, f in enumerate(features)}

    def get(f: str) -> np.ndarray:
        # returns column or zeros if not present (keeps shapes consistent)
        idx = feat_index.get(f, None)
        if idx is None:
            return np.zeros(K, dtype=float)
        return means_z[:, idx]

    # ---- Define signed-weight dictionaries (ONLY keys that might appear) ----
    # Normal (tonal) orientation
    w_norm = {
        "tonality_med":   +1.0,
        "nb_ratio_med":   +0.8,
        "crest_med":      +0.6,
        "sfm_med":        -1.0,  # lower flatness = better
        "snr_local_db":   +0.8,
    }

    # Downtown (DT) orientation — tuned for long, steep downsweeps with low centroid/flatness
    w_dt = {
        "sfm_med":        -1.2,  # lower flatness preferred
        "snr_local_db":   +0.6,  # cleaner events get a small boost
        "nb_ratio_med":   -0.8,  # lower NB ratio = better (broader-band)                                                               
    }

    def _score_with(weights: dict[str, float]) -> np.ndarray:
        s = np.zeros(K, dtype=float)
        wsum = 0.0
        for f, w in weights.items():
            if f in feat_index and w != 0.0:
                s += w * get(f)
                wsum += abs(w)
        if wsum == 0.0:
            # fallback: average over available dims to avoid NaNs
            return means_z.mean(axis=1)
        return s / wsum

    if style == "normal":
        return _score_with(w_norm)
    elif style == "dt":
        return _score_with(w_dt)
    elif style == "hybrid":
        sn = _score_with(w_norm)
        sd = _score_with(w_dt)
        return np.maximum(sn, sd)
    else:
        return _score_with(w_norm)


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
    import re

    def _normalize(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", s.strip().lower())

    try:
        df = pd.read_csv(path, engine="python", sep=None)
        header_mode = "headered"
    except Exception:
        df = pd.read_csv(path, sep=r"\s+", engine="python")
        header_mode = "headered"

    need_headerless_try = False
    if df.shape[1] == 1:
        need_headerless_try = True
    else:
        if all(isinstance(c, (int, np.integer)) for c in df.columns):
            need_headerless_try = True

    if need_headerless_try:
        try:
            df = pd.read_csv(path, engine="python", sep=None, header=None)
            header_mode = "headerless"
        except Exception:
            df = pd.read_csv(path, sep=r"\s+", engine="python", header=None)
            header_mode = "headerless"
        df.columns = [f"col{i}" for i in range(df.shape[1])]

    if header_mode == "headered":
        cols_norm = {_normalize(c): c for c in df.columns}
        start_candidates = [
            c for norm, c in cols_norm.items()
            if any(k in norm for k in ["begintime", "starttime", "start", "begin", "onset"])
        ]
        end_candidates = [
            c for norm, c in cols_norm.items()
            if any(k in norm for k in ["endtime", "end", "offset", "finish"])
        ]

        if start_candidates and end_candidates:
            start_col = start_candidates[0]
            end_col = end_candidates[0]
            out = df[[start_col, end_col]].copy()
            out.columns = ["start_s", "end_s"]
            out = out.apply(pd.to_numeric, errors="coerce")
            out = out.dropna(subset=["start_s", "end_s"])
            return out

    df_num = df.copy()
    for c in df_num.columns:
        df_num[c] = pd.to_numeric(df_num[c], errors="coerce")
    numeric_cols = [c for c in df_num.columns if pd.api.types.is_numeric_dtype(df_num[c])]
    if len(numeric_cols) < 2:
        raise ValueError("Truth file has fewer than two numeric columns; cannot infer start/end.")

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
            frac_valid = (e[mask] > s[mask]).mean()
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
    X = df.copy()
    for c in cols:
        if c in X.columns:
            vals = X[c].to_numpy()
            lo_v, hi_v = np.percentile(vals[~np.isnan(vals)], [lo, hi])
            X[c] = np.clip(X[c], lo_v, hi_v)
    return X


def component_means_in_space(responsibilities: np.ndarray, X_z: np.ndarray) -> np.ndarray:
    R = responsibilities
    Nk = R.sum(axis=0) + 1e-12
    means = (R.T @ X_z) / Nk[:, None]
    return means



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", required=True)
    ap.add_argument("--truth", default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--time_tolerance", type=float, default=0.20)
    ap.add_argument("--features", default="core", choices=list(FEATURE_PRESETS.keys()))
    ap.add_argument("--anchor_features", default="anchor_core", choices=list(FEATURE_PRESETS.keys()))
    ap.add_argument("--anchor_style", default="normal", choices=["normal", "dt"], help="Orientation scoring: normal, downtown (dt)")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--collapse", default="top1", choices=["top1", "top2", "top_delta", "positive"])
    ap.add_argument("--top_delta", type=float, default=0.15)
    ap.add_argument("--debug_orient", action="store_true")

    # Step 1 extra controls
    ap.add_argument("--covariance_type", default="diag", choices=["diag", "full", "tied", "spherical"])
    ap.add_argument("--n_init", type=int, default=10)
    ap.add_argument("--init_params", default="kmeans", choices=["kmeans", "random"])
    ap.add_argument("--reg_covar", type=float, default=1e-4)
    ap.add_argument("--winsor_lo", type=float, default=1.0)
    ap.add_argument("--winsor_hi", type=float, default=99.0)
    args = ap.parse_args()

    detections_path = find_file(args.detections, Path.cwd())
    truth_path = find_file(args.truth, Path.cwd()) if args.truth else None

    out_dir = Path.cwd() / "results" / "GMM"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (detections_path.stem + "_with_Clustered_labels.csv")

    print(f"Using detections: {detections_path}")
    if truth_path:
        print(f"Using truth: {truth_path}")
    print(f"Results will be written to: {out_path}")

    # Load detections
    det = pd.read_csv(detections_path)
    for col in ["start_s", "end_s"]:
        if col not in det.columns:
            raise ValueError(f"Detections file must include '{col}'.")
        
    # --- PREFILTER STEP (Fix 1) ---
    # Only run it if the needed columns exist, so it won't crash on older CSVs.
    if all(c in det.columns for c in ["snr_local_db", "tonality_med"]):
        prefilter_mask = (
            (det["snr_local_db"] > 0.0) |
            (det["tonality_med"] > 5.0)
        )
        before_n = len(det)
        det = det[prefilter_mask].reset_index(drop=True)
        after_n = len(det)
        print(f"[prefilter] kept {after_n}/{before_n} detections "
              f"({before_n - after_n} removed as obvious noise)")
    else:
        print("[prefilter] skipped (columns missing)")


    # Prepare features
    cluster_feats = choose_features(det, args.features)
    # Ensure numeric
    for c in cluster_feats:
        det[c] = pd.to_numeric(det[c], errors="coerce")

    Xc = det[cluster_feats].copy()
    Xc = winsorize_df(Xc, cluster_feats, args.winsor_lo, args.winsor_hi)

    # --- NEW: clean + impute ---
    Xc = Xc.replace([np.inf, -np.inf], np.nan)
    for c in cluster_feats:
        med = np.nanmedian(Xc[c].to_numpy())
        if not np.isfinite(med):
            med = 0.0
        Xc[c] = Xc[c].fillna(med)

    scaler_c = StandardScaler()
    Xc_z = scaler_c.fit_transform(Xc)

    anchor_feats = choose_features(det, args.anchor_features)
    for c in anchor_feats:
        det[c] = pd.to_numeric(det[c], errors="coerce")

    Xa = det[anchor_feats].copy()
    Xa = winsorize_df(Xa, anchor_feats, args.winsor_lo, args.winsor_hi)

    # (optional but consistent) clean + impute anchor too
    Xa = Xa.replace([np.inf, -np.inf], np.nan)
    for c in anchor_feats:
        med = np.nanmedian(Xa[c].to_numpy())
        if not np.isfinite(med):
            med = 0.0
        Xa[c] = Xa[c].fillna(med)

    scaler_a = StandardScaler()
    Xa_z = scaler_a.fit_transform(Xa)

    Xc_z = np.nan_to_num(Xc_z, nan=0.0, posinf=0.0, neginf=0.0)
    Xa_z = np.nan_to_num(Xa_z, nan=0.0, posinf=0.0, neginf=0.0)

    # Fit GMM
    gmm = GaussianMixture(
        n_components=args.k,
        covariance_type=args.covariance_type,
        n_init=args.n_init,
        reg_covar=args.reg_covar,
        init_params=args.init_params,
        random_state=42
    )
    gmm.fit(Xc_z)
    proba = gmm.predict_proba(Xc_z)


    # Orientation in anchor space
    means_anchor_z = component_means_in_space(proba, Xa_z)
    scores = anchor_scores(means_anchor_z, anchor_feats, style=args.anchor_style)
    if args.debug_orient:
        print(f"[orient] anchor scores per comp: {np.round(scores, 3)}")
        print(f"[orient] style={args.anchor_style}  feats={anchor_feats}")

    # Save anchor scores
    orient_out = out_dir / (detections_path.stem + "_anchor_scores.csv")
    pd.DataFrame({"component": np.arange(len(scores)), "anchor_score": scores}).to_csv(orient_out, index=False)

    # Decide call-like component(s)
    call_idxs: list[int]
    if args.collapse == "top1" or args.k == 2:
        call_idxs = [int(np.argmax(scores))]
    elif args.collapse == "top2":
        call_idxs = np.argsort(scores)[-2:].tolist()
    elif args.collapse == "top_delta":
        m = float(scores.max())
        call_idxs = np.where(scores >= m - args.top_delta)[0].tolist()
        if len(call_idxs) == 0:
            call_idxs = [int(np.argmax(scores))]
    elif args.collapse == "positive":
        call_idxs = np.where(scores > 0.0)[0].tolist()
        if len(call_idxs) == 0:
            call_idxs = [int(np.argmax(scores))]
    else:
        call_idxs = [int(np.argmax(scores))]

    print(f"[orient] selected call component(s): {call_idxs}")


    

    # Log run config
    run_cfg = {
        "detections": str(detections_path),
        "truth": str(truth_path) if truth_path else None,
        "k": args.k,
        "covariance_type": args.covariance_type,
        "n_init": args.n_init,
        "init_params": args.init_params,
        "reg_covar": args.reg_covar,
        "winsor_lo": args.winsor_lo,
        "winsor_hi": args.winsor_hi,
        "features": args.features,
        "anchor_features": args.anchor_features,
        "threshold": args.threshold,
        "time_tolerance": args.time_tolerance,
    }
    cfg_path = out_dir / (detections_path.stem + "_gmm_config.json")
    with open(cfg_path, "w", encoding="utf-8") as cf:
        json.dump(run_cfg, cf, indent=2)

    # Posteriors to p_call and label
    det["p_call"] = proba[:, call_idxs].sum(axis=1) if len(call_idxs) > 1 else proba[:, call_idxs[0]]
    # det["gmm_label"] = (det["p_call"] >= args.threshold).astype(int)
    # det["p_call_eff"] = det["p_call"] * det["conf_score"].clip(0,1)
    # det["gmm_label"] = (det["p_call_eff"] >= args.threshold).astype(int)
    # After you compute det["p_call"], insert:
    use_col = "p_call"
    if all(c in det.columns for c in ["td_slope_med", "dur"]):
        slope = pd.to_numeric(det["td_slope_med"], errors="coerce").fillna(0.0).to_numpy()
        dur   = pd.to_numeric(det["dur"],          errors="coerce").fillna(0.0).to_numpy()

        dt_score  = np.clip(((-slope - 120.0) / 120.0), 0.0, 1.0)   # 0 at −120, 1 at ≤ −240 Hz/s
        dur_score = np.clip(((dur - 0.35) / 0.35),       0.0, 1.0)  # 0 at 0.35s, 1 at ≥ 0.70s
        lift = np.minimum(dt_score, dur_score)

        gamma = 0.6
        p = det["p_call"].to_numpy(dtype=float)
        p_eff = p + gamma * lift * (1.0 - p)      # monotone boost
        det["p_call_eff"] = np.maximum(p, p_eff)
        use_col = "p_call_eff"

    det["gmm_label"] = (det[use_col] >= args.threshold).astype(int)



    # BIC
    bic = gmm.bic(Xc_z)
    print(f"BIC (k={args.k}): {bic:.2f}")
    metrics_path = out_dir / (detections_path.stem + "_gmm_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"BIC_k{args.k}={bic:.6f}\n")
        f.write(f"threshold={args.threshold}\n")
        f.write(f"time_tolerance={args.time_tolerance}\n")
    print(f"\nMetrics saved: {metrics_path}")

    # Evaluation (optional)
    if truth_path is not None:
        truth = load_truth(str(truth_path))
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


    print(pd.DataFrame(gmm.means_, columns=cluster_feats))


    # Save outputs
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
    det.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}   rows={len(det)}  cols={det.shape[1]}")


def mark_truth_overlaps(det_df: pd.DataFrame, truth_df: pd.DataFrame, tol: float) -> pd.Series:
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


if __name__ == "__main__":
    main()
