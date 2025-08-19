# Multi-feature CPD for Bryde's call detection (recall-first)
# Stage A (CPD): RMS + spectral flux + anchor (~42 Hz) narrow-band ratio + anchor tonality
# Downsweep slope is computed but NOT used in CPD; it's exported for later use (Stage B / clustering).

import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from scipy.signal import butter, filtfilt, get_window, stft
from scipy.ndimage import median_filter
import ruptures as rpt
from scipy.signal import find_peaks
import argparse

# --- Directories (based on your tree) ---
PROJ_ROOT   = Path(__file__).resolve().parents[1]
DATA_DIR    = PROJ_ROOT / "data" / "chunks"   # where your WAV + label files live
OUT_DIR     = PROJ_ROOT / "results"           # all detections/outputs
OUT_DIR.mkdir(parents=True, exist_ok=True)


def resolve_file(name_or_path: str | Path, base: Path) -> Path:
    p = Path(name_or_path)
    # 1) direct hit (absolute or relative to CWD)
    if p.is_file():
        return p.resolve()
    # 2) relative to project root (lets you pass 'data/chunks/1hrsamples/x.wav')
    q = (PROJ_ROOT / p).resolve()
    if q.is_file():
        return q
    # 3) under the base directory (e.g., data/chunks/<arg>)
    q = (base / p).resolve()
    if q.is_file():
        return q
    # 4) last resort: search by filename anywhere under base
    hits = list(base.rglob(p.name))
    if hits:
        return hits[0].resolve()
    raise FileNotFoundError(f"Could not find '{name_or_path}'. "
                            f"Tried CWD, PROJ_ROOT, {base}, and recursive search.")

def default_out_path(audio_path: Path, suffix: str) -> Path:
    """results/<audio_stem>_<suffix>.csv"""
    return (OUT_DIR / f"{audio_path.stem}_{suffix}").with_suffix(".csv")

def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--audio",  required=True,
                        help="Audio file (filename in data/chunks or full path).")
    parser.add_argument("--labels", default=None,
                        help="Label file (filename in same folder or full path).")
    parser.add_argument("--out",    default=None,
                        help="Output CSV path (defaults to results/<audio_stem>_output.csv).")

# ------------------- paths -------------------
# HERE = Path(__file__).resolve().parent
# AUDIO_PATH   = (HERE / "APR22_s1.wav").resolve()
# LABELS_PATH  = (HERE / "APR22_s1_selections.txt").resolve()
# OUT_DETS_CSV = (HERE / "APR22_s1_detections.csv").resolve()

# ------------------- front-end filtering & framing -------------------
# Recall-first band: keep low anchor + upper energy
LOW_HZ, HIGH_HZ = 40, 280
FRAME_S = 0.050
HOP_S   = 0.010

# ------------------- post-proc durations -------------------
MIN_EVENT_S = 0.3
MIN_GAP_S   = 0.10
ONSET_TOL_S = 0.25



# ------------------- CPD preset (recall tilt) -------------------
MODEL    = "l2"
PENALTY  = 70.0
Z_THRESH = 1.50
SMOOTH_S = 1.00
FLUX_W   = 0.30  # only used if you switch to 2-feature; we use multi-feature below

# ------------------- STFT config for spectral features -------------------
SPEC_NFFT   = 2048   # ~3.9 Hz bins at fs=8k, resolves ~42 Hz well
SPEC_FMAX   = 280
ANCHOR_BAND = (36.0, 48.0)   # slightly wider than before to catch drift
UPPER_BAND  = (60.0, 220.0)  # MT/BT/TD energy region
TD_BAND     = (150.0, 220.0) # for exported downsweep feature ONLY

# ------------------- Stage B toggle (keep OFF while maximising recall) -------------------
APPLY_STAGE_B = False

# ------------------- core helpers -------------------
def butter_bandpass(low, high, fs, order=4):
    ny = 0.5 * fs
    b, a = butter(order, [low/ny, high/ny], btype="band")
    return b, a

def frame_signal(x, fs, frame_s, hop_s, window="hann"):
    frame = int(round(frame_s*fs))
    hop   = int(round(hop_s*fs))
    pad = (-(len(x)-frame) % hop) % hop
    if pad: x = np.pad(x, (0, pad), mode="constant")
    n_frames = 1 + (len(x)-frame)//hop
    W = get_window(window, frame, fftbins=True).astype(np.float32)
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n_frames, frame),
        strides=(x.strides[0]*hop, x.strides[0]),
        writeable=False
    ).copy()
    frames *= W[None, :]
    times = np.arange(n_frames)*hop_s
    return frames, times

def frame_rms_from_frames(frames):
    return np.sqrt((frames**2).mean(axis=1) + 1e-12)

def spectral_flux(frames):
    mag = np.abs(np.fft.rfft(frames, axis=1))
    diff = np.diff(mag, axis=0, prepend=mag[:1])
    return np.maximum(diff, 0.0).sum(axis=1)

def robust_z(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return (x - med) / (1.4826*mad)

def segments_from_cpd(z, hop_s, penalty, model="l2", min_size_frames=3):
    algo = rpt.Pelt(model=model, min_size=min_size_frames).fit(z.reshape(-1,1))
    bkps = algo.predict(pen=penalty)
    segs, start = [], 0
    for b in bkps:
        segs.append((start, b))
        start = b
    return [(s*hop_s, e*hop_s) for s, e in segs]

def pick_event_segments(z, segs, z_thresh, min_event_s, min_gap_s, hop_s):
    events = []
    for (ts, te) in segs:
        s_idx = int(round(ts / hop_s))
        e_idx = int(round(te / hop_s))
        if e_idx <= s_idx: continue
        z_seg = z[s_idx:e_idx]
        if not len(z_seg): continue
        z_med = np.median(z_seg)
        z_p75 = np.percentile(z_seg, 75)
        if (z_med > z_thresh) and (z_p75 > 0.9*z_thresh) and ((te - ts) >= min_event_s):
            events.append([ts, te])
    # merge close events
    merged = []
    for st, en in events:
        if merged and (st - merged[-1][1] <= min_gap_s):
            merged[-1][1] = max(merged[-1][1], en)
        else:
            merged.append([st, en])
    return merged

def read_raven_table(path: Path, prefer_channel=None, dedupe_tol=1e-3) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", engine="python")
    begin_col = next(c for c in df.columns if c.lower().startswith("begin time"))
    end_col   = next(c for c in df.columns if c.lower().startswith("end time"))
    if prefer_channel is not None and "Channel" in df.columns:
        df = df[df["Channel"] == prefer_channel].copy()
    gt = df[[begin_col, end_col]].rename(columns={begin_col:"start_s", end_col:"end_s"})
    gt = gt[gt["end_s"] >= gt["start_s"]]
    r = gt.copy()
    r["sr"] = (r["start_s"] / dedupe_tol).round().astype(int)
    r["er"] = (r["end_s"]   / dedupe_tol).round().astype(int)
    gt = (r.drop_duplicates(subset=["sr", "er"])
            .drop(columns=["sr", "er"])
            .sort_values("start_s")
            .reset_index(drop=True))
    return gt

def det_onsets_from_cross(dets, z, hop_s, z_lo):
    ons = []
    z = np.asarray(z, dtype=np.float32)
    for ts, te in dets:
        s = int(round(ts / hop_s)); e = max(s+1, int(round(te / hop_s)))
        seg = z[s:e]
        if seg.size == 0: continue
        above = np.flatnonzero(seg >= z_lo)
        idx = above[0] if above.size else int(seg.argmax())
        ons.append((s + idx) * hop_s)
    return np.array(ons, dtype=np.float32)

def evaluate_with_cross(dets, gt, tol_s, z, hop_s, z_lo):
    det_on = det_onsets_from_cross(dets, z, hop_s, z_lo)
    if det_on.size == 0:
        return 0.0, 0.0, 0.0, 0, 0, len(gt)
    gt_on = gt["start_s"].values
    used_det = np.zeros(len(det_on), bool); used_gt = np.zeros(len(gt_on), bool)
    tp = 0
    for i, t in enumerate(gt_on):
        diffs = np.abs(det_on - t); diffs[used_det] = np.inf
        j = int(np.argmin(diffs))
        if diffs[j] <= tol_s:
            used_det[j] = True; used_gt[i] = True; tp += 1
    fp = int((~used_det).sum()); fn = int((~used_gt).sum())
    p = tp/(tp+fp) if (tp+fp) else 0.0
    r = tp/(tp+fn) if (tp+fn) else 0.0
    f1 = 2*p*r/(p+r) if (p+r) else 0.0
    return p, r, f1, tp, fp, fn

def split_events_by_valley(
    dets, z, hop_s,
    z_split,
    min_valley_s=0.08,
    min_event_s=0.30,
    merge_gap_s=0.02
):
    """Split [ts,te] at valleys (z < z_split) for at least min_valley_s, then lightly re-merge."""
    z = np.asarray(z, dtype=np.float32)
    min_valley_frames = max(1, int(round(min_valley_s / hop_s)))

    pieces = []
    for ts, te in dets:
        s = int(round(ts / hop_s))
        e = max(s+1, int(round(te / hop_s)))
        seg = z[s:e]
        if seg.size <= 1:
            pieces.append([ts, te]); continue

        below = seg < z_split
        splits = []
        i = 0
        while i < len(below):
            if below[i]:
                j = i
                while j < len(below) and below[j]:
                    j += 1
                if (j - i) >= min_valley_frames:
                    k = i + int(np.argmin(seg[i:j]))
                    splits.append(s + k)
                i = j
            else:
                i += 1

        if not splits:
            pieces.append([ts, te])
        else:
            idxs = [s] + sorted(splits) + [e]
            for a, b in zip(idxs[:-1], idxs[1:]):
                dur = (b - a) * hop_s
                if dur >= min_event_s:
                    pieces.append([a * hop_s, b * hop_s])

    pieces.sort(key=lambda p: p[0])
    merged = []
    for st, en in pieces:
        if merged and (st - merged[-1][1] <= merge_gap_s):
            merged[-1][1] = max(merged[-1][1], en)
        else:
            merged.append([st, en])
    return merged

# ------------------- STFT + per-frame spectral features -------------------
def compute_stft(x, fs, frame_s, hop_s, nfft=SPEC_NFFT, fmax=SPEC_FMAX):
    nper = int(round(frame_s * fs))
    hop  = int(round(hop_s * fs))
    # ensure valid overlap even if short
    step = max(1, min(hop, nper-1))
    nover = nper - step
    nfft_use = max(nfft, nper)
    f, t, Z = stft(x, fs=fs, nperseg=nper, noverlap=nover, nfft=nfft_use,
                   boundary=None, padded=False)
    keep = f <= fmax
    return f[keep], t, np.abs(Z[keep, :])**2  # power

def band_mask(f, lo, hi):
    return (f >= lo) & (f <= hi)

def per_frame_narrowband_ratio(P, f):
    """Energy in ~42 Hz band vs upper band per frame."""
    a = band_mask(f, *ANCHOR_BAND)
    u = band_mask(f, *UPPER_BAND)
    Ea = P[a, :].sum(axis=0) + 1e-12
    Eu = P[u, :].sum(axis=0) + 1e-12
    return (Ea / Eu).astype(np.float32)

def per_frame_anchor_tonality(P, f):
    """Crest factor in anchor band: max/mean (higher = more tonal)."""
    a = band_mask(f, *ANCHOR_BAND)
    Pa = P[a, :]
    mean = Pa.mean(axis=0) + 1e-12
    peak = Pa.max(axis=0) + 1e-12
    return (peak / mean).astype(np.float32)

# ---- Downsweep helpers (EXPORTED ONLY; not used in CPD z) ----
def band_centroid_hz(P, f, band):
    m = band_mask(f, *band)
    if not m.any():
        return np.zeros(P.shape[1], dtype=np.float32)
    sub = P[m, :]
    wsum = sub.sum(axis=0) + 1e-12
    cf   = ((f[m][:, None] * sub).sum(axis=0) / wsum)  # Hz
    return cf.astype(np.float32)

def downsweep_slope_hz_per_s(P, f, hop_s, band=TD_BAND):
    c_hz = band_centroid_hz(P, f, band)
    dcf = np.diff(c_hz, prepend=c_hz[:1]) / max(hop_s, 1e-6)  # Hz/s
    return dcf.astype(np.float32)


def split_by_bandflux_peaks(
    dets, P, f_axis, hop_s,
    band=(80.0, 220.0),
    smooth_s=0.20,             # short smooth to reveal local valleys
    min_peak_dist_s=0.55,      # min spacing between calls
    valley_min_s=0.06,         # valley must persist this long
    min_piece_s=0.25,          # don't create tiny slivers
    min_prom_frac=0.12,        # prominence as % of local dynamic range
    height_frac=0.25           # height as % above local baseline
):
    """
    For each detection [ts,te], look at band-limited energy E(t) in 80–220 Hz.
    Find peaks with adaptive prominence/height (per segment), and split at
    the deepest valley between adjacent significant peaks.
    """
    band_mask = (f_axis >= band[0]) & (f_axis <= band[1])
    if not np.any(band_mask):
        return dets[:]  # nothing to do

    E = P[band_mask, :].sum(axis=0).astype(np.float32)  # band-limited energy per frame

    # smooth energy
    k = max(1, int(round(smooth_s / hop_s)))
    if k > 1:
        E_s = median_filter(E, size=k)
    else:
        E_s = E

    out = []
    min_peak_dist = max(1, int(round(min_peak_dist_s / hop_s)))
    valley_min_f  = max(1, int(round(valley_min_s     / hop_s)))
    min_piece_f   = max(1, int(round(min_piece_s      / hop_s)))

    for ts, te in dets:
        s = int(round(ts / hop_s))
        e = max(s+2, int(round(te / hop_s)))
        seg = E_s[s:e]
        if seg.size < 3:
            out.append([ts, te]); continue

        # Adaptive thresholds from local distribution
        lo = float(np.percentile(seg, 20))
        hi = float(np.percentile(seg, 90))
        dyn = max(hi - lo, 1e-6)
        prom   = min_prom_frac * dyn
        height = lo + height_frac * dyn

        peaks, props = find_peaks(seg, distance=min_peak_dist,
                                  prominence=prom, height=height)
        if len(peaks) <= 1:
            out.append([ts, te]); continue

        # propose cuts at the deepest point between adjacent peaks
        cuts = []
        for p1, p2 in zip(peaks[:-1], peaks[1:]):
            if (p2 - p1) < valley_min_f:
                continue
            valley_rel = p1 + int(np.argmin(seg[p1:p2]))
            if (valley_rel) < min_piece_f or (len(seg) - valley_rel) < min_piece_f:
                continue
            cuts.append(s + valley_rel)

        if not cuts:
            out.append([ts, te]); continue

        idxs = [s] + cuts + [e]
        for a, b in zip(idxs[:-1], idxs[1:]):
            if (b - a) >= min_piece_f:
                out.append([a * hop_s, b * hop_s])

    # tiny de‑glitch merging only (5 ms)
    out.sort(key=lambda p: p[0])
    merged = []
    for st, en in out:
        if merged and (st - merged[-1][1] <= 0.005):
            merged[-1][1] = max(merged[-1][1], en)
        else:
            merged.append([st, en])
    return merged


def local_cpd_refine(
    dets, z_scalar, hop_s,
    model="l2",
    pen_ratio=0.6,         # use a lower penalty than global (e.g., 60%)
    min_event_s=0.25,
    min_sep_s=0.45         # don't create sub-events closer than this
):
    """
    Inside each detected [ts,te], re-run PELT with a lower penalty on z_scalar
    to add internal change points. Only keep splits that create reasonably
    sized pieces and are separated by >= min_sep_s.
    """
    out = []
    z = np.asarray(z_scalar, dtype=np.float32)
    min_event_f = max(1, int(round(min_event_s / hop_s)))
    min_sep_f   = max(1, int(round(min_sep_s   / hop_s)))

    for ts, te in dets:
        s = int(round(ts / hop_s))
        e = max(s+2, int(round(te / hop_s)))
        seg = z[s:e]
        if seg.size < 4:
            out.append([ts, te]); continue

        # run PELT locally with lower penalty
        try:
            algo = rpt.Pelt(model=model, min_size=min_event_f).fit(seg.reshape(-1,1))
            # estimate a local penalty from seg variance; fallback to global scale
            local_pen = max(1.0, np.var(seg) * 5.0)  # rough; we scale it below
            pen = max(1.0, local_pen * pen_ratio)
            bkps = algo.predict(pen=pen)
        except Exception:
            out.append([ts, te]); continue

        # convert to absolute frame indices
        idxs = [s]
        for b in bkps:
            ab = s + int(b)
            if (ab - idxs[-1]) >= min_sep_f and (e - ab) >= min_event_f:
                idxs.append(ab)
        if idxs[-1] != e:
            idxs.append(e)

        # build sub-events
        for a, b in zip(idxs[:-1], idxs[1:]):
            if (b - a) >= min_event_f:
                out.append([a * hop_s, b * hop_s])

    out.sort(key=lambda p: p[0])
    # avoid re-merging; we want separate calls
    return out

# ------------------- main -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline CPD")
    add_common_args(parser)
    args = parser.parse_args()

    AUDIO_PATH = resolve_file(args.audio, DATA_DIR)
    LABELS_PATH = resolve_file(args.labels, DATA_DIR) if args.labels else None
    OUT_DETS_CSV = Path(args.out) if args.out else default_out_path(AUDIO_PATH, "detections")
    OUT_DETS_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Audio + labels
    x, fs = sf.read(AUDIO_PATH)
    if x.ndim > 1: x = x.mean(axis=1)
    x = x.astype(np.float32)

    gt = None
    if LABELS_PATH:
        raw_df = pd.read_csv(LABELS_PATH, sep="\t", engine="python")
        gt = read_raven_table(LABELS_PATH, prefer_channel=None, dedupe_tol=1e-3)
        print(f"Raw GT rows: {len(raw_df)} | Unique selections after de-dupe: {len(gt)}")
    else:
        print("No labels provided: skipping evaluation.")

    # Front-end filter
    b, a = butter_bandpass(LOW_HZ, HIGH_HZ, fs, order=4)
    xf = filtfilt(b, a, x)

    # Frame-domain features (RMS + flux)
    frames, times = frame_signal(xf, fs, FRAME_S, HOP_S, window="hann")
    rms  = frame_rms_from_frames(frames).astype(np.float32)
    flux = spectral_flux(frames).astype(np.float32)

    # Spectral per-frame features aligned to frames
    f_axis, t_stft, P = compute_stft(xf, fs, FRAME_S, HOP_S, nfft=SPEC_NFFT, fmax=SPEC_FMAX)
    L = min(len(times), len(t_stft), len(rms), len(flux), P.shape[1])
    if (len(times) != L) or (len(t_stft) != L):
        times = times[:L]; rms = rms[:L]; flux = flux[:L]; P = P[:, :L]

    nb_ratio = per_frame_narrowband_ratio(P, f_axis)      # anchor vs upper
    tonality = per_frame_anchor_tonality(P, f_axis)       # crest factor in anchor band

    # Export-only downsweep features (NOT used in CPD z)
    td_slope_hzs = downsweep_slope_hz_per_s(P, f_axis, HOP_S, band=TD_BAND)
    td_energy    = P[band_mask(f_axis, *TD_BAND), :].sum(axis=0)  # TD-band energy per frame

    # Smooth + robust z on CPD features
    k = max(1, int(round(SMOOTH_S / HOP_S)))
    z_rms   = robust_z(median_filter(rms,      size=k))
    z_flux  = robust_z(median_filter(flux,     size=k))
    z_nb    = robust_z(median_filter(nb_ratio, size=k))
    z_tone  = robust_z(median_filter(tonality, size=k))

    # Combine (recall-tilted weights) — NOTE: NO downsweep in CPD z
    W_RMS, W_FLUX, W_NB, W_TONE = 0.15, 0.15, 0.45, 0.25
    z = (W_RMS*z_rms + W_FLUX*z_flux + W_NB*z_nb + W_TONE*z_tone).astype(np.float32)

    # CPD → candidate segments
    min_size_frames = max(3, int(round(MIN_EVENT_S / HOP_S)))
    segs = segments_from_cpd(z, HOP_S, PENALTY, model=MODEL, min_size_frames=min_size_frames)
    dets = pick_event_segments(z, segs, Z_THRESH, MIN_EVENT_S, MIN_GAP_S, HOP_S)

    # Valley split (helps separate close short calls)
    z_split = max(0.5*Z_THRESH, Z_THRESH - 0.7)
    dets = split_events_by_valley(
        dets, z, HOP_S,
        z_split=z_split,
        min_valley_s=0.08,
        min_event_s=0.30,
        merge_gap_s=0.02
    )

    # --- NEW 1: band-limited flux peak splitter (80–220 Hz) ---
    dets = split_by_bandflux_peaks(
        dets, P, f_axis, HOP_S,
        band=(80.0, 220.0),
        smooth_s=0.20,
        min_peak_dist_s=0.65,
        valley_min_s=0.06,
        min_piece_s=0.30,
        min_prom_frac=0.12,
        height_frac=0.25
    )

    # --- NEW 2: local CPD re-segmentation with lower penalty ---
    dets = local_cpd_refine(
        dets, z, HOP_S,
        model="l2",
        pen_ratio=0.7,         # try 0.5–0.7 if it under-splits / over-splits
        min_event_s=0.25,
        min_sep_s=0.6
    )

    # Save raw detections
    pd.DataFrame(dets, columns=["start_s", "end_s"]).to_csv(OUT_DETS_CSV, index=False)
    print(f"Saved detections: {OUT_DETS_CSV} (n={len(dets)})")

    # Evaluate (CPD only)
    z_lo = max(0.6*Z_THRESH, Z_THRESH - 0.4)
    print("\n=== CPD (multi-feature, recall-first; no downsweep in z) ===")
    print(f"Params | model={MODEL} | pen={PENALTY:.1f} | z={Z_THRESH:.2f} | smooth={SMOOTH_S:.2f}s | W=[{W_RMS:.2f},{W_FLUX:.2f},{W_NB:.2f},{W_TONE:.2f}]")
    p, r, f1, tp, fp, fn = evaluate_with_cross(dets, gt, ONSET_TOL_S, z, HOP_S, z_lo)
    print(f"GT n={len(gt)} | Det n={len(dets)}")
    print(f"TP={tp}  FP={fp}  FN={fn}")
    print(f"Precision={p:.3f}  Recall={r:.3f}  F1={f1:.3f}")

    # ---- Stage B (OFF by default; keep permissive if you enable it) ----
    accepted = dets
    if APPLY_STAGE_B:
        # Example: extremely light pass—keep anything whose median nb_ratio is above a gentle floor
        accepted = []
        for st, en in dets:
            s = int(round(st / HOP_S)); e = max(s+1, int(round(en / HOP_S)))
            nb_med = float(np.median(nb_ratio[s:e])) if e > s else 0.0
            if nb_med >= 0.55:  # very permissive; adjust only if you need to shave worst FPs
                accepted.append([st, en])

        p2, r2, f12, tp2, fp2, fn2 = evaluate_with_cross(accepted, gt, ONSET_TOL_S, z, HOP_S, z_lo)
        print("\n=== CPD + very light Stage B ===")
        print(f"GT n={len(gt)} | Det n={len(accepted)}")
        print(f"TP={tp2}  FP={fp2}  FN={fn2}")
        print(f"Precision={p2:.3f}  Recall={r2:.3f}  F1={f12:.3f}")

    # ---- Export per-event features for GMM / later analysis ----
    rows = []
    for st, en in accepted:
        s = int(round(st / HOP_S)); e = max(s+1, int(round(en / HOP_S)))
        row = {
            "start_s": st, "end_s": en, "dur": en - st,
            "nb_ratio_med":  float(np.median(nb_ratio[s:e])) if e > s else 0.0,
            "nb_ratio_p75":  float(np.percentile(nb_ratio[s:e], 75)) if e > s else 0.0,
            "tonality_med":  float(np.median(tonality[s:e])) if e > s else 0.0,
            "rms_med":       float(np.median(rms[s:e])) if e > s else 0.0,
            "flux_med":      float(np.median(flux[s:e])) if e > s else 0.0,
            # Downsweep stats (not used in CPD)
            "td_slope_med":  float(np.median(td_slope_hzs[s:e])) if e > s else 0.0,
            "td_slope_min":  float(np.min(td_slope_hzs[s:e])) if e > s else 0.0,
            "td_energy_med": float(np.median(td_energy[s:e])) if e > s else 0.0,
        }
        rows.append(row)
    feats_csv = default_out_path(AUDIO_PATH, "events_for_gmm")
    pd.DataFrame(rows).to_csv(feats_csv, index=False)
    print(f"Saved features for GMM: {feats_csv} (n={len(rows)})")

    # Raven TSV (for quick visual QA)
    raven = pd.DataFrame({
        "Selection": np.arange(1, len(dets)+1),
        "View": "Spectrogram 1",
        "Channel": 1,
        "Begin Time (s)": [st for st, _ in dets],
        "End Time (s)":   [en for _, en in dets],
    })
    raven_tsv = OUT_DETS_CSV.with_suffix(".raven.tsv")
    raven.to_csv(raven_tsv, sep="\t", index=False)
    print(f"Saved Raven selections: {raven_tsv}")

    # Optional quick visual
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 3))
        ax.plot(times[:len(z)], z, label="multi-feature z")
        for i, (st, en) in enumerate(dets):
            ax.axvspan(st, en, alpha=0.20, color="tab:orange", label="CPD det" if i == 0 else None)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("z-score"); ax.set_title("Detections over multi-feature z")
        if dets: ax.legend(loc="upper right")
        plt.show()
    except Exception as e:
        print("(Plotting skipped:", e, ")")
