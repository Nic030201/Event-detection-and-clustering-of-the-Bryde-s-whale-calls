# Recall-first multi-feature CPD for Bryde's whale calls
# Stage A: CPD on multi-feature scalar stream (RMS, flux, anchor-band ratio, tonality)
# Plus a low-threshold run-length fallback; take UNION → valley split → evaluate.
# Stage B is off (we focus on recall). Downsweep features are computed and exported only.

import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from scipy.signal import butter, filtfilt, get_window, stft
from scipy.ndimage import median_filter
import ruptures as rpt

# ------------------- paths -------------------
HERE = Path(__file__).resolve().parent
AUDIO_PATH   = (HERE / "APR22_s1.wav").resolve()
LABELS_PATH  = (HERE / "APR22_s1_selections.txt").resolve()
OUT_DETS_CSV = (HERE / "APR22_s1_detections.csv").resolve()

# ------------------- front-end filtering & framing -------------------
LOW_HZ, HIGH_HZ = 20, 280   # wider low cut keeps faint anchors
FRAME_S = 0.050
HOP_S   = 0.010

# ------------------- post-proc durations -------------------
MIN_EVENT_S = 0.30          # shorter to keep brief calls
MIN_GAP_S   = 0.15
ONSET_TOL_S = 0.25

# Hard cap for any single event (tune from your Raven: many Bryde calls are ~0.5–1.2 s)
MAX_EVENT_S = 1.25     # try 1.1–1.4; smaller -> more splits

# ------------------- CPD settings (recall-tilted) -------------------
MODEL    = "l2"             # kernel (rbf) is OFF for now
PENALTY  = 70.0
SMOOTH_S = 1.00

# ------------------- Hysteresis acceptance (on scalar z) -------------------
Z_LO = 1.15                 # low gate: segment median must exceed OR...
Z_HI = 1.95                 # ...segment 95th percentile must exceed this
Z_HI_QUANT = 95

# ------------------- STFT config for spectral features -------------------
SPEC_NFFT   = 2048          # ~3.9 Hz bins @ 8 kHz, resolves ~42 Hz
SPEC_FMAX   = 280
ANCHOR_BAND = (36.0, 48.0)  # ~42 Hz anchor (slightly wide)
UPPER_BAND  = (60.0, 220.0) # upper energy region (MT/BT/TD)
TD_BAND     = (150.0, 220.0)# downsweep band (export only)

# ------------------- Stage B toggle (keep off while maximising recall) -------------------
APPLY_STAGE_B = False

# ------------------- helpers -------------------
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

def compute_stft(x, fs, frame_s, hop_s, nfft=SPEC_NFFT, fmax=SPEC_FMAX):
    nper = int(round(frame_s * fs))
    hop  = int(round(hop_s * fs))
    step = max(1, min(hop, nper-1))   # ensure 0 <= noverlap < nperseg
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

def pick_event_segments_hysteresis(z_scalar, segs, z_lo, z_hi, min_event_s, min_gap_s, hop_s, hi_quant=95):
    """
    Accept a segment if it has either enough low support (median > z_lo)
    OR brief high evidence (quantile@hi_quant > z_hi).
    """
    events = []
    for (ts, te) in segs:
        s_idx = int(round(ts / hop_s)); e_idx = int(round(te / hop_s))
        if e_idx <= s_idx: continue
        z_seg = z_scalar[s_idx:e_idx]
        if z_seg.size == 0: continue
        if (np.median(z_seg) > z_lo) or (np.percentile(z_seg, hi_quant) > z_hi):
            if (te - ts) >= min_event_s:
                events.append([ts, te])

    # merge close events
    merged = []
    for st, en in events:
        if merged and (st - merged[-1][1] <= min_gap_s):
            merged[-1][1] = max(merged[-1][1], en)
        else:
            merged.append([st, en])
    return merged

def threshold_run_segments(z_scalar, hop_s, thr=1.15, min_len_s=0.20, max_gap_s=0.05):
    """
    Low-threshold run-length detector on z_scalar with tiny morphological closing.
    Returns list of [start_s, end_s].
    """
    z = np.asarray(z_scalar, dtype=np.float32)
    above = z >= thr
    max_gap = max(1, int(round(max_gap_s / hop_s)))
    i = 0
    runs = []
    while i < len(above):
        if above[i]:
            j = i
            while j < len(above) and above[j]:
                j += 1
            # stitch tiny gap if another run starts within max_gap
            k = j
            while (k < len(above)) and (k - j <= max_gap) and not above[k]:
                k += 1
            if k < len(above) and above[k]:
                above[j:k] = True
                i = j
                continue
            start_s = i * hop_s
            end_s   = j * hop_s
            if (end_s - start_s) >= min_len_s:
                runs.append([start_s, end_s])
            i = j
        else:
            i += 1
    return runs

def merge_interval_lists(a, b, merge_gap_s=0.04):
    """Union of two [st,en] lists with a tiny merge gap."""
    all_ = sorted(a + b, key=lambda p: p[0])
    if not all_:
        return []
    merged = [all_[0]]
    for st, en in all_[1:]:
        if st - merged[-1][1] <= merge_gap_s:
            merged[-1][1] = max(merged[-1][1], en)
        else:
            merged.append([st, en])
    return merged

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
                    k = i + int(np.argmin(seg[i:j]))  # deepest valley
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



from scipy.signal import find_peaks  # you already import scipy.signal; this is ok

def enforce_max_event_length(
    dets, z_scalar, hop_s,
    max_len_s=1.25,
    min_event_s=0.25,
    valley_quantile=0.40,   # valley threshold = 40th percentile of the segment
    min_valley_s=0.08,      # valley must last this long
):
    """
    If a detected [ts,te] is longer than max_len_s, split it at valleys of z_scalar.
    We place cuts at local minima below a per-segment threshold (quantile),
    requiring the 'below' run to last at least min_valley_s. Greedy repeat until all pieces fit.
    """
    z = np.asarray(z_scalar, dtype=np.float32)
    max_len_frames  = int(round(max_len_s   / hop_s))
    min_event_frames= int(round(min_event_s / hop_s))
    min_valley_frames = max(1, int(round(min_valley_s / hop_s)))

    out = []
    for ts, te in dets:
        s0 = int(round(ts / hop_s))
        e0 = max(s0+1, int(round(te / hop_s)))
        # nothing to do?
        if e0 - s0 <= max_len_frames:
            out.append([ts, te]); continue

        # work on this long segment
        starts = [s0]
        ends   = [e0]

        # we’ll iteratively refine each piece until all <= max_len_frames
        i = 0
        while i < len(starts):
            s = starts[i]; e = ends[i]
            if e - s <= max_len_frames:
                i += 1; continue

            seg = z[s:e]
            if seg.size < 2:
                i += 1; continue

            # segment-specific valley threshold
            thr = np.quantile(seg, valley_quantile)
            below = seg <= thr

            # find all continuous "below" runs that are long enough
            cuts = []
            j = 0
            while j < len(below):
                if below[j]:
                    k = j
                    while k < len(below) and below[k]:
                        k += 1
                    if (k - j) >= min_valley_frames:
                        # choose the deepest point in this valley as split
                        rel = j + int(np.argmin(seg[j:k]))
                        # don't cut too close to the ends
                        if (rel - s) >= min_event_frames and (e - (s+rel)) >= min_event_frames:
                            cuts.append(s + rel)
                    j = k
                else:
                    j += 1

            # If no valid valleys, force a split at the lowest point near the middle window
            if not cuts:
                mid_left  = s + max(min_event_frames, (e - s)//3)
                mid_right = e - max(min_event_frames, (e - s)//3)
                mid_left  = min(mid_left, e - min_event_frames - 1)
                mid_right = max(mid_right, s + min_event_frames + 1)
                if mid_right > mid_left:
                    rel = np.argmin(z[mid_left:mid_right]) + mid_left
                    cuts = [rel]

            # Use only one cut at a time (deepest valley); re-check pieces next loop
            if cuts:
                # pick the cut that gives two pieces both as close as possible to max_len (avoid tiny slivers)
                # score = minimum of piece lengths; maximize it
                best = None; best_score = -1
                for c in cuts:
                    L1 = c - s; L2 = e - c
                    score = min(L1, L2)
                    if score > best_score and L1 >= min_event_frames and L2 >= min_event_frames:
                        best = c; best_score = score
                if best is None:
                    i += 1; continue
                # replace the current piece with two sub-pieces
                starts[i:i+1] = [s, best]
                ends[i:i+1]   = [best, e]
            else:
                i += 1

        # append all final pieces
        for s, e in zip(starts, ends):
            if (e - s) >= min_event_frames:
                out.append([s * hop_s, e * hop_s])

    # keep output sorted and non-overlapping
    out.sort(key=lambda p: p[0])
    # tiny merge to eliminate numerically-adjacent borders only
    merged = []
    for st, en in out:
        if merged and (st - merged[-1][1] <= 0.01):  # 10 ms glue only
            merged[-1][1] = max(merged[-1][1], en)
        else:
            merged.append([st, en])
    return merged


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

# ------------------- main -------------------
if __name__ == "__main__":
    # Audio + labels
    x, fs = sf.read(AUDIO_PATH)
    if x.ndim > 1: x = x.mean(axis=1)
    x = x.astype(np.float32)

    raw_df = pd.read_csv(LABELS_PATH, sep="\t", engine="python")
    gt = read_raven_table(LABELS_PATH, prefer_channel=None, dedupe_tol=1e-3)
    print(f"Raw GT rows: {len(raw_df)} | Unique selections after de-dupe: {len(gt)}")

    # Front-end filter
    b, a = butter_bandpass(LOW_HZ, HIGH_HZ, fs, order=4)
    xf = filtfilt(b, a, x)

    # Frame-domain features
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
    td_energy    = P[band_mask(f_axis, *TD_BAND), :].sum(axis=0)

    # Smooth + robust z on CPD features
    k = max(1, int(round(SMOOTH_S / HOP_S)))
    z_rms   = robust_z(median_filter(rms,      size=k))
    z_flux  = robust_z(median_filter(flux,     size=k))
    z_nb    = robust_z(median_filter(nb_ratio, size=k))
    z_tone  = robust_z(median_filter(tonality, size=k))

    # Scalar z for gating/plots
    W_RMS, W_FLUX, W_NB, W_TONE = 0.15, 0.15, 0.45, 0.25
    z_scalar = (W_RMS*z_rms + W_FLUX*z_flux + W_NB*z_nb + W_TONE*z_tone).astype(np.float32)

    # ---------- CPD: l2-PELT on scalar z ----------
    min_size_frames = max(3, int(round(MIN_EVENT_S / HOP_S)))
    algo = rpt.Pelt(model=MODEL, min_size=min_size_frames).fit(z_scalar.reshape(-1,1))
    bkps = algo.predict(pen=PENALTY)

    # Convert breakpoints to segments [ts,te] in seconds
    segs = []
    start = 0
    for b in bkps:
        segs.append((start, b))
        start = b
    segs = [(s*HOP_S, e*HOP_S) for s, e in segs]

    # A) CPD segments -> hysteresis acceptance
    dets_cpd = pick_event_segments_hysteresis(
        z_scalar, segs,
        z_lo=Z_LO, z_hi=Z_HI, hi_quant=Z_HI_QUANT,
        min_event_s=MIN_EVENT_S, min_gap_s=MIN_GAP_S, hop_s=HOP_S
    )

    # B) No-CPD threshold fallback (very permissive, recall-first)
    runs_thr = threshold_run_segments(
        z_scalar, hop_s=HOP_S,
        thr=1.20,          # low gate
        min_len_s=0.20,    # short runs allowed
        max_gap_s=0.05
    )

    # C) Union of both detectors
    dets = merge_interval_lists(dets_cpd, runs_thr, merge_gap_s=0.05)

    # D) Valley split to separate paired short calls
    dets = split_events_by_valley(
        dets, z_scalar, HOP_S,
        z_split=max(0.5*Z_LO, Z_LO - 0.5),
        min_valley_s=0.08,
        min_event_s=0.30,
        merge_gap_s=0.02
    )

    # Enforce a maximum event length to stop multi-call swallowing
    dets = enforce_max_event_length(
        dets, z_scalar, HOP_S,
        max_len_s=MAX_EVENT_S,
        min_event_s=MIN_EVENT_S,   # keep in sync
        valley_quantile=0.40,      # 0.35–0.45 works well
        min_valley_s=0.08
    )


    # Save raw detections
    pd.DataFrame(dets, columns=["start_s", "end_s"]).to_csv(OUT_DETS_CSV, index=False)
    print(f"Saved detections: {OUT_DETS_CSV} (n={len(dets)})")

    # Evaluate (CPD + fallback union)
    z_eval_lo = max(0.6*Z_LO, Z_LO - 0.3)
    print("\n=== Recall-first detector (CPD ∪ threshold) ===")
    print(f"Params | model={MODEL} | pen={PENALTY:.1f} | smooth={SMOOTH_S:.2f}s | W=[{W_RMS:.2f},{W_FLUX:.2f},{W_NB:.2f},{W_TONE:.2f}]")
    p, r, f1, tp, fp, fn = evaluate_with_cross(dets, gt, ONSET_TOL_S, z_scalar, HOP_S, z_eval_lo)
    print(f"GT n={len(gt)} | Det n={len(dets)}")
    print(f"TP={tp}  FP={fp}  FN={fn}")
    print(f"Precision={p:.3f}  Recall={r:.3f}  F1={f1:.3f}")

    # ---- Export per-event features for clustering / later filters ----
    rows = []
    for st, en in dets:
        s = int(round(st / HOP_S)); e = max(s+1, int(round(en / HOP_S)))
        row = {
            "start_s": st, "end_s": en, "dur": en - st,
            "nb_ratio_med":  float(np.median(nb_ratio[s:e])) if e > s else 0.0,
            "nb_ratio_p75":  float(np.percentile(nb_ratio[s:e], 75)) if e > s else 0.0,
            "tonality_med":  float(np.median(tonality[s:e])) if e > s else 0.0,
            "rms_med":       float(np.median(rms[s:e])) if e > s else 0.0,
            "flux_med":      float(np.median(flux[s:e])) if e > s else 0.0,
            # Downsweep stats (export only)
            "td_slope_med":  float(np.median(td_slope_hzs[s:e])) if e > s else 0.0,
            "td_slope_min":  float(np.min(td_slope_hzs[s:e])) if e > s else 0.0,
            "td_energy_med": float(np.median(td_energy[s:e])) if e > s else 0.0,
        }
        rows.append(row)
    feats_csv = OUT_DETS_CSV.with_name("events_for_gmm.csv")
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
        ax.plot(times[:len(z_scalar)], z_scalar, label="multi-feature z")
        for i, (st, en) in enumerate(dets):
            ax.axvspan(st, en, alpha=0.20, color="tab:orange", label="Det" if i == 0 else None)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("z-score"); ax.set_title("Detections over z")
        if dets: ax.legend(loc="upper right")
        plt.show()
    except Exception as e:
        print("(Plotting skipped:", e, ")")
