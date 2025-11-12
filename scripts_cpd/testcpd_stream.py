# CPD_streaming.py 
# Streaming runner for your Bryde's CPD detector on very long audio.
# CURRENT BEST USES C.I.

import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from scipy.signal import butter, filtfilt, get_window, stft, find_peaks, spectrogram, resample_poly, sosfiltfilt
from scipy.ndimage import median_filter
import ruptures as rpt
import time
import argparse
import matplotlib.pyplot as plt
from scipy.signal.windows import dpss
import math
from matplotlib.colors import PowerNorm, Normalize
from matplotlib.ticker import ScalarFormatter


# --- Directories (based on your tree) ---
PROJ_ROOT   = Path(__file__).resolve().parents[1]
DATA_DIR    = PROJ_ROOT / "data" / "chunks"   # where your WAV + label files live
OUT_DIR     = PROJ_ROOT / "results" / "CPD"         # all detections/outputs
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
                        help="Output CSV path (defaults to results/CPD/<audio_stem>_output.csv).")

# ------------------- front-end filtering & framing -------------------
LOW_HZ, HIGH_HZ = 45, 240      
FRAME_S = 0.050
HOP_S   = 0.010

# ------------------- post-proc durations -------------------
MIN_EVENT_S = 0.20
DUR_MAX_MAIN = 0.7 # soft upper cap for main (non-burst) events
MIN_GAP_S   = 0.03
ONSET_TOL_S = 0.25

# ------------------- CPD preset (recall tilt) -------------------
MODEL    = "l2"
PENALTY  = 40.0
Z_THRESH = 1.2
SMOOTH_S = 0.50

# ------------------- STFT config for spectral features -------------------
SPEC_NFFT   = 2048
SPEC_FMAX   = 280
ANCHOR_BAND = (45.0, 65.0)
UPPER_BAND  = (60.0, 240.0)
TD_BAND     = (150.0, 220.0)  # exported only

# ------------------- splitters (your current settings) -------------------
# valley split is inside split_events_by_valley()
# band-limited flux peak splitter:
PEAK_BAND            = (80.0, 220.0)
PEAK_SMOOTH_S        = 0.12
PEAK_MIN_DIST_S      = 0.30
PEAK_VALLEY_MIN_S    = 0.05
PEAK_MIN_PIECE_S     = 0.2
PEAK_MIN_PROM_FRAC   = 0.1
PEAK_HEIGHT_FRAC     = 0.15

# local CPD refine:
LOCAL_PEN_RATIO      = 0.70
LOCAL_MIN_EVENT_S    = 0.20
LOCAL_MIN_SEP_S      = 0.40


# --- plotting/debug toggles ---
DEBUG_PLOTS  = False                  # per-chunk z-plot
PLOT_AT_END  = True                  # final stitched plot
PLOT_SAVEPATH = OUT_DIR / "debug_zstream_full.png"   # was a bare string

# accumulator for plotting across chunks
plot_accum = {
    "t": [],       # absolute times
    "z": [],       # fused z
    "dets": []     # list of (st_abs, en_abs)
}

# =======================================================================================
#                                  Core helpers
# =======================================================================================
_t0 = time.time()

def _rolling_mean_std(x: np.ndarray, win: int):
    """Centered rolling mean/std with 'same' length. Win must be >=1."""
    if win <= 1 or x.size == 0:
        return x.astype(np.float32), np.zeros_like(x, dtype=np.float32)
    w = int(win)
    ones = np.ones(w, dtype=np.float32)
    sum1 = np.convolve(x.astype(np.float32), ones, mode="same")
    sum2 = np.convolve((x*x).astype(np.float32), ones, mode="same")
    m = sum1 / w
    v = sum2 / w - m*m
    v[v < 0.0] = 0.0
    return m, np.sqrt(v + 1e-12)

def _progress(chunk_idx, start_frame, n_frames_total, fs, total_kept):
    done_s   = start_frame / fs
    total_s  = n_frames_total / fs
    frac     = min(1.0, done_s / total_s if total_s else 0.0)
    elapsed  = max(1e-6, time.time() - _t0)
    eta_s    = elapsed * (1-frac) / max(frac, 1e-6)
    print(f"[{chunk_idx:04d}] {done_s/3600:6.2f}/{total_s/3600:6.2f} h  "
          f"({frac*100:5.1f}%) | kept={total_kept} | "
          f"elapsed={elapsed/60:5.1f}m  ETA={eta_s/60:5.1f}m")

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
    times = np.arange(n_frames)*hop_s   # only used for alignment
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

def split_events_by_valley(
    dets, z, hop_s,
    z_split,
    min_valley_s=0.08,
    min_event_s=0.30,
    merge_gap_s=0.02
):
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

def split_by_anchor_valleys(
    dets, P, f_axis, hop_s,
    band=(35.0, 50.0),
    smooth_s=0.12,
    valley_drop_frac=0.16,
    min_valley_s=0.05,
    min_piece_s=0.18
):
    m = (f_axis >= band[0]) & (f_axis <= band[1])
    if not np.any(m):
        return dets[:]
    E = P[m, :].sum(axis=0).astype(np.float32)

    k = max(1, int(round(smooth_s / hop_s)))
    E_s = median_filter(E, size=k) if k > 1 else E

    out = []
    min_valley_f = max(1, int(round(min_valley_s / hop_s)))
    min_piece_f  = max(1, int(round(min_piece_s  / hop_s)))

    for ts, te in dets:
        s = int(round(ts / hop_s)); e = max(s+2, int(round(te / hop_s)))
        seg = E_s[s:e]
        if seg.size < 3:
            out.append([ts, te]); continue

        lo = float(np.percentile(seg, 15))
        hi = float(np.percentile(seg, 85))
        dyn = max(hi - lo, 1e-6)
        thr = hi - valley_drop_frac * dyn  # valleys below this count

        cuts = []
        i = 1
        while i < len(seg)-1:
            # local valley + deep enough
            if seg[i] < seg[i-1] and seg[i] <= seg[i+1] and seg[i] <= thr:
                j = i
                # extend flat valley
                while j < len(seg)-1 and seg[j+1] <= thr:
                    j += 1
                if (j - i + 1) >= min_valley_f:
                    cuts.append(s + i + int(np.argmin(seg[i:j+1])))
                i = j + 1
            else:
                i += 1

        if not cuts:
            out.append([ts, te]); continue
        idxs = [s] + cuts + [e]
        for a, b in zip(idxs[:-1], idxs[1:]):
            if (b - a) >= min_piece_f:
                out.append([a * hop_s, b * hop_s])

    out.sort(key=lambda p: p[0])
    # tiny deglitch merge (≤ one hop)
    merged = []
    for st, en in out:
        if merged and (st - merged[-1][1]) <= hop_s:
            merged[-1][1] = max(merged[-1][1], en)
        else:
            merged.append([st, en])
    return merged

def compute_stft(x, fs, frame_s, hop_s, nfft=SPEC_NFFT, fmax=SPEC_FMAX):
    nper = int(round(frame_s * fs))
    hop  = int(round(hop_s * fs))
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
    a = band_mask(f, *ANCHOR_BAND)
    u = band_mask(f, *UPPER_BAND)
    Ea = P[a, :].sum(axis=0) + 1e-12
    Eu = P[u, :].sum(axis=0) + 1e-12
    return (Ea / Eu).astype(np.float32)

def per_frame_anchor_tonality(P, f):
    a = band_mask(f, *ANCHOR_BAND)
    Pa = P[a, :]
    mean = Pa.mean(axis=0) + 1e-12
    peak = Pa.max(axis=0) + 1e-12
    return (peak / mean).astype(np.float32)

def band_centroid_hz(P, f, band):
    m = band_mask(f, *band)
    if not m.any():
        return np.zeros(P.shape[1], dtype=np.float32)
    sub = P[m, :]
    wsum = sub.sum(axis=0) + 1e-12
    cf   = ((f[m][:, None] * sub).sum(axis=0) / wsum)
    return cf.astype(np.float32)

def downsweep_slope_hz_per_s(P, f, hop_s, band=TD_BAND):
    c_hz = band_centroid_hz(P, f, band)
    dcf = np.diff(c_hz, prepend=c_hz[:1]) / max(hop_s, 1e-6)
    return dcf.astype(np.float32)

def split_by_bandflux_peaks(
    dets, P, f_axis, hop_s,
    band=(80.0, 220.0),
    smooth_s=0.20,
    min_peak_dist_s=0.65,
    valley_min_s=0.06,
    min_piece_s=0.30,
    min_prom_frac=0.12,
    height_frac=0.25
):
    band_m = (f_axis >= band[0]) & (f_axis <= band[1])
    if not np.any(band_m):
        return dets[:]
    E = P[band_m, :].sum(axis=0).astype(np.float32)
    k = max(1, int(round(smooth_s / hop_s)))
    E_s = median_filter(E, size=k) if k > 1 else E
    out = []
    min_peak_dist = max(1, int(round(min_peak_dist_s / hop_s)))
    valley_min_f  = max(1, int(round(valley_min_s   / hop_s)))
    min_piece_f   = max(1, int(round(min_piece_s    / hop_s)))
    for ts, te in dets:
        s = int(round(ts / hop_s)); e = max(s+2, int(round(te / hop_s)))
        seg = E_s[s:e]
        if seg.size < 3:
            out.append([ts, te]); continue
        lo = float(np.percentile(seg, 20))
        hi = float(np.percentile(seg, 90))
        dyn = max(hi - lo, 1e-6)
        prom   = min_prom_frac * dyn
        height = lo + height_frac * dyn
        peaks, _ = find_peaks(seg, distance=min_peak_dist, prominence=prom, height=height)
        if len(peaks) <= 1:
            out.append([ts, te]); continue
        cuts = []
        for p1, p2 in zip(peaks[:-1], peaks[1:]):
            if (p2 - p1) < valley_min_f: continue
            valley_rel = p1 + int(np.argmin(seg[p1:p2]))
            if (valley_rel) < min_piece_f or (len(seg) - valley_rel) < min_piece_f: continue
            cuts.append(s + valley_rel)
        if not cuts:
            out.append([ts, te]); continue
        idxs = [s] + cuts + [e]
        for a, b in zip(idxs[:-1], idxs[1:]):
            if (b - a) >= min_piece_f:
                out.append([a * hop_s, b * hop_s])
    out.sort(key=lambda p: p[0])
    merged = []
    for st, en in out:
        if merged and (st - merged[-1][1] <= 0.005):
            merged[-1][1] = max(merged[-1][1], en)
        else:
            merged.append([st, en])
    return merged

class RobustZEMA:
    def __init__(self, alpha=0.05, eps=1e-12):
        self.alpha, self.eps = alpha, eps
        self.med = None; self.mad = None
    def __call__(self, x):
        x = np.asarray(x, dtype=np.float32)
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med))) + self.eps
        if self.med is None:
            self.med, self.mad = med, mad
        else:
            a = self.alpha
            self.med = (1-a)*self.med + a*med
            self.mad = (1-a)*self.mad + a*mad
        return (x - self.med) / (1.4826*self.mad + self.eps)

rz_rms, rz_flux, rz_nb, rz_tone, rz_ent = RobustZEMA(), RobustZEMA(), RobustZEMA(), RobustZEMA(), RobustZEMA()
rz_td_energy = RobustZEMA()
rz_sfm   = RobustZEMA()
rz_crest = RobustZEMA()
rz_snr   = RobustZEMA()   # (or rz_ber if you prefer BER)





def band_entropy_tonality(P, f_axis, band):
    m = (f_axis >= band[0]) & (f_axis <= band[1])
    if not np.any(m):
        return np.zeros(P.shape[1], dtype=np.float32)
    sub = P[m, :].astype(np.float32) + 1e-12
    sub = sub / sub.sum(axis=0, keepdims=True)
    H = -(sub * np.log2(sub)).sum(axis=0)
    Hnorm = H / (np.log2(max(int(m.sum()), 2)))  # 0..1
    return (1.0 - Hnorm).astype(np.float32)      # high = tonal


def local_cpd_refine(
    dets, z_scalar, hop_s,
    model="l2",
    pen_ratio=0.7,
    min_event_s=0.2,
    min_sep_s=0.60
):
    out = []
    z = np.asarray(z_scalar, dtype=np.float32)
    min_event_f = max(1, int(round(min_event_s / hop_s)))
    min_sep_f   = max(1, int(round(min_sep_s   / hop_s)))
    for ts, te in dets:
        s = int(round(ts / hop_s)); e = max(s+2, int(round(te / hop_s)))
        seg = z[s:e]
        if seg.size < 4:
            out.append([ts, te]); continue
        try:
            algo = rpt.Pelt(model=model, min_size=min_event_f).fit(seg.reshape(-1,1))
            local_pen = max(1.0, np.var(seg) * 5.0)
            pen = max(1.0, local_pen * pen_ratio)
            bkps = algo.predict(pen=pen)
        except Exception:
            out.append([ts, te]); continue
        idxs = [s]
        for b in bkps:
            ab = s + int(b)
            if (ab - idxs[-1]) >= min_sep_f and (e - ab) >= min_event_f:
                idxs.append(ab)
        if idxs[-1] != e:
            idxs.append(e)
        for a, b in zip(idxs[:-1], idxs[1:]):
            if (b - a) >= min_event_f:
                out.append([a * hop_s, b * hop_s])
    out.sort(key=lambda p: p[0])
    return out




# =======================================================================================
# ------------------------ CPD EVALUATION HELPERS ------------------------
# =======================================================================================
def _interval_iou(a, b):
    """IoU of two [start, end] intervals in seconds."""
    s1, e1 = a
    s2, e2 = b
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / union if union > 0 else 0.0

def match_events(
    dets, truths, tol_s=0.5, use_iou=False, iou_min=0.3
):
    """
    dets, truths: list of (start_s, end_s)
    Returns:
      matches: list of (i_det, i_truth)
      fp_idx: indices of dets with no match
      fn_idx: indices of truths with no match
    """
    dets = list(dets); truths = list(truths)
    nD, nT = len(dets), len(truths)
    matched_det = np.zeros(nD, dtype=bool)
    matched_tru = np.zeros(nT, dtype=bool)
    matches = []

    for i_t, (ts, te) in enumerate(truths):
        best_j, best_score = -1, -1.0
        for j_d, (ds, de) in enumerate(dets):
            if matched_det[j_d]:
                continue
            if use_iou:
                score = _interval_iou((ds, de), (ts, te))
                ok = score >= iou_min
            else:
                # onset/offset tolerance match: any overlap after padding
                ok = (ds <= te + tol_s) and (de >= ts - tol_s)
                score = 1.0 if ok else -1.0
            if ok and score > best_score:
                best_score = score
                best_j = j_d
        if best_j >= 0:
            matched_det[best_j] = True
            matched_tru[i_t]  = True
            matches.append((best_j, i_t))

    fp_idx = np.where(~matched_det)[0].tolist()
    fn_idx = np.where(~matched_tru)[0].tolist()
    return matches, fp_idx, fn_idx

def compute_cpd_metrics(
    dets, truths, audio_dur_s, tol_s=0.5, use_iou=False, iou_min=0.3
):
    """
    Returns dict with: precision, recall, f1, fp_per_hour, tpr, arrays of onset/offset/duration errors.
    """
    matches, fp_idx, fn_idx = match_events(dets, truths, tol_s, use_iou, iou_min)
    tp = len(matches); fp = len(fp_idx); fn = len(fn_idx)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall    = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1        = (2*precision*recall / (precision+recall)) if (precision+recall)>0 else 0.0
    fp_per_hour = fp / max(audio_dur_s/3600.0, 1e-9)
    tpr = recall

    # error arrays for TPs
    onset_errs, offset_errs, dur_errs = [], [], []
    for j_d, i_t in matches:
        ds, de = dets[j_d]
        ts, te = truths[i_t]
        onset_errs.append(ds - ts)
        offset_errs.append(de - te)
        dur_errs.append((de - ds) - (te - ts))

    return dict(
        precision=precision, recall=recall, f1=f1, fp_per_hour=fp_per_hour, tpr=tpr,
        tp=tp, fp=fp, fn=fn,
        onset_errs=np.array(onset_errs, float),
        offset_errs=np.array(offset_errs, float),
        dur_errs=np.array(dur_errs, float),
        matches=matches, fp_idx=fp_idx, fn_idx=fn_idx
    )


def plot_z_overlay(t0, window_s, t_z, z, z_thresh, dets, truths, out_png):
    t1 = t0 + window_s
    sel = (t_z >= t0) & (t_z <= t1)
    plt.figure(figsize=(10, 3))
    plt.plot(t_z[sel], z[sel], lw=1)
    plt.axhline(z_thresh, ls='--', lw=1, label='Z_THRESH')
    # detections
    for (ds, de) in dets:
        if de < t0 or ds > t1: 
            continue
        xs = [max(ds, t0), min(de, t1)]
        plt.axvspan(xs[0], xs[1], color='tab:green', alpha=0.2)
    # truths
    for (ts, te) in truths:
        if te < t0 or ts > t1:
            continue
        xs = [max(ts, t0), min(te, t1)]
        plt.axvspan(xs[0], xs[1], color='tab:red', alpha=0.2)
    plt.xlim(t0, t1)
    plt.xlabel("Time (s)", fontsize=10); plt.ylabel("$z$-statistic", fontsize=10)
    plt.title("Composite detection statistic (close-up)", fontsize=11)
    plt.tight_layout()
    plt.legend()
    plt.savefig("results/figs/z_closeup.png", dpi=600, bbox_inches="tight")
    plt.close()

def save_z_closeup(time_s, z, out_png, t0=None, t1=None, threshold=None):
    # choose a window if not specified (e.g., around the max z)
    if t0 is None or t1 is None:
        t_peak = time_s[np.argmax(z)]
        half = 7.5
        t0, t1 = max(time_s.min(), t_peak-half), min(time_s.max(), t_peak+half)

    m = (time_s >= t0) & (time_s <= t1)
    t_win, z_win = time_s[m], z[m]

    if threshold is None:
        # or set your fixed threshold here
        threshold = np.quantile(z_win, 0.95)

    above = z_win >= threshold
    rising = np.diff(above.astype(int), prepend=0) == 1
    det_idx = np.where(rising)[0]

    plt.figure(figsize=(8, 3))
    plt.plot(t_win, z_win, linewidth=1.2, label="z(t)")
    plt.axhline(threshold, linestyle="--", linewidth=1.2, label="threshold")
    if det_idx.size:
        plt.scatter(t_win[det_idx], z_win[det_idx], s=18, marker="o", label="detections")
    plt.xlabel("Time (s)")
    plt.ylabel("z(t)")
    plt.title("Composite detection statistic: close-up view")
    plt.xlim(t0, t1)
    # Uncomment if you want a legend:
    # plt.legend(loc="upper right", frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def per_type_recall(dets, truths, truth_types, tol_s=0.5, use_iou=False, iou_min=0.3):
    """truth_types: list of strings aligned with truths."""
    out = {}
    if truth_types is None:
        return out
    types = sorted(set(truth_types))
    for ty in types:
        idx = [i for i, t in enumerate(truth_types) if t == ty]
        truths_ty = [truths[i] for i in idx]
        m = compute_cpd_metrics(dets, truths_ty, audio_dur_s=1.0, tol_s=tol_s, use_iou=use_iou, iou_min=iou_min)
        out[ty] = m['recall']
    return out

def build_summary_row(run_name, metrics, per_type=None):
    onset_ms = float(np.mean(metrics['onset_errs']))*1000 if metrics['onset_errs'].size else np.nan
    row = dict(
        run=run_name,
        precision=metrics['precision'],
        recall=metrics['recall'],
        f1=metrics['f1'],
        fp_per_hour=metrics['fp_per_hour'],
        onset_bias_ms=onset_ms,
        tp=metrics['tp'], fp=metrics['fp'], fn=metrics['fn']
    )
    if per_type:
        for ty, rec in per_type.items():
            row[f'recall_{ty}'] = rec
    return row





def plot_z_with_detections(
    time_s: np.ndarray,
    z: np.ndarray,
    det_intervals: list[tuple[float, float]] | None = None,
    det_times: np.ndarray | list[float] | None = None,
    det_window: float = 2.0,
    title: str = "Detections over multi-feature z",
    out_png: str | None = None,
):
    """
    Plot z(t) with shaded CPD detections.

    Provide either:
      - det_intervals: list of (start_s, end_s) tuples, OR
      - det_times + det_window: centres with a half-width window (i.e., [t - w/2, t + w/2])

    Args
    ----
    time_s : array-like
        Time axis (seconds).
    z : array-like
        Composite z-score (same length as time_s).
    det_intervals : list of (start, end)
        Detection intervals in seconds (optional).
    det_times : array-like
        Detection centre times in seconds (optional).
    det_window : float
        Width of the band if using det_times (seconds).
    title : str
        Figure title.
    out_png : str
        If given, save to this path (dpi=300). Otherwise just show.
    """
    time_s = np.asarray(time_s)
    z = np.asarray(z)

    # Build intervals if only centres provided
    if det_intervals is None and det_times is not None:
        det_times = np.asarray(det_times, dtype=float)
        half = det_window / 2.0
        det_intervals = [(float(t - half), float(t + half)) for t in det_times]

    fig, ax = plt.subplots(figsize=(9, 2.6))

    # Shade detection intervals
    if det_intervals:
        first = True
        for s, e in det_intervals:
            ax.axvspan(s, e, color = "black", alpha=0.15, label="CPD det" if first else None)
            first = False

    # z(t) trace
    ax.plot(time_s, z, linewidth=1.1, label="multi-feature z")

    ax.set_title(title)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("$z$-statistic", fontsize=10)
    ax.set_xlim(float(time_s.min()), float(time_s.max()))
    ax.legend(loc="upper right", frameon=False)
    plt.tight_layout()

    if out_png:
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()
        return out_png
    else:
        plt.show()

def event_confidence(z_seg, Z_THRESH, win_frames=5):
    """
    Measure temporal consistency of an event.

    z_seg:      1D array of your z() values for just this event
    Z_THRESH:   same decision threshold you already use
    win_frames: size of the local voting window (5 ≈ ~50 ms if hop ~10 ms)

    Returns:
        conf_score in [0,1], where higher = more locally stable "call" behaviour.
    """
    import numpy as np

    if len(z_seg) == 0:
        return 0.0

    # frame-level yes/no: is this frame whale-y enough?
    call_mask = (z_seg > Z_THRESH).astype(np.int32)

    half = win_frames // 2
    conf_vals = []

    for i in range(len(call_mask)):
        left  = max(0, i - half)
        right = min(len(call_mask), i + half + 1)
        window = call_mask[left:right]

        # fraction of the window that agrees "this is a call"
        conf_vals.append(window.mean())

    conf_vals = np.array(conf_vals, dtype=np.float32)

    # strict version: confidence is the weakest local agreement inside the event
    return float(np.mean(conf_vals))

def merge_close_events(events, gap_s=0.15):
    """
    Merge events that are basically the same call but got split by tiny gaps.

    events: list of dicts. Each dict MUST have "start_s" and "end_s".
            (This matches what we append to accepted_rows.)
    gap_s: max allowed gap between consecutive events to merge.

    Returns: new list of merged event dicts.
    """
    if not events:
        return []

    # sort by start time
    evs = sorted(events, key=lambda e: e["start_s"])
    merged = [evs[0].copy()]

    for ev in evs[1:]:
        prev = merged[-1]
        # if the gap between prev end and this start is tiny, merge
        if ev["start_s"] - prev["end_s"] < gap_s:
            # extend the previous event's end time
            prev["end_s"] = max(prev["end_s"], ev["end_s"])
            # recompute duration
            prev["dur_s"] = prev["end_s"] - prev["start_s"]

            # OPTIONAL: you can also fuse confidence fields.
            # For example, keep the max conf_score, max nb_med, etc.
            prev["conf_score"] = max(prev.get("conf_score", 0.0),
                                     ev.get("conf_score", 0.0))
            prev["z_med"]   = max(prev.get("z_med",   0.0),
                                  ev.get("z_med",   0.0))
            prev["z_p75"]   = max(prev.get("z_p75",   0.0),
                                   ev.get("z_p75",   0.0))
            prev["nb_med"]  = max(prev.get("nb_med",  0.0),
                                   ev.get("nb_med",  0.0))
            prev["ent_med"] = max(prev.get("ent_med", 0.0),
                                   ev.get("ent_med", 0.0))
            # if you store band_ratio or flatness later, also fuse with max

        else:
            # it's far enough to be its own detection
            merged.append(ev.copy())

    return merged
# =======================================================================================
#                           Per-chunk detector
# =======================================================================================
def detect_on_chunk(x_chunk, fs, chunk_s):
    """Run pipeline on one padded audio chunk.
       Returns dets_local (seconds relative to padded t0) and per-frame features needed by caller."""
    # filter
    b, a = butter_bandpass(LOW_HZ, HIGH_HZ, fs, order=4)
    xf = filtfilt(b, a, x_chunk)

    # frames + features
    frames, times = frame_signal(xf, fs, FRAME_S, HOP_S, window="hann")
    rms  = frame_rms_from_frames(frames).astype(np.float32)
    flux = spectral_flux(frames).astype(np.float32)

    # STFT features
    f_axis, t_stft, P = compute_stft(xf, fs, FRAME_S, HOP_S, nfft=SPEC_NFFT, fmax=SPEC_FMAX)
    L = min(len(times), len(t_stft), len(rms), len(flux), P.shape[1])
    times = times[:L]; rms = rms[:L]; flux = flux[:L]; P = P[:, :L]

    # smoothing window
    k = max(1, int(round(SMOOTH_S / HOP_S)))

    # ---------------- Per-frame features ----------------
    eps = 1e-12

    # Analysis bands
    B_ALL  = (f_axis >= 20.0) & (f_axis <= 300.0)
    B_CALL = (f_axis >= 30.0) & (f_axis <= 120.0)
    B_LOW  = (f_axis >= 20.0) & (f_axis < 100.0)
    B_HIGH = (f_axis >= 100.0) & (f_axis <= 300.0)

    Sb_all = P[B_ALL, :] + eps

    # Crest factor per frame (peak / mean in analysis band)
    peak  = Sb_all.max(axis=0)
    meanp = Sb_all.mean(axis=0)
    crest = peak / (meanp + eps)

    # Spectral flatness per frame (GM / AM in analysis band)
    gm = np.exp(np.mean(np.log(Sb_all), axis=0))
    am = meanp
    sfm = gm / (am + eps)

    # Local SNR per frame using band energies (call band vs side bands)
    P_call = P[B_CALL, :].mean(axis=0) + eps
    P_low  = P[B_LOW,  :].mean(axis=0) + eps
    P_high = P[B_HIGH, :].mean(axis=0) + eps
    snr_db = 10.0 * np.log10(P_call / ((P_low + P_high) * 0.5))

    # entropy-based tonality in mid band (ST/MT focus)
    ENT_BAND = (70.0, 160.0)
    ent_mid  = band_entropy_tonality(P, f_axis, ENT_BAND)
    z_ent    = rz_ent(median_filter(ent_mid, size=k))

    nb_ratio = per_frame_narrowband_ratio(P, f_axis)
    tonality = per_frame_anchor_tonality(P, f_axis)
    td_slope_hzs = downsweep_slope_hz_per_s(P, f_axis, HOP_S, band=TD_BAND)  # exported only
    td_energy    = P[band_mask(f_axis, *TD_BAND), :].sum(axis=0)             # exported only

    # rolling z-scores
    z_td_energy = rz_td_energy(median_filter(td_energy, size=k))
    z_nb    = rz_nb(   median_filter(nb_ratio, size=k) )
    z_tone  = rz_tone( median_filter(tonality, size=k))
    z_crest = rz_crest(median_filter(crest,   size=k))
    z_sfm   = rz_sfm(  median_filter(sfm,     size=k))
    z_snr   = rz_snr(  median_filter(snr_db,  size=k))

    # -------- Anchor-aligned CPD mix (your weights) --------
    # z(t) = 0.45·z_nb + 0.30·z_tone + 0.15·z_snr + 0.05·z_crest + 0.1·z_td_energy − 0.05·z_sfm
    z = (0.45 * z_nb
       + 0.30 * z_tone
       + 0.15 * z_snr
       + 0.05 * z_crest
       + 0.10 * z_td_energy
       - 0.05 * z_sfm).astype(np.float32)

    # ===================== CPD & Splitting =====================
    min_size_frames = max(3, int(round(MIN_EVENT_S / HOP_S)))
    n_frames_here   = len(z)

    BASE_N = int(round(max(chunk_s, 1e-6) / HOP_S))
    penalty_here = max(1.0, PENALTY * (np.log(max(n_frames_here, 2)) / np.log(max(BASE_N, 2))))

    segs = segments_from_cpd(z, HOP_S, penalty_here, model=MODEL, min_size_frames=min_size_frames)
    dets = pick_event_segments(z, segs, Z_THRESH, MIN_EVENT_S, MIN_GAP_S, HOP_S)

    # Valley split
    z_split = max(0.5 * Z_THRESH, Z_THRESH - 0.7)
    dets = split_events_by_valley(
        dets, z, HOP_S,
        z_split=z_split,
        min_valley_s=0.08,
        min_event_s=0.24,
        merge_gap_s=0.02
    )

    # band-limited flux peak splitter
    dets = split_by_bandflux_peaks(
        dets, P, f_axis, HOP_S,
        band=PEAK_BAND,
        smooth_s=PEAK_SMOOTH_S,
        min_peak_dist_s=PEAK_MIN_DIST_S,
        valley_min_s=PEAK_VALLEY_MIN_S,
        min_piece_s=PEAK_MIN_PIECE_S,
        min_prom_frac=PEAK_MIN_PROM_FRAC,
        height_frac=PEAK_HEIGHT_FRAC
    )

    # micro-split on anchor envelope (helps ST/MT multi-pulse)
    dets = split_by_anchor_valleys(
        dets, P, f_axis, HOP_S,
        band=(35.0, 50.0),
        smooth_s=0.12,
        valley_drop_frac=0.18,
        min_valley_s=0.05,
        min_piece_s=0.18
    )

    # local CPD refine
    dets = local_cpd_refine(
        dets, z, HOP_S,
        model="l2",
        pen_ratio=LOCAL_PEN_RATIO,
        min_event_s=LOCAL_MIN_EVENT_S,
        min_sep_s=LOCAL_MIN_SEP_S
    )

 # ===================== POST FILTER (with confidence) =====================
    rescue_ct = 0
    accepted_rows = []   # CHANGED: we'll store dicts instead of just [st,en]

    for st, en in dets:
        s = int(round(st / HOP_S))
        e = max(s+1, int(round(en / HOP_S)))
        if e <= s:
            continue

        # clip to frame bounds
        s = max(0, min(s, len(z)-1))
        e = max(0, min(e, len(z)))

        # --- per-event stats
        z_seg   = z[s:e]
        z_med   = float(np.median(z_seg))
        z_p75   = float(np.percentile(z_seg, 75))
        ent_med = float(np.median(z_ent[s:e]))
        nb_med  = float(np.median(z_nb[s:e]))  # z-scored nb_ratio
        dur     = en - st

        keep_main   = (z_med > (Z_THRESH - 0.1)) and (z_p75 > 0.8 * Z_THRESH)
        keep_rescue = (ent_med > 0.4) and (nb_med > 0.1) and (dur >= 0.2)

        # long-blobby reject rule
        if dur > DUR_MAX_MAIN and not (nb_med > 0.20 and ent_med > 0.35):
            continue

        # --- NEW: temporal stability confidence
        conf_score = event_confidence(z_seg, Z_THRESH, win_frames=5)
        HIGH_CONF_THRESH = 0.8
        high_conf = (conf_score >= HIGH_CONF_THRESH)

        if (keep_main and high_conf) or keep_rescue:
            # track rescue count like before
            if (not keep_main) and keep_rescue:
                rescue_ct += 1

            # NEW: instead of just [st, en], store rich info in a dict
            accepted_rows.append({
                "start_s": float(st),
                "end_s": float(en),
                "dur_s": float(dur),
                "z_med": z_med,
                "z_p75": z_p75,
                "nb_med": float(nb_med),
                "ent_med": float(ent_med),
                "conf_score": float(conf_score),
            })

    print(f"Rescued by entropy: {rescue_ct}")
    
    # --- NEW: merge nearby events to kill duplicate splits ---
    MERGE_GAP_S = 0.15  # tune this
    accepted_rows_merged = merge_close_events(accepted_rows, gap_s=MERGE_GAP_S)



    # NOTE: we now return `accepted_rows` instead of `dets`
    return accepted_rows_merged, nb_ratio, tonality, rms, flux, td_slope_hzs, td_energy, times, z

# =======================================================================================
#                           Streaming orchestrator
# =======================================================================================
def run_streaming(audio_path: Path,
                  labels_path: Path | None,
                  out_csv: Path,
                  chunk_s=600.0,
                  overlap_s=15.0):
    info = sf.info(str(audio_path))
    fs = info.samplerate
    n_frames_total = info.frames
    dur_s = n_frames_total / fs
    print(f"Audio: {audio_path.name} | fs={fs} Hz | dur={dur_s/3600:.2f} h")

    chunk_frames   = int(round(chunk_s * fs))
    overlap_frames = int(round(overlap_s * fs))
    stride_frames  = chunk_frames - overlap_frames
    assert stride_frames > 0, "chunk_s must be larger than overlap_s"

    all_dets = []            # global dets (only start/end for merge) [[start_s,end_s], ...]
    all_feat_rows = []       # per-event feature rows (will go to events_for_gmm CSV)

    start_frame = 0
    chunk_idx = 0
    while start_frame < n_frames_total:
        end_frame = min(n_frames_total, start_frame + chunk_frames)

        # compute padded read window
        pad_left  = overlap_frames
        pad_right = overlap_frames
        read_start = max(0, start_frame - pad_left)
        read_end   = min(n_frames_total, end_frame + pad_right)

        x, _fs = sf.read(
            str(audio_path),
            start=read_start,
            frames=read_end - read_start,
            dtype="float32"
        )
        if x.ndim > 1:
            x = x.mean(axis=1)

        # detect on the PADDED chunk
        # NOTE: detect_on_chunk now returns a list of dicts, not list of [st,en]
        dets_local, nb_ratio, tonality, rms, flux, td_slope_hzs, td_energy, times_local, z_local = detect_on_chunk(
            x, fs, chunk_s
        )

        # absolute start time (s) of this padded chunk
        t0_padded = read_start / fs

        # absolute per-frame time axis for z
        t_abs = t0_padded + times_local[:len(z_local)]

        # absolute detections with metadata
        # We'll build a new list of detection dicts but with absolute times added.
        dets_abs_meta = []
        for det in dets_local:
            st_loc = det["start_s"]
            en_loc = det["end_s"]

            det_abs = det.copy()
            det_abs["start_abs_s"] = t0_padded + st_loc
            det_abs["end_abs_s"]   = t0_padded + en_loc
            dets_abs_meta.append(det_abs)

        # stash for end-of-run plots if enabled
        if PLOT_AT_END:
            plot_accum["t"].append(t_abs.astype(np.float32))
            plot_accum["z"].append(z_local.astype(np.float32))

            # for plotting, we just need start/end times, so extract those
            plot_accum["dets"].extend(
                [(d["start_abs_s"], d["end_abs_s"]) for d in dets_abs_meta]
            )

        # global z timeline (could be used later for overlay)
        t_z_global = t0_padded + times_local[:len(z_local)]
        if 't_z' not in globals():
            # capture first chunk for overlay demo
            t_z, z = t_z_global.copy(), z_local.copy()

        # ---------------- ownership window ----------------
        # only keep events whose midpoint lies in the "center" of the *un-padded* chunk
        edge_keep_s = overlap_s / 2.0
        center_left  = (start_frame / fs) + (edge_keep_s if start_frame > 0 else 0.0)
        center_right = (end_frame   / fs) - (edge_keep_s if end_frame   < n_frames_total else 0.0)

        kept_meta = []
        for det in dets_abs_meta:
            st_abs = det["start_abs_s"]
            en_abs = det["end_abs_s"]
            mid = 0.5 * (st_abs + en_abs)
            if (mid >= center_left) and (mid <= center_right):
                kept_meta.append(det)

        # Add kept detections (start/end only) to global list for merging later
        for det in kept_meta:
            all_dets.append([det["start_abs_s"], det["end_abs_s"]])

        # ---------------- per-event features for GMM / analysis ----------------
        # We now also want to store conf_score etc. So we pull from kept_meta,
        # which already has z_med, nb_med, conf_score, etc. (from detect_on_chunk).
        for det in kept_meta:
            # local chunk-relative times for slicing per-frame features
            st_local = det["start_s"]       # seconds relative to padded chunk start
            en_local = det["end_s"]
            s = int(round(st_local / HOP_S))
            e = max(s + 1, int(round(en_local / HOP_S)))

            # clip to arrays just in case
            s = max(0, min(s, len(nb_ratio)-1))
            e = max(0, min(e, len(nb_ratio)))

            row = {
                # absolute timing
                "start_s": det["start_abs_s"],
                "end_s":   det["end_abs_s"],
                "dur":     det["dur_s"],

                # stats already computed in detect_on_chunk
                "z_med":       det["z_med"],
                "z_p75":       det["z_p75"],
                "nb_med":      det["nb_med"],
                "ent_med":     det["ent_med"],
                "conf_score":  det["conf_score"],

                # extra per-event summaries computed here (legacy stuff you used to export)
                "nb_ratio_med":  float(np.median(nb_ratio[s:e]))   if e > s else 0.0,
                "nb_ratio_p75":  float(np.percentile(nb_ratio[s:e], 75)) if e > s else 0.0,
                "tonality_med":  float(np.median(tonality[s:e]))   if e > s else 0.0,
                "rms_med":       float(np.median(rms[s:e]))        if e > s else 0.0,
                "flux_med":      float(np.median(flux[s:e]))       if e > s else 0.0,
                "td_slope_med":  float(np.median(td_slope_hzs[s:e])) if e > s else 0.0,
                "td_slope_min":  float(np.min(td_slope_hzs[s:e]))    if e > s else 0.0,
                "td_energy_med": float(np.median(td_energy[s:e]))    if e > s else 0.0,

                # optional bookkeeping
                "chunk_idx":    chunk_idx,
                "chunk_start":  start_frame / fs,
            }
            all_feat_rows.append(row)

        chunk_idx += 1
        start_frame += stride_frames
        print(f"Chunk {chunk_idx:04d}: kept {len(kept_meta)} dets (global so far: {len(all_dets)})")
        _progress(chunk_idx, start_frame, n_frames_total, fs, len(all_dets))

    print(f"Pre-merge dets: {len(all_dets)}")

    # ---------------- merge overlapping / touching events ----------------
    all_dets.sort(key=lambda p: p[0])
    gaps = []
    for (a1, b1), (a2, b2) in zip(all_dets, all_dets[1:]):
        gaps.append(max(0.0, a2 - b1))
    if gaps:
        for q in (5, 10, 25, 50, 75, 90, 95):
            print(f"gap p{q}: {np.percentile(gaps, q):.3f} s")

    merged = []
    for st, en in all_dets:
        if merged:
            mst, men = merged[-1]
            overlap = st <= men
            touching = (st - men) <= 0.5  # consider within one hop as continuous
            if overlap or touching:
                merged[-1][1] = max(men, en)
                continue
        merged.append([st, en])

    # Save raw detections (merged)
    pd.DataFrame(merged, columns=["start_s", "end_s"]).to_csv(out_csv, index=False)
    print(f"\nSaved detections: {out_csv} (n={len(merged)})")

    # Save Raven TSV
    raven_tsv = out_csv.with_suffix(".raven.tsv")
    raven = pd.DataFrame({
        "Selection": np.arange(1, len(merged)+1),
        "View": "Spectrogram 1",
        "Channel": 1,
        "Begin Time (s)": [st for st, _ in merged],
        "End Time (s)":   [en for _, en in merged],
    })
    raven.to_csv(raven_tsv, sep="\t", index=False)
    print(f"Saved Raven selections: {raven_tsv}")

    # Save features for GMM (one row per kept event, BEFORE merge)
    feats_csv = default_out_path(audio_path, "events_for_gmm")
    pd.DataFrame(all_feat_rows).to_csv(feats_csv, index=False)
    print(f"Saved features for GMM: {feats_csv} (n={len(all_feat_rows)})")

    # ---------------- ground truth / evaluation ----------------
    gt = None
    truths = []
    truth_types = None
    if labels_path and Path(labels_path).exists():
        gt = read_raven_table(labels_path, prefer_channel=None, dedupe_tol=1e-3)
        truths = list(gt[['start_s', 'end_s']].itertuples(index=False, name=None))
        truth_types = gt['type'].tolist() if 'type' in gt.columns else None

    dets = [tuple(x) for x in merged]
    audio_dur_s = float(dur_s)

    if truths:
        m = compute_cpd_metrics(
            dets, truths, audio_dur_s=audio_dur_s,
            tol_s=ONSET_TOL_S, use_iou=False, iou_min=0.3
        )
        print("\n=== CPD eval (unified) ===")
        print(f"GT n={len(truths)} | Det n={len(dets)}")
        print(f"TP={m['tp']}  FP={m['fp']}  FN={m['fn']} | "
              f"Precision={m['precision']:.3f}  Recall={m['recall']:.3f}  F1={m['f1']:.3f}  "
              f"FP/h={m['fp_per_hour']:.2f}")

        # --- Per-type recall (if you have 'type' in ground truth) ---
        ptype = per_type_recall(dets, truths, truth_types, tol_s=ONSET_TOL_S) if truth_types else {}

        outdir = Path("results/cpd_eval")
        outdir.mkdir(parents=True, exist_ok=True)

        summary_row = build_summary_row("cpd_current", m, per_type=ptype)
        pd.DataFrame([summary_row]).to_csv(outdir / "summary.csv", index=False)
        print(f"[cpd_eval] Saved summary → {outdir/'summary.csv'}")

        


        if plot_accum["t"] and plot_accum["z"]:
            globals()["t_z"] = np.concatenate(plot_accum["t"])
            globals()["z"] = np.concatenate(plot_accum["z"])

        #     T = np.concatenate(plot_accum["t"])
        #     Z = np.concatenate(plot_accum["z"])

        #    # Pick last 5 minutes (300 s)
        #     t1 = float(T.max())
        #     t0 = max(float(T.min()), t1 - 300.0)

        #     # Mask z(t) to the window
        #     m = (T >= t0) & (T <= t1)
        #     T_win = T[m]
        #     Z_win = Z[m]

        #     # Trim detections to the same window
        #     def _overlaps(a, b, c, d):  # [a,b] overlaps [c,d]?
        #         return (a <= d) and (b >= c)

        #     det_intervals_win = []
        #     det_intervals = [ (float(s), float(e)) for (s, e) in merged ]

            
        #     for s, e in det_intervals:  # your CPD (start,end) list
        #         if _overlaps(s, e, t0, t1):
        #             det_intervals_win.append((max(s, t0), min(e, t1)))

        #     # Plot the shorter trace
        #     plot_z_with_detections(
        #         T_win, Z_win,
        #         det_intervals=det_intervals_win,
        #         title=f"Detections over multi-feature z",
        #         out_png="results/figs/z_with_cpd_last5min.png",
        #     )


        outdir = Path("results/cpd_eval")
        outdir.mkdir(parents=True, exist_ok=True)

        # --- Optional z overlay (first chunk) ---
        if "t_z" in globals() and "z" in globals():
            plot_z_overlay(
                t0=0.0,
                window_s=30.0,
                t_z=globals()["t_z"],
                z=globals()["z"],
                z_thresh=Z_THRESH,
                dets=dets,
                truths=truths,  # [] if no labels
                out_png=outdir / "z_overlay.png",
            )
            print(f"[cpd_eval] Saved z overlay → {outdir/'z_overlay.png'}")
        else:
            print("[cpd_eval][warn] z overlay skipped: no accumulated z timeline.")

    else:
        print("[cpd_eval] No labels provided; metrics skipped.")

# ------------------- run -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming CPD")
    add_common_args(parser)
    parser.add_argument("--chunk_s",   type=float, default=600.0)
    parser.add_argument("--overlap_s", type=float, default=15.0)
    args = parser.parse_args()

    AUDIO_PATH  = resolve_file(args.audio, DATA_DIR)
    LABELS_PATH = resolve_file(args.labels, DATA_DIR) if args.labels else None
    OUT_DETS_CSV = Path(args.out) if args.out else default_out_path(AUDIO_PATH, "output")
    OUT_DETS_CSV.parent.mkdir(parents=True, exist_ok=True)

    run_streaming(AUDIO_PATH, LABELS_PATH, OUT_DETS_CSV,
                  chunk_s=args.chunk_s, overlap_s=args.overlap_s)
