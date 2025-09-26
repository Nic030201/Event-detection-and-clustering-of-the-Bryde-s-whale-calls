# CPD_streaming.py 
# Streaming runner for your Bryde's CPD detector on very long audio.

import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from scipy.signal import butter, filtfilt, get_window, stft, find_peaks, spectrogram, resample_poly
from scipy.ndimage import median_filter
import ruptures as rpt
import time
import argparse
import matplotlib.pyplot as plt
from scipy.signal.windows import dpss
import math
from matplotlib.colors import PowerNorm, Normalize




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
LOW_HZ, HIGH_HZ = 35, 200      
FRAME_S = 0.050
HOP_S   = 0.010

# ------------------- post-proc durations -------------------
MIN_EVENT_S = 0.20
DUR_MAX_MAIN = 0.90  # soft upper cap for main (non-burst) events
MIN_GAP_S   = 0.03
ONSET_TOL_S = 0.25

# ------------------- CPD preset (recall tilt) -------------------
MODEL    = "l2"
PENALTY  = 40.0
Z_THRESH = 1.25
SMOOTH_S = 0.70

# ------------------- STFT config for spectral features -------------------
SPEC_NFFT   = 2048
SPEC_FMAX   = 280
ANCHOR_BAND = (35.0, 50.0)
UPPER_BAND  = (60.0, 240.0)
TD_BAND     = (150.0, 220.0)  # exported only

# ------------------- splitters (your current settings) -------------------
# valley split is inside split_events_by_valley()
# band-limited flux peak splitter:
PEAK_BAND            = (80.0, 220.0)
PEAK_SMOOTH_S        = 0.12
PEAK_MIN_DIST_S      = 0.40
PEAK_VALLEY_MIN_S    = 0.05
PEAK_MIN_PIECE_S     = 0.16
PEAK_MIN_PROM_FRAC   = 0.06
PEAK_HEIGHT_FRAC     = 0.15

# local CPD refine:
LOCAL_PEN_RATIO      = 0.70
LOCAL_MIN_EVENT_S    = 0.25
LOCAL_MIN_SEP_S      = 0.60

# =======================================================================================
#                                  Core helpers
# =======================================================================================
_t0 = time.time()

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
    min_event_s=0.25,
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

def pr_curve_from_scores(
    det_events,   # list of dicts with {'start','end','score'} at minimum
    truths, audio_dur_s,
    score_name='score',  # e.g. 'z_p75'
    thresholds=np.linspace(-1.0, 4.0, 51),  # adjust range to your score stats
    tol_s=0.5, use_iou=False, iou_min=0.3
):
    pr_points = []
    for th in thresholds:
        dets = [(d['start'], d['end']) for d in det_events if d.get(score_name, 0.0) >= th]
        m = compute_cpd_metrics(dets, truths, audio_dur_s, tol_s, use_iou, iou_min)
        pr_points.append((m['precision'], m['recall'], th))
    P = [p for p, r, t in pr_points]
    R = [r for p, r, t in pr_points]
    return np.array(P), np.array(R), thresholds


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
    plt.xlabel('Time (s)'); plt.ylabel('z-stat')
    plt.title(f'z overlay [{t0:.1f}–{t1:.1f}s]')
    plt.tight_layout()
    plt.legend()
    plt.savefig(out_png, dpi=160)
    plt.close()

def plot_onset_hist(onset_errs, out_png, bin_s=0.1, range_s=(-1.0, 1.0)):
    if onset_errs.size == 0:
        return
    bins = int((range_s[1]-range_s[0]) / bin_s)
    plt.figure(figsize=(5,3))
    plt.hist(onset_errs, bins=bins, range=range_s, edgecolor='k')
    mu = np.mean(onset_errs); med = np.median(onset_errs)
    plt.axvline(mu, color='tab:orange', ls='--', label=f'mean={mu*1000:.0f} ms')
    plt.axvline(med, color='tab:green', ls='--', label=f'median={med*1000:.0f} ms')
    plt.xlabel('Onset error (s)'); plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


    

def save_spectro_thumb(
    wav_path,
    center_s,
    out_png,
    *,
    mode="report",          # "inspection" (crisp) or "report" (smoother)
    fmin=20.0,
    fmax=350.0,
    crop_s=None,                # None -> 4.0 (inspection) / 5.0 (report)
    target_fs=2000,             # upper bound after band-aware downsampling
    dyn_db=70,                  # dynamic range (try 65–75)
    hi_pct=99.3,                # white point percentile (lower -> brighter mids)
    lo_pct=15,                  # lower percentile clip (use None to revert to dyn_db)
    whitening=False,            # subtract per-frequency median (stationary floor)
    fft_pad=8,                  # frequency zero-padding multiplier (1, 4, 8, 16)
    time_up=1,                  # cosmetic time upsampling factor (1 or 2)
    cmap="inferno",             # "inferno"/"magma"/"turbo"/"cividis"
    gamma=0.75,                 # gamma < 1 boosts mid-tones (0.6–0.85)
    adaptive_floor=True,        # remove slow time-varying floor
    floor_seconds=0.5,          # baseline window along time (s) if adaptive_floor
    figsize=(9.6, 3.4),
    dpi=340,
    # NEW: simple overlays (vertical lines only)
    truth_spans=None,           # list of (start_s, end_s) -> solid black lines
    det_spans=None,             # list of (start_s, end_s) -> dashed black lines
    truth_lw=2.2,
    det_lw=1.6,
):
    """
    Save a clear, band-limited spectrogram thumbnail around `center_s` (sec).
    Works well for Bryde’s whale thumbnails in 20–350 Hz.
    Draws optional vertical lines at truth/detection spans.
    """


    truth_spans = truth_spans or []
    det_spans   = det_spans or []

    # --- Load mono
    y, fs = sf.read(str(wav_path))
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)

    # --- Crop (reflect-pad at edges)
    if crop_s is None:
        crop_s = 4.0 if mode == "inspection" else 5.0
    crop_s = max(3.0, float(crop_s))

    t_total = len(y) / fs
    t0 = max(0.0, center_s - crop_s / 2)
    t1 = min(t_total, center_s + crop_s / 2)

    pad_left = max(0, int(np.ceil((crop_s / 2 - center_s) * fs)))
    pad_right = max(0, int(np.ceil((center_s + crop_s / 2 - t_total) * fs)))
    if pad_left or pad_right:
        y = np.pad(y, (pad_left, pad_right), mode="reflect")
        t0 += pad_left / fs
        t1 += pad_left / fs

    seg = y[int(round(t0 * fs)): int(round(t1 * fs))]
    seg_dur = len(seg) / fs

    # --- Band-aware downsample (enough for fmax, not wasteful)
    fs_goal = int(min(target_fs, max(1200, 6 * int(fmax))))
    if fs > fs_goal:
        g = math.gcd(int(fs), int(fs_goal))
        up, down = fs_goal // g, fs // g
        seg = resample_poly(seg, up, down)
        fs = fs_goal

    # --- Optional time upsampling (cosmetic smoothing of columns)
    if time_up > 1:
        seg = resample_poly(seg, time_up, 1)
        fs *= time_up

    # --- STFT params
    if mode == "inspection":
        win_s = 0.240     # ~4–5 Hz resolution at fs~2 kHz
        hop_s = 0.012     # ~95% overlap -> many time bins
        smooth = (1, 1)   # no extra smoothing
        interp = "nearest"
    else:  # "report"
        win_s = 0.264
        hop_s = 0.010
        smooth = (1, 3)   # light time smoothing
        interp = "bilinear"

    nper = max(256, int(round(win_s * fs)))
    nover = max(0, int(round((win_s - hop_s) * fs)))
    # FFT zero-padding (frequency interpolation)
    pad_mult = max(1, int(fft_pad))
    base = max(nper * pad_mult, 512)
    nfft = 1 << int(np.ceil(np.log2(base)))

    # --- Spectrogram (with fallback for older SciPy)
    try:
        f, t, S = spectrogram(
            seg, fs=fs, window="hann",
            nperseg=nper, noverlap=nover, nfft=nfft,
            detrend=False, scaling="density", mode="psd",
            boundary="zeros", padded=True
        )
    except TypeError:
        f, t, S = spectrogram(
            seg, fs=fs, window="hann",
            nperseg=nper, noverlap=nover, nfft=nfft,
            detrend=False, scaling="density", mode="psd"
        )
    if S.size == 0:
        plt.figure(figsize=figsize, dpi=dpi)
        plt.text(0.5, 0.5, "no data", ha="center", va="center")
        plt.axis("off")
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close()
        return

    # --- Band limit
    keep = (f >= fmin) & (f <= min(fmax, fs / 2))
    if not np.any(keep):
        keep = slice(None)
    f = f[keep]
    S = S[keep, :]

    # --- Convert to dB
    eps = 1e-12
    Sdb = 10.0 * np.log10(S + eps)

    # Optional per-frequency whitening (remove stationary floor)
    if whitening:
        Sdb = Sdb - np.median(Sdb, axis=1, keepdims=True)

    # Optional adaptive floor removal along time (drifting backgrounds)
    if adaptive_floor:
        win_cols = max(9, int(round(floor_seconds / max(hop_s, 1e-6))))
        baseline = median_filter(Sdb, size=(1, win_cols))
        Sdb = Sdb - baseline

    # Optional light smoothing for the "report" mode
    if smooth != (1, 1):
        Sdb = median_filter(Sdb, size=smooth)

    # --- Robust contrast
    v_hi = np.percentile(Sdb, hi_pct)
    if lo_pct is None:
        v_lo = v_hi - float(dyn_db)
    else:
        v_lo = np.percentile(Sdb, lo_pct)

    if not np.isfinite(v_lo) or not np.isfinite(v_hi) or v_hi <= v_lo:
        v_lo = np.nanpercentile(Sdb, 20.0)
        v_hi = np.nanpercentile(Sdb, 99.7)

    # Gamma mapping (mid-tone boost) and draw
    norm = PowerNorm(gamma=gamma, vmin=v_lo, vmax=v_hi, clip=True) if gamma else \
           Normalize(vmin=v_lo, vmax=v_hi, clip=True)

    extent = [t0, t0 + seg_dur, float(f[0]), float(f[-1])]

    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(
        Sdb, origin="lower", aspect="auto",
        extent=extent, cmap=cmap, norm=norm, interpolation=interp
    )
    plt.xlim(extent[0], extent[1])
    plt.ylim(fmin, min(fmax, fs / 2))
    plt.xlabel("Time (s)")
    plt.ylabel("Freq (Hz)")
    plt.grid(alpha=0.2, linewidth=0.5)

    # ---- overlays: simple vertical lines only (black)
    ax = plt.gca()
    ymin, ymax = fmin, min(fmax, fs / 2)

    def _vlines(spans, lw=2.0, ls="-"):
        for s, e in spans:
            ax.vlines([s, e], ymin=ymin, ymax=ymax,
                      colors="w", linewidth=lw, linestyles=ls, zorder=5)

    if truth_spans:
        _vlines(truth_spans, lw=truth_lw, ls="--")   # solid black
    if det_spans:
        _vlines(det_spans,   lw=det_lw,   ls="--")  # dashed black

    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close()



def make_error_gallery(
    wav_path, dets, truths, matches, fp_idx, fn_idx, out_dir, max_each=10
):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    # FPs: dets with no match
    for k, j in enumerate(fp_idx[:max_each]):
        ds, de = dets[j]
        dur = de - ds   # or te - ts for FN
        win_s = max(1.0, min(2.0, dur + 0.5))
        save_spectro_thumb(wav_path, (ds+de)/2, out_dir/f'FP_{k:02d}_t{ds:.2f}-{de:.2f}.png', crop_s=win_s, mode="inspection",det_spans=[(ds, de)]
)
    # FNs: truths with no match
    for k, i in enumerate(fn_idx[:max_each]):
        ts, te = truths[i]
        win_s = max(1.0, min(2.0, (te-ts) + 0.5))
        save_spectro_thumb(wav_path, (ts+te)/2, out_dir/f'FN_{k:02d}_t{ts:.2f}-{te:.2f}.png', crop_s=win_s, mode="inspection",truth_spans=[(ts, te)] 
)

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

    # entropy-based tonality in mid band (ST/MT focus)
    ENT_BAND = (70.0, 160.0)
    ent_mid  = band_entropy_tonality(P, f_axis, ENT_BAND)
    z_ent    = rz_ent(median_filter(ent_mid, size=k))

    nb_ratio = per_frame_narrowband_ratio(P, f_axis)
    tonality = per_frame_anchor_tonality(P, f_axis)
    td_slope_hzs = downsweep_slope_hz_per_s(P, f_axis, HOP_S, band=TD_BAND)  # exported only
    td_energy    = P[band_mask(f_axis, *TD_BAND), :].sum(axis=0)             # exported only

    # rolling z-scores
    z_rms  = rz_rms( median_filter(rms,      size=k) )
    z_flux = rz_flux(median_filter(flux,     size=k) )
    z_nb   = rz_nb(  median_filter(nb_ratio, size=k) )
    z_tone = rz_tone(median_filter(tonality, size=k))

    # weights tuned for recall on tonal ST/MT
    W_RMS, W_FLUX, W_NB, W_TONE = 0.10, 0.05, 0.50, 0.35
    z = (W_RMS*z_rms + W_FLUX*z_flux + W_NB*z_nb + W_TONE*z_tone).astype(np.float32)

    # CPD → candidates
    min_size_frames = max(3, int(round(MIN_EVENT_S / HOP_S)))
    n_frames_here = len(z)

    # scale penalty to the configured chunk length (single source of truth)
    BASE_N = int(round(max(chunk_s, 1e-6) / HOP_S))
    penalty_here = max(1.0, PENALTY * (np.log(max(n_frames_here,2)) / np.log(max(BASE_N,2))))

    segs = segments_from_cpd(z, HOP_S, penalty_here, model=MODEL, min_size_frames=min_size_frames)
    dets = pick_event_segments(z, segs, Z_THRESH, MIN_EVENT_S, MIN_GAP_S, HOP_S)

    # Valley split
    z_split = max(0.5*Z_THRESH, Z_THRESH - 0.7)
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

    # entropy/nb rescue
    rescue_ct = 0
    accepted = []
    for st, en in dets:
        s = int(round(st / HOP_S)); e = max(s+1, int(round(en / HOP_S)))
        if e <= s: 
            continue
        z_med   = float(np.median(z[s:e]))
        z_p75   = float(np.percentile(z[s:e], 75))
        ent_med = float(np.median(z_ent[s:e]))
        nb_med  = float(np.median(z_nb[s:e]))  # z-scored nb_ratio
        keep_main   = (z_med > (Z_THRESH - 0.1)) and (z_p75 > 0.8*Z_THRESH)
        keep_rescue = (ent_med > 0.38) and (nb_med > 0.08) and ((en - st) >= 0.16)
        dur = en - st
        # If it's longer than our main cap and not very narrowband/tonal, drop it.
        if dur > DUR_MAX_MAIN and not (nb_med > 0.20 and ent_med > 0.35):
            continue

        if keep_main or keep_rescue:
            accepted.append([st, en])
            if (not keep_main) and keep_rescue:
                rescue_ct += 1

    dets = accepted
    print(f"Rescued by entropy: {rescue_ct}")

    # return only what's needed by the orchestrator
    return dets, nb_ratio, tonality, rms, flux, td_slope_hzs, td_energy, times, z

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

    # chk_csv = out_csv.with_suffix(".partial.csv")
    # if chk_csv.exists():
    #     chk_csv.unlink()
    # wrote_header = False

    all_dets = []            # global dets [start_s, end_s]
    all_feat_rows = []       # feature rows for GMM export

    start_frame = 0
    chunk_idx = 0
    while start_frame < n_frames_total:
        end_frame = min(n_frames_total, start_frame + chunk_frames)

        # compute padded read window
        pad_left  = overlap_frames
        pad_right = overlap_frames
        read_start = max(0, start_frame - pad_left)
        read_end   = min(n_frames_total, end_frame + pad_right)

        x, _fs = sf.read(str(audio_path),
                         start=read_start,
                         frames=read_end - read_start,
                         dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)

        # detect on the PADDED chunk
        dets_local, nb_ratio, tonality, rms, flux, td_slope_hzs, td_energy, times_local, z_local = detect_on_chunk(x, fs, chunk_s)

        # map local → global using padded t0
        t0_padded = read_start / fs
        dets_global = [[t0_padded + st, t0_padded + en] for (st, en) in dets_local]

        t_z_global = t0_padded + times_local[:len(z_local)]

        # Example: stash one 30 s window around 60 s for the overlay figure
        if 't_z' not in globals():
            t_z, z = t_z_global.copy(), z_local.copy()

        # keep only the CENTER ownership region of this chunk
        edge_keep_s = overlap_s / 2.0
        center_left  = (start_frame / fs) + (edge_keep_s if start_frame > 0 else 0.0)
        center_right = (end_frame   / fs) - (edge_keep_s if end_frame   < n_frames_total else 0.0)

        kept = []
        for st, en in dets_global:
            mid = 0.5*(st+en)
            if (mid >= center_left) and (mid <= center_right):
                kept.append([st, en])
        all_dets.extend(kept)

        df_kept = pd.DataFrame(kept, columns=["start_s","end_s"])
        # df_kept.to_csv(chk_csv, mode="a", header=not wrote_header, index=False)
        wrote_header = True

        # --- per-event features (computed in chunk time then shifted) ---
        for st_g, en_g in kept:
            st = st_g - t0_padded; en = en_g - t0_padded  # back to chunk sec
            s = int(round(st / HOP_S)); e = max(s+1, int(round(en / HOP_S)))
            row = {
                "start_s": st_g, "end_s": en_g, "dur": en_g - st_g,
                "nb_ratio_med":  float(np.median(nb_ratio[s:e])) if e > s else 0.0,
                "nb_ratio_p75":  float(np.percentile(nb_ratio[s:e], 75)) if e > s else 0.0,
                "tonality_med":  float(np.median(tonality[s:e])) if e > s else 0.0,
                "rms_med":       float(np.median(rms[s:e])) if e > s else 0.0,
                "flux_med":      float(np.median(flux[s:e])) if e > s else 0.0,
                "td_slope_med":  float(np.median(td_slope_hzs[s:e])) if e > s else 0.0,
                "td_slope_min":  float(np.min(td_slope_hzs[s:e])) if e > s else 0.0,
                "td_energy_med": float(np.median(td_energy[s:e])) if e > s else 0.0,
            }
            all_feat_rows.append(row)

        chunk_idx += 1
        start_frame += (chunk_frames - overlap_frames)
        print(f"Chunk {chunk_idx:04d}: kept {len(kept)} dets (global so far: {len(all_dets)})")
        _progress(chunk_idx, start_frame, n_frames_total, fs, len(all_dets))

    print(f"Pre-merge dets: {len(all_dets)}")

    # --- Gap histogram diagnostic ---
    gaps = []
    all_dets.sort(key=lambda p: p[0])
    for (a1, b1), (a2, b2) in zip(all_dets, all_dets[1:]):
        gaps.append(max(0.0, a2 - b1))
    if gaps:
        for q in (5, 10, 25, 50, 75, 90, 95):
            print(f"gap p{q}: {np.percentile(gaps, q):.3f} s")

    # --- merge: only overlaps or ≤ one hop apart ---
    all_dets.sort(key=lambda p: p[0])
    merged = []
    for st, en in all_dets:
        if merged:
            mst, men = merged[-1]
            overlap = st <= men
            touching = (st - men) <= HOP_S  # <= 1 frame hop (10 ms)
            if overlap or touching:
                merged[-1][1] = max(men, en)
                continue
        merged.append([st, en])

    # Save detections
    pd.DataFrame(merged, columns=["start_s","end_s"]).to_csv(out_csv, index=False)
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

    # Save features for GMM (only for kept events; merged boundaries may shift slightly)
    feats_csv = default_out_path(audio_path, "events_for_gmm")
    pd.DataFrame(all_feat_rows).to_csv(feats_csv, index=False)
    print(f"Saved features for GMM: {feats_csv} (n={len(all_feat_rows)})")

    # --- Build ground truth once (if labels exist) ---
    gt = None
    truths = []
    truth_types = None
    if labels_path and Path(labels_path).exists():
        gt = read_raven_table(labels_path, prefer_channel=None, dedupe_tol=1e-3)
        truths = list(gt[['start_s','end_s']].itertuples(index=False, name=None))
        truth_types = gt['type'].tolist() if 'type' in gt.columns else None

    # --- Unified console eval (uses SAME logic as summary) ---
    if truths:
        dets = [tuple(x) for x in merged]
        m = compute_cpd_metrics(
            dets, truths, audio_dur_s=float(dur_s),
            tol_s=ONSET_TOL_S, use_iou=False, iou_min=0.3
        )
        print("\n=== CPD eval (unified) ===")
        print(f"GT n={len(truths)} | Det n={len(dets)}")
        print(f"TP={m['tp']}  FP={m['fp']}  FN={m['fn']} | "
              f"Precision={m['precision']:.3f}  Recall={m['recall']:.3f}  F1={m['f1']:.3f}  "
              f"FP/h={m['fp_per_hour']:.2f}")
    else:
        print("[cpd_eval] No labels provided; metrics skipped.")

        
    # ------------------------ CPD EVALUATION CALLS ------------------------
    # Build detection & truth lists from what's computed above.
    dets = [tuple(x) for x in merged]           # list[(start_s, end_s)]
    audio_dur_s = float(dur_s)

    # If no truths, skip metrics (nothing to compare against)
    if truths:
        # --- Core metrics (FP/h, TPR/Recall, Onset/Offset error, Duration bias) ---
        tol_s = ONSET_TOL_S  # use your configured onset tolerance
        metrics = compute_cpd_metrics(dets, truths, audio_dur_s,
                                      tol_s=tol_s, use_iou=False, iou_min=0.3)

        # --- Per-type recall (optional) ---
        ptype = per_type_recall(dets, truths, truth_types, tol_s=tol_s) if truth_types else {}

        # --- Summary table ---
        outdir = Path("results/cpd_eval"); outdir.mkdir(parents=True, exist_ok=True)
        summary_row = build_summary_row("cpd_current", metrics, per_type=ptype)
        pd.DataFrame([summary_row]).to_csv(outdir / "summary.csv", index=False)
        print(f"[cpd_eval] Saved summary → {outdir/'summary.csv'}")

        # --- PR curve (score sweep) ---
        # Use an event-level "score". You already collect per-event features in all_feat_rows.
        # Pick a monotonic-with-confidence proxy like nb_ratio_p75 (or tonality_med).
        det_event_dicts = []
        for r in all_feat_rows:
            sc = r.get('nb_ratio_p75', None)
            if sc is None:
                sc = r.get('tonality_med', 0.0)
            det_event_dicts.append({'start': r['start_s'], 'end': r['end_s'], 'score': float(sc)})

        try:
            scores = np.array([d['score'] for d in det_event_dicts], float)
            if scores.size >= 5 and np.isfinite(scores).all():
                lo, hi = np.percentile(scores, [1, 99])
                if not np.isfinite(lo): lo = np.nanmin(scores)
                if not np.isfinite(hi): hi = np.nanmax(scores)
                if hi <= lo: hi = lo + 1e-3
                P, R, TH = pr_curve_from_scores(
                    det_event_dicts, truths, audio_dur_s,
                    score_name='score',
                    thresholds=np.linspace(lo, hi, 41),
                    tol_s=tol_s, use_iou=False, iou_min=0.3
                )
                plt.figure(figsize=(4,4))
                plt.plot(R, P, marker='o', lw=1)
                plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('CPD PR (score sweep)')
                plt.grid(True, ls='--', alpha=0.4)
                plt.tight_layout(); plt.savefig(outdir / "cpd_pr_curve.png", dpi=160); plt.close()
                print(f"[cpd_eval] Saved PR curve → {outdir/'cpd_pr_curve.png'}")
            else:
                print("[cpd_eval][warn] PR curve skipped: insufficient or invalid scores.")
        except Exception as e:
            print("[cpd_eval][warn] PR curve failed:", e)

        # --- Onset error histogram ---
        plot_onset_hist(metrics['onset_errs'], outdir/"onset_error_hist.png",
                        bin_s=0.05, range_s=(-1.0, 1.0))
        print(f"[cpd_eval] Saved onset error histogram → {outdir/'onset_error_hist.png'}")

        # --- Error gallery (FP/FN spectrograms) ---
        make_error_gallery(audio_path, dets, truths, metrics['matches'],
                           metrics['fp_idx'], metrics['fn_idx'],
                           out_dir=outdir/"error_gallery", max_each=12)
        print(f"[cpd_eval] Saved error gallery → {outdir/'error_gallery'}")

        # --- z-signal overlay (optional) ---
        # NOTE: You currently don't expose t_z and z. If you want this plot,
        # return (times, z) from detect_on_chunk and stitch to global time:
        #   1) In detect_on_chunk(...): return `times` and `z` too.
        #   2) In the loop, after t0_padded is known, do:
        #        t_z_global = t0_padded + times[:len(z)]
        #        collect a slice that covers t0..t0+window.
        # For now, we skip unless you wire those arrays.
        if 't_z' in globals() and 'z' in globals():
            plot_z_overlay(t0=60.0, window_s=30.0, t_z=t_z, z=z, z_thresh=Z_THRESH,
                           dets=dets, truths=truths, out_png=outdir/"z_overlay_60s.png")
            print(f"[cpd_eval] Saved z overlay → {outdir/'z_overlay_60s.png'}")
        else:
            print("[cpd_eval][info] z overlay skipped (t_z/z not provided).")
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
