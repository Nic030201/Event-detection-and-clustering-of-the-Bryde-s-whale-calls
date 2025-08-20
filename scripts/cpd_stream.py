# CPD_streaming.py (clean)
# Streaming (chunked) runner for your Bryde's CPD detector on very long audio (e.g., 72h).
# Same features/CPD/splitters as your latest script, but tidied and de-duped.

import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from scipy.signal import butter, filtfilt, get_window, stft, find_peaks
from scipy.ndimage import median_filter
import ruptures as rpt
import time
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

# ------------------- front-end filtering & framing -------------------
LOW_HZ, HIGH_HZ = 35, 300           # passband
FRAME_S = 0.050
HOP_S   = 0.010

# ------------------- post-proc durations -------------------
MIN_EVENT_S = 0.20
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
    W_RMS, W_FLUX, W_NB, W_TONE = 0.15, 0.15, 0.45, 0.25
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
        if keep_main or keep_rescue:
            accepted.append([st, en])
            if (not keep_main) and keep_rescue:
                rescue_ct += 1

    dets = accepted
    print(f"Rescued by entropy: {rescue_ct}")

    # return only what's needed by the orchestrator
    return dets, nb_ratio, tonality, rms, flux, td_slope_hzs, td_energy

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

    chk_csv = out_csv.with_suffix(".partial.csv")
    if chk_csv.exists():
        chk_csv.unlink()
    wrote_header = False

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
        dets_local, nb_ratio, tonality, rms, flux, td_slope_hzs, td_energy = detect_on_chunk(x, fs, chunk_s)

        # map local → global using padded t0
        t0_padded = read_start / fs
        dets_global = [[t0_padded + st, t0_padded + en] for (st, en) in dets_local]

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
        df_kept.to_csv(chk_csv, mode="a", header=not wrote_header, index=False)
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

    # Optional: coarse eval if labels exist (uses begins, tolerant; no z re-compute)
    if labels_path and Path(labels_path).exists():
        gt = read_raven_table(labels_path, prefer_channel=None, dedupe_tol=1e-3)
        det_on = np.array([st for st, _ in merged], dtype=np.float32)
        gt_on  = gt["start_s"].values.astype(np.float32)
        used_det = np.zeros(len(det_on), bool); used_gt = np.zeros(len(gt_on), bool)
        tp = 0
        for i, t in enumerate(gt_on):
            diffs = np.abs(det_on - t); diffs[used_det] = np.inf
            j = int(np.argmin(diffs))
            if diffs[j] <= ONSET_TOL_S:
                used_det[j] = True; used_gt[i] = True; tp += 1
        fp = int((~used_det).sum()); fn = int((~used_gt).sum())
        p = tp/(tp+fp) if (tp+fp) else 0.0
        r = tp/(tp+fn) if (tp+fn) else 0.0
        f1 = 2*p*r/(p+r) if (p+r) else 0.0
        print(f"\n=== Streaming eval (approx) ===")
        print(f"GT n={len(gt)} | Det n={len(merged)}")
        print(f"TP={tp}  FP={fp}  FN={fn} | Precision={p:.3f}  Recall={r:.3f}  F1={f1:.3f}")

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
