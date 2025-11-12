#!/usr/bin/env python3
"""
Feature Engineering for Noise vs Whale-Call Clustering (GMM-ready)

Given:
  - A WAV audio file (mono or stereo).
  - An events CSV with at least: start_s, end_s (in seconds).

Produces:
  - A new CSV with per-event pooled stats of LFCC/MFCC/GTCC (and their deltas)
    plus anchor-like features (tonality, crest, flux, downsweep slope),
    and extra robust cues: band energy ratio (BER), local SNR, spectral flatness (SFM).

Usage (example):
  python feature_engineering_gmm_v2.py \
      --audio path/to/audio.wav \
      --events path/to/events.csv \
      --out path/to/events_with_features.csv \
      --sr 8000 \
      --fmin 20 --fmax 300 \
      --n_mels 48 --n_lfcc 48 \
      --n_mfcc 13 --n_lfcc_ceps 13 \
      --frame_length_s 0.256 --hop_length_s 0.128 \
      --add_mfcc --add_lfcc --add_mfcc_deltas --add_lfcc_deltas --add_gtcc
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from scipy.fft import dct


# ------------------------- Helper functions -------------------------

def pooled_stats(mat, prefix):
    """Return mean/std/p10/p90 per row (coefficient) with a prefix."""
    if mat.ndim == 1:
        mat = mat[:, None]
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    stats = {}
    for i in range(mat.shape[0]):
        v = mat[i, :]
        stats[f"{prefix}{i:02d}_mean"] = float(np.mean(v)) if v.size else 0.0
        stats[f"{prefix}{i:02d}_std"]  = float(np.std(v))  if v.size else 0.0
        stats[f"{prefix}{i:02d}_p10"]  = float(np.percentile(v, 10)) if v.size else 0.0
        stats[f"{prefix}{i:02d}_p90"]  = float(np.percentile(v, 90)) if v.size else 0.0
    return stats


def linear_filterbank(sr, n_fft, n_filters=48, fmin=20.0, fmax=300.0):
    """Evenly spaced triangular filters on linear frequency between fmin and fmax."""
    freqs = np.linspace(0, sr/2, 1 + n_fft//2, endpoint=True)
    centers = np.linspace(fmin, fmax, n_filters)
    bw = (centers[1] - centers[0]) if n_filters > 1 else (fmax - fmin)
    f_lower = np.clip(centers - bw, fmin, fmax)
    f_upper = np.clip(centers + bw, fmin, fmax)

    fbanks = np.zeros((n_filters, freqs.size), dtype=np.float32)
    for i in range(n_filters):
        left, center, right = f_lower[i], centers[i], f_upper[i]
        rising = (freqs >= left) & (freqs < center) & (center > left)
        if rising.any():
            fbanks[i, rising] = (freqs[rising] - left) / (center - left)
        falling = (freqs >= center) & (freqs <= right) & (right > center)
        if falling.any():
            fbanks[i, falling] = (right - freqs[falling]) / (right - center)
    fbanks = fbanks / (fbanks.sum(axis=1, keepdims=True) + 1e-10)
    return fbanks


def compute_lfcc(y, sr, n_fft, hop_length, n_filters=48, n_ceps=13, fmin=20.0, fmax=300.0):
    S = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, center=True, window="hann"))**2
    fbanks = linear_filterbank(sr=sr, n_fft=n_fft, n_filters=n_filters, fmin=fmin, fmax=fmax)
    E = np.dot(fbanks, S)
    logE = np.log(E + 1e-10)
    C = dct(logE, type=2, axis=0, norm='ortho')[:n_ceps, :]
    return C


def compute_mfcc_lowband(y, sr, n_fft, hop_length, n_mels=48, n_mfcc=13, fmin=20.0, fmax=300.0):
    S = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, center=True, window="hann"))**2
    mel_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, norm='slaney')
    melE = np.dot(mel_fb, S)
    log_mel = np.log(melE + 1e-10)
    C = dct(log_mel, type=2, axis=0, norm='ortho')[:n_mfcc, :]
    return C


# ---- ERB helpers + GTCC ----

def hz_to_erb(f):
    return 21.4 * np.log10(4.37e-3 * f + 1.0)

def erb_to_hz(e):
    return (10**(e / 21.4) - 1.0) / 4.37e-3

def erb_filterbank(sr, n_fft, n_filters=24, fmin=20.0, fmax=300.0):
    """ERB-spaced triangular filters over [fmin, fmax]."""
    freqs = np.linspace(0, sr/2, 1 + n_fft//2, endpoint=True)
    e_min, e_max = hz_to_erb(fmin), hz_to_erb(fmax)
    centers_erb = np.linspace(e_min, e_max, n_filters)
    centers_hz = erb_to_hz(centers_erb)
    bw_erb = (e_max - e_min) / (n_filters - 1) if n_filters > 1 else (e_max - e_min)
    bw_hz = erb_to_hz(centers_erb + bw_erb/2) - erb_to_hz(centers_erb - bw_erb/2)
    f_lower = np.clip(centers_hz - bw_hz, fmin, fmax)
    f_upper = np.clip(centers_hz + bw_hz, fmin, fmax)

    fbanks = np.zeros((n_filters, freqs.size), dtype=np.float32)
    for i in range(n_filters):
        left, center, right = f_lower[i], centers_hz[i], f_upper[i]
        rising = (freqs >= left) & (freqs < center) & (center > left)
        if rising.any():
            fbanks[i, rising] = (freqs[rising] - left) / (center - left)
        falling = (freqs >= center) & (freqs <= right) & (right > center)
        if falling.any():
            fbanks[i, falling] = (right - freqs[falling]) / (right - center)
    fbanks = fbanks / (fbanks.sum(axis=1, keepdims=True) + 1e-10)
    return fbanks

def compute_gtcc(seg, sr, n_fft, hop_length, n_filters=24, n_ceps=13, fmin=20.0, fmax=300.0):
    """Gammatone-like cepstra via ERB triangular filterbank + DCT."""
    S = np.abs(librosa.stft(y=seg, n_fft=n_fft, hop_length=hop_length, center=True, window='hann'))**2
    fbanks = erb_filterbank(sr=sr, n_fft=n_fft, n_filters=n_filters, fmin=fmin, fmax=fmax)
    E = np.dot(fbanks, S)
    logE = np.log(E + 1e-10)
    C = dct(logE, type=2, axis=0, norm='ortho')[:n_ceps, :]
    return C


# ---- Anchor-like cues ----

def simple_anchor_features(seg, sr, fmin=20.0, fmax=300.0, n_fft=None, hop_length=None):
    if n_fft is None:
        n_fft = 1 << (int(round(0.256 * sr)) - 1).bit_length()
    if hop_length is None:
        hop_length = int(round(0.128 * sr))
    S = np.abs(librosa.stft(seg, n_fft=n_fft, hop_length=hop_length, window="hann"))**2
    freqs = np.linspace(0, sr/2, 1 + n_fft//2, endpoint=True)
    band = (freqs >= fmin) & (freqs <= fmax)
    Sb = S[band, :]

    # narrowband energies (dB)
    def band_db(f0, f1):
        i = (freqs >= f0) & (freqs < f1)
        return float(np.median(10.0*np.log10(S[i, :] + 1e-12))) if i.any() else np.nan

    E30_60   = band_db(30, 60)
    E60_120  = band_db(60, 120)
    E120_240 = band_db(120, 240)
    E240_360 = band_db(240, 360)
    others = [v for v in [E30_60, E120_240, E240_360] if np.isfinite(v)]
    nb_ratio_med = E60_120 - (np.median(others) if others else 0.0)
    nb_ratio_p75 = E60_120 - (np.percentile(others, 75) if others else 0.0)

    # per-frame tonality/crest
    if Sb.size:
        max_t = Sb.max(axis=0)
        mean_t = Sb.mean(axis=0) + 1e-9
        tonality_t = max_t / mean_t
        crest_t = np.percentile(Sb, 90, axis=0) / mean_t
        flux_t = np.abs(np.diff(Sb, axis=1)).mean(axis=0) if Sb.shape[1] >= 2 else np.array([0.0])
        tonality_med = float(np.median(tonality_t))
        crest_med = float(np.median(crest_t))
        flux_med = float(np.median(flux_t))
    else:
        tonality_med = crest_med = flux_med = np.nan

    # downsweep slope via peak track in 30–120 Hz
    band2 = (freqs >= 30) & (freqs <= 120)
    if band2.any() and S.shape[1] >= 3:
        Sb2 = S[band2, :]; fb = freqs[band2]
        pk = np.argmax(Sb2, axis=0); fpk = fb[pk]
        t = np.arange(len(fpk)) * (hop_length / sr)
        slope = np.nan
        if len(fpk) >= 3:
            # gate by strong frames
            e = 10*np.log10(Sb2.max(axis=0)+1e-12)
            gate = e >= (np.median(e) + (np.percentile(e, 75) - np.median(e)))
            X = t[gate] if gate.any() else t
            y = fpk[gate] if gate.any() else fpk
            if len(X) >= 3:
                a = np.polyfit(X, y, 1)
                slope = float(a[0])
        f_peak_med = float(np.median(fpk))
    else:
        slope, f_peak_med = np.nan, np.nan

    return dict(
        nb_ratio_med=nb_ratio_med,
        nb_ratio_p75=nb_ratio_p75,
        tonality_med=tonality_med,
        crest_med=crest_med,
        flux_med=flux_med,
        f_peak_med=f_peak_med,
        td_slope_med=slope,
    )


def td_stats(seg, sr, win_s=0.256, hop_s=0.128):
    # time-domain frame RMS and median energy (dB)
    frame_length = int(round(win_s * sr))
    hop_length = int(round(hop_s * sr))
    rms = librosa.feature.rms(y=seg, frame_length=frame_length, hop_length=hop_length, center=True)[0]
    rms_med = float(np.median(rms)) if rms.size else np.nan
    td_energy_med = float(np.median(10.0*np.log10(rms**2 + 1e-12))) if rms.size else np.nan
    return rms_med, td_energy_med


def slope_min_from_peak_track(S, freqs, sr, hop_length, band=(30, 120)):
    # robust “minimum-ish” downsweep from instantaneous slope
    b = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(b) or S.shape[1] < 3:
        return np.nan
    Sb = S[b, :]; fb = freqs[b]
    fpk = fb[np.argmax(Sb, axis=0)]
    dt = hop_length / float(sr)
    if len(fpk) < 3:
        return np.nan
    dfdt = np.diff(fpk) / dt
    return float(np.percentile(dfdt, 10))


def band_energy_ratio(S, freqs, low_lo=20.0, low_hi=100.0, hi_lo=100.0, hi_hi=300.0):
    """Compute log10 energy ratio between two bands using power spectrogram S (freq x time)."""
    bl = (freqs >= low_lo) & (freqs < low_hi)
    bh = (freqs >= hi_lo) & (freqs <= hi_hi)
    if not bl.any() or not bh.any():
        return np.nan
    El = np.median(S[bl, :], axis=1).mean()
    Eh = np.median(S[bh, :], axis=1).mean()
    return float(np.log10((El + 1e-12) / (Eh + 1e-12)))


def local_snr_db(y_full, sr, t0, dur, band=(20.0, 120.0), n_fft=2048, hop_length=256):
    """Estimate local SNR around event using a same-length noise window before/after."""
    win = int(round(dur * sr))
    s_idx = int(round(t0 * sr))
    e_idx = s_idx + win
    # prefer before-window; if insufficient, use after
    n0 = max(0, s_idx - win); n1 = s_idx
    if n1 - n0 < win:
        n0 = e_idx; n1 = min(len(y_full), e_idx + win)
    if n1 - n0 <= 0:
        return np.nan
    seg = y_full[s_idx:e_idx].astype(np.float32)
    noi = y_full[n0:n1].astype(np.float32)

    def band_median_power(sig):
        S = np.abs(librosa.stft(sig, n_fft=n_fft, hop_length=hop_length, window='hann'))**2
        freqs = np.linspace(0, sr/2, 1 + n_fft//2, endpoint=True)
        b = (freqs >= band[0]) & (freqs <= band[1])
        if not b.any():
            return np.nan
        return float(np.median(S[b, :]))

    Pe = band_median_power(seg); Pn = band_median_power(noi)
    if not np.isfinite(Pe) or not np.isfinite(Pn) or Pn <= 0:
        return np.nan
    return float(10.0 * np.log10((Pe + 1e-12) / (Pn + 1e-12)))


def spectral_flatness_med(S):
    """Spectral flatness (geometric mean / arithmetic mean) per frame; return median across frames."""
    eps = 1e-12
    gm = np.exp(np.mean(np.log(S + eps), axis=0))
    am = np.mean(S + eps, axis=0)
    sf = gm / (am + eps)
    return float(np.median(sf)) if sf.size else np.nan


# ------------------------- Main -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sr", type=int, default=8000)
    ap.add_argument("--frame_length_s", type=float, default=0.256)
    ap.add_argument("--hop_length_s", type=float, default=0.128)
    ap.add_argument("--fmin", type=float, default=20.0)
    ap.add_argument("--fmax", type=float, default=300.0)
    # MFCC / LFCC
    ap.add_argument("--add_mfcc", action="store_true")
    ap.add_argument("--n_mels", type=int, default=48)
    ap.add_argument("--n_mfcc", type=int, default=13)
    ap.add_argument("--add_mfcc_deltas", action="store_true")
    ap.add_argument("--add_lfcc", action="store_true")
    ap.add_argument("--n_lfcc", type=int, default=48)
    ap.add_argument("--n_lfcc_ceps", type=int, default=13)
    ap.add_argument("--add_lfcc_deltas", action="store_true")
    # GTCC / robust extras
    ap.add_argument("--add_gtcc", action="store_true")
    ap.add_argument("--n_gtcc_filters", type=int, default=24)
    ap.add_argument("--n_gtcc_ceps", type=int, default=13)
    ap.add_argument("--ber_low_hi", type=float, nargs=2, default=(20.0, 100.0))
    ap.add_argument("--ber_hi_hi", type=float, default=300.0)
    ap.add_argument("--snr_band", type=float, nargs=2, default=(20.0, 120.0))
    # pooling / padding
    ap.add_argument("--min_event_s", type=float, default=0.5)
    args = ap.parse_args()

    # Load audio
    y, sr_in = sf.read(args.audio)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr_in != args.sr:
        y = librosa.resample(y, orig_sr=sr_in, target_sr=args.sr)
    sr = args.sr

    # STFT params
    n_fft = 1 << (int(round(args.frame_length_s * sr)) - 1).bit_length()
    hop_length = int(round(args.hop_length_s * sr))

    # Load events
    df = pd.read_csv(args.events)
    out_rows = []

    # --- clean & dedupe events ---
    for col in ("start_s", "end_s"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # drop non-numeric or inverted intervals
    before = len(df)
    df = df.dropna(subset=["start_s", "end_s"])
    df = df[df["end_s"] > df["start_s"]]

    # collapse near-duplicates (e.g., same event written twice with tiny rounding drift)
    df["_start_r"] = df["start_s"].round(3)
    df["_end_r"]   = df["end_s"].round(3)
    df = df.drop_duplicates(subset=["_start_r", "_end_r"]).drop(columns=["_start_r", "_end_r"])

    print(f"[debug] cleaned input rows: {len(df)} (was {before})")

    for _, row in df.iterrows():
        t0, t1 = float(row["start_s"]), float(row["end_s"])
        if t1 <= t0:
            continue
        dur = t1 - t0
        target_len = max(args.min_event_s, dur)
        s_idx = int(round(t0 * sr))
        e_idx = int(round((t0 + target_len) * sr))
        seg = y[s_idx:e_idx].astype(np.float32)
        if seg.size < hop_length * 2:
            seg = np.pad(seg, (0, hop_length*2 - seg.size), mode="constant")

        # Prepare row dict early
        pooled = {"start_s": t0, "end_s": t1, "dur": dur}
 

        # Reusable STFT (power) and freqs for this segment
        S_full = np.abs(librosa.stft(seg, n_fft=n_fft, hop_length=hop_length, window="hann"))**2
        freqs = np.linspace(0, sr/2, 1 + n_fft//2, endpoint=True)

        # --- Domain-invariant spectral/temporal features ---
        # Use power spectrogram S_full for stability (works across gain changes).
        try:
            _cent = librosa.feature.spectral_centroid(S=S_full, sr=sr).mean()
            _bw   = librosa.feature.spectral_bandwidth(S=S_full, sr=sr).mean()
        except Exception:
            _cent, _bw = np.nan, np.nan

        # Zero-crossing rate on the raw segment (robust to amplitude scaling)
        try:
            _zcr = librosa.feature.zero_crossing_rate(seg).mean()
        except Exception:
            _zcr = np.nan

        # Raw (Hz) + normalised to Nyquist for cross-sample-rate comparability
        pooled["centroid_hz"]   = float(_cent) if np.isfinite(_cent) else np.nan
        pooled["bandwidth_hz"]  = float(_bw)   if np.isfinite(_bw)   else np.nan
        pooled["zcr_mean"]      = float(_zcr)  if np.isfinite(_zcr)  else np.nan
        nyq = sr / 2.0
        pooled["centroid_norm"]  = pooled["centroid_hz"]  / nyq if np.isfinite(pooled["centroid_hz"])  else np.nan
        pooled["bandwidth_norm"] = pooled["bandwidth_hz"] / nyq if np.isfinite(pooled["bandwidth_hz"]) else np.nan

        # Robust cues
        ber_low_lo, ber_low_hi = args.ber_low_hi
        ber_hi_lo, ber_hi_hi = ber_low_hi, args.ber_hi_hi
        pooled["ber_20_100__100_300"] = band_energy_ratio(
            S_full, freqs, low_lo=ber_low_lo, low_hi=ber_low_hi, hi_lo=ber_hi_lo, hi_hi=ber_hi_hi
        )

        band_mask = (freqs >= args.fmin) & (freqs <= args.fmax)
        pooled["sfm_med"] = spectral_flatness_med(S_full[band_mask, :]) if band_mask.any() else np.nan

        pooled["snr_local_db"] = local_snr_db(
            y_full=y, sr=sr, t0=t0, dur=target_len, band=tuple(args.snr_band),
            n_fft=n_fft, hop_length=hop_length
        )

        # Anchor-like features
        anchors = simple_anchor_features(seg, sr, fmin=args.fmin, fmax=args.fmax, n_fft=n_fft, hop_length=hop_length)
        pooled.update(anchors)

        # Time-domain stats
        rms_med, td_energy_med = td_stats(seg, sr, win_s=args.frame_length_s, hop_s=args.hop_length_s)
        pooled["rms_med"] = rms_med
        pooled["td_energy_med"] = td_energy_med

        # Extra downsweep metric
        pooled["td_slope_min"] = slope_min_from_peak_track(S_full, freqs, sr, hop_length, band=(30, 120))

        # Cepstra
        if args.add_mfcc:
            mfcc = compute_mfcc_lowband(seg, sr, n_fft, hop_length, args.n_mels, args.n_mfcc, args.fmin, args.fmax)
            pooled.update(pooled_stats(mfcc, "mfcc_"))
            if args.add_mfcc_deltas:
                mfcc_delta = librosa.feature.delta(mfcc, width=9, order=1, axis=1, mode="nearest")
                pooled.update(pooled_stats(mfcc_delta, "dmfcc_"))

        if args.add_gtcc:
            gtcc = compute_gtcc(seg, sr, n_fft, hop_length, n_filters=args.n_gtcc_filters,
                                n_ceps=args.n_gtcc_ceps, fmin=args.fmin, fmax=args.fmax)
            pooled.update(pooled_stats(gtcc, "gtcc_"))

        if args.add_lfcc:
            lfcc = compute_lfcc(seg, sr, n_fft, hop_length, args.n_lfcc, args.n_lfcc_ceps, args.fmin, args.fmax)
            pooled.update(pooled_stats(lfcc, "lfcc_"))
            if args.add_lfcc_deltas:
                lfcc_delta = librosa.feature.delta(lfcc, width=9, order=1, axis=1, mode="nearest")
                pooled.update(pooled_stats(lfcc_delta, "dlfcc_"))

        # Pass-through of any extra columns from the events file
        for col in df.columns:
            if col not in ("start_s", "end_s", "dur"):
                if col not in pooled:
                    pooled[col] = row[col]

        out_rows.append(pooled)

    out_df = pd.DataFrame(out_rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote: {args.out}  ({len(out_df)} events, {out_df.shape[1]} columns)")


if __name__ == "__main__":
    main()
