#!/usr/bin/env python3
"""Did the echoes survive the phone's audio processing?

Finds every chirp in a recording, numbers them from the session marker, and plots
the 72.5 ms after each direct chirp on a distance axis, so a wall shows up as a bump.

  .venv/bin/python tools/check_recording.py audio/recordings/test_1m_laptop.wav
  .venv/bin/python tools/check_recording.py ~/Downloads/memo.m4a --channel 1 --min-dist 0.6 --out docs/check.png

Non-WAV files (Voice Memos .m4a) are converted with afconvert (macOS) or ffmpeg.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import ndimage, signal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FS = 48000
C = 343.0
WINDOW_S = 0.0725
MARKER_TO_FIRST_S = 1.5  # 1.0 s marker + 0.5 s gap


def load(path):
    """Mono-or-multichannel float64 at 48 kHz, converting/resampling as needed."""
    try:
        x, fs = sf.read(path, dtype="float64", always_2d=True)
    except sf.LibsndfileError:
        tmp = os.path.join(tempfile.mkdtemp(), "converted.wav")
        if shutil.which("afconvert"):
            cmd = ["afconvert", "-f", "WAVE", "-d", "LEI24@%d" % FS, path, tmp]
        elif shutil.which("ffmpeg"):
            cmd = ["ffmpeg", "-loglevel", "error", "-i", path, "-ar", str(FS), "-c:a", "pcm_s24le", tmp]
        else:
            sys.exit("can't read %s: install ffmpeg or convert it to WAV first" % path)
        subprocess.run(cmd, check=True)
        x, fs = sf.read(tmp, dtype="float64", always_2d=True)
    if fs != FS:
        g = np.gcd(int(fs), FS)
        x = signal.resample_poly(x, FS // g, int(fs) // g, axis=0)
    return x


def bandpass(x, band):
    if not band or band[0] <= 0:
        return x
    sos = signal.butter(6, band, "bandpass", fs=FS, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


def matched(x, tpl):
    """Raw matched-filter output; index i = template starting at sample i."""
    return signal.fftconvolve(x, tpl[::-1], mode="valid")


def find_marker(x, marker):
    c = np.abs(matched(x, marker))
    i = int(c.argmax())
    strength = c[i] / (np.median(c) + 1e-12)
    return (i, strength) if strength > 20 else (None, strength)


def find_chirps(env, min_gap, rel_height):
    peaks, _ = signal.find_peaks(env, height=rel_height * env.max(), distance=min_gap)
    return peaks


def on_grid(peaks, index, env, tol_s=0.002):
    """Drop peaks that aren't on the chirp grid (bumps, voices, a second peak in the same slot).

    Keeps the strongest peak per index, then repeatedly fits position vs index and drops
    anything more than tol_s off the line. Returns (peaks, index, number rejected).
    """
    n0 = len(peaks)
    best = {}
    for p, k in zip(peaks, index):
        if k >= 0 and (k not in best or env[p] > env[best[k]]):
            best[k] = p
    index = np.array(sorted(best), dtype=int)
    peaks = np.array([best[k] for k in index], dtype=int)
    for _ in range(5):
        if len(peaks) < 3:
            break
        res = peaks - np.polyval(np.polyfit(index, peaks, 1), index)
        ok = np.abs(res) <= max(tol_s * FS, 3 * np.median(np.abs(res)))
        if ok.all():
            break
        peaks, index = peaks[ok], index[ok]
    return peaks, index, n0 - len(peaks)


def first_echo(profile, dist, min_dist, factor=4.0, local_m=0.5, lobe_m=0.25):
    """First echo past min_dist that stands factor x above the local noise (median over local_m).

    Local rather than global, so the slowly fading tail of the direct chirp never counts as an echo.
    A chirp's matched-filter peak has side ripples up to half the chirp length (3 ms = 0.26 m)
    either side, so the first crossing is often a ripple: take the highest point in the next lobe_m.
    """
    local = ndimage.median_filter(profile, size=max(3, int(local_m / (dist[1] - dist[0]))), mode="nearest")
    start = np.searchsorted(dist, min_dist)
    peaks = start + signal.find_peaks(profile[start:])[0]
    peaks = peaks[profile[peaks] > factor * local[peaks]]
    if not len(peaks):
        return None
    i = peaks[0]
    j = np.searchsorted(dist, dist[i] + lobe_m)
    return dist[i + int(np.argmax(profile[i:j]))]


def analyse(x, args):
    tpl = sf.read(args.template, dtype="float64")[0]
    marker = sf.read(args.marker, dtype="float64")[0]
    band = None if args.band[0] <= 0 else args.band
    xb, tplb = bandpass(x, band), bandpass(tpl, band)

    m_idx, m_strength = find_marker(x, marker)  # marker sweep is 200 Hz–8 kHz, so search the unfiltered signal
    corr = matched(xb, tplb)
    env = np.abs(signal.hilbert(corr))
    min_gap = int(args.min_gap * FS) if args.min_gap else int(0.8 * args.interval * FS)
    peaks = find_chirps(env, min_gap, args.threshold)
    if m_idx is not None:
        peaks = peaks[peaks > m_idx + int(0.5 * MARKER_TO_FIRST_S * FS)]
        index = np.round((peaks - m_idx - MARKER_TO_FIRST_S * FS) / (args.interval * FS)).astype(int)
    else:
        index = np.round((peaks - peaks[0]) / (args.interval * FS)).astype(int) if len(peaks) else np.array([], int)

    peaks, index, rejected = on_grid(peaks, index, env)
    W = int(WINDOW_S * FS)
    keep = peaks + W < len(corr)
    peaks, index = peaks[keep], index[keep]
    segs = np.array([corr[p:p + W] for p in peaks])
    segs /= np.abs(segs[:, :1]) + 1e-12
    envs = np.abs(signal.hilbert(segs, axis=1))
    coherent = np.abs(signal.hilbert(segs.mean(axis=0)))
    coherent /= coherent[0]
    dist = np.arange(W) / FS * C / 2

    out = dict(peaks=peaks, index=index, rejected=rejected, marker=m_idx, marker_strength=m_strength, envs=envs, coherent=coherent, dist=dist)
    if len(peaks) > 2:
        slope, icpt = np.polyfit(index, peaks, 1)
        out["spacing"] = slope
        out["drift_ppm"] = (slope / (args.interval * FS) - 1) * 1e6
        out["jitter"] = np.std(peaks - (slope * index + icpt))
        out["missing"] = int(index.max() - index.min() + 1 - len(np.unique(index)))
    firsts = [first_echo(e, dist, args.min_dist) for e in envs]
    firsts = [f for f in firsts if f is not None]
    out["first_echo_median"] = float(np.median(firsts)) if firsts else None
    out["first_echo_count"] = len(firsts)
    out["coherent_first_echo"] = first_echo(coherent, dist, args.min_dist)
    return out


def report(name, r, args):
    n = len(r["peaks"])
    print("%s" % name)
    if r["marker"] is not None:
        print("  marker at %.4f s (%.0fx median); chirps numbered from it" % (r["marker"] / FS, r["marker_strength"]))
    else:
        print("  marker NOT found (%.0fx median); chirps numbered from the first one" % r["marker_strength"])
    print("  chirps found: %d (%d off-grid peaks rejected)" % (n, r["rejected"]))
    if "spacing" in r:
        print("  mean spacing: %.3f samples = %.6f s (expected %.0f); drift %+.1f ppm; jitter %.1f samples; missing in range: %d"
              % (r["spacing"], r["spacing"] / FS, args.interval * FS, r["drift_ppm"], r["jitter"], r["missing"]))
    fe = r["first_echo_median"]
    print("  first echo past %.2f m: median %s over %d/%d chirps; coherent average %s"
          % (args.min_dist, "%.2f m" % fe if fe is not None else "none", r["first_echo_count"], n,
             "%.2f m" % r["coherent_first_echo"] if r["coherent_first_echo"] is not None else "none"))


def plot(results, args, title):
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 7), squeeze=False, layout="constrained")
    rng = np.random.default_rng(0)
    for ax, (name, r) in zip(axes[0], results):
        d, envs = r["dist"], r["envs"]
        pick = np.sort(rng.choice(len(envs), size=min(20, len(envs)), replace=False))
        for row, i in enumerate(pick):
            ax.plot(d, row + envs[i] / max(envs[i][d > args.min_dist].max(), 1e-12) * 0.9, lw=0.7, color="C0")
        top = len(pick)
        ax.plot(d, top + 0.5 + r["coherent"] / max(r["coherent"][d > args.min_dist].max(), 1e-12) * 2.0, lw=1.4, color="C3")
        ax.text(args.max_dist, top + 2.6, "coherent average of %d" % len(envs), ha="right", color="C3", fontsize=9)
        ax.axvspan(0, args.min_dist, color="0.85")
        fe = r["first_echo_median"]
        if fe is not None:
            ax.axvline(fe, color="C2", ls="--", lw=1, label="median first echo %.2f m" % fe)
            ax.legend(loc="lower right", fontsize=9)
        ax.set(xlim=(0, args.max_dist), ylim=(-0.3, top + 3), yticks=range(top), yticklabels=["#%d" % r["index"][i] for i in pick],
               xlabel="distance (m) = samples / 48000 × 343 / 2", title=name)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle(title + ("  [band %g–%g Hz]" % tuple(args.band) if args.band[0] > 0 else "  [full band]"))
    fig.savefig(args.out, dpi=110)
    print("saved", args.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording")
    ap.add_argument("--template", default=os.path.join(ROOT, "audio", "chirp_3ms.wav"))
    ap.add_argument("--marker", default=os.path.join(ROOT, "audio", "marker_1s.wav"))
    ap.add_argument("--interval", type=float, default=0.5, help="chirp interval in seconds (0.5 collect, 0.15 live)")
    ap.add_argument("--min-gap", type=float, help="minimum seconds between chirp peaks (default 0.8 x interval)")
    ap.add_argument("--threshold", type=float, default=0.3, help="peak height as a fraction of the strongest chirp")
    ap.add_argument("--band", type=float, nargs=2, default=[2000, 15000], metavar=("LO", "HI"),
                    help="band-pass before matching (default 2000 15000; '0 0' = full band). Removes mains hum.")
    ap.add_argument("--channel", default="mix", help="'mix', 'all', or a channel number")
    ap.add_argument("--min-dist", type=float, default=0.2, help="blind zone in metres before looking for the first echo")
    ap.add_argument("--max-dist", type=float, default=4.0, help="x-axis limit of the plot, metres")
    ap.add_argument("--out", help="PNG path (default: next to the recording)")
    args = ap.parse_args()
    if not args.out:
        args.out = os.path.splitext(args.recording)[0] + "_check.png"

    x = load(args.recording)
    if args.channel == "mix":
        chans = [("mix of %d ch" % x.shape[1], x.mean(axis=1))]
    elif args.channel == "all":
        chans = [("channel %d" % i, x[:, i]) for i in range(x.shape[1])]
    else:
        chans = [("channel %s" % args.channel, x[:, int(args.channel)])]
    print("%s: %.1f s, %d channel(s), peak level %.3f%s" % (os.path.basename(args.recording), len(x) / FS, x.shape[1],
          np.abs(x).max(), "  ** CLIPPING **" if np.abs(x).max() > 0.99 else ""))

    results = []
    for name, sig in chans:
        r = analyse(sig, args)
        report(name, r, args)
        results.append((name, r))
    plot(results, args, os.path.basename(args.recording))


if __name__ == "__main__":
    main()
