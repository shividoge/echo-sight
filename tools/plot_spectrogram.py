#!/usr/bin/env python3
"""Spectrogram + chirp spacing for a chirp-train WAV (stands in for Audacity's spectrogram view).

  .venv/bin/python tools/plot_spectrogram.py audio/train_collect_10min.wav --out docs/collect_spectrogram.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav")
    ap.add_argument("--template", default=os.path.join(ROOT, "audio", "chirp_3ms.wav"))
    ap.add_argument("--seconds", type=float, default=8.0, help="length of the overview panel")
    ap.add_argument("--out", default="spectrogram.png")
    a = ap.parse_args()

    x, fs = sf.read(a.wav, dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    tpl, tfs = sf.read(a.template, dtype="float64")
    assert tfs == fs, "template is %d Hz, recording is %d Hz" % (tfs, fs)

    # Matched filter: chirp onsets are the correlation peaks.
    corr = np.abs(signal.fftconvolve(x, tpl[::-1], mode="valid"))
    peaks, _ = signal.find_peaks(corr, height=0.5 * corr.max(), distance=int(0.04 * fs))
    steps = np.diff(peaks)
    print("%s: %.1f s at %d Hz, %d chirps found" % (os.path.basename(a.wav), len(x) / fs, fs, len(peaks)))
    if len(steps):
        vals, counts = np.unique(steps, return_counts=True)
        print("first chirp at %.6f s; spacing (samples: count) = %s" % (peaks[0] / fs, dict(zip(vals.tolist(), counts.tolist()))))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw=dict(height_ratios=[3, 2]))
    n = min(len(x), int(a.seconds * fs))
    f, t, S = signal.spectrogram(x[:n], fs, nperseg=512, noverlap=448, mode="magnitude")
    axes[0].pcolormesh(t, f / 1000, 20 * np.log10(S + 1e-9), shading="auto", cmap="magma", vmin=-100, vmax=-20)
    axes[0].set(title="%s: first %g s" % (os.path.basename(a.wav), a.seconds), xlabel="time (s)", ylabel="kHz")

    if len(peaks):
        p = peaks[0]
        seg = x[max(0, p - int(0.001 * fs)):p + int(0.005 * fs)]
        f, t, S = signal.spectrogram(seg, fs, nperseg=48, noverlap=46, mode="magnitude")
        axes[1].pcolormesh(t * 1000 - 1, f / 1000, 20 * np.log10(S + 1e-9), shading="auto", cmap="magma", vmin=-90, vmax=-20)
        axes[1].set(title="zoom on chirp 0 (at %.4f s)" % (p / fs), xlabel="ms from chirp start", ylabel="kHz")
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print("saved", a.out)


if __name__ == "__main__":
    main()
