#!/usr/bin/env python3
"""Render Echo Sight chirp-train WAV files (48 kHz, 16-bit mono).

Standard library only. Must agree sample-for-sample with conductor/index.html;
tools/test_chirp_match.py checks that.

  python3 tools/make_chirp_train.py                 # all standard files into audio/
  python3 tools/make_chirp_train.py --preset live --minutes 2
  python3 tools/make_chirp_train.py --interval 0.25 --minutes 1 --f0 20 --f1 20000
"""
import argparse
import csv
import math
import os
import struct
import wave

FS = 48000
FADE_FRAC = 0.05
CHIRP = (20.0, 20000.0, 0.003)
NEAR = (17000.0, 20000.0, 0.003)
MARKER = (200.0, 8000.0, 1.0)
MARKER_GAP_S = 0.5
LEAD_S = 1.0  # silence before the marker (lets Bluetooth speakers wake up)
DEFAULT_GAIN = 0.7  # the page's default volume

PRESETS = {
    "collect": dict(chirp=CHIRP, interval=0.5, minutes=10),
    "live": dict(chirp=CHIRP, interval=0.15, minutes=5),
    "nearsilent": dict(chirp=NEAR, interval=0.15, minutes=5),
}


def js_round(x):
    """Math.round from JavaScript (half rounds up), so sizes match the page."""
    return math.floor(x + 0.5)


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def sweep(f0, f1, dur):
    """Linear sweep with a raised-cosine fade, as float32 values (same arithmetic order as the page)."""
    n = js_round(dur * FS)
    fade = js_round(FADE_FRAC * n)
    k = (f1 - f0) / (2 * dur)
    x = []
    for i in range(n):
        t = i / FS
        v = math.sin(2 * math.pi * (f0 * t + k * t * t))
        if i < fade:
            v *= 0.5 * (1 - math.cos(math.pi * i / fade))
        elif i >= n - fade:
            v *= 0.5 * (1 - math.cos(math.pi * (n - 1 - i) / fade))
        x.append(f32(v))
    return x


def train_layout(interval_s, chirp_len, marker, lead_s, minutes):
    step = js_round(interval_s * FS)
    lead = js_round(lead_s * FS)
    marker_len = js_round(MARKER[2] * FS) if marker else 0
    gap = js_round(MARKER_GAP_S * FS) if marker else 0
    first = lead + marker_len + gap
    total = js_round((lead_s + minutes * 60) * FS)
    starts = list(range(first, total - chirp_len + 1, step))
    return dict(step=step, total=total, marker_start=lead if marker else None, first=first, chirp_starts=starts)


def to_int16(v):
    return math.floor(max(-1.0, min(1.0, v)) * 32767 + 0.5)


def pcm(samples, gain):
    """Float32 samples times a float32 gain (as a Web Audio GainNode does), quantized to 16-bit bytes."""
    g = f32(gain)
    return struct.pack("<%dh" % len(samples), *(to_int16(f32(s * g)) for s in samples))


def render_train(chirp, interval_s, minutes, gain=DEFAULT_GAIN, marker=True, lead_s=LEAD_S):
    """Returns (pcm_bytes, layout) for a whole train."""
    c = sweep(*chirp)
    lay = train_layout(interval_s, len(c), marker, lead_s, minutes)
    chirp_pcm = pcm(c, gain)
    out = bytearray(lay["total"] * 2)
    if marker:
        m = pcm(sweep(*MARKER), gain)
        out[lay["marker_start"] * 2:lay["marker_start"] * 2 + len(m)] = m
    for s in lay["chirp_starts"]:
        out[s * 2:s * 2 + len(chirp_pcm)] = chirp_pcm
    return bytes(out), lay


def write_wav(path, data):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(FS)
        w.writeframes(data)


def write_times(path, lay):
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["kind", "index", "start_sample", "start_s"])
        if lay["marker_start"] is not None:
            wr.writerow(["marker", -1, lay["marker_start"], "%.6f" % (lay["marker_start"] / FS)])
        for i, s in enumerate(lay["chirp_starts"]):
            wr.writerow(["chirp", i, s, "%.6f" % (s / FS)])


def num(x):
    return "%g" % x  # same file names as the page's downloads


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=["all", *PRESETS], default="all")
    ap.add_argument("--interval", type=float, help="seconds between chirps")
    ap.add_argument("--minutes", type=float, help="train length after the lead-in silence")
    ap.add_argument("--f0", type=float, help="chirp start frequency, Hz")
    ap.add_argument("--f1", type=float, help="chirp end frequency, Hz")
    ap.add_argument("--gain", type=float, default=DEFAULT_GAIN, help="0..1, matches the page volume (default 0.7)")
    ap.add_argument("--out", default="audio", help="output folder (default audio/)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    write_wav(os.path.join(a.out, "chirp_3ms.wav"), pcm(sweep(*CHIRP), 1.0))
    write_wav(os.path.join(a.out, "chirp_nearsilent_3ms.wav"), pcm(sweep(*NEAR), 1.0))
    write_wav(os.path.join(a.out, "marker_1s.wav"), pcm(sweep(*MARKER), 1.0))

    custom = any(v is not None for v in (a.interval, a.minutes, a.f0, a.f1))
    names = list(PRESETS) if a.preset == "all" and not custom else [a.preset if a.preset != "all" else "collect"]
    for name in names:
        p = PRESETS[name]
        f0, f1, dur = p["chirp"]
        chirp = (a.f0 if a.f0 is not None else f0, a.f1 if a.f1 is not None else f1, dur)
        interval = a.interval if a.interval is not None else p["interval"]
        minutes = a.minutes if a.minutes is not None else p["minutes"]
        stem = "train_%s_%smin" % (name, num(minutes))
        if a.interval is not None:
            stem += "_%ss" % num(interval)
        if a.f0 is not None or a.f1 is not None:
            stem += "_%s-%sHz" % (num(chirp[0]), num(chirp[1]))
        data, lay = render_train(chirp, interval, minutes, a.gain)
        write_wav(os.path.join(a.out, stem + ".wav"), data)
        write_times(os.path.join(a.out, stem + "_times.csv"), lay)
        print("%s.wav: %d chirps every %d samples (%.4f s), first at %.3f s, %.1f MB"
              % (stem, len(lay["chirp_starts"]), lay["step"], lay["step"] / FS, lay["first"] / FS, len(data) / 1e6))


if __name__ == "__main__":
    main()
