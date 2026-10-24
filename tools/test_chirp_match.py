#!/usr/bin/env python3
"""Check that conductor/index.html and make_chirp_train.py make identical audio.

  python3 tools/test_chirp_match.py                      # runs the page's DSP code in Node and diffs it
  python3 tools/test_chirp_match.py --page-json page_chirps.json
  python3 tools/test_chirp_match.py --page-wav train_collect_10min.wav --mode collect --minutes 10

--page-json takes the page's "Export chirp JSON" download. --page-wav takes a page
"Download WAV" file; pass the same mode/minutes/interval/volume that were on the page.
Every WAV checked also gets its chirp spacing measured to the sample.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_chirp_train as mct  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(ROOT, "conductor", "index.html")
COPIES = [os.path.join(ROOT, "recorder", "index.html")]  # pages carrying a copy of the conductor's DSP block
MODES = {"collect": ("collect", mct.CHIRP), "live": ("live", mct.CHIRP), "near": ("nearsilent", mct.NEAR)}

failures = []


def check(ok, msg):
    print(("PASS  " if ok else "FAIL  ") + msg)
    if not ok:
        failures.append(msg)


def compare(name, page, py):
    if len(page) != len(py):
        check(False, "%s: page has %d samples, script has %d" % (name, len(page), len(py)))
        return
    diffs = [i for i, (a, b) in enumerate(zip(page, py)) if a != b]
    worst = max((abs(page[i] - py[i]) for i in diffs), default=0.0)
    check(not diffs, "%s: %d samples, %d differ (max diff %.3g)" % (name, len(py), len(diffs), worst))


def dsp_block(path):
    m = re.search(r"// BEGIN DSP.*?\n(.*?)// END DSP", open(path, encoding="utf-8").read(), re.S)
    return m.group(1) if m else None


def page_dsp_via_node():
    block = dsp_block(PAGE)
    if block is None:
        check(False, "could not find the BEGIN DSP / END DSP block in conductor/index.html")
        return None
    js = block + """
const layouts = [[0.5, 144, true, 1.0, 10], [0.15, 144, true, 1.0, 5], [0.25, 144, true, 1.0, 0.1], [0.123, 144, false, 0.0, 0.05]]
  .map(([intervalS, chirpLen, marker, leadS, minutes]) => trainLayout({ intervalS, chirpLen, marker, leadS, minutes }));
const q = [-1.5, -1, -0.5, -1e-5, 0, 1e-5, 0.49999, 0.5, 1, 1.5].map(toInt16);
process.stdout.write(JSON.stringify({
  chirp: Array.from(sweep(CHIRP.f0, CHIRP.f1, CHIRP.dur)),
  near: Array.from(sweep(NEAR.f0, NEAR.f1, NEAR.dur)),
  marker: Array.from(sweep(MARKER.f0, MARKER.f1, MARKER.dur)),
  layouts, q,
}));
"""
    out = subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def chirp_onsets(samples, min_gap):
    """First non-zero sample after at least min_gap zeros. The trains are digital silence between chirps."""
    onsets, zeros = [], min_gap
    for i, v in enumerate(samples):
        if v == 0:
            zeros += 1
        else:
            if zeros >= min_gap:
                onsets.append(i)
            zeros = 0
    return onsets


def read_wav(path):
    with wave.open(path, "rb") as w:
        check(w.getframerate() == mct.FS and w.getsampwidth() == 2 and w.getnchannels() == 1,
              "%s is 48 kHz 16-bit mono (got %d Hz, %d-byte, %d ch)" % (os.path.basename(path), w.getframerate(), w.getsampwidth(), w.getnchannels()))
        return w.readframes(w.getnframes())


def check_spacing(name, data, expected_step):
    import array
    a = array.array("h")
    a.frombytes(data)
    onsets = chirp_onsets(a, 1000)[1:]  # drop the marker
    steps = sorted({b - a_ for a_, b in zip(onsets, onsets[1:])})
    check(steps == [expected_step], "%s: %d chirps, spacing in samples = %s (expected %d)" % (name, len(onsets), steps[:5], expected_step))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--page-json")
    ap.add_argument("--page-wav")
    ap.add_argument("--mode", choices=list(MODES), default="collect")
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--interval", type=float, help="the page's interval override, if one was set")
    ap.add_argument("--volume", type=float, default=70, help="page volume in percent (default 70)")
    a = ap.parse_args()

    py = {"chirp": mct.sweep(*mct.CHIRP), "near": mct.sweep(*mct.NEAR), "marker": mct.sweep(*mct.MARKER)}

    for path in COPIES:
        if os.path.exists(path):
            check(dsp_block(path) == dsp_block(PAGE), "%s DSP block is identical to the conductor's" % os.path.relpath(path, ROOT))

    if shutil.which("node"):
        js = page_dsp_via_node()
        if js:
            for k in py:
                compare("node vs script, %s" % k, js[k], py[k])
            cases = [(0.5, 144, True, 1.0, 10), (0.15, 144, True, 1.0, 5), (0.25, 144, True, 1.0, 0.1), (0.123, 144, False, 0.0, 0.05)]
            for (iv, n, mk, lead, mins), lj in zip(cases, js["layouts"]):
                lp = mct.train_layout(iv, n, mk, lead, mins)
                same = (lj["step"], lj["total"], lj["markerStart"], lj["first"], lj["chirpStarts"]) == \
                       (lp["step"], lp["total"], lp["marker_start"], lp["first"], lp["chirp_starts"])
                check(same, "train layout matches (interval %g s, %g min, marker=%s): %d chirps" % (iv, mins, mk, len(lp["chirp_starts"])))
            pq = [mct.to_int16(v) for v in [-1.5, -1, -0.5, -1e-5, 0, 1e-5, 0.49999, 0.5, 1, 1.5]]
            check(js["q"] == pq, "16-bit quantization matches %s" % pq)
    else:
        print("SKIP  node not found; use --page-json instead")

    if a.page_json:
        pj = json.load(open(a.page_json))
        for k in py:
            compare("page JSON vs script, %s" % k, pj[k], py[k])

    if a.page_wav:
        file_mode, chirp = MODES[a.mode]
        interval = a.interval if a.interval is not None else mct.PRESETS[file_mode]["interval"]
        expected, lay = mct.render_train(chirp, interval, a.minutes, a.volume / 100)
        got = read_wav(a.page_wav)
        n_diff = sum(1 for i in range(0, min(len(got), len(expected)), 2) if got[i:i + 2] != expected[i:i + 2])
        check(len(got) == len(expected) and n_diff == 0,
              "page WAV vs script: %d vs %d samples, %d differ" % (len(got) // 2, len(expected) // 2, n_diff))
        check_spacing("page WAV", got, lay["step"])

    for name in ("train_collect_10min", "train_live_5min", "train_nearsilent_5min"):
        path = os.path.join(ROOT, "audio", name + ".wav")
        if os.path.exists(path):
            step = mct.PRESETS[name.split("_")[1]]["interval"] * mct.FS
            check_spacing(name + ".wav", read_wav(path), round(step))

    print("\n%s" % ("ALL PASSED" if not failures else "%d FAILED" % len(failures)))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
