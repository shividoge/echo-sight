#!/usr/bin/env python3
"""Synthetic check for check_recording.py: plant echoes at known distances and recover them.

  .venv/bin/python tools/test_check_recording.py
"""
import argparse
import os
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf
from scipy import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_recording as cr  # noqa: E402
import make_chirp_train as mct  # noqa: E402

WALL_M = 1.2
FAR_M = 2.9
DRIFT_PPM = 50


def delay(x, metres):
    n = int(round(2 * metres / cr.C * cr.FS))
    return np.concatenate([np.zeros(n), x[:-n]])


def synthetic_recording(path):
    data, lay = mct.render_train(mct.CHIRP, 0.5, 0.5)
    s = np.frombuffer(data, dtype="<i2").astype(np.float64) / 32767
    rng = np.random.default_rng(1)
    t = np.arange(len(s)) / cr.FS
    y = s + 0.25 * delay(s, WALL_M) + 0.1 * delay(s, FAR_M)
    y += 0.2 * np.sin(2 * np.pi * 60 * t) + 0.02 * rng.standard_normal(len(s))  # mains hum + noise
    y = signal.resample_poly(y, 20000 + 1, 20000)  # recorder clock runs 50 ppm fast
    y = signal.resample_poly(y, 147, 160)  # saved at 44.1 kHz
    sf.write(path, 0.5 * y / np.abs(y).max(), 44100, subtype="PCM_16")
    return lay


def args(**kw):
    a = argparse.Namespace(template=os.path.join(cr.ROOT, "audio", "chirp_3ms.wav"),
                           marker=os.path.join(cr.ROOT, "audio", "marker_1s.wav"),
                           interval=0.5, min_gap=None, threshold=0.3, band=[2000, 15000], min_dist=0.2)
    vars(a).update(kw)
    return a


class CheckRecording(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = os.path.join(tempfile.mkdtemp(), "synthetic.wav")
        cls.layout = synthetic_recording(cls.path)
        cls.x = cr.load(cls.path).mean(axis=1)
        cls.r = cr.analyse(cls.x, args())

    def test_resampled_to_48k(self):
        expected = len(self.layout["chirp_starts"]) and self.layout["total"] * (1 + DRIFT_PPM * 1e-6)
        self.assertAlmostEqual(len(self.x), expected, delta=10)

    def test_marker_and_numbering(self):
        self.assertIsNotNone(self.r["marker"])
        self.assertAlmostEqual(self.r["marker"] / cr.FS, mct.LEAD_S, delta=0.002)
        self.assertEqual(list(self.r["index"]), list(range(len(self.layout["chirp_starts"]))))

    def test_spacing_and_drift(self):
        self.assertEqual(self.r["missing"], 0)
        self.assertAlmostEqual(self.r["drift_ppm"], DRIFT_PPM, delta=3)
        self.assertLess(self.r["jitter"], 1.0)

    def test_first_echo_distance(self):
        self.assertAlmostEqual(self.r["first_echo_median"], WALL_M, delta=0.03)
        self.assertAlmostEqual(self.r["coherent_first_echo"], WALL_M, delta=0.03)

    def test_blind_zone_skips_to_far_echo(self):
        r = cr.analyse(self.x, args(min_dist=1.5))
        self.assertAlmostEqual(r["coherent_first_echo"], FAR_M, delta=0.03)


if __name__ == "__main__":
    unittest.main(verbosity=2)
