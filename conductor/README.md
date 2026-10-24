# Chirp conductor

Single-file page (`index.html`) that plays the Echo Sight chirp train. No build step and no libraries.

## Run it

Laptop (localhost counts as secure):

```bash
python3 -m http.server 8000 --directory conductor
```

Then open http://localhost:8000.

Phone: phones need HTTPS for audio unlock and the screen wake lock. Start the server above, then tunnel it:

```bash
npx localtunnel --port 8000
```

`cloudflared tunnel --url http://localhost:8000` works too. Open the https URL it prints on the phone.

## Use it

1. Start recording on every phone (left, right, camera).
2. Pick a mode, set the volume (70% default, keep it at 90% or below), then press **Start**.
3. Each train starts with the 1 s session marker (200 Hz → 8 kHz), then 0.5 s of silence, then the chirps.
4. Press **Stop**, then **Copy log** to get `{index, audio_time_s}` for every chirp that played.

**Download WAV** renders the exact same train offline at 48 kHz. Play that file through any speaker, including a Bluetooth one. Downloaded files start with 1 s of silence so a Bluetooth speaker can wake up before the marker. Phones may run out of memory rendering 10 minutes; use a laptop or `tools/make_chirp_train.py` for long files.

If the screen says the sample rate isn't 48000 Hz, the browser is resampling the chirp. Use the downloaded WAV for data collection in that case.

## Matching the Python generator

`tools/make_chirp_train.py` writes byte-identical WAVs. `python3 tools/test_chirp_match.py` runs the page's DSP block in Node and diffs it against the script. To check an actual page download:

```bash
python3 tools/test_chirp_match.py --page-wav ~/Downloads/train_collect_10min.wav --mode collect --minutes 10
```
