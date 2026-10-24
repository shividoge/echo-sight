# Echo Sight — Project Handoff (read this first)

This file is the single source of truth for the project. Any Claude session (on any account) can pick up the work from here without the original conversation.

**If you are a new Claude Code session:** read this whole file, then go to **"Next step"** under Status and continue from there. Do not ask the user to re-explain the project. Update the Status section and the Work log before you stop working (see "Handoff protocol" at the bottom).

---

## 1. What we are building

**Event:** Cal Hacks 13.0, Oct 23–25, 2026, Palace of Fine Arts, San Francisco. Three-person team. Shivin Anand (computer engineering) owns this repo's hardware/audio pieces. Development and filming happen at Shivin's brother's house nearby; judging happens at the Palace on Sunday.

**One-sentence pitch:** "Have you ever had to find your way through a room you couldn't see?" Echo Sight lets your phone see in the dark by listening, like a bat.

**How it works:** a speaker plays a short chirp, phones record the echoes, and two layers turn the echoes into a picture of the room:

1. **Classic layer (no training):** cross-correlate the recording with the chirp, find echo delays, convert to distance (`d = delay × 343 / 2`), and combine with phone heading into a live floor plan.
2. **Learned layer:** a U-Net turns a 2-channel echo spectrogram into a 128×128 depth image. Pretrained on public echo datasets, fine-tuned on our own recordings.

**Use cases in the pitch:** firefighters in smoke, blind and low-vision navigation, power outages. Fully offline (local Wi-Fi only).

**Target prize tracks:** Most Technically Advanced (primary), Cal Hacks Overall, Microsoft Built for the Real World, People-First Social Impact, Most Creative, Hacker's Choice.

## 2. Locked decisions (do not reopen without the user)

- **Chirp:** 3 ms linear sweep, 20 Hz → 20 kHz, matching the BatVision and VisualEchoes datasets so pretrained models transfer. A research study found depth accuracy drops when echoes stay ultrasonic, so the model uses the audible chirp. A near-silent 17–20 kHz chirp exists only for the classic distance mode.
- **Sample rate:** 48 kHz everywhere, 16-bit PCM WAV, never compressed.
- **Sync method:** every device hears every chirp. Align all recordings by finding the chirps in each file. No network clock sync.
- **Rig:** a ~24 cm bar, left and right "ear" phones at each end, speaker centered, camera phone on top. Wired speaker preferred; Bluetooth acceptable for the audible chirp because alignment uses the chirps themselves.
- **Datasets (in order):** VisualEchoes + Beyond Image to Depth pretrained weights (fastest), BatVision (best real data), synthetic rooms from pyroomacoustics (matched to our rig). **STARSS23 (zenodo 7880637) is the wrong dataset**: it labels sound *sources*, has no chirp echoes or depth. Do not use it for mapping.
- **Compute:** two ASUS Ascent GX10s (NVIDIA GB10, 128 GB unified memory, ARM Linux). Use NVIDIA's PyTorch container from NGC.
- **Rules:** Cal Hacks expects code written during the event. Study Echoscope (github.com/lifeart/echoscope) and EchoWalk (github.com/GroundstateTech/EchoWalk) as references only; write our own code. Credit every dataset and pretrained model in the Devpost.

## 3. The data contract (every component follows this)

| Item | Format |
| --- | --- |
| Chirp | 3 ms linear sweep 20 Hz → 20 kHz, 5% raised-cosine fade at each end |
| Session marker | 1.0 s linear sweep 200 Hz → 8 kHz, then 0.5 s silence, at the start of every session |
| Chirp rate | every 0.5 s while collecting data; every 0.15 s live |
| Raw recordings | 48 kHz, 16-bit mono WAV, one file per device per session, named `<session>_<role>.wav` (role = left / right / camera) |
| Training sample | `data/<session>/<room>/<sample_id>/` containing `left.wav`, `right.wav` (72.5 ms = 3,480 samples each, starting at the direct chirp), `depth.npy` (128×128 float32 meters, clipped at 12 m, NaN = invalid), `rgb.jpg`, `meta.json` |
| meta.json | `{session, room, chirp_index, chirp_time_s, direct_peak, holdout}` |
| Live stream | WebSocket `ws://<gx10>:8765/live`. Client sends JSON header `{phone, chirp_index, heading_deg}` then the window as Float32. Server broadcasts `{chirp_index, nearest_m, confidence, depth_png_b64, map_point}` on `/view` |

## 4. Repo layout (create as you go)

```
CLAUDE.md                 this file
conductor/                Task 1: chirp conductor web page
tools/
  make_chirp_train.py     Task 1b: renders chirp-train WAV files
  check_recording.py      Task 2: verifies chirps + echoes in a recording
recorder/                 Task 3: raw recorder web app (later becomes live client)
slicer/                   Task 4: recordings -> training samples
docs/                     exported plans and notes
```

---

## 5. TASK 1 (current): Chirp Conductor

The conductor is the thing that sends out the sound. It must produce the exact same chirp train every time so recordings line up and match the datasets.

### 1a. Conductor web page (`conductor/index.html`, single file, no build step)

Must run from a phone or laptop browser. Serve over HTTPS or localhost (phones need HTTPS for audio unlock and wake lock); include a one-line `python -m http.server` + tunnel note in a README.

**Features:**

1. **Start button** (audio may only start after a user tap). Creates an `AudioContext` at 48 kHz (fall back to the device rate and resample the buffer if 48 kHz is refused; show the actual rate on screen).
2. **Chirp generation in JS:** build the chirp as a Float32Array: `x[n] = sin(2π (f0·t + (f1−f0)/(2T)·t²))`, f0 = 20, f1 = 20000, T = 0.003 s, with a 5% raised-cosine fade in and out. Build the session marker the same way (200 → 8000 Hz, 1.0 s).
3. **Sample-accurate scheduling:** schedule each chirp as an `AudioBufferSourceNode` with `start(when)` on the AudioContext clock, scheduling ~2 s ahead in a lookahead loop. Never rely on `setInterval` timing for the audio itself.
4. **Modes (dropdown):**
   - *Collect:* marker, then one chirp every 0.5 s.
   - *Live:* marker, then one chirp every 0.15 s.
   - *Near-silent range:* 17–20 kHz chirp every 0.15 s (classic distance mode only).
   - *Single:* one chirp per tap (for testing).
5. **Controls:** volume slider (default 70%, warn above 90% about clipping), Stop button, interval override field.
6. **Display:** big chirp counter, elapsed time, actual sample rate, current mode, and a "recording should be running on all phones" reminder before Start.
7. **Chirp log:** record `{index, audio_time_s}` for every chirp; a "Copy log" button copies JSON (clipboard write inside the click handler, with a select-text fallback).
8. **Screen wake lock** while running (request it; ignore rejection).
9. **Download WAV of the exact train** (render the same schedule with `OfflineAudioContext`, encode 16-bit PCM WAV) so the same file can play through any speaker or Bluetooth device.

**Design:** plain, readable at arm's length, dark-friendly, large tap targets. No external libraries.

### 1b. `tools/make_chirp_train.py`

Python (numpy + soundfile, or the standard `wave` module) that writes WAV files with identical parameters to the page:

- `chirp_3ms.wav` (template used by every analysis script)
- `train_collect_10min.wav` (marker + chirps every 0.5 s for 10 minutes)
- `train_live_5min.wav` (marker + chirps every 0.15 s)
- `train_nearsilent_5min.wav` (17–20 kHz)
- Prints the chirp start times to `train_*_times.csv`.

CLI flags: `--interval`, `--minutes`, `--f0`, `--f1`, `--out`. The page and this script must agree sample-for-sample on the chirp; add a test that compares them (export the page's chirp as JSON or WAV and diff).

### Acceptance tests for Task 1

- [x] Spectrogram (Audacity) of the downloaded WAV shows the marker sweep, then evenly spaced vertical lines at the chosen interval. *(2026-10-23: checked with `tools/plot_spectrogram.py` because Audacity wasn't installed; see `docs/collect_spectrogram.png`.)*
- [x] Measured spacing between chirps in the downloaded WAV is exact to the sample. *(Matched filter: 1196/1196 gaps = 24000 samples in collect, 1989/1989 = 7200 in live.)*
- [ ] Recording the speaker with an iPhone (Voice Memos set to Lossless) shows the chirps clearly.
- [ ] Page runs 10 minutes on a phone without stopping or drifting (chirp count matches elapsed ÷ interval).

## 6. TASK 2 (next): `tools/check_recording.py`

Answers "did the echoes survive the phone's audio processing?"

1. Load a recording and `chirp_3ms.wav`; resample to 48 kHz if needed.
2. FFT cross-correlate, take the envelope (Hilbert magnitude), find chirp peaks ≥ 0.4 s apart above 30% of max.
3. Locate the session marker with its own template; number chirps from it.
4. For 20 random chirps, plot the first 72.5 ms of envelope after each direct peak, stacked, with a distance axis (meters = samples / 48000 × 343 / 2).
5. Print: chirps found, mean spacing, drift vs expected, and the median distance of the first echo past 0.2 m.
6. Save the plot as PNG.

**Pass:** a consistent echo bump appears at the right distance when the rig faces a wall at a measured distance (try 1 m and 2 m).

If echoes look smeared or missing, phone processing is the cause: move to Task 3 (recorder with processing disabled) or a native iOS app using `AVAudioSession` mode `.measurement`.

## 7. Later tasks (summaries; full detail in the implementation guide)

- **Task 3, recorder web app:** `getUserMedia` with `echoCancellation:false, noiseSuppression:false, autoGainControl:false`, AudioWorklet raw capture, 16-bit WAV out, role selector, level meter. Becomes the live client Sunday.
- **Task 4, slicer:** find chirps in every file, align devices by chirp index, correct clock drift with a linear fit, extract camera frames by the camera's audio track (`ffmpeg -i cam.mov -ac 1 -ar 48000 cam.wav`), cut 72.5 ms windows, write samples per the contract, skip clipped windows.
- **Classic ranging + spin-to-map:** matched filter, baseline subtraction, 0.2 m blind zone, threshold 4× local noise, median smoothing; heading from `webkitCompassHeading` (iOS, after permission) or `alpha` (Android); occupancy grid at 5 cm, RANSAC walls.
- **Model:** input `[2, 256, 256]` log spectrograms (FFT 512, window 64, hop 16); output 128×128 depth / 12 m; BatVision U-Net; L1 loss on valid pixels; AdamW; fine-tune at 1e-4 with encoder frozen 2 epochs; holdout one whole room; must beat an average-depth baseline.
- **Live system, hero video, website, Devpost.**

## 8. Key links

- BatVision dataset + U-Net: https://github.com/AmandineBtto/Batvision-Dataset (mirror: https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/HYLZNL)
- VisualEchoes data: https://github.com/facebookresearch/VisualEchoes
- Pretrained echo-to-depth weights: https://github.com/krantiparida/beyond-image-to-depth
- pyroomacoustics: https://pyroomacoustics.readthedocs.io/
- BatVision paper: https://arxiv.org/pdf/2303.07257
- Ultrasonic vs audible echoes study: https://arxiv.org/abs/2409.03336
- Team plan and implementation guide live as claude.ai docs on the original account; links from the original account may not open on another account, so export both into `docs/` (Markdown) as soon as possible.

---

## Status

**Last updated:** Fri Oct 23, 2026, ~10 PM PT (Claude Code session, Task 2 built)

**Done:**
- Idea locked (Echo Sight). Methods researched and ranked. Implementation guide and team plan written (claude.ai docs).
- Teammate on GX10s started with STARSS23; redirected to VisualEchoes + Beyond Image to Depth weights.
- Task 1a: `conductor/index.html` (all 9 features) + `conductor/README.md` (how to serve and tunnel).
- Task 1b: `tools/make_chirp_train.py`, standard library only (no numpy needed). It writes `audio/chirp_3ms.wav`, `chirp_nearsilent_3ms.wav`, `marker_1s.wav`, the three trains, and `train_*_times.csv`. `audio/` is gitignored except `chirp_3ms.wav`; rerun the script to regenerate.
- `tools/test_chirp_match.py` passes. It runs the page's `// BEGIN DSP … // END DSP` block in Node and checks chirp, near-silent chirp, marker, train layout and 16-bit quantization against the script: 0 samples differ. `--page-wav` / `--page-json` check real page downloads.
- Verified in a browser: the page's OfflineAudioContext WAV is byte-identical to the Python file for collect/live/near-silent, a custom 0.123 s interval at 95% volume, and the full 10-min collect train (10 s render on a laptop).
- Live run checked: AudioContext at 48000 Hz; chirp counter = expected count; logged spacing exactly 7200 samples in live mode; Stop/restart works; no console errors.
- Acceptance test 2 (spacing exact to the sample) passes on every generated train.

- First real recording (2026-10-23, no speaker yet): MacBook speakers played `train_collect_10min.wav` (35 s) and an iPhone recorded it in Voice Memos Lossless (48 kHz, 24-bit, stereo). The file is `audio/recordings/test_1m_laptop*.wav` (gitignored); plots are `docs/test_1m_laptop_*.png`. Results:
  - All 65 expected chirps were found. The marker was detected at 1500× the median level.
  - Mac-to-iPhone clock offset is 13 ppm, which the slicer's linear fit will correct. Peak jitter is about 8 samples.
  - The chirp only reached about 0.2–15 kHz (laptop speaker and/or iPhone processing). There was no clipping (peak 0.33).
  - Shivin says the laptop was about 1 m from the wall. Full-band analysis showed no clear echo there.
  - **What worked:** band-pass 2–15 kHz on both the recording and the template, matched filter, then a *coherent* average of all chirps (`docs/test_1m_laptop_coherent_2-15k.png`). Both mics then show a weak bump at about **0.85 m**, about 3 dB above the decaying ringing, at −35 to −38 dB re direct. That fits the iPhone being held in front of the laptop. There is also a strong reflection at about 2.3 m on both mics, probably the wall behind, not confirmed. The noise floor at 1–2 m is −45 to −55 dB.
  - **Lesson for Task 2 and the slicer:** below 2 kHz there was a smooth plateau at −22 to −28 dB that coherent averaging did not reduce. Probably 60 Hz mains hum: 0.5 s and 0.15 s are both whole numbers of 60 Hz cycles, so hum stays in step with the chirps. Always band-pass before ranging. The laptop speaker also rings for about 0.8 m worth of delay, which hides near echoes; a real speaker should shorten this.
  - `check_recording.py` (below) on the same file: drift +13 ppm, jitter 3 samples, 1 off-grid peak rejected per mic. Mic 0 has a sharp echo at **2.15 m** that shows in every chirp; there is nothing consistent at 1 m (the 0.85 m bump is under 4× local noise). Probable cause, and a mistake in my instructions: Shivin was told to face the laptop *screen* at the wall, but MacBook speakers fire up out of the keyboard deck, so the screen shielded the wall. The 2.15 m echo is probably behind or beside the user. Plot: `docs/test_1m_laptop_check.png`.
  - Test 3 is not ticked yet. Re-record with the keyboard/speakers facing the wall (or a phone/real speaker as the source), at a measured 1 m and then 2 m. The wall echo should move by 1 m between them.
- Task 2: `tools/check_recording.py` does everything in section 6 except item 1's "resample" fallback order (it resamples any rate to 48 k). Reads .m4a directly via afconvert/ffmpeg; `--channel mix|all|N`. Extras beyond the spec:
  - band-pass 2–15 kHz by default (`--band 0 0` for full band);
  - off-grid peaks rejected (best peak per index, then iterative line fit, ±2 ms);
  - drift from a linear fit, in ppm;
  - a coherent average drawn above the 20 stacked chirps;
  - first echo = first local max past `--min-dist` that is ≥4× the **local** median (0.5 m window), refined to the highest point within 0.25 m. A global noise threshold failed: after coherent averaging, the direct chirp's fading tail beat it.
- `tools/test_check_recording.py` passes (5 tests). It builds a synthetic recording (echoes at 1.2 m and 2.9 m, 60 Hz hum, noise, +50 ppm clock, saved at 44.1 kHz) and checks marker, numbering, drift ±3 ppm, jitter <1 sample, and echo distance ±3 cm.

**Implementation choices to know (not in the original spec):**
- Downloaded/generated WAVs start with **1.0 s of silence** before the marker (Bluetooth wake-up). First chirp is at 2.5 s. The live page starts the marker 0.3 s after Start.
- The near-silent train also starts with the (audible) session marker, so recordings can still be aligned.
- Fade = `round(0.05 × N)` samples (7 for the 144-sample chirp), `w = 0.5(1 − cos(π i / fade))`; the first and last samples are 0.
- Intervals are rounded to whole samples at 48 kHz, and every chirp time is `first + k × step`, so there is no drift.
- 16-bit quantization: `floor(clamp(x) × 32767 + 0.5)` in both page and script. Default gain 0.7 = page default volume.
- Chirp scheduling uses a Worker timer (200 ms tick, 2 s lookahead) so background tabs don't starve it. iOS: sets `navigator.audioSession.type = 'playback'` to play through the mute switch.

- Repo: https://github.com/shividoge/echo-sight (public; default branch `task1-conductor`, no `main` pushed yet). GitHub Pages serves that branch: the conductor is at https://shividoge.github.io/echo-sight/conductor/ (HTTPS, so it works on phones).

**In progress:** nothing.

**Next step:**
1. Waiting on Shivin (deferred "for later"): (a) re-record the wall test with the speakers *facing* the wall at a measured 1 m and 2 m; (b) the 10-min Collect run on a phone at https://shividoge.github.io/echo-sight/conductor/ (counter must equal the expected count). Check each recording with `.venv/bin/python tools/check_recording.py <file.m4a> --channel all --out docs/<name>_check.png`. Pass = the wall echo sits at the measured distance and moves 1 m between recordings. Then tick tests 3–4 in section 5. Shivin has no speaker yet: MacBook speakers or a second phone are the stopgap; a small wired speaker is needed for the rig.
2. Next code task: Task 3, the recorder web app (`recorder/index.html`, section 7). Start it while waiting for (1); its whole point is raw capture with `echoCancellation/noiseSuppression/autoGainControl: false`, which removes the question of whether Voice Memos processing hurts echoes. Host it on the same Pages site and reuse the conductor's style.

**Open questions for the user:**
- Does anyone on the team have an iPhone Pro (LiDAR) and a Mac with Xcode?
- What exact speaker will the rig use (wired or Bluetooth)?
- Cal Hacks Devpost submission deadline (confirm and write it here).

## Work log

Append one line per work chunk, newest last: `date time | who/which session | what changed | what's next`.

- 2026-10-23 20:46 | claude.ai planning session | created this handoff file | build Task 1
- 2026-10-23 20:56 | Claude Code (Shivin's Mac) | created repo at ~/echo-sight (branch task1-conductor); built conductor page, make_chirp_train.py, test_chirp_match.py; page and script byte-identical | hardware acceptance tests for Task 1, then Task 2
- 2026-10-23 21:20 | Claude Code (Shivin's Mac) | pushed to github.com/shividoge/echo-sight + Pages; added .venv and tools/plot_spectrogram.py; acceptance tests 1 and 2 pass | iPhone recording test + 10-min phone run
- 2026-10-23 22:00 | Claude Code (Shivin's Mac) | analysed first iPhone recording (chirps/timing good, no 1 m echo: speakers were shielded by the screen); built tools/check_recording.py + synthetic test (passes) | Shivin re-records later; next code task is Task 3 recorder

---

## Handoff protocol (how switching accounts works)

**For Claude (every session, every account):**
1. On start: read this file top to bottom; check `git log` and the repo tree to confirm Status matches reality.
2. While working: keep changes small; commit after each working piece with a clear message.
3. Before stopping, or whenever context or credits may run low: update **Status** (Done / In progress / Next step / Open questions) and append to **Work log**, then commit. Write the next step so concretely that a fresh session could start it with no questions.
4. Never delete decisions from section 2; if the user changes one, edit it and note the change in the Work log.

**For Shivin (switching accounts):**
1. Make sure the latest work is committed and pushed (GitHub), with this file at the repo root.
2. On the new account, open Claude Code in the repo folder (or clone it).
3. Say: **"Read CLAUDE.md and continue from Next step."**

Claude Code loads `CLAUDE.md` from the repo root automatically, so the new session starts with all of this context.
