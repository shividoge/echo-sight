# Raw recorder

Single-file page (`index.html`) that records the microphone with echo cancellation, noise suppression and auto gain turned off. It saves a 48 kHz, 16-bit mono WAV named `<session>_<role>.wav`.

Phone link: https://shividoge.github.io/echo-sight/recorder/ (the microphone needs HTTPS or localhost).

## Use it

1. Pick the role (left / right / camera) and use the **same session name** on every phone.
2. Press **Record** on every phone first, then **Start** on the conductor.
3. Watch the level meter. If it says it clipped, move the phone away from the speaker or turn the conductor down.
4. Press **Stop**. Under "Last take", the chirp check should show the number of chirps and a spacing of 24000 samples (0.5 s) for Collect mode, 7200 for Live.
5. **Save WAV**, or **Share / AirDrop** on iPhone, and send the file to the Mac. Check it there:

```bash
.venv/bin/python tools/check_recording.py <session>_<role>.wav --out docs/<session>_<role>_check.png
```

The "Echo cancel · noise · gain" box shows what the browser actually granted. It should read `off · off · off`. If any shows ON, that browser ignored the request; try another browser.

## Notes

- Keep the screen on and the page in front. Phones may pause the microphone for background tabs.
- Ten minutes is about 58 MB in memory. Save each take before recording the next one.
- Don't run the conductor and the recorder on the same iPhone: iOS tends to give the audio to only one of them.
