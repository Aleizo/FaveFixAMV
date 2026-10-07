# FAVE FIX — AMV

An anime-style music video for **Fave Fix**. Every frame is generated in code (no third-party footage), and every cut lands on the song's beat grid.

- **Output:** `out/fave_fix_amv.mp4` — 1920×1080, 30 fps, H.264 plus AAC, 3:02
- **Style:** Shinkai-like cel-shaded skies, a heroine drawn as a silhouette, an anime eye in extreme close-up, a neon city with kanji signs, a hyperspace warp, manga speed lines, screentone freeze frames and kinetic typography

## How the video follows the music

`src/audio.py` analyses the track. It runs at a steady **140 BPM**, with beat *k* at `0.075 + k·60/140` seconds. From that analysis it exports per-frame envelopes: kick, snare, low, mid and high bands, and RMS.

The edit (`src/timeline.py`) is written in **beats**, not seconds:

| Section | Beats | Time | Look |
|---|---|---|---|
| Intro | –1…17 | 0:00–0:07 | night sky → title card |
| Verse 1 | 17…77 | 0:07–0:33 | golden hour → dusk: clouds, power lines, rooftop, eye, train, city lights turning on |
| Pre | 77…81 | 0:33–0:35 | FAVE / FIX focus-line hits |
| Chorus 1 | 81…148 | 0:35–1:03 | a cut every bar: warp run, neon city, moon run, kaleidoscope, typography |
| Breakdown | 148…154 | 1:03–1:06 | screentone freeze → silence |
| Verse 2 | 154…182 | 1:06–1:18 | rain: window bokeh, streetlight, a tear |
| Build | 182…218 | 1:18–1:33 | cuts accelerate from 2 bars to 1 beat |
| Bass section | 222…282 | 1:35–2:01 | blood moon eclipse, red hex tunnel, glitch |
| Chorus 2 | 286…350 | 2:02–2:30 | daylight palette |
| Final chorus | 354…418 | 2:31–2:59 | dawn: everything at once, camera roll |
| Outro | 418…end | 2:59–3:02 | title over the moon |

Post-FX in `src/render.py` respond to the audio:

- a zoom punch and screen shake on kicks
- chromatic aberration
- a flash on every cut
- glitch slices on snares in the bass section
- bloom, grain and vignette
- letterboxing in the calm sections

## Rendering

```bash
pip install numpy opencv-python-headless pillow librosa
python3 src/audio.py                              # analysis -> analysis/features.npz
python3 src/render.py                             # full video (about 20 min on 4 cores)
python3 src/render.py --start 30 --end 45 --half  # quick half-res preview of a range
python3 src/still.py eye 20 1 4 '{"iris":"rose"}' # a single still of one scene
```

The renderer needs `ffmpeg` and the Inter and WenQuanYi Zen Hei fonts installed (paths are in `src/fx.py`).

## Layout

- `src/fx.py` — drawing primitives (anti-aliased polygons, text, noise) and post effects
- `src/scenes.py` — every shot type: `sky`, `poles`, `rooftop`, `eye`, `train`, `city`, `moon`, `warp`, `hextunnel`, `kaleido`, `window`, `streetlight`, `focus`, `kinetic`, `title`, `freeze`, `black`
- `src/timeline.py` — the edit decision list on the beat grid
- `src/render.py` — parallel frame renderer, which pipes frames to ffmpeg and muxes in the song
- `assets/fave_fix.mp3` — the song
