"""Render the AMV.

  python3 src/render.py                       # full video -> out/fave_fix_amv.mp4
  python3 src/render.py --start 30 --end 40   # preview a range -> out/preview.mp4
"""
import argparse
import math
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fx  # noqa: E402
import scenes  # noqa: E402
import timeline  # noqa: E402
from audio import Audio, AUDIO, FPS, DURATION, ROOT  # noqa: E402

OUT = os.path.join(ROOT, "out")
SHOTS = timeline.shots()
_A = None


class Ctx:
    pass


def audio():
    global _A
    if _A is None:
        _A = Audio()
    return _A


def shot_at(t):
    for i, s in enumerate(SHOTS):
        if s["t0"] <= t < s["t1"]:
            return i, s
    return len(SHOTS) - 1, SHOTS[-1]


def cut_strength(i):
    s = SHOTS[i]
    if i == 0 or s["sec"] in ("drop", "outro"):
        return 0.0
    prev = SHOTS[i - 1]
    if prev["sec"] != s["sec"]:
        return 1.0 if s["sec"] in ("chorus", "final", "bass", "verse") else 0.6
    return {"chorus": 0.45, "final": 0.55, "bass": 0.35, "build": 0.4, "pre": 0.6}.get(s["sec"], 0.0)


def render_frame(f):
    A = audio()
    t = f / FPS
    i, s = shot_at(t)
    c = Ctx()
    c.t, c.lt = t, t - s["t0"]
    c.dur = s["t1"] - s["t0"]
    c.u = c.lt / c.dur
    c.A, c.frame, c.seed, c.p = A, f, i, s["p"]
    img = scenes.SCENES[s["scene"]](c)

    I = timeline.INTENSITY[s["sec"]]
    kick = A.get("kick", t)
    rng = np.random.default_rng(f * 7919 + 13)

    # camera: slow push-in, kick punch, shake on the heavy sections
    sc = 1.0 + 0.04 * fx.ease_in_out(c.u) + 0.05 * kick * I
    shake = 9 * kick * max(0.0, I - 0.4)
    ang = 0.0
    if s["sec"] in ("final",):
        ang = 1.2 * math.sin(t * 1.7)
    img = fx.zoom(img, sc, angle=ang, dx=rng.normal(0, shake), dy=rng.normal(0, shake))

    img = fx.glow(img, 16, 0.25 + 0.25 * I, 0.62)
    img = fx.chroma(img, 2.0 + 16 * kick * I)

    cs = cut_strength(i)
    if cs > 0 and c.lt < 0.25:
        img = fx.glitch(img, rng, cs * 0.8 * (1 - c.lt / 0.25)) if s["sec"] in ("bass", "build", "final") else img
    if s["sec"] in ("bass",) and A.get("snare", t) > 0.75:
        img = fx.glitch(img, rng, 0.35)
    if s["sec"] == "drop":
        img = fx.glitch(img, rng, 0.25 * (rng.random() < 0.3))
    flash = cs * math.exp(-c.lt * 14)
    if flash > 0.01:
        img = img + flash

    img = fx.vignette(img, 0.9)
    img = fx.grain(img, f, 0.03)
    if s["sec"] in timeline.LETTERBOX:
        img = fx.letterbox(img, 1.0)
    # global fades
    fade = min(1.0, t / 1.2) * min(1.0, max(0.0, (DURATION - t) / 1.4))
    if fade < 1:
        img = img * fade
    return fx.finish(img)


def render_chunk(args):
    f0, f1, path, half = args
    w, h = (fx.W // 2, fx.H // 2) if half else (fx.W, fx.H)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium" if half else "slow",
           "-crf", "22" if half else "16", "-pix_fmt", "yuv420p", "-tune", "animation", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for f in range(f0, f1):
        fr = render_frame(f)
        if half:
            import cv2
            fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    p.wait()
    return f0, f1, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=DURATION)
    ap.add_argument("--out", default=None)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--chunk", type=int, default=150)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tmp = os.path.join(OUT, "chunks")
    os.makedirs(tmp, exist_ok=True)
    f0, f1 = int(round(a.start * FPS)), int(round(min(a.end, DURATION) * FPS))
    jobs = []
    for k, s in enumerate(range(f0, f1, a.chunk)):
        jobs.append((s, min(f1, s + a.chunk), os.path.join(tmp, f"c{k:04d}.mp4"), a.half))
    t0 = time.time()
    done = 0
    with Pool(a.jobs) as pool:
        for s, e, dt in pool.imap_unordered(render_chunk, jobs):
            done += e - s
            el = time.time() - t0
            print(f"[{done}/{f1 - f0}] chunk {s}-{e} {dt:.0f}s  elapsed {el:.0f}s  eta {el / done * (f1 - f0 - done):.0f}s",
                  flush=True)
    lst = os.path.join(tmp, "list.txt")
    with open(lst, "w") as fh:
        for j in jobs:
            fh.write(f"file '{os.path.basename(j[2])}'\n")
    out = a.out or os.path.join(OUT, "fave_fix_amv.mp4" if (f0 == 0 and f1 >= int(DURATION * FPS) - 1) else "preview.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                    "-ss", f"{f0 / FPS:.3f}", "-t", f"{(f1 - f0) / FPS:.3f}", "-i", AUDIO,
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                    "-movflags", "+faststart", "-shortest", out], check=True)
    for j in jobs:
        os.remove(j[2])
    os.remove(lst)
    print("wrote", out, f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
