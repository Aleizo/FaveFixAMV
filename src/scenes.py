"""Shot renderers. Each scene is f(ctx) -> float32 HxWx3 image.

ctx fields: t (song time), lt (time inside shot), dur, u (lt/dur), A (Audio),
frame (global frame index), seed, p (shot params dict).
"""
import functools
import math

import cv2
import numpy as np

from fx import (W, H, hexc, mix, smoothstep, ease_out, ease_in_out, vgrad, radial, catmull, capsule,
                ellipse_pts, new_mask, fill, stroke, circle, comp, add, blur, fast_blur, glow, fbm,
                draw_text, text_mask, place, paint_poly, FONT_BLACK, FONT_BLACK_IT, FONT_LIGHT, FONT_JP, halftone)

# ------------------------------------------------------------------ palettes
PAL = {
    "golden": dict(sky=[(0, "#27306e"), (0.45, "#9a5a9e"), (0.75, "#f4876b"), (1.0, "#ffd08a")],
                   cloud_sh="#6b4a8f", cloud_mid="#e98a8f", cloud_hi="#ffe2b0", sun="#fff3c4", sil="#1a1230",
                   rim="#ffb36b"),
    "dusk": dict(sky=[(0, "#0e1240"), (0.5, "#4a2f7a"), (0.8, "#c2507a"), (1.0, "#ff9a76")],
                 cloud_sh="#2c2357", cloud_mid="#7a3f7f", cloud_hi="#ff9d8a", sun="#ffd1a8", sil="#0b0a1c",
                 rim="#ff7aa0"),
    "night": dict(sky=[(0, "#03040f"), (0.55, "#0c1236"), (0.85, "#24235a"), (1.0, "#3a2c66")],
                  cloud_sh="#0a0d26", cloud_mid="#1d2152", cloud_hi="#5560a8", sun="#e8f0ff", sil="#02020a",
                  rim="#8fb6ff"),
    "day": dict(sky=[(0, "#1d5fd6"), (0.55, "#4f9bf0"), (0.85, "#a8d6ff"), (1.0, "#e8f6ff")],
                cloud_sh="#8aa6d8", cloud_mid="#dce9fb", cloud_hi="#ffffff", sun="#ffffff", sil="#16233f",
                rim="#ffffff"),
    "dawn": dict(sky=[(0, "#26357a"), (0.45, "#7a6cc0"), (0.75, "#ff9fa0"), (1.0, "#fff0c8")],
                 cloud_sh="#7d6db0", cloud_mid="#ffb0b8", cloud_hi="#fff6dc", sun="#fffbe8", sil="#221a3e",
                 rim="#ffe0a0"),
    "blood": dict(sky=[(0, "#07020a"), (0.5, "#2a0618"), (0.8, "#7a0f2a"), (1.0, "#ff3b3b")],
                  cloud_sh="#14040e", cloud_mid="#4a0a20", cloud_hi="#ff5a5a", sun="#ffe0d0", sil="#050006",
                  rim="#ff4060"),
}


def pal(name):
    d = PAL[name]
    return {k: ([(p, hexc(c)) for p, c in v] if k == "sky" else hexc(v)) for k, v in d.items()}


@functools.lru_cache(maxsize=8)
def sky_bg(name):
    return vgrad(pal(name)["sky"])


# ------------------------------------------------------------------ clouds

@functools.lru_cache(maxsize=6)
def cloud_layer(name, seed, w_mult=3, top=0.15, bottom=0.85, cover=0.52, scale=1.0):
    """Pre-rendered cel-shaded cumulus band as (rgb, alpha) of width W*w_mult."""
    P = pal(name)
    w = W * w_mult
    h = H
    d = fbm(h, w, seed, base=int(5 * scale * w_mult), octaves=6, persistence=0.55, wrap_x=True)
    d2 = fbm(h, w, seed + 1, base=int(2 * w_mult), octaves=2, wrap_x=True)
    ys = np.linspace(0, 1, h)[:, None].astype(np.float32)
    band = smoothstep(top, top + 0.2, ys) * (1 - smoothstep(bottom - 0.25, bottom, ys))
    dens = d * (0.6 + 0.6 * d2) * band
    a = smoothstep(cover * 0.62, cover * 0.62 + 0.012, dens)
    # lighting from below-right (low sun) -> undersides glow; computed on a smoothed field so the
    # shading reads as big cel-shaded lobes instead of noise
    ds = blur(dens, 10)
    sh = np.roll(np.roll(ds, -26, axis=0), -16, axis=1)
    lit = blur(ds - sh, 6) + (dens - ds) * 0.6
    lit = (lit - np.percentile(lit, 5)) / (np.percentile(lit, 95) - np.percentile(lit, 5) + 1e-6)
    t1 = smoothstep(0.40, 0.43, lit)
    t2 = smoothstep(0.70, 0.73, lit)
    core = smoothstep(cover * 0.62 + 0.02, cover * 0.62 + 0.16, dens)
    rgb = P["cloud_sh"] + (P["cloud_mid"] - P["cloud_sh"]) * t1[..., None]
    rgb = rgb + (P["cloud_hi"] - rgb) * t2[..., None]
    # silver lining at the edges
    edge = np.clip(a - core, 0, 1)
    rgb = rgb + (P["cloud_hi"] - rgb) * (edge * 0.75)[..., None]
    # slight vertical darkening toward top for depth
    rgb = rgb * (0.85 + 0.15 * ys[..., None])
    return rgb.astype(np.float32), a.astype(np.float32)


def scroll(layer, x):
    rgb, a = layer
    w = rgb.shape[1]
    x = x % w
    xi = int(x)
    fr = x - xi
    idx = (np.arange(W) + xi) % w
    r = rgb[:, idx]
    al = a[:, idx]
    if fr > 0.01:
        idx2 = (idx + 1) % w
        r = r * (1 - fr) + rgb[:, idx2] * fr
        al = al * (1 - fr) + a[:, idx2] * fr
    return r, al


def sky(ctx, name, pan=30.0, sun=None, stars=0.0, clouds=True, cover=0.52, top=0.12, bottom=0.8, seed=3,
        yoff=0.0):
    P = pal(name)
    img = sky_bg(name).copy()
    if stars > 0:
        star_field(img, ctx, stars)
    if sun is not None:
        sx, sy, sr = sun
        r = radial(sx, sy, sr)
        g = np.exp(-np.maximum(r - 1, 0) * 1.4)[..., None] * 0.55
        img += P["sun"] * g * 0.6
        m = np.clip((1.03 - r) * 30, 0, 1)
        img += (P["sun"] - img) * m[..., None]
    if clouds:
        far = cloud_layer(name, seed + 10, 3, top, bottom * 0.9, cover * 1.06, 0.7)
        rgb, a = scroll(far, ctx.lt * pan * 0.45 + 900)
        a = a * 0.8
        if yoff:
            rgb, a = np.roll(rgb, int(yoff * 0.5), 0), np.roll(a, int(yoff * 0.5), 0)
        img += (rgb * 0.92 - img) * a[..., None]
        near = cloud_layer(name, seed, 3, top + 0.05, bottom, cover)
        rgb, a = scroll(near, ctx.lt * pan)
        if yoff:
            rgb, a = np.roll(rgb, int(yoff), 0), np.roll(a, int(yoff), 0)
        img += (rgb - img) * a[..., None]
    return img


@functools.lru_cache(maxsize=4)
def _stars(seed, n=900):
    rng = np.random.default_rng(seed)
    return (rng.random(n) * W, rng.random(n) * H * 0.85, rng.random(n) ** 3, rng.random(n) * 6.28)


def star_field(img, ctx, amt, seed=11, drift=0.0):
    xs, ys, br, ph = _stars(seed)
    m = np.zeros((H, W), np.float32)
    tw = 0.55 + 0.45 * np.sin(ctx.t * 3 + ph)
    xi = ((xs + ctx.lt * drift) % W).astype(int)
    yi = ys.astype(int)
    np.add.at(m, (yi, xi), br * tw * amt)
    m = blur(m, 0.9) * 3.5
    img += np.clip(m, 0, 1.5)[..., None] * np.array([0.9, 0.95, 1.0], np.float32)
    big = br > 0.75
    for x, y, b in zip(xi[big], yi[big], (br * tw)[big]):
        sz = 6 + 10 * b
        cv2.line(img, (int(x - sz), y), (int(x + sz), y), (b * amt * .7,) * 3, 1, cv2.LINE_AA)
        cv2.line(img, (int(x), int(y - sz)), (int(x), int(y + sz)), (b * amt * .7,) * 3, 1, cv2.LINE_AA)
    return img


# ------------------------------------------------------------------ the heroine (silhouette)

def _rot(v, a):
    c, s = math.cos(a), math.sin(a)
    return np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c])


def girl_shapes(t, pose="stand", wind=1.0, phase=0.0):
    """Polygons for the heroine in local units (feet at y=0, head top ~ y=-1, facing +x)."""
    polys = []
    run = pose == "run"
    cyc = t * 2 * math.pi * 1.6 + phase  # running cadence
    lean = 0.28 if run else 0.0
    bob = (abs(math.sin(cyc)) * 0.03) if run else math.sin(t * 1.3) * 0.004
    hip = np.array([0.0, -0.47 - bob])

    def up(v):  # body points rotate around hip by lean
        return hip + _rot(np.array(v) - np.array([0, -0.47]), lean)

    # legs
    if run:
        for side, off in ((1, 0), (-1, math.pi)):
            a = math.sin(cyc + off)
            th_ang = a * 0.9
            knee = hip + _rot(np.array([0, 0.24]), -th_ang)
            bend = 0.2 + max(0.0, -math.cos(cyc + off)) * 1.5
            foot = knee + _rot(np.array([0, 0.24]), -th_ang + bend)
            polys.append(capsule(hip + [0.0, 0.0], knee, 0.034, 0.024))
            polys.append(capsule(knee, foot, 0.024, 0.016))
            polys.append(ellipse_pts(*(foot + _rot(np.array([0.025, 0.004]), -th_ang + bend)), 0.04, 0.016,
                                     -th_ang + bend))
    else:
        for x in (-0.025, 0.03):
            knee = np.array([x + 0.005, -0.24])
            foot = np.array([x + 0.008, -0.02])
            polys.append(capsule(hip + [x, 0.02], knee, 0.032, 0.022))
            polys.append(capsule(knee, foot, 0.022, 0.015))
            polys.append(ellipse_pts(foot[0] + 0.02, -0.012, 0.042, 0.016))
    # torso
    torso = [up([-0.05, -0.47]), up([-0.065, -0.62]), up([-0.07, -0.75]), up([-0.03, -0.79]), up([0.03, -0.79]),
             up([0.065, -0.74]), up([0.075, -0.68]), up([0.05, -0.6]), up([0.045, -0.47])]
    polys.append(catmull(torso, 6))
    # skirt with flutter
    hem = []
    n = 7
    for i in range(n):
        u = i / (n - 1)
        x = mix(0.12, -0.15, u) - wind * 0.06 * u
        y = -0.36 + 0.025 * math.sin(t * 7 + u * 9) * wind + (0.03 * u if run else 0)
        hem.append(up([x, y]) if run else np.array([x, y - bob]))
    skirt = [up([0.05, -0.56]), *hem, up([-0.055, -0.56])]
    polys.append(np.array(catmull(skirt, 6, closed=True)))
    # neck + head
    neck = up([0.005, -0.80])
    head_c = up([0.02, -0.885])
    polys.append(capsule(up([0.0, -0.76]), neck, 0.024, 0.022))
    polys.append(ellipse_pts(head_c[0], head_c[1], 0.066, 0.076, lean * 0.6))
    # face profile: brow, nose, lips, chin
    prof = [[0.0, -0.04], [0.058, -0.045], [0.068, -0.012], [0.086, 0.014], [0.068, 0.024], [0.071, 0.036],
            [0.063, 0.046], [0.06, 0.064], [0.03, 0.078], [0.0, 0.05]]
    polys.append(catmull([head_c + _rot(np.array(v), lean * 0.6) for v in prof], 5))
    # arms
    sh = up([0.0, -0.75])
    for side, off in ((1, 0.0), (-1, math.pi)):
        if run:
            a = math.sin(cyc + off + math.pi) * 1.1
            el = sh + _rot(np.array([0, 0.15]), -a)
            hand = el + _rot(np.array([0, 0.13]), -a - 1.3)
        else:
            sw = math.sin(t * 1.1 + off) * 0.03
            el = sh + np.array([0.01 * side + sw, 0.15])
            hand = el + np.array([0.02 + sw, 0.13])
        polys.append(capsule(sh, el, 0.02, 0.016))
        polys.append(capsule(el, hand, 0.016, 0.012))
        polys.append(ellipse_pts(hand[0], hand[1], 0.017, 0.02))
    # hair: crown + long flowing strands
    crown = [head_c + v for v in ([0.075, -0.02], [0.06, -0.075], [0.0, -0.1], [-0.06, -0.07], [-0.08, 0.0],
                                  [-0.06, 0.06])]
    polys.append(catmull(crown, 6))
    # long hair: each strand integrates a direction that bends from "down" toward "back" with wind
    flow = 0.35 + 0.45 * wind + (0.55 if run else 0)
    for k in range(6):
        root = head_c + np.array([-0.035 - 0.008 * k, -0.045 + 0.022 * k])
        L = (0.40 + 0.05 * math.sin(k * 2.3)) * (0.75 if run else 1.0)
        segs = 12
        pos = root.copy()
        pts_a, pts_b = [], []
        for i in range(segs + 1):
            u = i / segs
            ang = min(1.45, flow * (0.25 + 1.1 * u)) + lean * 0.8
            ang += math.sin(t * (3.2 + 0.35 * k) - u * 4.5 + k * 1.7) * 0.22 * u * (0.3 + wind)
            d = np.array([-math.sin(ang), math.cos(ang)])
            nrm = np.array([-d[1], d[0]])
            wdt = (0.05 - 0.004 * k) * (1 - u) ** 0.8 + 0.002
            pts_a.append(pos + nrm * wdt)
            pts_b.append(pos - nrm * wdt)
            pos = pos + d * (L / segs)
        polys.append(np.vstack([pts_a, pts_b[::-1]]))
    # scarf / ribbon tails from the neck
    for k, (ln, ph) in enumerate(((0.30, 0.0), (0.24, 1.3))):
        rib_a, rib_b = [], []
        pos = neck + np.array([-0.01, 0.015])
        for i in range(13):
            u = i / 12
            ang = 0.5 + (0.6 + 0.5 * wind + (0.4 if run else 0)) * u + k * 0.15
            ang += math.sin(t * 7 + ph - u * 6) * 0.35 * u * (0.4 + wind)
            d = np.array([-math.sin(ang), math.cos(ang)])
            nrm = np.array([-d[1], d[0]])
            wd = 0.02 * (1 - 0.3 * u)
            rib_a.append(pos + nrm * wd)
            rib_b.append(pos - nrm * wd)
            pos = pos + d * (ln / 12)
        polys.append(np.vstack([rib_a, rib_b[::-1]]))
    return polys


def girl_mask(x, y, s, t, pose="stand", wind=1.0, facing=1, phase=0.0, mask=None):
    m = new_mask() if mask is None else mask
    out = []
    for p in girl_shapes(t, pose, wind, phase):
        q = np.asarray(p, float).copy()
        q[:, 0] = x + facing * q[:, 0] * s
        q[:, 1] = y + q[:, 1] * s
        out.append(q)
    fill(m, out)
    return m


def rim_light(img, mask, color, dx=6, dy=-3, strength=1.0):
    m = mask.astype(np.float32) / 255.0
    sh = np.roll(np.roll(m, dy, 0), -dx, 1)
    rim = np.clip(m - sh, 0, 1)
    rim = blur(rim, 1.2)
    img += color * (rim * strength * 1.4)[..., None]
    return img


# ------------------------------------------------------------------ scene: sky pan

def sc_sky(ctx):
    p = ctx.p
    name = p.get("pal", "golden")
    tilt = p.get("tilt", 0.0)
    img = sky(ctx, name, pan=p.get("pan", 40), sun=p.get("sun", (W * 0.72, H * 0.86, 90)),
              stars=p.get("stars", 0.0), cover=p.get("cover", 0.52), yoff=tilt * ctx.lt)
    petals(img, ctx, p.get("petals", 0))
    return img


# ------------------------------------------------------------------ petals

@functools.lru_cache(maxsize=4)
def _petal_set(seed, n):
    rng = np.random.default_rng(seed)
    return dict(x=rng.random(n), y=rng.random(n), z=rng.random(n) ** 1.5 * 0.9 + 0.1, ph=rng.random(n) * 6.28,
                sp=rng.random(n) * 0.5 + 0.7, rot=rng.random(n) * 6.28)


def petals(img, ctx, n, color=None, seed=5, wind=1.0, speed=1.0):
    if n <= 0:
        return img
    S = _petal_set(seed, 260)
    color = hexc("#ffc6d9") if color is None else color
    dark = color * 0.75
    t = ctx.t * speed
    order = np.argsort(S["z"])[: n]
    for i in order:
        z = S["z"][i]
        x = (S["x"][i] * (W + 400) + t * (160 + 260 * wind) * z * S["sp"][i] + 60 * math.sin(t * 1.3 + S["ph"][i])) \
            % (W + 400) - 200
        y = (S["y"][i] * (H + 300) + t * 130 * z * S["sp"][i]) % (H + 300) - 150
        r = 14 + 34 * z
        flip = abs(math.sin(t * 2.2 * S["sp"][i] + S["ph"][i])) * 0.8 + 0.2
        rot = S["rot"][i] + t * 1.5 * S["sp"][i]
        pts = []
        for k in range(16):
            th = k / 16 * 2 * math.pi
            rr = r * (0.55 + 0.45 * math.cos(th)) * (1 - 0.25 * max(0, math.cos(th)) ** 8)
            v = np.array([rr * math.cos(th), rr * math.sin(th) * 0.55 * flip])
            pts.append(np.array([x, y]) + _rot(v, rot))
        c = mix(dark, color, flip)
        paint_poly(img, np.array(pts), c, 0.95, soft=(0.35 - z) * 18 if z < 0.35 else 0.0)
    return img


# ------------------------------------------------------------------ scene: rooftop girl

def rooftop_layer(seed=2, w=W):
    rng = np.random.default_rng(seed)
    polys = []
    x = -50
    while x < w + 50:
        bw = rng.integers(120, 340)
        bh = rng.integers(60, 230)
        polys.append(np.array([[x, H], [x, H - bh], [x + bw, H - bh], [x + bw, H]], float))
        if rng.random() < 0.4:  # water tank / antenna
            ax = x + rng.integers(20, max(21, bw - 30))
            polys.append(np.array([[ax, H - bh], [ax, H - bh - rng.integers(40, 120)], [ax + 4, H - bh - 30],
                                   [ax + 4, H - bh]], float))
        x += bw
    return polys


def sc_rooftop(ctx):
    p = ctx.p
    name = p.get("pal", "golden")
    P = pal(name)
    push = 1.0 + 0.04 * ctx.u
    img = sky(ctx, name, pan=20, sun=p.get("sun", (W * 0.66, H * 0.74, 120)), stars=p.get("stars", 0))
    # distant skyline silhouette
    sk = city_far(ctx.p.get("seed", 1))
    comp(img, sk, mix(P["sil"], P["cloud_sh"], 0.55))
    m = new_mask()
    # roof edge
    fill(m, np.array([[0, H * 0.84], [W, H * 0.80], [W, H], [0, H]], float))
    fill(m, np.array([[W * 0.08, H * 0.84], [W * 0.08, H * 0.72], [W * 0.085, H * 0.72], [W * 0.085, H * 0.84]]))
    stroke(m, np.array([[0, H * 0.765], [W, H * 0.725]]), 5)
    for k in range(18):
        xx = k * W / 17
        yy = H * (0.84 - 0.04 * k / 17)
        stroke(m, np.array([[xx, yy], [xx, yy - H * 0.075]]), 4)
    gx = W * p.get("gx", 0.6)
    girl_mask(gx, H * 0.825 - (H * 0.04 * (gx / W)), H * 0.62 * push, ctx.t, "stand",
              wind=p.get("wind", 1.0), facing=p.get("facing", 1), mask=m)
    comp(img, m, P["sil"])
    rim_light(img, m, P["rim"], dx=-7 if p.get("facing", 1) > 0 else 7, dy=-2, strength=0.9)
    petals(img, ctx, p.get("petals", 0))
    return img


@functools.lru_cache(maxsize=4)
def city_far(seed):
    rng = np.random.default_rng(seed)
    m = new_mask()
    x = 0
    while x < W:
        bw = rng.integers(40, 140)
        bh = rng.integers(40, 260) * (1.3 if rng.random() < 0.15 else 1)
        m = fill(m, np.array([[x, H], [x, H * 0.82 - bh], [x + bw, H * 0.82 - bh], [x + bw, H]], float))
        x += bw - rng.integers(0, 20)
    return m


# ------------------------------------------------------------------ scene: utility poles

def sc_poles(ctx):
    p = ctx.p
    name = p.get("pal", "golden")
    P = pal(name)
    img = sky(ctx, name, pan=12, sun=p.get("sun", (W * 0.3, H * 0.9, 80)), stars=p.get("stars", 0))
    speed = p.get("speed", 140)
    off = ctx.lt * speed
    m = new_mask()
    spacing = 820
    tops = []
    k0 = int((off - 400) // spacing)
    for k in range(k0, k0 + 5):
        x = k * spacing - off + 260
        depth = 1.0
        top = H * 0.08
        fill(m, np.array([[x - 11, H], [x - 7, top], [x + 7, top], [x + 11, H]], float))
        for j, yy in enumerate((0.16, 0.24)):
            ww = 150 - j * 30
            fill(m, np.array([[x - ww, H * yy], [x + ww, H * yy - 4], [x + ww, H * yy + 7], [x - ww, H * yy + 11]]))
            for q in (-ww + 18, -ww / 2, ww / 2, ww - 18):
                fill(m, np.array([[x + q - 5, H * yy - 2], [x + q - 4, H * yy - 18], [x + q + 4, H * yy - 18],
                                  [x + q + 5, H * yy - 2]]))
        fill(m, np.array([[x + 9, H * 0.36], [x + 70, H * 0.36], [x + 70, H * 0.43], [x + 9, H * 0.43]]))
        tops.append(x)
    for a, b in zip(tops[:-1], tops[1:]):
        for j, yy in enumerate((0.16, 0.24)):
            for q in (-130 + j * 30, -60, 60, 130 - j * 30):
                y0 = H * yy - 18
                sag = 70 + 25 * j
                xs = np.linspace(a + q, b + q, 40)
                ys = y0 + sag * (1 - ((xs - (a + b) / 2 - q) / ((b - a) / 2)) ** 2)
                stroke(m, np.stack([xs, ys], 1), 2)
    # low houses
    fill(m, rooftop_layer(9))
    comp(img, m, P["sil"])
    if p.get("birds"):
        birds(img, ctx, P["sil"])
    petals(img, ctx, p.get("petals", 0))
    return img


def birds(img, ctx, color, n=7):
    m = new_mask()
    rng = np.random.default_rng(4)
    for i in range(n):
        x = (rng.random() * W + ctx.lt * 160 * (0.7 + rng.random() * 0.6)) % (W + 200) - 100
        y = H * (0.2 + rng.random() * 0.3) + 20 * math.sin(ctx.lt * 1.5 + i)
        s = 10 + rng.random() * 12
        f = math.sin(ctx.lt * 12 + i * 2) * 0.6
        pts = np.array([[x - s, y - s * f], [x - s * 0.3, y - s * 0.2], [x, y + 2], [x + s * 0.3, y - s * 0.2],
                        [x + s, y - s * f], [x + s * 0.3, y + 1], [x, y + 4], [x - s * 0.3, y + 1]])
        fill(m, pts)
    comp(img, m, color)


# ------------------------------------------------------------------ scene: anime eye close-up

EYE_COLS = {
    "violet": ("#2a1150", "#7d4fd8", "#e9a8ff"),
    "rose": ("#4a0d2c", "#d9467e", "#ffc0d8"),
    "cyan": ("#08264a", "#2a9fe0", "#b8f7ff"),
    "gold": ("#4a2208", "#e0902a", "#fff0a8"),
    "red": ("#3a0408", "#e0203a", "#ffb0a0"),
}


def _eye_curves(o, n=48):
    x = np.linspace(-1, 1, n)
    up = -0.56 * (1 - x ** 2) ** 0.7 + 0.1 * x + 0.06
    low = 0.40 * (1 - x ** 2) ** 0.9 + 0.12 + 0.06 * x
    closed = low - 0.03 * (1 - x ** 2) - 0.02
    upo = closed + (up - closed) * o
    return x, upo, low


def sc_eye(ctx):
    p = ctx.p
    A = ctx.A
    dark, mid, light = (hexc(c) for c in EYE_COLS[p.get("iris", "violet")])
    img = vgrad([(0, hexc("#f9e0d6")), (0.6, hexc("#f4cfc4")), (1, hexc("#eab3ab"))])
    push = 1 + 0.06 * ease_out(ctx.u) + 0.015 * A.get("kick", ctx.t)
    S = W * 0.26 * push
    cx = W * 0.5 + math.sin(ctx.lt * 0.6) * 12
    cy = H * 0.50 + math.cos(ctx.lt * 0.5) * 8

    def E(xs, ys):
        return np.stack([cx + np.asarray(xs) * S, cy + np.asarray(ys) * S], 1)

    # blush with anime hatch lines
    bl = np.exp(-((radial(cx - 0.2 * S, cy + 0.95 * S, S * 0.9)) ** 2) * 2.0)
    img += (hexc("#ff8fa0") - img) * (bl * 0.35)[..., None]
    for k in range(5):
        x0 = cx - 0.55 * S + k * 0.17 * S
        paint_poly(img, np.array([[x0, cy + 0.78 * S], [x0 + 6, cy + 0.78 * S], [x0 - 0.1 * S + 6, cy + 1.0 * S],
                                  [x0 - 0.1 * S, cy + 1.0 * S]]), hexc("#ff6f86"), 0.55)
    # openness / blink
    o = 1.0
    for bt in p.get("blinks", []):
        d = abs(ctx.lt - bt)
        if d < 0.16:
            o = min(o, d / 0.16)
    if p.get("open_in"):
        o *= ease_out(ctx.lt / p["open_in"])
    o = max(o, 0.0)
    x, up, low = _eye_curves(o)
    eye_poly = np.vstack([E(x, up), E(x[::-1], low[::-1])])
    em = new_mask()
    fill(em, eye_poly)
    emf = em.astype(np.float32) / 255.0
    # sclera with lid shadow
    yy = (np.arange(H, dtype=np.float32)[:, None] - cy) / S
    sclera = hexc("#fdf8fb") + (hexc("#b9a9d6") - hexc("#fdf8fb")) * smoothstep(0.1, -0.45, yy)[..., None]
    img += (sclera - img) * emf[..., None]
    # iris
    if o > 0.05:
        ix, iy = cx + 0.04 * S, cy + 0.06 * S
        rx, ry = 0.40 * S, 0.52 * S
        r = np.sqrt(((np.arange(W, dtype=np.float32)[None, :] - ix) / rx) ** 2 + ((np.arange(H, dtype=np.float32)[:, None] - iy) / ry) ** 2)
        vy = (np.arange(H, dtype=np.float32)[:, None] - (iy - ry)) / (2 * ry)
        im_ = smoothstep(1.0, 0.985, r) * emf
        col = dark + (mid - dark) * smoothstep(0.15, 0.75, vy)[..., None]
        col = col + (light - col) * (smoothstep(0.62, 1.0, vy) * smoothstep(1.0, 0.55, r) * 0.9)[..., None]
        # radial fibres
        ang = np.arctan2(np.arange(H, dtype=np.float32)[:, None] - iy, np.arange(W, dtype=np.float32)[None, :] - ix)
        fib = (0.5 + 0.5 * np.sin(ang * 46 + np.sin(ang * 7) * 2)) * smoothstep(0.35, 0.9, r)
        col = col * (1 - 0.18 * fib[..., None]) + light * (0.10 * fib * smoothstep(0.5, 0.0, vy))[..., None]
        ring = smoothstep(0.86, 0.98, r) * (r < 1.0)
        col = col + (dark * 0.6 - col) * (ring * 0.9)[..., None]
        pul = 1 + 0.12 * A.get("kick", ctx.t)
        pr = np.sqrt(((np.arange(W, dtype=np.float32)[None, :] - ix) / (0.16 * S * pul)) ** 2 + ((np.arange(H, dtype=np.float32)[:, None] - iy + 0.04 * S) / (0.25 * S * pul)) ** 2)
        col = col + (dark * 0.35 - col) * smoothstep(1.0, 0.9, pr)[..., None]
        img += (col - img) * im_[..., None]
        # reflections: sparkles and a mirrored skyline glint
        tw = 0.6 + 0.4 * math.sin(ctx.t * 6)
        for (hx, hy, hrx, hry, a) in ((-0.17, -0.2, 0.13, 0.10, 1.0), (0.2, 0.26, 0.05, 0.045, 0.9),
                                      (0.24, -0.1, 0.025, 0.025, tw)):
            hm = new_mask()
            fill(hm, ellipse_pts(ix + hx * S, iy + hy * S, hrx * S, hry * S, -0.4))
            comp(img, hm.astype(np.float32) / 255.0 * emf, hexc("#ffffff"), a)
        for k in range(6):
            sx = ix + (-0.25 + 0.1 * k) * S
            sy = iy + 0.32 * S + 0.03 * S * math.sin(k * 3)
            s = (4 + 3 * math.sin(ctx.t * 5 + k * 1.7)) * S / 600
            paint_poly(img, np.array([[sx - s, sy], [sx, sy - s * 3], [sx + s, sy], [sx, sy + s * 3]]), light * 1.2,
                       0.8 * float(o > 0.5))
    # upper lash line (thick, heavier at outer corner) and flicks
    th = 0.03 + 0.07 * smoothstep(-0.6, 1.0, x)
    lash_top = E(x * 1.02, up - th - 0.01)
    lash_bot = E(x[::-1], up[::-1] + 0.012)
    lash = np.vstack([E([-1.05], [low[0] - 0.02]), lash_top, E([1.18, 1.32, 1.12], [up[-1] - 0.16, up[-1] - 0.12,
                                                                               up[-1] + 0.06]), lash_bot])
    paint_poly(img, lash, hexc("#1b1226"))
    for k, (fx_, fy_) in enumerate(((0.55, -0.42), (0.78, -0.33), (0.92, -0.2))):
        i = int((fx_ + 1) / 2 * (len(x) - 1))
        base = E([x[i]], [up[i] - th[i]])[0]
        tip = base + np.array([0.18 * S, (fy_ - 0.02) * S * 0.6 * (0.3 + 0.7 * o)])
        paint_poly(img, np.array([base - [0.03 * S, 0], tip, base + [0.03 * S, 0.01 * S]]), hexc("#1b1226"))
    # lower lash (outer third) and double-eyelid crease
    sel = x > 0.1
    lw = E(x[sel], low[sel])
    paint_poly(img, np.vstack([lw, (lw + [0, 0.025 * S])[::-1]]), hexc("#5a3a52"), 0.9)
    crease = E(x[6:-4] * 0.95, up[6:-4] - 0.16 - 0.04 * o)
    m = new_mask()
    stroke(m, crease, max(2, S * 0.012))
    comp(img, m, hexc("#c78a8a"), 0.8)
    # strands of hair across the frame
    hm = new_mask()
    for k in range(4):
        x0 = W * (0.03, 0.10, 0.90, 0.97)[k] + 30 * math.sin(ctx.t * 0.8 + k)
        pts_a, pts_b = [], []
        for i in range(12):
            u = i / 11
            xx = x0 + (u ** 1.5) * 160 * (1 if k < 2 else -1) + 20 * math.sin(ctx.t * 1.3 + u * 3 + k)
            yy = -20 + u * H * (0.35 + 0.18 * (k % 3))
            wd = 22 * (1 - u) + 2
            pts_a.append([xx - wd, yy])
            pts_b.append([xx + wd, yy])
        fill(hm, np.vstack([pts_a, pts_b[::-1]]))
    comp(img, hm, hexc("#2b1d3f"))
    if p.get("tear") and ctx.u > 0.4:
        ty = cy + (low[30] + 0.05) * S + (ctx.u - 0.4) * 400
        paint_poly(img, ellipse_pts(cx + 0.35 * S, ty, 12, 18), hexc("#dff4ff"), 0.85)
    return img


# ------------------------------------------------------------------ city (parallax night skyline)

@functools.lru_cache(maxsize=6)
def city_layer(seed, depth, mult=3):
    """Returns (building mask uint8, window group float [-1 = none], window colour rgb)."""
    rng = np.random.default_rng(seed + depth * 101)
    w = W * mult
    bm = np.zeros((H, w), np.uint8)
    wg = np.full((H, w), -1.0, np.float32)
    wc = np.zeros((H, w, 3), np.float32)
    base = H * (0.72 + 0.12 * depth)
    x = -20
    warm, cool = hexc("#ffd58a"), hexc("#a8d8ff")
    while x < w:
        bw = int(rng.integers(90, 260) * (1 + 0.5 * depth))
        bh = int(rng.integers(160, 520) * (1.25 - 0.3 * depth))
        if rng.random() < 0.12:
            bh = int(bh * 1.5)
        top = int(base - bh)
        cv2.rectangle(bm, (x, top), (min(w - 1, x + bw), H), 255, -1)
        if rng.random() < 0.3:  # stepped top
            cv2.rectangle(bm, (x + bw // 4, top - 40), (x + bw * 3 // 4, top), 255, -1)
        if rng.random() < 0.25:  # antenna
            cv2.rectangle(bm, (x + bw // 2 - 2, top - 120), (x + bw // 2 + 2, top), 255, -1)
        ww = int(8 + 5 * depth)
        wh = int(11 + 6 * depth)
        gx, gy = ww + int(6 + 3 * depth), wh + int(9 + 4 * depth)
        col = warm if rng.random() < 0.65 else cool
        for yy in range(top + 18, H - 10, gy):
            for xx in range(x + 10, x + bw - ww - 6, gx):
                if xx + ww >= w:
                    continue
                g = rng.random()
                wg[yy:yy + wh, xx:xx + ww] = g
                wc[yy:yy + wh, xx:xx + ww] = col * (0.75 + 0.25 * rng.random())
        x += bw + int(rng.integers(-10, 30))
    return bm, wg, wc


def _crop_wrap(a, x):
    w = a.shape[1]
    x = int(x) % w
    if x + W <= w:
        return a[:, x:x + W]
    return np.concatenate([a[:, x:], a[:, :W - (w - x)]], 1)


SIGN_TEXT = ["夜", "星", "光", "夢", "恋", "FAVE", "FIX", "♡", "愛", "空"]


@functools.lru_cache(maxsize=4)
def neon_signs(seed, mult=3):
    rng = np.random.default_rng(seed)
    out = []
    x = 100
    cols = ["#ff2fa8", "#29e6ff", "#8a5cff", "#ffe14d", "#ff5a3c", "#47ff9a"]
    while x < W * mult - 200:
        vertical = rng.random() < 0.6
        txt = SIGN_TEXT[rng.integers(0, len(SIGN_TEXT))]
        if vertical and len(txt) > 1 and txt.isascii():
            vertical = False
        out.append(dict(x=x, y=int(H * (0.35 + rng.random() * 0.3)), txt=txt, v=vertical,
                        col=hexc(cols[rng.integers(0, len(cols))]), ph=rng.random() * 6.28))
        x += int(rng.integers(220, 520))
    return out


def sc_city(ctx):
    p = ctx.p
    A = ctx.A
    name = p.get("pal", "night")
    P = pal(name)
    img = sky(ctx, name, pan=6, stars=p.get("stars", 0.8), clouds=p.get("clouds", False), cover=0.45)
    if p.get("moon"):
        moon(img, W * 0.75, H * 0.25, 120, ctx, hexc("#fff4d8"))
    speed = p.get("speed", 60)
    lit = p.get("lit", 0.7)
    if "lit_ramp" in p:
        a0, a1 = p["lit_ramp"]
        # windows switch on in beat-quantised steps
        k = math.floor((ctx.lt) / (60 / 140))
        lit = a0 + (a1 - a0) * min(1.0, k * (60 / 140) / ctx.dur)
    neon = new_mask()
    for depth in (0, 1, 2):
        bm, wg, wc = city_layer(p.get("seed", 1), depth)
        off = ctx.lt * speed * (0.35 + depth * 0.6) + depth * 777
        m = _crop_wrap(bm, off)
        g = _crop_wrap(wg, off)
        c = _crop_wrap(wc, off)
        shade = mix(P["sil"], P["sky"][-1][1] if isinstance(P["sky"], list) else P["sil"], 0.35 - depth * 0.15)
        comp(img, m, shade)
        on = ((g >= 0) & (g < lit)).astype(np.float32)
        flick = 0.85 + 0.15 * A.get("high", ctx.t)
        img += (c - img) * (on * flick * (0.65 + 0.15 * depth))[..., None]
        if depth == 2 and p.get("neon", True):
            for sgn in neon_signs(p.get("seed", 1)):
                sx = sgn["x"] - (off % (W * 3))
                if sx < -300:
                    sx += W * 3
                if not (-300 < sx < W + 300):
                    continue
                on_ = 0.15 if (math.sin(ctx.t * 9 + sgn["ph"]) > 0.9 and math.sin(ctx.t * 1.7 + sgn["ph"] * 3) > 0.6) else 1.0
                txt = sgn["txt"]
                if sgn["v"] and not txt.isascii():
                    tm = text_mask(txt, 110, FONT_JP)
                    bx = np.array([[sx - 70, sgn["y"] - 90], [sx + 70, sgn["y"] - 90], [sx + 70, sgn["y"] + 90],
                                   [sx - 70, sgn["y"] + 90]])
                else:
                    tm = text_mask(txt, 90, FONT_BLACK if txt.isascii() else FONT_JP)
                    bx = np.array([[sx - tm.shape[1] / 2 - 30, sgn["y"] - 75], [sx + tm.shape[1] / 2 + 30, sgn["y"] - 75],
                                   [sx + tm.shape[1] / 2 + 30, sgn["y"] + 75], [sx - tm.shape[1] / 2 - 30, sgn["y"] + 75]])
                paint_poly(img, bx, hexc("#0a0712"))
                mm = new_mask()
                stroke(mm, bx, 4, closed=True)
                tmf = place(tm, sx, sgn["y"])
                full = np.maximum(mm, tmf).astype(np.float32) / 255.0 * on_
                img += sgn["col"] * full[..., None] * 1.2
                neon_acc = full
                neon = np.maximum(neon, (neon_acc * 255).astype(np.uint8))
                ctx.__dict__.setdefault("_neon_cols", []).append((full, sgn["col"]))
    # glow for neon signs
    if hasattr(ctx, "_neon_cols"):
        acc = np.zeros_like(img)
        for full, col in ctx._neon_cols:
            ys, xs = np.nonzero(full > 0.05)
            if ys.size:
                acc[ys.min():ys.max() + 1, xs.min():xs.max() + 1] += col * full[ys.min():ys.max() + 1,
                                                                              xs.min():xs.max() + 1, None]
        img += fast_blur(acc, 22) * (1.6 + 0.8 * A.get("kick", ctx.t))
        del ctx._neon_cols
    if p.get("rain"):
        rain(img, ctx, p["rain"])
    if p.get("girl"):
        m = new_mask()
        girl_mask(W * 0.5, H * 1.02, H * 0.9, ctx.t, "stand", 0.6, facing=-1, mask=m)
        comp(img, m, hexc("#05040a"))
        rim_light(img, m, hexc("#ff4fb0"), dx=8, dy=0, strength=0.8)
        rim_light(img, m, hexc("#3fd8ff"), dx=-8, dy=0, strength=0.8)
    return img


def rain(img, ctx, amt, color=None, angle=0.18):
    rng = np.random.default_rng(int(ctx.t * 30) % 1000)
    color = hexc("#aeb8ff") if color is None else color
    n = int(260 * amt)
    m = np.zeros((H, W), np.float32)
    xs = rng.random(n) * (W + 200) - 100
    ys = rng.random(n) * H
    ls = 30 + rng.random(n) * 70
    for x, y, l in zip(xs, ys, ls):
        cv2.line(m, (int(x), int(y)), (int(x - l * angle), int(y + l)), 0.5 + 0.5 * (l / 100), 1, cv2.LINE_AA)
    img += color * m[..., None] * 0.45
    return img


def moon(img, x, y, r, ctx, color, shade=None, craters=True):
    rr = radial(x, y, r)
    halo = np.exp(-np.maximum(rr - 1, 0) * 2.2) * (rr > 0.98)
    img += color * (halo * 0.35)[..., None]
    disc = smoothstep(1.0, 0.985, rr)
    tex = moon_tex()
    s = int(r * 2)
    if s > 4 and craters:
        t2 = cv2.resize(tex, (s, s))
        full = np.zeros((H, W), np.float32)
        x0, y0 = int(x - r), int(y - r)
        xa, ya = max(0, x0), max(0, y0)
        xb, yb = min(W, x0 + s), min(H, y0 + s)
        if xb > xa and yb > ya:
            full[ya:yb, xa:xb] = t2[ya - y0:yb - y0, xa - x0:xb - x0]
        c = color[None, None, :] * (0.82 + 0.18 * full[..., None])
    else:
        c = color
    img += (c - img) * disc[..., None]
    if shade is not None:  # eclipse / crescent shadow
        sx, sy, scol = shade
        sd = smoothstep(1.0, 0.97, radial(x + sx * r, y + sy * r, r * 1.0)) * disc
        img += (scol - img) * sd[..., None]
    return img


@functools.lru_cache(maxsize=1)
def moon_tex():
    n = fbm(512, 512, 77, base=3, octaves=5)
    rng = np.random.default_rng(3)
    t = n.copy()
    for _ in range(28):
        cx, cy, rr = rng.random() * 512, rng.random() * 512, 8 + rng.random() ** 2 * 60
        d = radial(cx, cy, rr, 512, 512)
        t -= 0.25 * smoothstep(1.0, 0.7, d) - 0.15 * smoothstep(1.15, 1.0, d) * smoothstep(0.9, 1.0, d)
    t = (t - t.min()) / (t.max() - t.min())
    return np.round(t * 3) / 3  # cel-shaded steps


# ------------------------------------------------------------------ scene: big moon + girl

def sc_moon(ctx):
    p = ctx.p
    name = p.get("pal", "night")
    P = pal(name)
    img = sky(ctx, name, pan=10, stars=1.0, clouds=p.get("clouds", True), cover=0.42, top=0.35, bottom=0.95)
    r = H * p.get("r", 0.42) * (1 + 0.05 * ctx.u)
    mx, my = W * p.get("mx", 0.5), H * p.get("my", 0.48) - 30 * ctx.u
    mcol = hexc(p.get("mcol", "#fff1d6"))
    shade = None
    if p.get("eclipse"):
        e = p["eclipse"]
        shade = (0.25 + (1 - ctx.u) * e, -0.1, hexc("#120208"))
    moon(img, mx, my, r, ctx, mcol, shade=shade)
    m = new_mask()
    if p.get("pose", "run") == "run":
        gx = W * (-0.15 + 1.3 * ctx.u) if p.get("cross") else W * 0.5
        girl_mask(gx, H * 0.86, H * 0.55, ctx.t, "run", 1.2, mask=m)
        fill(m, np.array([[0, H * 0.86], [W, H * 0.86], [W, H], [0, H]], float))
    else:
        fill(m, catmull([[-100, H], [-100, H * 0.85], [W * 0.3, H * 0.80], [W * 0.6, H * 0.82], [W + 100, H * 0.78],
                         [W + 100, H]], 12))
        girl_mask(W * 0.47, H * 0.815, H * 0.42, ctx.t, "stand", 0.9, mask=m)
    comp(img, m, P["sil"])
    petals(img, ctx, p.get("petals", 0), color=hexc(p.get("petal_col", "#ffc6d9")))
    if p.get("lines"):
        speed_lines(img, ctx, p["lines"], color=hexc("#ffffff"))
    return img


# ------------------------------------------------------------------ manga speed / focus lines

def speed_lines(img, ctx, amt, color=None, cx=W / 2, cy=H / 2, inner=0.35, seed_rate=24):
    color = hexc("#ffffff") if color is None else color
    rng = np.random.default_rng(int(ctx.t * seed_rate))
    m = new_mask()
    R = math.hypot(W, H)
    polys = []
    for i in range(int(90 * amt)):
        a = rng.random() * 2 * math.pi
        w = rng.random() * 0.018 + 0.004
        r0 = (inner + rng.random() * 0.25) * H
        polys.append(np.array([[cx + math.cos(a) * r0, cy + math.sin(a) * r0],
                               [cx + math.cos(a - w) * R, cy + math.sin(a - w) * R],
                               [cx + math.cos(a + w) * R, cy + math.sin(a + w) * R]]))
    fill(m, polys)
    comp(img, m, color, 0.9)
    return img


def sc_focus(ctx):
    p = ctx.p
    A = ctx.A
    bg = hexc(p.get("bg", "#ff2f7a"))
    fg = hexc(p.get("fg", "#ffffff"))
    img = np.empty((H, W, 3), np.float32)
    img[:] = bg
    speed_lines(img, ctx, 1.4, color=fg, inner=0.28)
    txt = p.get("text", "FAVE")
    k = A.get("kick", ctx.t)
    sc = 1.0 + 0.08 * k + 0.05 * ctx.u
    font = FONT_JP if not txt.isascii() else FONT_BLACK_IT
    size = p.get("size", 300)
    draw_text(img, txt, W / 2 + 10, H / 2 + 12, size, hexc("#120a1a"), font=font, scale=sc)
    draw_text(img, txt, W / 2, H / 2, size, fg if p.get("solid") else bg, font=font, scale=sc,
              outline=(10, hexc("#120a1a")))
    return img


# ------------------------------------------------------------------ warp tunnel

@functools.lru_cache(maxsize=4)
def _warp_stars(seed, n=900):
    rng = np.random.default_rng(seed)
    a = rng.random(n) * 2 * math.pi
    r = 0.15 + rng.random(n) ** 0.6 * 3.0
    return np.cos(a) * r, np.sin(a) * r, rng.random(n), rng.integers(0, 4, n)


def sc_warp(ctx):
    p = ctx.p
    A = ctx.A
    cols = [hexc(c) for c in p.get("cols", ["#ff2fa8", "#29e6ff", "#8a5cff", "#ffffff"])]
    img = vgrad([(0, hexc(p.get("bg0", "#05020f"))), (1, hexc(p.get("bg1", "#1a0a33")))])
    cx = W / 2 + 60 * math.sin(ctx.lt * 0.7)
    cy = H / 2 + 40 * math.cos(ctx.lt * 0.9)
    core = np.exp(-radial(cx, cy, H * 0.35) ** 2)
    img += cols[0] * (core * 0.35)[..., None]
    X, Y, Z0, C = _warp_stars(p.get("seed", 1))
    speed = p.get("speed", 1.0)
    # integrate distance travelled using energy so kicks surge forward
    dist = ctx.lt * 0.9 * speed + 0.25 * A.get("energy", ctx.t) * ctx.lt
    surge = 0.35 * A.get("kick", ctx.t)
    z = (Z0 - dist) % 1.0 + 0.02
    z_prev = z + 0.035 * speed * (1 + 2.5 * surge)
    f = H * 0.5
    lay = np.zeros((H, W, 3), np.float32)
    for ci, col in enumerate(cols):
        sel = C == ci
        x1, y1 = cx + X[sel] / z[sel] * f * 0.25, cy + Y[sel] / z[sel] * f * 0.25
        x0, y0 = cx + X[sel] / z_prev[sel] * f * 0.25, cy + Y[sel] / z_prev[sel] * f * 0.25
        br = np.clip(1.2 - z[sel], 0, 1)
        m = np.zeros((H, W), np.float32)
        for a, b, c, d, e in zip(x0, y0, x1, y1, br):
            if -400 < c < W + 400 and -400 < d < H + 400:
                cv2.line(m, (int(a), int(b)), (int(c), int(d)), float(e), 2 if e > 0.6 else 1, cv2.LINE_AA)
        lay += m[..., None] * col
    img += lay + fast_blur(lay, 10) * 1.5
    # hoops
    if p.get("rings", True):
        for k in range(6):
            zz = ((k / 6) - dist * 0.5) % 1.0 + 0.05
            rr = H * 0.12 / zz
            if rr > W * 1.2:
                continue
            m = new_mask()
            th = max(2, 6 / zz)
            cv2.circle(m, (int(cx), int(cy)), int(rr), 255, int(th), cv2.LINE_AA)
            add(img, m, cols[k % len(cols)], min(1.0, 0.5 + 0.8 * A.get("kick", ctx.t)) * min(1, zz * 2))
    if p.get("girl"):
        m = new_mask()
        girl_mask(W * 0.5, H * 0.92, H * 0.62, ctx.t, "run", 1.3, mask=m)
        comp(img, m, hexc("#05030a"))
        rim_light(img, m, cols[0], dx=7, dy=0, strength=1.0)
        rim_light(img, m, cols[1], dx=-7, dy=0, strength=1.0)
    return img


# ------------------------------------------------------------------ title / typography

def sc_title(ctx):
    p = ctx.p
    A = ctx.A
    name = p.get("pal", "night")
    img = sky(ctx, name, pan=4, stars=1.2, clouds=False)
    if p.get("moon"):
        moon(img, W * 0.5, H * 0.46, H * 0.3, ctx, hexc("#fff3dc"))
        img *= 0.75
    a = ease_out((ctx.lt - p.get("delay", 0.0)) / 1.2)
    if a > 0:
        rng = np.random.default_rng(int(ctx.t * 30))
        jit = (1 - a) * 40
        tr = 40 + 30 * (1 - a)
        y = H * p.get("y", 0.47)
        draw_text(img, "FAVE FIX", W / 2 + 8 + rng.normal(0, jit) - 6, y, 230, hexc("#29e6ff"), tracking=tr, alpha=a * 0.8)
        draw_text(img, "FAVE FIX", W / 2 - 8 + rng.normal(0, jit) + 6, y, 230, hexc("#ff2fa8"), tracking=tr, alpha=a * 0.8)
        draw_text(img, "FAVE FIX", W / 2, y, 230, hexc("#fff8f0"), tracking=tr, alpha=a)
        b = ease_out((ctx.lt - p.get("delay", 0.0) - 0.6) / 1.0)
        if b > 0:
            draw_text(img, "フェイヴ・フィックス", W / 2, y + 175, 52, hexc("#fff8f0"), font=FONT_JP, tracking=18,
                      alpha=b * 0.9)
            m = new_mask()
            ww = 520 * b
            stroke(m, np.array([[W / 2 - ww, y + 120], [W / 2 + ww, y + 120]]), 2)
            comp(img, m, hexc("#fff8f0"), 0.7)
    return img


def sc_kinetic(ctx):
    """Beat-driven big words on flat colour fields; words switch every beat."""
    p = ctx.p
    A = ctx.A
    words = p.get("words", ["FAVE", "FIX", "FAVE", "FIX"])
    cols = p.get("cols", [("#0d0820", "#ff2fa8"), ("#ff2fa8", "#0d0820"), ("#29e6ff", "#0d0820"),
                          ("#0d0820", "#29e6ff")])
    k, dt = A.beat_phase(ctx.t)
    i = k % len(words)
    bg, fg = (hexc(c) for c in cols[k % len(cols)])
    img = np.empty((H, W, 3), np.float32)
    img[:] = bg
    # background stripes sliding
    m = new_mask()
    off = (ctx.lt * 600) % 240
    polys = [np.array([[x, 0], [x + 80, 0], [x + 80 - 300, H], [x - 300, H]]) for x in np.arange(-240, W + 600, 240) + off]
    fill(m, polys)
    comp(img, m, fg, 0.08)
    pop = ease_out(dt / 0.12)
    sc = 0.8 + 0.25 * pop + 0.04 * dt
    word = words[i]
    font = FONT_BLACK_IT if word.isascii() else FONT_JP
    size = 380 if len(word) <= 4 else 240
    draw_text(img, word, W / 2 + 16, H / 2 + 16, size, mix(bg, fg, 0.35), font=font, scale=sc)
    draw_text(img, word, W / 2, H / 2, size, fg, font=font, scale=sc)
    return img


# ------------------------------------------------------------------ train passing

def sc_train(ctx):
    p = ctx.p
    A = ctx.A
    name = p.get("pal", "golden")
    P = pal(name)
    img = sky(ctx, name, pan=8, sun=(W * 0.8, H * 0.7, 110))
    comp(img, city_far(3), mix(P["sil"], P["cloud_sh"], 0.6))
    m = new_mask()
    # crossing signal
    sx = W * 0.18
    fill(m, np.array([[sx - 8, H], [sx - 8, H * 0.3], [sx + 8, H * 0.3], [sx + 8, H]], float))
    fill(m, np.array([[sx - 120, H * 0.36], [sx + 120, H * 0.36], [sx + 120, H * 0.39], [sx - 120, H * 0.39]], float))
    fill(m, np.array([[sx - 60, H * 0.27], [sx + 60, H * 0.21], [sx + 66, H * 0.235], [sx - 54, H * 0.295]], float))
    fill(m, np.array([[sx - 60, H * 0.21], [sx + 60, H * 0.27], [sx + 54, H * 0.295], [sx - 66, H * 0.235]], float))
    circle(m, sx - 85, H * 0.42, 30)
    circle(m, sx + 85, H * 0.42, 30)
    # ground + girl on the far side
    fill(m, np.array([[0, H * 0.86], [W, H * 0.86], [W, H], [0, H]], float))
    girl_mask(W * 0.62, H * 0.87, H * 0.5, ctx.t, "stand", 0.8, facing=-1, mask=m)
    comp(img, m, P["sil"])
    k, _ = A.beat_phase(ctx.t)
    lamp = (sx - 85, sx + 85)[k % 2]
    lm = new_mask()
    circle(lm, lamp, H * 0.42, 22)
    add(img, lm, hexc("#ff3030"), 1.2)
    img += hexc("#ff2020") * (np.exp(-radial(lamp, H * 0.42, 70) ** 2) * 0.8)[..., None]
    # the train: present between u0 and u1, racing right-to-left
    u0, u1 = p.get("window", (0.18, 0.82))
    if u0 < ctx.u < u1:
        tl = np.zeros((H, W), np.float32)
        win = np.zeros((H, W), np.float32)
        car, gap = 1500, 90
        off = (ctx.lt * 2600) % (car + gap)
        top, bot = H * 0.44, H * 0.88
        stripe = np.zeros((H, W), np.float32)
        for j in range(-1, 3):
            x0 = j * (car + gap) - off
            cv2.rectangle(tl, (int(x0), int(top)), (int(x0 + car), int(bot)), 1.0, -1)
            for q in range(6):
                wx = x0 + 110 + q * 225
                cv2.rectangle(win, (int(wx), int(top + 50)), (int(wx + 150), int(top + 170)), 1.0, -1)
            cv2.rectangle(stripe, (int(x0), int(top + 205)), (int(x0 + car), int(top + 235)), 1.0, -1)
        tl = cv2.blur(tl, (61, 1))
        win = cv2.blur(win, (91, 1))
        stripe = cv2.blur(stripe, (61, 1))
        body = mix(hexc("#3a3550"), P["sil"], 0.45)
        img += (body - img) * tl[..., None]
        img += (hexc("#7fd8ff") - img) * (stripe * tl * 0.8)[..., None]
        img += (hexc("#ffe2b0") - img) * (win * tl * 0.85)[..., None]
        # sun glint sweeping along the roof line
        gl = np.exp(-((np.arange(H, dtype=np.float32)[:, None] - top) / 6.0) ** 2) * tl
        img += P["rim"] * gl[..., None] * 0.9
    return img


# ------------------------------------------------------------------ rain on window / bokeh

@functools.lru_cache(maxsize=2)
def _bokeh(seed):
    rng = np.random.default_rng(seed)
    acc = np.zeros((H, W, 3), np.float32)
    cols = [hexc(c) for c in ("#ff5fa8", "#ffcf6b", "#6bd6ff", "#a07bff", "#ff8a5c")]
    for _ in range(70):
        m = np.zeros((H, W), np.float32)
        r = int(20 + rng.random() * 70)
        cv2.circle(m, (int(rng.random() * W), int(H * (0.25 + rng.random() * 0.7))), r, 1.0, -1, cv2.LINE_AA)
        acc += m[..., None] * cols[rng.integers(0, len(cols))] * (0.15 + rng.random() * 0.25)
    return blur(acc, 4)


@functools.lru_cache(maxsize=2)
def _drops(seed, n=170):
    rng = np.random.default_rng(seed)
    return rng.random(n) * W, rng.random(n) * H, 4 + rng.random(n) ** 2 * 18, rng.random(n)


def sc_window(ctx):
    p = ctx.p
    A = ctx.A
    img = vgrad([(0, hexc("#070818")), (1, hexc("#1a1238"))])
    bk = _bokeh(p.get("seed", 2))
    shift = int(ctx.lt * 12)
    img += np.roll(bk, shift, axis=1) * (0.9 + 0.3 * A.get("rms", ctx.t))
    if p.get("girl"):
        m = new_mask()
        girl_mask(W * 0.3, H * 1.1, H * 1.05, ctx.t, "stand", 0.1, facing=1, mask=m)
        comp(img, m, hexc("#0a0814"), 0.85)
        rim_light(img, m, hexc("#ff7ab8"), dx=-6, dy=0, strength=0.7)
    xs, ys, rs, sp = _drops(p.get("seed", 2))
    for x, y, r, s in zip(xs, ys, rs, sp):
        yy = (y + ctx.lt * (8 + 60 * s ** 6)) % (H + 40) - 20
        c = cv2.getRectSubPix(img, (3, 3), (float(np.clip(x, 2, W - 3)), float(np.clip(yy + r * 2, 2, H - 3)))).mean((0, 1))
        paint_poly(img, ellipse_pts(x, yy, r * 0.85, r), c * 1.6 + 0.05, 0.85)
        paint_poly(img, ellipse_pts(x - r * 0.3, yy - r * 0.35, r * 0.22, r * 0.18), hexc("#ffffff"), 0.6)
    return img


# ------------------------------------------------------------------ streetlight in rain

def sc_streetlight(ctx):
    p = ctx.p
    A = ctx.A
    img = vgrad([(0, hexc("#04040c")), (0.7, hexc("#0e0c22")), (1, hexc("#1a1530"))])
    lx, ly = W * 0.56, H * 0.16
    cone = np.array([[lx - 30, ly + 30], [lx + 30, ly + 30], [lx + 420, H * 0.93], [lx - 420, H * 0.93]])
    cm = new_mask()
    fill(cm, cone)
    cmf = fast_blur(cm.astype(np.float32) / 255.0, 30)
    warm = hexc(p.get("light", "#ffd9a0"))
    flick = 0.85 + 0.15 * math.sin(ctx.t * 40) * (A.get("high", ctx.t) > 0.8)
    img += warm * (cmf * 0.30 * flick)[..., None]
    pool = np.exp(-((np.arange(W)[None, :] - lx) / 420.0) ** 2 - ((np.arange(H)[:, None] - H * 0.93) / 40.0) ** 2)
    img += warm * (pool * 0.45)[..., None].astype(np.float32)
    m = new_mask()
    fill(m, np.array([[lx - 220, H], [lx - 220, ly - 20], [lx - 205, ly - 40], [lx - 20, ly - 40], [lx + 10, ly],
                      [lx - 10, ly + 10], [lx - 30, ly - 22], [lx - 200, ly - 22], [lx - 205, H]], float))
    fill(m, np.array([[0, H * 0.93], [W, H * 0.93], [W, H], [0, H]], float))
    girl_mask(lx + 20, H * 0.935, H * 0.5, ctx.t, "stand", 0.25, facing=-1, mask=m)
    comp(img, m, hexc("#06050c"))
    rim_light(img, m, warm, dx=0, dy=6, strength=0.9)
    bulb = np.exp(-radial(lx, ly + 20, 40) ** 2)
    img += warm * (bulb * 1.5)[..., None]
    # rain lit inside the cone
    rl = np.zeros((H, W, 3), np.float32)
    rain(rl, ctx, 2.0, color=hexc("#ffffff"))
    img += rl * (0.15 + cmf[..., None] * 1.6)
    return img


# ------------------------------------------------------------------ bass section: neon hexagon tunnel

def sc_hextunnel(ctx):
    p = ctx.p
    A = ctx.A
    cols = [hexc(c) for c in p.get("cols", ["#ff2448", "#ff9a3c", "#7a1cff"])]
    img = vgrad([(0, hexc("#060008")), (1, hexc("#1a0310"))])
    cx, cy = W / 2, H / 2
    rot = ctx.lt * 0.25 + 0.15 * math.sin(ctx.lt * 0.9)
    dist = ctx.lt * 0.7 + 0.3 * A.get("energy", ctx.t) * ctx.lt
    lay = np.zeros((H, W), np.float32)
    lay2 = np.zeros((H, W, 3), np.float32)
    kk = A.get("kick", ctx.t)
    for k in range(14):
        z = ((k / 14) - dist * 0.35) % 1.0 + 0.03
        rr = H * 0.09 / z
        if rr > W * 1.5:
            continue
        pts = np.array([[cx + rr * math.cos(rot + z * 1.5 + j * math.pi / 3), cy + rr * math.sin(rot + z * 1.5 + j * math.pi / 3)]
                        for j in range(6)])
        m = new_mask()
        stroke(m, pts, max(2, 5 / z * 0.4), closed=True)
        a = min(1, z * 3) * (0.5 + 0.9 * kk)
        lay2 += (m.astype(np.float32) / 255.0 * a)[..., None] * cols[k % len(cols)]
    img += lay2 + fast_blur(lay2, 16) * 2.0
    if p.get("eclipse"):
        r = H * 0.18 * (1 + 0.08 * kk)
        moon(img, cx, cy, r, ctx, hexc("#ff3050"), shade=(0.0, 0.0, hexc("#050005")), craters=False)
        ring = np.exp(-((radial(cx, cy, r) - 1.0) * 14) ** 2)
        img += hexc("#ffb0a0") * (ring * (0.8 + kk))[..., None]
    if p.get("girl"):
        m = new_mask()
        girl_mask(W * 0.5, H * 1.0, H * 0.75, ctx.t, "stand", 1.6, facing=1, mask=m)
        comp(img, m, hexc("#040006"))
        rim_light(img, m, cols[0], dx=-8, dy=0, strength=1.2)
    return img


# ------------------------------------------------------------------ kaleidoscope of petals & light

@functools.lru_cache(maxsize=2)
def _kal_maps(seg):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx, dy = xx - W / 2, yy - H / 2
    r = np.sqrt(dx ** 2 + dy ** 2)
    a = np.arctan2(dy, dx)
    wedge = 2 * math.pi / seg
    a = np.abs(((a % wedge) + wedge) % wedge - wedge / 2)
    return r, a


def sc_kaleido(ctx):
    p = ctx.p
    A = ctx.A
    seg = p.get("seg", 10)
    src_ctx = type(ctx)()
    src_ctx.__dict__.update(ctx.__dict__)
    src_ctx.p = dict(pal="night", lit=0.8, speed=300, seed=p.get("seed", 4), stars=0)
    src = sc_city(src_ctx)
    petals(src, src_ctx, 50, color=hexc("#ff9ad0"))
    r, a = _kal_maps(seg)
    rot = ctx.lt * 0.6 + 0.3 * A.get("kick", ctx.t)
    zoomf = 1.0 - 0.15 * A.get("kick", ctx.t)
    ox, oy = W * 0.5 + 300 * math.sin(ctx.lt * 0.4), H * 0.62
    mx = (ox + r * zoomf * np.cos(a + rot)).astype(np.float32)
    my = (oy + r * zoomf * np.sin(a + rot) * 0.6).astype(np.float32)
    out = cv2.remap(src, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out


def sc_black(ctx):
    p = ctx.p
    img = np.zeros((H, W, 3), np.float32)
    txt = p.get("text")
    if txt:
        rng = np.random.default_rng(int(ctx.t * 30))
        a = 1.0 if rng.random() > 0.15 else 0.3
        draw_text(img, txt, W / 2 + rng.normal(0, 3), H / 2, p.get("size", 90), hexc("#f6f0ff"), tracking=30, alpha=a,
                  font=FONT_BLACK if txt.isascii() else FONT_JP)
    return img


def sc_freeze(ctx):
    """Manga-screentone freeze frame of another scene."""
    p = ctx.p
    src_ctx = type(ctx)()
    src_ctx.__dict__.update(ctx.__dict__)
    src_ctx.lt = p.get("at", 1.0)
    src_ctx.t = ctx.t - ctx.lt + src_ctx.lt
    src_ctx.u = 0.5
    src_ctx.p = p.get("src_p", {})
    src = SCENES[p["src"]](src_ctx)
    img = halftone(np.clip(src * 1.9, 0, 1) ** 0.8, cell=9, ink=hexc(p.get("ink", "#160c24")), paper=hexc(p.get("paper", "#f6eee4")))
    if p.get("text"):
        draw_text(img, p["text"], W * 0.5, H * 0.82, 120, hexc("#160c24"), font=FONT_BLACK_IT,
                  alpha=ease_out(ctx.lt / 0.3))
    return img


SCENES = {k[3:]: v for k, v in list(globals().items()) if k.startswith("sc_")}
