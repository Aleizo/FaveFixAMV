"""Drawing and post-processing primitives. Images are float32 HxWx3 in [0, 1]."""
import functools
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
SS = 4  # sub-pixel bits for cv2 drawing
FONT_BLACK = "/usr/share/fonts/opentype/inter/Inter-Black.otf"
FONT_BLACK_IT = "/usr/share/fonts/opentype/inter/Inter-BlackItalic.otf"
FONT_LIGHT = "/usr/share/fonts/opentype/inter/Inter-ExtraLight.otf"
FONT_JP = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"


# ---------------------------------------------------------------- colour utils

def hexc(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)


def mix(a, b, t):
    return a + (b - a) * t


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def ease_out(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def ease_in_out(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def vgrad(stops, h=H, w=W):
    """Vertical gradient from [(pos, colour), ...]."""
    ys = np.linspace(0, 1, h)
    col = np.zeros((h, 3), np.float32)
    pos = [s[0] for s in stops]
    for c in range(3):
        col[:, c] = np.interp(ys, pos, [s[1][c] for s in stops])
    return np.repeat(col[:, None, :], w, axis=1)


def radial(cx, cy, r, h=H, w=W):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r


# ---------------------------------------------------------------- geometry

def catmull(pts, n=10, closed=True):
    P = np.asarray(pts, np.float64)
    if closed:
        P = np.vstack([P[-1], P, P[0], P[1]])
    else:
        P = np.vstack([P[0], P, P[-1]])
    t = np.linspace(0, 1, n, endpoint=False)[:, None]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    if not closed:
        out.append(P[-2][None])
    return np.vstack(out)


def capsule(a, b, r0, r1=None, n=8):
    """Polygon for a tapered limb from a (radius r0) to b (radius r1)."""
    r1 = r0 if r1 is None else r1
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    ang = math.atan2(d[1], d[0])
    pts = []
    for k in range(n + 1):
        th = ang + math.pi / 2 + math.pi * k / n
        pts.append(b + r1 * np.array([math.cos(th), math.sin(th)]) * -1)
    for k in range(n + 1):
        th = ang + math.pi / 2 + math.pi * k / n
        pts.append(a + r0 * np.array([math.cos(th), math.sin(th)]))
    return np.array(pts)


def ellipse_pts(cx, cy, rx, ry, rot=0.0, n=64):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x, y = rx * np.cos(th), ry * np.sin(th)
    c, s = math.cos(rot), math.sin(rot)
    return np.stack([cx + x * c - y * s, cy + x * s + y * c], 1)


def to_cv(pts):
    return np.round(np.asarray(pts) * (1 << SS)).astype(np.int32)


# ---------------------------------------------------------------- masks / compositing

def new_mask(h=H, w=W):
    return np.zeros((h, w), np.uint8)


def fill(mask, polys, val=255):
    if isinstance(polys, np.ndarray) and polys.ndim == 2:
        polys = [polys]
    # one call per polygon: fillPoly uses even-odd filling, so overlaps would cancel out
    for p in polys:
        cv2.fillPoly(mask, [to_cv(p)], int(val), cv2.LINE_AA, shift=SS)
    return mask


def paint_poly(img, pts, color, alpha=1.0, soft=0.0, additive=False):
    """Fill one small polygon onto img using a local mask (much cheaper than a full-frame mask)."""
    pts = np.asarray(pts, np.float64)
    pad = int(3 * soft) + 3
    x0 = int(max(0, np.floor(pts[:, 0].min()) - pad))
    y0 = int(max(0, np.floor(pts[:, 1].min()) - pad))
    x1 = int(min(img.shape[1], np.ceil(pts[:, 0].max()) + pad))
    y1 = int(min(img.shape[0], np.ceil(pts[:, 1].max()) + pad))
    if x1 <= x0 or y1 <= y0:
        return img
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(m, [to_cv(pts - [x0, y0])], 255, cv2.LINE_AA, shift=SS)
    mf = m.astype(np.float32) / 255.0
    if soft > 0.3:
        mf = cv2.GaussianBlur(mf, (0, 0), soft)
    sub = img[y0:y1, x0:x1]
    if additive:
        sub += np.asarray(color, np.float32) * (mf * alpha)[..., None]
    else:
        sub += (np.asarray(color, np.float32) - sub) * (mf * alpha)[..., None]
    return img


def stroke(mask, pts, thick, val=255, closed=False):
    cv2.polylines(mask, [to_cv(pts)], closed, int(val), max(1, int(round(thick))), cv2.LINE_AA, shift=SS)
    return mask


def circle(mask, cx, cy, r, val=255):
    cv2.circle(mask, (int(round(cx * 16)), int(round(cy * 16))), max(1, int(round(r * 16))), int(val), -1,
               cv2.LINE_AA, shift=4)
    return mask


def comp(img, mask, color, alpha=1.0):
    """Paint `color` (rgb or HxWx3 array) through mask (uint8 or float) onto img in place."""
    m = mask.astype(np.float32) / 255.0 if mask.dtype == np.uint8 else mask
    if alpha != 1.0:
        m = m * alpha
    ys, xs = _bbox(m)
    if ys is None:
        return img
    sub = img[ys, xs]
    mm = m[ys, xs][..., None]
    c = color[ys, xs] if (isinstance(color, np.ndarray) and color.ndim == 3) else np.asarray(color, np.float32)
    sub += (c - sub) * mm
    return img


def add(img, mask, color, alpha=1.0):
    m = mask.astype(np.float32) / 255.0 if mask.dtype == np.uint8 else mask
    ys, xs = _bbox(m)
    if ys is None:
        return img
    c = color[ys, xs] if (isinstance(color, np.ndarray) and color.ndim == 3) else np.asarray(color, np.float32)
    img[ys, xs] += c * (m[ys, xs][..., None] * alpha)
    return img


def _bbox(m):
    rows = np.flatnonzero(m.max(1) > 0.002)
    if rows.size == 0:
        return None, None
    cols = np.flatnonzero(m[rows[0]:rows[-1] + 1].max(0) > 0.002)
    return slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)


def blur(a, sigma):
    if sigma <= 0.3:
        return a
    return cv2.GaussianBlur(a, (0, 0), sigma)


def fast_blur(a, sigma):
    """Large-radius blur via downsampling."""
    f = max(1, int(sigma // 6))
    if f == 1:
        return blur(a, sigma)
    h, w = a.shape[:2]
    s = cv2.resize(a, (w // f, h // f), interpolation=cv2.INTER_AREA)
    s = cv2.GaussianBlur(s, (0, 0), sigma / f)
    return cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)


def glow(img, sigma=18, strength=0.6, thresh=0.6):
    bright = np.clip(img - thresh, 0, None)
    g = fast_blur(bright, sigma) * strength + fast_blur(bright, sigma * 3) * strength * 0.6
    return img + g


# ---------------------------------------------------------------- noise

def fbm(h, w, seed, base=8, octaves=5, persistence=0.5, wrap_x=False):
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        gh = max(2, int(base * 2 ** o * h / max(h, w))) + 1
        gw = max(2, int(base * 2 ** o * w / max(h, w))) + 1
        g = rng.random((gh, gw)).astype(np.float32)
        if wrap_x:
            g[:, -1] = g[:, 0]
        out += cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC) * amp
        tot += amp
        amp *= persistence
    out /= tot
    return (out - out.min()) / (out.max() - out.min() + 1e-6)


# ---------------------------------------------------------------- text

@functools.lru_cache(maxsize=256)
def text_mask(text, size, font=FONT_BLACK, tracking=0, stroke_w=0):
    f = ImageFont.truetype(font, size)
    chars = list(text)
    widths = [f.getbbox(c)[2] - f.getbbox(c)[0] if c != " " else size * 0.3 for c in chars]
    asc, desc = f.getmetrics()
    tw = int(sum(widths) + tracking * (len(chars) - 1) + stroke_w * 2 + size * 0.2)
    th = int(asc + desc + stroke_w * 2 + size * 0.1)
    im = Image.new("L", (tw, th), 0)
    d = ImageDraw.Draw(im)
    x = stroke_w + size * 0.1
    for c, wdt in zip(chars, widths):
        if c != " ":
            d.text((x - f.getbbox(c)[0], stroke_w), c, font=f, fill=255, stroke_width=stroke_w, stroke_fill=255)
        x += wdt + tracking
    a = np.array(im)
    ys = np.flatnonzero(a.max(1))
    xs = np.flatnonzero(a.max(0))
    return a[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1].copy()


def place(mask_small, cx, cy, scale=1.0, angle=0.0, skew=0.0, h=H, w=W):
    """Warp a small mask into a full-frame mask centred at (cx, cy)."""
    mh, mw = mask_small.shape
    M = cv2.getRotationMatrix2D((mw / 2, mh / 2), angle, scale)
    M[0, 1] += skew * scale
    M[0, 2] += cx - mw / 2 - skew * scale * mh / 2
    M[1, 2] += cy - mh / 2
    return cv2.warpAffine(mask_small, M, (w, h), flags=cv2.INTER_LINEAR)


def draw_text(img, text, cx, cy, size, color, font=FONT_BLACK, scale=1.0, angle=0.0, alpha=1.0,
              tracking=0, skew=0.0, shadow=None, outline=None):
    m = text_mask(text, size, font, tracking)
    if outline is not None:
        mo = text_mask(text, size, font, tracking, stroke_w=outline[0])
        comp(img, place(mo, cx, cy, scale, angle, skew), outline[1], alpha)
    if shadow is not None:
        dx, dy, sc = shadow
        comp(img, place(m, cx + dx, cy + dy, scale, angle, skew), sc, alpha)
    full = place(m, cx, cy, scale, angle, skew)
    comp(img, full, color, alpha)
    return full


# ---------------------------------------------------------------- post

@functools.lru_cache(maxsize=1)
def _vignette():
    r = radial(W / 2, H / 2, math.hypot(W / 2, H / 2))
    return (1 - 0.55 * smoothstep(0.45, 1.05, r))[..., None].astype(np.float32)


@functools.lru_cache(maxsize=1)
def _grain_bank():
    rng = np.random.default_rng(7)
    bank = []
    for _ in range(12):
        g = rng.normal(0, 1, (H // 2, W // 2)).astype(np.float32)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)
        bank.append(g[..., None])
    return bank


def vignette(img, amt=1.0):
    v = _vignette()
    return img * (1 - (1 - v) * amt)


def grain(img, frame, amt=0.035):
    return img + _grain_bank()[frame % 12] * amt


def zoom(img, s, cx=W / 2, cy=H / 2, angle=0.0, dx=0.0, dy=0.0):
    if abs(s - 1) < 1e-4 and abs(angle) < 1e-4 and abs(dx) < 0.01 and abs(dy) < 0.01:
        return img
    M = cv2.getRotationMatrix2D((cx, cy), angle, s)
    M[0, 2] += dx
    M[1, 2] += dy
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def chroma(img, amt):
    """Radial chromatic aberration; amt is pixel offset at the frame edge."""
    if amt < 0.3:
        return img
    out = img.copy()
    for ch, s in ((0, 1 + amt / W * 2), (2, 1 - amt / W * 2)):
        M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, s)
        out[..., ch] = cv2.warpAffine(img[..., ch], M, (W, H), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT)
    return out


def glitch(img, rng, amt):
    """Horizontal slice displacement + channel split."""
    if amt <= 0.01:
        return img
    out = img.copy()
    n = int(4 + amt * 14)
    for _ in range(n):
        y0 = rng.integers(0, H - 10)
        hh = int(rng.integers(6, 30 + int(amt * 120)))
        sh = int(rng.normal(0, 60 * amt))
        out[y0:y0 + hh] = np.roll(img[y0:y0 + hh], sh, axis=1)
        if rng.random() < 0.5:
            out[y0:y0 + hh, :, 0] = np.roll(img[y0:y0 + hh, :, 0], sh + int(20 * amt), axis=1)
    return out


def halftone(img, cell=10, ink=None, paper=None):
    """Manga screentone rendering of the luminance."""
    ink = hexc("#15121c") if ink is None else ink
    paper = hexc("#f4efe6") if paper is None else paper
    lum = (img * np.array([0.3, 0.59, 0.11], np.float32)).sum(2)
    small = cv2.resize(lum, (W // cell, H // cell), interpolation=cv2.INTER_AREA)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # rotated dot grid
    a = math.radians(45)
    u = (xx * math.cos(a) - yy * math.sin(a)) / cell
    v = (xx * math.sin(a) + yy * math.cos(a)) / cell
    d = np.sqrt((u - np.round(u)) ** 2 + (v - np.round(v)) ** 2)
    L = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    r = np.sqrt(np.clip(1 - L, 0, 1)) * 0.72
    m = smoothstep(r + 0.06, r - 0.06, d)
    return paper + (ink - paper) * m[..., None]


def letterbox(img, amt):
    if amt <= 0:
        return img
    bh = int(round(H * 0.12 * amt))
    img[:bh] = 0
    img[H - bh:] = 0
    return img


def finish(img):
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
