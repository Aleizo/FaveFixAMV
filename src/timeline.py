"""Shot list for the "Fave Fix" AMV, cut on the 140 BPM beat grid (see audio.py).

Each entry: (start_beat, end_beat, scene, params, section). Section names drive post-FX intensity.
"""
from audio import beat_time, DURATION

INTENSITY = {
    "intro": 0.15, "verse": 0.3, "pre": 0.7, "chorus": 1.0, "break": 0.5, "quiet": 0.15, "build": 0.6,
    "drop": 0.2, "bass": 0.95, "final": 1.1, "outro": 0.1,
}
LETTERBOX = {"intro", "verse", "quiet", "outro"}

S = []


def shot(b0, b1, scene, sec, **p):
    S.append((b0, b1, scene, p, sec))


# ---- intro: night sky, then title ------------------------------------------------------------
shot(-1, 9, "sky", "intro", pal="night", stars=1.0, pan=8, sun=None, cover=0.42, tilt=-40)
shot(9, 17, "title", "intro", pal="night", moon=True, delay=0.2)

# ---- verse 1a: golden hour ---------------------------------------------------------------------
shot(17, 25, "sky", "verse", pal="golden", pan=45, petals=25)
shot(25, 33, "poles", "verse", pal="golden", birds=True, speed=120)
shot(33, 41, "rooftop", "verse", pal="golden", petals=35, wind=1.0)
shot(41, 49, "eye", "verse", iris="violet", open_in=0.7, blinks=[2.6])

# ---- verse 1b: dusk ----------------------------------------------------------------------------
shot(49, 57, "train", "verse", pal="dusk")
shot(57, 65, "city", "verse", pal="dusk", lit_ramp=(0.0, 0.85), stars=0.4, clouds=True, speed=40)
shot(65, 73, "moon", "verse", pal="dusk", pose="stand", petals=30, r=0.36, my=0.42)
shot(73, 77, "eye", "verse", iris="rose", blinks=[0.9])
shot(77, 79, "focus", "pre", text="FAVE", bg="#ff2f7a")
shot(79, 81, "focus", "pre", text="FIX", bg="#29c8ff", fg="#ffffff")

# ---- chorus 1 ----------------------------------------------------------------------------------
shot(81, 89, "warp", "chorus", girl=True)
shot(89, 93, "kinetic", "chorus")
shot(93, 97, "city", "chorus", pal="night", lit=0.8, speed=260, moon=True)
shot(97, 101, "moon", "chorus", pal="night", pose="run", cross=True, lines=0.6)
shot(101, 105, "eye", "chorus", iris="cyan")
shot(105, 109, "focus", "chorus", text="FIX", bg="#8a5cff")
shot(109, 113, "warp", "chorus", cols=["#29e6ff", "#ffffff", "#8a5cff", "#ff2fa8"], speed=1.6)
shot(113, 117, "rooftop", "chorus", pal="dusk", petals=60, wind=1.8)
shot(117, 121, "city", "chorus", pal="night", lit=0.7, speed=120, girl=True, rain=0.6)
shot(121, 125, "kinetic", "chorus", words=["お気に入り", "FAVE", "FIX", "♡"])
shot(125, 129, "kaleido", "chorus")
shot(129, 133, "moon", "chorus", pal="dusk", pose="run", lines=0.8, petals=40)
shot(133, 137, "warp", "chorus", girl=True, speed=1.4)
shot(137, 141, "focus", "chorus", text="FAVE", bg="#ff2f7a", solid=True)
shot(141, 145, "poles", "chorus", pal="dusk", speed=900)
shot(145, 148, "kinetic", "chorus", words=["FAVE", "FIX", "!!"])
# breakdown: screentone freeze then silence
shot(148, 151, "freeze", "break", src="rooftop", src_p=dict(pal="dusk", wind=1.2), text="FAVE FIX")
shot(151, 154, "black", "drop", text="…")

# ---- verse 2: rain, quiet ----------------------------------------------------------------------
shot(154, 162, "window", "quiet", girl=True)
shot(162, 170, "streetlight", "quiet")
shot(170, 176, "eye", "quiet", iris="cyan", tear=True, blinks=[1.6])
shot(176, 182, "city", "quiet", pal="night", lit=0.35, speed=20, rain=1.0, girl=True)

# ---- build: cuts accelerate --------------------------------------------------------------------
shot(182, 190, "sky", "build", pal="dusk", pan=160, tilt=25, petals=20)
shot(190, 198, "poles", "build", pal="night", speed=420, stars=1.0)
shot(198, 202, "train", "build", pal="dusk", window=(0.05, 0.95))
shot(202, 206, "rooftop", "build", pal="night", stars=1.0, petals=20, wind=1.5)
shot(206, 208, "eye", "build", iris="violet")
shot(208, 210, "city", "build", pal="night", lit=0.9, speed=500)
shot(210, 212, "focus", "build", text="FAVE", bg="#ffffff", fg="#ff2f7a", solid=True)
shot(212, 214, "moon", "build", pal="night", pose="run", lines=1.0)
shot(214, 218, "kinetic", "build", words=["FAVE", "FIX", "FAVE", "FIX"])
shot(218, 222, "black", "drop", text="FIX", size=140)

# ---- bass section: eclipse / red neon ----------------------------------------------------------
RED = ["#ff2448", "#ff9a3c", "#7a1cff", "#ffffff"]
shot(222, 230, "hextunnel", "bass", eclipse=True)
shot(230, 238, "moon", "bass", pal="blood", pose="stand", eclipse=0.6, mcol="#ff5a4a", r=0.4)
shot(238, 242, "hextunnel", "bass", girl=True)
shot(242, 246, "city", "bass", pal="blood", lit=0.5, speed=200, rain=0.8)
shot(246, 250, "eye", "bass", iris="red")
shot(250, 254, "warp", "bass", cols=RED, bg0="#0a0004", bg1="#2a0010", girl=True)
shot(254, 262, "hextunnel", "bass", eclipse=True, girl=True, cols=["#ff2448", "#ffffff", "#ff7a3c"])
shot(262, 266, "moon", "bass", pal="blood", pose="run", cross=True, lines=0.9, mcol="#ff6a5a")
shot(266, 270, "focus", "bass", text="FIX", bg="#d0102a", fg="#ffffff")
shot(270, 274, "kinetic", "bass", words=["FIX", "FIX", "FIX", "FAVE"],
     cols=[("#0a0004", "#ff2448"), ("#ff2448", "#0a0004"), ("#ffffff", "#d0102a"), ("#0a0004", "#ffffff")])
shot(274, 278, "warp", "bass", cols=RED, bg0="#0a0004", bg1="#2a0010", speed=1.8)
shot(278, 282, "hextunnel", "bass", eclipse=True)
shot(282, 286, "eye", "break", iris="cyan", open_in=0.9)

# ---- chorus 2: daylight ------------------------------------------------------------------------
CY = ["#29e6ff", "#ffffff", "#ff7ad9", "#a8f0ff"]
shot(286, 294, "sky", "chorus", pal="day", pan=120, petals=70, sun=(1500, 160, 70))
shot(294, 298, "rooftop", "chorus", pal="day", petals=50, wind=1.6, sun=(1300, 260, 80))
shot(298, 302, "kinetic", "chorus", cols=[("#ffffff", "#ff2f7a"), ("#29c8ff", "#ffffff"), ("#ff2f7a", "#ffffff"),
                                          ("#ffffff", "#29c8ff")])
shot(302, 306, "warp", "chorus", cols=CY, bg0="#04122a", bg1="#0a3a6a", girl=True)
shot(306, 310, "poles", "chorus", pal="day", speed=700, birds=True)
shot(310, 314, "focus", "chorus", text="FAVE", bg="#29c8ff")
shot(314, 318, "moon", "chorus", pal="dusk", pose="run", cross=True, lines=0.7, petals=30)
shot(318, 322, "city", "chorus", pal="night", lit=0.85, speed=300, moon=True)
shot(322, 326, "kaleido", "chorus", seed=9)
shot(326, 330, "eye", "chorus", iris="gold")
shot(330, 334, "train", "chorus", pal="golden", window=(0.0, 1.0))
shot(334, 338, "warp", "chorus", girl=True, speed=1.5)
shot(338, 342, "kinetic", "chorus", words=["フェイヴ", "フィックス", "FAVE", "FIX"])
shot(342, 346, "rooftop", "chorus", pal="golden", petals=80, wind=2.0)
shot(346, 350, "focus", "chorus", text="FIX", bg="#ff2f7a", solid=True)
shot(350, 354, "freeze", "break", src="moon", src_p=dict(pal="dusk", pose="run"), text="FAVE FIX",
     ink="#2a0c2a", paper="#ffe6ee")

# ---- final chorus: dawn, everything at once ----------------------------------------------------
shot(354, 362, "warp", "final", girl=True, cols=["#ffb36b", "#ff2fa8", "#29e6ff", "#ffffff"], speed=1.7)
shot(362, 366, "kinetic", "final", words=["FAVE", "FIX", "FAVE", "FIX"])
shot(366, 370, "moon", "final", pal="dawn", pose="run", cross=True, lines=0.9, petals=60)
shot(370, 374, "sky", "final", pal="dawn", pan=220, petals=120, sun=(960, 900, 110))
shot(374, 378, "city", "final", pal="night", lit=1.0, speed=420, girl=True)
shot(378, 382, "eye", "final", iris="violet")
shot(382, 386, "hextunnel", "final", cols=["#29e6ff", "#ff2fa8", "#ffe14d"], girl=True)
shot(386, 390, "kaleido", "final", seed=5, seg=12)
shot(390, 394, "focus", "final", text="FAVE", bg="#ffb020", fg="#ffffff")
shot(394, 398, "rooftop", "final", pal="dawn", petals=120, wind=2.2)
shot(398, 402, "warp", "final", girl=True, speed=2.0)
shot(402, 406, "poles", "final", pal="dawn", speed=1000, birds=True)
shot(406, 410, "kinetic", "final", words=["お気に入り", "FAVE", "FIX", "♡"])
shot(410, 414, "moon", "final", pal="dawn", pose="run", lines=1.0, petals=80)
shot(414, 418, "focus", "final", text="FAVE FIX", size=230, bg="#ff2f7a", solid=True)

# ---- outro -------------------------------------------------------------------------------------
shot(418, 999, "title", "outro", pal="night", moon=True, delay=0.0)


def shots():
    out = []
    for b0, b1, scene, p, sec in S:
        t0 = 0.0 if b0 < 0 else beat_time(b0)
        t1 = DURATION + 1 if b1 >= 999 else beat_time(b1)
        out.append(dict(t0=t0, t1=t1, scene=scene, p=p, sec=sec, b0=b0, b1=b1))
    for a, b in zip(out[:-1], out[1:]):
        assert abs(a["t1"] - b["t0"]) < 1e-6, (a, b)
    return out
