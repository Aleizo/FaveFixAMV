"""Per-video-frame audio features for "Fave Fix".

Run once to build analysis/features.npz; the renderer loads that file.
The track is a steady 140 BPM; beat k sits at BEAT0 + k * BEAT.
"""
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO = os.path.join(ROOT, "assets", "fave_fix.mp3")
FEATS = os.path.join(ROOT, "analysis", "features.npz")

FPS = 30
BPM = 140.0
BEAT = 60.0 / BPM
BEAT0 = 0.075
DURATION = 181.93


def beat_time(k):
    return BEAT0 + k * BEAT


def build():
    import librosa
    sr, hop = 22050, 256
    y, _ = librosa.load(AUDIO, sr=sr, mono=True)
    n_frames = int(np.ceil(len(y) / sr * FPS))
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    f = librosa.fft_frequencies(sr=sr, n_fft=2048)
    ft = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop)

    def band(lo, hi):
        return S[(f >= lo) & (f < hi)].mean(0)

    def norm(a, pct=99.0):
        a = np.asarray(a, float)
        lo, hi = np.percentile(a, 2), np.percentile(a, pct)
        return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)

    low = norm(band(30, 150))
    mid = norm(band(300, 2500))
    high = norm(band(5000, 11000))
    rms = norm(librosa.feature.rms(y=y, hop_length=hop)[0])
    # Kick transients: positive flux of the sub band, peak-normalised.
    sub = band(35, 120)
    flux = np.maximum(0, np.diff(np.log1p(sub * 50), prepend=0))
    flux = norm(flux, 99.5)
    yh, yp = librosa.effects.hpss(y)
    snare_env = librosa.onset.onset_strength(y=yp, sr=sr, hop_length=hop, fmin=1500, fmax=8000)
    snare_env = norm(snare_env, 99.5)

    vt = np.arange(n_frames) / FPS

    def at(a):
        return np.interp(vt, ft[: len(a)], a)

    # Envelope follower for kick hits: instant attack, exponential release.
    kick_raw = at(flux)
    kick = np.zeros(n_frames)
    v = 0.0
    for i, k in enumerate(kick_raw):
        v = max(k, v * 0.78)
        kick[i] = v
    snare = np.zeros(n_frames)
    v = 0.0
    for i, k in enumerate(at(snare_env)):
        v = max(k, v * 0.75)
        snare[i] = v

    def smooth(a, n):
        k = np.ones(n) / n
        return np.convolve(a, k, mode="same")

    np.savez_compressed(
        FEATS,
        low=smooth(at(low), 3), mid=smooth(at(mid), 3), high=smooth(at(high), 3),
        rms=smooth(at(rms), 5), energy=smooth(at(rms), 45), kick=kick, snare=snare,
    )
    print("frames", n_frames)


class Audio:
    def __init__(self):
        d = np.load(FEATS)
        self.d = {k: d[k] for k in d.files}
        self.n = len(self.d["rms"])

    def __getattr__(self, name):
        raise AttributeError(name)

    def get(self, name, t):
        a = self.d[name]
        i = min(max(int(round(t * FPS)), 0), self.n - 1)
        return float(a[i])

    @staticmethod
    def beat_phase(t):
        """(beat index, seconds since that beat)."""
        k = np.floor((t - BEAT0) / BEAT)
        return int(k), t - beat_time(k)

    @staticmethod
    def pulse(t, rate=7.0, every=1, offset=0):
        """Exponential pulse that fires on every `every`-th grid beat."""
        k = np.floor((t - BEAT0 - offset * BEAT) / (BEAT * every))
        dt = t - (BEAT0 + offset * BEAT + k * BEAT * every)
        return float(np.exp(-dt * rate))


if __name__ == "__main__":
    build()
