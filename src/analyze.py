import librosa, numpy as np, json, sys
y, sr = librosa.load("assets/fave_fix.mp3", sr=22050, mono=True)
dur = len(y)/sr
hop=512
oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
tempo, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=sr, hop_length=hop, units='time')
onsets = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, hop_length=hop, units='time')
rms = librosa.feature.rms(y=y, hop_length=hop)[0]
S = np.abs(librosa.stft(y, hop_length=hop, n_fft=2048))
freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
low = S[freqs<150].mean(0); mid=S[(freqs>=150)&(freqs<2000)].mean(0); high=S[freqs>=4000].mean(0)
# percussive onsets for kick/snare
yh, yp = librosa.effects.hpss(y)
pe = librosa.onset.onset_strength(y=yp, sr=sr, hop_length=hop)
# segmentation
chroma = librosa.feature.chroma_cqt(y=yh, sr=sr, hop_length=hop)
mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop, n_mfcc=13)
feat = np.vstack([librosa.util.normalize(chroma,axis=1), librosa.util.normalize(mfcc,axis=1)])
bf = librosa.util.sync(feat, librosa.time_to_frames(beats, sr=sr, hop_length=hop), aggregate=np.median)
bounds = librosa.segment.agglomerative(bf, 14)
btimes = [0.0]+[float(beats[min(b,len(beats)-1)]) for b in bounds[1:]]
t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
def norm(a): a=np.asarray(a,float); return ((a-a.min())/(np.percentile(a,99.5)-a.min()+1e-9)).clip(0,1)
out=dict(duration=dur, tempo=float(np.atleast_1d(tempo)[0]), beats=[float(b) for b in beats], onsets=[float(o) for o in onsets],
  seg_bounds=btimes, fps_feat=sr/hop, rms=norm(rms).round(4).tolist(), low=norm(low).round(4).tolist(), mid=norm(mid).round(4).tolist(),
  high=norm(high).round(4).tolist(), perc=norm(pe).round(4).tolist())
json.dump(out, open("analysis/audio.json","w"))
print("dur",dur,"tempo",out['tempo'],"nbeats",len(beats))
print("segments",[round(b,2) for b in btimes])
# energy per 4 seconds
for i in range(0,int(dur),4):
    m=(t>=i)&(t<i+4); print(f"{i:4d}s rms={norm(rms)[m].mean():.2f} low={norm(low)[m].mean():.2f} high={norm(high)[m].mean():.2f}")
print("first beats", [round(b,3) for b in beats[:16]])
