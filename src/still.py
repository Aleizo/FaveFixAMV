"""Render single stills of a scene for previewing: python3 src/still.py scene t lt dur [json params]"""
import sys, json, time, os, cv2
sys.path.insert(0, os.path.dirname(__file__))
import scenes, fx
from audio import Audio
class Ctx: pass
def make_ctx(t, lt, dur, p, seed=0):
    c = Ctx(); c.t=t; c.lt=lt; c.dur=dur; c.u=lt/dur; c.A=A; c.frame=int(t*30); c.seed=seed; c.p=p; return c
A = Audio()
if __name__ == "__main__":
    name=sys.argv[1]; t=float(sys.argv[2]); lt=float(sys.argv[3]); dur=float(sys.argv[4])
    p=json.loads(sys.argv[5]) if len(sys.argv)>5 else {}
    out=sys.argv[6] if len(sys.argv)>6 else f"/tmp/claude-0/-home-user-FaveFixAMV/c1a57004-d669-56c2-ab9e-87f0e9d377ef/scratchpad/stills/{name}.jpg"
    f=getattr(scenes,"sc_"+name)
    t0=time.time(); img=f(make_ctx(t,lt,dur,p)); t1=time.time()
    img=f(make_ctx(t+0.033,lt+0.033,dur,p)); t2=time.time()
    print(f"first {t1-t0:.2f}s, second {t2-t1:.3f}s")
    cv2.imwrite(out, cv2.cvtColor(fx.finish(img), cv2.COLOR_RGB2BGR)[::2,::2] if '--half' in sys.argv else cv2.cvtColor(fx.finish(img), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
