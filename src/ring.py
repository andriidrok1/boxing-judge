"""Ring control / aggression / knockdown from fighter positions, camera pan compensated.

Per frame with both fighters:
  d      = unit vector A -> B (hip centres)
  vX     = hip-centre velocity of X minus global pan (dx, dy) from shots.parquet, in torso-lengths/s
  A advancing  = vA.d > EPS ;  B retreating = vB.d > EPS   (and symmetric)
Knockdown candidate = one fighter's box wider than tall, or hips at knee height, for >= 0.8 s.
Writes data/ring.parquet (frame, t, dist, advA, advB, retA, retB, kdA, kdB).
"""
import numpy as np
import pandas as pd
from common import VIDEO, FIGHTERS, SHOTS, DATA, video_info, L_SHO, R_SHO, L_HIP, R_HIP, L_KNEE, R_KNEE

EPS = 0.3   # torso-lengths / s
RING = DATA / "ring.parquet"


def main():
    fps, n, W, H = video_info(VIDEO)
    f = pd.read_parquet(FIGHTERS)
    pan = pd.read_parquet(SHOTS).set_index("frame")[["dx", "dy", "shot"]]
    f = f[f.who != "ref"]
    f = f[f.groupby("frame").who.transform("nunique") == 2].copy()
    for c in ("x", "y"):
        f[f"hip{c}"] = 0.5 * (f[f"k{L_HIP}{c}"] + f[f"k{R_HIP}{c}"])
    f["torso"] = 0.5 * (np.hypot(f[f"k{L_SHO}x"] - f[f"k{L_HIP}x"], f[f"k{L_SHO}y"] - f[f"k{L_HIP}y"])
                        + np.hypot(f[f"k{R_SHO}x"] - f[f"k{R_HIP}x"], f[f"k{R_SHO}y"] - f[f"k{R_HIP}y"]))
    f["wide"] = (f.x2 - f.x1) > (f.y2 - f.y1)
    f["low"] = (0.5 * (f[f"k{L_KNEE}y"] + f[f"k{R_KNEE}y"]) - f.hipy) < 0.25 * f.torso
    names = sorted(f.who.unique())
    A, B = names[0], names[1]
    wide = f.pivot(index="frame", columns="who", values=["hipx", "hipy", "torso", "wide", "low", "shot"])
    wide = wide.sort_index().dropna()
    for c in ("hipx", "hipy", "torso"):
        wide[c] = wide[c].astype(float)
    fr = wide.index.values
    consec = np.r_[False, np.diff(fr) == 1]
    same_shot = np.r_[False, wide[("shot", A)].values[1:] == wide[("shot", A)].values[:-1]]
    ok = consec & same_shot
    torso = 0.5 * (wide[("torso", A)] + wide[("torso", B)]).rolling(9, center=True, min_periods=1).median().values.astype(float)
    out = pd.DataFrame({"frame": fr, "t": fr / fps})
    cA = wide[["hipx", "hipy"]].xs(A, axis=1, level=1).values.astype(float)
    cB = wide[["hipx", "hipy"]].xs(B, axis=1, level=1).values.astype(float)
    d = cB - cA
    out["dist"] = np.linalg.norm(d, axis=1) / torso
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-6
    p = pan.reindex(fr)[["dx", "dy"]].fillna(0).values
    vA = np.r_[[[0, 0]], np.diff(cA, axis=0)] - p
    vB = np.r_[[[0, 0]], np.diff(cB, axis=0)] - p
    vA = pd.DataFrame(vA).rolling(5, center=True, min_periods=1).mean().values * fps / torso[:, None]
    vB = pd.DataFrame(vB).rolling(5, center=True, min_periods=1).mean().values * fps / torso[:, None]
    fa, fb = (vA * d).sum(1), (vB * d).sum(1)
    out["advA"] = ok & (fa > EPS)
    out["retA"] = ok & (fa < -EPS)
    out["advB"] = ok & (fb < -EPS)
    out["retB"] = ok & (fb > EPS)
    for X in (A, B):
        down = (wide[("wide", X)].astype(bool) | wide[("low", X)].astype(bool)).astype(int)
        out[f"kd{X}"] = down.rolling(int(0.8 * fps), min_periods=1).sum().values >= int(0.8 * fps) * 0.8
    out.attrs["names"] = names
    out.to_parquet(RING, index=False)
    print(f"{len(out)} frames with both fighters; A={A} B={B}")
    for X in (A, B):
        kd = out[out[f"kd{X}"]]
        if len(kd):
            segs = np.split(kd.frame.values, np.flatnonzero(np.diff(kd.frame.values) > fps) + 1)
            print(f"knockdown candidates {X}:", [f"{s[0]/fps/60:.2f}min ({len(s)/fps:.1f}s)" for s in segs][:20])


if __name__ == "__main__":
    main()
