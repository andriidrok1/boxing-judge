"""Punches THROWN. Default mode "radial": the fist leaving the body (rate of change of wrist distance from the torso centre),
which catches hooks and uppercuts that the older "toward the opponent" mode missed. Wrist speed relative to the own shoulder (camera-pan invariant),
normalised by torso length (zoom invariant). No landed/blocked.

punch = peak of forward wrist speed (toward the opponent) above --thr torso-lengths/s,
        with elbow extended (angle >= EXT_MIN) around the peak, min 150 ms between punches of a hand.
kind  = jab if thrown by the lead hand (the shoulder nearer the opponent), else power.
Writes data/punches.parquet: frame, t, shot, who, hand, kind, speed, ext
"""
import argparse
import os
import json
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import json
from common import (VIDEO, FIGHTERS, PUNCHES, ROUNDS, SHOTS, IDENTITY, video_info,
                    L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI, L_HIP, R_HIP)

EXT_MIN = 110.0
STANCE = json.load(open(IDENTITY)).get("stance", {})
MIN_CONF = float(os.environ.get('MIN_CONF', 0.35))
MODE = os.environ.get('MODE', 'radial')   # 'radial' (fist leaving the body) or 'toward' (toward the opponent's hips)
ABS_MIN = float(os.environ.get('ABS_MIN', 2.0))  # radial mode: min absolute wrist speed, torso/s
JUMP = 0.5      # box centre jump (in box heights) between consecutive frames = missed camera cut
OPP_WIN = 6     # frames
DOWN_MAX = 0.8   # sin of the angle below horizontal above which wrist motion is an arm drop, not a punch
INTERP = 2      # bridge wrist dropouts up to this many frames
TORSO_PER_SHOULDER = 1.3   # torso length / shoulder width when hips are not visible
MERGE_S = 0.3   # min time between two punches of the same hand
EDGE = int(os.environ.get('EDGE', 0))   # frames at a segment edge (camera cut / chain change) are not trusted
CLINCH_IOU = float(os.environ.get('CLINCH_IOU', 0.4))  # fighter boxes overlapping this much = clinch, no punches counted


def xy(df, i):
    return df[[f"k{i}x", f"k{i}y"]].values


def step_dist(df):
    """Distance between consecutive rows in box heights, for continuity tests: shoulder midpoints when both rows have
    confident shoulders, else box centres for that pair (never mixed: a box balloons when an arm extends, and mixing
    the two anchors fakes a jump)."""
    ok = (df[[f"k{L_SHO}c", f"k{R_SHO}c"]].min(axis=1).values >= 0.5)
    sx = 0.5 * (df[f"k{L_SHO}x"].values + df[f"k{R_SHO}x"].values)
    sy = 0.5 * (df[f"k{L_SHO}y"].values + df[f"k{R_SHO}y"].values)
    bx = 0.5 * (df.x1.values + df.x2.values)
    by = 0.5 * (df.y1.values + df.y2.values)
    hh = (df.y2 - df.y1).values
    both = ok[1:] & ok[:-1]
    ds = np.hypot(np.diff(sx), np.diff(sy))
    db = np.hypot(np.diff(bx), np.diff(by))
    return np.where(both, ds, db) / np.maximum(hh[1:], hh[:-1])


def angle(a, b, c):
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def detect(g, opp_center, fps, thr):
    """g: one fighter's rows in one shot, consecutive frames. opp_center: (n,2) opponent hip centre."""
    torso = 0.5 * (np.linalg.norm(xy(g, L_SHO) - xy(g, L_HIP), axis=1)
                   + np.linalg.norm(xy(g, R_SHO) - xy(g, R_HIP), axis=1))
    hips_ok = g[[f"k{L_HIP}c", f"k{R_HIP}c"]].min(axis=1).values >= MIN_CONF
    # scale must not switch per frame between two different estimators (that alone fakes 8-12 torso/s spikes in clinches):
    # hips-based length where the hips are seen, carried over the unseen frames; shoulder width only when a whole segment has no hips
    t = pd.Series(np.where(hips_ok, torso, np.nan)).rolling(25, center=True, min_periods=1).median().ffill().bfill()
    if t.isna().all():
        t = pd.Series(TORSO_PER_SHOULDER * np.linalg.norm(xy(g, L_SHO) - xy(g, R_SHO), axis=1)).rolling(25, center=True, min_periods=1).median()
    torso = t.values
    sho_c = 0.5 * (xy(g, L_SHO) + xy(g, R_SHO))
    to_opp = opp_center - sho_c
    to_opp /= np.linalg.norm(to_opp, axis=1, keepdims=True) + 1e-6
    # lead hand from the fighter's stance (identity.json "stance"), not from a per-frame nearest-shoulder guess
    lead_left = np.full(len(g), STANCE.get(g.who.iloc[0], "orthodox") == "orthodox")
    out = []
    body_c = 0.5 * (sho_c + 0.5 * (xy(g, L_HIP) + xy(g, R_HIP)))
    for hand, S, E, Wr in (("L", L_SHO, L_ELB, L_WRI), ("R", R_SHO, R_ELB, R_WRI)):
        rel = xy(g, Wr) - xy(g, S)
        v = np.gradient(rel, axis=0) * fps / torso[:, None]
        v = pd.DataFrame(v).rolling(3, center=True, min_periods=1).mean().values
        if MODE == "radial":
            # punch = fist leaving the body: rate of change of |wrist - body centre|, plus a floor on absolute wrist speed.
            # frames where the wrist keypoint is not confident are NaN: a keypoint that drops out and reappears on the
            # extended arm must not read as a punch, so no speed is computed across such a frame
            rad = np.linalg.norm(xy(g, Wr) - body_c, axis=1) / torso
            rad = np.where(g[f"k{Wr}c"].values >= MIN_CONF, rad, np.nan)
            # bridge dropouts of <= INTERP frames (59-63% of low-confidence runs are 1-2 frames); longer runs stay NaN
            rad = pd.Series(rad).interpolate(limit=INTERP, limit_area="inside").values
            rspeed = np.gradient(rad) * fps
            rspeed = pd.Series(rspeed).rolling(3, center=True, min_periods=1).mean().values
            # a hand dropping to the side also "leaves the body": movement steeper than DOWN_MAX below horizontal is not a punch
            vnorm = np.linalg.norm(v, axis=1)
            down = np.where(vnorm > 0, v[:, 1] / np.maximum(vnorm, 1e-6), 0.0)   # image y grows downward
            fwd = np.where(np.isfinite(rspeed) & (vnorm >= ABS_MIN) & (down < DOWN_MAX), rspeed, 0.0)
        else:
            fwd = (v * to_opp).sum(1)
        ext = angle(xy(g, S), xy(g, E), xy(g, Wr))
        ext = np.where(g[[f"k{E}c", f"k{Wr}c"]].min(axis=1).values >= MIN_CONF, ext, np.nan)   # garbage angles from unseen joints
        ext_max = pd.Series(ext).rolling(5, center=True, min_periods=1).max().values
        conf = g[[f"k{S}c", f"k{E}c", f"k{Wr}c"]].min(axis=1).values
        # pad with zeros so a maximum on the segment edge (punch cut off by a camera cut) is still a peak
        peaks, props = find_peaks(np.r_[0.0, fwd, 0.0], height=thr, distance=max(1, int(0.15 * fps)))
        peaks = peaks - 1
        if os.environ.get("DEBUG_T"):
            t0, t1 = map(float, os.environ["DEBUG_T"].split(","))
            for p in peaks:
                if t0 <= g.frame.iloc[p] / fps <= t1:
                    print(f"  cand {g.who.iloc[0]} {hand} t={g.frame.iloc[p]/fps:.2f} fwd={fwd[p]:.1f} conf={conf[p]:.2f} ext_max={ext_max[p]:.0f} seg_len={len(fwd)} pos={p}")
            m = (g.frame.values / fps >= t0) & (g.frame.values / fps <= t1)
            if m.any():
                print(f"  seg {g.who.iloc[0]} {hand} frames {g.frame.values[m].min()}-{g.frame.values[m].max()} fwd={np.round(fwd[m], 1).tolist()}")
        for p in peaks:
            if conf[p] < MIN_CONF or ext_max[p] < EXT_MIN or p < EDGE or p >= len(fwd) - EDGE:
                continue
            is_lead = bool(lead_left[p]) if hand == "L" else not bool(lead_left[p])
            tw = float((v[p] * to_opp[p]).sum())   # wrist velocity component toward the opponent, torso/s
            hpx = float(g.y2.iloc[p] - g.y1.iloc[p])
            out.append((int(g.frame.iloc[p]), int(g.shot.iloc[p]), g.who.iloc[0], hand,
                        "jab" if is_lead else "power", float(fwd[p]), float(ext_max[p]), tw, hpx))
    return out


def main(thr, fighters_path=FIGHTERS, out_path=PUNCHES):
    fps, n, W, H = video_info(VIDEO)
    f = pd.read_parquet(fighters_path)
    f = f[f.who != "ref"].sort_values("frame").drop_duplicates(["frame", "who"], keep="first")
    f["shot"] = f.frame.map(pd.read_parquet(SHOTS).set_index("frame").shot).fillna(0).astype(int)
    # bridge single missing frames (a fighter dropped for exactly one frame): insert the average of the two neighbours,
    # keypoint confidence = min of the neighbours, so a punch whose peak lands on the gap is not cut in half
    kcols = [c for c in f.columns if c.startswith("k") and c[1:-1].isdigit()]
    fill = []
    for who, g in f.groupby("who"):
        g = g.sort_values("frame")
        fr = g.frame.values
        sd = step_dist(g)
        gap1 = np.flatnonzero(np.diff(fr) == 2)
        for i in gap1:
            a_, b_ = g.iloc[i], g.iloc[i + 1]
            if sd[i] > JUMP * np.sqrt(2):
                continue
            m = a_.copy()
            m["frame"] = fr[i] + 1
            for c in kcols + ["x1", "y1", "x2", "y2"]:
                m[c] = (a_[c] + b_[c]) / 2
            for c in [c for c in kcols if c.endswith("c")]:
                m[c] = min(a_[c], b_[c])
            fill.append(m)
    if fill:
        f = pd.concat([f, pd.DataFrame(fill)], ignore_index=True).sort_values("frame")
        print(f"bridged single-frame gaps: {len(fill)}")
    f["hx"] = 0.5 * (f[f"k{L_HIP}x"] + f[f"k{R_HIP}x"])
    f["hy"] = 0.5 * (f[f"k{L_HIP}y"] + f[f"k{R_HIP}y"])
    hips = f.pivot(index="frame", columns="who", values=["hx", "hy"])
    # clinch flag per frame: the two fighter boxes overlap a lot
    bx = f.pivot(index="frame", columns="who", values=["x1", "y1", "x2", "y2"])
    if bx.shape[1] == 8:
        A, B = sorted(f.who.unique())
        ix = np.minimum(bx[("x2", A)], bx[("x2", B)]) - np.maximum(bx[("x1", A)], bx[("x1", B)])
        iy = np.minimum(bx[("y2", A)], bx[("y2", B)]) - np.maximum(bx[("y1", A)], bx[("y1", B)])
        inter = ix.clip(lower=0) * iy.clip(lower=0)
        area = lambda X: (bx[("x2", X)] - bx[("x1", X)]) * (bx[("y2", X)] - bx[("y1", X)])
        clinch = (inter / (area(A) + area(B) - inter)) > CLINCH_IOU
    else:
        clinch = pd.Series(False, index=bx.index)
    rows = []
    for (who, shot), g in f.sort_values("frame").groupby(["who", "shot"]):
        # split at frame gaps so gradients are over consecutive frames only
        # split at frame gaps, chain changes and box-centre jumps (an undetected camera cut makes the wrist jump = fake punch)
        jump = step_dist(g) > JUMP
        brk = np.flatnonzero((np.diff(g.frame.values) != 1) | jump) + 1   # chain id changes alone are not cuts
        for seg in np.split(np.arange(len(g)), brk):
            if len(seg) < 5:
                continue
            gg = g.iloc[seg]
            other = [w for w in f.who.unique() if w != who][0]
            hx = hips["hx"].reindex(gg.frame)
            hy = hips["hy"].reindex(gg.frame)
            opp = np.c_[hx[other].fillna(hx[who]).values, hy[other].fillna(hy[who]).values]
            rows += detect(gg, opp, fps, thr)
    p = pd.DataFrame(rows, columns=["frame", "shot", "who", "hand", "kind", "speed", "ext", "toward", "box_h"])
    dbg = os.environ.get("DEBUG_T")
    def show(stage):
        if dbg:
            t0, t1 = map(float, dbg.split(","))
            d = p[(p.frame / fps >= t0) & (p.frame / fps <= t1)]
            print(f"[{stage}]", [(round(x / fps, 2), w[0], h, round(sp, 1)) for x, w, h, sp in zip(d.frame, d.who, d.hand, d.speed)])
    show("raw peaks")
    n0 = len(p)
    p = p[~p.frame.map(clinch).fillna(False).astype(bool)]
    print(f"clinch filter removed {n0 - len(p)}")
    show("after clinch")
    # the same hand cannot throw twice within MERGE_S; a detection gap splits one punch into two edge peaks -> keep the faster
    p = p.sort_values(["who", "hand", "frame"])
    keep = np.ones(len(p), bool)
    fr, sp = p.frame.values, p.speed.values
    same = np.r_[False, (p.who.values[1:] == p.who.values[:-1]) & (p.hand.values[1:] == p.hand.values[:-1]) & (np.diff(fr) < MERGE_S * fps)]
    for i in np.flatnonzero(same):
        j = i - 1
        while j >= 0 and not keep[j]:
            j -= 1
        if j < 0 or fr[i] - fr[j] >= MERGE_S * fps:
            continue
        if sp[i] <= sp[j]:
            keep[i] = False
        else:
            keep[j] = False
    n1 = len(p); p = p[keep]
    print(f"merged double peaks: {n1 - len(p)}")
    show("after merge")
    # a punch needs a target: the opponent must be in frame within +-OPP_WIN frames (kills arm swings after a knockdown / walking)
    both = f.groupby("frame").who.nunique()
    near = np.array([any(both.get(fr + d, 0) == 2 for d in range(-OPP_WIN, OPP_WIN + 1)) for fr in p.frame])
    n2 = len(p); p = p[near]
    print(f"no opponent in frame: {n2 - len(p)}")
    show("after opponent")
    p["t"] = p.frame / fps
    p.to_parquet(out_path, index=False)
    print(f"{len(p)} punches thrown detected (thr={thr})")
    if ROUNDS.exists():
        rounds = json.load(open(ROUNDS))
        p["round"] = 0
        for r in rounds:
            p.loc[(p.frame >= r["start"]) & (p.frame < r["end"]), "round"] = r["round"]
        tab = p[p["round"] > 0].pivot_table(index="round", columns=["who", "kind"], values="frame", aggfunc="size", fill_value=0)
        print(tab)
        print(tab.T.groupby(level=0).sum().T)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=3.5, help="forward wrist speed, torso-lengths per second")
    ap.add_argument("--fighters", default=str(FIGHTERS))
    ap.add_argument("--out", default=str(PUNCHES))
    a = ap.parse_args()
    main(a.thr, a.fighters, a.out)
