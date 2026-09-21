"""Who is who, per frame. Two persistent slots per camera shot, identity decided once per shot.

 1. candidates: tall, bare shoulders, not tiny relative to the largest person, not a referee crop (CLIP p_ref)
 2. per shot, two slots. Each frame's detections (largest 3) are assigned to slots by
    position continuity + colour similarity to the slot's running colour (trunks V, glove hue/V).
    A slot survives gaps up to GAP_MAX frames, so a missed detection does not create a new track.
 3. per shot, slot identity from all evidence at once:
      clip  : mean(p_fury - p_usyk) of slot 0 minus slot 1           (close-ups)
      rel   : relative colour slot0 vs slot1 over co-occurring frames (darker trunks / greener gloves = fury)
    score = W_CLIP*clip - W_REL*rel ; > 0 -> slot0 = fury. Weak score -> absolute trunks brightness.
 4. slots whose mean p_ref is high are the referee.
Writes data/fighters.parquet (track rows + who, chain = shot*2+slot, src, p).
usage: python fighters.py [tracks.parquet] [clip.parquet] [out.parquet]
"""
import json
import sys
import numpy as np
import pandas as pd
from common import VIDEO, TRACKS, FIGHTERS, IDENTITY, DATA, SHOTS, video_info, L_SHO, R_SHO

CLIP = DATA / "clip.parquet"
MIN_H_FRAC = 0.15
SKIN_MIN = 0.3
REL_H = 0.55
REF_P = 0.6          # per-detection CLIP p_ref above this -> not a fighter candidate
REF_SLOT_P = 0.45    # slot mean p_ref above this -> referee slot
GAP_MAX = 25         # frames a slot may be unseen and still continue
POS_MAX = 1.2        # max centre jump (in box heights) per frame gap unit
import os
W_POS, W_COL = 1.0, float(os.environ.get('W_COL', 0.6))
DEDUP_IOU = float(os.environ.get('DEDUP_IOU', 0.3))
CLOSE = 0.9          # boxes closer than this (in heights) -> colour weight W_COL_CLOSE
W_COL_CLOSE = float(os.environ.get('W_COL_CLOSE', 2.0))
REL_CONF = float(os.environ.get('REL_CONF', 1.0))   # |per-frame relative colour| above this -> confident frame label
W_CLIP, W_REL = 2.0, 1.0
WEAK = 0.25
TRUNK_V_SPLIT = 110
EMA = 0.15
UNASSIGNED = float(os.environ.get('UNASSIGNED', 5.0))   # penalty per unassigned detection among the two largest


def colour_vec(r):
    return np.array([r.trunk_v / 40 if pd.notna(r.trunk_v) else np.nan,
                     r.glove_h / 30 if pd.notna(r.glove_h) else np.nan])


def col_dist(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    return float(np.abs(a[m] - b[m]).mean()) if m.any() else 0.5


class Slot:
    def __init__(self, i):
        self.i, self.last_fr, self.cx, self.cy, self.h, self.col, self.rows = i, None, 0, 0, 1, None, []

    def cost(self, r, fr, wcol):
        if self.last_fr is None:
            return None
        gap = fr - self.last_fr
        if gap > GAP_MAX:
            return None
        h = max(self.h, r.y2 - r.y1)
        pos = np.hypot((r.x1 + r.x2) / 2 - self.cx, (r.y1 + r.y2) / 2 - self.cy) / h / max(1, gap) ** 0.5
        if pos > POS_MAX:
            return None
        return W_POS * pos + wcol * col_dist(self.col, colour_vec(r))

    def add(self, r, fr, idx):
        self.last_fr, self.cx, self.cy, self.h = fr, (r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2, r.y2 - r.y1
        c = colour_vec(r)
        if self.col is None:
            self.col = c
        else:
            m = ~np.isnan(c)
            self.col[m] = np.where(np.isnan(self.col[m]), c[m], (1 - EMA) * self.col[m] + EMA * c[m])
        self.rows.append(idx)


def assign_shot(g):
    """g: candidate rows of one shot, sorted by frame. Returns slot index per row (-1 = unassigned)."""
    slots = [Slot(0), Slot(1)]
    out = np.full(len(g), -1)
    for fr, idx in g.groupby("frame").indices.items():
        dets = [(k, g.iloc[k]) for k in idx][:3]
        # all one-to-one assignments of <=2 dets to 2 slots, pick min total cost; unmatched dets may open an empty slot
        best, best_cost = None, np.inf
        options = [(None, None)] + [(k, None) for k, _ in dets] + [(None, k) for k, _ in dets] \
                  + [(k1, k2) for k1, _ in dets for k2, _ in dets if k1 != k2]
        close = False
        if len(dets) >= 2:
            r0, r1 = dets[0][1], dets[1][1]
            close = np.hypot((r0.x1 + r0.x2 - r1.x1 - r1.x2) / 2, (r0.y1 + r0.y2 - r1.y1 - r1.y2) / 2) / max(r0.y2 - r0.y1, r1.y2 - r1.y1) < CLOSE
        wcol = W_COL_CLOSE if close else W_COL
        for a0, a1 in options:
            cost = 0.0
            ok = True
            for s, k in ((slots[0], a0), (slots[1], a1)):
                if k is None:
                    cost += 1.0 if s.last_fr is not None and fr - s.last_fr <= GAP_MAX else 0.0   # missing an active slot
                    continue
                c = s.cost(g.iloc[k], fr, wcol)
                if c is None:
                    if s.last_fr is None or fr - s.last_fr > GAP_MAX:
                        c = 0.8   # (re)start an empty slot
                    else:
                        ok = False; break
                cost += c
            if ok:
                cost += UNASSIGNED * sum(1 for k, _ in dets[:2] if k not in (a0, a1))   # leaving a big person unassigned costs
            if ok and cost < best_cost:
                best, best_cost = (a0, a1), cost
        for s, k in zip(slots, best):
            if k is not None:
                s.add(g.iloc[k], fr, k)
                out[k] = s.i
    return out


def main(tracks=TRACKS, clip=CLIP, out_path=FIGHTERS):
    fps, n, W, H = video_info(VIDEO)
    ident = json.load(open(IDENTITY))
    a, b = ident["fighters"]
    bright_name, dark_name = ident["trunks"]["bright"], ident["trunks"]["dark"]
    df = pd.read_parquet(tracks).merge(pd.read_parquet(clip), on=["frame", "tid"], how="left")
    df["shot"] = df.frame.map(pd.read_parquet(SHOTS).set_index("frame").shot).fillna(0).astype(int)
    df["h"] = df.y2 - df.y1
    df["area"] = (df.x2 - df.x1) * df.h
    sho_ok = df[[f"k{L_SHO}c", f"k{R_SHO}c"]].min(axis=1) > 0.3
    cand = df[sho_ok & (df.h > MIN_H_FRAC * H) & (df.torso_skin > SKIN_MIN) & ~(df.p_ref > REF_P)]
    cand = cand[cand.h > REL_H * cand.groupby("frame").h.transform("max")]
    cand = cand.sort_values(["frame", "area"], ascending=[True, False]).reset_index(drop=True)
    if DEDUP_IOU < 1.0:   # drop a smaller box that mostly overlaps a bigger one in the same frame (YOLO double detection)
        keep = np.ones(len(cand), bool)
        for fr, idx in cand.groupby("frame").indices.items():
            for i in range(1, len(idx)):
                for j in range(i):
                    if not keep[idx[j]]:
                        continue
                    A, B = cand.iloc[idx[j]], cand.iloc[idx[i]]
                    ix = max(0, min(A.x2, B.x2) - max(A.x1, B.x1)); iy = max(0, min(A.y2, B.y2) - max(A.y1, B.y1))
                    inter = ix * iy
                    if inter / (B.area + A.area - inter) > DEDUP_IOU:
                        keep[idx[i]] = False
        cand = cand[keep].reset_index(drop=True)
    cand["slot"] = -1
    for shot, g in cand.groupby("shot"):
        cand.loc[g.index, "slot"] = assign_shot(g)
    cand = cand[cand.slot >= 0].copy()
    cand["chain"] = cand.shot * 2 + cand.slot

    # per-frame confident label where both slots are present: brighter trunks / redder gloves = bright_name
    piv = cand.pivot_table(index=["shot", "frame"], columns="slot", values=["trunk_v", "glove_h", "glove_v"])
    rel_f = pd.Series(0.0, index=piv.index)
    for col, sc in (("trunk_v", 40), ("glove_h", 30), ("glove_v", 30)):
        if (col, 0) in piv and (col, 1) in piv:
            rel_f = rel_f + ((piv[(col, 0)] - piv[(col, 1)]) / sc).fillna(0)
    rel_f = rel_f.where(piv[("trunk_v", 0)].notna() | piv[("trunk_v", 1)].notna())
    lab0 = pd.Series(np.where(rel_f.abs() >= REL_CONF, np.where(rel_f > 0, bright_name, dark_name), None), index=rel_f.index)
    # CLIP per detection, confident close-ups
    cand["lab"] = None
    strong = (cand[f"p_{a}"] - cand[f"p_{b}"]).abs() > 0.6
    cand.loc[strong, "lab"] = np.where((cand[f"p_{a}"] - cand[f"p_{b}"])[strong] > 0, a, b)
    key_sf = list(zip(cand.shot, cand.frame))
    l0 = pd.Series([lab0.get(k) for k in key_sf], index=cand.index)
    l1 = l0.map({a: b, b: a})
    cand["lab"] = cand.lab.where(cand.lab.notna(), np.where(cand.slot == 0, l0, l1))
    # split each slot track into segments at label flips (median over a 9-frame window kills single-frame noise)
    cand["seg"] = 0
    seg_id = 0
    for (shot, slot), g in cand.groupby(["shot", "slot"]):
        lab = g.lab.map({a: 1, b: -1}).astype(float)
        sm = lab.rolling(9, center=True, min_periods=1).median()
        sm = sm.where(sm.abs() > 0.5)          # keep only clear majorities
        filled = sm.ffill()
        change = filled.ne(filled.shift()) & filled.notna() & filled.shift().notna()
        seg = change.cumsum().values + seg_id
        cand.loc[g.index, "seg"] = seg
        seg_id = seg.max() + 1
    who, src = {}, {}
    seg_lab = cand.groupby("seg").lab.agg(lambda x: x.dropna().mode().iloc[0] if x.notna().any() else None)
    seg_n = cand.groupby("seg").lab.agg(lambda x: x.notna().sum())
    for sg, lb in seg_lab.items():
        if lb is not None and seg_n[sg] >= 3:
            who[sg], src[sg] = lb, "frame"
    # referee slots and shot-level fallback for unlabelled segments
    for shot, g in cand.groupby("shot"):
        for slot in (0, 1):
            sl = g[g.slot == slot]
            if len(sl) and sl.p_ref.mean() > REF_SLOT_P:
                for sg in sl.seg.unique():
                    who[sg], src[sg] = "ref", "ref"
        for sg in g.seg.unique():
            if sg in who:
                continue
            sl = g[g.seg == sg]
            other = g[(g.slot != sl.slot.iloc[0]) & g.frame.isin(sl.frame)]
            other_lab = other.seg.map(who).dropna()
            other_lab = other_lab[other_lab != "ref"]
            if len(other_lab):   # the other slot is labelled in these frames -> we are the opposite
                who[sg], src[sg] = (b if other_lab.mode().iloc[0] == a else a), "opposite"
            else:
                tv = sl.trunk_v.mean()
                cd = float((sl[f"p_{a}"] - sl[f"p_{b}"]).mean()) if sl[f"p_{a}"].notna().any() else 0.0
                if abs(cd) > 0.3:
                    who[sg], src[sg] = (a if cd > 0 else b), "clip"
                elif pd.notna(tv):
                    who[sg], src[sg] = (bright_name if tv > TRUNK_V_SPLIT else dark_name), "trunks"
                else:
                    who[sg], src[sg] = "?", "none"
    cand["who"] = cand.seg.map(who).fillna("?")
    cand["src"] = cand.seg.map(src).fillna("none")
    cand["chain"] = cand.seg
    # same name twice in a frame: keep the detection whose segment has more confident frame labels, the other
    # becomes the opposite name if its own per-frame label does not object, else is dropped
    ev = cand.seg.map(seg_n).fillna(0) + cand.src.map({"frame": 100, "clip": 50}).fillna(0)
    cand["ev"] = ev
    fighters_mask = cand.who.isin([a, b])
    dup = cand[fighters_mask].groupby("frame").who.transform("nunique") == 1
    two = cand[fighters_mask].groupby("frame").who.transform("size") == 2
    conflict = cand[fighters_mask][dup & two]
    loser = conflict.sort_values("ev").groupby("frame").head(1)
    flip_ok = loser.lab.isna() | (loser.lab != loser.who)
    cand.loc[loser[flip_ok].index, "who"] = loser[flip_ok].who.map({a: b, b: a})
    cand.loc[loser[flip_ok].index, "src"] = "flip"
    cand = cand.drop(index=loser[~flip_ok].index)
    n_conf = len(loser)
    cand["p"] = cand[[f"p_{a}", f"p_{b}", "p_ref"]].max(axis=1)
    out = cand[cand.who != "?"].sort_values(["frame", "who"])
    out.to_parquet(out_path, index=False)
    per_frame = out[out.who != "ref"].groupby("frame").size()
    print(f"frames with 2 fighters: {(per_frame == 2).sum()} / {df.frame.nunique()} with detections, with 1: {(per_frame == 1).sum()}; "
          f"shots: {cand.shot.nunique()}; identity source per segment: {pd.Series(src).value_counts().to_dict()}; unknown dropped: {int((cand.who == '?').sum())}, same-name conflicts: {n_conf}")
    print(out.who.value_counts().to_dict())


if __name__ == "__main__":
    main(*sys.argv[1:])
