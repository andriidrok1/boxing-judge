"""Who is who, per frame. Each cue is used only where it is reliable:
 - box size + bare shoulders   -> candidates (the two fighters are the biggest people in nearly every shot)
 - chains (centre-distance linking inside a shot) -> temporal smoothing of everything below
 - CLIP p_ref averaged over a chain -> referee chains (per-detection p_ref is fooled by overlapping crops)
 - identity per chain: CLIP fury-vs-usyk when the chain is confident (close-ups), else trunks brightness
   (identity.json "trunks": {"bright": "usyk", "dark": "fury"}) via KMeans(2) on trunks HSV of all candidates
Writes data/fighters.parquet (track rows + who, chain, p).
usage: python fighters.py [tracks.parquet] [clip.parquet] [out.parquet]
"""
import json
import os
import sys
import numpy as np
import pandas as pd
from common import VIDEO, TRACKS, FIGHTERS, IDENTITY, DATA, SHOTS, video_info, L_SHO, R_SHO, L_HIP, R_HIP

CLIP = DATA / "clip.parquet"
MIN_H_FRAC = 0.15
SKIN_MIN = 0.3
LINK_DIST = 0.5
LINK_GAP = 3
SAME_MAN = 0.2      # skeleton centre distance (box heights) below which two boxes are the same man
OVERLAP_IOU = 0.3   # box overlap above which a crop's CLIP label is ignored (two men in one crop)
LAB_VETO = 0.5      # EMA of confident CLIP labels beyond this magnitude vetoes a contradicting confident crop
W_TV, W_PD, W_SKIN = 0.3, 0.3, 0.3   # soft cost weights in chain()
SKIN_VETO = 0.5     # torso skin fraction difference that forbids linking (fighter chain vs a shirt)
REF_P = 0.7          # per-detection CLIP referee probability that, with a covered torso, removes it from candidates
SMOOTH = 7          # frames, final per-chain median smoothing of the name
TV_VETO = 60        # trunks brightness difference that forbids linking a detection to a chain
REL_H = 0.55
REF_CHAIN_P = 0.4
WEAK_MIN_P = 0.15    # colour-only tiers need at least this fighter probability when the torso is covered
REF_SKIN_MAX = 0.6   # a referee wears a shirt: bare torso (skin fraction above this) is never the ref
STRONG_P = 0.7     # per-detection |p_a - p_b| above this = confident close-up label
SPLIT_WIN, SPLIT_MIN = 15, 5
CLIP_CONF = 0.5     # |mean(p_a - p_b)| over a chain above this -> trust CLIP for identity
TRUNK_V_SPLIT = 110
PAIR_DIST = 0.5     # median centre distance (box heights) for two chains to count as two people
PAIR_MIN = 0.1       # |relative colour| between co-occurring chains above this decides who is darker  # absolute fallback: mean trunks V above -> bright trunks


def chain(df, strong=None, tv=None, skin=None, raw_tv=None, raw_pd=None, raw_skin=None, sho_conf=None):
    """Link detections frame to frame inside a shot. Per frame, a joint assignment (Hungarian) between the
    frame's detections and the chains active in the last LINK_GAP frames.
    cost = geometry (shoulder midpoint when both shoulders are confident, else box centre; in box heights)
         + W_TV * |trunks brightness - chain colour| / 60 + W_PD * CLIP disagreement + W_SKIN * |skin - chain skin|
    Soft terms use raw values (also on overlapping crops); hard vetoes only on clean rows (tv/skin/strong are NaN/0
    on overlap rows). Chain colour and label are EMAs, so one polluted sample cannot veto the right chain.
    Unassigned detections start new chains."""
    from scipy.optimize import linear_sum_assignment
    cid = np.full(len(df), -1)
    next_id = 0
    df = df.reset_index(drop=True)
    bx = ((df.x1 + df.x2) / 2).values
    by = ((df.y1 + df.y2) / 2).values
    sx = ((df[f"k{L_SHO}x"] + df[f"k{R_SHO}x"]) / 2).values
    sy = ((df[f"k{L_SHO}y"] + df[f"k{R_SHO}y"]) / 2).values
    h = (df.y2 - df.y1).values
    shot = df.shot.values
    n = len(df)
    strong = np.zeros(n) if strong is None else np.asarray(strong)
    tv = np.full(n, np.nan) if tv is None else np.asarray(tv, dtype=float)
    skin = np.full(n, np.nan) if skin is None else np.asarray(skin, dtype=float)
    raw_tv = tv if raw_tv is None else np.asarray(raw_tv, dtype=float)
    raw_pd = strong if raw_pd is None else np.asarray(raw_pd, dtype=float)
    raw_skin = skin if raw_skin is None else np.asarray(raw_skin, dtype=float)
    sho_conf = np.ones(n) if sho_conf is None else np.asarray(sho_conf, dtype=float)
    last, lab, col, ncol, skn = {}, {}, {}, {}, {}
    for fr, idx in df.groupby("frame").indices.items():
        idx = list(idx)
        act = [c for c, v in last.items() if v[5] == shot[idx[0]]]
        cost = np.full((len(idx), len(act)), 1e6)
        for r, i in enumerate(idx):
            for k, c in enumerate(act):
                f0, bx0, by0, sx0, sy0, s0, h0, sc0 = last[c]
                # hard vetoes, clean rows only
                if strong[i] and abs(lab.get(c, 0)) > LAB_VETO and np.sign(strong[i]) != np.sign(lab[c]):
                    continue
                if np.isfinite(tv[i]) and ncol.get(c, 0) >= 3 and abs(tv[i] - col[c]) > TV_VETO:
                    continue
                if np.isfinite(skin[i]) and c in skn and abs(skin[i] - skn[c]) > SKIN_VETO:
                    continue
                hh = max(h[i], h0)
                if sho_conf[i] > 0.5 and sc0 > 0.5:
                    d = np.hypot(sx[i] - sx0, sy[i] - sy0) / hh
                else:
                    d = np.hypot(bx[i] - bx0, by[i] - by0) / hh
                if d >= LINK_DIST:
                    continue
                soft = 0.0
                if np.isfinite(raw_tv[i]) and c in col:
                    soft += W_TV * min(abs(raw_tv[i] - col[c]) / 60.0, 2.0)
                if c in lab and np.isfinite(raw_pd[i]):
                    soft += W_PD * max(0.0, -raw_pd[i] * lab[c])        # disagreement in sign, scaled by both strengths
                if np.isfinite(raw_skin[i]) and c in skn:
                    soft += W_SKIN * abs(raw_skin[i] - skn[c])
                cost[r, k] = d + soft
        assigned = {}
        if act:
            rows, cols = linear_sum_assignment(cost)
            for r, k in zip(rows, cols):
                if cost[r, k] < 1e6:
                    assigned[idx[r]] = act[k]
        for i in idx:
            c = assigned.get(i)
            if c is None:
                c, next_id = next_id, next_id + 1
            cid[i] = c
            last[c] = (fr, bx[i], by[i], sx[i], sy[i], shot[i], h[i], sho_conf[i])
            if strong[i]:
                lab[c] = strong[i] if c not in lab else 0.7 * lab[c] + 0.3 * strong[i]
            if np.isfinite(tv[i]):
                col[c] = tv[i] if c not in col else 0.8 * col[c] + 0.2 * tv[i]
                ncol[c] = ncol.get(c, 0) + 1
            if np.isfinite(skin[i]):
                skn[c] = skin[i] if c not in skn else 0.8 * skn[c] + 0.2 * skin[i]
        last = {c: v for c, v in last.items() if v[0] >= fr - LINK_GAP}
    return cid


def trunk_feat(df):
    ang = np.deg2rad(df.trunk_h * 2)
    w = df.trunk_s / 255
    return np.c_[np.cos(ang) * w, np.sin(ang) * w, df.trunk_s / 255, df.trunk_v / 255]


def main(tracks=TRACKS, clip=CLIP, out_path=FIGHTERS):
    fps, n, W, H = video_info(VIDEO)
    ident = json.load(open(IDENTITY))
    a, b = ident["fighters"]
    df = pd.read_parquet(tracks).merge(pd.read_parquet(clip), on=["frame", "tid"], how="left")
    df["shot"] = df.frame.map(pd.read_parquet(SHOTS).set_index("frame").shot).fillna(0).astype(int)
    df["h"] = df.y2 - df.y1
    df["area"] = (df.x2 - df.x1) * df.h
    sho_ok = df[[f"k{L_SHO}c", f"k{R_SHO}c"]].min(axis=1) > 0.3
    hip_ok = df[[f"k{L_HIP}c", f"k{R_HIP}c"]].max(axis=1) > 0.3     # a body, not a face at the frame edge
    # a covered torso with a clear referee crop is never a fighter candidate (white shirt under red light reads as skin ~0.3)
    ref_like = (df.p_ref > REF_P) & (df.torso_skin < REF_SKIN_MAX)
    cand = df[sho_ok & (df.h > MIN_H_FRAC * H) & (df.torso_skin > SKIN_MIN) & ~ref_like]
    cand = cand[cand.h > REL_H * cand.groupby("frame").h.transform("max")]   # crowd is much smaller than the fighters
    cand = cand.sort_values(["frame", "area"], ascending=[True, False], kind="stable").groupby("frame").head(3).copy()
    # a crop that overlaps another candidate's box holds two men: its CLIP label says nothing about which one
    ov = np.zeros(len(cand), bool)
    xs1, ys1, xs2, ys2, ar = cand.x1.values, cand.y1.values, cand.x2.values, cand.y2.values, cand.area.values
    for fr, idx in cand.groupby("frame").indices.items():
        for i in idx:
            for j in idx:
                if i == j:
                    continue
                inter = max(0, min(xs2[i], xs2[j]) - max(xs1[i], xs1[j])) * max(0, min(ys2[i], ys2[j]) - max(ys1[i], ys1[j]))
                # a crop holds two men when the boxes overlap, when it sits inside the other, OR when the other sits inside it
                if inter / (ar[i] + ar[j] - inter) > OVERLAP_IOU or inter > 0.6 * ar[i] or inter > 0.6 * ar[j]:
                    ov[i] = True
    cand["overlap"] = ov
    pd_ = cand[f"p_{a}"] - cand[f"p_{b}"]
    strong_lab = np.where((pd_.abs() > STRONG_P) & ~ov, np.sign(pd_), 0)
    # trunks colour sampled on an overlapping crop may belong to the other man: no veto and no update from such frames
    cand["chain"] = chain(cand, strong_lab, cand.trunk_v.where(~cand.overlap).values, cand.torso_skin.where(~cand.overlap).values,
                          raw_tv=cand.trunk_v.values, raw_pd=pd_.values, raw_skin=cand.torso_skin.values,
                          sho_conf=cand[[f"k{L_SHO}c", f"k{R_SHO}c"]].min(axis=1).values)
    if os.environ.get("DEBUG_FRAMES"):
        f0, f1 = map(int, os.environ["DEBUG_FRAMES"].split(","))
        cand["strong"] = strong_lab
        d = cand[(cand.frame >= f0) & (cand.frame <= f1)].assign(shx=lambda x: ((x[f"k{L_SHO}x"] + x[f"k{R_SHO}x"]) / 2).round(0))
        print(d[["frame", "tid", "chain", "x1", "x2", "h", "shx", "area", "strong", "overlap", "trunk_v", "torso_skin", "p_fury", "p_usyk", "p_ref"]].round(2).to_string(index=False))
    cand = cand.reset_index(drop=True)
    # split a chain where the per-detection CLIP label (confident close-ups only) flips for a sustained stretch
    strong = ((cand[f"p_{a}"] - cand[f"p_{b}"]).abs() > STRONG_P) & ~cand.overlap
    cand["lab"] = np.where(strong, np.sign(cand[f"p_{a}"] - cand[f"p_{b}"]), np.nan)
    n_split = 0
    new_chain = cand.chain.values.copy()
    next_id = cand.chain.max() + 1
    for c, g in cand.groupby("chain"):
        sm = g.lab.rolling(SPLIT_WIN, center=True, min_periods=SPLIT_MIN).median()
        sm = sm.where(sm.abs() > 0.5).ffill()
        flip = sm.ne(sm.shift()) & sm.notna() & sm.shift().notna()
        if flip.any():
            seg = flip.cumsum().values
            for k in range(1, seg.max() + 1):
                new_chain[g.index.values[seg == k]] = next_id; next_id += 1; n_split += 1
    cand["chain"] = new_chain
    g = cand.groupby("chain")
    clean = cand[~cand.overlap]
    gc = clean.groupby("chain")
    def agg(col_fn_clean, col_fn_all):
        v = col_fn_clean(gc).reindex(g.size().index)
        return v.where(v.notna(), col_fn_all(g))
    ch = pd.DataFrame({"n": g.size(),
                       "p_ref": agg(lambda x: x.p_ref.mean(), lambda x: x.p_ref.mean()),
                       "p_max_f": agg(lambda x: np.maximum(x[f"p_{a}"].mean(), x[f"p_{b}"].mean()), lambda x: np.maximum(x[f"p_{a}"].mean(), x[f"p_{b}"].mean())),
                       "skin": agg(lambda x: x.torso_skin.mean(), lambda x: x.torso_skin.mean()),
                       "cd": agg(lambda x: x[f"p_{a}"].mean() - x[f"p_{b}"].mean(), lambda x: x[f"p_{a}"].mean() - x[f"p_{b}"].mean())})
    # identity per chain, three tiers:
    #  clip   : |mean p_a - mean p_b| over the chain above CLIP_CONF (close-ups)
    #  pair   : relative colour vs a chain it shares frames with that is already labelled
    #           (darker trunks / greener gloves = fury), robust to lighting changes between shots
    #  trunks : absolute trunks brightness threshold, last resort
    bright_name = ident["trunks"]["bright"]
    dark_name = ident["trunks"]["dark"]
    feat = cand[~cand.overlap][["frame", "chain", "trunk_v", "glove_s", "glove_v"]].copy()
    feat["cx"], feat["cy"], feat["hh"] = (cand.x1 + cand.x2) / 2, (cand.y1 + cand.y2) / 2, cand.y2 - cand.y1
    skin = cand.groupby("chain").torso_skin.mean().reindex(ch.index)
    ch["who"] = np.where((ch.p_ref > REF_CHAIN_P) & (skin < REF_SKIN_MAX), "ref", "")
    ch["src"] = np.where(ch.who == "ref", "ref", "")
    conf = (ch.cd.abs() > CLIP_CONF) & (ch.who != "ref")
    ch.loc[conf, "who"] = np.where(ch.cd[conf] > 0, a, b)
    ch.loc[conf, "src"] = "clip"
    # pairwise relative brightness between chains that co-occur
    pairs = feat.merge(feat, on="frame", suffixes=("_x", "_y"))
    pairs = pairs[pairs.chain_x < pairs.chain_y]
    # > 0: x is the bright one. Bright trunks, bright gloves, LOW glove saturation (white vs green) = usyk
    rel = ((pairs.trunk_v_x - pairs.trunk_v_y) / 40).fillna(0) - ((pairs.glove_s_x - pairs.glove_s_y) / 30).fillna(0) \
        + ((pairs.glove_v_x - pairs.glove_v_y) / 30).fillna(0)
    pairs["rel"] = rel   # > 0: x is the bright one
    pairs["dist"] = np.hypot(pairs.cx_x - pairs.cx_y, pairs.cy_x - pairs.cy_y) / np.maximum(pairs.hh_x, pairs.hh_y)
    pr = pairs.groupby(["chain_x", "chain_y"]).agg(mean=("rel", "mean"), size=("rel", "size"), dist=("dist", "median"))
    pr = pr[(pr["size"] >= 3) & (pr["dist"] > PAIR_DIST)]   # two boxes on the same man are not a pair
    # tier "opposite": a chain that shares frames with a CLIP-labelled chain is the other fighter, whatever its colour
    votes = {}   # chain -> score for name a (positive) vs b (negative)
    for (cx, cy), row in pr.iterrows():
        wx, wy, sx, sy = ch.who.get(cx, ""), ch.who.get(cy, ""), ch.src.get(cx, ""), ch.src.get(cy, "")
        if sx == "clip" and wy == "" and row["size"] >= 5:
            votes[cy] = votes.get(cy, 0.0) + row["size"] * (1 if wx == b else -1)
        elif sy == "clip" and wx == "" and row["size"] >= 5:
            votes[cx] = votes.get(cx, 0.0) + row["size"] * (1 if wy == b else -1)
    for c, v in votes.items():
        v = v / 10.0 + 2.0 * ch.cd.get(c, 0.0)      # own CLIP tendency joins the vote
        if abs(v) >= 0.5:
            ch.loc[c, "who"], ch.loc[c, "src"] = (a if v > 0 else b), "opposite"
    pr = pr[pr["mean"].abs() > PAIR_MIN]
    # pair tier: the relative colour sign must agree with the known partner's name, then the other chain is the other man
    for _ in range(6):
        changed = 0
        for (cx, cy), row in pr.iterrows():
            wx, wy = ch.who.get(cx, ""), ch.who.get(cy, "")
            x_is_bright = row["mean"] > 0
            if wx in (a, b) and wy == "" and (wx == bright_name) == x_is_bright:
                ch.loc[cy, "who"], ch.loc[cy, "src"] = (dark_name if wx == bright_name else bright_name), "pair"; changed += 1
            elif wy in (a, b) and wx == "" and (wy == bright_name) == (not x_is_bright):
                ch.loc[cx, "who"], ch.loc[cx, "src"] = (dark_name if wy == bright_name else bright_name), "pair"; changed += 1
        if not changed:
            break
    tv = agg(lambda x: x.trunk_v.mean(), lambda x: x.trunk_v.mean())
    rest = (ch.who == "") & tv.reindex(ch.index).notna()
    ch.loc[rest, "who"] = np.where(tv.reindex(ch.index)[rest] > TRUNK_V_SPLIT, bright_name, dark_name)
    ch.loc[rest, "src"] = "trunks"
    ch.loc[ch.who == "", "who"] = "?"
    ch.loc[ch.src == "", "src"] = "none"
    # weak tiers (colour only) may not name a fighter when CLIP sees a referee more than a fighter and the torso is covered,
    # nor when there is neither a bare torso nor any fighter probability (crowd, cornermen, camera operators)
    weak = ch.src.isin(["trunks", "pair", "opposite"])
    to_ref = weak & (ch.p_ref > ch.p_max_f) & (ch.skin < REF_SKIN_MAX)
    ch.loc[to_ref, "who"], ch.loc[to_ref, "src"] = "ref", "ref"
    nobody = weak & (ch.skin < REF_SKIN_MAX) & (ch.p_max_f < WEAK_MIN_P)
    ch.loc[nobody, "who"], ch.loc[nobody, "src"] = "?", "none"
    n_gate = int(to_ref.sum() + nobody.sum())
    cand["who"] = cand.chain.map(ch.who)
    cand["src"] = cand.chain.map(ch.src)
    cand["clen"] = cand.chain.map(ch.n)

    fighters = cand[cand.who.isin([a, b])].copy()
    # local evidence for a row's own name: raw CLIP margin toward its name (clean crop) + trunks agreement + chain support
    own_sign = np.where(fighters.who == a, 1.0, -1.0)
    pdv = (fighters[f"p_{a}"] - fighters[f"p_{b}"]).fillna(0).values
    tvv = fighters.trunk_v.values
    tv_agree = np.where(np.isfinite(tvv), np.clip(((tvv - TRUNK_V_SPLIT) / 40.0) * np.where(fighters.who == bright_name, 1, -1), -1, 1), 0)
    src_w = fighters.src.map({"clip": 1.0, "opposite": 0.7, "pair": 0.5, "trunks": 0.2}).fillna(0).values
    fighters["ev"] = np.where(fighters.overlap, 0, pdv * own_sign) + 0.5 * np.where(fighters.overlap, 0, tv_agree) + 0.5 * src_w + fighters.area / 1e7
    # per frame: the best row per name, then the largest remaining row fills the second slot
    best = fighters.sort_values("ev", ascending=False).groupby(["frame", "who"]).head(1)
    rest = fighters.drop(index=best.index)
    need = best.groupby("frame").size()
    fill = rest[rest.frame.map(need).fillna(0) < 2].sort_values("area", ascending=False).groupby("frame").head(1)
    fighters = pd.concat([best, fill]).sort_values(["frame", "area"], ascending=[True, False]).copy()
    same = (fighters.groupby("frame").who.transform("nunique") == 1) & (fighters.groupby("frame").who.transform("size") == 2)
    dup = fighters[same].sort_values("area", ascending=False)
    big, small = dup.groupby("frame").head(1), dup.groupby("frame").tail(1)
    small = small.set_index("frame").loc[big.frame.values]
    # same man = same skeleton: shoulder centres and hip centres within a fraction of the box height.
    # two different men can have heavily overlapping boxes (a YOLO box that spans both), their skeletons do not coincide
    hh = np.maximum(big.h.values, small.h.values)
    d_sho = np.hypot((big[f"k{L_SHO}x"].values + big[f"k{R_SHO}x"].values) - (small[f"k{L_SHO}x"].values + small[f"k{R_SHO}x"].values),
                     (big[f"k{L_SHO}y"].values + big[f"k{R_SHO}y"].values) - (small[f"k{L_SHO}y"].values + small[f"k{R_SHO}y"].values)) / 2 / hh
    d_hip = np.hypot((big[f"k{L_HIP}x"].values + big[f"k{R_HIP}x"].values) - (small[f"k{L_HIP}x"].values + small[f"k{R_HIP}x"].values),
                     (big[f"k{L_HIP}y"].values + big[f"k{R_HIP}y"].values) - (small[f"k{L_HIP}y"].values + small[f"k{R_HIP}y"].values)) / 2 / hh
    same_man = (d_sho < SAME_MAN) & (d_hip < SAME_MAN)
    # a second box on the same person (YOLO double detection) is dropped; a real second person is flipped
    weak = fighters[same].sort_values("ev").groupby("frame").head(1)
    drop_frames = set(big.frame.values[same_man])
    drop = weak[weak.frame.isin(drop_frames)].index
    flip = weak[~weak.frame.isin(drop_frames)].index
    fighters.loc[flip, "who"] = fighters.loc[flip, "who"].map({a: b, b: a})
    fighters = fighters.drop(index=drop)
    # confident CLIP on the crop itself beats the chain label (smoothed: median of +-2 frames in the chain)
    lab_s = fighters.groupby("chain").lab.transform(lambda x: x.rolling(5, center=True, min_periods=3).median())
    clip_name = lab_s.map({1.0: a, -1.0: b})
    override = clip_name.notna() & (clip_name != fighters.who)
    # the override is a swap of the pair: the partner in the same frame takes the freed name, unless its own clean
    # per-crop label objects (then the partner keeps its name and the frame keeps the stronger crop only)
    fr_over = fighters[override].frame.values
    partner = fighters[fighters.frame.isin(fr_over) & ~override]
    partner_lab = clip_name.reindex(partner.index)
    # swap only when the partner is a clean crop (not overlapping) whose own label does not object
    # partner's own clean label: NaN = no opinion (swap), == its freed name = it wants the swap too, == its current
    # name = it objects, and a clean confirmation beats the overriding row -> the override is cancelled
    pl = partner_lab.reindex(partner.index)
    status = pd.Series(np.where(pl.isna(), "free", np.where(pl.values == partner.who.values, "objects", "agrees")), index=partner.index)
    st = pd.DataFrame({"frame": partner.frame, "st": status, "ov": partner.overlap}).groupby("frame").agg(
        obj=("st", lambda x: (x == "objects").any()), ov=("ov", "any"))
    cancel_frames = set(st[st.obj].index)
    override = override & ~fighters.frame.isin(cancel_frames)
    fighters.loc[override, "who"] = clip_name[override]
    fighters.loc[override, "src"] = "clip1"
    ok_frames = set(st[~st.obj & ~st.ov].index) & set(fighters[override].frame)
    swap_idx = partner[partner.frame.isin(ok_frames)].index
    fighters.loc[swap_idx, "who"] = fighters.loc[swap_idx, "who"].map({a: b, b: a})
    fighters.loc[swap_idx, "src"] = "swap"
    n_over = int(override.sum())
    # remaining same-name pairs (partner objected): keep the stronger clean crop
    fighters["pd"] = (fighters[f"p_{a}"] - fighters[f"p_{b}"]).abs().fillna(0).where(~fighters.overlap, 0)
    dup2 = (fighters.groupby("frame").who.transform("nunique") == 1) & (fighters.groupby("frame").who.transform("size") == 2)
    drop2 = fighters[dup2].sort_values("pd").groupby("frame").head(1).index
    fighters = fighters.drop(index=drop2).drop(columns=["pd", "ev"])
    # temporal smoothing: a name inside one chain cannot flip for fewer than SMOOTH//2+1 frames (median over SMOOTH frames)
    fighters = fighters.sort_values("frame")
    sign = fighters.who.map({a: 1.0, b: -1.0})
    sm = sign.groupby(fighters.chain).transform(lambda x: x.rolling(SMOOTH, center=True, min_periods=1).median())
    new_who = np.where(sm > 0, a, np.where(sm < 0, b, fighters.who))
    n_smooth = int((new_who != fighters.who.values).sum())
    fighters["who"] = new_who
    # smoothing may create a same-name pair in a frame: keep the row whose chain has the longer consistent run
    dup3 = (fighters.groupby("frame").who.transform("nunique") == 1) & (fighters.groupby("frame").who.transform("size") == 2)
    if dup3.any():
        runlen = fighters.groupby("chain").size()
        drop3 = fighters[dup3].assign(rl=lambda d: d.chain.map(runlen)).sort_values("rl").groupby("frame").head(1).index
        fighters = fighters.drop(index=drop3)
    ref = cand[cand.who == "ref"].sort_values("area", ascending=False).groupby("frame").head(1)
    out = pd.concat([fighters, ref]).sort_values(["frame", "who"])
    out.to_parquet(out_path, index=False)
    per_frame = out[out.who != "ref"].groupby("frame").size()
    print(f"frames with 2 fighters: {(per_frame == 2).sum()} / {df.frame.nunique()} with detections, with 1: {(per_frame == 1).sum()}; "
          f"chains: {len(ch)} (split by CLIP flips: {n_split}, ref {int((ch.who == 'ref').sum())}, ? {int((ch.who == '?').sum())}); identity source: {ch.src.value_counts().to_dict()}; flipped: {len(flip)}, dropped double boxes: {len(drop)}, weak-tier chains gated out: {n_gate}, per-crop CLIP overrides: {n_over}, smoothed flips: {n_smooth}, dropped after override: {len(drop2)}")
    print(out.who.value_counts().to_dict())


if __name__ == "__main__":
    main(*sys.argv[1:])
