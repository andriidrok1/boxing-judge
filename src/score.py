"""Round scoring, 10-point must.
composite = 0.6*punch_diff + 0.2*ring_diff + 0.2*aggr_diff  (all in [-1, 1], positive = A)
punch_diff: weighted thrown (power x1.5) share difference
ring_diff : (A advancing & B retreating) - (B advancing & A retreating), fraction of frames
aggr_diff : frac(A advancing) - frac(B advancing)
knockdown (from ring.py, or data/knockdowns.json override {"9": "fury"}) -> 10-8
"""
import json
import numpy as np
import pandas as pd
from common import PUNCHES, ROUNDS, DATA, OUT

RING = DATA / "ring.parquet"
KD_OVERRIDE = DATA / "knockdowns.json"
W_PUNCH, W_RING, W_AGGR, POWER_W = 0.6, 0.2, 0.2, 1.5


def main():
    p = pd.read_parquet(PUNCHES)
    r = pd.read_parquet(RING)
    rounds = json.load(open(ROUNDS))
    names = sorted(p.who.unique())
    A, B = names[0], names[1]
    kd_override = json.load(open(KD_OVERRIDE)) if KD_OVERRIDE.exists() else {}
    rows = []
    for rd in rounds:
        k, s, e = rd["round"], rd["start"], rd["end"]
        pp = p[(p.frame >= s) & (p.frame < e)]
        rr = r[(r.frame >= s) & (r.frame < e)]
        w = pp.assign(w=np.where(pp.kind == "power", POWER_W, 1.0)).groupby("who").w.sum()
        wa, wb = w.get(A, 0.0), w.get(B, 0.0)
        punch = (wa - wb) / (wa + wb + 1e-6)
        ring = (rr.advA & rr.retB).mean() - (rr.advB & rr.retA).mean() if len(rr) else 0.0
        aggr = rr.advA.mean() - rr.advB.mean() if len(rr) else 0.0
        comp = W_PUNCH * punch + W_RING * ring + W_AGGR * aggr
        kd_down = kd_override.get(str(k))   # pose-based knockdown detection is too noisy; facts only (data/knockdowns.json)
        winner = A if comp > 0 else B
        if kd_down is not None:
            winner = B if kd_down == A else A
        loser_pts = 8 if kd_down else 9
        rows.append(dict(round=k, thrown_A=int((pp.who == A).sum()), thrown_B=int((pp.who == B).sum()),
                         punch=round(punch, 3), ring=round(ring, 3), aggr=round(aggr, 3), comp=round(comp, 3),
                         kd=kd_down or "", winner=winner, **{A: 10 if winner == A else loser_pts, B: 10 if winner == B else loser_pts}))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "scores.csv", index=False)
    print(df.to_string(index=False))
    tot = df[[A, B]].sum()
    print(f"\nSYSTEM: {A} {tot[A]} - {tot[B]} {B}")
    cards = DATA / "cards.csv"
    if cards.exists():
        c = pd.read_csv(cards)
        for j in c.columns[1:]:
            jw = c[j].str.replace("_10-8", "")
            agree = (jw.values == df.winner.values).mean()
            print(f"agree with {j}: {agree*100:.0f}% of rounds")


if __name__ == "__main__":
    main()
