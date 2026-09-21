"""External validation only.
 1. thrown per round vs CompuBox (data/compubox.csv, rows with numeric round)
 2. punch detector vs hand labels: labels/round_N.csv with columns t (seconds in video), who, hand(optional)
    a detection matches a label if same fighter and |dt| <= TOL
 3. round winners vs the three judges' cards (data/cards.csv) -> done in score.py
"""
import glob
import json
import numpy as np
import pandas as pd
from common import PUNCHES, ROUNDS, DATA, LABELS, IDENTITY

TOL = 0.25


def main():
    p = pd.read_parquet(PUNCHES)
    rounds = json.load(open(ROUNDS))
    p["round"] = 0
    for r in rounds:
        p.loc[(p.frame >= r["start"]) & (p.frame < r["end"]), "round"] = r["round"]
    cb = pd.read_csv(DATA / "compubox.csv", comment="#")
    cb = cb[cb["round"].astype(str).str.isdigit()].astype({"round": int})
    if len(cb):
        sys_ = p[p["round"] > 0].groupby(["who", "round"]).size().rename("sys_thrown").reset_index()
        m = cb.merge(sys_, left_on=["fighter", "round"], right_on=["who", "round"])
        print("thrown per round: system vs CompuBox")
        print(m[["fighter", "round", "thrown", "sys_thrown"]].to_string(index=False))
        print(f"MAE={np.abs(m.thrown - m.sys_thrown).mean():.1f}  corr={np.corrcoef(m.thrown, m.sys_thrown)[0,1]:.2f}  (n={len(m)})")
    else:
        tot = cb if len(cb) else pd.read_csv(DATA / "compubox.csv", comment="#")
        tot = tot[tot["round"] == "total"]
        print("CompuBox per-round rows missing; totals only:")
        print(tot[["fighter", "thrown"]].to_string(index=False))
        print(p[p["round"] > 0].groupby("who").size().rename("system thrown, rounds only"))
    for path in sorted(glob.glob(str(LABELS / "round_*.csv"))):
        lab = pd.read_csv(path)
        rd = int(path.split("_")[-1].split(".")[0])
        det = p[p["round"] == rd]
        tp = 0
        used = set()
        for _, l in lab.iterrows():
            cand = det[(det.who == l.who) & (np.abs(det.t - l.t) <= TOL) & (~det.index.isin(used))]
            if len(cand):
                used.add(cand.index[0]); tp += 1
        prec = tp / max(len(det), 1)
        rec = tp / max(len(lab), 1)
        print(f"round {rd}: labels={len(lab)} detections={len(det)} TP={tp} precision={prec:.2f} recall={rec:.2f}")


if __name__ == "__main__":
    main()
