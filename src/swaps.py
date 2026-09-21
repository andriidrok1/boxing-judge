"""Identity-swap metric: within one shot, a fighter's box centre jumps by more than JUMP box heights
between frames closer than GAP, while the other fighter's box is near where this one just was.
Prints swap count per round and the first few timestamps. usage: python swaps.py [fighters.parquet]"""
import sys, json
import numpy as np
import pandas as pd
from common import FIGHTERS, ROUNDS, SHOTS, video_info, VIDEO

JUMP, GAP = 0.5, 10


def main(path=FIGHTERS):
    fps, n, W, H = video_info(VIDEO)
    f = pd.read_parquet(path, columns=["frame", "who", "x1", "y1", "x2", "y2"])
    f = f[f.who != "ref"].copy()
    f["shot"] = f.frame.map(pd.read_parquet(SHOTS).set_index("frame").shot)
    f["cx"], f["cy"], f["h"] = (f.x1 + f.x2) / 2, (f.y1 + f.y2) / 2, f.y2 - f.y1
    rounds = json.load(open(ROUNDS))
    events = []
    for who, g in f.groupby("who"):
        g = g.sort_values("frame")
        other = f[f.who != who].set_index("frame")
        d = g[["frame", "cx", "cy"]].diff()
        jump = (np.hypot(d.cx, d.cy) / g.h > JUMP) & (d.frame <= GAP) & (g.shot.values == g.shot.shift().values)
        for i in np.flatnonzero(jump.values):
            fr, pfr = int(g.frame.iloc[i]), int(g.frame.iloc[i - 1])
            o = other.loc[other.index == fr]
            if len(o) and np.hypot(o.cx.iloc[0] - g.cx.iloc[i - 1], o.cy.iloc[0] - g.cy.iloc[i - 1]) / g.h.iloc[i] < JUMP:
                events.append((fr, who))
    ev = pd.DataFrame(events, columns=["frame", "who"]).drop_duplicates("frame")
    ev["round"] = 0
    for r in rounds:
        ev.loc[(ev.frame >= r["start"]) & (ev.frame < r["end"]), "round"] = r["round"]
    ev = ev[ev["round"] > 0]
    print("swap events per round:", ev.groupby("round").size().to_dict(), " total:", len(ev))
    print("first in round 5:", [f"{fr/fps:.1f}s" for fr in ev[ev["round"] == 5].frame.head(15)])
    return ev


if __name__ == "__main__":
    main(*sys.argv[1:])
