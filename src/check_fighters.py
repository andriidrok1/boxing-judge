"""Visual QA: render N random in-round frames with fighter/ref boxes into out/qa/grid_K.jpg (12 per grid).
usage: python check_fighters.py [N=96] [seed=0]
"""
import json
import sys
import numpy as np
import cv2
import pandas as pd
from common import VIDEO, FIGHTERS, ROUNDS, OUT, frames

COL = {"fury": (0, 255, 0), "usyk": (255, 255, 255), "ref": (0, 0, 255)}


def main(n=96, seed=0):
    n, seed = int(n), int(seed)
    f = pd.read_parquet(FIGHTERS, columns=["frame", "who", "x1", "y1", "x2", "y2", "src"])
    rounds = json.load(open(ROUNDS))
    pool = np.concatenate([np.arange(r["start"], r["end"]) for r in rounds])
    picks = np.sort(np.random.default_rng(seed).choice(pool, n, replace=False))
    (OUT / "qa").mkdir(exist_ok=True)
    tiles = []
    for k, fr in enumerate(picks):
        img = next(frames(VIDEO, int(fr), int(fr) + 1))[1]
        for _, r in f[f.frame == fr].iterrows():
            c = COL[r.who]
            cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), c, 3)
            cv2.putText(img, f"{r.who} {r.src}", (int(r.x1) + 4, int(r.y1) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, c, 3)
        cv2.putText(img, str(fr), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        tiles.append(cv2.resize(img, (480, 270)))
        if len(tiles) == 12:
            grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, 12, 4)])
            cv2.imwrite(str(OUT / "qa" / f"grid_{k // 12}.jpg"), grid)
            tiles = []
    print("->", OUT / "qa")


if __name__ == "__main__":
    main(*sys.argv[1:])
