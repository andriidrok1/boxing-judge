"""Verified punch list for a demo window.
 1. python verify_clip.py --start-sec 1800 --dur 30            -> labels/clip_1800_candidates.csv + out/verify/clip_1800_*.jpg
    (low-threshold detector candidates, 5-frame strips with the wrist marked, 10 per sheet, plus a contact sheet of the
    whole window every 4 frames so punches the detector never proposed can be added by hand)
 2. review the sheets, write labels/clip_1800_verdicts.csv with columns idx,t,who,hand,verdict (tp / fp / clinch / unclear),
    extra rows with idx=-1 for punches added by hand
 3. python overlay.py --start-sec 1800 --dur 30 --verified labels/clip_1800_verdicts.csv
"""
import argparse
import os
import subprocess
import sys
import numpy as np
import pandas as pd
import cv2
from common import VIDEO, FIGHTERS, ROOT, LABELS, OUT, frames, video_info


def main(start, dur, thr):
    fps, n, W, H = video_info(VIDEO)
    s, e = int(start * fps), int((start + dur) * fps)
    tmp = ROOT / "data" / f"punches_cand_{int(start)}.parquet"
    subprocess.run([sys.executable, "punches.py", "--thr", str(thr), "--out", str(tmp)], check=True, capture_output=True)
    p = pd.read_parquet(tmp)
    p = p[(p.frame >= s) & (p.frame < e)].sort_values("frame").reset_index(drop=True)
    p.index.name = "idx"
    cand = LABELS / f"clip_{int(start)}_candidates.csv"
    p[["t", "who", "hand", "kind", "speed", "frame"]].round(2).to_csv(cand)
    f = pd.read_parquet(FIGHTERS)
    od = OUT / "verify"
    od.mkdir(exist_ok=True)
    rows, k = [], 0
    for i, d in p.iterrows():
        tiles = []
        W_ = 9 if d.hand == "L" else 10
        for fr in range(int(d.frame) - 4, int(d.frame) + 5, 2):
            img = next(frames(VIDEO, fr, fr + 1))[1]
            r = f[(f.frame == fr) & (f.who == d.who)]
            if len(r):
                r = r.iloc[0]
                cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), (0, 255, 0), 2)
                if r[f"k{W_}c"] > 0.2:
                    cv2.circle(img, (int(r[f"k{W_}x"]), int(r[f"k{W_}y"])), 12, (0, 0, 255), -1)
            img = cv2.resize(img, (320, 180))
            cv2.putText(img, f"{fr - int(d.frame):+d}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            tiles.append(img)
        lab = np.zeros((180, 100, 3), np.uint8)
        cv2.putText(lab, f"#{i}", (4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(lab, f"{d.who[:1].upper()}{d.hand}", (4, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(lab, f"{d.speed:.1f}", (4, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(lab, f"{d.t:.2f}", (4, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        rows.append(np.hstack([lab] + tiles))
        if len(rows) == 10:
            cv2.imwrite(str(od / f"clip_{int(start)}_s{k:02d}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85]); k += 1; rows = []
    if rows:
        cv2.imwrite(str(od / f"clip_{int(start)}_s{k:02d}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85]); k += 1
    # contact sheet of the whole window, every 4 frames, 8 per row, cropped to the two fighters
    tiles, m = [], 0
    for fr in range(s, e, 4):
        img = next(frames(VIDEO, fr, fr + 1))[1]
        b = f[(f.frame == fr) & (f.who != "ref")]
        if len(b):
            x1, y1, x2, y2 = int(b.x1.min()), int(b.y1.min()), int(b.x2.max()), int(b.y2.max())
            side = int(max(x2 - x1, y2 - y1) * 1.1); cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            x1, y1 = max(0, cx - side // 2), max(0, cy - side // 2); img = img[y1:min(H, y1 + side), x1:min(W, x1 + side)]
        t = cv2.resize(img, (220, 220)); cv2.putText(t, f"{fr / fps:.2f}", (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        tiles.append(t)
        if len(tiles) == 48:
            cv2.imwrite(str(od / f"clip_{int(start)}_sheet{m:02d}.jpg"), np.vstack([np.hstack(tiles[j:j + 8]) for j in range(0, 48, 8)]), [cv2.IMWRITE_JPEG_QUALITY, 80]); m += 1; tiles = []
    if tiles:
        while len(tiles) % 8: tiles.append(np.zeros((220, 220, 3), np.uint8))
        cv2.imwrite(str(od / f"clip_{int(start)}_sheet{m:02d}.jpg"), np.vstack([np.hstack(tiles[j:j + 8]) for j in range(0, len(tiles), 8)]), [cv2.IMWRITE_JPEG_QUALITY, 80]); m += 1
    os.remove(tmp)
    print(f"{len(p)} candidates -> {cand}; {k} strip sheets, {m} contact sheets in {od}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-sec", type=float, required=True)
    ap.add_argument("--dur", type=float, default=30)
    ap.add_argument("--thr", type=float, default=1.5)
    a = ap.parse_args()
    main(a.start_sec, a.dur, a.thr)
