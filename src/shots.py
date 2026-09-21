"""Camera-cut detection + per-frame global pan estimate.

Output data/shots.parquet: frame, shot, cut, hist_d, dx, dy, pan_resp
 - cut: 1 on the first frame of a new shot
 - dx, dy: estimated camera translation (pixels at full res) between frame-1 and frame,
   from phase correlation of downscaled grayscale frames. 0 on cuts.
"""
import argparse
import numpy as np
import cv2
import pandas as pd
from common import VIDEO, SHOTS, frames, video_info

SMALL = (320, 180)


def hist_of(bgr_small):
    hsv = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def main(cut_thr=0.35, min_shot=12):
    fps, n, W, H = video_info(VIDEO)
    sx, sy = W / SMALL[0], H / SMALL[1]
    win = cv2.createHanningWindow(SMALL, cv2.CV_32F)
    rows = []
    prev_hist = prev_gray = None
    shot = 0
    last_cut = -min_shot
    for i, img in frames(VIDEO):
        small = cv2.resize(img, SMALL, interpolation=cv2.INTER_AREA)
        hist = hist_of(small)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        cut, d, dx, dy, resp = 0, 0.0, 0.0, 0.0, 0.0
        if prev_hist is not None:
            # Bhattacharyya distance: 0 identical, 1 disjoint
            d = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
            if d > cut_thr and i - last_cut >= min_shot:
                cut, shot, last_cut = 1, shot + 1, i
            else:
                (dx, dy), resp = cv2.phaseCorrelate(prev_gray, gray, win)
                dx, dy = dx * sx, dy * sy
        rows.append((i, shot, cut, d, dx, dy, resp))
        prev_hist, prev_gray = hist, gray
        if i % 5000 == 0:
            print(f"{i}/{n} shots={shot}", flush=True)
    df = pd.DataFrame(rows, columns=["frame", "shot", "cut", "hist_d", "dx", "dy", "pan_resp"])
    df.to_parquet(SHOTS, index=False)
    print(f"done: {n} frames, {shot + 1} shots, median shot len {df.groupby('shot').size().median():.0f} frames")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut-thr", type=float, default=0.35)
    a = ap.parse_args()
    main(a.cut_thr)
