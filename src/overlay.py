"""Render a clip with skeletons, fighter names, thrown counters and advance/retreat arrows.
usage: python overlay.py --start-sec 1710 --dur 25 --out out/clip.mp4
"""
import argparse
import numpy as np
import cv2
import pandas as pd
import supervision as sv
from common import VIDEO, FIGHTERS, PUNCHES, DATA, OUT, frames, video_info, KP

RING = DATA / "ring.parquet"
from pathlib import Path
INSET, INSET_S = 220, 1.5   # punch photo size (px) and how long it stays on screen (s)
COLORS = {"A": (60, 200, 255), "B": (255, 120, 60), "ref": (200, 200, 200)}
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def main(start_sec, dur, out, verified=None, timecode=False):
    fps, n, W, H = video_info(VIDEO)
    s, e = int(start_sec * fps), int((start_sec + dur) * fps)
    f = pd.read_parquet(FIGHTERS)
    f = f[(f.frame >= s) & (f.frame < e)]
    if verified:
        v = pd.read_csv(verified)
        v = v[v.verdict == "tp"].copy()
        v["frame"] = (v.t * fps).round().astype(int)
        p = v[["frame", "who", "hand"]]
    else:
        p = pd.read_parquet(PUNCHES)
    p = p[(p.frame >= s) & (p.frame < e)]
    names = sorted(w for w in f.who.unique() if w != "ref")
    col = {names[0]: COLORS["A"], names[1]: COLORS["B"], "ref": COLORS["ref"]} if len(names) == 2 else COLORS
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    count = {w: 0 for w in names}
    flash = {}
    insets = []          # (until_frame, thumbnail, label)
    sheet = []
    for i, img in frames(VIDEO, s, e):
        raw = img.copy()
        for _, r in f[f.frame == i].iterrows():
            c = col.get(r.who, (255, 255, 255))
            k = np.array([[r[f"k{j}x"], r[f"k{j}y"], r[f"k{j}c"]] for j in range(KP)])
            for a, b in SKELETON:
                if k[a, 2] > 0.3 and k[b, 2] > 0.3:
                    cv2.line(img, tuple(k[a, :2].astype(int)), tuple(k[b, :2].astype(int)), c, 2, cv2.LINE_AA)
            # box from the skeleton, not the YOLO box (which spans both men in a clinch)
            vis = k[k[:, 2] > 0.3]
            if len(vis) >= 4:
                bx1, by1 = vis[:, :2].min(0); bx2, by2 = vis[:, :2].max(0)
                pad = 0.08 * (by2 - by1)
                bx1, by1, bx2, by2 = int(bx1 - pad), int(by1 - 2 * pad), int(bx2 + pad), int(by2 + pad)
            else:
                bx1, by1, bx2, by2 = int(r.x1), int(r.y1), int(r.x2), int(r.y2)
            cv2.rectangle(img, (bx1, by1), (bx2, by2), c, 2)
            cv2.putText(img, r.who.upper(), (bx1, by1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2, cv2.LINE_AA)
        for _, q in p[p.frame == i].iterrows():
            count[q.who] += 1
            flash[q.who] = i
            # photo of the punch: crop around both fighters at this frame, shown as an inset for INSET_S seconds
            b = f[(f.frame == i) & (f.who != "ref")]
            if len(b):
                x1, y1, x2, y2 = int(b.x1.min()), int(b.y1.min()), int(b.x2.max()), int(b.y2.max())
                side = int(max(x2 - x1, y2 - y1) * 1.1); cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                x1, y1 = max(0, cx - side // 2), max(0, cy - side // 2); crop = raw[y1:min(H, y1 + side), x1:min(W, x1 + side)]
            else:
                crop = raw
            th = cv2.resize(crop, (INSET, INSET))
            label = f"{q.who.upper()} {q.hand}  #{count[q.who]}" + (f"  {i / fps:.2f}s" if timecode else "")
            insets.append((i + int(INSET_S * fps), th, label, col[q.who]))
            sh = th.copy(); cv2.putText(sh, label, (6, INSET - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(sh, label, (6, INSET - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col[q.who], 2, cv2.LINE_AA); sheet.append(sh)
        insets = [x for x in insets if x[0] > i][-3:]
        for k, (until, th, label, c) in enumerate(reversed(insets)):
            x0 = W - (INSET + 16) * (k + 1); y0 = H - INSET - 16
            img[y0:y0 + INSET, x0:x0 + INSET] = th
            cv2.rectangle(img, (x0, y0), (x0 + INSET, y0 + INSET), c, 3)
            cv2.putText(img, label, (x0 + 4, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, label, (x0 + 4, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
        # timecode: video time and clip time, top right
        if timecode:
            tc = f"{i / fps:.2f}s   clip {int((i - s) / fps) // 60:02d}:{(i - s) / fps % 60:05.2f}"
            (tw, th_), _ = cv2.getTextSize(tc, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.putText(img, tc, (W - tw - 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(img, tc, (W - tw - 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        y = 40
        for w in names:
            hot = (i - flash.get(w, -99)) < 6
            cv2.putText(img, f"{w.upper()} thrown: {count[w]}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 255, 255) if hot else col[w], 3 if hot else 2, cv2.LINE_AA)
            y += 36
        vw.write(img)
    vw.release()
    if sheet:
        while len(sheet) % 5: sheet.append(np.zeros((INSET, INSET, 3), np.uint8))
        grid = np.vstack([np.hstack(sheet[j:j + 5]) for j in range(0, len(sheet), 5)])
        cv2.imwrite(str(Path(out).with_suffix("")) + "_punches.jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-sec", type=float, required=True)
    ap.add_argument("--dur", type=float, default=25)
    ap.add_argument("--out", default=str(OUT / "clip.mp4"))
    ap.add_argument("--verified", default=None, help="labels/clip_<start>_verdicts.csv: draw only hand-verified punches")
    ap.add_argument("--timecode", action="store_true", help="print video time top right (for reviewing)")
    a = ap.parse_args()
    main(a.start_sec, a.dur, a.out, a.verified, a.timecode)
