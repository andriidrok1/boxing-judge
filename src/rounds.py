"""Round boundaries from the broadcast graphic: the white clock box (lower-left) is on screen only
while a round is running. Samples every STEP frames, segments of presence >= MIN_LEN_S are rounds.
Writes data/rounds.json [{"round", "start", "end"}] (frames) and prints mm:ss. Expect 12.
Region for this broadcast (1280x720): x 60-170, y 632-664 -> identity.json "clock_box": [x1, y1, x2, y2]
"""
import json
import numpy as np
import cv2
from common import VIDEO, ROUNDS, IDENTITY, video_info

STEP = 5
GAP_S = 4
MIN_LEN_S = 150
V_MIN, S_MAX = 170, 60


def main():
    fps, n, W, H = video_info(VIDEO)
    ident = json.load(open(IDENTITY))
    x1, y1, x2, y2 = ident.get("clock_box", [60, 632, 170, 664])
    cap = cv2.VideoCapture(str(VIDEO))
    pres = np.zeros(n // STEP + 1, bool)
    i = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % STEP == 0:
            ok, img = cap.retrieve()
            hsv = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
            pres[i // STEP] = (hsv[..., 2].mean() > V_MIN) and (hsv[..., 1].mean() < S_MAX)
        i += 1
    cap.release()
    gap = int(GAP_S * fps / STEP)
    segs, start, last = [], None, None
    for k in np.flatnonzero(pres):
        if start is None:
            start = last = k
        elif k - last > gap:
            segs.append((start, last)); start = last = k
        else:
            last = k
    if start is not None:
        segs.append((start, last))
    rounds = [{"round": j + 1, "start": int(a * STEP), "end": int((b + 1) * STEP)}
              for j, (a, b) in enumerate(s for s in segs if (s[1] - s[0]) * STEP / fps >= MIN_LEN_S)]
    json.dump(rounds, open(ROUNDS, "w"), indent=1)
    for r in rounds:
        s, e = r["start"] / fps, r["end"] / fps
        print(f"R{r['round']:2d}  {int(s//60):02d}:{int(s%60):02d} - {int(e//60):02d}:{int(e%60):02d}  ({e-s:.0f}s)")
    print(f"{len(rounds)} rounds -> {ROUNDS};  short segments dropped: {sum((s[1]-s[0])*STEP/fps < MIN_LEN_S for s in segs)}")


if __name__ == "__main__":
    main()
