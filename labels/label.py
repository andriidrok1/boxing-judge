"""Hand-label punches thrown while watching a round. Ground truth independent of the tracker.
usage: python labels/label.py ROUND            (reads data/rounds.json for the frame range)
keys: f = Fury throws, u = Usyk throws, space = pause, a/d = -/+ 1 s, q = save & quit
writes labels/round_ROUND.csv with columns t (seconds in video), who
"""
import json
import sys
import time
import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rd = int(sys.argv[1])
rounds = {r["round"]: r for r in json.load(open(ROOT / "data/rounds.json"))}
s, e = rounds[rd]["start"], rounds[rd]["end"]
cap = cv2.VideoCapture(str(ROOT / "data/fury_usyk_1_h264.mp4"))
fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES, s)
out = ROOT / f"labels/round_{rd}.csv"
rows = []
paused = False
i = s
while i < e:
    if not paused:
        ok, img = cap.read()
        if not ok:
            break
        i += 1
    cv2.putText(img, f"R{rd} t={i/fps:7.2f}  fury={sum(r[1]=='fury' for r in rows)} usyk={sum(r[1]=='usyk' for r in rows)}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.imshow("label", img)
    k = cv2.waitKey(int(1000 / fps) if not paused else 30) & 0xFF
    if k == ord("q"):
        break
    elif k == ord(" "):
        paused = not paused
    elif k == ord("f"):
        rows.append((round(i / fps, 2), "fury"))
    elif k == ord("u"):
        rows.append((round(i / fps, 2), "usyk"))
    elif k in (ord("a"), ord("d")):
        i = max(s, i + (int(fps) if k == ord("d") else -int(fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
with open(out, "w") as f:
    f.write("t,who\n")
    for t, w in sorted(rows):
        f.write(f"{t},{w}\n")
print(f"saved {len(rows)} punches -> {out}")
