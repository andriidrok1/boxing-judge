"""Shared paths and helpers for boxing-judge."""
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "out"
LABELS = ROOT / "labels"
OUT.mkdir(exist_ok=True)

VIDEO = DATA / "fury_usyk_1_h264.mp4"
SHOTS = DATA / "shots.parquet"
TRACKS = DATA / "tracks.parquet"
FIGHTERS = DATA / "fighters.parquet"
PUNCHES = DATA / "punches.parquet"
ROUNDS = DATA / "rounds.json"
IDENTITY = DATA / "identity.json"

# COCO-17 keypoint indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16
KP = 17


def video_info(path=VIDEO):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, n, w, h


def frames(path=VIDEO, start=0, stop=None, step=1):
    """Yield (frame_idx, bgr) from start to stop (exclusive)."""
    cap = cv2.VideoCapture(str(path))
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    i = start
    while True:
        ok, img = cap.read()
        if not ok or (stop is not None and i >= stop):
            break
        if (i - start) % step == 0:
            yield i, img
        i += 1
    cap.release()
