"""YOLO11-pose per frame (no tracker: ByteTrack drops fighters on fast motion; identity is done in fighters.py).

Output data/tracks.parquet, one row per detection:
 frame, shot, tid, conf, x1,y1,x2,y2, k{i}x,k{i}y,k{i}c (i=0..16),
 torso_h, torso_s, torso_v, torso_skin (fraction of skin-like pixels on the torso patch),
 trunk_h, trunk_s, trunk_v (mean HSV of the trunks patch, hips->knees)
"""
import argparse
import numpy as np
import cv2
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from ultralytics import YOLO
from common import (VIDEO, SHOTS, TRACKS, ROOT, frames, video_info,
                    L_SHO, R_SHO, L_ELB, R_ELB, L_HIP, R_HIP, L_KNEE, R_KNEE, L_WRI, R_WRI, KP)

COLS = (["frame", "shot", "tid", "conf", "x1", "y1", "x2", "y2"]
        + [f"k{i}{a}" for i in range(KP) for a in "xyc"]
        + ["torso_h", "torso_s", "torso_v", "torso_skin", "trunk_h", "trunk_s", "trunk_v", "glove_h", "glove_s", "glove_v"])


def patch_stats(hsv, pts, W, H, shrink=0.25):
    """Mean HSV + skin fraction inside the bbox of pts, shrunk toward its center."""
    pts = np.asarray(pts, dtype=np.float32)
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    w, h = x1 - x0, y1 - y0
    x0, x1 = int(max(0, x0 + w * shrink)), int(min(W, x1 - w * shrink))
    y0, y1 = int(max(0, y0 + h * shrink)), int(min(H, y1 - h * shrink))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return np.nan, np.nan, np.nan, np.nan
    p = hsv[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    hue, sat, val = p[:, 0], p[:, 1], p[:, 2]
    skin = ((hue <= 25) | (hue >= 170)) & (sat >= 30) & (sat <= 180) & (val >= 60)
    # circular mean for hue (OpenCV hue in [0,180))
    ang = hue / 180 * 2 * np.pi
    mh = (np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) % (2 * np.pi)) / (2 * np.pi) * 180
    return float(mh), float(sat.mean()), float(val.mean()), float(skin.mean())


def main(start, stop, imgsz, out):
    fps, n, W, H = video_info(VIDEO)
    shots = pd.read_parquet(SHOTS).set_index("frame")["shot"]
    model = YOLO(str(ROOT / "yolo11m-pose.pt"))
    writer = pq.ParquetWriter(str(out), pa.schema([(c, pa.float32() if c not in ("frame", "shot", "tid") else pa.int32()) for c in COLS]))
    buf, prev_shot = [], None
    for i, img in frames(VIDEO, start, stop):
        shot = int(shots.get(i, 0))
        res = model.predict(img, classes=[0], imgsz=imgsz, half=True, verbose=False, conf=0.3)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        hsv = None
        ids = np.arange(len(res.boxes))  # per-frame index only
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        kps = res.keypoints.data.cpu().numpy()  # (N,17,3)
        for b, c, tid, k in zip(boxes, confs, ids, kps):
            if hsv is None:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            torso = trunk = glove = (np.nan,) * 4
            shos = k[[L_SHO, R_SHO], :2]
            sw = np.linalg.norm(shos[0] - shos[1])
            if min(k[[L_SHO, R_SHO], 2]) > 0.3 and sw > 4:
                # shoulder band: thin strip just below the shoulder line (bare on fighters, sleeve/shirt on the ref)
                band = [(shos[0, 0], shos[0, 1] + 0.05 * sw), (shos[1, 0], shos[1, 1] + 0.05 * sw),
                        (shos[0, 0], shos[0, 1] + 0.35 * sw), (shos[1, 0], shos[1, 1] + 0.35 * sw)]
                torso = patch_stats(hsv, band, W, H, shrink=0.1)
                if min(k[[L_HIP, R_HIP], 2]) > 0.3:
                    full = patch_stats(hsv, k[[L_SHO, R_SHO, L_HIP, R_HIP], :2], W, H)
                    if not np.isnan(full[3]):
                        torso = (*full[:3], max(full[3], torso[3]))
                gl = []
                for el, wr in ((L_ELB, L_WRI), (R_ELB, R_WRI)):
                    if k[wr, 2] > 0.3 and k[el, 2] > 0.3:
                        # glove centre: past the wrist along the forearm direction
                        fore = k[wr, :2] - k[el, :2]
                        cx, cy = k[wr, :2] + 0.35 * fore
                        r = 0.3 * sw
                        st = patch_stats(hsv, [(cx - r, cy - r), (cx + r, cy + r)], W, H, shrink=0.15)
                        if not np.isnan(st[0]):
                            gl.append(st[:3])
                if gl:
                    gl = np.array(gl)
                    ang = np.deg2rad(gl[:, 0] * 2)
                    mh = np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) % (2 * np.pi)) / 2
                    glove = (float(mh), float(gl[:, 1].mean()), float(gl[:, 2].mean()), np.nan)
            if min(k[[L_HIP, R_HIP], 2]) > 0.3:
                # trunks patch: from the hips down by 0.6 torso lengths, hip width wide (knees often out of frame)
                hips = k[[L_HIP, R_HIP], :2]
                shos = k[[L_SHO, R_SHO], :2]
                tl = np.linalg.norm(shos.mean(0) - hips.mean(0))
                hw = max(abs(hips[0, 0] - hips[1, 0]), 0.25 * tl)
                cx, top = hips.mean(0)
                quad = [(cx - hw, top), (cx + hw, top), (cx - hw, top + 0.6 * tl), (cx + hw, top + 0.6 * tl)]
                trunk = patch_stats(hsv, quad, W, H, shrink=0.2)
            buf.append([i, shot, int(tid), float(c), *b.tolist(), *k.flatten().tolist(),
                        *torso, *trunk[:3], *glove[:3]])
        if len(buf) >= 2000:
            writer.write_table(pa.Table.from_pandas(pd.DataFrame(buf, columns=COLS), preserve_index=False).cast(writer.schema))
            buf = []
        if i % 2000 == 0:
            print(f"{i}/{n}", flush=True)
    if buf:
        writer.write_table(pa.Table.from_pandas(pd.DataFrame(buf, columns=COLS), preserve_index=False).cast(writer.schema))
    writer.close()
    print("done ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default=str(TRACKS))
    a = ap.parse_args()
    main(a.start, a.stop, a.imgsz, a.out)
