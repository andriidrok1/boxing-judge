"""Zero-shot CLIP labels for every tall detection: who is it? (prompts in data/identity.json).
No training, no labels from the tracker. Writes data/clip.parquet: frame, tid, p_<name>...
usage: python classify.py [tracks.parquet] [out.parquet]
"""
import json
import sys
import numpy as np
import pandas as pd
import cv2
import torch
import open_clip
from PIL import Image
from common import VIDEO, TRACKS, DATA, IDENTITY, frames, video_info

CLIP_OUT = DATA / "clip.parquet"
MIN_H_FRAC = 0.15
BATCH = 256


def main(tracks=TRACKS, out=CLIP_OUT):
    fps, n, W, H = video_info(VIDEO)
    ident = json.load(open(IDENTITY))
    names = list(ident["prompts"])
    canvas = ident.get("canvas", [95, 130, 80])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device=dev)
    tok = open_clip.get_tokenizer("ViT-B-32")
    with torch.no_grad():
        tf = model.encode_text(tok([ident["prompts"][k] for k in names]).to(dev))
        tf = tf / tf.norm(dim=-1, keepdim=True)
    df = pd.read_parquet(tracks, columns=["frame", "tid", "x1", "y1", "x2", "y2"])
    df = df[(df.y2 - df.y1) > MIN_H_FRAC * H]
    by_frame = df.groupby("frame")
    rows, imgs, keys = [], [], []

    def flush():
        if not imgs:
            return
        with torch.no_grad(), torch.autocast(dev):
            x = torch.stack([pre(im) for im in imgs]).to(dev)
            f = model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            p = (100 * f @ tf.T).softmax(-1).float().cpu().numpy()
        for k, pr in zip(keys, p):
            rows.append((*k, *pr))
        imgs.clear(); keys.clear()

    want = set(df.frame.unique())
    fmin, fmax = df.frame.min(), df.frame.max()
    for i, img in frames(VIDEO, int(fmin), int(fmax) + 1):
        if i not in want:
            continue
        hsv = None
        for _, r in by_frame.get_group(i).iterrows():
            x1, y1, x2, y2 = int(r.x1), int(r.y1), int(r.x2), int(r.y2)
            imgs.append(Image.fromarray(img[y1:y2, x1:x2, ::-1]))
            # floor patch under the feet: fraction of ring-canvas colour (identity.json "canvas": [h_lo, h_hi, s_min])
            if hsv is None:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            bh = y2 - y1
            fx1, fx2 = x1 + (x2 - x1) // 3, x2 - (x2 - x1) // 3
            fy1, fy2 = max(0, int(y2 - 0.04 * bh)), min(H, int(y2 + 0.04 * bh))
            patch = hsv[fy1:fy2, fx1:fx2].reshape(-1, 3)
            blue = float(((patch[:, 0] >= canvas[0]) & (patch[:, 0] <= canvas[1]) & (patch[:, 1] >= canvas[2])).mean()) if len(patch) else 0.0
            keys.append((i, int(r.tid), blue, y2 >= H - 2))
        if len(imgs) >= BATCH:
            flush()
        if i % 5000 == 0:
            print(f"{i}/{n}", flush=True)
    flush()
    res = pd.DataFrame(rows, columns=["frame", "tid", "floor_canvas", "cut_bottom"] + [f"p_{k}" for k in names])
    res.to_parquet(out, index=False)
    print(res[[f"p_{k}" for k in names]].idxmax(axis=1).value_counts().to_dict(), "->", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
