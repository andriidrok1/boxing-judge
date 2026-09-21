"""Precision of data/punches.parquet against labels/round_5_verdicts.csv (verified proposals, thr 2.0).
A detection is judged by the nearest verified proposal of the same fighter within 0.2 s; unmatched = unknown."""
import numpy as np
import pandas as pd
from common import PUNCHES, LABELS

v = pd.read_csv(LABELS / "round_5_verdicts.csv")
import sys
p = pd.read_parquet(sys.argv[1] if len(sys.argv) > 1 else PUNCHES)
p = p[(p.t >= v.t.min() - 1) & (p.t <= v.t.max() + 1)]
res = []
for _, d in p.iterrows():
    c = v[(v.who == d.who) & ((v.t - d.t).abs() <= 0.2)]
    res.append(c.iloc[(c.t - d.t).abs().argmin()].verdict if len(c) else "unknown")
p = p.assign(verdict=res)
k = p[~p.verdict.isin(["unknown", "unclear"])]
print(f"round 5 detections: {len(p)}; judged: {len(k)}; precision {(k.verdict == 'tp').mean():.2f}; "
      f"tp {int((k.verdict=='tp').sum())} / identity {int((k.verdict=='identity').sum())} / clinch {int((k.verdict=='clinch').sum())} / "
      f"cut {int((k.verdict=='cut').sum())} / other fp {int((k.verdict=='fp').sum())}; unknown {int((p.verdict=='unknown').sum())}, unclear {int((p.verdict=='unclear').sum())}")
print(f"verified punches recovered: {int((k.verdict=='tp').sum())} / {int((v.verdict=='tp').sum())}")
