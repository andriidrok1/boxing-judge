"""Charts for the post (PNG, light surface). Fury = blue, Usyk = orange, fixed.
 out/chart_thrown.png    thrown per round, system, with CompuBox totals for scale
 out/chart_cards.png     round-by-round: three judges + system
 out/chart_pressure.png  who was walking forward, per round (ring control)
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from common import DATA, OUT, PUNCHES, ROUNDS

COL = {"fury": "#2a78d6", "usyk": "#eb6834"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})


def strip(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def thrown():
    sc = pd.read_csv(OUT / "scores.csv")
    cb = pd.read_csv(DATA / "compubox.csv", comment="#")
    tot = cb[cb["round"] == "total"].set_index("fighter").thrown
    a, b = sorted(["fury", "usyk"])            # score.py: A = first sorted name -> thrown_A
    series = {a: sc.thrown_A, b: sc.thrown_B}
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=200)
    x = np.arange(len(sc))
    w = 0.36
    for i, name in enumerate((a, b)):
        xs = x + (i - 0.5) * (w + 0.04)
        ax.bar(xs, series[name], w, color=COL[name], label=name.capitalize(), zorder=2)
        for xi, v in zip(xs, series[name]):
            ax.text(xi, v + 0.8, str(int(v)), ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x, [f"R{r}" for r in sc["round"]])
    ax.set_ylabel("punches thrown (system)")
    strip(ax)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.set_title(f"Punches thrown per round  |  fight totals: system Fury {int(series['fury'].sum())} / Usyk {int(series['usyk'].sum())},  "
                 f"CompuBox Fury {int(tot['fury'])} / Usyk {int(tot['usyk'])}", fontsize=10, color=INK, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "chart_thrown.png"); plt.close(fig)


def cards():
    c = pd.read_csv(DATA / "cards.csv")
    sc = pd.read_csv(OUT / "scores.csv")
    rows = [("Palomo (115-112 Usyk)", c.palomo), ("Fitzgerald (114-113 Usyk)", c.fitzgerald),
            ("Metcalfe (114-113 Fury)", c.metcalfe)]
    a, b = sorted(["fury", "usyk"])
    ta, tb = int(sc[a].sum()), int(sc[b].sum())
    label = f"System ({max(ta, tb)}-{min(ta, tb)} {(a if ta > tb else b).capitalize()})"
    sys_w = sc.winner + np.where(sc.kd.fillna("") != "", "_10-8", "")
    rows.append((label, sys_w))
    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=200)
    for r, (name, ser) in enumerate(rows):
        for k, v in enumerate(ser):
            who = v.replace("_10-8", "")
            ax.add_patch(Rectangle((k + 0.06, r + 0.08), 0.88, 0.84, color=COL[who], lw=0, alpha=0.9 if r < 3 else 1.0))
            ax.text(k + 0.5, r + 0.5, "10-8" if "10-8" in v else who[0].upper(), ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold")
    ax.set_xlim(0, 12); ax.set_ylim(len(rows), 0)
    ax.set_xticks(np.arange(12) + 0.5, [f"R{i+1}" for i in range(12)])
    ax.set_yticks(np.arange(len(rows)) + 0.5, [n for n, _ in rows])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    agree = [(sys_w.str.replace("_10-8", "") == ser.str.replace("_10-8", "")).mean() * 100 for _, ser in rows[:3]]
    ax.set_title("Round winners: three judges vs the system  (F = Fury, U = Usyk)", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("rounds where the system agrees with a judge: " + ", ".join(f"{n.split()[0]} {p:.0f}%" for (n, _), p in zip(rows, agree)),
                  fontsize=9, color=INK2, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "chart_cards.png"); plt.close(fig)


def pressure():
    sc = pd.read_csv(OUT / "scores.csv")
    a, b = sorted(["fury", "usyk"])
    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=200)
    x = np.arange(len(sc))
    v = sc.ring.values
    ax.bar(x, v, 0.6, color=[COL[a] if t > 0 else COL[b] for t in v], zorder=2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x, [f"R{r}" for r in sc["round"]])
    ax.set_ylabel("share of round")
    strip(ax)
    ax.set_title(f"Ring control per round: above zero = {a.capitalize()} walking {b.capitalize()} down, below = the reverse (camera pan compensated)", fontsize=10, color=INK, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "chart_pressure.png"); plt.close(fig)


if __name__ == "__main__":
    thrown(); cards(); pressure()
    print("charts ->", OUT)
