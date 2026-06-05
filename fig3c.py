# -*- coding: utf-8 -*-
"""Fig 3c — 2024 amount-vs-rate scatter, single clean panel.

Clean: no axis labels, no point labels.  Full box frame, large bold Arial
tick numbers, integer ticks + grid, transparent.  Bubble area ~ farm hours T;
power-law fit (log-log) + 1-sigma band.

Data: aggregates_path_a/pref_year_doy.csv  (sum over male+female, 2024).
Out:  figures/main/per_year_subpanels/fig3c_scatter/fig3c_2024.png
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _figlib as F

SIZE_SCALE = 12
OUT = F.OUT_ROOT / "fig3c_scatter"
OUT.mkdir(parents=True, exist_ok=True)
F.setup({"font.family": ["Arial", "Liberation Sans", "DejaVu Sans"]})


def main():
    df = F.load_agg("pref_year_doy", usecols=["year_jst", "pref_code", "sex", "T", "L"])
    df = df[df["year_jst"] == 2024]
    g = df.groupby("pref_code", as_index=False).agg(T=("T", "sum"), L=("L", "sum"))
    g["L_M"] = g["L"] / 1e6; g["T_M"] = g["T"] / 1e6; g["rate"] = 100 * g["L"] / g["T"]
    xs = g["L_M"].to_numpy(); ys = g["rate"].to_numpy(); sizes = g["T_M"].to_numpy() * SIZE_SCALE

    m = xs > 0.01
    lx = np.log(xs[m]); ly = np.log(np.maximum(ys[m], 0.05))
    b, a = np.polyfit(lx, ly, 1)
    sd = np.std(ly - (a + b * lx))
    xg = np.linspace(0.05, xs.max() * 1.05, 200); lp = a + b * np.log(xg)

    fig, ax = plt.subplots(figsize=(15.0, 4.2))
    ax.fill_between(xg, np.exp(lp - sd), np.exp(lp + sd), color=F.BRICK, alpha=0.15,
                    edgecolor="none", zorder=1)
    ax.plot(xg, np.exp(lp), color=F.DARK, lw=2.0, alpha=0.85, zorder=2)
    ax.scatter(xs, ys, s=sizes, c=F.BRICK, alpha=0.65, edgecolor=F.DARK,
               linewidth=1.5, zorder=3)

    ax.set_xlim(-0.15, 5.5); ax.set_ylim(-0.2, 6.8)        # x stops at 5 + a little
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
    # full box frame
    for s in ("top", "bottom", "left", "right"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_color(F.INK)
        ax.spines[s].set_linewidth(1.2)
    ax.grid(True, color="#dcdcdc", lw=0.7, zorder=0)
    ax.tick_params(labelsize=18, width=1.2, length=5, colors=F.INK)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    # no axis labels (clean)
    fig.savefig(OUT / "fig3c_2024.png", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"saved fig3c_2024.png  data xmax={xs.max():.2f} ymax={ys.max():.2f}  fit b={b:.2f}")


if __name__ == "__main__":
    main()
