# -*- coding: utf-8 -*-
"""Fig 4a - seasonal stacked-area of daily labor loss by crop (15-yr mean).
Crops stacked by descending annual loss (paddy rice bottom). X spans 1-366,
aligned with fig4b; x labels hidden. NO text annotations, NO y-axis label
(user adds those manually). Y tick numbers + month gridlines kept for alignment.
Data: aggregates_path_a/national_year_crop_doy.csv (sum over male+female)."""
from __future__ import annotations
import os, sys
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _figlib as F

N_YEARS = 15
OUT = F.OUT_ROOT / "fig4a_seasonal_curve"
OUT.mkdir(parents=True, exist_ok=True)
F.setup({"savefig.transparent": False, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.bbox": "standard"})

# stacked bottom -> top, ordered by descending 15-yr annual loss
CROPS = ["paddy rice", "sweet potato", "soybean", "carrot", "buckwheat", "potato", "onion", "wheat", "two-row barley", "six-row barley"]
COLORS = {"paddy rice": "#f2df9a", "sweet potato": "#8c2d2d", "soybean": "#cf5b46",
          "carrot": "#e8915a", "onion": "#e6bf57", "buckwheat": "#5e2626",
          "potato": "#d2694a", "wheat": "#bfa05a", "two-row barley": "#7c5a34",
          "six-row barley": "#3b2a22"}


def main():
    df = F.load_agg("national_year_crop_doy", usecols=["crop_en", "doy_jst", "L"])
    g = df.groupby(["crop_en", "doy_jst"], as_index=False)["L"].sum()
    g["L_M"] = g["L"] / N_YEARS / 1e6
    doy_axis = np.arange(1, 367)
    stacks = []
    for c in CROPS:
        arr = np.zeros(366)
        for _, r in g[g["crop_en"] == c].iterrows():
            d = int(r["doy_jst"])
            if 1 <= d <= 366:
                arr[d - 1] = r["L_M"]
        stacks.append(arr)
    total = np.sum(stacks, axis=0)

    fig, ax = plt.subplots(figsize=(13.0, 2.6))
    fig.subplots_adjust(left=0.14, right=0.985, top=0.97, bottom=0.06)
    ax.stackplot(doy_axis, *stacks, colors=[COLORS[c] for c in CROPS],
                 edgecolor="#7a2a2a", linewidth=0.15)
    ms = F.month_starts()
    ax.set_xticks(ms); ax.set_xticklabels([])
    for d in ms:
        ax.axvline(d, color="#dddddd", lw=0.5, zorder=0)
    ax.set_xlim(1, 366); ax.set_ylim(0, total.max() * 1.10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=10)
    fig.savefig(OUT / "fig4a.png", bbox_inches=None)
    plt.close(fig)
    print("saved fig4a.png  peak=%.2fM/day" % total.max())

    # crop colour legend, separate, same order as the stack
    figl, axl = plt.subplots(figsize=(2.2, 3.2)); figl.patch.set_facecolor("white")
    for k, c in enumerate(CROPS):
        yy = len(CROPS) - 1 - k
        axl.add_patch(plt.Rectangle((0.05, yy + 0.15), 0.5, 0.6, facecolor=COLORS[c],
                                    edgecolor="#7a2a2a", linewidth=0.4))
        axl.text(0.65, yy + 0.45, c, ha="left", va="center", fontsize=10, color=F.INK)
    axl.set_xlim(0, 2.4); axl.set_ylim(0, len(CROPS)); axl.axis("off")
    figl.savefig(OUT / "fig4a_legend.png", bbox_inches="tight")
    plt.close(figl)
    print("saved fig4a_legend.png")


if __name__ == "__main__":
    main()
