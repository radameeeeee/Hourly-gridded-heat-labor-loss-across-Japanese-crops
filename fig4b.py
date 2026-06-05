# -*- coding: utf-8 -*-
"""Fig 4b — crop phenology swimlane (full growing-season band + Wakai markers).

One lane per crop.  A continuous band spans the crop's full growing season
(sowing/planting -> harvest, split into two segments for autumn-sown winter
crops that wrap across the new year).  The band is colored at each day by that
crop's 15-yr-pooled daily share% (= L/T x 100).  Phenology stage markers are
overlaid at the Wakai-calendar day-of-year values (verified/corrected table).

Data: aggregates_path_a/national_year_crop_doy.csv  (sum over male+female).
Out:  figures/main/per_year_subpanels/fig4b_crop_swimlane/fig4b.png
                                                         /fig4b_legend.png  (stage key + colorbar)
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _figlib as F

VMAX = 8.0
OUT = F.OUT_ROOT / "fig4b_crop_swimlane"
OUT.mkdir(parents=True, exist_ok=True)
F.setup({"savefig.transparent": False, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.bbox": "standard"})

CROPS = ["paddy rice", "sweet potato", "soybean", "carrot", "buckwheat", "potato", "onion", "wheat", "two-row barley", "six-row barley"]
DISPLAY = {"paddy rice": "Paddy rice", "carrot": "Carrot", "sweet potato": "Sweet potato",
           "soybean": "Soybean", "onion": "Onion", "buckwheat": "Buckwheat",
           "potato": "Potato", "wheat": "Wheat",
           "two-row barley": "Two-row barley", "six-row barley": "Six-row barley"}

# Wakai-calendar stage markers (day-of-year), verified/corrected.
# S sowing, P planting, T transplanting, Sp sprouting, H heading, F flowering, R harvesting
STAGES = {
    "paddy rice":     {"S": 115, "T": 145, "H": 227, "R": 278},
    "carrot":         {"R": 186, "S": 206},
    "sweet potato":   {"R": 286},
    "soybean":        {"S": 166, "F": 222, "R": 304},
    "onion":          {"R": 146, "S": 260, "P": 316},
    "buckwheat":      {"S": 222, "F": 260, "R": 298},
    "potato":         {"P": 90, "Sp": 110, "F": 148, "R": 186},
    "wheat":          {"H": 118, "R": 166, "S": 308},
    "two-row barley": {"H": 105, "R": 150, "S": 319},
    "six-row barley": {"H": 115, "R": 161, "S": 301},
}
# Growing-season band segments (DOY), from Wakai sow/plant -> harvest.
# Winter cereals + onion are autumn-sown / next-year-harvested -> two segments.
ENVELOPE = {
    "paddy rice":     [(115, 278)],
    "carrot":         [(150, 206)],
    "sweet potato":   [(125, 286)],
    "soybean":        [(166, 304)],
    "onion":          [(260, 366), (1, 146)],
    "buckwheat":      [(222, 298)],
    "potato":         [(90, 186)],
    "wheat":          [(308, 366), (1, 166)],
    "two-row barley": [(319, 366), (1, 150)],
    "six-row barley": [(301, 366), (1, 161)],
}
STAGE_NAMES = [("S", "sowing"), ("P", "planting"), ("T", "transplanting"),
               ("Sp", "sprouting"), ("H", "heading"), ("F", "flowering"), ("R", "harvesting")]
LANE_H = 0.72


def main():
    df = F.load_agg("national_year_crop_doy", usecols=["crop_en", "doy_jst", "T", "L"])
    g = df.groupby(["crop_en", "doy_jst"], as_index=False).agg(T=("T", "sum"), L=("L", "sum"))
    g["share"] = np.where(g["T"] > 0, 100 * g["L"] / g["T"], 0.0)
    share = {(r.crop_en, int(r.doy_jst)): r.share for r in g.itertuples()}

    cmap = mpl.colormaps["Reds"]
    norm = mpl.colors.Normalize(vmin=0, vmax=VMAX)
    N = len(CROPS)
    fig, ax = plt.subplots(figsize=(13.0, 0.62 * N + 0.9))
    fig.subplots_adjust(left=0.14, right=0.985, top=0.95, bottom=0.10)

    for i, c in enumerate(CROPS):
        y = N - 1 - i
        ax.add_patch(patches.Rectangle((1, y - LANE_H / 2), 365, LANE_H,
                                       facecolor="#f4f4f4", edgecolor="none", zorder=1))
        for (s0, s1) in ENVELOPE[c]:
            doys = np.arange(int(s0), int(s1) + 1)
            vals = np.array([share.get((c, int(d)), 0.0) for d in doys]).reshape(1, -1)
            ax.imshow(vals, aspect="auto", cmap=cmap, norm=norm,
                      extent=(s0, s1, y - LANE_H / 2, y + LANE_H / 2), zorder=2)
            ax.add_patch(patches.Rectangle((s0, y - LANE_H / 2), s1 - s0, LANE_H,
                                           facecolor="none", edgecolor=F.DARK,
                                           linewidth=0.6, zorder=4))
        for st, doy in STAGES[c].items():
            ax.plot([doy, doy], [y - LANE_H / 2 - 0.05, y + LANE_H / 2 + 0.05],
                    color=F.INK, lw=1.0, zorder=5)
            ax.text(doy, y + LANE_H / 2 + 0.10, st, ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color=F.INK, zorder=6)

    ax.set_yticks([N - 1 - i for i in range(N)])
    ax.set_yticklabels([DISPLAY[c] for c in CROPS], fontsize=12)
    ax.set_ylim(-0.7, N - 0.2); ax.set_xlim(1, 366)
    ms = F.month_starts()
    ax.set_xticks(ms); ax.set_xticklabels(F.MONTH_LABELS, fontsize=11)
    for d in ms:
        ax.axvline(d, color="#dddddd", lw=0.5, zorder=0)
    ax.set_xlabel("Day of year", fontsize=12, fontweight="bold", color=F.INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(F.INK); ax.spines["bottom"].set_color(F.INK)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    fig.savefig(OUT / "fig4b.png", bbox_inches=None)
    plt.close(fig)
    print("saved fig4b.png")

    # ---------- companion legend: stage key + colorbar ----------
    figl = plt.figure(figsize=(3.2, 4.2)); figl.patch.set_facecolor("white")
    figl.text(0.08, 0.96, "Stage markers", fontsize=12, fontweight="bold",
              color=F.INK, ha="left", va="top")
    for k, (sh, nm) in enumerate(STAGE_NAMES):
        figl.text(0.10, 0.88 - k * 0.075, sh, fontsize=11, fontweight="bold", color=F.INK)
        figl.text(0.26, 0.88 - k * 0.075, nm, fontsize=11, color=F.INK)
    cax = figl.add_axes([0.12, 0.06, 0.76, 0.05])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = figl.colorbar(sm, cax=cax, orientation="horizontal", extend="max")
    cb.set_label("Share of labor lost (%)", fontsize=10.5, fontweight="bold", color=F.INK)
    cb.ax.tick_params(labelsize=9)
    figl.savefig(OUT / "fig4b_legend.png")
    plt.close(figl)
    print("saved fig4b_legend.png")
    for c in CROPS:
        mx = max((share.get((c, d), 0) for d in range(1, 367)), default=0)
        print(f"  {c:16} env={ENVELOPE[c]}  max_share={mx:.2f}%")


if __name__ == "__main__":
    main()
