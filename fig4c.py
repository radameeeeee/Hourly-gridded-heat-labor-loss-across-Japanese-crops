# -*- coding: utf-8 -*-
"""Fig 4c — crop x prefecture choropleth small-multiples, 2024.

Six hero crops; each gets a mini Japan map where prefectures are colored by
that crop's loss rate (share% = L/T x 100).  Prefectures with no production
of that crop are grey.

Data: aggregates_path_a/pref_year_crop.csv  (sum over male+female, 2024).
Polys: aggregated/_assets/jp_prefectures.pkl
Out:   figures/main/per_year_subpanels/fig4c_crop_maps/fig4c_{crop}.png x6 + composite.png
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _figlib as F

HERO = ["paddy rice", "sweet potato", "soybean", "carrot", "buckwheat", "potato"]
DISPLAY = {"paddy rice": "Paddy rice", "carrot": "Carrot", "sweet potato": "Sweet potato",
           "soybean": "Soybean", "onion": "Onion", "buckwheat": "Buckwheat", "potato": "Potato"}
VMAX = 6.0
OUT = F.OUT_ROOT / "fig4c_crop_maps"
OUT.mkdir(parents=True, exist_ok=True)
F.setup({"savefig.transparent": False, "figure.facecolor": "white", "axes.facecolor": "white"})


def main():
    df = F.load_agg("pref_year_crop", usecols=["year_jst", "pref_code", "crop_en", "sex", "T", "L"])
    df = df[df["year_jst"] == 2024]
    g = df.groupby(["crop_en", "pref_code"], as_index=False).agg(T=("T", "sum"), L=("L", "sum"))
    g["share"] = np.where(g["T"] > 0, 100 * g["L"] / g["T"], np.nan)
    rate = {(r.crop_en, int(r.pref_code)): r.share for r in g.itertuples()}

    polys = F.load_polys()
    en, jp = F.pref_lookup()
    cmap = mpl.colormaps["Reds"]
    norm = mpl.colors.Normalize(vmin=0, vmax=VMAX)

    lons, lats = [], []
    for plist in polys.values():
        for poly in plist:
            for p in poly[0]:
                lons.append(p[0]); lats.append(p[1])
    BB = (min(lons) - 0.1, max(lons) + 0.1, min(lats) - 0.1, max(lats) + 0.1)

    def draw(ax, crop):
        for pc in range(1, 48):
            name = jp.get(pc)
            if name not in polys:
                continue
            r = rate.get((crop, pc), np.nan)
            face = F.LAND_FILL if (r is None or not np.isfinite(r) or r <= 0) else cmap(norm(min(r, VMAX)))
            for poly in polys[name]:
                outer = poly[0]
                ax.add_patch(MplPolygon([(p[0], p[1]) for p in outer], closed=True,
                                        facecolor=face, edgecolor="#ffffff", linewidth=0.25))
        ax.set_xlim(BB[0], BB[1]); ax.set_ylim(BB[2], BB[3])
        ax.set_aspect("equal"); ax.axis("off")

    # individual PNGs
    for crop in HERO:
        fig, ax = plt.subplots(figsize=(5.0, 5.0))
        fig.subplots_adjust(left=.01, right=.99, top=.93, bottom=.01)
        draw(ax, crop)
        slug = crop.replace(" ", "_")
        fig.savefig(OUT / f"fig4c_{slug}.png"); plt.close(fig)
        print(f"  saved fig4c_{slug}.png  max share={g[g.crop_en==crop].share.max():.2f}%")

    # composite 3x2
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.0))
    for ax, crop in zip(axes.ravel(), HERO):
        draw(ax, crop)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cax = fig.add_axes([0.30, 0.055, 0.40, 0.022])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="max")
    cb.set_label("Share of labor lost (%)", fontsize=12, fontweight="bold", color=F.INK)
    cb.ax.tick_params(labelsize=10)
    fig.subplots_adjust(left=.01, right=.99, top=.96, bottom=.10, wspace=.02, hspace=.12)
    fig.savefig(OUT / "composite.png"); plt.close(fig)
    print("saved composite.png")


if __name__ == "__main__":
    main()
