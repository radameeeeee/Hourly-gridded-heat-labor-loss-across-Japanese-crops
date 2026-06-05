# -*- coding: utf-8 -*-
"""Fig 3a — TWO clean component sets:

(1) prefecture choropleth, ONE PNG PER YEAR (2010-2024):
    47 prefectures colored by share% = L/T x 100, RdYlBu_r, vmin0 vmax6,
    white prefecture borders, transparent, no text.  Source: pref_year_doy.

(2) within-prefecture grid zoom, ONE PNG per top-5 prefecture (2024):
    Ibaraki(8) Saitama(11) Saga(41) Fukuoka(40) Aichi(23) — smooth grid raster
    clipped to the prefecture, RdYlBu_r vmax6, black outline.  Source: year_pref_grid.

Out: figures/main/per_year_subpanels/fig3a_choropleth_with_zoom/
       pref_map_{year}.png x15
       zoom_{en}.png       x5
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, PathPatch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _figlib as F

YEARS = list(range(2010, 2025))
TOP5 = [(8, "Ibaraki"), (11, "Saitama"), (41, "Saga"), (40, "Fukuoka"), (23, "Aichi")]
VMIN, VMAX = 0.0, 6.0
CMAP = "RdYlBu_r"
OUT = F.OUT_ROOT / "fig3a_choropleth_with_zoom"
OUT.mkdir(parents=True, exist_ok=True)
F.setup()


def main():
    polys = F.load_polys()
    _, jp = F.pref_lookup()
    BB = F.poly_bounds([p for nm in polys for p in polys[nm]])
    cmap = mpl.colormaps[CMAP].copy()
    cmap.set_bad("none", alpha=0.0)
    norm = mpl.colors.Normalize(vmin=VMIN, vmax=VMAX, clip=True)

    mask, mextent = F.land_mask(polys)

    # ---------- (1) prefecture choropleth per year ----------
    pdy = F.load_agg("pref_year_doy", usecols=["year_jst", "pref_code", "sex", "T", "L"])
    for year in YEARS:
        d = pdy[pdy["year_jst"] == year]
        g = d.groupby("pref_code", as_index=False).agg(T=("T", "sum"), L=("L", "sum"))
        g["share"] = 100 * g["L"] / g["T"]
        share = dict(zip(g["pref_code"].astype(int), g["share"]))

        fig, ax = plt.subplots(figsize=(14.0 / 2.54, 17.5 / 2.54), dpi=300)
        ax.axis("off")
        for pc in range(1, 48):
            nm = jp.get(pc)
            if nm not in polys:
                continue
            r = share.get(pc, np.nan)
            face = F.LAND_FILL if not np.isfinite(r) else cmap(norm(min(r, VMAX)))
            for poly in polys[nm]:
                ax.add_patch(MplPolygon([(p[0], p[1]) for p in poly[0]], closed=True,
                                        facecolor=face, edgecolor="#ffffff",
                                        linewidth=0.4, zorder=2))
        F.draw_coastline(ax, mask, mextent, color="#202124", lw=0.6, zorder=6)
        ax.set_xlim(BB[0] - 0.3, BB[2] + 0.3)
        ax.set_ylim(BB[1] - 0.3, BB[3] + 0.3)
        ax.set_aspect("equal", adjustable="datalim")
        fig.savefig(OUT / f"pref_map_{year}.png", transparent=True,
                    bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        nat = 100 * g["L"].sum() / g["T"].sum()
        print(f"  pref_map_{year}: national share={nat:.2f}%")

    # ---------- (2) within-prefecture zoom (2024) ----------
    gd = F.add_lonlat(F.load_agg("year_pref_grid",
                                 usecols=["year_jst", "pref_code", "grid_id", "sex", "T", "L"]))
    gd = gd[gd["year_jst"] == 2024]
    pdy24 = pdy[pdy["year_jst"] == 2024].groupby("pref_code", as_index=False).agg(
        T=("T", "sum"), L=("L", "sum"))
    pshare = {int(r.pref_code): 100 * r.L / r.T for r in pdy24.itertuples()}

    for pc, name_en in TOP5:
        sub = gd[gd["pref_code"] == pc].copy()
        sub = sub.groupby(["grid_id", "grid_lat", "grid_lon"], as_index=False).agg(
            T=("T", "sum"), L=("L", "sum"))
        sub["rate"] = np.clip(100 * sub["L"] / sub["T"], VMIN, VMAX)
        nm = jp.get(pc)
        plist = polys.get(nm, [])
        mnx, mny, mxx, mxy = F.poly_bounds(plist)
        raster, extent = F.build_smooth_raster(sub, (mnx, mny, mxx, mxy),
                                               upsample=10, sigma=1.4)
        fig, ax = plt.subplots(figsize=(5.0, 5.2), dpi=300)
        ax.axis("off")
        im = ax.imshow(np.ma.masked_invalid(raster), extent=extent, origin="lower",
                       cmap=cmap, norm=norm, interpolation="bilinear", zorder=2)
        path = F.polys_to_mpl_path(plist)
        clip = PathPatch(path, transform=ax.transData, facecolor="none", edgecolor="none")
        ax.add_patch(clip); im.set_clip_path(clip)
        for poly in plist:
            xy = poly[0]
            ax.plot([p[0] for p in xy], [p[1] for p in xy], color=F.INK, lw=0.8, zorder=5)
        px = (mxx - mnx) * 0.06; py = (mxy - mny) * 0.06
        ax.set_xlim(mnx - px, mxx + px); ax.set_ylim(mny - py, mxy + py)
        ax.set_aspect("equal", adjustable="datalim")
        fig.savefig(OUT / f"zoom_{name_en}.png", transparent=True,
                    bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"  zoom_{name_en}: pref share={pshare.get(pc, float('nan')):.2f}%  cells={len(sub)}")

    # ---------- shared legend ----------
    fig, ax = plt.subplots(figsize=(6.0, 1.0))
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=ax, orientation="horizontal", extend="max")
    cb.set_label("Share of labor lost (%)", fontsize=12, fontweight="bold", color=F.INK)
    cb.ax.tick_params(labelsize=10)
    fig.savefig(OUT / "legend.png", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print("done: 15 pref maps + 5 zooms + legend")


if __name__ == "__main__":
    main()
