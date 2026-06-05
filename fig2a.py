# -*- coding: utf-8 -*-
"""Fig 2a — nationwide 0.1-deg grid loss-rate map, ONE CLEAN PNG PER YEAR.

Matches the reference style of make_15yr_grid_loss_rate_clean.py:
  - RdYlBu_r colormap, vmin=0 vmax=6 (%), pixel raster (nearest, no smoothing)
  - grey landmass underneath, prefecture/coast boundary lines on top
  - transparent background, NO title / legend / text  (user composes manually)

Years: every year 2010-2024 (15 PNGs).
Data:  aggregates_path_a/year_grid.csv  (sum over male+female).
Polys: aggregated/_assets/jp_prefectures.pkl
Out:   figures/main/per_year_subpanels/fig2a_grid_raster/loss_rate_{year}.png x15 + legend.png
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
VMIN, VMAX = 0.0, 6.0
CMAP = "RdYlBu_r"
BORDER = "#444444"
OUT = F.OUT_ROOT / "fig2a_grid_raster"
OUT.mkdir(parents=True, exist_ok=True)
F.setup()


def grid_to_raster_nearest(g, cell=0.1):
    """Regular 0.1-deg raster, nearest (no smoothing). g has grid_lat/lon/rate."""
    lat = g["grid_lat"].round(4).to_numpy()
    lon = g["grid_lon"].round(4).to_numpy()
    val = g["rate"].to_numpy()
    lat_min, lat_max = lat.min(), lat.max()
    lon_min, lon_max = lon.min(), lon.max()
    nlat = int(round((lat_max - lat_min) / cell)) + 1
    nlon = int(round((lon_max - lon_min) / cell)) + 1
    raster = np.full((nlat, nlon), np.nan)
    for la, lo, v in zip(lat, lon, val):
        i = int(round((la - lat_min) / cell)); j = int(round((lo - lon_min) / cell))
        if 0 <= i < nlat and 0 <= j < nlon:
            raster[i, j] = v
    half = cell / 2
    extent = (lon_min - half, lon_max + half, lat_min - half, lat_max + half)
    return raster, extent


def main():
    df = F.add_lonlat(F.load_agg("year_grid", usecols=["year_jst", "grid_id", "sex", "T", "L"]))
    polys = F.load_polys()
    _, jp = F.pref_lookup()
    all_polys = [p for nm in polys for p in polys[nm]]
    land = F.polys_to_mpl_path(all_polys)
    BB = F.poly_bounds(all_polys)

    cmap = mpl.colormaps[CMAP].copy()
    cmap.set_bad("none", alpha=0.0)
    norm = mpl.colors.Normalize(vmin=VMIN, vmax=VMAX, clip=True)

    for year in YEARS:
        sub = df[df["year_jst"] == year]
        g = sub.groupby(["grid_id", "grid_lat", "grid_lon"], as_index=False).agg(
            T=("T", "sum"), L=("L", "sum"))
        g = g[g["T"] > 0].copy()
        g["rate"] = np.clip(100 * g["L"] / g["T"], VMIN, VMAX)
        raster, extent = F.build_smooth_raster(
            g, (BB[0], BB[1], BB[2], BB[3]), upsample=8, sigma=1.3)

        fig, ax = plt.subplots(figsize=(14.0 / 2.54, 17.5 / 2.54), dpi=300)
        ax.axis("off")
        # grey landmass
        for nm in polys:
            for poly in polys[nm]:
                ax.add_patch(MplPolygon([(p[0], p[1]) for p in poly[0]], closed=True,
                                        facecolor=F.LAND_FILL, edgecolor="none", zorder=1))
        # raster, clipped to land
        im = ax.imshow(np.ma.array(raster, mask=~np.isfinite(raster)), extent=extent,
                       origin="lower", cmap=cmap, norm=norm, interpolation="bilinear",
                       zorder=2, rasterized=True)
        clip = PathPatch(land, transform=ax.transData, facecolor="none", edgecolor="none")
        ax.add_patch(clip); im.set_clip_path(clip)
        # boundary lines (prefecture + coastline, single colour)
        for nm in polys:
            for poly in polys[nm]:
                xy = poly[0]
                ax.plot([p[0] for p in xy], [p[1] for p in xy], color=BORDER,
                        lw=0.3, zorder=3, solid_capstyle="round")
        ax.set_xlim(BB[0] - 0.3, BB[2] + 0.3)
        ax.set_ylim(BB[1] - 0.3, BB[3] + 0.3)
        ax.set_aspect("equal", adjustable="datalim")
        fig.savefig(OUT / f"loss_rate_{year}.png", transparent=True,
                    bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        nat = 100 * g["L"].sum() / g["T"].sum()
        print(f"  {year}: national share={nat:.2f}%  cells={len(g)}  max_rate={g.rate.max():.2f}%")

    # standalone horizontal legend
    fig, ax = plt.subplots(figsize=(6.0, 1.0))
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=ax, orientation="horizontal", extend="max")
    cb.set_label("Share of labor lost (%)", fontsize=12, fontweight="bold", color=F.INK)
    cb.ax.tick_params(labelsize=10)
    fig.savefig(OUT / "legend.png", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print("saved legend.png + 15 yearly maps")


if __name__ == "__main__":
    main()
