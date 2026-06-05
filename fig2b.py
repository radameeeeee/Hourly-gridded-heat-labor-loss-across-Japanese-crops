# -*- coding: utf-8 -*-
"""Fig 2b — national daily L curve, ONE CLEAN PNG PER YEAR (2010-2024).

Matches make_daily_curves_4_years.py reference style:
  - narrow strip (for vertical stacking), brick-red area fill + dark line
  - ONLY the 2024 panel carries x-axis month labels (shared axis at bottom)
  - shared y-cap across all 15 panels so years are comparable
  - NO y-label, NO year box, NO header  (user composes the strip manually)

Data: aggregates_path_a/national_year_doy.csv  (sum over male+female).
Out:  figures/main/per_year_subpanels/fig2b_daily_curves/daily_L_{year}.png x15
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("MPLBACKEND", "Agg")
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _figlib as F

YEARS = list(range(2010, 2025))
FILL = "#B85450"      # brick red
LINE = "#5C211F"      # deep wine line
OUT = F.OUT_ROOT / "fig2b_daily_curves"
OUT.mkdir(parents=True, exist_ok=True)
F.setup()


def doy_md(doy, year):
    return (date(year, 1, 1) + timedelta(days=int(doy) - 1)).strftime("%b %d")


def main():
    df = F.load_agg("national_year_doy", usecols=["year_jst", "doy_jst", "sex", "L"])
    g = df.groupby(["year_jst", "doy_jst"], as_index=False)["L"].sum()
    g["L_M"] = g["L"] / 1e6
    Y_CAP = 4.5    # shared cap covering the true max daily peak (2013 = 4.16 M)
    print(f"shared y_cap = {Y_CAP} M/day  (raw max {g['L_M'].max():.2f})")

    for year in YEARS:
        sub = g[g["year_jst"] == year].sort_values("doy_jst")
        doy = sub["doy_jst"].to_numpy()
        val = sub["L_M"].to_numpy()
        val_disp = np.minimum(val, Y_CAP)

        fig, ax = plt.subplots(figsize=(10.0, 1.8))
        ax.fill_between(doy, 0, val_disp, color=FILL, alpha=1.0, zorder=2)
        ax.plot(doy, val_disp, color=LINE, lw=1.3, zorder=3)

        # top-3 separated peaks (skip any spike above the cap = artifact)
        used, labels = [], []
        for i in np.argsort(val)[::-1]:
            if val[i] > Y_CAP:
                continue
            if any(abs(doy[i] - u) < 9 for u in used):
                continue
            used.append(doy[i]); labels.append((int(doy[i]), float(val[i])))
            if len(labels) >= 3:
                break
        for d, v in labels:
            ax.annotate(doy_md(d, year), xy=(d, v), xytext=(d, v + Y_CAP * 0.06),
                        ha="center", va="bottom", fontsize=8, color=F.INK,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cfd2d6", lw=0.6),
                        arrowprops=dict(arrowstyle="-", color=F.INK, lw=0.5))

        # window Jul 1 - Sep 30; only 2024 carries month labels
        jul1 = date(year, 7, 1).timetuple().tm_yday
        sep30 = date(year, 9, 30).timetuple().tm_yday
        ticks = [date(year, m, 1).timetuple().tm_yday for m in (7, 8, 9)] + [sep30]
        labs = ["Jul 1", "Aug 1", "Sep 1", "Sep 30"]
        ax.set_xticks(ticks)
        if year == 2024:
            ax.set_xticklabels(labs, fontsize=9)
            ax.tick_params(axis="x", length=3)
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", length=0)
        ax.set_xlim(jul1, sep30 + 6)
        ax.set_ylim(0, Y_CAP)
        ax.set_yticks([0, 1, 2, 3, 4])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#cfd2d6"); ax.spines["left"].set_color("#cfd2d6")
        ax.grid(False)

        fig.savefig(OUT / f"daily_L_{year}.png", transparent=True,
                    bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"  {year}: annual={val.sum():.1f}M peak@{doy_md(labels[0][0], year)}={labels[0][1]:.2f}M")


if __name__ == "__main__":
    main()
