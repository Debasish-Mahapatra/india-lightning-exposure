#!/usr/bin/env python3
"""Supplementary Figure S3 (revision 7; the detection map, the regional values
and the peak-current scatter moved to main Fig. 3, fig3_detection.py).
a: satellite-expected CG flash density, 2022-2023: ISS LIS total lightning
   (0.5 degree sums of flashes over view time) converted with the IITM ratios
   of cloud to CG flashes by region and season (Ghosh et al. 2023).
b: NRSC flashes per satellite-expected CG flash by State/UT (larger than
   20,000 km2) against records (strokes) per flash, 2022-2025.
Inputs: data/vulnerability/detection_025.nc, results/detection_state.csv
Output: manuscript/supplementary/figS3_detection.png/.pdf
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

from common import DATA, REGIONS, RESULTS, india_area
from plotting import BLUE, GRID, INK2, MM, PC, REGION_COLOR, classes, colorbar, field, india_map, label, save

TO_REGION = {s: r for r, ss in REGIONS.items() for s in ss}
A = india_area()
d = xr.open_dataset(DATA / "vulnerability/detection_025.nc")
sat = d.iss_cg_2022_2023.values
st = pd.read_csv(RESULTS / "detection_state.csv", index_col=0)
st["region"] = [TO_REGION.get(s, "") for s in st.index]
st = st[(st.area_km2 > 2e4) & (st.region != "")]
fig = plt.figure(figsize=(183 * MM, 85 * MM))

ax = fig.add_axes([0.0, 0.1, 0.5, 0.88], projection=PC)
india_map(ax, ticks=False)
cmap, norm = classes([0.25, 0.5, 1, 2, 3, 4, 6, 8, 10, 12], BLUE, extend="both")
m = field(ax, sat, cmap, norm)
colorbar(fig, m, fig.add_axes([0.06, 0.07, 0.38, 0.025]),
         "Satellite-expected CG flashes per km$^2$ per year (2022\u20132023)", fmt="{x:g}")
label(ax, "a", x=0.07, y=0.92)

bx = fig.add_axes([0.62, 0.2, 0.34, 0.7])
for r, g in st.groupby("region"):
    bx.scatter(g.records_per_flash, g.de, s=16, color=REGION_COLOR[r], edgecolor="white", lw=0.4, zorder=3)
rho = stats.spearmanr(st.records_per_flash, st.de)
bx.text(0.03, 0.97, f"Rank correlation {rho.statistic:+.2f}\n({len(st)} States/UTs)", transform=bx.transAxes, va="top",
        ha="left", fontsize=6, color=INK2)
bx.set_xlabel("Records per flash (NRSC, 2022\u20132025)")
bx.set_ylabel("NRSC flashes per\nsatellite-expected CG flash")
bx.set_ylim(0, 1.0)
bx.grid(color=GRID, lw=0.5)
bx.set_axisbelow(True)
bx.legend(handles=[Line2D([], [], marker="o", ls="", color=REGION_COLOR[r], markersize=4, label=r) for r in REGIONS],
          loc="lower right", fontsize=5.6, frameon=False, handletextpad=0.2, labelspacing=0.3)
label(bx, "b", x=-0.2, y=1.02)
print(f"detection against records per flash across {len(st)} States/UTs: rho {rho.statistic:+.2f}")
print(save(fig, "figS3_detection", sup=True))
