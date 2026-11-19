#!/usr/bin/env python3
"""Check: the annual flash density on the 0.25 degree grid of the analyses and
on the 0.1 degree grid of Fig. 1a, side by side on the colour scale of Fig. 1a,
and how many flashes the cells hold (2022-2025, cells at least half inside
India's official boundary).
Output: results/check_fig1_grid.png; printed (results/log_check_fig1_grid.txt)
"""
import numpy as np
import matplotlib.pyplot as plt

from common import (LAT, LAT01, LON, LON01, RESULTS, YEARS, cell_area, cell_area_01, flashes_01, india_area,
                    india_area_01, nrsc_clim)
from plotting import MM, PC, BLUE, classes, colorbar, field, india_map, label

A25, A01 = india_area(), india_area_01()
f25 = nrsc_clim("flashes").sum(("month", "hour_utc")).values
f01 = flashes_01().sum(0)
d25, d01 = f25 / A25, f01 / A01
for name, f, a, full in (("0.25 degree", f25, A25, cell_area()), ("0.1 degree", f01, A01, cell_area_01())):
    k = np.isfinite(a) & (a >= 0.5 * full)
    n = np.nan_to_num(f[k]) * len(YEARS)                       # flashes in 2022-2025
    print(f"{name}: {np.nansum(f) / 1e6:.2f} million flashes a year, {np.nansum(f) / np.nansum(a):.2f} per km2; "
          f"highest cell {np.nanmax(np.where(k, f / a, np.nan)):.1f}; median cell {np.median(a[k]):.0f} km2 with "
          f"{np.median(n):.0f} flashes in 2022-2025; {100 * np.mean(n < 10):.1f}% of cells with fewer than 10, "
          f"{100 * np.mean(n < 100):.1f}% with fewer than 100")

fig = plt.figure(figsize=(183 * MM, 100 * MM))
cmap, norm = classes([0.25, 0.5, 1, 2, 3, 4, 6, 8, 10, 12], BLUE, extend="both")
for k, (data, lat, lon, title) in enumerate(((d25, LAT, LON, "0.25° grid (cells about 27 km across)"),
                                             (d01, LAT01, LON01, "0.1° grid (cells about 11 km across)"))):
    ax = fig.add_axes([0.01 + 0.5 * k, 0.13, 0.48, 0.82], projection=PC)
    india_map(ax, ticks=False)
    m = field(ax, data, cmap, norm, lat=lat, lon=lon)
    ax.set_title(title, fontsize=7, pad=3)
    label(ax, "ab"[k], x=0.04, y=0.94)
cax = fig.add_axes([0.3, 0.07, 0.4, 0.02])
colorbar(fig, m, cax, "Flashes per km$^2$ per year (NRSC, 2022-2025)", fmt="{x:g}")
fig.savefig(RESULTS / "check_fig1_grid.png", dpi=200)
print(RESULTS / "check_fig1_grid.png")
