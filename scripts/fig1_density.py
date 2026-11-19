#!/usr/bin/env python3
"""Figure 1: where India's lightning strikes (NRSC, 2022-2025 mean, each month
scaled up for its days without data).
a: flashes per km2 per year on a 0.1 degree grid (enough flashes per cell in a
year: median 742 in 2022-2025; check_fig1_grid.py compares it with 0.25 degree);
b-e: flashes per km2 per season on the 0.25 degree grid of the analyses, on one
colour scale (winter is December-February of the same calendar year), because a
season has too few flashes for 0.1 degree cells.
Output: manuscript/figures/fig1_density.png/.pdf
"""
import numpy as np
import matplotlib.pyplot as plt

from common import (LAT, LAT01, LON, LON01, SEASONS, cell_area, cell_area_01, flashes_01, india_area,
                    india_area_01, nrsc_clim)
from plotting import BLUE, MM, PC, classes, colorbar, field, india_map, label, save

A = india_area()
mon = nrsc_clim("flashes").sum("hour_utc").values                  # (month, lat, lon)
total = np.nansum(mon)
dens = np.where(A >= 0.5 * cell_area(), mon.sum(0) / A, np.nan)      # cells at least half in India
j, i = np.unravel_index(np.nanargmax(dens), dens.shape)
print(f"India: {total / 1e6:.2f} million flashes a year, {total / np.nansum(A):.2f} per km2 over "
      f"{np.nansum(A) / 1e6:.3f} million km2; highest 0.25 degree cell {dens[j, i]:.1f} at {LAT[j]} N, {LON[i]} E")
A1 = india_area_01()
dens01 = flashes_01().sum(0) / A1
half01 = np.where(A1 >= 0.5 * cell_area_01(), dens01, np.nan)
j1, i1 = np.unravel_index(np.nanargmax(half01), half01.shape)
print(f"0.1 degree grid: {np.nansum(dens01 * A1) / 1e6:.2f} million flashes a year; highest cell {half01[j1, i1]:.1f} "
      f"at {LAT01[j1]} N, {LON01[i1]} E")

fig = plt.figure(figsize=(183 * MM, 112 * MM))
ax = fig.add_axes([0.0, 0.12, 0.5, 0.86], projection=PC)
india_map(ax)
cmap, norm = classes([0.25, 0.5, 1, 2, 3, 4, 6, 8, 10, 12], BLUE, extend="both")
m = field(ax, dens01, cmap, norm, lat=LAT01, lon=LON01)
ax.set_title(f"Year: {total / 1e6:.1f} million flashes", fontsize=7, pad=3)
cax = fig.add_axes([0.06, 0.07, 0.38, 0.018])
colorbar(fig, m, cax, "Flashes per km$^2$ per year", fmt="{x:g}")
label(ax, "a", x=0.04, y=0.94)

cmap2, norm2 = classes([0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 3, 5, 8], BLUE, extend="both")
for k, (name, months) in enumerate(SEASONS.items()):
    f = np.nansum(mon[[mm - 1 for mm in months]], 0)
    share = 100 * f.sum() / total
    print(f"{name}: {share:.1f}% of flashes, {f.sum() / 1e6:.2f} million a year")
    bx = fig.add_axes([0.52 + 0.24 * (k % 2), 0.56 - 0.43 * (k // 2), 0.23, 0.39], projection=PC)
    india_map(bx, ticks=False)
    m2 = field(bx, f / A, cmap2, norm2)
    bx.set_title(f"{name.split(' (')[0]}, {name.split('(')[1][:-1]}: {share:.0f}%", fontsize=6.3, pad=2)
    label(bx, "bcde"[k], x=0.07, y=0.92)
cax2 = fig.add_axes([0.58, 0.07, 0.36, 0.018])
colorbar(fig, m2, cax2, "Flashes per km$^2$ per season", fmt="{x:g}")
print(save(fig, "fig1_density"))
