#!/usr/bin/env python3
"""Supplementary Figure S1: the growth of the NRSC network. a: flashes over
India each year by region, 2020-2025 (each month scaled up for its days
without data); b-g: flash density each year. The totals rise until 2022 and
then level off, so absolute numbers in the paper use 2022-2025.
Output: manuscript/supplementary/figS1_network.png/.pdf
"""
import numpy as np
import matplotlib.pyplot as plt

from common import REGIONS, india_area, nrsc_clim, region_grid
from plotting import (BLUE, INK, INK2, MM, PC, REGION_COLOR, classes, colorbar, field, india_map,
                      label, save)

A = india_area()
reg = region_grid()
years = range(2020, 2026)
fig = plt.figure(figsize=(183 * MM, 120 * MM))
bx = fig.add_axes([0.07, 0.14, 0.22, 0.78])
cmap, norm = classes([0.25, 0.5, 1, 2, 3, 4, 6, 8, 10, 12], BLUE, extend="both")
bottom = np.zeros(len(years))
by_year = {r: [] for r in REGIONS}
for k, y in enumerate(years):
    f = nrsc_clim("flashes", years=[y]).sum(("month", "hour_utc")).values
    for r in REGIONS:
        by_year[r].append(np.nansum(f[reg == r]) / 1e6)
    print(f"{y}: {np.nansum(f) / 1e6:.2f} million flashes over India")
    ax = fig.add_axes([0.33 + 0.225 * (k % 3), 0.53 - 0.44 * (k // 3), 0.215, 0.43], projection=PC)
    india_map(ax, ticks=False)
    m = field(ax, f / A, cmap, norm)
    ax.set_title(f"{y}: {np.nansum(f) / 1e6:.1f} million", fontsize=6.5, pad=2)
    label(ax, "bcdefg"[k], x=0.08, y=0.9)
for r in REGIONS:
    v = by_year[r]
    print(f"{r}: " + ", ".join(f"{y} {x:.2f}" for y, x in zip(years, v))
          + f" million; 2021 / 2020 {v[1] / v[0]:.2f}, 2022 / 2020 {v[2] / v[0]:.2f}, 2022-2025 range {min(v[2:]):.2f}-{max(v[2:]):.2f}")
cax = fig.add_axes([0.45, 0.07, 0.42, 0.016])
colorbar(fig, m, cax, "Flashes per km$^2$ per year", fmt="{x:g}")
for r in REGIONS:
    bx.bar(years, by_year[r], bottom=bottom, width=0.72, color=REGION_COLOR[r], edgecolor="white",
           linewidth=0.6, label=r)
    bottom += np.array(by_year[r])
for x, t in zip(years, bottom):
    bx.text(x, t + 0.15, f"{t:.1f}", ha="center", va="bottom", fontsize=6, color=INK)
bx.axvspan(2021.55, 2025.45, color="#f0efec", zorder=0, lw=0)
bx.text(2024.4, 14.4, "study period\n2022-2025", ha="center", va="top", fontsize=6, color=INK2)
bx.set_ylim(0, 15)
bx.set_xticks(list(years), [str(y) for y in years], rotation=90)
bx.set_ylabel("Flashes over India (millions per year)")
bx.tick_params(axis="x", length=0)
h, l = bx.get_legend_handles_labels()
bx.legend(h[::-1], l[::-1], loc="upper left", fontsize=5.5, handlelength=0.9, handletextpad=0.4,
          labelspacing=0.3)
label(bx, "a", x=-0.3, y=1.0)
print(save(fig, "figS1_network", sup=True))
