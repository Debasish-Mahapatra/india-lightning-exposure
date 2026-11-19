#!/usr/bin/env python3
"""Supplementary Figure S2: hour of the diurnal peak of lightning in each 0.25
degree cell, by season (NRSC, 2022-2025). The peak is the maximum of the
24-hour harmonic of the hourly flash counts (as in Methods; the UTC hour
h is placed at its middle, h+6 IST). Cells are shown where the season has at
least 25 flashes a year and the harmonic's amplitude is at least 40% of the
mean (r >= 0.2 below).
Output: manuscript/supplementary/figS2_peak_hour.png/.pdf
"""
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from common import LAT, LON, REGIONS, SEASONS, india_area, nrsc_clim, region_grid
from plotting import MM, PC, colorbar, field, india_map, label, save

NMIN, RMIN = 25, 0.2

A = india_area()
reg = region_grid()
f = nrsc_clim("flashes").values                        # (month, hour_utc, lat, lon), per year
t = (np.arange(24) + 6.0) % 24                         # middle of each UTC hour, in IST
w = np.exp(-2j * np.pi * t / 24)

# 24 hourly classes on a cyclic palette: night blue, morning green and yellow,
# afternoon orange and red, evening purple (anchors at the middle of each
# 3-hour block, interpolated in between and back to the start at 24:00)
ANCHORS = ["#253494", "#2c8fb8", "#8fd1a6", "#f4d35e", "#f28c28", "#d7263d", "#a3136f", "#5b2c83"]
anch = np.array([mcolors.to_rgb(c) for c in ANCHORS + ANCHORS[:1]])
hh = np.arange(0, 25, 3) + 1.5
cmap = mcolors.ListedColormap([[np.interp((h + 0.5 - 1.5) % 24 + 1.5, hh, anch[:, i]) for i in range(3)]
                               for h in range(24)])
norm = mcolors.BoundaryNorm(np.arange(25), 24)

fig = plt.figure(figsize=(183 * MM, 196 * MM))
for k, (name, months) in enumerate(SEASONS.items()):
    x = np.nansum(f[[m - 1 for m in months]], 0)       # (hour, lat, lon)
    n = x.sum(0)
    z = np.tensordot(w, x, (0, 0))
    peak = (-np.angle(z) * 24 / (2 * np.pi)) % 24
    r = np.abs(z) / np.where(n > 0, n, np.nan)
    show = np.isfinite(A) & (n >= NMIN) & (r >= RMIN)
    ax = fig.add_axes([0.01 + 0.5 * (k % 2), 0.555 - 0.465 * (k // 2), 0.47, 0.43], projection=PC)
    india_map(ax)
    ax.tick_params(labelleft=k % 2 == 0, labelbottom=k // 2 == 1)
    m = field(ax, np.where(show, peak, np.nan), cmap, norm)
    head, rng = name.split(" (")
    ax.set_title(f"{head} ({rng[:-1].replace('-', '–')})", fontsize=7, pad=3)
    label(ax, "abcd"[k], x=0.06, y=0.93)
    inside = np.isfinite(A) & (n > 0)
    print(f"{name}: {show.sum()} cells shown, {100 * n[show].sum() / n[inside].sum():.0f}% of the season's flashes")
    for rg in REGIONS:
        cells = show & (reg == rg)
        if cells.sum() < 20:
            continue
        # flash-weighted circular mean and spread of the peak hour over the region's cells
        zz = np.sum(n[cells] * np.exp(2j * np.pi * peak[cells] / 24))
        mean = (np.angle(zz) * 24 / (2 * np.pi)) % 24
        night = np.mean((peak[cells] >= 21) | (peak[cells] < 6)) * 100
        allc = np.isfinite(A) & (reg == rg)
        aft = 100 * n[cells & (peak >= 15) & (peak < 21)].sum() / np.nansum(n[allc])
        nig = 100 * n[cells & ((peak >= 21) | (peak < 6))].sum() / np.nansum(n[allc])
        print(f"  {rg:15s} {cells.sum():4d} cells, mean peak {mean:5.1f} IST, {night:3.0f}% of cells peak at "
              f"21-06 IST; of all its flashes {aft:3.0f}% in cells peaking at 15-21 and {nig:3.0f}% at 21-06 IST")
    allc = np.isfinite(A)
    print(f"  India: {100 * n[show & (peak >= 15) & (peak < 21)].sum() / np.nansum(n[allc]):.0f}% of flashes "
          f"in cells peaking at 15-21 IST")
cax = fig.add_axes([0.2, 0.045, 0.6, 0.014])
cb = colorbar(fig, m, cax, "Hour of the diurnal peak of lightning (IST)", ticks=range(0, 25, 3))
cb.ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)])
print(save(fig, "figS2_peak_hour", sup=True))
