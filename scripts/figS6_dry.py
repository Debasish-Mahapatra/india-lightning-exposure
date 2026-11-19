#!/usr/bin/env python3
"""Supplementary Figure S6: lightning with little rain (NRSC flashes, IMERG V07
Final daily rain, 0.25 degree, January 2022 to September 2025).
a-d: share of flashes on days when their cell gets less than 2.5 mm of rain,
by season (cells with at least 20 flashes a season); e: the share over India
by month for thresholds of 1, 2.5 and 5 mm a day.
Output: manuscript/supplementary/figS6_dry.png/.pdf
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

from common import DATA, NRSC_DETAIL, REGIONS, SEASONS, india_area, region_grid
from plotting import GRID, INK, INK2, MM, PC, classes, colorbar, field, india_map, label, save

A = india_area()
F = xr.concat([xr.open_dataset(NRSC_DETAIL / f"nrsc_detail_025_{y}.nc")["daily"] for y in range(2022, 2026)], "time")
R = xr.concat([xr.open_dataset(DATA / f"imerg_025/imerg_daily_025_{y}.nc")["precipitation"] for y in range(2022, 2026)], "time")
F, R = xr.align(F, R, join="inner")
ok = np.isfinite(F).any(("lat", "lon")).values
days = pd.DatetimeIndex(F.time.values)[ok]
F, R = np.nan_to_num(F.values[ok]), R.values[ok]
month = days.month.values
reg = region_grid()
fig = plt.figure(figsize=(183 * MM, 72 * MM))
cm, nm = classes([5, 10, 15, 20, 30, 40, 50, 60], "Oranges", extend="both")
for k, (name, months) in enumerate(SEASONS.items()):
    sel = np.isin(month, months)
    ny = np.mean([len(set(days.year[month == m])) for m in months])
    f = F[sel].sum(0)
    dry = np.where(R[sel] < 2.5, F[sel], 0).sum(0)
    share = np.where(np.isfinite(A) & (f / ny >= 20), 100 * dry / np.maximum(f, 1), np.nan)
    print(f"{name}: {100 * dry.sum() / f.sum():.1f}% of India's flashes with less than 2.5 mm; by region: "
          + ", ".join(f"{r} {100 * dry[reg == r].sum() / f[reg == r].sum():.1f}%" for r in REGIONS)
          + f"; cells shown with at least 30%: {100 * np.mean(share[np.isfinite(share)] >= 30):.0f}%")
    ax = fig.add_axes([0.005 + 0.165 * k, 0.24, 0.16, 0.62], projection=PC)
    ax.set_anchor("S")
    india_map(ax, ticks=False)
    m = field(ax, share, cm, nm)
    ax.set_title(name.replace(" (", "\n("), fontsize=6, pad=2)
    label(ax, "abcd"[k], x=0.1, y=0.9)
colorbar(fig, m, fig.add_axes([0.12, 0.16, 0.42, 0.028]), "Share of flashes with less than 2.5 mm of rain in the cell that day (%)", fmt="{x:g}")
ex = fig.add_axes([0.73, 0.2, 0.25, 0.7])
for thr, ls in ((1, ":"), (2.5, "-"), (5, "--")):
    y = [100 * np.where(R[month == m] < thr, F[month == m], 0).sum() / F[month == m].sum() for m in range(1, 13)]
    ex.plot(range(1, 13), y, color=INK, ls=ls, lw=1.3, label=f"less than {thr:g} mm")
    print(f"India, less than {thr:g} mm, January-December (%): " + " ".join(f"{v:.1f}" for v in y))
ex.set_xticks(range(1, 13), list("JFMAMJJASOND"))
ex.set_ylim(0, 70)
ex.set_ylabel("Share of India's flashes (%)")
ex.grid(axis="y", color=GRID, lw=0.5)
ex.set_axisbelow(True)
ex.legend(loc="upper right", fontsize=5.8, handlelength=1.8)
label(ex, "e", x=-0.25, y=1.0)
print(save(fig, "figS6_dry", sup=True))
