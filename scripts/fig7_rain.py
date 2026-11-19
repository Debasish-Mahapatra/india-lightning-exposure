#!/usr/bin/env python3
"""Figure 7: lightning per unit rain, and lightning with little rain (NRSC
flashes and IMERG V07 Final daily rain on the 0.25 degree grid, January 2022
to September 2025, days with both).
a-d: flashes per million tonnes of rain (flashes per km2 per metre of rain;
1 mm on 1 km2 is 1,000 t) by season, where the season brings at least 20 mm a
year. e: share of flashes that fall on days when their 0.25 degree cell gets
less than 2.5 mm of rain ("dry" lightning), in the pre-monsoon season (cells
with at least 20 flashes a season). Revision 7 removed the former panels e and
g (the same quantities by month and region); their values are still printed,
as is the rain yield per flash (kg), for comparison with the literature
(which counts all flashes, not only cloud-to-ground).
Output: manuscript/figures/fig7_rain.png/.pdf
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

from common import DATA, NRSC_DETAIL, REGIONS, SEASONS, india_area, region_grid
from plotting import MM, PC, classes, colorbar, field, india_map, label, save

A = india_area()
reg = region_grid()
F = xr.concat([xr.open_dataset(NRSC_DETAIL / f"nrsc_detail_025_{y}.nc")["daily"] for y in range(2022, 2026)], "time")
R = xr.concat([xr.open_dataset(DATA / f"imerg_025/imerg_daily_025_{y}.nc")["precipitation"] for y in range(2022, 2026)], "time")
F, R = xr.align(F, R, join="inner")
ok = np.isfinite(F).any(("lat", "lon")).values                   # days with an NRSC file
days = pd.DatetimeIndex(F.time.values)[ok]
F, R = F.values[ok], R.values[ok]
month = days.month.values
nyears = {m: len(set(days.year[month == m])) for m in range(1, 13)}
print(f"{len(days)} days with NRSC and IMERG, {days[0]:%Y-%m-%d} to {days[-1]:%Y-%m-%d}")

DRY = 2.5                                                         # mm per day
F0 = np.nan_to_num(F)
dry = np.where(R < DRY, F0, 0)
print(f"flashes with less than {DRY} mm of rain in their cell that day: {100 * dry.sum() / F0.sum():.1f}% "
      f"(1 mm: {100 * np.where(R < 1, F0, 0).sum() / F0.sum():.1f}%, 5 mm: {100 * np.where(R < 5, F0, 0).sum() / F0.sum():.1f}%)")

fig = plt.figure(figsize=(183 * MM, 150 * MM))
POS = [(0.005, 0.53), (0.335, 0.53), (0.665, 0.53), (0.005, 0.05)]   # a-d; e at (0.335, 0.05)
cmap, norm = classes([0.5, 1, 1.5, 2, 3, 4, 5, 7, 10], "Purples", extend="both")
inside = np.isfinite(A)
for k, (name, months) in enumerate(SEASONS.items()):
    sel = np.isin(month, months)
    f, r = np.nansum(F[sel], 0), np.nansum(R[sel], 0)
    per_year = r / np.mean([nyears[m] for m in months])
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.where(per_year >= 20, (f / A) / (r / 1000), np.nan)
    tot = np.nansum(f[inside]) / np.nansum((A * r / 1000)[inside])
    print(f"{name}: {tot:.2f} flashes per Mt of rain over India, rain yield {1e9 / tot:.2e} kg per flash, "
          f"{100 * dry[sel].sum() / F0[sel].sum():.1f}% of flashes with little rain")
    ax = fig.add_axes([*POS[k], 0.32, 0.42], projection=PC)
    india_map(ax, ticks=False)
    m = field(ax, y, cmap, norm)
    ax.set_title(f"{name}\nIndia: {tot:.1f}", fontsize=6.5, pad=2)
    label(ax, "abcd"[k], x=0.06, y=0.93)
cax = fig.add_axes([0.69, 0.33, 0.28, 0.016])
colorbar(fig, m, cax, "a\u2013d: flashes per million tonnes of rain\n(flashes per km$^2$ per metre of rain)", fmt="{x:g}")


def by_region(values, name):
    """Monthly values by region (the former panels e and g), printed only."""
    for r in list(REGIONS) + ["India"]:
        cells = inside if r == "India" else reg == r
        v = [values(month == m, cells) for m in range(1, 13)]
        print(f"{name}, {r}, January-December: " + " ".join(f"{x:.1f}" for x in v))


by_region(lambda t, c: np.nansum(F[t][:, c]) / np.nansum(R[t][:, c] * A[c] / 1000), "Flashes per million tonnes of rain")

fx = fig.add_axes([0.335, 0.05, 0.32, 0.42], projection=PC)
india_map(fx, ticks=False)
pre = np.isin(month, SEASONS["Pre-monsoon (March-May)"])
fs = F0[pre].sum(0) / nyears[4]
share = np.where(inside & (fs >= 20), 100 * dry[pre].sum(0) / np.maximum(F0[pre].sum(0), 1), np.nan)
cm2, nm2 = classes([5, 10, 15, 20, 30, 40, 50, 60], "Oranges", extend="both")
m2 = field(fx, share, cm2, nm2)
y = np.where(inside, (F0[pre].sum(0) / A) / (R[pre].sum(0) / 1000), np.nan)
ok = inside & (R[pre].sum(0) / nyears[4] >= 20)
print(f"Pre-monsoon: cells with more than 10 flashes per Mt of rain {100 * np.mean(y[ok] > 10):.0f}%, more than 5 {100 * np.mean(y[ok] > 5):.0f}%")
fx.set_title("Pre-monsoon (March-May)", fontsize=6.5, pad=2)
cax2 = fig.add_axes([0.69, 0.15, 0.28, 0.016])
colorbar(fig, m2, cax2, f"e: share of flashes with less than {DRY:g} mm of rain\nin their cell that day (%)", fmt="{x:g}")
label(fx, "e", x=0.06, y=0.93)
by_region(lambda t, c: 100 * dry[t][:, c].sum() / max(F0[t][:, c].sum(), 1),
          f"Flashes with less than {DRY:g} mm of rain in the cell that day (%)")
print(save(fig, "fig7_rain"))
