#!/usr/bin/env python3
"""Figure 4: lightning exposure (NRSC 2022-2025, WorldPop 2020).
a: flash density where people live, by district (Survey of India; population-
weighted mean); b: the 20 districts with the largest share of India's
exposure (population times flash density); c: exposure by hour of day (IST)
and season, as a share of the year's exposure (flashes weighted by the people
under them); grey: all lightning over India's area by hour, whoever lives
there, as a share of the year's flashes.
Output: manuscript/figures/fig4_exposure.png/.pdf
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr

from common import DISTRICTS, POP, REGIONS, RESULTS, SEASONS, india_area, nrsc_clim, region_grid
from plotting import (BLUE, GRID, INK, INK2, MM, MUTED, PC, REGION_COLOR, SEASON_COLOR, classes, colorbar,
                      india_map, label, save, _INDIA)

TO_REGION = {s: r for r, ss in REGIONS.items() for s in ss}

dt = pd.read_csv(RESULTS / "district_table.csv", index_col=0)
dt["region"] = [TO_REGION.get(s, "") for s in dt.state]
geo = gpd.read_file(DISTRICTS).to_crs(4326)
geo["geometry"] = geo.geometry.simplify(0.005)                  # about 500 m, small files
geo = geo.join(dt[["exposure"]])

fig = plt.figure(figsize=(183 * MM, 118 * MM))
ax = fig.add_axes([0.0, 0.1, 0.46, 0.88], projection=PC)
india_map(ax, ticks=False)
cmap, norm = classes([0.25, 0.5, 1, 2, 3, 4, 6, 8, 10, 12], BLUE, extend="both")
ok = geo.exposure.notna()
geo[ok].plot(ax=ax, column="exposure", cmap=cmap, norm=norm, transform=PC, edgecolor="white",
             linewidth=0.1, zorder=1)
ax.add_geometries([_INDIA], PC, facecolor="none", edgecolor=INK, lw=0.5, zorder=4)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cax = fig.add_axes([0.06, 0.07, 0.34, 0.018])
colorbar(fig, sm, cax, "Flashes per km$^2$ per year where people live\n(by district, 2022-2025)", fmt="{x:g}")
label(ax, "a", x=0.07, y=0.93)

bx = fig.add_axes([0.685, 0.55, 0.3, 0.39])
top = dt.head(20).iloc[::-1]
names = [f"{d} ({s})" for d, s in zip(top.DISTRICT, top.ST_NM)]
bx.barh(range(len(top)), top.share, color=[REGION_COLOR.get(r, MUTED) for r in top.region], height=0.72)
bx.set_yticks(range(len(top)), names, fontsize=5.8)
bx.tick_params(axis="y", length=0)
for k, (v, e) in enumerate(zip(top.share, top.exposure)):
    bx.text(v + 0.03, k, f"{e:.1f}", va="center", fontsize=5.5, color=INK2)
bx.set_xlabel("Share of India's exposure (%)")
bx.set_xlim(0, 2.3)
n = (dt.share.cumsum() <= 50).sum() + 1
bx.set_title(f"Half of India's exposure is in {n} of {len(dt)} districts\n"
             f"({dt.population_m.iloc[:n].sum():.0f} million people); number: flashes per km$^2$ per year",
             fontsize=6, color=INK2, loc="right", pad=3)
used = [r for r in REGIONS if r in set(top.region)]
bx.legend([plt.Rectangle((0, 0), 1, 1, color=REGION_COLOR[r]) for r in used], used, loc="lower right",
          fontsize=5.8, handlelength=0.9, handletextpad=0.4)
label(bx, "b", x=-0.6, y=1.02)                       # right of panel a, not on its frame

A = india_area()
pop = np.nan_to_num(xr.open_dataset(POP / "pop2020_025.nc").population.values)
f = np.nan_to_num(nrsc_clim("flashes").values / A)                   # (month, hour UTC, lat, lon), per km2
E = np.einsum("mhjk,jk->mh", f, pop) / pop.sum()                    # people-weighted
G = np.einsum("mhjk,jk->mh", f, np.nan_to_num(A)) / np.nansum(A)   # area-weighted
ist = (np.arange(24) + 6) % 24
o = np.argsort(ist)
cx = fig.add_axes([0.575, 0.1, 0.41, 0.34])
hours = np.append(ist[o], 24)
close = lambda y: np.append(y, y[0])
cx.plot(hours, close(100 * G.sum(0)[o] / G.sum()), color=MUTED, lw=1.2, ls="--", label="All lightning (India's area)")
cx.plot(hours, close(100 * E.sum(0)[o] / E.sum()), color=INK, lw=2.0, label="Exposure, all year")
for (name, months), c in zip(SEASONS.items(), SEASON_COLOR):
    y = E[[m - 1 for m in months]].sum(0)[o]
    cx.plot(hours, close(100 * y / E.sum()), color=c, lw=1.2, label=f"Exposure, {name.split(' (')[0].lower()}")
cx.set_xticks(range(0, 25, 3))
cx.set_xlim(0, 24)
cx.set_ylim(0, 11.5)
cx.set_xlabel("Hour of day (IST)")
cx.set_ylabel("Share of the year's\ntotal (% per hour)")
cx.grid(axis="y", color=GRID, lw=0.5)
cx.set_axisbelow(True)
cx.legend(loc="upper left", fontsize=5.4, handlelength=1.5, ncol=2, columnspacing=0.7)
cx.set_yticks(range(0, 11, 2))
label(cx, "c", x=-0.14, y=1.02)
day = E.sum(0)[o]
print(f"exposure peaks at {np.argmax(day)} IST; 12-19 IST holds {100 * day[12:19].sum() / day.sum():.0f}% "
      f"and 06-18 IST {100 * day[6:18].sum() / day.sum():.0f}% of the year's exposure; "
      f"monsoon {100 * E[5:9].sum() / E.sum():.0f}%, pre-monsoon {100 * E[2:5].sum() / E.sum():.0f}%")
g = G.sum(0)[o]
night = np.r_[21:24, 0:6]                                           # 20:30-05:30 IST
print(f"all lightning over India's area peaks at {np.argmax(g)} IST, 11:30-18:30 holds "
      f"{100 * g[12:19].sum() / g.sum():.0f}%; 20:30-05:30 holds {100 * day[night].sum() / day.sum():.0f}% "
      f"of the exposure and {100 * g[night].sum() / g.sum():.0f}% of all lightning")
# Why the seasons differ (panel c): each season's share of India's flashes and of the exposure, how much more
# lightning falls where people live than over India's area, and the regions that hold the season's exposure
R = region_grid()
Fm = f.sum(1)                                                        # per km2 per year, by month and cell
people = {r: pop[R == r].sum() / pop.sum() for r in REGIONS}
print("people by region: " + ", ".join(f"{r} {100 * v:.0f}%" for r, v in people.items()))
print("exposure by month (% of the year): " + ", ".join(f"{100 * v:.1f}" for v in E.sum(1) / E.sum()))
for name, months in SEASONS.items():
    k = [m - 1 for m in months]
    reg = {r: (Fm[k].sum(0) * pop)[R == r].sum() for r in REGIONS}
    top = sorted(reg.items(), key=lambda x: -x[1])
    print(f"{name}: {100 * G[k].sum() / G.sum():.0f}% of the flashes over India's area and "
          f"{100 * E[k].sum() / E.sum():.0f}% of the exposure ({100 * E[k].sum() / E.sum() / len(k):.1f}% per month); "
          f"flash density where people live {E[k].sum() / G[k].sum():.2f} times the mean over India's area; "
          f"exposure by region: " + ", ".join(f"{r} {100 * v / sum(reg.values()):.0f}%" for r, v in top))
print(save(fig, "fig4_exposure"))
