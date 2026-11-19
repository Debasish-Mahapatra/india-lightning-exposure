#!/usr/bin/env python3
"""Figure 2: the seven regions, and their monthly and diurnal cycles of lightning
(NRSC, 2022-2025). Monthly cycle: share of the region's yearly flashes in each
month. Diurnal cycle: share of the season's flashes in each hour, in Indian
Standard Time (IST = UTC + 5:30; the UTC hour h covers h+5:30 to h+6:30 IST and
is plotted at h+6 IST).
Output: manuscript/figures/fig2_cycles.png/.pdf
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from common import REGIONS, SEASONS, india_area, nrsc_clim, region_grid, states
from plotting import (GRID, INK, INK2, MM, PC, REGION_COLOR, india_map, label, save)

A = india_area()
reg = region_grid()
f = nrsc_clim("flashes").values                               # (month, hour UTC, lat, lon)
cells = {r: reg == r for r in REGIONS} | {"India": np.isfinite(A)}
mh = {r: np.nansum(f[:, :, k], 2) for r, k in cells.items()}  # (month, hour UTC)
dens = {r: mh[r].sum() / np.nansum(A[k]) for r, k in cells.items()}

fig = plt.figure(figsize=(183 * MM, 112 * MM))
ax = fig.add_axes([0.0, 0.02, 0.43, 0.96], projection=PC)
india_map(ax, ticks=False, states=False)
st = states()
st["region"] = st.state.map({s: r for r, ss in REGIONS.items() for s in ss})
rg = st.dropna(subset=["region"]).dissolve("region")
rg["geometry"] = rg.geometry.buffer(0.02).buffer(-0.02).simplify(0.005)   # close slivers, fewer points
for r, row in rg.iterrows():
    ax.add_geometries([row.geometry], PC, facecolor=REGION_COLOR[r], edgecolor="white", lw=0.5,
                      alpha=0.85, zorder=2)
spots = {"Himalaya": (76.5, 33.2), "Northwest": (73.0, 25.5), "Gangetic Plain": (81.0, 26.8),
         "Central": (78.5, 21.0), "East": (85.6, 22.4), "Northeast": (93.5, 26.6), "South": (77.5, 14.5)}
for r, (x, y) in spots.items():
    ax.text(x, y, f"{r}\n{dens[r]:.1f}", ha="center", va="center", fontsize=6.5, color=INK,
            transform=PC, zorder=5, path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])
ax.text(0.97, 0.97, "Number: flashes per\nkm$^2$ per year", transform=ax.transAxes,
        fontsize=6, color=INK2, va="top", ha="right")
label(ax, "a", x=0.04, y=0.93)


def lines(axis, x, series, close=False):
    for r in list(REGIONS) + ["India"]:
        y = series[r]
        xs, ys = (np.append(x, x[0] + 24), np.append(y, y[0])) if close else (x, y)
        axis.plot(xs, ys, color=INK if r == "India" else REGION_COLOR[r], lw=2.0 if r == "India" else 1.2,
                  ls="--" if r == "India" else "-", label="All India" if r == "India" else r,
                  zorder=4 if r == "India" else 3)
    axis.grid(axis="y", color=GRID, lw=0.5)
    axis.set_axisbelow(True)


bx = fig.add_axes([0.5, 0.62, 0.37, 0.34])
lines(bx, np.arange(1, 13), {r: 100 * v.sum(1) / v.sum() for r, v in mh.items()})
bx.set_xticks(range(1, 13), list("JFMAMJJASOND"))
bx.set_xlim(0.7, 12.3)
bx.set_ylim(0, 29)
bx.set_ylabel("Share of the year's flashes (%)")
bx.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6, handlelength=1.8)
label(bx, "b", x=-0.12, y=1.0)

ist = (np.arange(24) + 6) % 24
order = np.argsort(ist)
for k, name in enumerate(["Pre-monsoon (March-May)", "Monsoon (June-September)"]):
    cx = fig.add_axes([0.5 + 0.265 * k, 0.1, 0.225, 0.36])
    ms = [m - 1 for m in SEASONS[name]]
    lines(cx, ist[order], {r: (100 * v[ms].sum(0) / v[ms].sum())[order] for r, v in mh.items()}, close=True)
    cx.set_xticks(range(0, 25, 6))
    cx.set_xlim(0, 24)
    cx.set_ylim(0, 12.5)
    cx.set_xlabel("Hour of day (IST)")
    if k == 0:
        cx.set_ylabel("Share of flashes (% per hour)")
    cx.set_title(name, fontsize=7, pad=3)
    label(cx, "cd"[k], x=-0.16 if k == 0 else -0.06, y=1.0)
print(save(fig, "fig2_cycles"))
