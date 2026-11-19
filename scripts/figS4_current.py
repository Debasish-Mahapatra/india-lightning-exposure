#!/usr/bin/env python3
"""Supplementary Figure S4: peak current of India's flashes (NRSC, 2022-2025).
a: mean peak current per 0.25 degree cell (cells with at least 20 flashes a
year); b: share of flashes in NRSC's four classes (C1 up to 20 kA, C2 20-50,
C3 50-100, C4 above 100 kA) by region. Weak flashes go undetected where
sensors are sparse, which raises the mean and the share of strong flashes.
Output: manuscript/supplementary/figS4_current.png/.pdf
"""
import numpy as np
import matplotlib.pyplot as plt

from common import DATA, LAT, LON, REGIONS, YEARS, india_area, nrsc_clim, region_grid
from plotting import INK, INK2, MM, PC, classes, colorbar, field, india_map, label, save

A = india_area()
reg = region_grid()
F = np.nansum(nrsc_clim("flashes").sum("hour_utc").values, 0)
C = np.nansum(nrsc_clim("current_sum").values, 0)
mean_ka = np.where(np.isfinite(A) & (F >= 20), C / np.maximum(F, 1) / 1e3, np.nan)
edges = [0, 20e3, 50e3, 100e3, np.inf]
counts = {r: np.zeros(4) for r in list(REGIONS) + ["All India"]}
ka = {r: [0.0, 0] for r in counts}
for y in YEARS:
    z = np.load(DATA / f"nrsc_flashes/flashes_{y}.npz")
    j = np.rint((z["lat"] - LAT[0]) / 0.25).astype(int)
    i = np.rint((z["lon"] - LON[0]) / 0.25).astype(int)
    cls = np.digitize(z["amps"], edges) - 1
    ok = (cls >= 0) & (cls < 4)
    rr = reg[j, i]
    for r in REGIONS:
        counts[r] += np.bincount(cls[ok & (rr == r)], minlength=4)
        ka[r][0] += z["amps"][ok & (rr == r)].sum() / 1e3
        ka[r][1] += (ok & (rr == r)).sum()
    counts["All India"] += np.bincount(cls[ok], minlength=4)
    ka["All India"][0] += z["amps"][ok].sum() / 1e3
    ka["All India"][1] += ok.sum()
fig = plt.figure(figsize=(183 * MM, 85 * MM))
ax = fig.add_axes([0.0, 0.12, 0.45, 0.86], projection=PC)
india_map(ax, ticks=False)
cmap, norm = classes([9, 10, 11, 12, 13, 14, 16, 18, 20, 24], "Oranges", extend="both")
m = field(ax, mean_ka, cmap, norm)
colorbar(fig, m, fig.add_axes([0.05, 0.08, 0.35, 0.025]), "Mean peak current (kA)", fmt="{x:g}")
label(ax, "a", x=0.06, y=0.94)
bx = fig.add_axes([0.58, 0.18, 0.4, 0.74])
names = list(counts)
share = np.array([100 * counts[r] / counts[r].sum() for r in names])
colors = ["#fdd0a2", "#fd8d3c", "#d94801", "#7f2704"]
left = np.zeros(len(names))
for c, (lab, col) in enumerate(zip(["C1 <20 kA", "C2 20 to <50 kA", "C3 50 to <100 kA", r"C4 $\geq$100 kA"], colors)):
    bx.barh(range(len(names)), share[:, c], left=left, color=col, edgecolor="white", lw=0.5, label=lab, height=0.7)
    left += share[:, c]
for k, r in enumerate(names):
    bx.text(101, k, f"{share[k, 1:].sum():.0f}%", va="center", fontsize=6, color=INK2)
    print(f"{r}: " + ", ".join(f"{v:.2f}%" for v in share[k]) + f"; mean peak current {ka[r][0] / ka[r][1]:.1f} kA")
bx.set_yticks(range(len(names)), names, fontsize=6.5)
bx.invert_yaxis()
bx.set_xlim(0, 100)
bx.set_xlabel("Share of flashes (%); number: share of 20 kA or more")
bx.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=5.8, handlelength=0.9, columnspacing=0.8)
bx.tick_params(axis="y", length=0)
label(bx, "b", x=-0.3, y=1.08)
print(save(fig, "figS4_current", sup=True))
