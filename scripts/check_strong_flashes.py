#!/usr/bin/env python3
"""Checks of the peak-current pattern (NRSC, 2022-2025). This was Figure 3 until
revision 6; revision 7 shows the detection evidence in fig3_detection.py and keeps
these printed numbers for Supplementary Note 3. Panels: a, b: density of flashes
of at least 20 kA and at least 50 kA; c: share of flashes of at least 20 kA;
d: that share by month and region; e: the share against flash density per cell.
Where sensors are sparse the network misses weak flashes, so the share of strong
flashes rises where few flashes are seen (e).
Output: results/check_strong_flashes.png
"""
import numpy as np
import matplotlib.pyplot as plt

from common import LAT, LON, REGIONS, RESULTS, cell_area, india_area, nrsc_clim, region_grid
from plotting import (GRID, INK, INK2, MM, MUTED, PC, REGION_COLOR, classes, colorbar, field,
                      india_map, label, save)

A = india_area()
reg = region_grid()
inside = np.isfinite(A)
f = nrsc_clim("flashes").sum("hour_utc").values                  # (month, lat, lon)
s20 = nrsc_clim("flashes_ge20kA").values
s50 = nrsc_clim("flashes_ge50kA").values
F, S20, S50 = (np.nansum(x, 0) for x in (f, s20, s50))
for x in (F, S20, S50):
    x[~inside] = np.nan
enough = F >= 20                                                  # flashes per cell per year
share = np.where(enough, 100 * S20 / F, np.nan)

fig = plt.figure(figsize=(183 * MM, 140 * MM))
maps = [(S20 / A, [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5], "Oranges",
         "Flashes of 20 kA or more\nper km$^2$ per year"),
        (1e3 * S50 / A, [2, 5, 10, 20, 30, 50, 75, 100], "Oranges",
         "Flashes of 50 kA or more\nper 1,000 km$^2$ per year"),
        (share, [5, 7.5, 10, 12.5, 15, 20, 30, 40], "Purples",
         "Share of flashes of 20 kA\nor more (%)")]
for k, (data, levels, cm, name) in enumerate(maps):
    ax = fig.add_axes([0.005 + 0.333 * k, 0.555, 0.32, 0.44], projection=PC)
    india_map(ax, ticks=False)
    cmap, norm = classes(levels, cm, extend="both")
    m = field(ax, data, cmap, norm)
    cax = fig.add_axes([0.04 + 0.333 * k, 0.525, 0.25, 0.013])
    colorbar(fig, m, cax, name, fmt="{x:g}")
    label(ax, "abc"[k], x=0.06, y=0.93)

dx = fig.add_axes([0.07, 0.075, 0.4, 0.28])
months = np.arange(1, 13)
for r in list(REGIONS) + ["India"]:
    k = inside if r == "India" else reg == r
    y = 100 * np.nansum(s20[:, k], 1) / np.nansum(f[:, k], 1)
    dx.plot(months, y, color=INK if r == "India" else REGION_COLOR[r], lw=2.0 if r == "India" else 1.2,
            ls="--" if r == "India" else "-", label="All India" if r == "India" else r)
dx.set_xticks(months, list("JFMAMJJASOND"))
dx.set_xlim(0.7, 12.3)
dx.set_ylim(0, 50)
dx.set_ylabel("Share of flashes of\n20 kA or more (%)")
dx.grid(axis="y", color=GRID, lw=0.5)
dx.set_axisbelow(True)
dx.legend(loc="upper left", bbox_to_anchor=(0.0, 1.12), ncol=4, fontsize=5.8, handlelength=1.6,
          columnspacing=0.8)
label(dx, "d", x=-0.14, y=1.04)

ex = fig.add_axes([0.6, 0.075, 0.36, 0.28])
dens = F / A
ok = enough & inside & (A >= 0.5 * cell_area())                   # cells at least half in India
groups = [("Other regions", ~np.isin(reg, ["Himalaya", "Northeast"]), MUTED, 2),
          ("Himalaya", reg == "Himalaya", REGION_COLOR["Himalaya"], 3),
          ("Northeast", reg == "Northeast", REGION_COLOR["Northeast"], 3)]
for name, k, c, z in groups:
    ex.scatter(dens[ok & k], share[ok & k], s=2, color=c, alpha=0.5 if z == 2 else 0.8, lw=0,
               label=name, zorder=z, rasterized=True)
bins = np.logspace(np.log10(0.02), np.log10(20), 16)
mid, med = [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    sel = ok & (dens >= lo) & (dens < hi)
    if sel.sum() >= 20:
        mid.append(np.sqrt(lo * hi))
        med.append(np.median(share[sel]))
ex.plot(mid, med, color=INK, lw=1.6, label="Bin median", zorder=4)
r = np.corrcoef(np.log(dens[ok]), share[ok])[0, 1]
ex.text(0.97, 0.95, f"r = {r:.2f} (log density)", transform=ex.transAxes, ha="right", va="top",
        fontsize=6.5, color=INK2)
ex.set_xscale("log")
ex.xaxis.set_major_formatter(lambda v, _: f"{v:g}")
ex.set_xlabel("Recorded flashes per km$^2$ per year")
ex.set_ylabel("Share of flashes of\n20 kA or more (%)")
ex.set_ylim(0, 70)
ex.legend(loc="upper left", bbox_to_anchor=(0.0, 1.12), ncol=4, fontsize=5.8, handlelength=1.0,
          columnspacing=0.8, markerscale=3)
label(ex, "e", x=-0.17, y=1.04)
print(f"India: {100 * np.nansum(S20) / np.nansum(F):.1f}% of flashes are 20 kA or more, "
      f"{100 * np.nansum(S50) / np.nansum(F):.2f}% are 50 kA or more; r(log density, share) = {r:.2f}")
for lo, hi in [(0, 0.5), (0.5, 1), (1, 2), (2, 5), (5, np.inf)]:
    sel = ok & (dens >= lo) & (dens < hi)
    print(f"cells with {lo}-{hi} flashes per km2 per year: {sel.sum()}, median share {np.median(share[sel]):.0f}%")
hn = ok & np.isin(reg, ["Himalaya", "Northeast"])
print(f"Himalaya and Northeast cells: median share {np.median(share[hn]):.0f}%; r without them = "
      f"{np.corrcoef(np.log(dens[ok & ~hn]), share[ok & ~hn])[0, 1]:.2f}")
d50 = np.where(ok, 1e3 * S50 / A, np.nan)
j, i = np.unravel_index(np.nanargmax(d50), d50.shape)
print(f"densest cell for 50 kA or more: {LAT[j]} N, {LON[i]} E, {d50[j, i]:.0f} per 1,000 km2 per year "
      f"(median of cells {np.nanmedian(d50):.1f})")
fig.savefig(RESULTS / "check_strong_flashes.png", dpi=200)   # no longer a manuscript figure (revision 7)
