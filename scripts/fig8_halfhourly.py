#!/usr/bin/env python3
"""Figure 8 (revision 7; formerly Fig. 9 a-c) and Supplementary Figure S8
(formerly Fig. 9 d, e): lightning, rain and cloud tops through the day.
Top (IMERG V07 Final half-hourly rain and NRSC flashes, 2022 to September
2025, days with both; data/obs_hh_025 from hh_cycle.py):
a: share of the season's flashes and rain in each half hour over India
   (Indian Standard Time: UTC + 5:30, so half hours map exactly);
b: time of the peak of lightning (filled) and rain (open) by region, before
   and during the monsoon (maximum of the 24-hour harmonic of each cycle);
   the line is the lag of rain behind lightning;
c: flashes per million tonnes of rain in each half hour, over India.
Supplementary Fig. S8 (MERGIR 4 km brightness temperature; data/mergir_light
from mergir_light.py; shown when available):
a: cloud-top temperature over flashes (coldest pixel within 12 km), by season;
b: mean flashes per 1 degree box and half hour against the box's area of cloud
   colder than 235 K and 208 K, by season.
Output: manuscript/figures/fig8_halfhourly.png/.pdf and
        manuscript/supplementary/figS8_cloud[_provisional].png/.pdf (provisional
        until MERGIR covers all of 2022-2025)
Usage: fig8_halfhourly.py [final]
"""
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import xarray as xr

from common import DATA, REGIONS, SEASONS, india_area, region_grid
from plotting import GRID, INK, INK2, MM, MUTED, SEASON_COLOR, label, save

A = india_area()
reg = region_grid()
inside = np.isfinite(A)
hh = [xr.open_dataset(f) for f in sorted((DATA / "obs_hh_025").glob("hh_cycle_*.nc"))]
rain = sum(np.nan_to_num(d.rain.values) for d in hh)          # (month, 48, lat, lon), mm summed over days
flash = sum(np.nan_to_num(d.flashes.values) for d in hh)
ndays = sum(d.days.values for d in hh)
print(f"{len(hh)} years, {ndays.sum()} days with IMERG and NRSC")
ist = (np.arange(48) + 11) % 48                                # IST half-hour index of each UTC half hour
order = np.argsort(ist)
tmid = np.arange(48) * 0.5 + 0.25                              # middle of each IST half hour


def cycle(x, months, cells, weights=None):
    """Half-hourly cycle summed over the months and cells, in IST order; rain is
    weighted by India's area in each cell (mm x km2), flashes are counts."""
    w = np.nan_to_num(A) if weights is None else weights
    return np.einsum("mhjk,jk->h", x[[m - 1 for m in months]], np.where(cells, w, 0))[order]


def peak_time(v):
    """Time (IST hours) of the maximum of the first diurnal harmonic."""
    c = np.sum(v * np.exp(-2j * np.pi * tmid / 24))
    return (-np.angle(c) * 24 / (2 * np.pi)) % 24


fig = plt.figure(figsize=(183 * MM, 82 * MM))
ax = fig.add_axes([0.07, 0.17, 0.26, 0.68])
for (name, months), c in zip(list(SEASONS.items())[1:3], SEASON_COLOR[1:3]):
    fv = cycle(flash, months, inside, np.ones_like(A))
    rv = cycle(rain, months, inside)
    short = name.split(" (")[0]
    ax.plot(tmid, 100 * fv / fv.sum(), color=c, lw=1.6, label=f"{short}: flashes")
    ax.plot(tmid, 100 * rv / rv.sum(), color=c, lw=1.2, ls="--", label=f"{short}: rain")
    print(f"India {short}: flashes peak {tmid[np.argmax(fv)]:.2f} IST (harmonic {peak_time(fv):.1f}), "
          f"rain peak {tmid[np.argmax(rv)]:.2f} IST (harmonic {peak_time(rv):.1f})")
    day_, night = (tmid >= 12) & (tmid < 20), (tmid >= 21) | (tmid < 6)
    print(f"India {short}: 12:00-20:00 IST holds {100 * fv[day_].sum() / fv.sum():.0f}% of flashes and "
          f"{100 * rv[day_].sum() / rv.sum():.0f}% of rain; 21:00-06:00 holds {100 * fv[night].sum() / fv.sum():.0f}% "
          f"of flashes and {100 * rv[night].sum() / rv.sum():.0f}% of rain")
ax.set_xticks(range(0, 25, 3))
ax.set_xlim(0, 24)
ax.set_ylim(0, 7)
ax.set_xlabel("Hour of day (IST)")
ax.set_ylabel("Share of the season (% per half hour)")
ax.grid(axis="y", color=GRID, lw=0.5)
ax.set_axisbelow(True)
ax.legend(loc="upper left", fontsize=5.6, handlelength=1.8)
label(ax, "a", x=-0.2, y=1.0)

bx = fig.add_axes([0.45, 0.17, 0.2, 0.68])
names = list(REGIONS) + ["India"]
for (name, months), c, dy in zip(list(SEASONS.items()), SEASON_COLOR, (0, -0.16, 0.16, 0)):
    for k, r in enumerate(names):
        cells = inside if r == "India" else reg == r
        fv = cycle(flash, months, cells, np.ones_like(A))
        rv = cycle(rain, months, cells)
        if fv.sum() < 2000:
            continue
        lag = (peak_time(rv) - peak_time(fv) + 12) % 24 - 12
        print(f"{r} {name.split(' (')[0]}: rain peak {lag:+.1f} h after the lightning peak")
        if name.startswith(("Winter", "Post")):                 # shown: the two main seasons
            continue
        tl = peak_time(fv) % 24
        tl = tl + 24 if tl < 12 else tl                         # axis from noon to noon
        bx.plot([tl, tl + lag], [k + dy] * 2, color=c, lw=1.3, solid_capstyle="round")
        bx.plot(tl, k + dy, "o", color=c, ms=4)
        bx.plot(tl + lag, k + dy, "o", mfc="white", mec=c, mew=1.1, ms=4)
bx.set_yticks(range(len(names)), [n if n != "India" else "All India" for n in names], fontsize=6)
bx.invert_yaxis()
bx.set_xlim(12, 36)
bx.set_xticks(range(12, 37, 6), ["12", "18", "0", "6", "12"])
bx.set_xlabel("Time of the diurnal peak (IST)")
bx.grid(axis="x", color=GRID, lw=0.5)
bx.set_axisbelow(True)
bx.tick_params(axis="y", length=0)
bx.legend(handles=[Line2D([0], [0], ls="", marker="o", color=INK2, ms=4, label="Lightning"),
                   Line2D([0], [0], ls="", marker="o", mfc="white", mec=INK2, mew=1.1, ms=4, label="Rain")],
          loc="lower right", fontsize=5.6, handletextpad=0.2, borderpad=0.3)
label(bx, "b", x=-0.45, y=1.0)

cx = fig.add_axes([0.74, 0.17, 0.24, 0.68])
for (name, months), c in zip(SEASONS.items(), SEASON_COLOR):
    fv = cycle(flash, months, inside, np.ones_like(A))
    rv = cycle(rain, months, inside)                           # mm times km2
    y = fv / (rv / 1000)                                       # flashes per million tonnes
    cx.plot(tmid, y, color=c, lw=1.3, label=name.split(" (")[0])
    hm = lambda t: f"{int(t):02d}:{int(round(60 * (t % 1))):02d}"
    am, pm = (tmid >= 6) & (tmid < 12), (tmid >= 12) & (tmid < 18)
    print(f"India {name.split(' (')[0]}: flashes per Mt of rain lowest {y[am].min():.1f} at {hm(tmid[am][np.argmin(y[am])])} IST, "
          f"highest {y[pm].max():.1f} at {hm(tmid[pm][np.argmax(y[pm])])} IST")
cx.set_yscale("log")
cx.set_yticks([0.3, 0.5, 1, 2, 5, 10], ["0.3", "0.5", "1", "2", "5", "10"])
cx.set_ylim(0.3, 12)
cx.set_xticks(range(0, 25, 3))
cx.set_xlim(0, 24)
cx.set_xlabel("Hour of day (IST)")
cx.set_ylabel("Flashes per million tonnes of rain")
cx.grid(axis="y", color=GRID, lw=0.5, which="both")
cx.set_axisbelow(True)
SEASON_LEGEND = [Line2D([0], [0], color=c, marker="o", ms=3.5, lw=1.3, label=n.split(" (")[0])
                 for n, c in zip(SEASONS, SEASON_COLOR)]
fig.legend(handles=SEASON_LEGEND, loc="lower center", bbox_to_anchor=(0.5, 0.92), ncol=4, fontsize=6, frameon=False,
           handlelength=2.0, columnspacing=1.5)                 # one season legend for b (dots) and c (lines)
label(cx, "c", x=-0.22, y=1.0)
print(save(fig, "fig8_halfhourly"))

fig = plt.figure(figsize=(183 * MM, 82 * MM))                 # Supplementary Fig. S8: cloud tops
fig.legend(handles=SEASON_LEGEND, loc="lower center", bbox_to_anchor=(0.5, 0.92), ncol=4, fontsize=6, frameon=False,
           handlelength=2.0, columnspacing=1.5)

final = len(sys.argv) > 1 and sys.argv[1] == "final"
ml = sorted((DATA / "mergir_light").glob("mergir_light_*.nc")) if final else []
dx = fig.add_axes([0.08, 0.2, 0.38, 0.66])
ex = fig.add_axes([0.58, 0.2, 0.38, 0.66])
if ml:
    m = [xr.open_dataset(f) for f in ml]
    tb = m[0].tb.values
    h = sum(d.tb_hist.values for d in m)                       # (month, region, tb)
    for (name, months), c in zip(SEASONS.items(), SEASON_COLOR):
        v = h[[mm - 1 for mm in months]].sum((0, 1))
        if v.sum() < 1000:
            continue
        dx.plot(tb, 100 * np.convolve(v, np.ones(3) / 3, mode="same") / v.sum(), color=c, lw=1.3,
                label=name.split(" (")[0])                  # 3 K running mean
        cdf = np.cumsum(v) / v.sum()
        print(f"MERGIR {name.split(' (')[0]}: median cloud top over flashes {tb[np.searchsorted(cdf, 0.5)]:.0f} K, "
              f"{100 * v[tb < 235].sum() / v.sum():.0f}% below 235 K, {100 * v[tb < 208].sum() / v.sum():.0f}% below 208 K")
    for thr, ls in ((235, "-"), (208, "--")):
        n = sum(d[f"bin{thr}_n"].values for d in m)
        f = sum(d[f"bin{thr}_flashes"].values for d in m)
        area = m[0].area_bin.values
        for (name, months), c in zip(SEASONS.items(), SEASON_COLOR):
            nn = n[[mm - 1 for mm in months]].sum((0, 1))
            ff = f[[mm - 1 for mm in months]].sum((0, 1))
            ok = nn >= 200
            ex.plot(area[ok], (ff / np.maximum(nn, 1))[ok], color=c, lw=1.3, ls=ls,
                    label=f"{name.split(' (')[0]}, {thr} K" if thr == 235 else None)
            at = [np.argmin(abs(area - a)) for a in (2000, 5000, 10000)]
            print(f"MERGIR {name.split(' (')[0]}, colder than {thr} K: mean flashes per box and half hour at 2,000/5,000/10,000 km2: "
                  + "/".join(f"{ff[k] / nn[k]:.0f}" if nn[k] >= 200 else "-" for k in at))
    dx.set_xlim(180, 300)
    dx.set_xlabel("Coldest cloud-top brightness temperature\nwithin about 12 km of the flash (K)")
    dx.set_ylabel("Matched flashes (% per K)")
    dx.axvline(235, color=MUTED, lw=0.6)
    dx.axvline(208, color=MUTED, lw=0.6, ls="--")
    ex.set_xlabel("Area of cold cloud in the 1-degree box (km$^2$)\nsolid: colder than 235 K; dashed: colder than 208 K")
    ex.set_ylabel("Mean flashes per box and half hour")
    for a_ in (dx, ex):
        a_.grid(axis="y", color=GRID, lw=0.5)
        a_.set_axisbelow(True)
else:
    for a_ in (dx, ex):
        a_.text(0.5, 0.5, "MERGIR analysis pending", ha="center", va="center", transform=a_.transAxes,
                color=INK2, fontsize=8)
        a_.set_xticks([])
        a_.set_yticks([])
label(dx, "a", x=-0.12, y=1.0)
label(ex, "b", x=-0.12, y=1.0)
complete = final and len(ml) == 4                                # all of 2022-2025
print(f"MERGIR years: {[f.stem[-4:] for f in ml]}")
print(save(fig, "figS8_cloud" if complete else "figS8_cloud_provisional", sup=True))
