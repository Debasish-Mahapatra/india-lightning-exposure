#!/usr/bin/env python3
"""Check, no longer a figure of the paper (Supplementary Fig. S7 of revision 7, main Fig. 8 before): how lightning
follows CAPE, rain and CAPE times rain (NRSC
flashes and ERA5 daily fields on the 0.25 degree grid, 2022-2025, days with an
NRSC file, cells inside India). Descriptive only: the test of lightning
proxies under warming is in the WRF paper.
ERA5: cape = daily mean CAPE, tp = daily rain, capep = sum over the hours of
CAPE times the next hour's rain (era5_daily.py of the WRF paper).
a: monthly cycle over India, each as a share of its yearly total;
b: mean flash density against CAPE times rain, by season;
c: day-to-day rank correlation per cell of flashes and CAPE times rain.
Printed only (the former panels b, c and e): mean flash density by daily CAPE
and rain, flashes per million tonnes of ERA5 rain against CAPE by season, and
the spatial correlation (log-log, across cells) of the seasonal mean maps.
Output: results/check_drivers.png; printed (results/log_check_drivers.txt)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from scipy import stats

from common import ERA5, NRSC_DETAIL, RESULTS, SEASONS, YEARS, india_area
from plotting import DRIVER_COLOR, GRID, INK, MM, PC, SEASON_COLOR, classes, colorbar, field, india_map, label, save

A = india_area()
F = xr.concat([xr.open_dataset(NRSC_DETAIL / f"nrsc_detail_025_{y}.nc")["daily"] for y in YEARS], "time")
E = xr.concat([xr.open_dataset(ERA5 / f"era5_daily_{y}.nc")[["cape", "tp", "capep"]] for y in YEARS], "time")
F, E = xr.align(F, E, join="inner")
ok = np.isfinite(F).any(("lat", "lon")).values
month = pd.DatetimeIndex(F.time.values)[ok].month.values
cells = np.isfinite(E.cape.isel(time=0).values) & np.isfinite(A)
f = F.values[ok][:, cells] / A[cells]                               # flashes per km2 per day
X = {"CAPE": E.cape.values[ok][:, cells], "Rain": E.tp.values[ok][:, cells],
     "CAPE times rain": E.capep.values[ok][:, cells]}
w = A[cells]
print(f"{ok.sum()} days, {cells.sum()} cells")

fig = plt.figure(figsize=(183 * MM, 72 * MM))

# a: monthly cycle over India
ax = fig.add_axes([0.06, 0.2, 0.26, 0.72])
share = lambda x: np.array([np.average(x[month == m].mean(0), weights=w) for m in range(1, 13)])
fm = share(f)
ax.plot(range(1, 13), 100 * fm / fm.sum(), color=INK, lw=2.2, label="Flashes")
for name, x in X.items():
    xm = share(x)
    r = np.corrcoef(xm, fm)[0, 1]
    ax.plot(range(1, 13), 100 * xm / xm.sum(), color=DRIVER_COLOR[name], lw=1.2, label=f"{name} (r = {r:.2f})")
    print(f"monthly cycle over India: r(flashes, {name}) = {r:.2f}")
ax.set_xticks(range(1, 13), list("JFMAMJJASOND"))
ax.set_xlim(0.7, 12.3)
ax.set_ylim(0, 36)
ax.set_ylabel("Normalized monthly\nmean (%)")
ax.grid(axis="y", color=GRID, lw=0.5)
ax.set_axisbelow(True)
ax.legend(loc="upper left", fontsize=5.6, handlelength=1.4)
label(ax, "a", x=-0.2, y=1.0)

# flash density by CAPE and rain (printed only)
cb = np.array([0, 100, 300, 600, 1000, 1500, 2000, 3000, 6000])
rb = np.array([0, 0.5, 2, 5, 10, 20, 40])
fr, cr, rr = f.ravel(), X["CAPE"].ravel(), X["Rain"].ravel()
ci, ri = np.digitize(cr, cb) - 1, np.digitize(rr, rb) - 1
M = np.full((len(rb) - 1, len(cb) - 1), np.nan)
for a in range(len(rb) - 1):
    for b in range(len(cb) - 1):
        s = (ri == a) & (ci == b)
        if s.sum() >= 200:
            M[a, b] = fr[s].mean()
print("mean flash density (per km2 per day) by rain bin (rows, mm) and CAPE bin (columns, J/kg):")
print(pd.DataFrame(M, index=[f"{v:g}" for v in rb[:-1]], columns=[f"{v:g}" for v in cb[:-1]]).round(4).to_string())

# flashes per unit rain against CAPE (printed only), and b: flash density against CAPE times rain, by season
seasons = list(SEASONS.items())[1:]
mid = 0.5 * (cb[:-1] + cb[1:])
pb = np.array([10, 100, 300, 1000, 3000, 10000, 30000, 100000])
dx = fig.add_axes([0.41, 0.2, 0.26, 0.72])
pr = X["CAPE times rain"].ravel()
mon = np.repeat(month, cells.sum())
for (name, months), c in zip(seasons, SEASON_COLOR[1:]):
    sel = np.isin(mon, months)
    y = []
    for b in range(len(cb) - 1):
        k = sel & (ci == b) & (rr > 0.5)
        y.append(fr[k].sum() / (rr[k].sum() / 1000) if k.sum() >= 1000 else np.nan)
    short = name.split(" (")[0]
    pi = np.digitize(pr, pb) - 1
    y2 = [fr[sel & (pi == b)].mean() if (sel & (pi == b)).sum() >= 500 else np.nan for b in range(len(pb) - 1)]
    dx.plot(np.sqrt(pb[:-1] * pb[1:]), y2, color=c, lw=1.3, marker="o", ms=2.5, label=short)
    print(f"{short}: flashes per Mt of ERA5 rain by CAPE bin {np.round(y, 1)}")
dx.set_xscale("log")
dx.set_yscale("log")
dx.xaxis.set_major_formatter(lambda v, _: f"{v:g}")
dx.yaxis.set_major_locator(plt.LogLocator(base=10, subs=(1, 2, 5)))  # the range holds only one power of ten
dx.yaxis.set_major_formatter(lambda v, _: f"{v:g}")
dx.set_xlabel("Daily CAPE times rain (J kg$^{-1}$ mm)")
dx.set_ylabel("Flashes per km$^2$ per day")
dx.grid(color=GRID, lw=0.4)
dx.set_axisbelow(True)
dx.legend(loc="upper left", fontsize=6)
label(dx, "b", x=-0.2, y=1.0)

# spatial correlation of seasonal mean maps (printed only)
groups = [("Year", np.ones(len(month), bool))] + [(n.split(" (")[0].replace("-", "-\n"), np.isin(month, ms))
                                                   for n, ms in SEASONS.items()]
for k, (name, x) in enumerate(X.items()):
    r = []
    for g, sel in groups:
        a_, b_ = f[sel].mean(0), x[sel].mean(0)
        good = (a_ > 0) & (b_ > 0)
        r.append(np.corrcoef(np.log(a_[good]), np.log(b_[good]))[0, 1])
    print(f"spatial r (log-log) with {name}: " + ", ".join(f"{g.replace(chr(10), '')} {v:.2f}" for (g, _), v in zip(groups, r)))

# c: day-to-day rank correlation with CAPE times rain
rank = lambda v: stats.rankdata(v, axis=0)
ra, rb_ = rank(f), rank(X["CAPE times rain"])
ra, rb_ = ra - ra.mean(0), rb_ - rb_.mean(0)
rho = (ra * rb_).sum(0) / np.sqrt((ra ** 2).sum(0) * (rb_ ** 2).sum(0))
rho[(f > 0).sum(0) < 30] = np.nan
grid = np.full(A.shape, np.nan)
grid[cells] = rho
print(f"day-to-day rank correlation with CAPE times rain: median {np.nanmedian(rho):.2f}")
for name in ("Rain", "CAPE"):                                     # the same with rain alone and CAPE alone (printed only)
    rx = rank(X[name])
    rx = rx - rx.mean(0)
    r1 = (ra * rx).sum(0) / np.sqrt((ra ** 2).sum(0) * (rx ** 2).sum(0))
    r1[(f > 0).sum(0) < 30] = np.nan
    both = np.isfinite(rho) & np.isfinite(r1)
    print(f"day-to-day rank correlation with {name}: median {np.nanmedian(r1):.2f}; CAPE times rain is higher in "
          f"{100 * np.mean(rho[both] > r1[both]):.0f}% of cells")
fx = fig.add_axes([0.71, 0.13, 0.29, 0.85], projection=PC)
india_map(fx, ticks=False)
cm2, nm2 = classes([0.2, 0.3, 0.4, 0.5, 0.6, 0.7], "Reds", extend="both")
m2 = field(fx, grid, cm2, nm2)
cax2 = fig.add_axes([0.74, 0.1, 0.23, 0.025])
colorbar(fig, m2, cax2, "Day-to-day rank correlation of\nflashes and CAPE times rain", fmt="{x:g}")
label(fx, "c", x=0.06, y=0.93)
fig.savefig(RESULTS / "check_drivers.png", dpi=200)
print(RESULTS / "check_drivers.png")
