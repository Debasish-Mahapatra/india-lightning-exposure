#!/usr/bin/env python3
"""Figure 3: how completely the NRSC network records cloud-to-ground (CG)
lightning (revision 7; replaces the peak-current figure fig3_strong.py and
takes panels b, c and e of the former Supplementary Fig. S4).
a: NRSC flashes per satellite-expected CG flash, 2022-2023 (both summed over
   3 x 3 cells of 0.25 degree; cells with at least 0.5 expected flashes per
   km2 per year); expected CG flashes from ISS LIS total lightning and the
   IITM ratios of cloud to CG flashes (sr_detection.py).
b: share of recorded flashes of 20 kA or more, 2022-2025 (cells with at least
   20 recorded flashes a year).
c: detection by region, with the range from the ends of the published ratios.
d: detection by State/UT (larger than 20,000 km2) against the share of flashes
   of 20 kA or more.
Inputs: data/vulnerability/detection_025.nc, results/detection_state.csv,
results/detection_region.csv, data/nrsc_detail
Output: manuscript/figures/fig3_detection.png/.pdf
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from scipy.ndimage import uniform_filter

from common import DATA, REGIONS, RESULTS, india_area, nrsc_clim
from plotting import GRID, INK, INK2, MM, MUTED, PC, REGION_COLOR, classes, colorbar, field, india_map, label, save

TO_REGION = {s: r for r, ss in REGIONS.items() for s in ss}
A = india_area()
aa = np.nan_to_num(A)
d = xr.open_dataset(DATA / "vulnerability/detection_025.nc")
sm = lambda x: uniform_filter(np.nan_to_num(x) * aa, 3) / np.maximum(uniform_filter(aa, 3), 1e-9)
net_s, sat_s = sm(d.nrsc_2022_2023.values), sm(d.iss_cg_2022_2023.values)
det = np.where(np.isfinite(A) & (sat_s >= 0.5), net_s / np.maximum(sat_s, 1e-9), np.nan)
F = np.nansum(nrsc_clim("flashes").sum("hour_utc").values, 0)
S20 = np.nansum(nrsc_clim("flashes_ge20kA").values, 0)
share = np.where(np.isfinite(A) & (F >= 20), 100 * S20 / np.maximum(F, 1e-9), np.nan)
st = pd.read_csv(RESULTS / "detection_state.csv", index_col=0)
st["region"] = [TO_REGION.get(s, "") for s in st.index]
st = st[(st.area_km2 > 2e4) & (st.region != "")]
st["s20"] = 100 * st.share_ge20kA
rg = pd.read_csv(RESULTS / "detection_region.csv", index_col=0)

fig = plt.figure(figsize=(183 * MM, 160 * MM))
ax = fig.add_axes([0.0, 0.5, 0.48, 0.48], projection=PC)
india_map(ax, ticks=False)
cmap, norm = classes([0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0], "Purples", extend="both")
m = field(ax, det, cmap, norm)
colorbar(fig, m, fig.add_axes([0.06, 0.475, 0.37, 0.013]), "NRSC flashes per satellite-expected CG flash (2022–2023)",
         fmt="{x:g}")
label(ax, "a", x=0.07, y=0.92)

bx = fig.add_axes([0.5, 0.5, 0.48, 0.48], projection=PC)
india_map(bx, ticks=False)
cmap2, norm2 = classes([5, 7.5, 10, 12.5, 15, 20, 30, 40], "Oranges", extend="both")
m2 = field(bx, share, cmap2, norm2)
colorbar(fig, m2, fig.add_axes([0.56, 0.475, 0.37, 0.013]), "Share of recorded flashes of 20 kA or more (%, 2022–2025)",
         fmt="{x:g}")
label(bx, "b", x=0.07, y=0.92)

cx = fig.add_axes([0.13, 0.07, 0.3, 0.32])
order = rg.drop("India").sort_values("detection").index.tolist() + ["India"]
for k, r in enumerate(order):
    v = rg.loc[r]
    c = INK if r == "India" else REGION_COLOR[r]
    cx.barh(k, v.detection, color=c, height=0.62)
    cx.plot([v.detection_lo, v.detection_hi], [k, k], color=INK2 if r != "India" else MUTED, lw=1.0)
    cx.text(v.detection_hi + 0.03, k, f"{v.detection:.2f}", va="center", fontsize=6, color=INK)
cx.set_yticks(range(len(order)), order, fontsize=6.5)
cx.set_xlim(0, 1.15)
cx.set_xlabel("NRSC flashes per\nsatellite-expected CG flash")
cx.tick_params(axis="y", length=0)
for sp in ("top", "right"):
    cx.spines[sp].set_visible(False)
label(cx, "c", x=-0.36, y=1.02)

dx = fig.add_axes([0.6, 0.07, 0.36, 0.32])
for r, g in st.groupby("region"):
    dx.scatter(g.s20, g.de, s=16, color=REGION_COLOR[r], edgecolor="white", lw=0.4, zorder=3)
rho = stats.spearmanr(st.s20, st.de)
dx.text(0.97, 0.97, f"Rank correlation {rho.statistic:+.2f}\n({len(st)} States/UTs)", transform=dx.transAxes,
        va="top", ha="right", fontsize=6, color=INK2)
dx.set_xlabel("Share of recorded flashes of 20 kA or more (%)")
dx.set_ylabel("NRSC flashes per\nsatellite-expected CG flash")
dx.set_ylim(0, 1.0)
dx.grid(color=GRID, lw=0.5)
dx.set_axisbelow(True)
dx.legend(handles=[Line2D([], [], marker="o", ls="", color=REGION_COLOR[r], markersize=4, label=r) for r in REGIONS],
          loc="upper right", bbox_to_anchor=(1.0, 0.84), fontsize=5.6, frameon=False, handletextpad=0.2,
          labelspacing=0.3)
label(dx, "d", x=-0.17, y=1.02)
print(f"India: detection {rg.loc['India', 'detection']:.2f} ({rg.loc['India', 'detection_lo']:.2f}-"
      f"{rg.loc['India', 'detection_hi']:.2f}); share of flashes of 20 kA or more {100 * np.nansum(S20) / np.nansum(F):.1f}%; "
      f"detection against that share across {len(st)} States/UTs rho {rho.statistic:+.2f}")
print(save(fig, "fig3_detection"))
