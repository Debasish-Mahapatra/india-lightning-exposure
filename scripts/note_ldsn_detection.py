#!/usr/bin/env python3
"""Two figures for a short technical note to NRSC on the detection efficiency
of its Lightning Detection Sensor Network (LDSN), from the satellite comparison
of sr_detection.py (2022-2023).
Figure 1: a, satellite-expected cloud-to-ground (CG) flash density (ISS LIS
   total lightning times the CG fraction); b, CG flash density recorded by the
   LDSN on the same scale; c, b divided by a (3 x 3 cell sums, as in
   Supplementary Fig. S4b).
Figure 2: a, detection by region with the range from the ends of the published
   cloud-to-ground ratios; b, expected CG flashes the network does not record,
   per year, by region; c, d, detection by State/UT against records per flash
   and the share of flashes of 20 kA or more (2022-2025).
Also prints the numbers quoted in the note (missed flashes by region,
population-weighted detection).
Inputs: data/vulnerability/detection_025.nc, results/detection_state.csv,
results/detection_region.csv, data/population/pop_state_cell.csv
Output: notes/ldsn_detection/fig1_ldsn_detection_map.png/.pdf,
notes/ldsn_detection/fig2_ldsn_detection_regions.png/.pdf
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from scipy.ndimage import uniform_filter

from common import DATA, POP, REGIONS, RESULTS, WS, india_area, region_grid
from plotting import BLUE, GRID, INK, INK2, MM, MUTED, PC, REGION_COLOR, classes, colorbar, field, india_map, label

OUT = WS / "notes/ldsn_detection"
OUT.mkdir(parents=True, exist_ok=True)
TO_REGION = {s: r for r, ss in REGIONS.items() for s in ss}
A = india_area()
reg = region_grid()
d = xr.open_dataset(DATA / "vulnerability/detection_025.nc")
net, sat = d.nrsc_2022_2023.values, d.iss_cg_2022_2023.values
aa = np.nan_to_num(A)
sm = lambda x: uniform_filter(np.nan_to_num(x) * aa, 3) / np.maximum(uniform_filter(aa, 3), 1e-9)
net_s, sat_s = sm(net), sm(sat)
det = np.where(np.isfinite(A) & (sat_s >= 0.5), net_s / np.maximum(sat_s, 1e-9), np.nan)
st = pd.read_csv(RESULTS / "detection_state.csv", index_col=0)
st["region"] = [TO_REGION.get(s, "") for s in st.index]
st = st[(st.area_km2 > 2e4) & (st.region != "")]
rg = pd.read_csv(RESULTS / "detection_region.csv", index_col=0)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(OUT / f"{name}.png")


# numbers for the text: flashes per year over each region (2022-2023), missed = expected - recorded
pop = pd.read_csv(POP / "pop_state_cell.csv")
P = np.zeros_like(aa)
np.add.at(P, (pop.j, pop.i), pop.population_all)
k0 = np.isfinite(A) & np.isfinite(sat) & np.isfinite(net)
missed = {}
for r in list(REGIONS) + ["India"]:
    k = k0 & ((reg == r) if r != "India" else True)
    n, c = (net * A)[k].sum(), (sat * A)[k].sum()
    missed[r] = (c - n) / 1e6
    print(f"{r:15s} area {A[k].sum() / 1e3:5.0f} thousand km2 ({100 * A[k].sum() / A[k0].sum():4.1f}%), people "
          f"{P[k].sum() / 1e6:5.0f} million ({100 * P[k].sum() / P[k0].sum():4.1f}%); recorded {n / 1e6:5.2f}, expected "
          f"{c / 1e6:5.2f}, missed {(c - n) / 1e6:5.2f} million a year; population-weighted detection "
          f"{(P[k] * net[k]).sum() / (P[k] * sat[k]).sum():.2f}")
print("Share of India's missed flashes: " + ", ".join(f"{r} {100 * missed[r] / missed['India']:.1f}%" for r in REGIONS))

# Figure 1: expected, recorded, ratio
fig = plt.figure(figsize=(183 * MM, 76 * MM))
cmap, norm = classes([0.25, 0.5, 1, 2, 3, 4, 6, 8, 10, 12], BLUE, extend="both")
titles = ("Expected from satellite (ISS LIS)", "Recorded by the LDSN", "Detection efficiency (b divided by a)")
for p, (x0, data, t) in enumerate(zip((0.0, 0.335, 0.67), (sat, net, det), titles)):
    ax = fig.add_axes([x0, 0.16, 0.32, 0.78], projection=PC)
    india_map(ax, ticks=False)
    if p < 2:
        m = field(ax, data, cmap, norm)
    else:
        cmap2, norm2 = classes([0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0], "Purples", extend="both")
        m2 = field(ax, data, cmap2, norm2)
    ax.text(0.045, 1.012, t, transform=ax.transAxes, fontsize=7, color=INK, va="bottom")
    label(ax, "abc"[p], x=0.035, y=1.0)
colorbar(fig, m, fig.add_axes([0.06, 0.11, 0.53, 0.025]),
         "Cloud-to-ground flashes per km$^2$ per year, 2022–2023 (a and b)", fmt="{x:g}")
colorbar(fig, m2, fig.add_axes([0.70, 0.11, 0.26, 0.025]), "Recorded per expected flash", fmt="{x:g}")
save(fig, "fig1_ldsn_detection_map")

# Figure 2: regions, missed flashes, internal signs
fig = plt.figure(figsize=(183 * MM, 58 * MM))
order = rg.drop("India").sort_values("detection").index.tolist() + ["India"]
ax = fig.add_axes([0.095, 0.2, 0.17, 0.68])
bx = fig.add_axes([0.285, 0.2, 0.15, 0.68])
for k, r in enumerate(order):
    v = rg.loc[r]
    c = INK if r == "India" else REGION_COLOR[r]
    ax.barh(k, v.detection, color=c, height=0.62)
    ax.plot([v.detection_lo, v.detection_hi], [k, k], color=INK2 if r != "India" else MUTED, lw=1.0)
    ax.text(v.detection_hi + 0.03, k, f"{v.detection:.2f}", va="center", fontsize=5.8, color=INK)
    if r == "India":
        bx.text(0.06, k, f"India {missed[r]:.1f} million", va="center", fontsize=5.8, color=INK)
    else:
        bx.barh(k, missed[r], color=c, height=0.62)
        bx.text(missed[r] + 0.04, k, f"{100 * missed[r] / missed['India']:.0f}%", va="center", fontsize=5.8,
                color=INK)
ax.set_yticks(range(len(order)), order, fontsize=6)
ax.set_xlim(0, 1.15)
ax.set_xlabel("Recorded per expected\ncloud-to-ground flash")
bx.set_yticks(range(len(order)), [""] * len(order))
bx.set_xlim(0, 2.0)
bx.set_xlabel("Expected flashes not recorded\n(million per year)")
for a_ in (ax, bx):
    a_.tick_params(axis="y", length=0)
ax.set_title("Detection efficiency", fontsize=7, color=INK)
bx.set_title("Missed flashes (share)", fontsize=7, color=INK)
label(ax, "a", x=-0.55, y=1.04)
label(bx, "b", x=-0.03, y=1.04)


def scatter(ax, x, xlabel):
    for r, g in st.groupby("region"):
        ax.scatter(g[x], g.de, s=12, color=REGION_COLOR[r], edgecolor="white", lw=0.4, zorder=3)
    rho = stats.spearmanr(st[x], st.de)
    ax.text(0.03 if rho.statistic > 0 else 0.97, 0.97, f"Rank correlation {rho.statistic:+.2f}\n({len(st)} States/UTs)",
            transform=ax.transAxes, va="top", ha="left" if rho.statistic > 0 else "right", fontsize=5.8, color=INK2)
    ax.set_xlabel(xlabel)
    ax.set_ylim(0, 1.0)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)


cx = fig.add_axes([0.53, 0.2, 0.2, 0.68])
scatter(cx, "records_per_flash", "Records (strokes) per flash")
cx.set_ylabel("Detection efficiency")
cx.set_title("Fewer strokes recorded", fontsize=7, color=INK)
label(cx, "c", x=-0.2, y=1.04)
ex = fig.add_axes([0.78, 0.2, 0.2, 0.68])
st["s20"] = 100 * st.share_ge20kA
scatter(ex, "s20", "Flashes of 20 kA or more (%)")
ex.set_yticklabels([])
ex.set_title("Weak flashes missed", fontsize=7, color=INK)
ex.legend(handles=[Line2D([], [], marker="o", ls="", color=REGION_COLOR[r], markersize=3.5, label=r) for r in REGIONS],
          loc="upper right", bbox_to_anchor=(1.0, 0.8), fontsize=5.2, frameon=False, handletextpad=0.2,
          labelspacing=0.25)
label(ex, "d", x=-0.03, y=1.04)
save(fig, "fig2_ldsn_detection_regions")
