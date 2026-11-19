#!/usr/bin/env python3
"""Four figures for a newsletter article on how completely NRSC's Lightning
Detection Sensor Network (LDSN) records cloud-to-ground (CG) lightning, from
the satellite comparison of sr_detection.py (2022-2023).
1: the estimate in four panels (ISS LIS total lightning L, expected CG flashes
   N_exp, CG flashes recorded by the LDSN N_rec, detection efficiency
   DE = N_rec / N_exp), with India's mean of each.
2: share recorded by region, as a map.
3: CG flashes recorded and missed per year by region.
4: flashes recorded per year by region, 2020-2025, as the network grew.
Inputs: data/vulnerability/detection_025.nc, results/detection_region.csv,
data/nrsc_detail
Output: notes/ldsn_newsletter/nl_fig1_method ... nl_fig4_growth (.png, .pdf)
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, Patch
from scipy.ndimage import uniform_filter

from common import DATA, REGIONS, RESULTS, WS, india_area, nrsc_clim, region_grid, states
from plotting import BLUE, INK, INK2, MM, MUTED, PC, REGION_COLOR, classes, colorbar, field, india_map

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 7.5})
OUT = WS / "notes/ldsn_newsletter"
OUT.mkdir(parents=True, exist_ok=True)
A = india_area()
aa = np.nan_to_num(A)
reg = region_grid()
d = xr.open_dataset(DATA / "vulnerability/detection_025.nc")
tot, cg, net = d.iss_2022_2023.values, d.iss_cg_2022_2023.values, d.nrsc_2022_2023.values
rg = pd.read_csv(RESULTS / "detection_region.csv", index_col=0)
k0 = np.isfinite(A) & np.isfinite(cg) & np.isfinite(net) & np.isfinite(tot)
mean = lambda x: (x * A)[k0].sum() / A[k0].sum()
DET_LEVELS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(OUT / f"{name}.png")


# 1: the estimate in four steps
sm = lambda x: uniform_filter(np.nan_to_num(x) * aa, 3) / np.maximum(uniform_filter(aa, 3), 1e-9)
det = np.where(np.isfinite(A) & (sm(cg) >= 0.5), sm(net) / np.maximum(sm(cg), 1e-9), np.nan)
print(f"India: all lightning {mean(tot):.1f}, expected CG {mean(cg):.2f}, recorded {mean(net):.2f} flashes per km2 "
      f"per year; share recorded {mean(net) / mean(cg):.3f}; CG fraction {mean(cg) / mean(tot):.3f}")
fig = plt.figure(figsize=(183 * MM, 58 * MM))
W, X0 = 0.215, (0.0, 0.262, 0.524, 0.786)
panels = [("a", "ISS LIS total lightning, $L$", tot, f"{mean(tot):.1f}"),
          ("b", "Expected CG flashes, $N_\\mathrm{exp}$", cg, f"{mean(cg):.1f}"),
          ("c", "LDSN CG flashes, $N_\\mathrm{rec}$", net, f"{mean(net):.1f}"),
          ("d", "DE = $N_\\mathrm{rec}\\,/\\,N_\\mathrm{exp}$", det, f"{100 * mean(net) / mean(cg):.0f}%")]
cm_tot = classes([2, 5, 10, 15, 20, 30, 40, 60], "Oranges", extend="both")
cm_cg = classes([0.25, 0.5, 1, 2, 3, 4, 6, 8, 10, 12], BLUE, extend="both")
cm_det = classes(DET_LEVELS, "Purples", extend="both")
maps = []
for p, (x0, (letter, title, data, india)) in enumerate(zip(X0, panels)):
    ax = fig.add_axes([x0, 0.27, W, 0.66], projection=PC)
    india_map(ax, ticks=False)
    ax.spines["geo"].set_visible(False)
    cmap, norm = (cm_tot, cm_cg, cm_cg, cm_det)[p]
    maps.append(field(ax, data, cmap, norm))
    ax.text(0.0, 1.02, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", color=INK, va="bottom")
    ax.text(0.09, 1.02, title, transform=ax.transAxes, fontsize=7.5, color=INK, va="bottom")
    ax.text(0.0, 0.97, f"India\n{india}", transform=ax.transAxes, fontsize=7.5, color=INK, va="top",
            linespacing=1.1)
for x in (0.215, 0.477, 0.739):
    fig.add_artist(FancyArrowPatch((x + 0.006, 0.6), (x + 0.042, 0.6), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=12, color=MUTED, lw=1.4))
colorbar(fig, maps[0], fig.add_axes([0.02, 0.12, 0.18, 0.03]), "Flashes km$^{-2}$ yr$^{-1}$ (a)", fmt="{x:g}")
colorbar(fig, maps[1], fig.add_axes([0.29, 0.12, 0.44, 0.03]), "CG flashes km$^{-2}$ yr$^{-1}$ (b, c)", fmt="{x:g}")
colorbar(fig, maps[3], fig.add_axes([0.80, 0.12, 0.18, 0.03]), "DE (d)", ticks=[0.1, 0.3, 0.6, 1.0],
         fmt="{x:.0%}")
save(fig, "nl_fig1_method")

# 2: share recorded by region, as a map
st = states()
st["region"] = st.state.map({s: r for r, ss in REGIONS.items() for s in ss})
rgeo = st.dropna(subset=["region"]).dissolve("region")
rgeo["geometry"] = rgeo.geometry.buffer(0.02).buffer(-0.02).simplify(0.005)   # close slivers, as in fig2_cycles
cmap, norm = cm_det
fig = plt.figure(figsize=(88 * MM, 92 * MM))
ax = fig.add_axes([0, 0.12, 1, 0.88], projection=PC)
india_map(ax, ticks=False)
ax.spines["geo"].set_visible(False)
for r, g in rgeo.iterrows():
    ax.add_geometries([g.geometry], PC, facecolor=cmap(norm(rg.loc[r, "detection"])),
                      edgecolor="white", lw=1.0, zorder=1)
POS = {"Himalaya": (77.6, 33.6), "Northwest": (72.9, 26.4), "Gangetic Plain": (80.6, 26.6),
       "Central": (78.0, 21.3), "East": (85.4, 21.7), "Northeast": (94.6, 28.6), "South": (77.0, 14.6)}
for r, (x, y) in POS.items():
    v = rg.loc[r, "detection"]
    c = "white" if v >= 0.4 else INK
    ax.text(x, y, f"{r}\n{100 * v:.0f}%", transform=PC, ha="center", va="center", fontsize=7.5, fontweight="bold",
            color=c, linespacing=1.05, zorder=6,
            path_effects=[pe.withStroke(linewidth=1.5, foreground=INK if c == "white" else "white")])
ax.text(0.03, 0.10, f"India\n{100 * rg.loc['India', 'detection']:.0f}%", transform=ax.transAxes, fontsize=11,
        fontweight="bold", color=INK, va="bottom")
ax.text(0.03, 0.09, f"range {100 * rg.loc['India', 'detection_lo']:.0f}–{100 * rg.loc['India', 'detection_hi']:.0f}%",
        transform=ax.transAxes, fontsize=6.5, color=INK2, va="top")
sc = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
colorbar(fig, sc, fig.add_axes([0.12, 0.08, 0.76, 0.025]),
         "Detection efficiency (DE) of the LDSN, 2022–2023", fmt="{x:.0%}")
save(fig, "nl_fig2_map")

# 3: recorded and missed flashes per year by region
rows = []
for r in REGIONS:
    k = k0 & (reg == r)
    rows.append((r, (net * A)[k].sum() / 1e6, (cg * A)[k].sum() / 1e6))
t = pd.DataFrame(rows, columns=["region", "recorded", "expected"]).set_index("region")
t = t.sort_values("expected")
print(t.round(2).to_string())
fig, ax = plt.subplots(figsize=(88 * MM, 62 * MM))
for k, (r, v) in enumerate(t.iterrows()):
    c = REGION_COLOR[r]
    ax.barh(k, v.recorded, color=c, height=0.66)
    ax.barh(k, v.expected - v.recorded, left=v.recorded, color=mcolors.to_rgba(c, 0.25), height=0.66,
            edgecolor=c, lw=0.6)
    ax.text(v.expected + 0.05, k, f"DE {100 * v.recorded / v.expected:.0f}%", va="center", fontsize=7,
            color=INK2)
ax.set_yticks(range(len(t)), t.index)
ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 4.2)
ax.set_xlabel("CG flashes per year, 2022–2023 (millions)")
ax.legend(handles=[Patch(color=INK2, label="Recorded, $\\sum N_\\mathrm{rec}A$"),
                   Patch(facecolor=mcolors.to_rgba(INK2, 0.25), edgecolor=INK2, lw=0.6,
                         label="Missed, $\\sum (N_\\mathrm{exp} - N_\\mathrm{rec})A$")],
          loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, handlelength=1.2)
save(fig, "nl_fig3_missed")

# 4: flashes recorded per year, 2020-2025
years = range(2020, 2026)
by = pd.DataFrame({y: {r: np.nansum(nrsc_clim("flashes", years=[y]).sum(("month", "hour_utc")).values[reg == r])
                       for r in REGIONS} for y in years}) / 1e6
allin = {y: np.nansum(nrsc_clim("flashes", years=[y]).sum(("month", "hour_utc")).values) / 1e6 for y in years}
print(by.round(2).to_string())
print({y: round(v, 2) for y, v in allin.items()})
fig, ax = plt.subplots(figsize=(88 * MM, 62 * MM))
order = ["Central", "East", "South", "Gangetic Plain", "Northwest", "Northeast", "Himalaya"]
base = np.zeros(len(years))
for r in order:
    ax.bar(list(years), by.loc[r].values, bottom=base, color=REGION_COLOR[r], width=0.7, label=r)
    base += by.loc[r].values
for y in years:
    ax.text(y, allin[y] + 0.12, f"{allin[y]:.1f}", ha="center", fontsize=7.5, fontweight="bold", color=INK)
ax.set_ylim(0, 9)
ax.set_ylabel("LDSN flashes over India\n(millions per year)")
ax.set_xticks(list(years))
h, l = ax.get_legend_handles_labels()
ax.legend(h[::-1], l[::-1], loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=7,
          handlelength=1.0, labelspacing=0.3)
save(fig, "nl_fig4_growth")
