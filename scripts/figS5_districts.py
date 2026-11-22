#!/usr/bin/env python3
"""Supplementary Figure S5: where hazard and vulnerability meet.
a, b: do lightning and farm work coincide? The 640 districts of the 2011 Census: share of people working in
agriculture (cultivators and agricultural labourers, Census 2011) against exposure, the population-weighted
flash density, from (a) the NRSC network (2022-2025) and (b) satellite-expected cloud-to-ground lightning
(ISS LIS 2022-2023 times the CG fraction from the IITM network). Colour: region.
c: deaths per unit residential exposure (network), relative to India, against literacy (literate people as a
share of the population, Census 2011), States/UTs with at least 10 deaths in 2022-2024; drawn like Fig. 6c
(symbol size: population; open symbols: the network records less than half of the satellite-expected CG flashes).
Input: results/district2011_table.csv (sr_districts2011.py), results/vulnerability_state.csv,
detection_state.csv, deaths_ratios_by_measure.csv
Output: manuscript/supplementary/figS5_districts.png/.pdf
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from scipy import stats

from common import REGIONS, RESULTS
from plotting import GRID, INK, INK2, MM, MUTED, REGION_COLOR, label, save

t = pd.read_csv(RESULTS / "district2011_table.csv")
t["region"] = t.region.fillna("")
fig = plt.figure(figsize=(183 * MM, 150 * MM))
for k, (e, name) in enumerate((("E_net", "NRSC network, 2022–2025"),
                               ("E_sat", "Satellite-expected CG, 2022–2023"))):
    ax = fig.add_axes([0.085 + 0.49 * k, 0.6, 0.37, 0.31])
    for r in REGIONS:
        g = t[t.region == r]
        ax.scatter(100 * g.agri, g[e], s=4, color=REGION_COLOR[r], alpha=0.8, lw=0, zorder=3)
    other = t[~t.region.isin(list(REGIONS))]
    ax.scatter(100 * other.agri, other[e], s=4, color=MUTED, alpha=0.8, lw=0, zorder=4)
    rho = stats.spearmanr(t.agri, t[e])
    ax.set_title(f"{name}: rank correlation {rho.statistic:+.2f} ({len(t)} districts)", fontsize=6.5, loc="left", pad=3)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}")
    ax.set_ylim(0.01 if e == "E_net" else 0.03, 30)
    ax.set_xlim(-1, 55)
    ax.set_xlabel("Agricultural workers (% of population, Census 2011)")
    ax.set_ylabel("Exposure: flashes per km$^2$ per year\nwhere people live")
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    label(ax, "ab"[k], x=-0.14, y=1.03)
    print(f"{'ab'[k]}: {name}, Spearman {rho.statistic:+.3f}, {len(t)} districts")

# c: deaths per unit residential exposure against literacy, by state
TO_REGION = {s: r for r, ss in REGIONS.items() for s in ss}
SHORT = {"MADHYA PRADESH": "MP", "WEST BENGAL": "WB", "UTTAR PRADESH": "UP", "TAMIL NADU": "TN",
         "ANDHRA PRADESH": "AP", "HIMACHAL PRADESH": "HP"}
v = pd.read_csv(RESULTS / "vulnerability_state.csv", index_col=0).join(
    pd.read_csv(RESULTS / "detection_state.csv", index_col=0)[["de"]])
r10 = pd.read_csv(RESULTS / "deaths_ratios_by_measure.csv", index_col=0)[["R_E"]].join(v[["pop_m", "census_literacy", "de"]])
r10["x"] = 100 * r10.census_literacy
r10["region"] = [TO_REGION.get(s, "") for s in r10.index]
cx = fig.add_axes([0.085, 0.075, 0.52, 0.4])
for s, r in r10.iterrows():
    c = REGION_COLOR.get(r.region, MUTED)
    cx.scatter(r.x, r.R_E, s=8 + 0.7 * r.pop_m, facecolor=c if r.de >= 0.5 else "white", edgecolor=c, lw=0.9,
               alpha=0.9, zorder=3)
cx.axhline(1, color=MUTED, lw=0.8, ls="--", zorder=1)
# labels placed by hand (offset in points, alignment), with a leader line when set well away from the dot
LEADER = dict(arrowstyle="-", color=INK2, lw=0.4, shrinkA=0, shrinkB=0)
LAB = {"MADHYA PRADESH": (7, 0, "left"), "BIHAR": (-8, 0, "right"), "RAJASTHAN": (-6, 2, "right"),
       "JHARKHAND": (-5, -7, "right"), "TELANGANA": (-2, 16, "right"), "CHHATTISGARH": (10, 12, "left"),
       "ODISHA": (6, 0, "left"), "KARNATAKA": (6, 0, "left"), "UTTAR PRADESH": (-10, -3, "right"),
       "ANDHRA PRADESH": (6, 0, "left"), "GUJARAT": (-6, 0, "right"), "MAHARASHTRA": (7, 2, "left"),
       "TAMIL NADU": (-6, -3, "right"), "HIMACHAL PRADESH": (5, -2, "left"), "MEGHALAYA": (-4, 0, "right"),
       "ASSAM": (-5, 0, "right"), "WEST BENGAL": (-7, -2, "right"), "UTTARAKHAND": (6, 2, "left"),
       "TRIPURA": (5, 0, "left"), "PUNJAB": (-5, 0, "right"), "HARYANA": (5, -2, "left"), "KERALA": (0, 8, "center")}
for s, (ox, oy, ha) in LAB.items():
    r = r10.loc[s]
    cx.annotate(SHORT.get(s, s.title()), (r.x, r.R_E), xytext=(ox, oy), textcoords="offset points", ha=ha, va="center",
                fontsize=5.8, color=INK, zorder=4, path_effects=[pe.withStroke(linewidth=1.4, foreground="white")],
                arrowprops=LEADER if abs(ox) + abs(oy) > 13 else None)
rho = stats.spearmanr(r10.census_literacy, r10.R_E)
cx.set_yscale("log")
cx.yaxis.set_major_formatter(lambda y, _: f"{y:g}")
cx.set_yticks([0.25, 0.5, 1, 2])
cx.set_ylim(0.18, 3.2)
cx.set_xlim(43, 90)
cx.set_xlabel("Literate people (% of population, Census 2011)")
cx.set_ylabel("Deaths per unit exposure,\nrelative to India")
cx.set_title(f"States and Union Territories: rank correlation \u2212{abs(rho.statistic):.2f} ({len(r10)} with at least 10 deaths)",
             fontsize=6.5, loc="left", pad=3)
cx.grid(color=GRID, lw=0.5)
cx.set_axisbelow(True)
label(cx, "c", x=-0.105, y=1.03)
print(f"c: deaths per unit residential exposure against literacy, Spearman {rho.statistic:+.3f} (p = {rho.pvalue:.4f}), "
      f"{len(r10)} States/UTs; literacy {r10.x.min():.0f}-{r10.x.max():.0f}%")
fig.legend(handles=[Line2D([], [], marker="o", ls="", color=REGION_COLOR[r], markersize=4, label=r) for r in REGIONS],
           loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=7, fontsize=6, frameon=False, handletextpad=0.2,
           columnspacing=1.2)
fig.legend(handles=[Line2D([], [], marker="o", ls="", markerfacecolor=INK2, markeredgecolor=INK2, markersize=4,
                           label="Network records at least half of the\nsatellite-expected cloud-to-ground flashes"),
                    Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=INK2, markersize=4,
                           label="Network records less than half"),
                    Line2D([], [], marker="o", ls="", markerfacecolor=INK2, markeredgecolor=INK2, markersize=2.2,
                           label="Symbol size: population")],
           loc="upper left", bbox_to_anchor=(0.635, 0.46), fontsize=6, frameon=False, handletextpad=0.4, labelspacing=0.9,
           title="Panel c", title_fontsize=6.5, alignment="left")
print(save(fig, "figS5_districts", sup=True))
