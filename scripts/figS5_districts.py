#!/usr/bin/env python3
"""Supplementary Figure S5: do lightning and farm work coincide? The 640 districts
of the 2011 Census: share of people working in agriculture (cultivators and
agricultural labourers, Census 2011) against exposure, the population-weighted
flash density, from (a) the NRSC network (2022-2025) and (b) satellite-expected
cloud-to-ground lightning (ISS LIS 2022-2023 times the CG fraction from the
IITM network). Colour: region.
Input: results/district2011_table.csv (sr_districts2011.py)
Output: manuscript/supplementary/figS5_districts.png/.pdf
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

from common import REGIONS, RESULTS
from plotting import GRID, INK2, MM, MUTED, REGION_COLOR, label, save

t = pd.read_csv(RESULTS / "district2011_table.csv")
t["region"] = t.region.fillna("")
fig = plt.figure(figsize=(183 * MM, 82 * MM))
for k, (e, name) in enumerate((("E_net", "NRSC network, 2022\u20132025"),
                               ("E_sat", "Satellite-expected CG, 2022\u20132023"))):
    ax = fig.add_axes([0.08 + 0.48 * k, 0.16, 0.4, 0.66])
    for r in REGIONS:
        g = t[t.region == r]
        ax.scatter(100 * g.agri, g[e], s=5, color=REGION_COLOR[r], alpha=0.8, lw=0, zorder=3)
    other = t[~t.region.isin(list(REGIONS))]
    ax.scatter(100 * other.agri, other[e], s=5, color=MUTED, alpha=0.8, lw=0, zorder=4)
    rho = stats.spearmanr(t.agri, t[e])
    ax.set_title(f"{name}: rank correlation {rho.statistic:+.2f} ({len(t)} districts)", fontsize=6.5, loc="left",
                 pad=3)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}")
    ax.set_ylim(0.01 if e == "E_net" else 0.03, 30)
    ax.set_xlim(-1, 55)
    ax.set_xlabel("People working in agriculture (% of population, Census 2011)")
    ax.set_ylabel("Exposure: flashes per km$^2$ per year\nwhere people live")
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    label(ax, "ab"[k], x=-0.14, y=1.02)
fig.legend(handles=[Line2D([], [], marker="o", ls="", color=REGION_COLOR[r], markersize=4, label=r) for r in REGIONS],
           loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=7, fontsize=6, frameon=False, handletextpad=0.2,
           columnspacing=1.2)
print(save(fig, "figS5_districts", sup=True))
