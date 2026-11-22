#!/usr/bin/env python3
"""Figure 5: lightning deaths (NCRB, 2022-2024) against exposure (NRSC
2022-2025, flash density where people live), by State/UT.
a: deaths per million people per year against exposure (States/UTs with at
least a million people and at least one death; bars: 95% Poisson intervals of
the death rate); dashed: all-India deaths per unit exposure; solid: fit on log
scales. b: deaths per unit exposure relative to all India (the ratio in
results/state_table.csv; grey: fewer than 10 deaths in 2022-2024). c: year to
year within states (2022-2024; states with at least 60 deaths in 2022-2024):
deviation of each year's deaths and exposure from the state's own mean; line:
the Poisson slope with state fixed effects (results/deaths_models.csv, 95%
interval from the wild cluster bootstrap of sr_deaths_models.py).
Output: manuscript/figures/fig5_deaths.png/.pdf
"""
import numpy as np
from adjustText import adjust_text
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy import stats

from common import NCRB, POP, REGIONS, RESULTS, india_area, nrsc_clim, states
from plotting import (GRID, INK, INK2, MM, MUTED, PC, REGION_COLOR, diverging, colorbar, india_map,
                      label, save)

TO_REGION = {s: r for r, ss in REGIONS.items() for s in ss}
SHORT = {"DELHI (UT)": "Delhi", "JAMMU & KASHMIR": "J&K", "D & N HAVELI AND DAMAN & DIU": "DNH&DD",
         "A & N ISLANDS": "A&N Islands"}
short = lambda s: SHORT.get(s, s.title())
st = pd.read_csv(RESULTS / "state_table.csv", index_col=0)
india_rate = st.deaths_per_year.sum() / st.population_m.sum()
india_exp = (st.exposure * st.population_m).sum() / st.population_m.sum()
nat = india_rate / india_exp

fig = plt.figure(figsize=(183 * MM, 125 * MM))
ax = fig.add_axes([0.075, 0.1, 0.42, 0.84])
big = st[(st.population_m >= 1) & (st.deaths_per_year > 0)]
d3 = big.deaths_per_year * 3
lo = stats.chi2.ppf(0.025, 2 * d3) / 2 / 3 / big.population_m
hi = stats.chi2.ppf(0.975, 2 * d3 + 2) / 2 / 3 / big.population_m
texts = []
for s, r in big.iterrows():
    c = REGION_COLOR.get(TO_REGION.get(s, ""), MUTED)
    ax.plot([r.exposure] * 2, [lo[s], hi[s]], color=c, lw=0.8, alpha=0.8, zorder=2)
    ax.scatter(r.exposure, r.deaths_per_million, s=8 + 0.9 * r.population_m, color=c, alpha=0.85,
               edgecolor="white", linewidth=0.5, zorder=3)
    texts.append(ax.text(r.exposure, r.deaths_per_million, short(s), fontsize=5.8, color=INK, zorder=4,
                         ha="center", va="center", path_effects=[pe.withStroke(linewidth=1.5, foreground="white")]))
x = np.array([0.3, 14])
ax.plot(x, nat * x, color=INK2, lw=0.9, ls="--", zorder=1, label="All-India deaths per unit exposure")
lx, ly = np.log(big.exposure), np.log(big.deaths_per_million)
fit = stats.linregress(lx, ly)
ax.plot(x, np.exp(fit.intercept) * x ** fit.slope, color=INK, lw=1.2, zorder=1,
        label=f"Fit: deaths $\\propto$ exposure$^{{{fit.slope:.2f}}}$, r = {fit.rvalue:.2f}")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.3, 14)
ax.set_ylim(0.08, 12)
for axis in (ax.xaxis, ax.yaxis):
    axis.set_major_formatter(lambda v, _: f"{v:g}")
ax.set_xlabel("Exposure: flashes per km$^2$ per year where people live (NRSC, 2022-2025)")
ax.set_ylabel("Lightning deaths per million people per year (NCRB, 2022-2024)")
ax.legend(loc="upper left", fontsize=6)
fig.canvas.draw()                                               # labels avoid the whole of each dot:
pts = []                                                        # its centre and 12 points on its edge
for s, r in big.iterrows():
    cx, cy = ax.transData.transform((r.exposure, r.deaths_per_million))
    rad = np.sqrt(8 + 0.9 * r.population_m) / 2 * fig.dpi / 72
    pts += [(cx, cy)] + [(cx + rad * np.cos(a), cy + rad * np.sin(a)) for a in np.arange(12) * np.pi / 6]
pts = ax.transData.inverted().transform(np.array(pts))
adjust_text(texts, x=pts[:, 0], y=pts[:, 1], ax=ax, expand=(1.15, 1.4),
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.4), min_arrow_len=8)
ax.grid(color=GRID, lw=0.4, which="major")
ax.set_axisbelow(True)
label(ax, "a", x=-0.1, y=1.0)
tq = stats.t.ppf(0.975, len(big) - 2)
print(f"{len(big)} states: log-log fit slope {fit.slope:.2f} (95% interval {fit.slope - tq * fit.stderr:.2f} to "
      f"{fit.slope + tq * fit.stderr:.2f}), r = {fit.rvalue:.2f}, "
      f"p = {fit.pvalue:.1e}; India {india_rate:.2f} deaths per million, exposure {india_exp:.2f}")

bx = fig.add_axes([0.55, 0.43, 0.44, 0.56], projection=PC)
india_map(bx, ticks=False)
geo = states().set_index("state").join(st[["ratio", "deaths_per_year"]])
geo["geometry"] = geo.geometry.simplify(0.005)                  # about 500 m, small files
levels = [0.25, 0.4, 0.6, 0.8, 1.25, 1.67, 2.5, 4]
cmap, norm = diverging(levels)
few = geo.deaths_per_year * 3 < 10
geo[~few].plot(ax=bx, column="ratio", cmap=cmap, norm=norm, transform=PC, edgecolor="white", lw=0.3, zorder=1)
geo[few].plot(ax=bx, color="#dcdbd6", transform=PC, edgecolor="white", lw=0.3, zorder=1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cax = fig.add_axes([0.6, 0.39, 0.34, 0.013])
colorbar(fig, sm, cax, "Deaths per unit exposure, relative to all India\n(grey: fewer than 10 deaths in 2022-2024)",
         fmt="{x:g}")
label(bx, "b", x=0.06, y=0.93)

# c: year to year within states
A = india_area()
dens = {y: nrsc_clim("flashes", years=[y]).sum(("month", "hour_utc")).values / A for y in range(2022, 2025)}
pop = pd.read_csv(POP / "pop_state_cell.csv")
deaths = pd.read_csv(NCRB / "lightning_deaths.csv")
rows = []
for y, d in dens.items():
    p = pop.assign(pd=pop.population * d[pop.j, pop.i])
    e = p.groupby("state").pd.sum() / p.groupby("state").population.sum()
    rows += [(y, s, v) for s, v in e.items()]
m = pd.DataFrame(rows, columns=["year", "state", "exposure"]).merge(deaths[["year", "state", "total"]], on=["year", "state"])
keep = m.groupby("state").total.sum()
m = m[m.state.isin(keep[keep >= 60].index) & (m.year >= 2022)].copy()
m["x"] = np.log(m.exposure) - m.groupby("state").exposure.transform(lambda v: np.log(v).mean())
m["y"] = np.log(m.total) - m.groupby("state").total.transform(lambda v: np.log(v).mean())
r = np.corrcoef(m.x, m.y)[0, 1]
dof = len(m) - m.state.nunique() - 1
t = r * np.sqrt(dof / (1 - r * r))
p = 2 * stats.t.sf(abs(t), dof)
wf = pd.read_csv(RESULTS / "deaths_models.csv")
wf = wf[wf["sample"] == "within60"].iloc[0]
slope = wf.slope
cx = fig.add_axes([0.64, 0.08, 0.32, 0.19])
for s, g in m.groupby("state"):
    cx.scatter(100 * (np.exp(g.x) - 1), 100 * (np.exp(g.y) - 1), s=9,
               color=REGION_COLOR.get(TO_REGION.get(s, ""), MUTED), alpha=0.85, lw=0)
xx = np.linspace(m.x.min(), m.x.max(), 20)
cx.plot(100 * (np.exp(xx) - 1), 100 * (np.exp(slope * xx) - 1), color=INK, lw=1.0)
cx.axhline(0, color=GRID, lw=0.6, zorder=0)
cx.axvline(0, color=GRID, lw=0.6, zorder=0)
cx.set_xlabel("Exposure, change from geometric mean (%)", fontsize=6.3)
cx.set_ylabel("Deaths, change from\ngeometric mean (%)", fontsize=6.3)
cx.text(0.98, 0.04, f"{m.state.nunique()} states, 2022-2024; r = {r:.2f}\nPoisson coefficient {slope:.2f} "
        f"({wf.lo:.2f} to {wf.hi:.2f})", transform=cx.transAxes, va="bottom", ha="right", fontsize=5.8, color=INK2)
label(cx, "c", x=-0.3, y=1.0)
print(f"within states: r = {r:.2f}, p = {p:.3f} ({dof} degrees of freedom, as if independent), least-squares slope "
      f"{np.polyfit(m.x, m.y, 1)[0]:.2f}; Poisson slope {slope:.2f} ({wf.lo:.2f} to {wf.hi:.2f}, wild cluster bootstrap)")
print(save(fig, "fig5_deaths"))
