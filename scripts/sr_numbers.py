#!/usr/bin/env python3
"""Numbers quoted in the text that no figure script prints: flashes and
exposure by region, the regional shares of India's flashes and deaths, the
flashes inside the official boundary by year, the monthly shares and the peak
hour (IST) of each region and season, the Northeast's night-time share, the
link between deaths per unit of exposure and the time of day of exposure
across states, the share of flashes of 20 kA or more by region and month,
and the storm of 11 November 2022 on the Cauvery delta.
Output: printed, and results/numbers.txt
"""
import numpy as np
import pandas as pd
from scipy import stats

from common import DATA, LAT, LON, NCRB, POP, REGIONS, RESULTS, SEASONS, india_area, nrsc_clim, region_grid

A = india_area()
reg = region_grid()
f = nrsc_clim("flashes").sum(("month", "hour_utc")).values
out = []
tot = np.nansum(f)
out.append(f"India 2022-2025: {tot / 1e6:.2f} million flashes a year, {tot / np.nansum(A):.2f} per km2, "
           f"area {np.nansum(A) / 1e6:.3f} million km2")
for y in range(2020, 2026):
    z = np.load(DATA / f"nrsc_flashes/flashes_{y}.npz")
    out.append(f"{y}: {len(z['t']) / 1e6:.2f} million flashes inside the boundary on {len(z['dates'])} days (not scaled)")
st = pd.read_csv(RESULTS / "state_table.csv", index_col=0)
to_region = {s: r for r, ss in REGIONS.items() for s in ss}
st["region"] = [to_region.get(s, "") for s in st.index]
pop = pd.read_csv(POP / "pop_state_cell.csv")
pop["region"] = pop.state.map(to_region).fillna("")
dens = f / A
pop["pd"] = pop.population * dens[pop.j, pop.i]
g = pop.groupby("region")
for r in REGIONS:
    k = reg == r
    s = st[st.region == r]
    exp = g.pd.sum()[r] / g.population.sum()[r]
    out.append(f"{r}: {np.nansum(f[k]) / 1e6:.2f} million flashes a year ({100 * np.nansum(f[k]) / tot:.1f}%), "
               f"{np.nansum(f[k]) / np.nansum(A[k]):.2f} per km2, {g.population.sum()[r] / 1e6:.0f} million people, "
               f"exposure {exp:.2f}, {100 * g.pd.sum()[r] / pop.pd.sum():.1f}% of India's exposure, "
               f"{s.deaths_per_year.sum():.0f} deaths a year ({100 * s.deaths_per_year.sum() / st.deaths_per_year.sum():.1f}%)")
# months and hours: nrsc_clim gives flashes by month and UTC hour; UTC hour h is centred on h + 6 IST
fmh = nrsc_clim("flashes").values                                 # (month, hour_utc, lat, lon)
ist = lambda h: (h + 6) % 24
by_month = np.nansum(fmh, axis=(1, 2, 3))
out.append("India, share of the year's flashes by month: " + ", ".join(f"{m + 1}: {100 * v / by_month.sum():.1f}%" for m, v in enumerate(by_month)))
for r in REGIONS:
    k = reg == r
    mr = np.nansum(fmh[:, :, k], axis=(1, 2))
    peaks = []
    for sname, months in SEASONS.items():
        hr = np.nansum(fmh[[m - 1 for m in months]][:, :, k], axis=(0, 2))
        peaks.append(f"{sname.split(' (')[0]} {ist(np.argmax(hr))}")
    out.append(f"{r}: busiest months " + ", ".join(f"{m + 1} ({100 * mr[m] / mr.sum():.0f}%)" for m in np.argsort(mr)[::-1][:3])
               + "; peak hour IST by season: " + ", ".join(peaks))
hne = np.nansum(fmh[:, :, reg == "Northeast"], axis=(0, 2))
out.append(f"Northeast: {100 * hne[15:24].sum() / hne.sum():.0f}% of flashes between 20:30 and 05:30 IST (UTC hours 15-23)")
# time of day of exposure by state (exposure by UTC hour), against deaths per unit of exposure
fh = np.nansum(fmh, axis=0) / A                                    # (hour_utc, lat, lon) flashes per km2 per year
eh = np.stack([(pop.population * fh[h][pop.j, pop.i]).groupby(pop.state).sum() / pop.groupby("state").population.sum()
               for h in range(24)], axis=1)
eh = pd.DataFrame(eh, index=pop.groupby("state").population.sum().index)
big = st[st.deaths_per_year * 3 >= 10]
for name, hours in (("11:30-19:30 IST", range(6, 14)), ("05:30-17:30 IST", range(0, 12))):
    share = eh.loc[big.index, list(hours)].sum(1) / eh.loc[big.index].sum(1)
    rho, pv = stats.spearmanr(share, big.ratio)
    out.append(f"States with at least 10 deaths in 2022-2024 ({len(big)}): rank correlation of deaths per unit of exposure "
               f"with the share of exposure {name}: {rho:.2f} (p = {pv:.2f})")
# strong flashes: share of 20 kA or more by region and by month
s20 = nrsc_clim("flashes_ge20kA").values                          # (month, lat, lon)
fm = np.nansum(fmh, axis=1)                                        # (month, lat, lon)
out.append("Share of flashes of 20 kA or more by region: " + ", ".join(
    f"{r}: {100 * np.nansum(s20[:, reg == r]) / np.nansum(fm[:, reg == r]):.1f}%" for r in REGIONS))
out.append("Share of flashes of 20 kA or more over India by month: " + ", ".join(
    f"{m + 1}: {100 * np.nansum(s20[m]) / np.nansum(fm[m]):.1f}%" for m in range(12)))
# a single storm: the cell on the Cauvery delta, 11 November 2022
z = np.load(DATA / "nrsc_flashes/flashes_2022.npz")
t = z["t"].astype(np.int64)
j = np.rint((z["lat"] - LAT[0]) / 0.25).astype(int)
i = np.rint((z["lon"] - LON[0]) / 0.25).astype(int)
day = pd.Timestamp("2022-11-11").dayofyear - 1
k = (t // 86400 == day) & (j == np.argmin(abs(LAT - 11.25))) & (i == np.argmin(abs(LON - 79.75)))
hh = np.bincount((t[k] % 86400) // 3600, minlength=24)
best = max(range(20), key=lambda h: hh[h:h + 5].sum())
out.append(f"Cauvery delta cell (11.25 N, 79.75 E), 11 November 2022: {k.sum():,} flashes, {100 * hh[best:best + 5].sum() / k.sum():.0f}% "
           f"within five hours ({ist(best) - 0.5:.1f}-{ist(best + 5) - 0.5:.1f} IST), {np.sum(z['amps'][k] >= 5e4)} of 50 kA or more")
d = pd.read_csv(NCRB / "lightning_deaths.csv")
for y, v in d.groupby("year")[["male", "female", "total"]].sum().iterrows():
    out.append(f"NCRB {y}: {v.total} lightning deaths, {100 * v.male / v.total:.0f}% male")
text = "\n".join(out)
print(text)
(RESULTS / "numbers.txt").write_text(text + "\n")
