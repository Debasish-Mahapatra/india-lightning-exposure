#!/usr/bin/env python3
"""Check: how do the paper's main numbers change if every NRSC record counts
as a flash, instead of grouping the records into flashes (nrsc_flashes.py)?
Recomputes each number from data/nrsc_flashes/records_YYYY.npz and
flashes_YYYY.npz in the same way as the paper's scripts (nrsc_clim,
sr_exposure.py, fig5_deaths.py, fig7_rain.py, figS2):
India's density and exposure, regional and seasonal shares, the hour of the
diurnal peak, strong flashes, deaths per unit exposure by state and the fits of
deaths against exposure, districts holding half of the exposure, and flashes
per unit rain and with little rain.
Output: results/check_grouping_effect.csv (and the printed table)
Usage: check_grouping_effect.py
"""
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from common import DATA, LAT, LON, NCRB, POP, REGIONS, RESULTS, SEASONS, YEARS, india_area, region_grid

A = india_area()
inside = np.isfinite(A)
reg = region_grid()
NY, NX = len(LAT), len(LON)
cell_km2 = (111.2 * 0.25) ** 2 * np.cos(np.deg2rad(LAT))[:, None] * np.ones(NX)
big_cell = inside & (np.nan_to_num(A) >= 0.5 * cell_km2)
season_of = {m: s for s, ms in SEASONS.items() for m in ms}
V = ("records", "flashes")

month_dens = {v: {} for v in V}                          # year -> (12, NY, NX) scaled for missing days
daily = {v: [] for v in V}                               # (day, NY, NX) counts on days with a file
hourly = {v: np.zeros((4, len(REGIONS) + 1, 24)) for v in V}   # season, region (+ India), UTC hour
strong = {v: np.zeros(3) for v in V}                     # all, >= 20 kA, >= 50 kA
day_index = []
for y in YEARS:
    for v in V:
        z = np.load(DATA / f"nrsc_flashes/{v}_{y}.npz")
        t = z["t"].astype(np.int64)
        doy = t // 86400
        j = np.clip(np.rint((z["lat"] - LAT[0]) / 0.25).astype(int), 0, NY - 1)
        i = np.clip(np.rint((z["lon"] - LON[0]) / 0.25).astype(int), 0, NX - 1)
        dates = pd.DatetimeIndex(z["dates"])
        month = (pd.Timestamp(f"{y}-01-01") + pd.to_timedelta(doy, "D")).month.values
        ndays = np.array([pd.Period(f"{y}-{m:02d}").days_in_month for m in range(1, 13)])
        have = np.array([np.sum(dates.month == m) for m in range(1, 13)])
        dd = (dates - pd.Timestamp(f"{y}-01-01")).days.values           # days with a file
        pos = np.full(367, -1)
        pos[dd] = np.arange(dd.size)
        assert (pos[doy] >= 0).all(), f"{y}: records on days without a file"
        r_of = np.array([list(REGIONS).index(r) if r else -1 for r in reg[j, i]])
        s_of = np.array([list(SEASONS).index(season_of[m]) for m in range(1, 13)])[month - 1]
        hour = (t % 86400) // 3600
        c = np.bincount(((month - 1) * NY + j) * NX + i, minlength=12 * NY * NX).reshape(12, NY, NX)
        month_dens[v][y] = c * (ndays / have)[:, None, None]
        c = np.bincount((pos[doy] * NY + j) * NX + i, minlength=dd.size * NY * NX)
        daily[v].append(c.reshape(dd.size, NY, NX).astype(np.float32))
        kr = r_of >= 0
        np.add.at(hourly[v], (s_of[kr], r_of[kr], hour[kr]), 1)
        np.add.at(hourly[v], (s_of, len(REGIONS), hour), 1)              # India
        a = z["amps"]
        strong[v] += [t.size, np.sum(a >= 2e4), np.sum(a >= 5e4)]
    day_index += list(dates)
days = pd.DatetimeIndex(day_index)

out = {v: {} for v in V}
pop = pd.read_csv(POP / "pop_state_cell.csv")
deaths = pd.read_csv(NCRB / "lightning_deaths.csv")
dc = pd.read_csv(POP / "pop_district_cell.csv")
dc = dc[dc.district >= 0]
R = xr.concat([xr.open_dataset(DATA / f"imerg_025/imerg_daily_025_{y}.nc")["precipitation"] for y in YEARS], "time")
rain_days = days[days.isin(R.time.to_index())]
Rv = R.sel(time=rain_days).values
for v in V:
    o = out[v]
    dens = sum(month_dens[v].values()).sum(0) / len(YEARS) / A          # flashes per km2 per year
    o["India, per km2 per year"] = np.nansum(dens * A) / np.nansum(A)
    o["Max cell (>= half in India)"] = np.nanmax(np.where(big_cell, dens, np.nan))
    tot = np.nansum(dens * A)
    for r in REGIONS:
        o[f"{r}: share of India's flashes, %"] = 100 * np.nansum((dens * A)[reg == r]) / tot
    for s in SEASONS:
        ms = SEASONS[s]
        o[f"{s.split(' (')[0]}: share of the year, %"] = 100 * sum(np.nansum(month_dens[v][y][[m - 1 for m in ms]] ) for y in YEARS) / (tot * len(YEARS))
    # exposure and deaths (sr_exposure.py, fig5_deaths.py)
    p = pop.assign(pd=pop.population * dens[pop.j, pop.i])
    g = p.groupby("state")
    st = pd.DataFrame({"population_m": g.population.sum() / 1e6, "exposure": g.pd.sum() / g.population.sum()})
    d3 = deaths[deaths.year.between(2022, 2024)].groupby("state")["total"].sum().reindex(st.index)
    st["dpy"] = d3 / 3
    st["dpm"] = st.dpy / st.population_m
    india_exp = (st.exposure * st.population_m).sum() / st.population_m.sum()
    nat = st.dpy.sum() / st.population_m.sum() / india_exp
    st["ratio"] = st.dpm / st.exposure / nat
    o["India exposure (where people live)"] = india_exp
    for s in ("MADHYA PRADESH", "RAJASTHAN", "TELANGANA", "ODISHA", "BIHAR", "KARNATAKA", "WEST BENGAL", "ASSAM", "KERALA"):
        o[f"Deaths per exposure, {s.title()}"] = st.ratio[s]
    big = st[(st.population_m >= 1) & (st.dpy > 0)]
    fit = stats.linregress(np.log(big.exposure), np.log(big.dpm))
    tq = stats.t.ppf(0.975, len(big) - 2)
    o["States: r (log deaths per million, log exposure)"] = fit.rvalue
    o["States: slope"] = fit.slope
    o["States: slope 95% low"] = fit.slope - tq * fit.stderr
    o["States: slope 95% high"] = fit.slope + tq * fit.stderr
    rows = []
    for y in (2022, 2023, 2024):
        dy = month_dens[v][y].sum(0) / A
        py = pop.assign(pd=pop.population * dy[pop.j, pop.i])
        e = py.groupby("state").pd.sum() / py.groupby("state").population.sum()
        rows += [(y, s, x) for s, x in e.items()]
    m = pd.DataFrame(rows, columns=["year", "state", "exposure"]).merge(deaths[["year", "state", "total"]], on=["year", "state"])
    keep_states = deaths[deaths.year.between(2022, 2024)].groupby("state").total.sum()
    m = m[m.state.isin(keep_states[keep_states >= 60].index)].copy()
    m["x"] = np.log(m.exposure) - m.groupby("state").exposure.transform(lambda q: np.log(q).mean())
    m["y"] = np.log(m.total) - m.groupby("state").total.transform(lambda q: np.log(q).mean())
    rr = np.corrcoef(m.x, m.y)[0, 1]
    dof = len(m) - m.state.nunique() - 1
    o["Within states: r"] = rr
    o["Within states: p"] = 2 * stats.t.sf(abs(rr * np.sqrt(dof / (1 - rr * rr))), dof)
    q = dc.assign(pd=dc.population * dens[dc.j, dc.i]).groupby("district").pd.sum().sort_values(ascending=False)
    o["Districts holding half of the exposure"] = int((100 * q.cumsum() / q.sum() <= 50).sum() + 1)
    # rain (fig7_rain.py): days with a file and IMERG
    F = np.concatenate(daily[v])[days.isin(rain_days)]
    mo = rain_days.month.values
    for s, ms in SEASONS.items():
        sel = np.isin(mo, ms)
        f, r = F[sel].sum(0), np.nansum(Rv[sel], 0)
        o[f"{s.split(' (')[0]}: flashes per Mt of rain"] = f[inside].sum() / np.nansum((A * r / 1000)[inside])
    dry = F[Rv < 2.5].sum() / F.sum()
    o["Share with < 2.5 mm that day, %"] = 100 * dry
    pre = np.isin(mo, SEASONS["Pre-monsoon (March-May)"])
    o["Pre-monsoon share with < 2.5 mm, %"] = 100 * F[pre][Rv[pre] < 2.5].sum() / F[pre].sum()
    # diurnal cycle (figS2): hour of the peak, IST (UTC hour h is plotted at h + 6)
    for k, s in enumerate(SEASONS):
        for ri, r in enumerate(list(REGIONS) + ["India"]):
            o[f"Peak hour IST, {s.split(' (')[0]}, {r}"] = (np.argmax(hourly[v][k, ri]) + 6) % 24
    o["Share >= 20 kA, %"] = 100 * strong[v][1] / strong[v][0]
    o["Share >= 50 kA, %"] = 100 * strong[v][2] / strong[v][0]
    o["_dens"] = dens

d0, d1 = out["records"].pop("_dens"), out["flashes"].pop("_dens")
ok = big_cell & np.isfinite(d0) & (d0 > 0)
print(f"Maps: correlation of the density of all records and of flashes, cells at least half in India: "
      f"{np.corrcoef(d0[ok], d1[ok])[0, 1]:.3f}; of their logarithms {np.corrcoef(np.log(d0[ok]), np.log(np.maximum(d1[ok], 1e-3)))[0, 1]:.3f}; "
      f"flashes / records by cell: 5-95% {np.round(np.percentile(d1[ok] / d0[ok], [5, 50, 95]), 2)}")
t = pd.DataFrame(out)
t["change, %"] = 100 * (t.flashes / t.records - 1)
pd.set_option("display.width", 200)
print(t.round(3).to_string())
t.to_csv(RESULTS / "check_grouping_effect.csv", float_format="%.4f")
