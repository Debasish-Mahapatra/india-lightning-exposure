#!/usr/bin/env python3
"""Within states, do deaths follow the number of flashes or the number of days
with lightning where people live? (npj Natural Hazards paper)

For each State/UT and year 2021-2024: population-weighted mean number of days
on which the person's 0.25 degree cell had at least 1, 5 or 20 flashes (NRSC
daily counts inside India, data/nrsc_detail), scaled for days without data,
and the population-weighted flash density. Poisson regression of each year's
deaths on the log of each measure with state fixed effects, 2022-2024, for
the 15 states with at least 60 deaths in 2022-2024.
Output: printed; results/log_sr_lightning_days.txt
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr

from common import NCRB, NRSC_DETAIL, POP

pop = pd.read_csv(POP / "pop_state_cell.csv")
deaths = pd.read_csv(NCRB / "lightning_deaths.csv")
rows = []
for y in (2021, 2022, 2023, 2024):
    d = xr.open_dataset(NRSC_DETAIL / f"nrsc_detail_025_{y}.nc")
    daily = np.nan_to_num(d.daily.values)                          # (days, lat, lon) flashes inside India
    have = np.isfinite(d.daily.values).any(axis=(1, 2)).sum()
    scale = len(d.time) / max(have, 1)
    meas = {f"days{k}": (daily >= k).sum(0) * scale for k in (1, 5, 20)}
    meas["flashes"] = daily.sum(0) * scale
    for s, g in pop.groupby("state"):
        w = g.population.values
        rows.append({"year": y, "state": s, **{k: (v[g.j, g.i] * w).sum() / w.sum() for k, v in meas.items()}})
m = pd.DataFrame(rows).merge(deaths[["year", "state", "total"]], on=["year", "state"])
keep = m[m.year >= 2022].groupby("state").total.sum()        # deaths 2022-2024
mm = m[m.state.isin(keep[keep >= 60].index) & (m.year >= 2022)].copy()
X0 = pd.get_dummies(mm.state, dtype=float)
print(f"{mm.state.nunique()} states, 2022-2024; year-to-year spread (standard deviation of log deviations):")
for k in ("flashes", "days1", "days5", "days20"):
    dev = np.log(mm[k]) - mm.groupby("state")[k].transform(lambda v: np.log(v).mean())
    X = X0.assign(logH=np.log(mm[k]))
    r = sm.GLM(mm.total, X, family=sm.families.Poisson()).fit(cov_type="HC0")
    ci = r.conf_int().loc["logH"]
    print(f"  {k:8s}: spread {100 * dev.std():4.1f}%, slope {r.params['logH']:.2f} ({ci[0]:.2f} to {ci[1]:.2f})")
print(mm.groupby("state")[["flashes", "days1", "days5", "days20"]].mean().round(1).head(15).to_string())
