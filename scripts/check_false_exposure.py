#!/usr/bin/env python3
"""How much do the doubtful flashes (check_false_flashes.py: no cloud colder than
273 K within 20 km in the MERGIR image nearest in time) contribute to exposure,
to exposure while people are outside the home and during outdoor work, and do
the main count models change without them? (npj Natural Hazards paper)

The tables of check_false_flashes.py list the flashes of each day in the order
of flashes_YYYY.npz, leaving out those whose nearest image was missing; they
are matched back to the flashes in that order by month, time of day and peak
current. Doubtful flashes are then gridded by month and UTC hour like
data/nrsc_detail (each month scaled for its days without data) and removed from
the flash density before the exposures are recomputed as in sr_vulnerability.py.
Models: negative binomial across the 30 States/UTs of sr_deaths_models.py
(exposure, exposure during outdoor work, each with the farm share), and the
Poisson slope with state fixed effects within the 15 states with at least 60
deaths in 2021-2024.
Output: printed (results/log_check_false_exposure.txt)
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr

from common import DATA, LAT, LON, NCRB, NRSC_DETAIL, POP, RESULTS, india_area

A = india_area()
YEARS = range(2022, 2026)


def match(year):
    """Doubtful flag (True/False) of every flash of the year, NaN where not checked."""
    z = np.load(DATA / f"nrsc_flashes/flashes_{year}.npz")
    t = z["t"].astype(np.int64)
    month = pd.to_datetime(f"{year}-01-01") + pd.to_timedelta(t // 86400, unit="D")
    fm = month.month.values
    fh = ((t % 86400) / 3600 + 5.5) % 24
    fa = np.nan_to_num(z["amps"].astype(float), nan=-1.0)
    d = pd.read_csv(RESULTS / f"check_false_flashes_{year}.csv.gz", usecols=["month", "hour_ist", "amps", "d273"])
    rm, rh, ra = d.month.values, d.hour_ist.values, np.nan_to_num(d.amps.values, nan=-1.0)
    rs = (d.d273.values > 20)
    flag = np.full(t.size, np.nan)
    r, nr = 0, len(d)
    fm_, fh_, fa_ = fm.tolist(), fh.tolist(), fa.tolist()
    rm_, rh_, ra_, rs_ = rm.tolist(), rh.tolist(), ra.tolist(), rs.tolist()
    for k in range(t.size):
        if r >= nr:
            break
        if fm_[k] == rm_[r] and abs(fh_[k] - rh_[r]) <= 0.051 and abs(fa_[k] - ra_[r]) <= 0.051:
            flag[k] = rs_[r]
            r += 1
    print(f"{year}: {t.size:,} flashes, {len(d):,} checked rows, {r:,} matched; doubtful "
          f"{100 * np.nanmean(flag):.2f}% of the checked flashes")
    assert r == nr, (r, nr)
    j = np.rint((z["lat"] - LAT[0]) / 0.25).astype(int)
    i = np.rint((z["lon"] - LON[0]) / 0.25).astype(int)
    return fm, (t % 86400) // 3600, j, i, flag


def doubtful_density(year):
    """Density of doubtful flashes by month and UTC hour (per km2 per year), scaled like nrsc_clim."""
    fm, hr, j, i, flag = match(year)
    g = np.zeros((12, 24, len(LAT), len(LON)))
    s = flag == 1
    np.add.at(g, (fm[s] - 1, hr[s], j[s], i[s]), 1.0)
    d = xr.open_dataset(NRSC_DETAIL / f"nrsc_detail_025_{year}.nc")
    ndays = np.array([pd.Period(f"{year}-{m:02d}").days_in_month for m in range(1, 13)])
    tot = d["flashes"].values
    return g * (ndays / d["days"].values)[:, None, None, None], tot * (ndays / d["days"].values)[:, None, None, None]


sus, tot = {}, {}
for y in YEARS:
    sus[y], tot[y] = doubtful_density(y)
pop = pd.read_csv(POP / "pop_state_cell.csv")
deaths = pd.read_csv(NCRB / "lightning_deaths.csv")
tus = pd.read_csv(DATA / "vulnerability/tus_outside.csv", dtype={"sex": str})
tus = tus[tus.sex.isin(["1", "2"])]


def outside(col):
    d = tus.assign(a=tus[col] * tus.weight)
    share = d.groupby(["state", "subround", "slot"]).a.sum() / d.groupby(["state", "subround", "slot"]).weight.sum()
    india = d.groupby(["subround", "slot"]).a.sum() / d.groupby(["subround", "slot"]).weight.sum()
    s1 = ((np.arange(24) * 60 + 330) % 1440) // 30
    s2 = (s1 + 1) % 48
    res = {}
    for s in pop.state.unique():
        x = share.loc[s].unstack().values if s in share.index.get_level_values(0) else india.unstack().values
        res[s] = 0.5 * (x[:, s1] + x[:, s2])
    return res


SUB = np.arange(12) // 3
O = {c: outside(c) for c in ("out_noschool", "out_open")}


def exposures(dens):
    """Population-weighted density by state: total, while outside, during outdoor work."""
    f = np.nan_to_num(dens / A)
    out = {}
    for s, g in pop.groupby("state"):
        w = g.population.values
        if w.sum() <= 0:
            continue
        rho = np.einsum("mhc,c->mh", f[:, :, g.j.values, g.i.values], w) / w.sum()
        out[s] = (rho.sum(), (O["out_noschool"][s][SUB] * rho).sum(), (O["out_open"][s][SUB] * rho).sum())
    return pd.DataFrame(out, index=["E", "E_out", "E_open"]).T


mean = lambda d: sum(d[y] for y in YEARS) / len(YEARS)
all_ = exposures(mean(tot))
bad = exposures(mean(sus))
wpop = pop.groupby("state").population.sum()
wpop = wpop[wpop > 0]
print("\nShare of India's exposure from doubtful flashes (2022-2025, enumerated area, population weighted):")
for c, lab in (("E", "where people live"), ("E_out", "while people are outside"), ("E_open", "during outdoor work")):
    print(f"  {lab:28s} {100 * (bad[c] * wpop).sum() / (all_[c] * wpop).sum():.2f}%")
r = (bad.E / all_.E).reindex(wpop.index)
big_states = wpop[wpop >= 1e6].index
print(f"  by State/UT with at least a million people: {100 * r[big_states].min():.2f}-{100 * r[big_states].max():.2f}% "
      f"(largest: {', '.join(f'{s.title()} {100 * v:.1f}%' for s, v in r[big_states].sort_values(ascending=False).head(4).items())})")
rm = sum(np.nansum(sus[y], (1, 2, 3)) for y in YEARS) / sum(np.nansum(tot[y], (1, 2, 3)) for y in YEARS)
print("  share of the flashes by month (%): " + ", ".join(f"{m + 1}: {100 * v:.1f}" for m, v in enumerate(rm)))

# count models with and without doubtful flashes
v = pd.read_csv(RESULTS / "vulnerability_state.csv", index_col=0)
clean = exposures(mean(tot) - mean(sus))
t = v[["pop_m", "deaths", "census_agri", "E", "E_open"]].copy()
print(f"\ncheck: recomputed exposure against vulnerability_state.csv, largest difference "
      f"{np.nanmax(np.abs(all_.E.reindex(t.index) / t.E - 1)):.4f} (E), "
      f"{np.nanmax(np.abs(all_.E_open.reindex(t.index) / t.E_open - 1)):.4f} (E_open)")
big = t[(t.pop_m >= 1) & t.deaths.notna()].copy()
big["n3"] = np.rint(big.deaths * 3)
big["offset"] = np.log(3 * big.pop_m * 1e6)


def nb(df, cols):
    X = sm.add_constant(df[cols])
    ml = sm.NegativeBinomial(df.n3, X, offset=df.offset).fit(disp=0, maxiter=200)
    res = sm.GLM(df.n3, X, family=sm.families.NegativeBinomial(alpha=ml.params["alpha"]), offset=df.offset).fit()
    ci = res.conf_int()
    return ", ".join(f"{c} {res.params[c]:.2f} ({ci.loc[c, 0]:.2f} to {ci.loc[c, 1]:.2f})" for c in cols) + \
        f"; AIC {ml.aic:.1f}"


print(f"\nNegative binomial across {len(big)} States/UTs, with all flashes and without the doubtful ones:")
for lab, ex in (("all flashes", all_), ("without doubtful flashes", clean)):
    b = big.assign(logE=np.log(ex.E.reindex(big.index)), logW=np.log(ex.E_open.reindex(big.index)))
    print(f"  {lab}:")
    for name, cols in (("exposure", ["logE"]), ("exposure + farm share", ["logE", "census_agri"]),
                       ("exposure during outdoor work", ["logW"]),
                       ("exposure during outdoor work + farm share", ["logW", "census_agri"])):
        print(f"    {name:44s} {nb(b, cols)}")

# within states, 2022-2024
rows = []
for y in (2021, 2022, 2023, 2024):
    if y == 2021:
        continue
    for lab, dens in (("all", tot[y]), ("clean", tot[y] - sus[y])):
        e = exposures(dens).E
        rows += [(y, s, lab, val) for s, val in e.items()]
m = pd.DataFrame(rows, columns=["year", "state", "version", "E"]).merge(deaths[["year", "state", "total"]],
                                                                       on=["year", "state"])
keep = deaths[deaths.year.between(2021, 2024)].groupby("state").total.sum()
m = m[m.state.isin(keep[keep >= 60].index)]
print(f"\nWithin {m.state.nunique()} states (Poisson, state fixed effects, 2022-2024):")
for lab in ("all", "clean"):
    mm = m[m.version == lab]
    X = pd.get_dummies(mm.state, dtype=float)
    X["logE"] = np.log(mm.E.values)
    res = sm.GLM(mm.total.values, X, family=sm.families.Poisson()).fit()
    print(f"  {'all flashes' if lab == 'all' else 'without doubtful flashes':26s} slope {res.params['logE']:.3f}")
