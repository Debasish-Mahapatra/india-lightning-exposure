#!/usr/bin/env python3
"""Lightning exposure by State/UT and district (Survey of India), and lightning
deaths by State/UT (npj Natural Hazards paper).

Flash density per 0.25 degree cell: NRSC mean over 2022-2025 (the years with a
stable network, each month scaled up for its days without data), flashes per
km2 per year; every 1 km WorldPop cell takes the density of its 0.25 degree
cell (sr_population.py). For every State/UT and district:
  density     mean flash density over its area (populated 1 km cells)
  exposure    flash density where people live (population-weighted mean)
  share       share of India's exposure (population times density), %
For States/UTs also NCRB lightning deaths (mean of 2022-2024), per million
people, and ratio: deaths per million divided by exposure, relative to all
India (above 1: more deaths than the state's exposure alone implies), with a
95% interval from the Poisson error of the three years' deaths.
Output: results/state_table.csv, results/district_table.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2

from common import NCRB, POP, RESULTS, india_area, nrsc_clim

A = india_area()
dens = nrsc_clim("flashes").sum(("month", "hour_utc")).values / A


def table(cells, key):
    cells = cells.assign(d=dens[cells.j, cells.i])
    cells["pd"] = cells.population * cells.d
    g = cells.groupby(key)
    t = pd.DataFrame({"area_km2": g.area_km2.sum(), "population_m": g.population.sum() / 1e6,
                      "density": g.apply(lambda x: (x.area_km2 * x.d).sum(), include_groups=False) / g.area_km2.sum(),
                      "exposure": g.pd.sum() / g.population.sum()})
    t["share"] = 100 * g.pd.sum() / cells.pd.sum()
    return t


st = table(pd.read_csv(f"{POP}/pop_state_cell.csv"), "state")
deaths = pd.read_csv(f"{NCRB}/lightning_deaths.csv")
d3 = deaths[deaths.year.between(2022, 2024)].groupby("state")["total"].sum().reindex(st.index)
st["deaths_per_year"] = d3 / 3
st["deaths_per_million"] = st.deaths_per_year / st.population_m
india_rate = st.deaths_per_year.sum() / st.population_m.sum()
india_exp = (st.exposure * st.population_m).sum() / st.population_m.sum()
nat = india_rate / india_exp
st["ratio"] = st.deaths_per_million / st.exposure / nat
lo = np.where(d3 > 0, chi2.ppf(0.025, 2 * d3) / 2, 0)            # Poisson 95% on the 3-year count
hi = chi2.ppf(0.975, 2 * d3 + 2) / 2
st["ratio_lo"] = st.ratio * lo / d3.where(d3 > 0)
st["ratio_hi"] = (hi / 3 / st.population_m) / st.exposure / nat
st.loc[d3 == 0, "ratio_lo"] = 0
st = st.sort_values("deaths_per_year", ascending=False)
st.round(4).to_csv(RESULTS / "state_table.csv")

names = pd.read_csv(f"{POP}/districts.csv", index_col="district")
dc = pd.read_csv(f"{POP}/pop_district_cell.csv")
dt = table(dc[dc.district >= 0], "district")
dt = names.join(dt, how="inner").sort_values("share", ascending=False)
dt.round(4).to_csv(RESULTS / "district_table.csv")

print(f"India: {st.population_m.sum():,.0f} million people; flash density {np.nansum(dens * A) / np.nansum(A):.2f} "
      f"per km2 per year over India's area (2022-2025), {india_exp:.2f} where people live; "
      f"{st.deaths_per_year.sum():,.0f} lightning deaths a year (2022-2024), {india_rate:.2f} per million")
print(st.head(16).round(2).to_string())
print("\nDistricts with the largest share of India's exposure:")
print(dt.head(12).round(2).to_string())
n = (dt.share.cumsum() <= 50).sum() + 1
print(f"\n{n} of {len(dt)} districts hold half of India's exposure, with "
      f"{dt.population_m.iloc[:n].sum():.0f} million people")
# whole official boundary (districts) against the area enumerated by the 2011 Census (States/UTs, with deaths)
pw = dt.population_m.sum()
ew = (dt.exposure * dt.population_m).sum() / pw
print(f"whole official boundary: {pw:,.1f} million people, exposure {ew:.3f}; enumerated area: "
      f"{st.population_m.sum():,.1f} million, exposure {india_exp:.3f}")
# benchmark: what does ranking by lightning add to ranking by population alone?
top = dt.iloc[:n]
rest = dt.iloc[n:]
e_top = (top.exposure * top.population_m).sum() / top.population_m.sum()
e_rest = (rest.exposure * rest.population_m).sum() / rest.population_m.sum()
pop_top = dt.sort_values("population_m", ascending=False).iloc[:n]
print(f"these {n} districts hold {100 * top.population_m.sum() / pw:.1f}% of the people; their exposure "
      f"{e_top:.2f} is {e_top / ew:.2f} times India's and {e_top / e_rest:.2f} times that of the other districts")
print(f"the {n} most populous districts hold {100 * pop_top.population_m.sum() / pw:.1f}% of the people and "
      f"{pop_top.share.sum():.1f}% of the exposure; {len(set(top.index) & set(pop_top.index))} districts are in both sets")
k = (dt.sort_values("population_m", ascending=False).share.cumsum() <= 50).sum() + 1
print(f"ranked by population alone, {k} districts are needed to hold half of India's exposure")
