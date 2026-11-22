#!/usr/bin/env python3
"""Do lightning hazard and vulnerability coincide? District-level test with the
640 districts of the 2011 Census (npj Natural Hazards paper).

Every WorldPop 2020 1 km cell inside India's official boundary goes to the
2011 district containing its centre and takes the flash density of its 0.25
degree cell: the network (NRSC, 2022-2025) and satellite cloud-to-ground
lightning (ISS LIS 2022-2023 times the CG fraction from the IITM network,
sr_detection.py). District exposure is the population-weighted mean. Census
2011 covariates: agricultural workers (cultivators and agricultural labourers)
per person, literacy, rural share of households.
Output: results/district2011_table.csv; printed correlations
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.features
import xarray as xr
from scipy import stats

from common import DATA, LAT, LON, POP, REGIONS, RESULTS, india_area, india_geometry, nrsc_clim

A = india_area()
net = nrsc_clim("flashes").sum(("month", "hour_utc")).values / A
det = xr.open_dataset(DATA / "vulnerability/detection_025.nc")
sat = det.iss_cg_2022_2023.values
dist = gpd.read_file(DATA / "boundaries/districts2011/India-Districts-2011Census.shp").to_crs(4326)
dist["code"] = pd.to_numeric(dist.censuscode, errors="coerce")
india = india_geometry()
parts = []
for tif in ("ind_ppp_2020_1km_Aggregated_UNadj.tif", "pak_ppp_2020_1km_Aggregated_UNadj.tif"):
    with rasterio.open(POP / tif) as src:
        pop = src.read(1).astype(np.float64)
        nod, tr, shape = src.nodata, src.transform, src.shape
    valid = np.isfinite(pop) & (pop != nod) & (pop >= 0)
    did = rasterio.features.rasterize(((g, k + 1) for k, g in enumerate(dist.geometry)), out_shape=shape,
                                      transform=tr, fill=0, dtype="int32")
    inside = rasterio.features.rasterize([(india, 1)], out_shape=shape, transform=tr, fill=0,
                                         dtype="uint8").astype(bool)
    r, c = np.nonzero(valid & inside & (did > 0))
    lon, lat = tr.c + (c + 0.5) * tr.a, tr.f + (r + 0.5) * tr.e
    j, i = np.rint((lat - LAT[0]) / 0.25).astype(int), np.rint((lon - LON[0]) / 0.25).astype(int)
    ok = (j >= 0) & (j < len(LAT)) & (i >= 0) & (i < len(LON))
    parts.append(pd.DataFrame({"d": did[r, c][ok] - 1, "pop": pop[r, c][ok], "net": net[j[ok], i[ok]],
                               "sat": sat[j[ok], i[ok]]}))
df = pd.concat(parts)
df = df[np.isfinite(df.net) & np.isfinite(df.sat)]
g = df.assign(pn=df["pop"] * df.net, ps=df["pop"] * df.sat).groupby("d")[["pop", "pn", "ps"]].sum()
t = dist[["DISTRICT", "ST_NM", "code"]].join(g, how="inner")
t["E_net"], t["E_sat"] = t.pn / t["pop"], t.ps / t["pop"]
c = pd.read_csv(DATA / "vulnerability/india-districts-census-2011.csv")
c["agri"] = (c.Cultivator_Workers + c.Agricultural_Workers) / c.Population
c["literacy"] = c.Literate / c.Population
c["rural"] = c.Rural_Households / (c.Rural_Households + c.Urban_Households)
t = t.merge(c[["District code", "agri", "literacy", "rural", "Population"]], left_on="code",
            right_on="District code").drop(columns=["District code", "pn", "ps"])
# regions of this paper from the 2011 state names (Telangana was part of Andhra Pradesh; Ladakh of J&K)
name = {"Jammu & Kashmir": "JAMMU & KASHMIR", "NCT of Delhi": "DELHI (UT)", "Orissa": "ODISHA",
        "Andaman & Nicobar Island": "A & N ISLANDS", "Dadara & Nagar Havelli": "D & N HAVELI AND DAMAN & DIU",
        "Daman & Diu": "D & N HAVELI AND DAMAN & DIU", "Puducherry": "PUDUCHERRY", "Arunanchal Pradesh": "ARUNACHAL PRADESH"}
TO_REGION = {s: r for r, ss in REGIONS.items() for s in ss}
t["state"] = t.ST_NM.map(lambda s: name.get(s, s.upper()))
t["region"] = t.state.map(lambda s: TO_REGION.get(s, TO_REGION.get("TELANGANA") if s == "ANDHRA PRADESH" else ""))
t.round(5).to_csv(RESULTS / "district2011_table.csv", index=False)
print(f"{len(t)} districts, {t['pop'].sum() / 1e6:,.0f} million people (WorldPop 2020)")


def corr(d, x, y):
    r = stats.spearmanr(d[x], d[y])
    return f"{r.statistic:+.2f}"


print("\nSpearman correlation across districts of exposure with agricultural workers per person, literacy, rural share:")
for lab, d in [("all India", t)] + [(r, t[t.region == r]) for r in REGIONS]:
    print(f"  {lab:15s} ({len(d):3d})  network: {corr(d, 'E_net', 'agri')}, {corr(d, 'E_net', 'literacy')}, "
          f"{corr(d, 'E_net', 'rural')};  satellite CG: {corr(d, 'E_sat', 'agri')}, {corr(d, 'E_sat', 'literacy')}, "
          f"{corr(d, 'E_sat', 'rural')}")
best = t[t.region.isin(["Central", "East", "Gangetic Plain"])]
print(f"  best-detected regions together ({len(best)}): network {corr(best, 'E_net', 'agri')}, satellite CG "
      f"{corr(best, 'E_sat', 'agri')}")
# how much of India's farm work is in the most exposed districts
t["agri_n"] = t.agri * t["pop"]
for e in ("E_net", "E_sat"):
    s = t.sort_values(e, ascending=False)
    cum = s["pop"].cumsum() / s["pop"].sum()
    top = cum <= 0.25
    print(f"{e}: districts with the most exposed quarter of India's people hold "
          f"{100 * s.agri_n[top].sum() / s.agri_n.sum():.0f}% of its agricultural workers; "
          f"population-weighted exposure of agricultural workers / of everyone: "
          f"{(s[e] * s.agri_n).sum() / s.agri_n.sum() / ((s[e] * s['pop']).sum() / s['pop'].sum()):.2f}")

# Does the concentration of exposure in few districts come from the network's uneven detection? The 741 current
# districts (Survey of India, as in sr_exposure.py; population of the whole official boundary) with exposure from
# the network (2022-2025) and from satellite-expected CG lightning (2022-2023), on the cells that have both
names = pd.read_csv(POP / "districts.csv", index_col="district")
dc = pd.read_csv(POP / "pop_district_cell.csv")
dc = dc[dc.district >= 0]
dc = dc.assign(net=net[dc.j.values, dc.i.values], sat=sat[dc.j.values, dc.i.values])
dc = dc[np.isfinite(dc.net) & np.isfinite(dc.sat)]
g = dc.assign(pn=dc.population * dc.net, ps=dc.population * dc.sat).groupby("district")[["population", "pn", "ps"]].sum()


def half(col):
    """Districts that hold half of India's exposure, largest shares first (as sr_exposure.py)."""
    v = g[col].sort_values(ascending=False)
    return set(v.index[:int((100 * v.cumsum() / v.sum() <= 50).sum() + 1)])


hn, hs = half("pn"), half("ps")
people = lambda ids: 100 * g.population[list(ids)].sum() / g.population.sum()
r = stats.spearmanr(g.pn, g.ps)
print(f"\nCurrent districts ({len(g)}, {g.population.sum() / 1e6:,.1f} million people): half of India's exposure lies in "
      f"{len(hn)} districts with network exposure ({people(hn):.0f}% of the people) and in {len(hs)} districts with "
      f"satellite-expected CG exposure ({people(hs):.0f}% of the people); {len(hn & hs)} districts are in both sets; "
      f"rank correlation of the district shares {r.statistic:.2f}")
reg = names.state.map(TO_REGION).reindex(g.index).fillna("")
print("share of India's exposure by region, network / satellite-expected CG (%): " + ", ".join(
    f"{k} {100 * g.pn[reg == k].sum() / g.pn.sum():.0f} / {100 * g.ps[reg == k].sum() / g.ps.sum():.0f}" for k in REGIONS))
top = lambda col, n=20: set(g[col].sort_values(ascending=False).index[:n])
print(f"of the 20 districts with the largest network shares, {len(top('pn') & top('ps'))} are among the 20 largest "
      f"satellite shares and {len(top('pn') & hs)} among the districts that hold half of the satellite-based exposure")
