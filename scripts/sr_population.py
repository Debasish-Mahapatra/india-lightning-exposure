#!/usr/bin/env python3
"""Population inside India's official boundary, by state and by 0.25 degree
cell of the NRSC/ERA5 grid.

WorldPop 2020, 1 km, UN-adjusted, for all of India within its official
boundary (Survey of India; the WorldPop grids of India and of Pakistan together
cover it). Every 1 km cell goes to the state (names as in NCRB's tables), the
district (Survey of India's current districts) and the 0.25 degree cell that
contain its centre.
Output in data/population/: pop_state_cell.csv (state, j, i, population of the
area enumerated by the 2011 Census, to which the death records refer, as
"population"; the whole official boundary as "population_all"; area_km2),
pop_district_cell.csv (district number, j, i, population of the whole official
boundary, area_km2), districts.csv (district number, name, state) and
pop2020_025.nc (population per 0.25 degree cell, whole official boundary);
results/population_conservation.csv (by State/UT, with the 2011 Census).
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.features
import xarray as xr
from common import DISTRICTS, LAT, LON, NCRB_NAME, POP, STATES, india_geometry, state_label
R = 6371.0


def conservation(tab, cells):
    """Population by assignment: national, States/UTs and districts, for the whole official
    boundary and for the enumerated area, against the 2011 Census (results/population_conservation.csv)."""
    from common import DATA, RESULTS
    s = tab.groupby("state")[["population_all", "population"]].sum() / 1e6
    c = pd.read_csv(DATA / "vulnerability/india-districts-census-2011.csv")
    cname = {"JAMMU AND KASHMIR": "JAMMU & KASHMIR", "NCT OF DELHI": "DELHI (UT)", "ORISSA": "ODISHA",
             "DAMAN AND DIU": "D & N HAVELI AND DAMAN & DIU", "DADRA AND NAGAR HAVELI": "D & N HAVELI AND DAMAN & DIU",
             "PONDICHERRY": "PUDUCHERRY", "ANDAMAN AND NICOBAR ISLANDS": "A & N ISLANDS"}
    c["state"] = c["State name"].map(lambda x: cname.get(x, x))
    tg = ["Adilabad", "Nizamabad", "Karimnagar", "Medak", "Hyderabad", "Rangareddy", "Mahbubnagar", "Nalgonda",
          "Warangal", "Khammam"]
    c.loc[(c.state == "ANDHRA PRADESH") & c["District name"].isin(tg), "state"] = "TELANGANA"
    c.loc[(c.state == "JAMMU & KASHMIR") & c["District name"].isin(["Leh(Ladakh)", "Kargil"]), "state"] = "LADAKH"
    s["census2011"] = c.groupby("state").Population.sum() / 1e6
    s.round(4).to_csv(RESULTS / "population_conservation.csv")
    print(f"population (million): official boundary {s.population_all.sum():,.1f}; enumerated area "
          f"{s.population.sum():,.1f}; elsewhere {s.population_all.sum() - s.population.sum():.1f}; "
          f"sum over States/UTs {s.population_all.sum():,.1f}; Census 2011 {s.census2011.sum():,.1f}")
    for st_ in ("JAMMU & KASHMIR", "LADAKH"):
        r = s.loc[st_]
        print(f"  {st_}: official boundary {r.population_all:.2f}, enumerated area {r.population:.2f}, "
              f"Census 2011 {r.census2011:.2f}")
    other = s.drop(["JAMMU & KASHMIR", "LADAKH"])
    print(f"  all other States/UTs: official boundary {other.population_all.sum():,.1f}, enumerated "
          f"{other.population.sum():,.1f}; WorldPop 2020 / Census 2011 {other.population.sum() / other.census2011.sum():.3f} "
          f"(States/UTs {(other.population / other.census2011).min():.2f}-{(other.population / other.census2011).max():.2f})")
    print(f"  districts: {cells.groupby('district').population_all.sum().sum() / 1e6:,.1f} million in "
          f"{cells.district[cells.district >= 0].nunique()} districts")


def main():
    st = gpd.read_file(STATES).to_crs(4326)
    st["ncrb"] = [NCRB_NAME.get(n, n) for n in st["STATE"]]
    st = st.dissolve("ncrb").reset_index()
    names = list(st["ncrb"])
    dist = gpd.read_file(DISTRICTS).to_crs(4326)
    dist["state"] = [NCRB_NAME.get(n, n) for n in dist["STATE"]]
    dist["DISTRICT"] = (dist["District"].str.replace(">", "A").str.replace("|", "I")   # SoI writes long a and i
                        .str.upper().str.title().str.replace(" - ", "-").str.replace("Twenty-Four", "24"))          # (Kolk>ta, Son|pat) as > and |
    dist["ST_NM"] = [state_label(n) for n in dist["state"]]
    dist[["DISTRICT", "ST_NM", "state", "DISTRICT_L"]].rename_axis("district").to_csv(f"{POP}/districts.csv")
    india = india_geometry()
    parts = []
    for tif in ("ind_ppp_2020_1km_Aggregated_UNadj.tif", "pak_ppp_2020_1km_Aggregated_UNadj.tif"):
        with rasterio.open(f"{POP}/{tif}") as src:
            pop = src.read(1).astype(np.float64)
            nod = src.nodata
            tr, shape = src.transform, src.shape
        valid = np.isfinite(pop) & (pop != nod) & (pop >= 0)
        sid = rasterio.features.rasterize(((g, k + 1) for k, g in enumerate(st.geometry)),
                                          out_shape=shape, transform=tr, fill=0, dtype="int32")
        did = rasterio.features.rasterize(((g, k + 1) for k, g in enumerate(dist.geometry)),
                                          out_shape=shape, transform=tr, fill=0, dtype="int32")
        inside = rasterio.features.rasterize([(india, 1)], out_shape=shape, transform=tr,
                                             fill=0, dtype="uint8").astype(bool)
        rows, cols = np.nonzero(valid & inside & (sid > 0))
        lon = tr.c + (cols + 0.5) * tr.a
        lat = tr.f + (rows + 0.5) * tr.e
        dlat = abs(tr.e)
        area = R**2 * np.deg2rad(tr.a) * np.abs(np.sin(np.deg2rad(lat + dlat / 2)) - np.sin(np.deg2rad(lat - dlat / 2)))
        parts.append(pd.DataFrame({"state": np.array(names)[sid[rows, cols] - 1],
                                   "district": did[rows, cols] - 1,           # -1: in no district
                                   "j": np.rint((lat - LAT[0]) / 0.25).astype(int),
                                   "i": np.rint((lon - LON[0]) / 0.25).astype(int),
                                   "population": pop[rows, cols], "area_km2": area, "source": tif[:3]}))
        print(f"{tif[:3]}: {len(rows):,} cells inside India's official boundary, "
              f"{pop[rows, cols].sum() / 1e6:,.1f} million people", flush=True)
    df = pd.concat(parts)
    ok = (df.j >= 0) & (df.j < len(LAT)) & (df.i >= 0) & (df.i < len(LON))
    # State tables go with the death records, which cover the area enumerated by the 2011
    # Census: the WorldPop grid "ind". "population" is that area's population and
    # "population_all" the whole official boundary (maps and district tables use the latter).
    df["population_all"] = df.population
    df["population"] = np.where(df.source == "ind", df.population_all, 0.0)
    tab = df[ok].groupby(["state", "j", "i"])[["population", "population_all", "area_km2"]].sum().reset_index()
    tab.to_csv(f"{POP}/pop_state_cell.csv", index=False)
    conservation(tab, df[ok])
    df["population"] = df.population_all
    dtab = df[ok].groupby(["district", "j", "i"])[["population", "area_km2"]].sum().reset_index()
    dtab.to_csv(f"{POP}/pop_district_cell.csv", index=False)
    print(f"{df[ok].population[df[ok].district < 0].sum() / 1e6:.2f} million people in no district")
    grid = np.full((len(LAT), len(LON)), np.nan)
    g = tab.groupby(["j", "i"])["population_all"].sum()
    grid[g.index.get_level_values(0), g.index.get_level_values(1)] = g.values
    xr.Dataset({"population": (("lat", "lon"), grid, {"long_name": "WorldPop 2020 inside India's official boundary"})},
               coords={"lat": LAT, "lon": LON}).to_netcdf(f"{POP}/pop2020_025.nc")
    s = tab.groupby("state")["population_all"].sum().sort_values(ascending=False) / 1e6
    print(f"total {s.sum():,.1f} million in {len(s)} States/UTs; largest:", s.head(5).round(1).to_dict())


if __name__ == "__main__":
    main()
