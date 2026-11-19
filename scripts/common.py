"""Paths and shared helpers for the paper on lightning exposure in India (observations only).

Everything is over India's land inside its official (India-view) boundary on
the 0.25 degree grid 67-98 E, 6-38 N (the ERA5 grid of MARS).
"""
import pathlib
import struct

import numpy as np
import shapely

WS = pathlib.Path("/ec/res4/hpcperm/bel5089/scirep_india_lightning")
DATA = WS / "data"
NRSC = DATA / "nrsc"                  # link, shared: raw daily files in all/, daily 0.25 degree counts in grid025/
NRSC_DETAIL = DATA / "nrsc_detail"    # counts by month and UTC hour, and peak-current counts, per year
ERA5 = DATA / "era5_daily"            # link, shared: daily ERA5 fields (era5_daily_YYYY.nc)
IMERG_DAILY = DATA / "imerg_daily"    # link, shared: IMERG V07 Final daily, cut to 55-110 E, 5 S-45 N
OBS_HH = DATA / "obs_hh"              # link to scratch: half-hourly IMERG and MERGIR over the India box
POP = DATA / "population"
NCRB = DATA / "ncrb"
# Survey of India, Administrative Boundary Database (1:1M, product OVSF/1M/7; copy with invalid
# geometries repaired, github.com/surajreddyr/India-State-District-Administrative-Boundaries-Shapefiles):
# India's official boundary is the union of its State/UT polygons
STATES = DATA / "boundaries/soi_abdb/states/STATE_BOUNDARY_Corrected.shp"
DISTRICTS = DATA / "boundaries/soi_abdb/districts/DISTRICT_BOUNDARY_Corrected.shp"
FIG = WS / "manuscript/figures"
SUP = WS / "manuscript/supplementary"
RESULTS = WS / "results"
SCRATCH = "/ec/res4/scratch/bel5089/experiments/mrdsgo"   # job logs are in logs/

LAT = np.arange(6.0, 38.0001, 0.25)
LON = np.arange(67.0, 98.0001, 0.25)
R_EARTH = 6371.0                      # km
YEARS = range(2022, 2026)             # stable network: every absolute number uses 2022-2025

SEASONS = {"Winter (December-February)": [12, 1, 2], "Pre-monsoon (March-May)": [3, 4, 5],
           "Monsoon (June-September)": [6, 7, 8, 9], "Post-monsoon (October-November)": [10, 11]}

# State/UT names of the Survey of India file -> NCRB's spelling (the rest are the same; four
# small strips named "DISPUTED (...)" lie between states and keep their names)
NCRB_NAME = {"ANDAMAN & NICOBAR": "A & N ISLANDS", "DADRA & NAGAR HAVELI & DAMAN & DIU": "D & N HAVELI AND DAMAN & DIU",
             "DELHI": "DELHI (UT)", "JAMMU AND KASHM|R": "JAMMU & KASHMIR"}
# NCRB names -> names for tables and figures (the rest: title case)
STATE_LABEL = {"JAMMU & KASHMIR": "Jammu and Kashmir", "DELHI (UT)": "Delhi",
               "A & N ISLANDS": "Andaman and Nicobar Islands",
               "D & N HAVELI AND DAMAN & DIU": "Dadra and Nagar Haveli and Daman and Diu"}


def state_label(ncrb_name):
    return STATE_LABEL.get(ncrb_name, ncrb_name.title())

# Regions for the seasonal and diurnal cycles, as groups of States/UTs (islands left out)
REGIONS = {
    "Himalaya": ["JAMMU & KASHMIR", "LADAKH", "HIMACHAL PRADESH", "UTTARAKHAND"],
    "Northwest": ["RAJASTHAN", "GUJARAT", "PUNJAB", "HARYANA", "DELHI (UT)", "CHANDIGARH",
                  "D & N HAVELI AND DAMAN & DIU"],
    "Gangetic Plain": ["UTTAR PRADESH", "BIHAR"],
    "Central": ["MADHYA PRADESH", "CHHATTISGARH", "MAHARASHTRA"],
    "East": ["JHARKHAND", "WEST BENGAL", "ODISHA"],
    "Northeast": ["ASSAM", "MEGHALAYA", "ARUNACHAL PRADESH", "NAGALAND", "MANIPUR", "MIZORAM",
                  "TRIPURA", "SIKKIM"],
    "South": ["TELANGANA", "ANDHRA PRADESH", "KARNATAKA", "GOA", "KERALA", "TAMIL NADU", "PUDUCHERRY"],
}


def cell_area():
    """Area of each 0.25 degree cell (km2), shape (lat, lon)."""
    band = np.abs(np.sin(np.deg2rad(LAT + 0.125)) - np.sin(np.deg2rad(LAT - 0.125)))
    return (R_EARTH**2 * np.deg2rad(0.25) * band)[:, None] * np.ones(len(LON))[None, :]


def india_geometry():
    """India's official boundary (lon, lat): the union of the Survey of India's State and
    Union Territory polygons. Cached."""
    f = DATA / "boundaries/soi_india_4326.wkb"
    if f.exists():
        return shapely.from_wkb(f.read_bytes())
    import geopandas as gpd
    g = shapely.make_valid(shapely.union_all(gpd.read_file(STATES).to_crs(4326).geometry.values))
    g = shapely.union_all([shapely.Polygon(p.exterior) for p in getattr(g, "geoms", [g])
                           if p.geom_type == "Polygon"])      # fill slivers where state lines do not meet
    f.write_bytes(shapely.to_wkb(g))
    return g


# 0.1 degree grid, used only for the annual map of Fig. 1a
LAT01 = np.round(np.arange(6.05, 38, 0.1), 2)
LON01 = np.round(np.arange(67.05, 98, 0.1), 2)


def cell_area_01():
    """Area of each 0.1 degree cell (km2), shape (lat, lon)."""
    band = np.abs(np.sin(np.deg2rad(LAT01 + 0.05)) - np.sin(np.deg2rad(LAT01 - 0.05)))
    return (R_EARTH**2 * np.deg2rad(0.1) * band)[:, None] * np.ones(len(LON01))[None, :]


def india_area_01():
    """Area (km2) of India's part (official boundary) of every 0.1 degree cell, NaN where the
    cell misses India. Cached."""
    f = DATA / "boundaries/india_area_010.npy"
    if f.exists():
        return np.load(f)
    lon2, lat2 = np.meshgrid(LON01, LAT01)
    boxes = shapely.box(lon2 - 0.05, lat2 - 0.05, lon2 + 0.05, lat2 + 0.05)
    geom = india_geometry()
    shapely.prepare(geom)
    full = cell_area_01()
    out = np.full(lon2.shape, np.nan)
    inside = shapely.contains(geom, boxes)
    out[inside] = full[inside]
    edge = shapely.intersects(geom, boxes) & ~inside
    out[edge] = shapely.area(shapely.intersection(geom, boxes[edge])) / 0.01 * full[edge]
    np.save(f, out)
    return out


def flashes_01(years=YEARS):
    """Mean yearly number of flashes inside India by month on the 0.1 degree grid, shape
    (12, lat, lon), from data/nrsc_flashes, each month scaled up for its days without data
    (as nrsc_clim does on the 0.25 degree grid)."""
    import pandas as pd
    mon = np.zeros((12, len(LAT01), len(LON01)))
    for y in years:
        z = np.load(DATA / f"nrsc_flashes/flashes_{y}.npz")
        day = z["t"].astype(np.int64) // 86400
        m = (pd.Timestamp(f"{y}-01-01") + pd.to_timedelta(day, unit="D")).month.values - 1
        j = np.floor((z["lat"] - 6) / 0.1).astype(int)
        i = np.floor((z["lon"] - 67) / 0.1).astype(int)
        ok = (j >= 0) & (j < len(LAT01)) & (i >= 0) & (i < len(LON01))
        c = np.bincount((m[ok] * len(LAT01) + j[ok]) * len(LON01) + i[ok], minlength=mon.size).reshape(mon.shape)
        have = pd.to_datetime(z["dates"]).month.value_counts().reindex(range(1, 13), fill_value=0).values
        ndays = np.array([pd.Period(f"{y}-{k:02d}").days_in_month for k in range(1, 13)])
        mon += c * (ndays / np.maximum(have, 1))[:, None, None]
    return mon / len(years)


def india_mask():
    """True for 0.25 degree cells whose centre is inside India's official boundary."""
    lon2, lat2 = np.meshgrid(LON, LAT)
    return shapely.contains_xy(india_geometry(), lon2, lat2)


def india_area():
    """Area (km2) of India's part (official boundary) of every 0.25 degree cell, NaN where
    the cell misses India. Flash density = flashes inside India / india_area. Cached."""
    f = DATA / "boundaries/india_area_025.npy"
    if f.exists():
        return np.load(f)
    lon2, lat2 = np.meshgrid(LON, LAT)
    boxes = shapely.box(lon2 - 0.125, lat2 - 0.125, lon2 + 0.125, lat2 + 0.125)
    geom = india_geometry()
    hit = shapely.intersects(geom, boxes)
    out = np.full(lon2.shape, np.nan)
    out[hit] = shapely.area(shapely.intersection(geom, boxes[hit])) / 0.0625 * cell_area()[hit]
    np.save(f, out)
    return out


def states():
    """State/UT polygons (lon, lat) with NCRB's names in column "state", one row each."""
    import geopandas as gpd
    st = gpd.read_file(STATES).to_crs(4326)
    st["state"] = [NCRB_NAME.get(n, n) for n in st["STATE"]]
    return st.dissolve("state").reset_index()[["state", "geometry"]]


def state_grid():
    """State/UT (NCRB name) of every 0.25 degree cell with a part inside India's official
    boundary: the one that contains the cell centre, else the nearest; "" for cells that miss
    India. Shape (lat, lon). Cached."""
    f = DATA / "boundaries/state_grid_025.npy"
    if f.exists():
        return np.load(f).astype(object)
    st = states()
    lon2, lat2 = np.meshgrid(LON, LAT)
    out = np.full(lon2.shape, "", dtype=object)
    for name, geom in zip(st.state, st.geometry):
        out[shapely.contains_xy(geom, lon2, lat2)] = name
    inside = np.isfinite(india_area())
    for j, i in zip(*np.nonzero(inside & (out == ""))):
        p = shapely.Point(LON[i], LAT[j])
        out[j, i] = st.state.iloc[int(np.argmin([g.distance(p) for g in st.geometry]))]
    out[~inside] = ""
    np.save(f, out.astype(str))
    return out


def region_grid():
    """Region name (REGIONS) of every 0.25 degree cell, "" outside the regions."""
    to_region = {s: r for r, ss in REGIONS.items() for s in ss}
    return np.vectorize(lambda s: to_region.get(s, ""), otypes=[object])(state_grid())


def nrsc_clim(var="flashes", years=YEARS):
    """Mean over the years of a variable of data/nrsc_detail (flashes by month and UTC hour, or
    flashes_ge20kA, flashes_ge50kA, current_sum by month) per 0.25 degree cell, each month
    scaled up for its days without data. Only flashes inside India's official boundary; NaN
    for cells that miss India (xarray sums turn these into 0: divide by india_area())."""
    import pandas as pd
    import xarray as xr
    tot = 0
    for y in years:
        d = xr.open_dataset(NRSC_DETAIL / f"nrsc_detail_025_{y}.nc")
        ndays = xr.DataArray([pd.Period(f"{y}-{m:02d}").days_in_month for m in range(1, 13)],
                             coords={"month": d.month})
        tot = tot + d[var] * (ndays / d["days"])
    return tot / len(years)


def read_day(stem, current=False, seconds=False):
    """Longitude, latitude and UTC hour of every flash in one NRSC daily file
    (point geometry; Time is UTC on a 12-hour clock), and with current=True
    also its peak current in A (NaN where not recorded, part of January 2020).
    With seconds=True the time is in seconds after 00:00 UTC instead of hours."""
    shp = pathlib.Path(f"{stem}.shp").read_bytes()
    rec = np.dtype([("num", ">i4"), ("len", ">i4"), ("type", "<i4"), ("x", "<f8"), ("y", "<f8")])
    pts = np.frombuffer(shp, dtype=rec, offset=100)
    assert (pts["type"] == 1).all() and (pts["len"] == 10).all(), stem
    dbf = pathlib.Path(f"{stem}.dbf").read_bytes()
    nrec, hlen, rlen = struct.unpack("<IHH", dbf[4:12])
    off, start, cur = 32, 1, None
    while dbf[off] != 0x0D:
        name, width = dbf[off:off + 11].split(b"\0")[0].decode(), dbf[off + 16]
        if name == "Time":
            tcol = (start, width)
        if name in ("CurrentD", "Current"):    # "Current" in files up to October 2020
            cur = (start, width)
        start += width
        off += 32
    assert nrec == len(pts), stem
    recs = np.frombuffer(dbf, dtype=f"S{rlen}", count=nrec, offset=hlen)
    uniq, inv = np.unique(np.array([r[tcol[0]:tcol[0] + tcol[1]] for r in recs]), return_inverse=True)
    secs = []
    for t in uniq:                      # b"12:00:02 AM" -> 2 s, b"1:05:09 PM" -> 13:05:09; unreadable -> -1
        t = t.decode(errors="replace").strip()
        try:
            h, m, s = (int(v) for v in t.split()[0].split(":"))
            secs.append(((h % 12) + (12 if t.upper().endswith("PM") else 0)) * 3600 + m * 60 + s)
        except (ValueError, IndexError):
            secs.append(-1)
    secs = np.array(secs)[inv]
    hours = secs if seconds else np.where(secs >= 0, secs // 3600, -1)
    if not current:
        return pts["x"], pts["y"], hours
    amps = np.array([float(r[cur[0]:cur[0] + cur[1]] or 0) for r in recs]) if cur else np.zeros(nrec)
    amps[amps <= 0] = np.nan
    return pts["x"], pts["y"], hours, amps


def _xyz(lon, lat):
    """Points on a sphere of radius R_EARTH (km): chord lengths equal great-circle
    distances within a metre up to 10 km."""
    la, lo = np.deg2rad(np.asarray(lat, float)), np.deg2rad(np.asarray(lon, float))
    return R_EARTH * np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


def _pairs(lon, lat, t, r, tmax):
    """Records sorted by time (o), and every pair (i earlier, j later, in sorted
    order) within r km and tmax s, with its distance and time difference."""
    from scipy.spatial import cKDTree
    o = np.lexsort((np.arange(t.size), t))          # of records in the same second, the one listed first is earlier
    xyz, s = _xyz(lon, lat)[o], np.asarray(t, float)[o]
    p = cKDTree(np.c_[xyz, s * r / tmax]).query_pairs(np.hypot(r, r) + 1e-6, output_type="ndarray")
    i, j = p.min(1), p.max(1)
    d, dt = np.linalg.norm(xyz[i] - xyz[j], axis=1), s[j] - s[i]
    k = (d <= r) & (dt <= tmax)
    return o, i[k], j[k], d[k], dt[k]


def earlier_neighbour(lon, lat, t, r=10.0, tmax=60):
    """For every record, the time (s), distance (km) and index of the nearest
    earlier record within r km and tmax s (NaN, NaN and -1 if none): smallest
    time difference first, then distance."""
    o, i, j, d, dt = _pairs(lon, lat, t, r, tmax)
    m = np.lexsort((d, dt, j))                      # for each j keep the smallest dt, then d
    m = m[np.unique(j[m], return_index=True)[1]]
    out_dt, out_d, out_i = np.full(t.size, np.nan), np.full(t.size, np.nan), np.full(t.size, -1)
    out_dt[o[j[m]]], out_d[o[j[m]]], out_i[o[j[m]]] = dt[m], d[m], o[i[m]]
    return out_dt, out_d, out_i


def group_flashes(lon, lat, t, r=10.0, tmax=1):
    """Index of the first record of every record's flash, adapted from the rule of
    the US National Lightning Detection Network (Cummins et al. 1998; without its
    500 ms, 15-stroke and overlap conditions) with times to the second: in time order, a record joins the nearest flash whose first record lies
    within tmax s and r km, or else starts a new flash. NRSC's records include
    later strokes of one flash (check_grouping.py)."""
    o, i, j, d, dt = _pairs(lon, lat, t, r, tmax)
    m = np.lexsort((d, j))                          # candidates of each later record, nearest first
    i, j = i[m], j[m]
    head = np.arange(t.size)
    is_head = np.ones(t.size, bool)
    starts = np.r_[0, np.nonzero(np.diff(j))[0] + 1] if j.size else np.zeros(0, int)
    for a, b in zip(starts, np.r_[starts[1:], j.size]):
        for ii in i[a:b]:                           # j's are in time order, so heads before j are final
            if is_head[ii]:
                head[j[a]], is_head[j[a]] = ii, False
                break
    out = np.empty(t.size, int)
    out[o] = o[head]
    return out
