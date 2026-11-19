#!/usr/bin/env python3
"""NRSC flashes inside India's official boundary on ERA5's 0.25 degree grid,
per year, from data/nrsc_flashes (nrsc_flashes.py): counts by calendar month
and UTC hour (diurnal cycles by region and season), by UTC day, of flashes of
at least 20 kA and at least 50 kA, and the summed peak current, by month.
Cells that miss India are NaN; india_area is the area of India's part of each
cell, so flash density = count / india_area.
Output: data/nrsc_detail/nrsc_detail_025_YYYY.nc
Usage: nrsc_detail.py YEAR [YEAR ...]
"""
import sys

import numpy as np
import pandas as pd
import xarray as xr

from common import DATA, LAT, LON, NRSC_DETAIL, india_area

AREA = india_area()


def one_year(year):
    ny, nx = len(LAT), len(LON)
    z = np.load(DATA / f"nrsc_flashes/flashes_{year}.npz")
    t, a = z["t"].astype(np.int64), z["amps"]
    i = np.rint((z["lon"] - LON[0]) / 0.25).astype(np.int64)
    j = np.rint((z["lat"] - LAT[0]) / 0.25).astype(np.int64)
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31")
    doy = t // 86400
    hour = (t % 86400) // 3600
    month = days.month.values[doy] - 1
    cell = j * nx + i
    count = lambda idx, n, w=None: np.bincount(idx, weights=w, minlength=n * ny * nx).reshape(n, ny, nx)
    hourly = count(((month * 24 + hour) * ny * nx + cell), 12 * 24).reshape(12, 24, ny, nx)
    daily = count(doy * ny * nx + cell, len(days))
    mcell = month * ny * nx + cell
    s20 = count(mcell, 12, a >= 20000)
    s50 = count(mcell, 12, a >= 50000)
    amps = count(mcell, 12, np.nan_to_num(a).astype(np.float64))
    have = np.isin(days.strftime("%Y-%m-%d"), z["dates"])
    ndays = np.bincount(days.month.values[have] - 1, minlength=12)
    f = lambda v: np.where(np.isfinite(AREA), v, np.nan).astype(np.float32)
    daily = f(daily)
    daily[~have] = np.nan
    ds = xr.Dataset(
        {"flashes": (("month", "hour_utc", "lat", "lon"), f(hourly), {"long_name": "flashes per cell, by month and UTC hour"}),
         "daily": (("time", "lat", "lon"), daily, {"long_name": "flashes per cell and UTC day (NaN: no NRSC file)"}),
         "flashes_ge20kA": (("month", "lat", "lon"), f(s20)),
         "flashes_ge50kA": (("month", "lat", "lon"), f(s50)),
         "current_sum": (("month", "lat", "lon"), f(amps), {"units": "A"}),
         "days": (("month",), ndays, {"long_name": "days with an NRSC file"}),
         "india_area": (("lat", "lon"), AREA.astype(np.float32), {"units": "km2", "long_name": "area of India's part of the cell"})},
        coords={"month": np.arange(1, 13), "hour_utc": np.arange(24), "time": days, "lat": LAT, "lon": LON},
        attrs={"year": year, "note": "flashes inside India's official boundary only"})
    ds.to_netcdf(NRSC_DETAIL / f"nrsc_detail_025_{year}.nc",
                 encoding={v: {"zlib": True} for v in ("flashes", "daily", "flashes_ge20kA", "flashes_ge50kA", "current_sum")})
    return f"{year}: {int(np.nansum(f(hourly))):,} flashes inside India, {have.sum()} days"


if __name__ == "__main__":
    for y in sys.argv[1:]:
        print(one_year(int(y)), flush=True)
