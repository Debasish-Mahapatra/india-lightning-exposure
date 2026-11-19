#!/usr/bin/env python3
"""Half-hourly IMERG V07 Final rain (data/obs_hh/imerg, 0.1 degree, mm/hr) and
NRSC flashes (data/nrsc_flashes) by calendar month and half hour of the day
(UTC) on the 0.25 degree grid, per year, for the days with both: summed rain
(mm), flash counts and the number of days. Rain is put on the 0.25 degree
grid by area overlap; cells with a part inside India only.
Output: data/obs_hh_025/hh_cycle_YYYY.nc
Usage: hh_cycle.py YEAR [YEAR ...]   (one process per year)
"""
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import xarray as xr

from common import DATA, LAT, LON, OBS_HH, india_area

OUT = DATA / "obs_hh_025"
INSIDE = np.isfinite(india_area())


def weights(src, dst, dsrc=0.1, ddst=0.25):
    """Overlap of each target cell with each source cell, rows summing to one."""
    lo = np.maximum((dst - ddst / 2)[:, None], (src - dsrc / 2)[None, :])
    hi = np.minimum((dst + ddst / 2)[:, None], (src + dsrc / 2)[None, :])
    w = np.clip(hi - lo, 0, None)
    return w / w.sum(1, keepdims=True)                 # edge rows: only the part the file covers


def one_year(year):
    ny, nx = len(LAT), len(LON)
    rain = np.zeros((12, 48, ny, nx))
    days = np.zeros(12, int)
    z = np.load(DATA / f"nrsc_flashes/flashes_{year}.npz")
    have_nrsc = set(z["dates"])
    t = z["t"].astype(np.int64)
    use = np.zeros(t.size, bool)
    wy = wx = None
    for k, day in enumerate(pd.date_range(f"{year}-01-01", f"{year}-12-31")):
        f = OBS_HH / f"imerg/{year}/imerg_{day:%Y%m%d}.nc"
        if not f.exists() or day.strftime("%Y-%m-%d") not in have_nrsc:
            continue
        d = xr.open_dataset(f)["precipitation"].transpose("time", "lat", "lon")
        if d.sizes["time"] != 48:
            continue
        if wy is None:
            wy = weights(np.round(d.lat.values.astype(np.float64), 3), LAT)
            wx = weights(np.round(d.lon.values.astype(np.float64), 3), LON)
        rain[day.month - 1] += 0.5 * ((wy @ np.nan_to_num(d.values)) @ wx.T)   # mm/hr -> mm per half hour
        days[day.month - 1] += 1
        use |= (t // 86400) == k
    t = t[use]
    i = np.rint((z["lon"][use] - LON[0]) / 0.25).astype(np.int64)
    j = np.rint((z["lat"][use] - LAT[0]) / 0.25).astype(np.int64)
    month = pd.date_range(f"{year}-01-01", f"{year}-12-31").month.values[t // 86400] - 1
    hh = (t % 86400) // 1800
    flashes = np.bincount(((month * 48 + hh) * ny + j) * nx + i, minlength=12 * 48 * ny * nx).reshape(12, 48, ny, nx)
    f32 = lambda v: np.where(INSIDE, v, np.nan).astype(np.float32)
    ds = xr.Dataset({"rain": (("month", "halfhour", "lat", "lon"), f32(rain), {"units": "mm", "long_name": "IMERG rain summed over the days"}),
                     "flashes": (("month", "halfhour", "lat", "lon"), f32(flashes)),
                     "days": (("month",), days, {"long_name": "days with IMERG and NRSC"})},
                    coords={"month": np.arange(1, 13), "halfhour": np.arange(48) * 0.5, "lat": LAT, "lon": LON},
                    attrs={"note": "halfhour: start of the half hour, UTC hours; India's official boundary only"})
    ds.to_netcdf(OUT / f"hh_cycle_{year}.nc", encoding={v: {"zlib": True} for v in ("rain", "flashes")})
    return f"{year}: {days.sum()} days, {int(flashes.sum()):,} flashes"


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    years = [int(y) for y in sys.argv[1:]]
    with Pool(min(len(years), 2)) as pool:
        for msg in pool.imap_unordered(one_year, years):
            print(msg, flush=True)
