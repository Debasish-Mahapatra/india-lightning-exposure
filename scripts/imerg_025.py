#!/usr/bin/env python3
"""IMERG V07 Final daily rain (data/imerg_daily, 0.1 degree, mm/day) on the
0.25 degree NRSC/ERA5 grid by area overlap (conservative for a regular grid),
for the cells with a part inside India's official boundary (NaN elsewhere).
Output: data/imerg_025/imerg_daily_025_YYYY.nc (mm per UTC day)
Usage: imerg_025.py YEAR [YEAR ...]   (2020 to September 2025)
"""
import sys

import numpy as np
import pandas as pd
import xarray as xr

from common import DATA, IMERG_DAILY, LAT, LON, india_area

OUT = DATA / "imerg_025"
INSIDE = np.isfinite(india_area())


def weights(src, dst, dsrc, ddst=0.25):
    """Overlap of each target cell with each source cell, as a fraction of the target cell."""
    lo = np.maximum((dst - ddst / 2)[:, None], (src - dsrc / 2)[None, :])
    hi = np.minimum((dst + ddst / 2)[:, None], (src + dsrc / 2)[None, :])
    return np.clip(hi - lo, 0, None) / ddst


def one_year(year):
    months = [m for m in pd.date_range(f"{year}-01", f"{year}-12", freq="MS")
              if (IMERG_DAILY / f"imerg_daily_{m:%Y%m}.nc").exists()]
    out = []
    for m in months:
        d = xr.open_dataset(IMERG_DAILY / f"imerg_daily_{m:%Y%m}.nc")["precipitation"].transpose("time", "lat", "lon")
        wy = weights(np.round(d.lat.values.astype(np.float64), 3), LAT, 0.1)
        wx = weights(np.round(d.lon.values.astype(np.float64), 3), LON, 0.1)     # float32 in the files
        assert np.allclose(wy.sum(1), 1) and np.allclose(wx.sum(1), 1)
        p = (wy @ np.nan_to_num(d.values)) @ wx.T
        p[:, ~INSIDE] = np.nan
        out.append(xr.DataArray(p.astype(np.float32), dims=("time", "lat", "lon"),
                                coords={"time": d.time.values.astype("datetime64[ns]"), "lat": LAT, "lon": LON}))
    r = xr.concat(out, "time").rename("precipitation")
    r.attrs = {"units": "mm", "long_name": "IMERG V07 Final daily rain per UTC day, 0.25 degree"}
    r.to_dataset().to_netcdf(OUT / f"imerg_daily_025_{year}.nc", encoding={"precipitation": {"zlib": True}})
    return f"{year}: {len(months)} months, {r.sizes['time']} days, India mean {np.nanmean(r.values) * 365:.0f} mm a year"


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for y in sys.argv[1:]:
        print(one_year(int(y)), flush=True)
