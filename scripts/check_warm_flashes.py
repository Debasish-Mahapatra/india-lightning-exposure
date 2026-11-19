#!/usr/bin/env python3
"""Check: NRSC flashes under warm cloud tops. For each flash on the days with a
MERGIR file, the coldest brightness temperature within about 6, 12 and 20 km
(3 x 3, 7 x 7 and 11 x 11 pixels: MERGIR views India at 40-60 degrees from
the vertical, so cloud tops are shifted by parallax) of the image nearest in
time, and the half-hourly IMERG rain of its
0.25 degree cell. Flashes under tops of 273 K or warmer (no ice) are unlikely
to be real lightning. Prints their share by month, region, peak current and
hour, and saves the per-flash table.
Output: results/check_warm_flashes_YYYY.csv (month, region, share warm, ...)
Usage: check_warm_flashes.py YEAR
"""
import sys

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import minimum_filter

from common import DATA, LAT, LON, OBS_HH, RESULTS, region_grid

year = int(sys.argv[1])
reg = region_grid()
z = np.load(DATA / f"nrsc_flashes/flashes_{year}.npz")
t = z["t"].astype(np.int64)
rows = []
for f in sorted((OBS_HH / f"mergir/{year}").glob("mergir_*.nc")):
    day = pd.Timestamp(f.stem.split("_")[1])
    sel = np.nonzero((t // 86400) == day.dayofyear - 1)[0]
    if sel.size == 0:
        continue
    d = xr.open_dataset(f)
    tb = d["Tb"].values
    lat, lon = d.lat.values.astype(float), d.lon.values.astype(float)
    k_img = ((d.time.values - np.datetime64(day)) / np.timedelta64(30, "m")).round().astype(int)
    img = {k: n for n, k in enumerate(k_img)}
    rain_f = OBS_HH / f"imerg/{year}/imerg_{day:%Y%m%d}.nc"
    rain = xr.open_dataset(rain_f)["precipitation"].transpose("time", "lat", "lon").values if rain_f.exists() else None
    fk = np.rint((t[sel] % 86400) / 1800).astype(int)
    fr = np.clip(np.rint((z["lat"][sel] - lat[0]) / np.diff(lat).mean()).astype(int), 0, lat.size - 1)
    fc = np.clip(np.rint((z["lon"][sel] - lon[0]) / np.diff(lon).mean()).astype(int), 0, lon.size - 1)
    cold = {w: np.full(sel.size, np.nan) for w in (3, 7, 11)}
    for k in np.unique(fk):
        if k not in img:
            continue
        here = fk == k
        x = tb[img[k]]
        if np.isnan(x).mean() > 0.2:
            continue
        for w in cold:
            cold[w][here] = minimum_filter(np.nan_to_num(x, nan=999.0), size=w)[fr[here], fc[here]]
    for w in cold:
        cold[w][cold[w] > 900] = np.nan
    hh = (t[sel] % 86400) // 1800
    rr = np.full(sel.size, np.nan)
    if rain is not None:                            # IMERG 0.1 degree pixel of the flash, same half hour (mm/hr)
        ri = np.clip(np.rint((z["lat"][sel] - 6.05) / 0.1).astype(int), 0, rain.shape[1] - 1)
        ci = np.clip(np.rint((z["lon"][sel] - 67.05) / 0.1).astype(int), 0, rain.shape[2] - 1)
        rr = rain[hh, ri, ci]
    j = np.rint((z["lat"][sel] - LAT[0]) / 0.25).astype(int)
    i = np.rint((z["lon"][sel] - LON[0]) / 0.25).astype(int)
    rows.append(pd.DataFrame({"month": day.month, "region": reg[j, i], "tb": cold[3], "tb7": cold[7],
                              "tb11": cold[11], "amps": z["amps"][sel],
                              "hour_ist": ((t[sel] % 86400) / 3600 + 5.5) % 24, "rain": rr}))
df = pd.concat(rows)
df = df[np.isfinite(df.tb)]
df["warm"] = df.tb >= 273
df["cold235"] = df.tb < 235
print(f"{year}: {len(df):,} flashes with a cloud top, {100 * df.warm.mean():.1f}% under 273 K or warmer, "
      f"{100 * df.cold235.mean():.1f}% under cloud colder than 235 K")
for c, km in (("tb", 6), ("tb7", 12), ("tb11", 20)):
    print(f"  coldest within {km} km: {100 * np.mean(df[c] >= 273):.1f}% 273 K or warmer, "
          f"{100 * np.mean(df[c] < 235):.1f}% colder than 235 K, {100 * np.mean(df[c] < 208):.1f}% colder than 208 K")
w = df.tb11 >= 273
print(f"  warm even within 20 km: {100 * w.mean():.1f}%; of these, rain at the flash {100 * np.mean(df.rain[w] > 0.1):.0f}%; "
      f"by month " + ", ".join(f"{m}: {100 * v:.1f}%" for m, v in w.groupby(df.month).mean().items()))
g = df.groupby("month").agg(n=("tb", "size"), warm=("warm", "mean"), cold235=("cold235", "mean"))
print((g.assign(warm=100 * g.warm, cold235=100 * g.cold235)).round(1).to_string())
g = df.groupby("region").agg(n=("tb", "size"), warm=("warm", "mean"))
print((g.assign(warm=100 * g.warm)).round(1).to_string())
for name, x in (("warm", df[df.warm]), ("cold", df[~df.warm])):
    print(f"{name}: median current {np.nanmedian(x.amps) / 1e3:.1f} kA, share >= 20 kA {100 * np.mean(x.amps >= 2e4):.1f}%, "
          f"IMERG rain at the flash > 0.1 mm/h: {100 * np.mean(x.rain > 0.1):.0f}%, "
          f"hours IST (quartiles) {np.round(np.nanpercentile(x.hour_ist, [25, 50, 75]), 1)}")
out = RESULTS / f"check_warm_flashes_{year}.csv"
df.to_csv(out, index=False, float_format="%.2f")
print(out)
