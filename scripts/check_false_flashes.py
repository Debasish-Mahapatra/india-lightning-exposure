#!/usr/bin/env python3
"""Check: are NRSC flashes under warm cloud tops false detections, or real
flashes matched to the wrong place or time? For every flash on the days with
a MERGIR file (image nearest in time, within 15 minutes):
  d273, d235   distance (km) to the nearest pixel colder than 273 K (ice is
               possible) and 235 K (deep cloud) in that image
  dt_*         distance to the nearest pixel colder than 273 K in the images
               0.5 to 2 hours before and after, and 12 hours before or after
               (a 12-hour shift would mean an AM/PM error in the flash time)
  n_cell       flashes in the same 0.25 degree cell and half hour (isolated
               flashes have 1)
  rain_day     IMERG daily rain of the flash's 0.25 degree cell (mm), as in the
               analysis of lightning with little rain
A flash with no cloud colder than 273 K within 20 km is "suspect". Prints how
suspect flashes differ from the others and saves the per-flash table.
Output: results/check_false_flashes_YYYY.csv.gz
Usage: check_false_flashes.py YEAR
"""
import sys

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import distance_transform_edt

from common import DATA, LAT, LON, OBS_HH, RESULTS, region_grid

year = int(sys.argv[1])
reg = region_grid()
z = np.load(DATA / f"nrsc_flashes/flashes_{year}.npz")
t = z["t"].astype(np.int64)
SHIFTS = [-4, -3, -2, -1, 1, 2, 3, 4]                 # half hours
R = xr.open_dataset(DATA / f"imerg_025/imerg_daily_025_{year}.nc")["precipitation"]
rows = []
for f in sorted((OBS_HH / f"mergir/{year}").glob("mergir_*.nc")):
    day = pd.Timestamp(f.stem.split("_")[1])
    sel = np.nonzero((t // 86400) == day.dayofyear - 1)[0]
    if sel.size == 0:
        continue
    d = xr.open_dataset(f)
    tb = d["Tb"].values
    lat, lon = d.lat.values.astype(float), d.lon.values.astype(float)
    dy = 111.2 * np.diff(lat).mean()
    dx = 111.2 * np.diff(lon).mean() * np.cos(np.deg2rad(22))
    k_img = ((d.time.values - np.datetime64(day)) / np.timedelta64(30, "m")).round().astype(int)
    good = {k: n for n, k in enumerate(k_img) if 0 <= k < 48 and np.isnan(tb[n]).mean() < 0.2}
    dist273, dist235 = {}, {}
    for k, n in good.items():                         # distance (km) to the nearest cold pixel
        x = np.nan_to_num(tb[n], nan=999.0)
        dist273[k] = distance_transform_edt(x >= 273, sampling=(dy, dx)).astype(np.float32)
        dist235[k] = distance_transform_edt(x >= 235, sampling=(dy, dx)).astype(np.float32)
    ft = t[sel] % 86400
    fk = np.rint(ft / 1800).astype(int)
    fr = np.clip(np.rint((z["lat"][sel] - lat[0]) / np.diff(lat).mean()).astype(int), 0, lat.size - 1)
    fc = np.clip(np.rint((z["lon"][sel] - lon[0]) / np.diff(lon).mean()).astype(int), 0, lon.size - 1)
    look = lambda dist, k: np.array([dist[kk][r, c] if kk in dist else np.nan for kk, r, c in zip(k, fr, fc)])
    out = {"month": day.month, "hour_ist": (ft / 3600 + 5.5) % 24, "amps": z["amps"][sel],
           "d273": look(dist273, fk), "d235": look(dist235, fk)}
    for s in SHIFTS:
        out[f"dt{s:+d}"] = look(dist273, fk + s)
    k12 = np.where(fk < 24, fk + 24, fk - 24)         # the other half of the same UTC day
    out["dt12h"] = look(dist273, k12)
    j = np.rint((z["lat"][sel] - LAT[0]) / 0.25).astype(int)
    i = np.rint((z["lon"][sel] - LON[0]) / 0.25).astype(int)
    key = (j * len(LON) + i) * 48 + np.minimum(ft // 1800, 47)
    u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    out["n_cell"] = cnt[inv]
    out["region"] = reg[j, i]
    out["rain_day"] = R.sel(time=day).values[j, i] if day in R.time.to_index() else np.nan
    rows.append(pd.DataFrame(out))
df = pd.concat(rows)
df = df[np.isfinite(df.d273)]
sus = df.d273 > 20
print(f"{year}: {len(df):,} flashes with a MERGIR image; {100 * sus.mean():.1f}% suspect "
      f"(no cloud colder than 273 K within 20 km)")
print("share suspect by month: " + ", ".join(f"{m}: {100 * v:.1f}%" for m, v in sus.groupby(df.month).mean().items()))
q = lambda x: np.round(np.nanpercentile(x, [25, 50, 75, 90]), 0)
print(f"suspect: distance to cloud colder than 273 K (km) quartiles and 90%: {q(df.d273[sus])}; "
      f"to cloud colder than 235 K: {q(df.d235[sus])}")
for name, m in (("suspect", sus), ("others", ~sus)):
    x = df[m]
    near = lambda c: 100 * np.mean(x[c] <= 20)
    print(f"{name}: cold cloud within 20 km 0.5/1/1.5/2 h earlier: "
          + "/".join(f"{near(f'dt{s:+d}'):.0f}%" for s in (-1, -2, -3, -4))
          + ", later: " + "/".join(f"{near(f'dt{s:+d}'):.0f}%" for s in (1, 2, 3, 4))
          + f", 12 h away: {near('dt12h'):.0f}%; isolated (alone in cell and half hour): "
          f"{100 * np.mean(x.n_cell == 1):.0f}%; median current {np.nanmedian(x.amps) / 1e3:.1f} kA")
print("suspect share by region: " + ", ".join(f"{r}: {100 * v:.1f}%" for r, v in sus.groupby(df.region).mean().items() if r))
h = np.histogram(df.hour_ist[sus], bins=np.arange(0, 25, 3))[0]
h2 = np.histogram(df.hour_ist[~sus], bins=np.arange(0, 25, 3))[0]
print("share by 3-hour IST bin, suspect: " + " ".join(f"{100 * v / h.sum():.0f}" for v in h)
      + " | others: " + " ".join(f"{100 * v / h2.sum():.0f}" for v in h2))
dry = df.rain_day < 2.5
for m, g in df.groupby("month"):
    d, s_ = g.rain_day < 2.5, g.d273 > 20
    print(f"month {m}: {100 * d.mean():.1f}% of flashes with less than 2.5 mm that day; suspect flashes are "
          f"{100 * (d & s_).sum() / max(d.sum(), 1):.1f}% of these; without suspect flashes the share would be "
          f"{100 * (d & ~s_).sum() / (~s_).sum():.1f}%")
out = RESULTS / f"check_false_flashes_{year}.csv.gz"
df.to_csv(out, index=False, float_format="%.1f")
print(out)
