#!/usr/bin/env python3
"""Check: is each NRSC record one flash, or one stroke of a flash? Taori et al.
(2024, JASTP 261:106286) grouped the network's records within 10 km and 1 minute
into one flash before comparing with WWLLN. For every record inside India (records_YYYY.npz): the
time (s) and distance (km) to the nearest earlier record within 10 km and 60 s.
Strokes of one flash follow each other within about a second and a few km, so
if the records were strokes many would have a neighbour within 1 s, well above
the rate at 2-4 s (other flashes of the same storm).
Prints the share of records with an earlier record within 1 s and 10 km by
year, region, season and hour, and the peak currents. (The flashes themselves
are grouped in nrsc_flashes.py; check_grouping_effect.py compares the results.)
Also repeats Taori et al. (2024, Table 1) for 2-9 October 2022: daily counts
above 1, 5 and 12 kA, and above 12 kA after grouping within 10 km and 60 s.
Output: printed summary (results/check_grouping.txt)
Usage: check_grouping.py
"""
import numpy as np
import pandas as pd
from common import DATA, LAT, LON, SEASONS, YEARS, earlier_neighbour, region_grid

reg = region_grid()
season_of = {m: s for s, months in SEASONS.items() for m in months}
edges = np.array([0, 1, 2, 5, 10, 20, 30, 61])
rows = []
for year in YEARS:
    z = np.load(DATA / f"nrsc_flashes/records_{year}.npz")
    t, lon, lat, amps = z["t"].astype(np.int64), z["lon"], z["lat"], z["amps"]
    day = t // 86400
    dts, ds, nb = np.full(t.size, np.nan), np.full(t.size, np.nan), np.full(t.size, -1)
    for dd in np.unique(day):
        sel = np.nonzero(day == dd)[0]
        a, b, c = earlier_neighbour(lon[sel], lat[sel], t[sel])
        dts[sel], ds[sel] = a, b
        nb[sel] = np.where(c >= 0, sel[np.maximum(c, 0)], -1)
    has = np.isfinite(dts)
    h = np.histogram(dts[has], edges)[0]
    print(f"{year}: {t.size:,} records; {100 * has.mean():.1f}% have an earlier record within 10 km and 60 s; "
          f"nearest earlier one at " + ", ".join(f"{a}-{b - 1} s: {100 * n / t.size:.1f}%" for a, b, n in zip(edges[:-1], edges[1:], h)))
    later = has & (dts <= 1)
    print(f"   within 1 s: {100 * later.mean():.2f}% of records, median distance {np.nanmedian(ds[later]):.1f} km; "
          f"within 1 s and 2 km: {100 * np.mean(later & (ds <= 2)):.2f}%")
    ratio = amps[later] / amps[nb[later]]
    print(f"   peak current below 5 kA: {100 * np.mean(amps < 5e3):.1f}%, below 12 kA: {100 * np.mean(amps < 12e3):.1f}%; "
          f"median current: other records {np.median(amps[~later]) / 1e3:.1f} kA, later ones {np.median(amps[later]) / 1e3:.1f} kA; "
          f"later / earlier current: median {np.median(ratio):.2f}, within 10% of each other {100 * np.mean(abs(ratio - 1) < 0.1):.0f}%")
    j = np.clip(np.rint((lat - LAT[0]) / 0.25).astype(int), 0, len(LAT) - 1)
    i = np.clip(np.rint((lon - LON[0]) / 0.25).astype(int), 0, len(LON) - 1)
    month = (pd.Timestamp(f"{year}-01-01") + pd.to_timedelta(day, "D")).month
    rows.append(pd.DataFrame({"year": year, "region": reg[j, i], "season": pd.Series(month).map(season_of).values,
                              "hour_ist": ((t % 86400) / 3600 + 5.5) % 24 // 3 * 3, "later": later}))
    if year == 2022:
        print("   Taori et al. (2024) Table 1, LDSN over India, 2-9 October 2022:")
        print("   day      >1 kA  >5 kA >12 kA  >12 kA grouped   (ours, UTC days, inside India)")
        for dd in range(274, 282):                  # 2-9 October (day of year 275-282)
            sel = np.nonzero(day == dd)[0]
            a = amps[sel]
            g = sel[a > 12e3]
            dt12 = earlier_neighbour(lon[g], lat[g], t[g])[0]
            print(f"   Oct {dd - 272:02d} {np.sum(a > 1e3):7,} {np.sum(a > 5e3):6,} {np.sum(a > 12e3):6,} {np.sum(~np.isfinite(dt12)):9,}")
df = pd.concat(rows)
print(f"\n2022-2025: {100 * df.later.mean():.1f}% of records have an earlier record within 1 s and 10 km")
for key in ("region", "season", "hour_ist", "year"):
    share = df[df[key] != ""].groupby(key).later.mean()
    print(f"by {key}: " + ", ".join(f"{k}: {100 * v:.1f}%" for k, v in share.items()))
