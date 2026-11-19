#!/usr/bin/env python3
"""NRSC records and flashes inside India's official boundary, per year, as
tables of longitude, latitude, time (seconds after 1 January 00:00 UTC) and
peak current (A; NaN where not recorded). Records with an unreadable time are
dropped. Also the dates with an NRSC file.
About a quarter of the records follow an earlier record within a second and a
few km: later strokes of one flash (check_grouping.py). A record within 1 s
and 10 km of an earlier record joins that record's flash (common.group_flashes,
applied to all records of the day, inside and outside India). A flash takes
the place and time of its first record and the largest peak current of its
records, and counts if its first record lies inside India.
Output: data/nrsc_flashes/flashes_YYYY.npz (also n: records per flash; used by
        every analysis), data/nrsc_flashes/records_YYYY.npz (every record, for checks)
Usage: nrsc_flashes.py YEAR [YEAR ...]   (one process per year)
"""
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import shapely

from common import DATA, NRSC, group_flashes, india_geometry, read_day

OUT = DATA / "nrsc_flashes"
INDIA = india_geometry()
shapely.prepare(INDIA)


def table(cols, dates):
    out = {"lon": np.concatenate(cols["lon"]).astype(np.float32),
           "lat": np.concatenate(cols["lat"]).astype(np.float32),
           "t": np.concatenate(cols["t"]).astype(np.int32),
           "amps": np.concatenate(cols["amps"]).astype(np.float32),
           "dates": np.array(dates)}
    if "n" in cols:
        out["n"] = np.concatenate(cols["n"]).astype(np.int16)
    return out


def one_year(year):
    rec = {"lon": [], "lat": [], "t": [], "amps": []}
    fl = {"lon": [], "lat": [], "t": [], "amps": [], "n": []}
    dates, n_all, n_bad = [], 0, 0
    for k, day in enumerate(pd.date_range(f"{year}-01-01", f"{year}-12-31")):
        stem = NRSC / f"all/{year}/{year}/00{day:%m%Y}/{day:%d%m%Y}_ECVD"
        if not (stem.with_suffix(".shp").exists() and stem.with_suffix(".dbf").exists()):
            continue
        dates.append(day.strftime("%Y-%m-%d"))
        x, y, s, a = read_day(stem, current=True, seconds=True)
        n_all += len(x)
        n_bad += int((s < 0).sum())
        ok = s >= 0
        x, y, s, a = x[ok], y[ok], s[ok], a[ok]
        inside = shapely.contains_xy(INDIA, x, y)
        for c, v in zip(rec, (x[inside], y[inside], s[inside] + k * 86400, a[inside])):
            rec[c].append(v)
        root = group_flashes(x, y, s)
        first = inside & (root == np.arange(root.size))
        amax = np.full(root.size, np.nan)
        np.fmax.at(amax, root, a)
        n = np.bincount(root, minlength=root.size)
        for c, v in zip(fl, (x[first], y[first], s[first] + k * 86400, amax[first], n[first])):
            fl[c].append(v)
    r, f = table(rec, dates), table(fl, dates)
    np.savez_compressed(OUT / f"records_{year}.npz", **r)
    np.savez_compressed(OUT / f"flashes_{year}.npz", **f)
    return (f"{year}: {len(dates)} days, {n_all:,} records in the files ({n_bad} with an unreadable time); "
            f"inside India's official boundary {len(r['t']):,} records, {len(f['t']):,} flashes "
            f"({len(r['t']) / len(f['t']):.2f} records per flash; flashes with one record "
            f"{100 * np.mean(f['n'] == 1):.0f}%, with 2 {100 * np.mean(f['n'] == 2):.0f}%, "
            f"with 3 or more {100 * np.mean(f['n'] >= 3):.0f}%)")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    years = [int(y) for y in sys.argv[1:]]
    with Pool(min(len(years), 4)) as pool:
        for msg in pool.imap_unordered(one_year, years):
            print(msg, flush=True)
