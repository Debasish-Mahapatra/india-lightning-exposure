#!/usr/bin/env python3
"""Cloud tops over India's lightning (MERGIR, light version: no storm tracking),
per year, from the half-hourly MERGIR files (data/obs_hh/mergir, 4 km, Tb in K)
and the NRSC flashes inside India's official boundary (data/nrsc_flashes).
Each flash goes with the image nearest in time (within 15 minutes).
  tb_hist     flashes by month, region and cloud-top brightness temperature: the
              coldest pixel within about 12 km of the flash (7 x 7 pixels, to
              allow for parallax, which shifts cloud tops by 10 km or more at the
              slant views of India, and for location errors), 1 K bins
  box_*       per month and 1 degree box: summed area (km2 x half hours) of
              cloud colder than 235 K and 208 K (pixels inside India only),
              flashes, and the number of valid images
  bin_*       per month, region and bin of the box's cold-cloud area (boxes at
              least half inside India): number of box-images, and summed flashes
              and squared flashes, for 235 K and 208 K
  day_*       per day and region: flashes and cold-cloud area (km2 x half hours)
Output: data/mergir_light/mergir_light_YYYY.nc
Usage: mergir_light.py YEAR [YEAR ...]   (one process per year)
"""
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import shapely
import xarray as xr
from scipy.ndimage import minimum_filter

from common import DATA, LAT, LON, OBS_HH, REGIONS, india_geometry, region_grid

OUT = DATA / "mergir_light"
TB = np.arange(170, 331)                       # bin edges, K
AREA = np.array([0, 1, 250, 500, 1000, 2000, 3000, 4000, 6000, 8000, 10000, 13000])   # km2 per box
NAMES = list(REGIONS)
NR = len(NAMES)


def geometry(lat, lon):
    """India mask, region number (-1: none), 1 degree box number and pixel area of the MERGIR grid."""
    lon2, lat2 = np.meshgrid(lon, lat)
    inside = shapely.contains_xy(india_geometry(), lon2, lat2)
    reg = region_grid()
    j = np.clip(np.rint((lat2 - LAT[0]) / 0.25).astype(int), 0, len(LAT) - 1)
    i = np.clip(np.rint((lon2 - LON[0]) / 0.25).astype(int), 0, len(LON) - 1)
    rnum = np.vectorize({n: k for k, n in enumerate(NAMES)}.get, otypes=[int])(np.where(reg == "", "x", reg), -1)[j, i]
    rnum[~inside] = -1
    box = np.clip(np.floor(lat2 - 6).astype(int), 0, 31) * 31 + np.clip(np.floor(lon2 - 67).astype(int), 0, 30)
    dlat, dlon = np.diff(lat).mean(), np.diff(lon).mean()
    area = (111.32 ** 2) * dlat * dlon * np.cos(np.deg2rad(lat2))
    return inside, rnum, box, area


def one_year(year):
    files = sorted((OBS_HH / f"mergir/{year}").glob("mergir_*.nc"))
    z = np.load(DATA / f"nrsc_flashes/flashes_{year}.npz")
    t = z["t"].astype(np.int64)
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31")
    g = None
    tb_hist = np.zeros((12, NR, TB.size - 1))
    box_a235, box_a208, box_f, box_n = (np.zeros((12, 32 * 31)) for _ in range(4))
    bins = {c: np.zeros((3, 12, NR, AREA.size - 1)) for c in (235, 208)}
    day_f, day_a235, day_a208 = (np.full((days.size, NR), np.nan) for _ in range(3))
    for f in files:
        day = pd.Timestamp(f.stem.split("_")[1])
        d = xr.open_dataset(f)
        if g is None:
            lat, lon = d.lat.values.astype(np.float64), d.lon.values.astype(np.float64)
            inside, rnum, box, parea = geometry(lat, lon)
            lat0, lon0, dlat, dlon = lat[0], lon[0], np.diff(lat).mean(), np.diff(lon).mean()
            boxfrac = np.bincount(box[inside], minlength=992) / np.maximum(np.bincount(box.ravel(), minlength=992), 1)
            boxreg = np.array([np.bincount(rnum[inside & (box == b) & (rnum >= 0)], minlength=NR).argmax()
                               if (inside & (box == b) & (rnum >= 0)).any() else -1 for b in range(992)])
            good_box = (boxfrac >= 0.5) & (boxreg >= 0)
        tb = d["Tb"].values                                            # (48, lat, lon)
        k_img = ((d.time.values - np.datetime64(day)) / np.timedelta64(30, "m")).round().astype(int)
        valid = np.isfinite(tb[:, inside]).mean(1) > 0.9               # images covering India
        m, doy = day.month - 1, day.dayofyear - 1
        # cold-cloud area per image and box (India pixels only)
        a235 = np.zeros((48, 992))
        a208 = np.zeros((48, 992))
        for n, k in enumerate(k_img):
            if not valid[n] or not 0 <= k < 48:
                continue
            x = tb[n]
            a235[k] = np.bincount(box[inside], weights=parea[inside] * (x[inside] < 235), minlength=992)
            a208[k] = np.bincount(box[inside], weights=parea[inside] * (x[inside] < 208), minlength=992)
        ok_k = np.zeros(48, bool)
        ok_k[k_img[valid & (k_img >= 0) & (k_img < 48)]] = True
        # flashes of this day, with the nearest image
        sel = (t // 86400) == doy
        ft = t[sel] % 86400
        fk = np.rint(ft / 1800).astype(int)
        keep = (fk < 48)
        keep[keep] &= ok_k[fk[keep]]
        flat, flon, fk = z["lat"][sel][keep], z["lon"][sel][keep], fk[keep]
        fr = np.clip(np.rint((flat - lat0) / dlat).astype(int), 0, lat.size - 1)
        fc = np.clip(np.rint((flon - lon0) / dlon).astype(int), 0, lon.size - 1)
        fbox = box[fr, fc]
        freg = rnum[fr, fc]
        fl = np.bincount(fk * 992 + fbox, minlength=48 * 992).reshape(48, 992).astype(float)
        # coldest pixel near each flash
        for n, k in enumerate(k_img):
            if not valid[n] or not 0 <= k < 48:
                continue
            here = fk == k
            if not here.any():
                continue
            cold = minimum_filter(np.nan_to_num(tb[n], nan=999.0), size=7)[fr[here], fc[here]]
            use = (cold < 900) & (freg[here] >= 0)
            np.add.at(tb_hist, (m, freg[here][use], np.clip(np.floor(cold[use]).astype(int) - TB[0], 0, TB.size - 2)), 1)
        # accumulate
        box_a235[m] += a235[ok_k].sum(0)
        box_a208[m] += a208[ok_k].sum(0)
        box_f[m] += fl[ok_k].sum(0)
        box_n[m] += ok_k.sum()
        for c, a in ((235, a235), (208, a208)):
            ab = np.clip(np.digitize(a[ok_k][:, good_box], AREA) - 1, 0, AREA.size - 2)
            rb = np.broadcast_to(boxreg[good_box], ab.shape)
            ff = fl[ok_k][:, good_box]
            for q, w in enumerate((np.ones_like(ff), ff, ff ** 2)):
                np.add.at(bins[c][q], (m, rb.ravel(), ab.ravel()), w.ravel())
        if ok_k.any():
            day_f[doy] = np.bincount(freg[freg >= 0], minlength=NR)
            for arr, a in ((day_a235, a235), (day_a208, a208)):
                arr[doy] = np.bincount(boxreg[boxreg >= 0], weights=a[ok_k].sum(0)[boxreg >= 0], minlength=NR)
    ds = xr.Dataset(
        {"tb_hist": (("month", "region", "tb"), tb_hist, {"long_name": "flashes by coldest cloud top within 12 km"}),
         "box_area235": (("month", "box"), box_a235, {"units": "km2 half-hours"}),
         "box_area208": (("month", "box"), box_a208, {"units": "km2 half-hours"}),
         "box_flashes": (("month", "box"), box_f), "box_images": (("month", "box"), box_n),
         "box_india_fraction": (("box",), boxfrac), "box_region": (("box",), boxreg),
         **{f"bin{c}_{q}": (("month", "region", "area_bin"), bins[c][k])
            for c in (235, 208) for k, q in enumerate(("n", "flashes", "flashes2"))},
         "day_flashes": (("time", "region"), day_f), "day_area235": (("time", "region"), day_a235),
         "day_area208": (("time", "region"), day_a208)},
        coords={"month": np.arange(1, 13), "region": NAMES, "tb": TB[:-1] + 0.5, "box": np.arange(992),
                "area_bin": 0.5 * (AREA[:-1] + AREA[1:]), "time": days},
        attrs={"box": "1 degree boxes, number = floor(lat - 6) * 31 + floor(lon - 67)", "area_edges": str(AREA.tolist()),
               "files": len(files)})
    ds.to_netcdf(OUT / f"mergir_light_{year}.nc")
    return f"{year}: {len(files)} days, {int(tb_hist.sum()):,} flashes with a cloud-top temperature"


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    years = [int(y) for y in sys.argv[1:]]
    with Pool(min(len(years), 4)) as pool:
        for msg in pool.imap_unordered(one_year, years):
            print(msg, flush=True)
