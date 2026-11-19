#!/usr/bin/env python3
"""Numbers of Supplementary Note 11 on the flashes with no cloud colder than 273 K within 20 km,
from the tables written by check_false_flashes.py (2022-2025).
Output: printed; keep it in results/log_check_false_flashes.txt
"""
import numpy as np
import pandas as pd

from common import RESULTS

COLS = ["month", "d273", "dt-4", "dt+4", "dt12h", "n_cell", "rain_day"]
d = pd.concat([pd.read_csv(RESULTS / f"check_false_flashes_{y}.csv.gz", usecols=COLS).assign(year=y)
               for y in range(2022, 2026)])
d["sus"] = d.d273 > 20
print(f"{len(d):,} flashes matched to an image, {100 * d.sus.mean():.1f}% with no cloud colder than 273 K within 20 km")

by = d.groupby(["year", "month"]).sus.agg(["sum", "size"])
share = 100 * by["sum"] / by["size"]
print(f"Single months (48): flagged {by['sum'].min():,}-{by['sum'].max():,}, {by['sum'].max() / by['sum'].min():.0f}-fold; "
      f"all flashes {by['size'].min():,}-{by['size'].max():,}, {by['size'].max() / by['size'].min():.0f}-fold")
cal = by.groupby("month").mean()
print(f"Calendar months (mean of the four years): flagged {cal['sum'].min():,.0f}-{cal['sum'].max():,.0f}, "
      f"{cal['sum'].max() / cal['sum'].min():.1f}-fold; all flashes {cal['size'].min():,.0f}-{cal['size'].max():,.0f}, "
      f"{cal['size'].max() / cal['size'].min():.0f}-fold")
quiet = share[by["size"] < 100000]
print(f"{len(quiet)} months with fewer than 100,000 flashes: flagged share {quiet.min():.1f}-{quiet.max():.1f}%, median {quiet.median():.1f}%")
for name, months in (("winter", [12, 1, 2]), ("monsoon", [6, 7, 8, 9])):
    print(f"{name}: flagged share {100 * d.sus[d.month.isin(months)].mean():.2f}%")

for y, g in d.groupby("year"):
    s = g[g.sus]
    near = lambda c, x: 100 * (x[c] <= 20).sum() / x[c].notna().sum()
    print(f"{y}: alone in cell and half hour {100 * (s.n_cell == 1).mean():.1f}% (others {100 * (g[~g.sus].n_cell == 1).mean():.1f}%); "
          f"median distance {s.d273.median():.1f} km, 12 h away {s.dt12h.median():.1f} km; "
          f"cold cloud within 20 km 12 h away {near('dt12h', s):.0f}% (others {near('dt12h', g[~g.sus]):.0f}%), "
          f"2 h away {near('dt-4', s):.0f}% and {near('dt+4', s):.0f}%; closer 12 h away {100 * (s.dt12h < s.d273).sum() / s.dt12h.notna().sum():.0f}%")

j = d[(d.year == 2024) & (d.month == 1)]
print(f"January 2024: share of flashes below 2.5 mm {100 * (j.rain_day < 2.5).mean():.1f}%, "
      f"without the flagged flashes {100 * (j[~j.sus].rain_day < 2.5).mean():.1f}%")
