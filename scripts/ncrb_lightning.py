#!/usr/bin/env python3
"""Lightning deaths by State/UT from NCRB's "Accidental Deaths & Suicides in
India" (ADSI) reports: Table 1.9 (deaths due to forces of nature, by cause and
gender), from the PDFs in data/ncrb/. Finds the page whose header names Lightning, reads each State/UT row
(four numbers per cause: male, female, transgender, total) and writes
data/ncrb/lightning_deaths.csv (year, state, male, female, transgender, total).
Usage: ncrb_lightning.py ADSI2021.pdf ADSI2022.pdf ... (year from the name)
"""
import csv
import re
import sys

import pypdf

from common import NCRB

OUT = NCRB / "lightning_deaths.csv"
CAUSES = ["Avalanche", "Exposure to Cold", "Cold", "Cyclone", "Tornado", "Starvation", "Earthquake",
          "Epidemic", "Flood", "Heat", "Forest Fire", "Landslide", "Lightning", "Torrential Rain",
          "Other"]


def lightning_rows(path):
    r = pypdf.PdfReader(path)
    for page in r.pages:
        t = page.extract_text() or ""
        if "Table 1.9" not in t or "Lightning" not in t:
            continue
        head = t.split("(1)")[0]                          # header, before the column numbers
        order = sorted((head.find(c), c) for c in ["Avalanche", "Cold", "Cyclone", "Starvation",
                       "Earthquake", "Epidemic", "Flood", "Heat", "Fire", "Landslide", "Lightning",
                       "Torrential", "Other"] if c in head)
        names = [c for _, c in order]
        if "Lightning" not in names:
            continue
        k = names.index("Lightning")
        rows = {}
        text = re.sub(r"AND\s*\n\s*DAMAN", "AND DAMAN", t)
        for m in re.finditer(r"^\s*(\d+)\s+([A-Z][A-Z &.()\-]+?)\s+((?:\d+\s+){%d}\d+)\s*$" % (4 * len(names) - 1),
                             text, re.M):
            v = [int(x) for x in m.group(3).split()]
            rows[" ".join(m.group(2).split())] = v[4 * k:4 * k + 4]     # one space between words, whatever the PDF reader gives
        tot = re.search(r"TOTAL \(ALL INDIA\)\s+((?:\d+\s+){%d}\d+)" % (4 * len(names) - 1), text)
        if rows and tot:
            v = [int(x) for x in tot.group(1).split()]
            return rows, v[4 * k:4 * k + 4]
    raise SystemExit(f"{path}: lightning table not found")


def main(paths):
    out = []
    for p in paths:
        year = int(re.search(r"(20\d\d)", p.split("/")[-1]).group(1))
        rows, total = lightning_rows(p)
        s = sum(v[3] for v in rows.values())
        assert s == total[3], f"{year}: states sum to {s}, table total {total[3]}"
        for state, v in rows.items():
            out.append([year, state] + v)
        print(f"{year}: {len(rows)} States/UTs, {total[3]} lightning deaths (male {total[0]}, female {total[1]})")
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["year", "state", "male", "female", "transgender", "total"])
        w.writerows(out)


if __name__ == "__main__":
    main(sys.argv[1:])
