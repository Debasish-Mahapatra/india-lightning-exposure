#!/usr/bin/env python3
"""How completely does the NRSC network record India's cloud-to-ground (CG)
lightning? Comparison with satellite total lightning and two internal signs of
missed weak flashes (npj Natural Hazards paper).

Satellites (LIS/OTD reprocessed flash climatology, GHRC lorfc V2, 0.1 degree,
flash counts scaled for detection efficiency and view time): ISS LIS in
2022-2023, the same years as the network, and the combined OTD, TRMM LIS and
ISS LIS climatology (1995-2023). Satellites see total lightning (cloud flashes
and CG), the network only CG, so NRSC flashes per satellite flash
= detection efficiency x CG fraction, and the CG fraction 1/(1+Z) varies with
the ratio Z of cloud to CG flashes (6-10 in the northwest, 0.5-3.5 in the
northeast; Ghosh et al. 2023, IITM network 2019-2022). The satellite counts are
annual, so each cell's expected CG density is L / (1 + sum_s w_s Z_s), with w_s
the network's share of CG flashes in season s (smoothed over 5 x 5 cells).
Rates: sums of flashes over sums of view time x area in 0.5 degree boxes
(5 x 5 cells of 0.1 degree) around each 0.25 degree cell.
Internal signs of missed weak flashes, where sensors are sparse: a larger share
of flashes of 20 kA or more, and fewer records (return strokes) per flash.
Output: results/detection_state.csv, results/detection_region.csv,
data/vulnerability/detection_025.nc
"""
import numpy as np
import pandas as pd
import xarray as xr
from pyhdf.SD import SD, SDC
from scipy import stats

from common import DATA, LAT, LON, POP, REGIONS, RESULTS, india_area, nrsc_clim, region_grid

V = DATA / "vulnerability"
A = india_area()
SEC_YEAR = 365.25 * 86400
pop = pd.read_csv(POP / "pop_state_cell.csv")


def satellite(fname, years=None):
    """Flash rate (flashes per km2 per year) on the 0.25 degree grid from 0.1
    degree counts and view times, as sums over 0.5 degree boxes."""
    h = SD(str(V / fname), SDC.READ)
    lat = h.select("Latitude" if "ISS" in fname and "COMB" not in fname else "LATITUDE")[:]
    lon = h.select("Longitude" if "ISS" in fname and "COMB" not in fname else "LONGITUDE")[:]
    j = np.where((lat > 4) & (lat < 40))[0].tolist()
    i = np.where((lon > 65) & (lon < 100))[0].tolist()
    f = h.select("flashcount_scaled")[j[0]:j[-1] + 1, i[0]:i[-1] + 1]
    vt = h.select("viewtime")[j[0]:j[-1] + 1, i[0]:i[-1] + 1]
    if years is not None:
        yr = list(h.select("Year")[:].astype(int))
        k = [yr.index(y) for y in years]
        f, vt = f[..., k].sum(-1), vt[..., k].sum(-1)
    area = h.select("grid_areas")[j[0]:j[-1] + 1, i[0]:i[-1] + 1]
    la, lo = lat[j[0]:j[-1] + 1], lon[i[0]:i[-1] + 1]
    out = np.full((len(LAT), len(LON)), np.nan)
    for a, y in enumerate(LAT):
        rj = np.abs(la - y) <= 0.25
        for b, x in enumerate(LON):
            ri = np.abs(lo - x) <= 0.25
            va = (vt[np.ix_(rj, ri)] * area[np.ix_(rj, ri)]).sum()
            if va > 0:
                out[a, b] = f[np.ix_(rj, ri)].sum() / va * SEC_YEAR
    return out


iss = satellite("ISSYearlyTimeSeries.hdf", years=[2022, 2023])
comb = satellite("COMB_OTD_TRMM_ISS_AnnualMean.hdf")

# Z (cloud flashes per CG flash) from the IITM network, 2019-2022 (Ghosh et al. 2023, Sect. 3.1): middle and
# ends of the ranges given for eight boxes, pre-monsoon and monsoon; elsewhere the all-India land means
# (3.81 and 3.04, standard deviations 0.79 and 0.50). Winter and post-monsoon take the mean of the two.
BOXES = [  # lon0, lon1, lat0, lat1, (pre-monsoon low, high), (monsoon low, high)
    (83, 88, 20, 27, (3, 6), (1, 4)),          # R1 east
    (73, 83, 25, 32, (4, 8), (6, 10)),         # R2 northwest
    (90, 98, 22, 30, (1.5, 4.5), (0.5, 3.5)),  # R3 northeast
    (80, 85, 15, 20, (5, 10), (2, 4)),         # R5 Odisha and Andhra coasts
    (75, 83, 20, 25, (3, 4.5), (2.5, 3.5)),    # R7 central
    (73, 78, 32, 36, (3, 4.5), (4.2, 6.5))]    # R8 Jammu and Kashmir
ALL = {"pre": (3.81 - 0.79, 3.81 + 0.79), "mon": (3.04 - 0.50, 3.04 + 0.50)}


def zfield(season, case):
    k = {"low": 0, "high": 1}
    z = np.full((len(LAT), len(LON)), np.mean(ALL[season]) if case == "mid" else ALL[season][k[case]])
    for lo0, lo1, la0, la1, pre, mon in BOXES:
        rng = pre if season == "pre" else mon
        v = np.mean(rng) if case == "mid" else rng[k[case]]
        z[np.ix_((LAT >= la0) & (LAT < la1), (LON >= lo0) & (LON < lo1))] = v
    return z


from scipy.ndimage import uniform_filter
mon = np.nan_to_num(nrsc_clim("flashes").sum("hour_utc").values)             # (month, lat, lon)
sm = lambda x: uniform_filter(x, size=5, mode="nearest")                       # about 1 degree
seas = {"pre": sm(mon[2:5].sum(0)), "mon": sm(mon[5:9].sum(0)), "other": sm(mon[[0, 1, 9, 10, 11]].sum(0))}
tot = seas["pre"] + seas["mon"] + seas["other"]
# CG fraction of each cell's annual total lightning. With w_s the share of the year's CG flashes in season s
# (the network's, smoothed) and Z_s the ratio of cloud to CG flashes, the year's cloud flashes are
# CG x sum_s w_s Z_s, so total = CG (1 + sum_s w_s Z_s) and CG = total / (1 + sum_s w_s Z_s): the ratios are
# averaged with CG weights before the conversion. (Averaging the fractions 1/(1+Z_s) with CG weights, as before
# 30 September 2026, would need shares of total lightning and overstates the CG fraction.)
fcg, fcg_old = {}, {}
for case in ("mid", "low", "high"):
    zp, zm = zfield("pre", case), zfield("mon", case)
    zo = 0.5 * (zp + zm)
    den = np.where(tot > 0, tot, np.nan)
    fcg[case] = 1 / (1 + (seas["pre"] * zp + seas["mon"] * zm + seas["other"] * zo) / den)
    fcg_old[case] = (seas["pre"] / (1 + zp) + seas["mon"] / (1 + zm) + seas["other"] / (1 + zo)) / den
# satellite CG: the low Z case gives the most CG flashes
iss_cg = {"mid": iss * fcg["mid"], "high_cg": iss * fcg["low"], "low_cg": iss * fcg["high"]}
# worked example: one cell on the Odisha coast (box R5, where Z differs most between the seasons)
j0, i0 = np.argmin(np.abs(LAT - 19.5)), np.argmin(np.abs(LON - 84.75))
w = np.array([seas[k][j0, i0] for k in ("pre", "mon", "other")]) / tot[j0, i0]
zp0, zm0 = zfield("pre", "mid")[j0, i0], zfield("mon", "mid")[j0, i0]
z = np.array([zp0, zm0, 0.5 * (zp0 + zm0)])
L = iss[j0, i0]
C = L / (1 + (w * z).sum())
G = C * w                                                   # CG flashes in each season
print(f"Worked example, cell {LAT[j0]:.2f} N {LON[i0]:.2f} E: ISS LIS total L = {L:.2f} flashes per km2 per year; "
      f"network CG shares (pre-monsoon, monsoon, rest) {np.round(w, 3)}; Z {np.round(z, 2)}")
print(f"  CG = L / (1 + sum w Z) = {L:.2f} / {1 + (w * z).sum():.3f} = {C:.3f}; CG by season {np.round(G, 3)}, cloud "
      f"flashes {np.round(z * G, 3)}; CG + cloud = {(G * (1 + z)).sum():.3f} = L; stored value "
      f"{iss_cg['mid'][j0, i0]:.3f}; the former fraction-averaging gave {L * fcg_old['mid'][j0, i0]:.3f}")
k_ = np.isfinite(iss) & np.isfinite(A)
print(f"India: CG fraction {np.nansum((iss * fcg['mid'] * A)[k_]) / np.nansum((iss * A)[k_]):.3f} with the ratios averaged "
      f"first, {np.nansum((iss * fcg_old['mid'] * A)[k_]) / np.nansum((iss * A)[k_]):.3f} with the fractions averaged")
net = nrsc_clim("flashes", years=[2022, 2023]).sum(("month", "hour_utc")).values / A     # same years as ISS
s20 = np.nansum(nrsc_clim("flashes_ge20kA").values, 0)
fl = nrsc_clim("flashes").sum(("month", "hour_utc")).values
# records per flash on the grid, 2022-2025
rec, nfl = np.zeros_like(A), np.zeros_like(A)
for y in range(2022, 2026):
    z = np.load(DATA / f"nrsc_flashes/flashes_{y}.npz")
    jj = np.rint((z["lat"] - LAT[0]) / 0.25).astype(int)
    ii = np.rint((z["lon"] - LON[0]) / 0.25).astype(int)
    np.add.at(rec, (jj, ii), z["n"])
    np.add.at(nfl, (jj, ii), 1)
xr.Dataset({"nrsc_2022_2023": (("lat", "lon"), net), "iss_2022_2023": (("lat", "lon"), iss),
            "lis_otd_1995_2023": (("lat", "lon"), comb), "share_ge20kA": (("lat", "lon"), s20 / fl),
            "records_per_flash": (("lat", "lon"), rec / np.maximum(nfl, 1)),
            "cg_fraction": (("lat", "lon"), fcg["mid"]), "iss_cg_2022_2023": (("lat", "lon"), iss_cg["mid"]),
            "iss_cg_low": (("lat", "lon"), iss_cg["low_cg"]), "iss_cg_high": (("lat", "lon"), iss_cg["high_cg"])},
           coords={"lat": LAT, "lon": LON}).to_netcdf(V / "detection_025.nc")

# by state: area means of densities, population-weighted exposures and the diagnostics
g = pop.assign(net=net[pop.j, pop.i], iss=iss[pop.j, pop.i], comb=comb[pop.j, pop.i],
               cg=iss_cg["mid"][pop.j, pop.i], cg_lo=iss_cg["low_cg"][pop.j, pop.i], cg_hi=iss_cg["high_cg"][pop.j, pop.i],
               s20=s20[pop.j, pop.i], fl=fl[pop.j, pop.i], rec=rec[pop.j, pop.i], nfl=nfl[pop.j, pop.i])
g = g[np.isfinite(g.iss) & np.isfinite(g.net)]
wa = lambda d, c: (d.area_km2 * d[c]).sum() / d.area_km2.sum()
wp = lambda d, c: (d.population * d[c]).sum() / d.population.sum()
t = pd.DataFrame({"area_km2": g.groupby("state").area_km2.sum(),
                  "net": g.groupby("state").apply(wa, "net", include_groups=False),
                  "iss": g.groupby("state").apply(wa, "iss", include_groups=False),
                  "comb": g.groupby("state").apply(wa, "comb", include_groups=False),
                  "E_iss": g.groupby("state").apply(wp, "iss", include_groups=False),
                  "E_comb": g.groupby("state").apply(wp, "comb", include_groups=False),
                  "cg": g.groupby("state").apply(wa, "cg", include_groups=False),
                  "E_cg": g.groupby("state").apply(wp, "cg", include_groups=False),
                  "E_cg_low": g.groupby("state").apply(wp, "cg_lo", include_groups=False),
                  "E_cg_high": g.groupby("state").apply(wp, "cg_hi", include_groups=False),
                  "share_ge20kA": g.groupby("state").s20.sum() / g.groupby("state").fl.sum(),
                  "records_per_flash": g.groupby("state").rec.sum() / g.groupby("state").nfl.sum()})
t["net_per_iss"] = t.net / t.iss
t["net_per_comb"] = t.net / t.comb
t["de"] = t.net / t.cg                                     # detection relative to satellite CG, middle case
t.round(5).to_csv(RESULTS / "detection_state.csv")

reg = region_grid()
rows = []
print("NRSC CG flashes per satellite flash by region (area means; ISS LIS and NRSC both 2022-2023):")
for r in list(REGIONS) + ["India"]:
    k = ((reg == r) if r != "India" else np.isfinite(A)) & np.isfinite(iss) & np.isfinite(net) & np.isfinite(A)
    num, den = np.nansum((net * A)[k]), np.nansum((iss * A)[k])
    num2 = np.nansum((comb * A)[k])
    print(f"  {r:15s} ISS {den / np.nansum(A[k]):5.1f}, long-term {num2 / np.nansum(A[k]):5.1f}, NRSC "
          f"{num / np.nansum(A[k]):4.2f} per km2 per year; NRSC/ISS {num / den:.3f}, NRSC/long-term {num / num2:.3f}; "
          f"share >= 20 kA {100 * np.nansum(s20[k]) / np.nansum(fl[k]):4.1f}%, records per flash "
          f"{rec[k].sum() / nfl[k].sum():.2f}")
    cgm, cgl, cgh = (np.nansum((iss_cg[c] * A)[k]) for c in ("mid", "low_cg", "high_cg"))
    rows.append({"region": r, "satellite_total": den / np.nansum(A[k]), "satellite_cg": cgm / np.nansum(A[k]),
                 "cg_fraction": cgm / den, "network": num / np.nansum(A[k]), "detection": num / cgm,
                 "detection_lo": num / cgh, "detection_hi": num / cgl,
                 "share_ge20kA": np.nansum(s20[k]) / np.nansum(fl[k]), "records_per_flash": rec[k].sum() / nfl[k].sum()})
    print(f"  {'':15s} CG fraction {cgm / den:.2f} ({cgl / den:.2f}-{cgh / den:.2f}); satellite CG "
          f"{cgm / np.nansum(A[k]):4.1f} per km2 per year; network detection {num / cgm:.2f} ({num / cgh:.2f}-{num / cgl:.2f})")
pd.DataFrame(rows).round(5).to_csv(RESULTS / "detection_region.csv", index=False)
k = np.isfinite(iss) & np.isfinite(net) & np.isfinite(A)
print(f"  India           NRSC/ISS {np.nansum((net * A)[k]) / np.nansum((iss * A)[k]):.3f}; network detection of "
      f"satellite CG {np.nansum((net * A)[k]) / np.nansum((iss_cg['mid'] * A)[k]):.2f} "
      f"({np.nansum((net * A)[k]) / np.nansum((iss_cg['high_cg'] * A)[k]):.2f}-"
      f"{np.nansum((net * A)[k]) / np.nansum((iss_cg['low_cg'] * A)[k]):.2f}); satellite CG over India "
      f"{np.nansum((iss_cg['mid'] * A)[k]) / 1e6:.1f} million a year against the network's {np.nansum((net * A)[k]) / 1e6:.1f}")
big = t[t.area_km2 > 2e4]
print(f"\n{len(big)} states larger than 20,000 km2: rank correlation of NRSC/ISS with")
for c in ("share_ge20kA", "records_per_flash", "net_per_comb", "de"):
    r = stats.spearmanr(big.net_per_iss, big[c])
    print(f"  {c:18s}: rho = {r.statistic:+.2f} (p = {r.pvalue:.3f})")
# the same across 1 degree boxes with enough flashes, where the grid is finer than states
bj, bi = (np.arange(len(LAT)) // 4), (np.arange(len(LON)) // 4)
df = pd.DataFrame({"bj": np.repeat(bj, len(LON)), "bi": np.tile(bi, len(LAT)), "net": (net * A).ravel(),
                   "iss": (iss * A).ravel(), "s20": s20.ravel(), "fl": fl.ravel(), "rec": rec.ravel(),
                   "nfl": nfl.ravel()}).dropna()
b = df.groupby(["bj", "bi"]).sum()
b = b[(b.fl >= 2000) & (b.iss > 0)]
b["ratio"], b["s"], b["m"] = b.net / b.iss, b.s20 / b.fl, b.rec / b.nfl
for c, name in (("s", "share >= 20 kA"), ("m", "records per flash")):
    r = stats.spearmanr(b.ratio, b[c])
    print(f"  1 degree boxes ({len(b)}): NRSC/ISS against {name}: rho = {r.statistic:+.2f}")
print("\nDetection against satellite CG (de) by state and the internal signs, rank correlations:")
for c in ("share_ge20kA", "records_per_flash"):
    r = stats.spearmanr(big.de, big[c])
    print(f"  de vs {c}: rho = {r.statistic:+.2f} (p = {r.pvalue:.3f})")
print(t.sort_values("net_per_iss")[["net", "iss", "cg", "de", "net_per_iss", "net_per_comb", "share_ge20kA",
                                    "records_per_flash"]].round(3).to_string())
