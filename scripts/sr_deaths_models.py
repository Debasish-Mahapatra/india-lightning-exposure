#!/usr/bin/env python3
"""Lightning deaths against hazard and vulnerability, with count models
(npj Natural Hazards paper).

Deaths are counts, so the models are Poisson regressions with the log of
person-years as offset and a negative binomial check for extra variance:
  across states (2022-2024):  log E[deaths] = log(3 x population) + a + b log H + c X
  within states (2022-2024):  log E[deaths_sy] = log(population) + a_s + b log H_sy
where H is a hazard measure and X a vulnerability covariate. Hazard measures:
  E        NRSC flash density where people live (2022-2025)
  E_out    the same while people are outside the home (Time Use Survey 2024)
  E_iss    satellite total lightning where people live (ISS LIS 2022-2023)
  E_cg     satellite CG where people live: ISS LIS total lightning times the
           CG fraction 1/(1+Z), Z from the IITM network by region and season
           (Ghosh et al. 2023; sr_detection.py), with low and high cases
  E_out_cg E_cg with the state's NRSC timing of outdoor exposure
Covariates: Census 2011 agricultural workers per person, literacy; Time Use
Survey 2024 median consumption per person (log) and share in katcha homes.
  E_day    E times the state's whole-day share of people outside (no timing)
  E_open   the flash density where people live while they do outdoor work
           (tus_outdoor.py); E_open_act the same for the activity codes only
           (without the usual-industry rule), E_open_eq with each half hour
           shared equally among the activities recorded in it; *_cg the
           satellite CG versions (E_cg times the state's outdoor-work share)
Samples: the 30 States/UTs with at least a million people in the area enumerated
by the 2011 Census, including the 5 with no recorded death (the 25 with deaths
for comparison), and the states where the network records at least 1.35
records per flash (the best-detected ones).
Also: benchmarks for the timing of time outside (separate hazard and activity
terms), the farm-share coefficient with outdoor-work exposure, a matched table
of models for network and satellite exposure (coefficients, dispersion, AIC,
deviance explained), and the stability of effective detection within states
(annual exposure corrected cell by cell with the detection estimated from ISS
LIS; the internal signs by state and year). Within states, standard errors are
clustered by state (CR1, t with G - 1 degrees of freedom), and a wild cluster
score bootstrap (Rademacher weights, all 2^G sign patterns; Kline and Santos
2012) gives p-values and, by test inversion, 95% intervals; with and without
year fixed effects.
Output: printed (results/log_sr_deaths_models.txt); every fit in results/deaths_models.csv;
the matched table in results/deaths_models_table.csv
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import DATA, NCRB, POP, REGIONS, RESULTS, india_area, nrsc_clim

A = india_area()
t = pd.read_csv(RESULTS / "vulnerability_state.csv", index_col=0)
d = pd.read_csv(RESULTS / "detection_state.csv", index_col=0)
t = t.join(d[["net", "iss", "E_iss", "E_comb", "net_per_iss", "records_per_flash", "share_ge20kA", "de",
            "E_cg", "E_cg_low", "E_cg_high"]])
TO_REGION = {s: r for r, ss in REGIONS.items() for s in ss}
t["region"] = [TO_REGION.get(s, "") for s in t.index]
t["out_frac"] = t.E_out / t.E
for c in ("E_cg", "E_cg_low", "E_cg_high"):
    t[c.replace("E_", "E_out_")] = t[c] * t.out_frac
t["E_open_cg"] = t.E_cg * t.E_open / t.E
for v in ("act", "eq"):
    t[f"E_open_{v}_cg"] = t.E_cg * t[f"E_open_{v}"] / t.E
t["mpce_log"] = np.log(t.tus_mpce)
t["O_timing"] = t.E_out / t.E
big = t[(t.pop_m >= 1) & t.deaths.notna()].copy()
big["n3"] = np.rint(big.deaths * 3)
big["offset"] = np.log(3 * big.pop_m * 1e6)
pos = big[big.deaths > 0]
best = big[big.records_per_flash >= 1.35]
N = len(big)
print(f"{N} States/UTs with at least a million people; {N - len(pos)} with no recorded death "
      f"({', '.join(big.index[big.deaths == 0].str.title())}); {len(best)} best-detected (at least 1.35 records per "
      f"flash): {', '.join(best.index.str.title())}")


def model(df, hazard, cov=None, family="poisson"):
    X = pd.DataFrame({"logH": np.log(df[hazard])}, index=df.index)
    if cov:
        X[cov] = df[cov]
    X = sm.add_constant(X)
    fam = sm.families.Poisson() if family == "poisson" else sm.families.NegativeBinomial(alpha=family)
    return sm.GLM(df.n3, X, family=fam, offset=df.offset).fit()


def nb_alpha(df, hazard, cov=None):
    """Negative binomial dispersion by maximum likelihood."""
    X = pd.DataFrame({"logH": np.log(df[hazard])}, index=df.index)
    if cov:
        X[cov] = df[cov]
    X = sm.add_constant(X)
    return sm.NegativeBinomial(df.n3, X, offset=df.offset).fit(disp=0).params["alpha"]


FITS = []


def show(label, r, sample="all", hazard="", cov=""):
    ci = r.conf_int()
    FITS.append({"sample": sample, "hazard": hazard, "covariate": cov, "label": label,
                 "intercept": r.params["const"], "slope": r.params["logH"], "lo": ci.loc["logH", 0],
                 "hi": ci.loc["logH", 1], "cov_coef": r.params[cov] if cov else np.nan,
                 "cov_lo": ci.loc[cov, 0] if cov else np.nan, "cov_hi": ci.loc[cov, 1] if cov else np.nan,
                 "deviance_explained": 1 - r.deviance / r.null_deviance, "n": int(r.nobs)})
    s = f"slope {r.params['logH']:.2f} ({ci.loc['logH', 0]:.2f} to {ci.loc['logH', 1]:.2f})"
    for k in r.params.index[2:]:
        s += f", {k} {r.params[k]:.2f} ({ci.loc[k, 0]:.2f} to {ci.loc[k, 1]:.2f})"
    dev = 1 - r.deviance / r.null_deviance
    print(f"  {label:44s} {s}; deviance explained {100 * dev:.0f}%")


hazards = [("E", "network"), ("E_out", "network, outdoors"), ("E_iss", "satellite total lightning"),
           ("E_cg", "satellite CG (Z middle)"), ("E_cg_low", "satellite CG (Z high, fewer CG)"),
           ("E_cg_high", "satellite CG (Z low, more CG)"), ("E_out_cg", "satellite CG, outdoors"),
           ("E_day", "network, whole-day time outside"), ("E_open", "network, outdoor work"),
           ("E_open_cg", "satellite CG, outdoor work"), ("E_open_act", "network, outdoor work (activities only)"),
           ("E_open_eq", "network, outdoor work (equal allocation)"),
           ("E_open_act_cg", "satellite CG, outdoor work (activities only)")]
for name, df, smp in ((f"All {N} States/UTs", big, "all"), (f"Only the {len(pos)} with deaths", pos, "positive"),
                      (f"Best-detected {len(best)} states", best, "best")):
    print(f"\n{name}: negative binomial (dispersion by maximum likelihood), deaths 2022-2024")
    for h, lab in hazards:
        a = nb_alpha(df, h)
        show(f"{lab}", model(df, h, family=a), smp, h)
print(f"\nWith one vulnerability covariate (negative binomial, all {N} States/UTs; farm share also for the {len(pos)} "
      "with deaths):")
for h, lab in hazards[:2] + hazards[3:4] + hazards[6:]:
    for cov in ("census_agri", "census_literacy", "mpce_log", "tus_katcha"):
        a = nb_alpha(big, h, cov)
        show(f"{lab} + {cov}", model(big, h, cov, family=a), "all", h, cov)
    a = nb_alpha(pos, h, "census_agri")
    show(f"{lab} + census_agri (with deaths only)", model(pos, h, "census_agri", family=a), "positive", h, "census_agri")


def nb_general(df, cols):
    """Negative binomial with several regressors: dispersion and AIC from the maximum-likelihood fit."""
    X = sm.add_constant(df[cols])
    ml = sm.NegativeBinomial(df.n3, X, offset=df.offset).fit(disp=0)
    r = sm.GLM(df.n3, X, family=sm.families.NegativeBinomial(alpha=ml.params["alpha"]), offset=df.offset).fit()
    return r, ml.aic


print(f"\nWhat the timing of time outside adds (negative binomial, all {N} States/UTs; AIC from the "
      "maximum-likelihood fit, lower is better):")
b2 = big.assign(logE=np.log(big.E), logO_day=np.log(big.O_day), logO_timing=np.log(big.O_timing),
                logE_day=np.log(big.E_day), logE_out=np.log(big.E_out), logE_open=np.log(big.E_open),
                logO_open=np.log(big.E_open / big.E), logE_open_cg=np.log(big.E_open_cg))
for lab, cols in (("hazard alone", ["logE"]), ("whole-day weighting (one slope)", ["logE_day"]),
                  ("timing weighting (one slope)", ["logE_out"]), ("outdoor work (one slope)", ["logE_open"]),
                  ("hazard + whole-day share outside", ["logE", "logO_day"]),
                  ("hazard + timing share outside", ["logE", "logO_timing"]),
                  ("hazard + outdoor-work share", ["logE", "logO_open"]),
                  ("hazard + farm share", ["logE", "census_agri"]),
                  ("hazard + timing share outside + farm share", ["logE", "logO_timing", "census_agri"]),
                  ("hazard + farm share of the 2024 survey", ["logE", "tus_agri"]),
                  ("outdoor work + farm share of the 2024 survey", ["logE_open", "tus_agri"]),
                  ("outdoor work + farm share (Census 2011)", ["logE_open", "census_agri"]),
                  ("satellite CG, outdoor work (one slope)", ["logE_open_cg"]),
                  ("satellite CG, outdoor work + farm share", ["logE_open_cg", "census_agri"])):
    r, aic = nb_general(b2, cols)
    ci = r.conf_int()
    txt = ", ".join(f"{c} {r.params[c]:.2f} ({ci.loc[c, 0]:.2f} to {ci.loc[c, 1]:.2f})" for c in cols)
    print(f"  {lab:44s} {txt}; deviance explained {100 * (1 - r.deviance / r.null_deviance):.0f}%, AIC {aic:.1f}")
    FITS.append({"sample": "all", "hazard": "+".join(cols), "covariate": "", "label": "benchmark: " + lab,
                 "intercept": r.params["const"], "slope": r.params[cols[0]], "lo": ci.loc[cols[0], 0],
                 "hi": ci.loc[cols[0], 1], "cov_coef": r.params[cols[1]] if len(cols) > 1 else np.nan,
                 "cov_lo": ci.loc[cols[1], 0] if len(cols) > 1 else np.nan,
                 "cov_hi": ci.loc[cols[1], 1] if len(cols) > 1 else np.nan,
                 "deviance_explained": 1 - r.deviance / r.null_deviance, "n": int(r.nobs)})

pe, pc = big.E * big.pop_m, big.E_cg * big.pop_m
print(f"\nShare of the exposure that falls while people are outside, India ({N} States/UTs): network "
      f"{100 * (big.O_timing * pe).sum() / pe.sum():.1f}%, satellite CG {100 * (big.O_timing * pc).sum() / pc.sum():.1f}%, "
      f"network without the timing {100 * (big.O_day * pe).sum() / pe.sum():.1f}%, satellite CG without the timing "
      f"{100 * (big.O_day * pc).sum() / pc.sum():.1f}%")

# matched models for network and satellite exposure: one row per model and exposure source
def nb_full(df, cols):
    X = sm.add_constant(df[cols])
    ml = sm.NegativeBinomial(df.n3, X, offset=df.offset).fit(disp=0, maxiter=200)
    r = sm.GLM(df.n3, X, family=sm.families.NegativeBinomial(alpha=ml.params["alpha"]), offset=df.offset).fit()
    return r, ml.params["alpha"], ml.aic


_tus = pd.read_csv(DATA / "vulnerability/tus_outside.csv", dtype={"sex": str})
_tus = _tus[_tus.sex.isin(["1", "2"])]
tus_open_day = (_tus.out_open * _tus.weight).groupby(_tus.state).sum() / _tus.weight.groupby(_tus.state).sum()
b3 = big.assign(logO=np.log(big.O_timing), logW=np.log(big.E_open / big.E), logW_act=np.log(big.E_open_act / big.E),
                logW_eq=np.log(big.E_open_eq / big.E))
MODELS = [("Exposure", ["logH"]),
          ("Exposure + share of exposure outside", ["logH", "logO"]),
          ("Exposure while outside", ["logHO"]),
          ("Exposure + share in outdoor work", ["logH", "logW"]),
          ("Exposure during outdoor work", ["logHW"]),
          ("Exposure + farm share", ["logH", "census_agri"]),
          ("Exposure during outdoor work + farm share", ["logHW", "census_agri"]),
          ("Exposure + share in outdoor work + farm share", ["logH", "logW", "census_agri"]),
          ("Exposure during outdoor work, activities only", ["logHWa"]),
          ("  + farm share", ["logHWa", "census_agri"]),
          ("Exposure during outdoor work, equal allocation", ["logHWe"]),
          ("  + farm share", ["logHWe", "census_agri"])]
table = []
print(f"\nMatched negative binomial models, all {N} States/UTs (coefficient and 95% interval; dispersion alpha and AIC "
      "from the maximum-likelihood fit; deviance explained = 1 - D/D0 at that alpha, D0 for the intercept and offset only):")
for src, H in (("network", "E"), ("satellite CG", "E_cg")):
    d_ = b3.assign(logH=np.log(b3[H]))
    d_ = d_.assign(logHO=d_.logH + d_.logO, logHW=d_.logH + d_.logW, logHWa=d_.logH + d_.logW_act,
                   logHWe=d_.logH + d_.logW_eq)
    for lab, cols in MODELS:
        r, alpha, aic = nb_full(d_, cols)
        ci = r.conf_int()
        row = {"source": src, "model": lab.strip(), "alpha": alpha, "aic": aic,
               "deviance_explained": 1 - r.deviance / r.null_deviance}
        for c in cols:
            key = {"logH": "exposure", "logHO": "exposure", "logHW": "exposure", "logHWa": "exposure",
                   "logHWe": "exposure", "logO": "share", "logW": "share", "census_agri": "farm"}[c]
            row[key], row[key + "_lo"], row[key + "_hi"] = r.params[c], ci.loc[c, 0], ci.loc[c, 1]
        table.append(row)
        txt = ", ".join(f"{c} {r.params[c]:.2f} ({ci.loc[c, 0]:.2f} to {ci.loc[c, 1]:.2f})" for c in cols)
        print(f"  {src:12s} {lab:48s} {txt}; alpha {alpha:.3f}, AIC {aic:.1f}, deviance explained "
              f"{100 * row['deviance_explained']:.0f}%")
pd.DataFrame(table).to_csv(RESULTS / "deaths_models_table.csv", index=False, float_format="%.6f")
print(f"  across the {N} States/UTs, the share of exposure in outdoor work and the farm share (Census 2011): Pearson r "
      f"{np.corrcoef(b3.E_open / b3.E, b3.census_agri)[0, 1]:.2f} (log share {np.corrcoef(b3.logW, b3.census_agri)[0, 1]:.2f}); "
      f"time in outdoor work (whole day) and the farm share r "
      f"{np.corrcoef([tus_open_day.get(s_, np.nan) for s_ in b3.index], b3.census_agri)[0, 1]:.2f}")
for src in ("network", "satellite CG"):
    tt = pd.DataFrame(table)
    tt = tt[(tt.source == src) & tt.model.isin(["Exposure + farm share", "Exposure during outdoor work + farm share"])]
    print(f"  {src}: farm-share rate ratio per 10 percentage points " +
          ", ".join(f"{m}: {np.exp(0.1 * r.farm):.2f} ({np.exp(0.1 * r.farm_lo):.2f}-{np.exp(0.1 * r.farm_hi):.2f})"
                    for m, r in zip(tt.model, tt.itertuples())))


# within states: Poisson with state fixed effects, 2022-2024, exposure from each year's NRSC data
pop = pd.read_csv(POP / "pop_state_cell.csv")
deaths = pd.read_csv(NCRB / "lightning_deaths.csv")
tus = pd.read_csv(DATA / "vulnerability/tus_outside.csv", dtype={"sex": str})
# detection of each 0.25 degree cell, from ISS LIS 2022-2023 (sr_detection.py): network flashes per expected CG
# flash, both summed over 3 x 3 cells where the expected density is at least 0.5 per km2 per year, else the
# region's value; kept between 0.05 and 1. Three cases for the CG fraction (middle, low and high Z).
import xarray as xr
from scipy.ndimage import uniform_filter
det = xr.open_dataset(DATA / "vulnerability/detection_025.nc")
reg_de = pd.read_csv(RESULTS / "detection_region.csv", index_col=0).detection
from common import region_grid
rgrid = region_grid()
box = lambda x: uniform_filter(np.nan_to_num(x), size=3, mode="constant")
DE = {}
for case, var in (("mid", "iss_cg_2022_2023"), ("low", "iss_cg_high"), ("high", "iss_cg_low")):
    nr, cg = box(det.nrsc_2022_2023.values), box(det[var].values)
    fallback = np.vectorize(lambda r: reg_de.get(r, reg_de["India"]))(rgrid)
    DE[case] = np.clip(np.where(cg >= 0.5, nr / np.maximum(cg, 1e-9), fallback), 0.05, 1.0)
rows = []
for yr in (2022, 2023, 2024, 2025):
    f = np.nan_to_num(nrsc_clim("flashes", years=[yr]).values / A)          # (12, 24, lat, lon)
    tot = f.sum((0, 1))
    g20 = np.nan_to_num(nrsc_clim("flashes_ge20kA", years=[yr]).values / A).sum(0)
    for s, g in pop.groupby("state"):
        w = g.population.values
        if w.sum() <= 0:
            continue
        e = (tot[g.j, g.i] * w).sum() / w.sum()
        ec = {c: (tot[g.j, g.i] / DE[c][g.j, g.i] * w).sum() / w.sum() for c in DE}
        rows.append((yr, s, e, ec["mid"], ec["low"], ec["high"], (g20[g.j, g.i] * w).sum() / max((tot[g.j, g.i] * w).sum(), 1e-12)))
ex = pd.DataFrame(rows, columns=["year", "state", "E", "Ec", "Ec_low", "Ec_high", "s20"])
m = ex.merge(deaths[["year", "state", "total"]], on=["year", "state"])            # 2022-2024: no 2025 deaths yet
keep = m.groupby("state").total.sum()                                            # deaths 2022-2024, as in the models

# the models across states pair deaths for 2022-2024 with exposure averaged over 2022-2025; refit them with
# exposure for 2022-2024 only (outdoor work: the 2022-2025 share of exposure in outdoor work)
e4, e3 = ex.groupby("state").E.mean(), ex[ex.year <= 2024].groupby("state").E.mean()
bb = big.assign(E_2224=e3.reindex(big.index), E_2225=e4.reindex(big.index))
bb["E_open_2224"] = bb.E_2224 * bb.E_open / bb.E
print(f"\nExposure for 2022-2024 instead of 2022-2025 (negative binomial, all {N} States/UTs, deaths 2022-2024); "
      f"annual means over 2022-2025 against the exposure used above: largest difference "
      f"{100 * (bb.E_2225 / bb.E - 1).abs().max():.2f}%")
for h, lab in (("E", "network, 2022-2025"), ("E_2224", "network, 2022-2024"),
               ("E_open", "network, outdoor work, 2022-2025"), ("E_open_2224", "network, outdoor work, 2022-2024")):
    show(lab, model(bb, h, family=nb_alpha(bb, h)), "period", h)
from itertools import product
from scipy import stats as st_
GRID_B = np.round(np.arange(-1.0, 2.0001, 0.005), 3)


def within_fit(dd, h, year_fe=False, extra=()):
    """Poisson regression of annual deaths on log exposure with state (and year) fixed effects. The slope's 95%
    interval from standard errors clustered by state (CR1, G/(G-1); t with G-1 degrees of freedom), from
    heteroskedasticity-robust errors (HC0, as before), and from a wild cluster score bootstrap (restricted
    scores, Rademacher weights over all 2^G sign patterns, or 99,999 random ones for G > 16; Kline and Santos
    2012) inverted over a grid of slopes, with its p-value for a slope of 0."""
    dd = dd.reset_index(drop=True)
    y, x = dd.total.values.astype(float), np.log(dd[h].values)
    Z = pd.get_dummies(dd.state, dtype=float)
    if year_fe:
        Z = Z.join(pd.get_dummies(dd.year, prefix="y", drop_first=True, dtype=float))
    for c in extra:
        Z[c] = dd[c].values
    Z = Z.values
    grp = pd.factorize(dd.state)[0]
    G = grp.max() + 1

    def xt_(mu):                                      # x after the mu-weighted projection on the other terms
        w = np.sqrt(mu)
        return x - Z @ np.linalg.lstsq(Z * w[:, None], x * w, rcond=None)[0]

    r = sm.GLM(y, np.column_stack([Z, x]), family=sm.families.Poisson()).fit()
    b, mu = r.params[-1], r.mu
    xt = xt_(mu)
    sg = np.bincount(grp, weights=xt * (y - mu), minlength=G)
    se = np.sqrt(G / (G - 1) * (sg ** 2).sum()) / (mu * xt ** 2).sum()
    tq = st_.t.ppf(0.975, G - 1)
    hc = sm.GLM(y, np.column_stack([Z, x]), family=sm.families.Poisson()).fit(cov_type="HC0").conf_int()[-1]
    W = (np.array(list(product([-1.0, 1.0], repeat=G))) if G <= 16 else
         np.random.default_rng(1).choice([-1.0, 1.0], size=(99_999, G)))

    def pval(b0):
        mu0 = sm.GLM(y, Z, family=sm.families.Poisson(), offset=b0 * x).fit().mu
        s0 = np.bincount(grp, weights=xt_(mu0) * (y - mu0), minlength=G)
        den = np.sqrt((s0 ** 2).sum())
        return np.mean(np.abs(W @ s0) / den >= abs(s0.sum()) / den - 1e-12)

    pv = np.array([pval(b0) for b0 in GRID_B])
    acc = GRID_B[pv > 0.05]
    gaps = np.any(np.diff(acc) > 0.0051) if acc.size else True
    return {"slope": b, "cr_lo": b - tq * se, "cr_hi": b + tq * se, "hc_lo": hc[0], "hc_hi": hc[1],
            "bs_lo": acc.min() if acc.size else np.nan, "bs_hi": acc.max() if acc.size else np.nan,
            "p_boot": pval(0.0), "p_cr": 2 * st_.t.sf(abs(b / se), G - 1), "G": G, "n": len(dd),
            "pearson_disp": r.pearson_chi2 / r.df_resid, "gaps": gaps}


def show_within(lab, w_, sample, hazard, cov=""):
    print(f"  {lab:60s} slope {w_['slope']:.2f}; 95% interval: wild cluster bootstrap {w_['bs_lo']:.2f} to "
          f"{w_['bs_hi']:.2f} (p {w_['p_boot']:.4f}){' (not contiguous)' if w_['gaps'] else ''}, clustered by state "
          f"{w_['cr_lo']:.2f} to {w_['cr_hi']:.2f} (p {w_['p_cr']:.4f}), HC0 {w_['hc_lo']:.2f} to {w_['hc_hi']:.2f}; "
          f"{w_['G']} states, {w_['n']} state-years")
    FITS.append({"sample": sample, "hazard": hazard, "covariate": cov, "label": lab, "intercept": np.nan,
                 "slope": w_["slope"], "lo": w_["bs_lo"], "hi": w_["bs_hi"], "n": w_["n"], "cr_lo": w_["cr_lo"],
                 "cr_hi": w_["cr_hi"], "p_boot": w_["p_boot"], "p_cr": w_["p_cr"]})


print("\nWithin states (Poisson, state fixed effects, deaths 2022-2024):")
for lab, sel, smp in (("states with at least 60 deaths in 2022-2024", keep[keep >= 60].index, "within60"),
                      ("all states with at least 10 deaths in 2022-2024", keep[keep >= 10].index, "within10")):
    mm = m[m.state.isin(sel) & (m.year >= 2022)].copy()
    for yfe in (False, True):
        w_ = within_fit(mm, "E", year_fe=yfe)
        show_within(f"within {mm.state.nunique()} {lab}" + (", with year effects" if yfe else ""), w_,
                    smp + ("_year" if yfe else ""), "E")
    print(f"    Pearson dispersion {w_['pearson_disp']:.2f}")

# does effective detection stay the same from year to year within states?
mm = m[m.state.isin(keep[keep >= 60].index) & (m.year >= 2022)].copy()
mm["lE"], mm["lD"] = np.log(mm.E), np.log(mm.E / mm.Ec)
for a_, b_ in (("lE", "dE"), ("lD", "dD"), ("s20", "dS")):
    mm[b_] = mm[a_] - mm.groupby("state")[a_].transform("mean")
print(f"\nEffective detection within the {mm.state.nunique()} states with at least 60 deaths, 2022-2024 (annual exposure "
      "against exposure corrected cell by cell with the ISS LIS detection):")
print(f"  year-to-year spread of log exposure {100 * mm.dE.std():.1f}%, of effective detection {100 * mm.dD.std():.1f}%; "
      f"correlation between them {np.corrcoef(mm.dE, mm.dD)[0, 1]:+.2f}; effective detection by state "
      f"{mm.groupby('state').apply(lambda d: (d.E / d.Ec).mean(), include_groups=False).min():.2f}-"
      f"{mm.groupby('state').apply(lambda d: (d.E / d.Ec).mean(), include_groups=False).max():.2f}")
print(f"  share of flashes of 20 kA or more where people live: year-to-year spread {100 * mm.dS.std():.2f} percentage "
      f"points (mean {100 * mm.s20.mean():.1f}%)")
best15 = mm[mm.state.isin(best.index)].state.unique()
tests = [("network exposure (as in the main text)", mm, "E", []),
         ("corrected exposure, middle CG fraction", mm, "Ec", []),
         ("corrected exposure, detection at the low end", mm, "Ec_low", []),
         ("corrected exposure, detection at the high end", mm, "Ec_high", []),
         ("network exposure + share of 20 kA or more", mm, "E", ["s20"]),
         (f"network exposure, {len(best15)} best-detected states", mm[mm.state.isin(best15)], "E", []),
         ("network exposure, without the Northwest and Northeast", mm[~mm.state.isin(REGIONS["Northwest"] + REGIONS["Northeast"])], "E", [])]
for lab, dd, h, extra in tests:
    show_within(lab, within_fit(dd, h, extra=extra), "within_check", h, "+".join(extra))
for lab, dd, h in (("network exposure, with year effects", mm, "E"),
                   ("corrected exposure, middle CG fraction, with year effects", mm, "Ec")):
    show_within(lab, within_fit(dd, h, year_fe=True), "within_check_year", h)

# leave one state out: the best-detected sample and all States/UTs, network outdoor exposure
print("\nLeave one state out, network outdoor exposure (negative binomial), range of the slope:")
for name, df in ((f"all {N}", big), ("best-detected", best)):
    sl = [model(df.drop(s), "E_out", family=nb_alpha(df.drop(s), "E_out")).params["logH"] for s in df.index]
    print(f"  {name}: {min(sl):.2f} to {max(sl):.2f}")
pd.DataFrame(FITS).to_csv(RESULTS / "deaths_models.csv", index=False, float_format="%.4f")
print(f"\nBy state (all {N}): hazard measures and deaths per million")
print(big[["region", "M", "E", "E_out", "E_iss", "E_cg", "de", "records_per_flash", "census_agri",
           "tus_mpce"]].sort_values("M", ascending=False).round(2).to_string())

# do hazard and vulnerability coincide across states, and does the network's detection follow poverty?
from scipy import stats
print(f"\nState level ({len(pos)} States/UTs with deaths): Pearson r of log exposure with")
for e in ("E", "E_cg"):
    out = []
    for cv in ("census_agri", "census_literacy", "mpce_log"):
        r = stats.pearsonr(np.log(pos[e]), pos[cv])
        out.append(f"{cv} {r.statistic:+.2f} (p {r.pvalue:.2f})")
    print(f"  {e:5s}: " + ", ".join(out))
r, r2 = stats.spearmanr(pos.de, pos.mpce_log), stats.spearmanr(pos.de, pos.census_agri)
print(f"  network detection vs consumption rho {r.statistic:+.2f} (p {r.pvalue:.3f}); vs agricultural workers "
      f"rho {r2.statistic:+.2f} (p {r2.pvalue:.3f})")

# deaths per unit exposure relative to India under each exposure measure
ten = t[t.deaths * 3 >= 10].copy()
for e in ("E", "E_out", "E_cg", "E_out_cg"):
    nat = t.deaths.sum() / t.pop_m.sum() / ((t[e] * t.pop_m).sum() / t.pop_m.sum())
    ten["R_" + e] = ten.M / ten[e] / nat
ten[["region", "R_E", "R_E_out", "R_E_cg", "R_E_out_cg", "de", "census_agri", "tus_mpce"]].round(4).to_csv(
    RESULTS / "deaths_ratios_by_measure.csv")
print("\nDeaths per unit exposure relative to India, by exposure measure (states with at least 10 deaths):")
print(ten[["R_E", "R_E_out", "R_E_cg", "R_E_out_cg", "de"]].sort_values("R_E", ascending=False).round(2).to_string())
r = stats.spearmanr(ten.R_E, ten.R_E_out_cg)
print(f"rank correlation of the network ratio with the satellite outdoor ratio: {r.statistic:.2f}")
for cv in ("census_agri", "tus_mpce"):
    r = stats.spearmanr(ten.R_E_out_cg, ten[cv])
    print(f"satellite outdoor ratio vs {cv}: rho {r.statistic:+.2f} (p {r.pvalue:.3f})")
