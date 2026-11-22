#!/usr/bin/env python3
"""Sampling uncertainty of the Time Use Survey results (npj Natural Hazards
paper): bootstrap over first-stage units (villages and urban blocks,
resampled with replacement within each state), 200 replicates.
For each replicate: share of people outside the home (school excluded) by
state, quarter and half hour, and by sex; outdoor exposure by state (NRSC
2022-2025 timing); India's share of exposure that falls outdoors; the share
of deaths expected among men; and the negative binomial slope of deaths
against outdoor exposure across the 25 States/UTs, alone and with the share
of agricultural workers.
Output: printed (results/log_tus_bootstrap.txt) and results/tus_bootstrap.csv
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import DATA, POP, RESULTS, india_area, nrsc_clim

rng = np.random.default_rng(29092026)
_src = open(__file__.replace("tus_bootstrap.py", "tus_outdoor.py")).read()
_cut = "# every activity record, expanded to person and half-hour slot"
assert _cut in _src, "tus_outdoor.py: the comment before the activity records has changed"
exec(_src.split(_cut)[0])        # people, weights, states and the activity records (rec)
# person x half-hour matrix of being outside (school excluded)
key_index = pd.Series(np.arange(len(p)), index=p.key)
M = np.zeros((len(p), 48), dtype=bool)
ch = rec[rec["where"].isin(["2", "3"]) & ~rec.act.str[:2].isin(["61", "62", "63"])]
t0 = ch.t0.str[:2].astype(int) * 60 + ch.t0.str[3:5].astype(int)
t1 = ch.t1.str[:2].astype(int) * 60 + ch.t1.str[3:5].astype(int)
n = np.maximum(((t1 - t0) % 1440) // 30, 1).values
idx, s0 = key_index.reindex(ch.key).values.astype(int), t0.values // 30
for j in range(n.max()):
    sel = n > j
    M[idx[sel], (s0[sel] + j) % 48] = True
print(f"{M.sum():,} person half hours outside (school excluded)")
print("survey states not on the population grid:", sorted(set(p.state.dropna()) - set(pd.read_csv(POP / "pop_state_cell.csv").state)))

A = india_area()
popc = pd.read_csv(POP / "pop_state_cell.csv")
f = np.nan_to_num(nrsc_clim("flashes").values / A)
rho = {s: np.einsum("mhc,c->mh", f[:, :, g.j.values, g.i.values], g.population.values) / g.population.sum()
       for s, g in popc.groupby("state")}
states = sorted(rho)
s1 = ((np.arange(24) * 60 + 330) % 1440) // 30
s2 = (s1 + 1) % 48
SR = np.arange(12) // 3
t = pd.read_csv(RESULTS / "vulnerability_state.csv", index_col=0)
big = t[(t.pop_m >= 1) & (t.deaths > 0)].copy()
n3, off = np.rint(big.deaths * 3).values, np.log(3 * big.pop_m.values * 1e6)
pop_state = popc.groupby("state").population.sum()

p = p.reset_index(drop=True)
st_code = pd.Categorical(p.state, categories=states)
stc = st_code.codes.astype(np.int64)                               # int8 codes would overflow below
sex = p.sex.values
sr = p.subround.astype(int).values - 1
fsu_codes, fsu_idx = np.unique(p.fsu.values, return_inverse=True)
fsu_state = pd.Series(p.state.values).groupby(fsu_idx).first().values
by_state = {s: np.where(fsu_state == s)[0] for s in np.unique(fsu_state)}


def estimate(w):
    """Outdoor exposure by state (all people, men, women) for person weights w."""
    res = {}
    known = stc >= 0                                     # states on the population grid
    for lab, sel in (("all", known), ("1", known & (sex == "1")), ("2", known & (sex == "2"))):
        g = stc[sel] * 4 + sr[sel]
        ww = w[sel]
        den = np.bincount(g, ww, minlength=len(states) * 4)
        num = np.stack([np.bincount(g, ww * M[sel, k], minlength=len(states) * 4) for k in range(48)], 1)
        share = (num / np.where(den > 0, den, np.nan)[:, None]).reshape(len(states), 4, 48)
        india = np.nansum(num.reshape(len(states), 4, 48), 0) / den.reshape(len(states), 4).sum(0)[:, None]
        e = {}
        for k, s in enumerate(states):
            x = share[k] if np.isfinite(share[k]).all() else india
            e[s] = ((0.5 * (x[:, s1] + x[:, s2]))[SR] * rho[s]).sum()
        res[lab] = pd.Series(e)
        res["w_" + lab] = pd.Series(np.bincount(stc[sel], ww, minlength=len(states)), index=states)
    return res


def slopes(e_out):
    x = np.log(e_out.reindex(big.index).values)
    out = []
    for X in (sm.add_constant(x), sm.add_constant(np.column_stack([x, big.census_agri.values]))):
        a = sm.NegativeBinomial(n3, X, offset=off).fit(disp=0).params[-1]
        out.append(sm.GLM(n3, X, family=sm.families.NegativeBinomial(alpha=a), offset=off).fit().params[1])
    return out


def summary(res):
    E = pd.Series({s: rho[s].sum() for s in states})
    frac = (res["all"] * pop_state).sum() / (E * pop_state).sum()
    male = (res["w_1"] * res["1"]).sum() / ((res["w_1"] * res["1"]).sum() + (res["w_2"] * res["2"]).sum())
    return [frac, male] + slopes(res["all"])


w0 = p.w.values
base = summary(estimate(w0))
reps = []
for b in range(200):
    mult = np.zeros(len(fsu_codes))
    for s, ids in by_state.items():
        np.add.at(mult, rng.choice(ids, len(ids)), 1)
    reps.append(summary(estimate(w0 * mult[fsu_idx])))
reps = np.array(reps)
lab = ["share of exposure outdoors", "share of deaths expected among men", "slope, outdoor exposure",
       "slope, outdoor exposure with agricultural workers"]
out = []
for k, name in enumerate(lab):
    lo, hi = np.percentile(reps[:, k], [2.5, 97.5])
    out.append({"quantity": name, "estimate": base[k], "lo": lo, "hi": hi})
    print(f"{name}: {base[k]:.3f} (bootstrap 95%: {lo:.3f} to {hi:.3f})")
male_pop = float(np.bincount(stc[stc >= 0], w0[stc >= 0] * (sex[stc >= 0] == "1")).sum() / w0[stc >= 0].sum())
out.append({"quantity": "share of men among people aged 6 and above", "estimate": male_pop})
pd.DataFrame(out).to_csv(RESULTS / "tus_bootstrap.csv", index=False, float_format="%.4f")
