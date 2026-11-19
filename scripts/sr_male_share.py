#!/usr/bin/env python3
"""Men's share of lightning deaths: recorded (NCRB 2022-2024) against the share
expected from time outside the home (Time Use Survey 2019 and NRSC timing,
sr_vulnerability.py), by State/UT and for India (npj Natural Hazards paper).

States with at least 30 deaths in 2022-2024. Tests: (1) do the recorded shares
differ between states by more than chance (chi-square test of one common
share)? (2) do they fit the state predictions within binomial noise
(chi-square)? (3) what correlation between predicted and recorded shares would
chance allow if the predictions were exact (5,000 binomial simulations)?
Output: results/male_share_states.csv; the log is results/log_sr_male_share.txt
"""
import numpy as np
import pandas as pd
from scipy import stats

from common import NCRB, REGIONS, RESULTS

rng = np.random.default_rng(1)
TO_REGION = {s: r for r, ss in REGIONS.items() for s in ss}
t = pd.read_csv(RESULTS / "vulnerability_state.csv", index_col=0)
boot = pd.read_csv(RESULTS / "tus_bootstrap.csv", index_col=0)
d = pd.read_csv(NCRB / "lightning_deaths.csv")
d = d[d.year.between(2022, 2024)].groupby("state")[["male", "total"]].sum()
k = d[d.total >= 30].join(t[["male_share_pred"]]).dropna()
n, m, p = k.total.values, k.male.values, k.male_share_pred.values
obs = m / n
k["observed"] = obs
k["obs_lo"] = [stats.binomtest(int(a), int(b)).proportion_ci().low for a, b in zip(m, n)]
k["obs_hi"] = [stats.binomtest(int(a), int(b)).proportion_ci().high for a, b in zip(m, n)]
k["region"] = [TO_REGION.get(s, "") for s in k.index]
N, M = int(d.total.sum()), int(d.male.sum())
ci = stats.binomtest(M, N).proportion_ci()
nat = boot.loc["share of deaths expected among men"]
k.loc["INDIA"] = {"male": M, "total": N, "male_share_pred": nat.estimate, "observed": M / N, "obs_lo": ci.low,
                  "obs_hi": ci.high, "region": "India"}
k.loc["INDIA", "pred_lo"], k.loc["INDIA", "pred_hi"] = nat.lo, nat.hi
k.round(4).to_csv(RESULTS / "male_share_states.csv")

print(f"{len(n)} states with at least 30 deaths in 2022-2024; deaths per state {n.min()}-{n.max()} "
      f"(median {np.median(n):.0f}); binomial noise of the median state's share "
      f"{100 * np.median(np.sqrt(obs * (1 - obs) / n)):.1f} points")
print(f"recorded male share {100 * obs.min():.0f}-{100 * obs.max():.0f}%, expected {100 * p.min():.0f}-{100 * p.max():.0f}%")
chi, pv, dof, _ = stats.chi2_contingency(np.c_[m, n - m])
print(f"one common share for all states: chi2 = {chi:.1f}, dof = {dof}, p = {pv:.3f}")
x2 = ((m - n * p) ** 2 / (n * p * (1 - p))).sum()
print(f"state predictions: chi2 = {x2:.1f}, dof = {len(n)}, p = {stats.chi2.sf(x2, len(n)):.1e}")
r = stats.pearsonr(p, obs).statistic
sims = np.array([stats.pearsonr(p, rng.binomial(n, p) / n).statistic for _ in range(5000)])
print(f"correlation of expected and recorded shares {r:.2f}; if the predictions were exact: median "
      f"{np.median(sims):.2f}, 90% of simulations {np.percentile(sims, 5):.2f} to {np.percentile(sims, 95):.2f}")
print(f"India: recorded {100 * M / N:.1f}% of {N} deaths (95% {100 * ci.low:.1f}-{100 * ci.high:.1f}); expected "
      f"{100 * nat.estimate:.1f}% ({100 * nat.lo:.1f}-{100 * nat.hi:.1f}); men's share of people aged 6 and above "
      f"{100 * boot.loc['share of men among people aged 6 and above', 'estimate']:.1f}%")
