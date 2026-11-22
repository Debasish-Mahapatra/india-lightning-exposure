#!/usr/bin/env python3
"""Time outside the home by state, sex, quarter and half hour, and household
covariates by state, from India's Time Use Survey 2024 (NSO, January to
December 2024; unit-level text files TUS106PER.TXT and TUS106HH.TXT of
MoSPI's microdata catalogue, copy from data.opencity.in,
data/vulnerability/tus2024/raw; layout and codes beside them).

The survey records every activity of each member aged 6 and above in 30-minute
slots over 24 hours (04:00 to 04:00), with where it was done: 1 within the
premises of the household's dwelling, 2 outside them in a fixed location,
3 outside them in a non-fixed location. A person counts as outside in a slot
if any activity in that slot was done outside the premises (2 or 3).
Two versions: all activities, and excluding formal education and study
(ICATUS 61x-63x), since schools are substantial buildings; travel to school
(64x) stays. A third version counts only outdoor work, every activity of
these kinds done outside the premises (2 or 3):
  A  for the market in household enterprises: growing crops, raising animals,
     forestry and logging, fishing, aquaculture, mining and quarrying,
     construction (ICATUS 121-126, 128);
  B  for own final use: agriculture, forestry, fishing, mining and quarrying
     (211-218), construction (230), gathering firewood and fetching water
     (241, 242);
  C  employment in corporations, government and non-profit institutions
     (110) and activities ancillary to employment and breaks (141, 142) of
     people whose usual principal activity is in agriculture, forestry,
     fishing, mining or construction (NIC 2008 01-03, 05-09, 41-43); people
     without a usual industry (not working) have none.
A half hour counts if any qualifying activity in it was done outside; the
equal-allocation variant (out_open_eq) instead gives it the share of the
activities recorded in it that qualify, as in the survey's own estimates.
out_open_act counts A and B only (no industry rule), out_open_ind C only.
The person file has one row for every activity, with the person's details
repeated; members without a time diary (children under 6 and a few others)
have one row without an activity. Weights: final weight = MULT/100 of the
person record, the weight for estimates from the four sub-rounds together.
Sub-rounds: 1 January-March, 2 April-June, 3 July-September, 4 October-December.

Output: data/vulnerability/tus_outside.csv (state, sex, subround, slot,
         out_all, out_noschool, out_open, out_open_act, out_open_ind,
         out_open_eq, weight, persons),
        results/tus_outdoor_work_codes.csv (share of outdoor-work time by
         activity code and sex),
        data/vulnerability/tus_state.csv (covariates by NCRB state name)
"""
import numpy as np
import pandas as pd

from common import DATA, RESULTS

T = DATA / "vulnerability/tus2024/raw"
OUT = DATA / "vulnerability"
# state codes of the survey (first two digits of the NSS region) -> NCRB state names
NSS = {"01": "JAMMU & KASHMIR", "02": "HIMACHAL PRADESH", "03": "PUNJAB", "04": "CHANDIGARH", "05": "UTTARAKHAND",
       "06": "HARYANA", "07": "DELHI (UT)", "08": "RAJASTHAN", "09": "UTTAR PRADESH", "10": "BIHAR", "11": "SIKKIM",
       "12": "ARUNACHAL PRADESH", "13": "NAGALAND", "14": "MANIPUR", "15": "MIZORAM", "16": "TRIPURA",
       "17": "MEGHALAYA", "18": "ASSAM", "19": "WEST BENGAL", "20": "JHARKHAND", "21": "ODISHA",
       "22": "CHHATTISGARH", "23": "MADHYA PRADESH", "24": "GUJARAT", "25": "D & N HAVELI AND DAMAN & DIU",
       "27": "MAHARASHTRA", "28": "ANDHRA PRADESH", "29": "KARNATAKA",
       "30": "GOA", "31": "LAKSHADWEEP", "32": "KERALA", "33": "TAMIL NADU", "34": "PUDUCHERRY",
       "35": "A & N ISLANDS", "36": "TELANGANA", "37": "LADAKH"}
# byte positions (first, last) in the text files, from Data_Layout_TUS_2024.xlsx
PER = {"fsu": (4, 8), "sector": (16, 16), "state": (17, 18), "district": (20, 21), "subround": (28, 28),
       "hh": (33, 34), "mem": (35, 37), "sex": (39, 39), "age": (40, 42), "status": (46, 47), "nic": (48, 49),
       "t0": (72, 76), "t1": (77, 81), "act": (85, 87), "where": (88, 88), "mult": (95, 104)}
HH = {"fsu": (4, 8), "state": (17, 18), "district": (20, 21), "hh": (33, 34), "size": (41, 43), "mce": (98, 107),
      "structure": (114, 114), "mult": (123, 132)}


def read(name, spec):
    s = pd.read_csv(T / name, header=None, names=["x"], sep="\x01", dtype=str, quoting=3)["x"]
    d = pd.DataFrame({k: s.str.slice(a - 1, b).str.strip() for k, (a, b) in spec.items()})
    assert d.state.isin(list(NSS)).all(), sorted(set(d.state) - set(NSS))
    d["state"] = d.state.map(NSS)
    return d


# every activity record, with the details of its person; members without a diary have an empty activity
rec = read("TUS106PER.TXT", PER)
rec["key"] = rec.fsu + rec.hh + rec.mem
m = rec.drop_duplicates("key")[["key", "state", "district", "sector", "age", "sex", "status", "nic", "mult"]].copy()
rec = rec[rec.act != ""].reset_index(drop=True)
# persons with time-use records: weights and design
p = rec.drop_duplicates("key")[["fsu", "hh", "mem", "state", "district", "sector", "subround", "age", "mult", "key",
                                "sex", "status", "nic"]].reset_index(drop=True)
p["w"] = p.mult.astype(float) / 100
rec = rec[["key", "t0", "t1", "act", "where"]]
print(f"{len(p):,} persons with time use among {len(m):,} household members, youngest {p.age.astype(int).min()}; "
      f"{p.w.sum() / 1e6:,.0f} million people aged 6 and above")

OPEN_NIC = [f"{k:02d}" for k in (1, 2, 3, 5, 6, 7, 8, 9, 41, 42, 43)]
CODES_A = ["121", "122", "123", "124", "125", "126", "128"]
worker = p.status.isin(["11", "12", "21", "31", "41", "51"])
print(f"usual principal status a worker: {100 * p.w[worker].sum() / p.w.sum():.1f}% of people aged 6 and above; "
      f"workers without an industry code: {100 * p.w[worker & (p.nic.fillna('').str.strip() == '')].sum() / p.w[worker].sum():.2f}%; "
      f"in agriculture, forestry, fishing, mining or construction: "
      f"{100 * p.w[p.nic.isin(OPEN_NIC)].sum() / p.w.sum():.1f}% of people")
kid = pd.Series(np.arange(len(p)), index=p.key)
open_workers = p.nic.isin(OPEN_NIC).values
# every activity record, expanded to person and half-hour slot
cols = {c: [] for c in ("k", "slot", "out", "school", "A", "B", "C", "act")}
durations, where_codes, n_bad = [], [], 0
for first in range(0, len(rec), 2_000_000):
    ch = rec.iloc[first:first + 2_000_000]
    bad = ~(ch.t0.str.match(r"^\d\d:\d\d") & ch.t1.str.match(r"^\d\d:\d\d")).fillna(False).astype(bool)
    n_bad += int(bad.sum())
    ch = ch[~bad]
    where = ch["where"]
    act = ch.act
    where_codes.append(pd.crosstab(act.str[:2], where))
    t0 = ch.t0.str[:2].astype(int) * 60 + ch.t0.str[3:5].astype(int)
    t1 = ch.t1.str[:2].astype(int) * 60 + ch.t1.str[3:5].astype(int)
    dur = (t1 - t0) % 1440
    out = where.isin(["2", "3"]).values
    durations.append(dur[out].value_counts())
    kk = kid.reindex(ch.key).values
    ok = ~np.isnan(kk)
    kk = kk[ok].astype(np.int64)
    a, b = act.isin(CODES_A).values[ok], act.str[:2].isin(["21", "23", "24"]).values[ok]
    c = act.str[:2].isin(["11", "14"]).values[ok] & open_workers[kk]
    school = act.str[:2].isin(["61", "62", "63"]).values[ok]
    slot0, n = (t0 // 30).values[ok], np.maximum(dur // 30, 1).values[ok]
    acts = pd.to_numeric(act, errors="coerce").fillna(0).astype(np.int16).values[ok]
    for j in range(n.max()):                                  # records longer than one slot
        s_ = n > j
        for name, v in (("k", kk[s_]), ("slot", (slot0[s_] + j) % 48), ("out", out[ok][s_]),
                        ("school", school[s_]), ("A", a[s_]), ("B", b[s_]), ("C", c[s_]), ("act", acts[s_])):
            cols[name].append(v)
f = pd.DataFrame({c: np.concatenate(v) for c, v in cols.items()})
print(f"{n_bad} activity records without valid times left out")
dur = pd.concat(durations).groupby(level=0).sum()
print("durations of outside records (minutes):", dur.sort_values(ascending=False).head(4).to_dict())
wc = pd.concat(where_codes).groupby(level=0).sum()
print("location code 3 (share of records, the activity divisions with the most):",
      f"{100 * wc['3'].sum() / wc.values.sum():.2f}%;",
      ", ".join(f"{i}: {100 * v / wc['3'].sum():.0f}%" for i, v in wc['3'].sort_values(ascending=False).head(4).items()))
f["open"] = f.out & (f.A | f.B | f.C)
f["open_act"] = f.out & (f.A | f.B)
f["open_ind"] = f.out & f.C
g_ = f.groupby(["k", "slot"])
ps = pd.DataFrame({"n": g_.size(), "out_all": g_.out.any(), "out_noschool": (f.out & ~f.school).groupby([f.k, f.slot]).any(),
                   "out_open": g_.open.any(), "out_open_act": g_.open_act.any(), "out_open_ind": g_.open_ind.any(),
                   "n_open": g_.open.sum()}).reset_index()
ps["out_open_eq"] = ps.n_open / ps.n
print(f"{ps.out_all.sum():,} person half hours outside, {ps.out_noschool.sum():,} excluding school, "
      f"{ps.out_open.sum():,} in outdoor work ({ps.out_open_act.sum():,} from activities A and B, "
      f"{ps.out_open_ind.sum():,} from employment by usual industry C, {(ps.out_open_act & ps.out_open_ind).sum():,} both); "
      f"{100 * (ps.n > 1).mean():.1f}% of person half hours have more than one activity")
ps["key"] = p.key.values[ps.k]
# composition of outdoor-work time by activity code and sex: each qualifying record's slot, weighted, shared
# equally among the qualifying records of the slot
q = f[f.open].merge(ps[["k", "slot", "n_open"]], on=["k", "slot"])
q = q.assign(w=p.w.values[q.k] / q.n_open, sex=p.sex.values[q.k])
q["branch"] = np.where(q.A, "A", np.where(q.B, "B", "C"))
comp = q.groupby(["branch", "act", "sex"]).w.sum().unstack("sex").fillna(0)
comp = comp[["1", "2"]].rename(columns={"1": "men", "2": "women"})
comp["all"] = comp.men + comp.women
comp = comp / comp.sum()
comp.round(5).to_csv(RESULTS / "tus_outdoor_work_codes.csv")
print("share of outdoor-work time (%) by branch:", (100 * comp.groupby(level=0).sum()).round(1).to_dict("index"))
print("largest codes (% of all outdoor-work time):",
      ", ".join(f"{a} {100 * v:.1f}" for (b_, a), v in comp["all"].sort_values(ascending=False).head(8).items()))

# weighted share outside by state, sex, sub-round and slot
g = ["state", "sex", "subround"]
tot = p.groupby(g).agg(weight=("w", "sum"), persons=("w", "size")).reset_index()
OUT_COLS = ["out_all", "out_noschool", "out_open", "out_open_act", "out_open_ind", "out_open_eq"]
o = ps.merge(p[["key", "w"] + g], on="key")
res = o.assign(**{c: o[c] * o.w for c in OUT_COLS}).groupby(g + ["slot"])[OUT_COLS].sum().reset_index()
full = tot.merge(pd.DataFrame({"slot": range(48)}), how="cross")
res = full.merge(res, on=g + ["slot"], how="left").fillna({c: 0 for c in OUT_COLS})
for c in OUT_COLS:
    res[c] = res[c] / res.weight
res.to_csv(OUT / "tus_outside.csv", index=False, float_format="%.5g")
nat = res.assign(a=res.out_noschool * res.weight).groupby(["sex", "slot"]).a.sum() / \
      res.groupby(["sex", "slot"]).weight.sum()
nat_open = res.assign(a=res.out_open * res.weight).groupby(["sex", "slot"]).a.sum() / \
      res.groupby(["sex", "slot"]).weight.sum()
for s in ("1", "2"):
    v, o = nat.loc[s], nat_open.loc[s]
    print(f"sex {s}: share outside (no school) at 08:00 {v[16]:.2f}, 12:00 {v[24]:.2f}, 16:00 {v[32]:.2f}, "
          f"20:00 {v[40]:.2f}, 02:00 {v[4]:.2f}; mean {v.mean():.3f}; in open-air work at 16:00 {o[32]:.2f}, "
          f"mean {o.mean():.3f} ({100 * o.mean() / v.mean():.0f}% of the time outside)")
for c in OUT_COLS[3:]:
    x = res.assign(a=res[c] * res.weight).groupby("sex").a.sum() / res.assign(a=res.out_open * res.weight).groupby("sex").a.sum()
    print(f"{c}: {100 * x.get('1', np.nan):.1f}% (men) and {100 * x.get('2', np.nan):.1f}% (women) of the outdoor-work time")

# covariates by state: rural share, agricultural workers, dwelling structure, consumption
m["w"] = m.mult.astype(float) / 100
worker = m.status.isin(["11", "12", "21", "31", "41", "51"])
m["agri"] = worker & m.nic.isin(["01", "02", "03"])
h = read("TUS106HH.TXT", HH)
h = h[h["size"] != ""].copy()                                     # two casualty households have no entries
h["wp"] = h.mult.astype(float) / 100 * h["size"].astype(float)     # persons
h["mpce"] = h.mce.astype(float) / h["size"].astype(float)
print(f"{len(h):,} households, {h.wp.sum() / 1e6:,.0f} million people; dwelling structure codes "
      f"{h.structure.value_counts().sort_index().to_dict()}")


def wmedian(x, w):
    o = np.argsort(x.values)
    c = np.cumsum(w.values[o])
    return x.values[o][np.searchsorted(c, c[-1] / 2)]


print(f"median monthly consumption per person, India: {wmedian(h.mpce, h.wp):,.0f}")
cov = pd.DataFrame({
    "tus_rural": m.groupby("state").apply(lambda d: (d.w * (d.sector == "1")).sum() / d.w.sum(), include_groups=False),
    "tus_agri": m.groupby("state").apply(lambda d: (d.w * d.agri).sum() / d.w.sum(), include_groups=False),
    "tus_katcha": h.groupby("state").apply(lambda d: (d.wp * (d.structure == "1")).sum() / d.wp.sum(), include_groups=False),
    "tus_nonpucca": h.groupby("state").apply(lambda d: (d.wp * d.structure.isin(["1", "2"])).sum() / d.wp.sum(),
                                             include_groups=False),
    "tus_mpce": h.groupby("state").apply(lambda d: wmedian(d.mpce, d.wp), include_groups=False),
    "tus_households": h.groupby("state").size(),
    "tus_persons": p.groupby("state").size()})
cov.index.name = "state"
cov.to_csv(OUT / "tus_state.csv", float_format="%.4g")
print(cov.loc[["MADHYA PRADESH", "WEST BENGAL", "KERALA", "BIHAR", "ODISHA", "PUNJAB"]].round(3).to_string())
