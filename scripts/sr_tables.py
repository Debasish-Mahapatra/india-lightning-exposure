#!/usr/bin/env python3
"""Rows of Supplementary Table S1 (the 20 districts with the largest share of
India's exposure), Table S2 (States and Union Territories with at least a
million people, by deaths), Table S3 (deaths per unit exposure under four
exposure measures, with detection and vulnerability indicators, for states
with at least 10 deaths) and Table S4 (main numbers with every record counted
as a flash and with grouped flashes), from results/district_table.csv,
state_table.csv, deaths_ratios_by_measure.csv, vulnerability_state.csv and
check_grouping_effect.csv, written into manuscript/main.tex between
"% rows written by scripts/sr_tables.py (districts|states|vulnerability|grouping|codes|models)"
and "% end of rows". The ratio of Table S2 is given as a dash for fewer than 10
deaths in 2022-2024. Also the table of activities counted as outdoor work
(results/tus_outdoor_work_codes.csv) and the matched count models
(results/deaths_models_table.csv).
Usage: sr_tables.py   (after sr_exposure.py, tus_outdoor.py, sr_deaths_models.py and check_grouping_effect.py)
"""
import re

import pandas as pd

from common import REGIONS, RESULTS, WS, state_label

st = pd.read_csv(RESULTS / "state_table.csv", index_col=0)
states = []
for s, r in st[st.population_m >= 1].iterrows():
    ratio = f"{r.ratio:.2f} ({r.ratio_lo:.2f}--{r.ratio_hi:.2f})" if r.deaths_per_year * 3 >= 10 else "--"
    states.append(f"{state_label(s)} & {r.population_m:.1f} & {r.density:.2f} & {r.exposure:.2f} & "
                  f"{r.share:.1f} & {r.deaths_per_year:{'.1f' if 0 < r.deaths_per_year < 1 else '.0f'}} & "
                  f"{r.deaths_per_million:.2f} & {ratio} \\\\")
dt = pd.read_csv(RESULTS / "district_table.csv").head(20).reset_index(drop=True)
districts = [f"{k + 1} & {r.DISTRICT} & {r.ST_NM} & {r.population_m:.1f} & {r.exposure:.1f} & {r.share:.2f} \\\\"
             for k, r in dt.iterrows()]
v = pd.read_csv(RESULTS / "deaths_ratios_by_measure.csv", index_col=0).join(
    pd.read_csv(RESULTS / "vulnerability_state.csv", index_col=0)[["out_frac"]])
vulnerability = [f"{state_label(s)} & {r.de:.2f} & {100 * r.out_frac:.0f} & {100 * r.census_agri:.0f} & "
                 f"{r.tus_mpce:,.0f} & {r.R_E:.2f} & {r.R_E_out:.2f} & {r.R_E_cg:.2f} & {r.R_E_out_cg:.2f} \\\\"
                 for s, r in v.sort_values("R_E", ascending=False).iterrows()]
g = pd.read_csv(RESULTS / "check_grouping_effect.csv", index_col=0)


def row(label, key, fmt, scale=1.0):
    return f"{label} & {fmt.format(g.loc[key, 'records'] * scale)} & {fmt.format(g.loc[key, 'flashes'] * scale)} \\\\"


SEAS = ("Winter", "Pre-monsoon", "Monsoon", "Post-monsoon")
slope = [f"{g.loc['States: slope', c]:.2f} ({g.loc['States: slope 95% low', c]:.2f}--{g.loc['States: slope 95% high', c]:.2f})"
         for c in ("records", "flashes")]
grouping = [row("Flashes a year inside India (million)", "India, per km2 per year", "{:.1f}", 3.278),
            row("Flash density over India", "India, per km2 per year", "{:.2f}"),
            row("Flash density where people live", "India exposure (where people live)", "{:.2f}"),
            row("Highest flash density of a 0.25$^\\circ$ cell", "Max cell (>= half in India)", "{:.1f}"),
            *[row(f"Share of India's flashes, {r} (\\%)", f"{r}: share of India's flashes, %", "{:.1f}") for r in REGIONS],
            *[row(f"Share of the year's flashes, {s.lower()} (\\%)", f"{s}: share of the year, %", "{:.1f}") for s in SEAS],
            row("Peak hour over India, pre-monsoon season (IST)", "Peak hour IST, Pre-monsoon, India", "{:.0f}:00"),
            row("Peak hour over India, monsoon season (IST)", "Peak hour IST, Monsoon, India", "{:.0f}:00"),
            row("Flashes of 20\\,kA or more (\\%)", "Share >= 20 kA, %", "{:.1f}"),
            row("Flashes of 50\\,kA or more (\\%)", "Share >= 50 kA, %", "{:.2f}"),
            row("Districts holding half of India's exposure", "Districts holding half of the exposure", "{:.0f}"),
            row("Deaths against exposure across states: correlation", "States: r (log deaths per million, log exposure)", "{:.2f}"),
            f"Deaths against exposure across states: slope & {slope[0]} & {slope[1]} \\\\",
            row("Deaths against exposure within states: correlation", "Within states: r", "{:.2f}"),
            row("Deaths against exposure within states: $p$", "Within states: p", "{:.3f}"),
            *[row(f"Deaths per unit of exposure, {s}", f"Deaths per exposure, {s}", "{:.2f}")
              for s in ("Madhya Pradesh", "Rajasthan", "Telangana", "Odisha", "Bihar", "Karnataka", "West Bengal", "Assam", "Kerala")],
            *[row(f"Flashes per million tonnes of rain, {s.lower()}", f"{s}: flashes per Mt of rain", "{:.2f}") for s in SEAS],
            row("Flashes with less than 2.5\\,mm of rain that day (\\%)", "Share with < 2.5 mm that day, %", "{:.1f}"),
            row("The same in the pre-monsoon season (\\%)", "Pre-monsoon share with < 2.5 mm, %", "{:.1f}")]
# activities counted as outdoor work: ICATUS 2016 codes of the Time Use Survey 2019
ACT = {121: "Growing crops", 122: "Raising animals", 123: "Forestry and logging", 124: "Fishing",
       125: "Aquaculture", 126: "Mining and quarrying", 128: "Construction",
       211: "Growing crops and kitchen gardening", 212: "Farming of animals", 213: "Hunting and trapping",
       214: "Forestry and logging", 215: "Gathering wild products", 216: "Fishing", 217: "Aquaculture",
       218: "Mining and quarrying", 230: "Construction", 241: "Gathering firewood and other fuel",
       242: "Fetching water",
       110: "Employment in corporations, government and non-profit institutions",
       141: "Activities ancillary to employment", 142: "Breaks during working time"}
BRANCH = {"A": "For the market, in household enterprises",
          "B": "For own final use",
          "C": "Employment in agriculture, forestry, fishing, mining or construction"}
cc = pd.read_csv(RESULTS / "tus_outdoor_work_codes.csv")
pct = lambda v: f"{100 * v:.1f}" if v >= 0.0005 else "<0.1"
codes = []
for b_, g in cc.groupby("branch"):
    tot_ = g[["men", "women", "all"]].sum()
    codes.append(f"\\multicolumn{{2}}{{@{{}}l}}{{\\textit{{{BRANCH[b_]}}}}} & \\textit{{{pct(tot_['men'])}}} & "
                 f"\\textit{{{pct(tot_['women'])}}} & \\textit{{{pct(tot_['all'])}}} \\\\")
    codes += [f"{r.act} & {ACT[r.act]} & {pct(r.men)} & {pct(r.women)} & {pct(r.all)} \\\\" for r in g.itertuples()]
# matched negative binomial models
mt = pd.read_csv(RESULTS / "deaths_models_table.csv")


def coef(r, k):
    if pd.isna(r[k]):
        return ""
    lo, hi = r[k + "_lo"], r[k + "_hi"]
    rng = f"{lo:.2f}--{hi:.2f}" if lo >= 0 else f"${lo:.2f}$ to {hi:.2f}"
    return f"${r[k]:.2f}$ ({rng})" if r[k] < 0 else f"{r[k]:.2f} ({rng})"


MODEL_TEX = {"Exposure": "$E$", "Exposure + share of exposure outside": "$E$ + $O$",
             "Exposure while outside": "$E \\times O$", "Exposure + share in outdoor work": "$E$ + $W$",
             "Exposure during outdoor work": "$E \\times W$", "Exposure + farm share": "$E$ + $F$",
             "Exposure during outdoor work + farm share": "$E \\times W$ + $F$",
             "Exposure + share in outdoor work + farm share": "$E$ + $W$ + $F$",
             "Exposure during outdoor work, activities only": "$E \\times W_{\\mathrm{codes}}$",
             "Exposure during outdoor work, equal allocation": "$E \\times W_{\\mathrm{equal}}$"}
models = []
for src, g in mt.groupby("source", sort=False):
    head = "Network exposure" if src == "network" else "Satellite-based exposure"
    models.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{head}}}}} \\\\")
    prev = ""
    for _, r in g.iterrows():
        lab = prev + " + $F$" if r.model == "+ farm share" else MODEL_TEX[r.model]
        prev = lab
        models.append(f"{lab} & {coef(r, 'exposure')} & {coef(r, 'share')} & {coef(r, 'farm')} & "
                      f"{r.alpha:.2f} & {r.aic:.1f} & {100 * r.deviance_explained:.0f} \\\\")
f = WS / "manuscript/main.tex"
rows = {"states": states, "districts": districts, "vulnerability": vulnerability, "grouping": grouping,
        "codes": codes, "models": models}
tex, n = re.subn(r"(% rows written by scripts/sr_tables\.py \((states|districts|vulnerability|grouping|codes|models)\)\n)(.*?)(% end of rows\n)",
                 lambda m: m.group(1) + "\n".join(rows[m.group(2)]) + "\n" + m.group(4), f.read_text(), flags=re.S)
assert n == 6, n
f.write_text(tex)
print(f"{f}: {len(districts)} districts, {len(states)} states, {len(vulnerability)} states by exposure measure, "
      f"{len(grouping)} numbers with and without grouping, {len(codes)} rows of activity codes, {len(models)} rows of models")
