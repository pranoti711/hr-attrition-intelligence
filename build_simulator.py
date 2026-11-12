#!/usr/bin/env python3
"""
Layer 3: build the interactive what-if simulator as ONE self-contained HTML file
(reports/what_if_simulator.html). Open it in any browser, or host it for free
on GitHub Pages - no server and no database needed once it is built.

Data: staging.hr_employee + mart.employee_attrition_risk (IBM HR - SYNTHETIC),
      mart.attrition_comparable (REAL company data from Layer 1) for the default target.
Run:  python build_simulator.py        (needs DATABASE_URL)
Re-run after any change to the risk model; the numbers are baked into the HTML.
"""
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sqlalchemy import create_engine

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports" / "what_if_simulator.html"
OUT.parent.mkdir(exist_ok=True)

USD_INR, REALIZATION, FALLBACK_ATTRITION = 95.0, 0.5, 13.0
RAISES = (5, 10, 15, 20)  # pay-correction sizes offered in the UI
PROTECTED = ["age", "gender", "marital_status"]
DERIVED = ["employee_id", "attrition_flag", "tenure_band", "age_band", "income_quartile",
           "risk_probability", "risk_tier"]

url = os.environ.get("DATABASE_URL")
if not url:
    raise SystemExit("Set DATABASE_URL first, e.g. postgresql+psycopg2://postgres:PASSWORD@localhost:5432/hr_attrition")
engine = create_engine(url)
emp = pd.read_sql("SELECT * FROM staging.hr_employee", engine)
risk = pd.read_sql("SELECT employee_id, risk_probability, risk_tier FROM mart.employee_attrition_risk", engine)
emp = emp.merge(risk, on="employee_id", how="inner")

try:
    b = pd.read_sql("SELECT DISTINCT ON (company) company, attrition_pct FROM mart.attrition_comparable "
                    "WHERE company IN ('TCS','Infosys') ORDER BY company, period_end DESC", engine)
    target = float(b["attrition_pct"].astype(float).mean()) if not b.empty else FALLBACK_ATTRITION
except Exception:
    target = FALLBACK_ATTRITION


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def design(frame):
    raw = frame.drop(columns=[c for c in DERIVED + PROTECTED if c in frame.columns]).copy()
    raw["is_overtime"] = raw["is_overtime"].astype(int)
    cats = raw.select_dtypes(exclude=[np.number]).columns.tolist()
    return pd.get_dummies(raw, columns=cats).astype(float)


X_all = design(emp)
gbm = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3,
                                 subsample=0.8, random_state=42).fit(X_all, emp["attrition_flag"])
cols = X_all.columns


def model_logit(frame):
    return logit(gbm.predict_proba(design(frame).reindex(columns=cols, fill_value=0.0))[:, 1])


cur = emp[emp["attrition_flag"] == 0].reset_index(drop=True)
base = model_logit(cur)
low_income = (cur["monthly_income"] < cur["monthly_income"].quantile(0.5)).to_numpy()
cond = {
    "ot": cur["is_overtime"].to_numpy().astype(bool),
    "so": (cur["stock_option_level"] == 0).to_numpy(),
    "tr": (cur["business_travel"] == "Travel_Frequently").to_numpy(),
    "pay": low_income,
}


def change(frame, key, mask, raise_pct=10):
    f = frame.copy()
    if key == "ot":
        f.loc[mask, "is_overtime"] = False
    elif key == "so":
        f.loc[mask, "stock_option_level"] = 1
    elif key == "tr":
        f.loc[mask, "business_travel"] = "Travel_Rarely"
    elif key == "pay":
        f.loc[mask, "monthly_income"] = f.loc[mask, "monthly_income"] * (1 + raise_pct / 100)
    return f


delta = {k: model_logit(change(cur, k, cond[k])) - base for k in ("ot", "so", "tr")}
for r in RAISES:
    delta[f"pay{r}"] = model_logit(change(cur, "pay", cond["pay"], r)) - base

tier_code = cur["risk_tier"].map({"High": "H", "Medium": "M", "Low": "L"})
records = []
for i, row in cur.iterrows():
    rec = {"role": row["job_role"], "dept": row["department"], "tier": tier_code[i],
           "lvl": int(row["job_level"]), "inc": float(row["monthly_income"]),
           "z": round(float(logit(row["risk_probability"])), 4),
           "ot": int(cond["ot"][i]), "so": int(cond["so"][i]), "tr": int(cond["tr"][i]),
           "lo": int(cond["pay"][i]),
           "dOT": round(float(delta["ot"][i]), 4), "dSO": round(float(delta["so"][i]), 4),
           "dTR": round(float(delta["tr"][i]), 4)}
    for r in RAISES:
        rec[f"dP{r}"] = round(float(delta[f"pay{r}"][i]), 4)
    records.append(rec)

defaults = {"target": round(target, 1), "usd": USD_INR, "scale": 1.0, "mult": 1.0,
            "real": int(REALIZATION * 100)}

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Attrition what-if simulator</title>
<style>
:root{--bg:#ffffff;--fg:#1c1e21;--muted:#5f6670;--card:#f4f6f8;--line:#d9dde2;--accent:#1f6feb;--good:#1a7f37;--bad:#cf222e}
@media (prefers-color-scheme: dark){:root{--bg:#0f1318;--fg:#e6e9ee;--muted:#9aa3ae;--card:#1a2028;--line:#2d3641;--accent:#58a6ff;--good:#3fb950;--bad:#f85149}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{padding:20px 24px 8px;max-width:1280px;margin:0 auto}
h1{margin:0 0 4px;font-size:22px}
.sub{color:var(--muted);font-size:13px}
.wrap{display:grid;grid-template-columns:minmax(260px,330px) 1fr;gap:20px;padding:12px 24px 32px;align-items:start;max-width:1280px;margin:0 auto}
@media (max-width:860px){.wrap{grid-template-columns:1fr}}
.panel,.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.panel h2,.card h2{margin:0 0 10px;font-size:14px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
label{display:block;font-size:13px;margin:10px 0 2px}
label .v{float:right;font-weight:600}
input[type=range]{width:100%}
select,input[type=number]{width:100%;padding:4px;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px}
.row{display:flex;gap:10px;align-items:center;margin-top:8px;font-size:13px}
.row input[type=range]{flex:1}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:14px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.kpi .n{font-size:20px;font-weight:700}
.kpi .t{font-size:12px;color:var(--muted)}
.good{color:var(--good)} .bad{color:var(--bad)}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:600}
.bar{height:10px;border-radius:5px;background:var(--accent)}
.card{margin-bottom:14px}
.note{font-size:12px;color:var(--muted);margin-top:8px}
.matrix-chart{overflow-x:auto}
.matrix-chart svg{display:block;width:100%;min-width:760px;max-width:900px;height:auto;margin:0 auto}
svg text{fill:var(--fg);font-size:11px}
</style>
</head>
<body>
<header>
  <h1>Attrition what-if simulator</h1>
  <div class="sub">Synthetic IBM HR data (method demonstration), calibrated to real IT-services attrition. Change the assumptions on the left; every number updates live.</div>
</header>
<div class="wrap">
  <div class="panel">
    <h2>Assumptions</h2>
    <label>Annual attrition target <span class="v" id="v_target"></span></label>
    <input type="range" id="target" min="5" max="30" step="0.5">
    <label>Effect that is really achievable <span class="v" id="v_real"></span></label>
    <input type="range" id="real" min="0" max="100" step="5">
    <label>Replacement cost scale (1.0 = 0.5x to 2x salary by level) <span class="v" id="v_mult"></span></label>
    <input type="range" id="mult" min="0.5" max="1.5" step="0.05">
    <label>Pay scale (1.0 = dataset's US-level pay) <span class="v" id="v_scale"></span></label>
    <input type="range" id="scale" min="0.1" max="1.5" step="0.05">
    <label>USD to INR <span class="v" id="v_usd"></span></label>
    <input type="range" id="usd" min="70" max="110" step="0.5">

    <h2 style="margin-top:18px">Who gets treated</h2>
    <div class="row"><label style="margin:0"><input type="checkbox" id="tH" checked> High</label>
      <label style="margin:0"><input type="checkbox" id="tM" checked> Medium</label>
      <label style="margin:0"><input type="checkbox" id="tL"> Low</label></div>

    <h2 style="margin-top:18px">Levers (cost = % of salary per person)</h2>
    <div class="row"><input type="checkbox" id="on_ot" checked><span style="flex:0 0 120px">Cap overtime</span>
      <input type="range" id="c_ot" min="0" max="10" step="0.5"><span id="v_c_ot" style="width:38px;text-align:right"></span></div>
    <div class="row"><input type="checkbox" id="on_so" checked><span style="flex:0 0 120px">Stock options</span>
      <input type="range" id="c_so" min="0" max="10" step="0.5"><span id="v_c_so" style="width:38px;text-align:right"></span></div>
    <div class="row"><input type="checkbox" id="on_tr" checked><span style="flex:0 0 120px">Cut travel</span>
      <input type="range" id="c_tr" min="0" max="10" step="0.5"><span id="v_c_tr" style="width:38px;text-align:right"></span></div>
    <div class="row"><input type="checkbox" id="on_pay"><span style="flex:0 0 120px">Pay correction</span>
      <select id="raise"><option value="5">+5%</option><option value="10" selected>+10%</option><option value="15">+15%</option><option value="20">+20%</option></select></div>
    <div class="note">The package is the ticked levers. Pay correction costs the size of the raise itself.</div>
  </div>

  <div>
    <div class="kpis" id="kpis"></div>
    <div class="card"><h2>Levers</h2><div class="scroll"><table id="levers"></table></div>
      <div class="note">"Worth treating" = people for whom the estimated saving exceeds the cost. Package effects are added on the log-odds scale (an approximation of the joint model).</div></div>
    <div class="card"><h2>Package net benefit if the effect is only a quarter / half / three-quarters / fully real</h2><div id="sens"></div></div>
    <div class="card"><h2>Risk x cost matrix by job role</h2><div id="matrix"></div>
      <div class="note">Right = riskier roles. Up = costlier to replace. Bubble = expected cost. Dashed lines = workforce averages.</div></div>
    <div class="note">Model estimates show association, not proven cause: pilot before scaling, and expect the benefit of picking the best-looking people to be overstated. Data is synthetic.</div>
  </div>
</div>
<script>
const DATA = /*__DATA__*/[];
const DEFAULTS = /*__DEFAULTS__*/{};
const MULT = {1: 0.5, 2: 0.75, 3: 1.0, 4: 1.5, 5: 2.0};
const sig = z => 1 / (1 + Math.exp(-z));
const lgt = p => { p = Math.min(Math.max(p, 1e-4), 1 - 1e-4); return Math.log(p / (1 - p)); };
const CR = 1e7;
const inr = x => Math.abs(x) >= CR ? "Rs " + (x / CR).toFixed(1) + " crore" : "Rs " + (x / 1e5).toFixed(1) + " lakh";

function leverDef(P) {
  return [
    {k: "ot", name: "Cap overtime", on: P.on_ot, pct: P.c_ot / 100, cond: d => d.ot, dl: d => d.dOT},
    {k: "so", name: "Stock-option grant", on: P.on_so, pct: P.c_so / 100, cond: d => d.so, dl: d => d.dSO},
    {k: "tr", name: "Cut frequent travel", on: P.on_tr, pct: P.c_tr / 100, cond: d => d.tr, dl: d => d.dTR},
    {k: "pay", name: "Pay correction +" + P.raise + "%", on: P.on_pay, pct: P.raise / 100, cond: d => d.lo, dl: d => d["dP" + P.raise]},
  ];
}

function simulate(P, realization) {
  const real = realization === undefined ? P.real / 100 : realization;
  const n = DATA.length;
  const z = DATA.map(d => d.z);
  let lo = -6, hi = 6;
  for (let i = 0; i < 60; i++) {                       // intercept shift so mean risk = target
    const mid = (lo + hi) / 2;
    let m = 0; for (let j = 0; j < n; j++) m += sig(z[j] + mid);
    if (m / n < P.target / 100) lo = mid; else hi = mid;
  }
  const shift = (lo + hi) / 2;
  const tierOK = d => (d.tier === "H" && P.tH) || (d.tier === "M" && P.tM) || (d.tier === "L" && P.tL);
  const rows = DATA.map((d, j) => {
    const p = sig(z[j] + shift);
    const sal = d.inc * 12 * P.usd * P.scale;
    const repl = sal * MULT[d.lvl] * P.mult;
    return {d, p, sal, repl};
  });
  const exits = rows.reduce((a, r) => a + r.p, 0);
  const cost = rows.reduce((a, r) => a + r.p * r.repl, 0);
  const defs = leverDef(P);
  const result = {exits, cost, levers: [], rows};
  function evalSet(name, parts) {
    let elig = 0, saved = 0, spend = 0, benefit = 0, pos = 0, posNet = 0, posSpend = 0, posSaved = 0;
    rows.forEach(r => {
      const act = parts.filter(L => tierOK(r.d) && L.cond(r.d));
      if (!act.length) return;
      const dl = act.reduce((a, L) => a + L.dl(r.d), 0);
      const c = act.reduce((a, L) => a + L.pct * r.sal, 0);
      const s = r.p - sig(lgt(r.p) + real * dl);
      const b = s * r.repl;
      elig++; saved += s; spend += c; benefit += b;
      if (b - c > 0) { pos++; posNet += b - c; posSpend += c; posSaved += s; }
    });
    return {name, elig, saved, spend, net: benefit - spend, pos, posNet, posSpend, posSaved};
  }
  defs.forEach(L => result.levers.push(evalSet(L.name, [L])));
  const chosen = defs.filter(L => L.on);
  result.pkg = chosen.length ? evalSet("Selected package", chosen) : null;
  return result;
}

function matrix(res) {
  const g = {};
  res.rows.forEach(r => {
    const k = r.d.role;
    g[k] = g[k] || {n: 0, p: 0, repl: 0, cost: 0};
    g[k].n++; g[k].p += r.p; g[k].repl += r.repl; g[k].cost += r.p * r.repl;
  });
  const segs = Object.entries(g).map(([k, v]) => ({role: k, risk: v.p / v.n, repl: v.repl / v.n, cost: v.cost, n: v.n}));
  const n = res.rows.length;
  return {segs, avgRisk: res.exits / n, avgRepl: res.rows.reduce((a, r) => a + r.repl, 0) / n};
}

if (typeof document !== "undefined") {
  const $ = id => document.getElementById(id);
  const ids = ["target", "real", "mult", "scale", "usd", "c_ot", "c_so", "c_tr", "raise"];
  ids.forEach(id => { if (DEFAULTS[id] !== undefined) $(id).value = DEFAULTS[id]; });
  $("c_ot").value = 3; $("c_so").value = 4; $("c_tr").value = 1;
  function params() {
    const P = {tH: $("tH").checked, tM: $("tM").checked, tL: $("tL").checked,
      on_ot: $("on_ot").checked, on_so: $("on_so").checked, on_tr: $("on_tr").checked, on_pay: $("on_pay").checked};
    ids.forEach(id => P[id] = parseFloat($(id).value));
    return P;
  }
  function render() {
    const P = params();
    $("v_target").textContent = P.target + "%"; $("v_real").textContent = P.real + "%";
    $("v_mult").textContent = P.mult.toFixed(2) + "x"; $("v_scale").textContent = P.scale.toFixed(2) + "x";
    $("v_usd").textContent = P.usd.toFixed(1);
    ["ot", "so", "tr"].forEach(k => $("v_c_" + k).textContent = P["c_" + k] + "%");
    const res = simulate(P);
    const pk = res.pkg;
    const cards = [
      ["Expected exits (12 months)", res.exits.toFixed(0), ""],
      ["Expected replacement cost", inr(res.cost), ""],
      ["Average cost per exit", inr(res.cost / res.exits), ""],
      ["Package: exits avoided", pk ? pk.posSaved.toFixed(1) : "-", ""],
      ["Package: programme cost", pk ? inr(pk.posSpend) : "-", ""],
      ["Package: net benefit", pk ? inr(pk.posNet) : "-", pk && pk.posNet > 0 ? "good" : "bad"],
      ["Package: return on spend", pk && pk.posSpend ? Math.round(100 * pk.posNet / pk.posSpend) + "%" : "-", pk && pk.posNet > 0 ? "good" : "bad"],
    ];
    $("kpis").innerHTML = cards.map(c => `<div class="kpi"><div class="n ${c[2]}">${c[1]}</div><div class="t">${c[0]}</div></div>`).join("");
    const all = res.levers.concat(pk ? [pk] : []);
    const maxNet = Math.max(1, ...all.map(l => Math.abs(l.posNet)));
    $("levers").innerHTML = "<tr><th>Lever</th><th>Eligible</th><th>Cost if all eligible</th><th>Net if all eligible</th><th>Worth treating</th><th>Net if targeted</th><th></th></tr>" +
      all.map(l => `<tr><td>${l.name}</td><td>${l.elig}</td><td>${inr(l.spend)}</td><td class="${l.net >= 0 ? "good" : "bad"}">${inr(l.net)}</td><td>${l.pos}</td><td class="${l.posNet > 0 ? "good" : ""}">${inr(l.posNet)}</td><td style="width:90px"><div class="bar" style="width:${Math.round(80 * l.posNet / maxNet)}px"></div></td></tr>`).join("");
    if (pk) {
      $("sens").innerHTML = "<div class='scroll'><table><tr>" + [0.25, 0.5, 0.75, 1].map(r => `<th>${r * 100}% real</th>`).join("") + "</tr><tr>" +
        [0.25, 0.5, 0.75, 1].map(r => { const s = simulate(P, r).pkg; return `<td class="${s.posNet > 0 ? "good" : ""}">${inr(s.posNet)}</td>`; }).join("") + "</tr></table></div>";
    } else { $("sens").textContent = "Tick at least one lever."; }
    const m = matrix(res);
    const W = 840, H = Math.max(360, m.segs.length * 34 + 90);
    const pad = 48, plotRight = 555, labelX = 600;
    const xs = m.segs.map(s => s.risk), ys = m.segs.map(s => s.repl), cs = m.segs.map(s => s.cost);
    const xmax = Math.max(...xs) * 1.15, ymax = Math.max(...ys) * 1.15, cmax = Math.max(...cs);
    const X = v => pad + (plotRight - pad) * v / xmax, Y = v => H - pad - (H - 2 * pad) * v / ymax;
    const points = m.segs.map(s => ({s, x: X(s.risk), y: Y(s.repl),
      r: 6 + 22 * Math.sqrt(s.cost / cmax)})).sort((a, b) => a.y - b.y);
    const labelTop = 28, labelBottom = H - 58;
    const labelStep = points.length > 1 ? (labelBottom - labelTop) / (points.length - 1) : 0;
    $("matrix").innerHTML = `<div class="matrix-chart"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Average attrition risk and replacement cost by job role">
      <line x1="${pad}" y1="${H - pad}" x2="${W - 10}" y2="${H - pad}" stroke="currentColor" opacity=".4"/>
      <line x1="${pad}" y1="10" x2="${pad}" y2="${H - pad}" stroke="currentColor" opacity=".4"/>
      <line x1="${X(m.avgRisk)}" y1="10" x2="${X(m.avgRisk)}" y2="${H - pad}" stroke="currentColor" opacity=".5" stroke-dasharray="4 3"/>
      <line x1="${pad}" y1="${Y(m.avgRepl)}" x2="${plotRight}" y2="${Y(m.avgRepl)}" stroke="currentColor" opacity=".5" stroke-dasharray="4 3"/>
      <text x="${(pad + plotRight) / 2}" y="${H - 8}" text-anchor="middle">Average 12-month attrition risk (%)</text>
      <text x="12" y="${H / 2}" transform="rotate(-90 12 ${H / 2})" text-anchor="middle">Average replacement cost (Rs lakh)</text>` +
      points.map((p, i) => {
        const labelY = points.length === 1 ? (labelTop + labelBottom) / 2 : labelTop + i * labelStep;
        return `<line x1="${Math.min(p.x + p.r + 2, plotRight)}" y1="${p.y}" x2="${labelX - 10}" y2="${labelY}" stroke="currentColor" opacity=".25"/>
          <circle cx="${p.x}" cy="${p.y}" r="${p.r}" fill="var(--accent)" fill-opacity=".45" stroke="var(--accent)"><title>${p.s.role}: risk ${(100 * p.s.risk).toFixed(1)}%, replacement ${(p.s.repl / 1e5).toFixed(0)} lakh, expected cost ${inr(p.s.cost)}</title></circle>
          <text x="${labelX}" y="${labelY}" dominant-baseline="middle">${p.s.role}</text>`;
      }).join("") +
      `<text x="${plotRight - 8}" y="22" text-anchor="end" opacity=".7">Fix first</text></svg></div>`;
  }
  document.querySelectorAll("input,select").forEach(e => e.addEventListener("input", render));
  render();
}
</script>
</body>
</html>
"""

html = (TEMPLATE.replace("/*__DATA__*/[]", json.dumps(records, separators=(",", ":")))
                .replace("/*__DEFAULTS__*/{}", json.dumps(defaults)))
OUT.write_text(html, encoding="utf-8")
print(f"{len(cur)} current employees; default target attrition {target:.1f}%")
print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB). Open it in a browser.")
