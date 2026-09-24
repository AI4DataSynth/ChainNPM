#!/usr/bin/env python3
"""Keep the manuscript's four-dataset tables in sync with the archived TSVs.

Tables live inline in VLDB-submission/manuscript/main.tex between markers:

    % BEGIN GENERATED tab:four-ds     ... rows ...  % END GENERATED tab:four-ds
    % BEGIN GENERATED tab:re-appendix ... rows ...  % END GENERATED tab:re-appendix

so the caption stays hand-editable while numbers are regenerated, not retyped.

Inputs (read only): results/tables/p7_effect.tsv, results/tables/tab_re_appendix.tsv
Output:            VLDB-submission/manuscript/main.tex (marker regions only)

Cell convention for the effect table
------------------------------------
Both planting protocols' effects are summarised without picking a strength:

  plant_product   cell = ``min--max'' over the FOUR planting strengths
                  (t = 0.1, 0.2, 0.316, 0.4) of the per-seed median retention
                  error.  A range, because reporting one strength would be a
                  choice the data does not make for us.
  plant_xor       cell = single value (the only strength, phi = 0.8).

Markers:  dagger  any cell in the group is partial (privpetal n < 10)
          ddagger every strength in the group has `syn_reused` = 1, i.e. the
                  synthetic effect coincides across strengths
"""
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
T = os.path.join(ROOT, "results", "tables")
MAIN = os.path.join(ROOT, "VLDB-submission", "manuscript", "main.tex")

SERIES = [("chainnpm", r"\chainnpm"), ("pertable", "PrivMRF"),
          ("privbayes", "PrivBayes"), ("pbpgm", "PB-PGM"),
          ("lavaprop", "PrivLava"), ("privpetal", r"\privpetal")]
DATASETS = ["financial", "imdb", "instacart", "movielens"]
FUNCTIONAL = {"financial": "delta", "imdb": "contrast",
              "instacart": "contrast", "movielens": "contrast"}
EPS_LIST = [0.8, 3.2]
NCOL_EFFECT = 8
T_VALUES = [0.1, 0.2, 0.316, 0.4]


def load_effect():
    with open(os.path.join(T, "p7_effect.tsv")) as fh:
        lines = [l for l in fh if not l.startswith("#")]
    hdr = lines[0].rstrip("\n").split("\t")
    groups = {}
    for r in csv.DictReader(lines, delimiter="\t", fieldnames=hdr):
        if r["dataset"] == "dataset":
            continue
        key = (r["mode"], r["dataset"], float(r["eps"]), r["target"], r["series"])
        groups.setdefault(key, []).append(r)
    for key in groups:
        groups[key].sort(key=lambda r: float(r["t"]))
    return groups


def cell_product(rows):
    """min--max over the four planting strengths."""
    vals, partial, reused = [], False, True
    for r in rows:
        v = r["ret_err_med"]
        if v in ("", "nan", "None"):
            continue
        vals.append(float(v))
        if r["partial"] == "1":
            partial = True
        if r["syn_reused"] != "1":
            reused = False
    if not vals:
        return "--"
    lo, hi = min(vals), max(vals)
    s = f"{lo:.3f}" if abs(hi - lo) < 5e-4 else f"{lo:.3f}--{hi:.3f}"
    if partial:
        s += r"$^{\dagger}$"
    if reused and len(rows) >= len(T_VALUES):
        s += r"$^{\ddagger}$"
    return s


def cell_xor(rows):
    rows = [r for r in rows if r["ret_err_med"] not in ("", "nan", "None")]
    if not rows:
        return "--"
    r = rows[0]
    s = f"{float(r['ret_err_med']):.3f}"
    if r["partial"] == "1":
        s += r"$^{\dagger}$"
    if r["syn_reused"] == "1":
        s += r"$^{\ddagger}$"
    return s


def effect_body(groups):
    out = []
    for k, mode in enumerate(("plant_product", "plant_xor")):
        if k:
            out.append(r"\midrule")
        out.append(r"\multicolumn{%d}{@{}l}{\emph{%s}: }\\"
                   % (NCOL_EFFECT, mode.replace("_", r"\_")))
        for ds in DATASETS:
            for eps in EPS_LIST:
                cells = []
                for s, _ in SERIES:
                    rows = groups.get((mode, ds, eps, FUNCTIONAL[ds], s), [])
                    if not rows:
                        cells.append("--")
                    elif mode == "plant_product":
                        cells.append(cell_product(rows))
                    else:
                        cells.append(cell_xor(rows))
                out.append("%s & %.1f & %s \\\\" % (ds, eps, " & ".join(cells)))
    return "\n".join(out)


def re_body():
    with open(os.path.join(T, "tab_re_appendix.tsv")) as fh:
        lines = [l for l in fh if not l.startswith("#")]
    hdr = lines[0].rstrip("\n").split("\t")
    by = {}
    for l in lines[1:]:
        if not l.strip() or l.startswith("dataset"):
            continue
        r = dict(zip(hdr, l.rstrip("\n").split("\t")))
        by[(r["dataset"], r["method"])] = r
    eps_cols = ["0.1", "0.2", "0.4", "0.8", "1.6", "3.2"]
    out = []
    for k, ds in enumerate(DATASETS):
        if k:
            out.append(r"\midrule")
        for sname, slabel in SERIES:
            r = by.get((ds, sname))
            if r is None:
                continue
            vals = []
            for e in eps_cols:
                v = r.get("eps_" + e, "")
                vals.append(v if v not in ("nan", "", None) else "--")
            out.append("%s & %s & %s \\\\" % (slabel, ds, " & ".join(vals)))
    return "\n".join(out)


def patch(text, tag, body):
    begin, end = "%% BEGIN GENERATED %s" % tag, "%% END GENERATED %s" % tag
    i, j = text.find(begin), text.find(end)
    if i < 0 or j < 0:
        raise SystemExit("markers for %s not found in main.tex" % tag)
    nl = text.find("\n", i)
    after_begin = text[nl + 1:j]
    comments = ""
    for line in after_begin.split("\n"):
        if line.startswith("%"):
            comments += line + "\n"
        else:
            break
    old = text[nl + 1:j]
    new = comments + body + "\n"
    if old == new:
        return text, False
    return text[:nl + 1] + new + text[j:], True


def main():
    groups = load_effect()
    text = open(MAIN, encoding="utf-8").read()
    before = text
    text, c1 = patch(text, "tab:four-ds", effect_body(groups))
    text, c2 = patch(text, "tab:re-appendix", re_body())
    if text != before:
        open(MAIN, "w", encoding="utf-8").write(text)
        print("main.tex updated (tab:four-ds changed=%s, tab:re-appendix changed=%s)"
              % (c1, c2))
    else:
        print("main.tex already in sync (no change)")
    print("\n--- effect rows ---")
    print(effect_body(groups))


if __name__ == "__main__":
    main()