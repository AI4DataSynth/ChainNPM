#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""受控基准合并图（替换原 Fig 7 fig:evidence 与 Fig 8 fig:recovery）。

版式
  单栏 figure：自然宽 241.6 pt == \columnwidth，印刷 1:1，1 行 × 2 列 panel，
  两个面板在宽、高两个方向都缩小（宽度约 2.4 倍、高度约 1.2 倍）；
  --layout 1x2 复原此前的跨双栏 figure* 版式。
  常驻版式自检：文字出画布、文字两两重叠、文字压到旁边面板的数据区，
  任一出现即拒绝出图（sys.exit）。
  (a) Δ_syn/Δ_real vs ε  —— 效应恢复，六方法（含 PrivMRF 三次运行的范围带）；
  (b) 2-hop 查询 RE vs ε —— 查询精度，同一批六方法。
  x 三档 ε ∈ {0.8, 3.2, 8.0}；每点 3 次运行，竖段给出运行跨度。

为什么合并
  原 fig:evidence（四个基线、无 Chain-NPM）与原 fig:recovery（同一 Δ 恢复、
  六方法）画的是**同一个量**，只有方法集不同——读稿时被判定为内容重叠。
  合并后 (a) 承载效应恢复、(b) 承载查询精度，两 panel 各自回答一个不同的问题。

诚实性约束（两个 panel 讲述的方向并不一致，图注不得模糊这一点）
  * (a) 中四个 per-FK 基线在 0 附近（效应被结构性丢弃），Chain-NPM 接近 1；
  * (b) 中 Chain-NPM 在 ε=0.8 **高于**四个基线（0.79 vs 0.19–0.41），
    到 ε≥3.2 才反超——查询精度与效应恢复是两个不同的问题，
    该交叉必须由图注明确说明，不能靠视觉掩盖。
  * 自检：四个基线的 Δ 中位数必须与正文一致（±0.011），否则拒绝出图。
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TMP = os.path.join(ROOT, "tmp")
BASE_NAME = os.path.join(HERE, "fig_controlled_2panel")
COVERAGE_TXT = os.path.join(HERE, "plot_controlled_2panel_coverage.txt")
# --- probe additions -------------------------------------------------
LAYOUT = "col1sbs"   # set from --layout in main(); the manuscript layout
COL1_W_IN = 3.355     # 241.6 pt: == \columnwidth, so the pdf prints 1:1
SIZE = {"1x2": (7.05, 2.35), "col1sbs": (COL1_W_IN, 2.30)}

EPS = [("0.8", 0.8), ("3.2", 3.2), ("8.0", 8.0)]
BASE_KEYS = ["pertable", "lava_asis", "lava_prop", "denorm"]
CHAIN_KEYS = ["chain_direct", "chain_mixture"]
# 正文口径的基线 Δ 中位数（ε=0.8/3.2/8.0），用于自检。
# 2026-10-21：受控台架改为 10 runs 并与正文 tab:gap 同步重设。
EXP_DELTA = {"pertable": [0.035, 0.002, 0.009], "lava_asis": [0.009, 0.038, -0.011],
             "lava_prop": [-0.009, 0.001, 0.009], "denorm": [0.205, 0.239, 0.248]}
STYLES = [
    ("pertable",     "PrivMRF",         "#2b2b2b", "o", "-"),
    ("lava_asis",    "PrivLava (released)", "#6e6e6e", "^", "-"),
    ("lava_prop",    "PrivLava (Alg. 3)",  "#a8a8a8", "s", "-"),
    ("denorm",       "PrivMRF (denorm)", "#1f4e8c", "D", "--"),
    ("chain_direct", "Chain-NPM-direct", "#c0392b", "v", "-"),
    ("chain_mixture", "Chain-NPM-mixture", "#e67e22", "P", "-"),
]


def load():
    """Prefer the self-contained results/controlled/*_10run.json shipped with
    the artifact; fall back to the working tree's tmp/ inputs locally."""
    out = {}
    for eps_str, _ in EPS:
        paths = {}
        for name, prefix in (("base", "multihop_final"), ("chain", "chain_results")):
            shipped = os.path.join(ROOT, "results", "controlled",
                                   "%s_results_eps%s_10run.json" % (name, eps_str))
            local = os.path.join(TMP, "%s_eps%s.json" % (prefix, eps_str))
            paths[name] = shipped if os.path.exists(shipped) else local
        b = json.load(open(paths["base"]))["results"][eps_str]
        c = json.load(open(paths["chain"]))["results"][eps_str]
        out[eps_str] = dict(b, **c)
    return out




def audit_layout(fig, axes, layout, tol_pt=1.2, min_pt=0.8, intrude_pt=2.0):
    """Lay the figure out and report text that overflows the canvas or collides.

    Text-only guard.  Window extents come back in device pixels, so the
    thresholds are converted with the figure's dpi: tol_pt is the slack allowed
    outside the canvas (matplotlib clips a fraction of a legend's padding),
    min_pt is how far two boxes must overlap in BOTH directions before the pair
    counts -- the ascent/descent padding of stacked tick labels and of the
    successive lines of one paragraph stay below it, real collisions do not.
    A legend is a container and is never reported against its own entries; the
    same string drawn twice is invisible when it overlaps itself.
    In-panel annotations covering drawn data points are reported separately.
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    sc = fig.dpi / 72.0
    fb = fig.bbox
    tol, smin = tol_pt * sc, min_pt * sc
    from matplotlib.text import Text
    # a locator may return ticks past the view interval; matplotlib lays their
    # labels out but never draws them, so they must not count as collisions
    undrawn = set()
    for ax in fig.axes:
        for axis, lim in ((ax.yaxis, ax.get_ylim()), (ax.xaxis, ax.get_xlim())):
            lo, hi = min(lim), max(lim)
            for tick in axis.get_major_ticks():
                try:
                    loc = tick.get_loc()
                except Exception:
                    continue
                if loc < lo - 1e-9 or loc > hi + 1e-9:
                    for lab in (tick.label1, tick.label2):
                        undrawn.add(id(lab))
    legend_of = {}
    for leg in fig.legends:
        for t in leg.get_texts():
            legend_of[id(t)] = leg
    items = []                                # (kind, bbox, owner, snippet)
    for art in fig.findobj(Text):
        if not art.get_visible() or not art.get_text().strip():
            continue
        if id(art) in undrawn:
            continue
        try:
            bb = art.get_window_extent(rend)
        except Exception:
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        leg = legend_of.get(id(art))
        items.append(("legend entry" if leg is not None else "text", bb, leg,
                      art.get_text().replace(chr(10), " ")[:26], art))
    for leg in fig.legends:
        items.append(("LEGEND", leg.get_window_extent(rend), leg, "<legend>",
                      None))

    def box(b):
        return "x %.0f..%.0f y %.0f..%.0f" % ((b.x0 - fb.x0) / sc,
                                              (b.x1 - fb.x0) / sc,
                                              (b.y0 - fb.y0) / sc,
                                              (b.y1 - fb.y0) / sc)

    out_of_box = [(k, b, s) for k, b, _c, s, _a in items
                  if (b.x0 < fb.x0 - tol or b.y0 < fb.y0 - tol
                      or b.x1 > fb.x1 + tol or b.y1 > fb.y1 + tol)]
    collisions = []
    for i, (k1, b1, p1, s1, _a1) in enumerate(items):
        for (k2, b2, p2, s2, _a2) in items[i + 1:]:
            if p1 is not None and p1 is p2:       # legend box vs its own entry
                continue
            if s1 and s1 == s2:
                # the same string drawn twice (a shared axis label on two
                # panels) is invisible when it overlaps, so do not fail
                continue
            ix = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
            iy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
            if ix > smin and iy > smin:
                collisions.append((s1, b1, s2, b2, ix / sc, iy / sc))
    collisions.sort(key=lambda c: -(c[4] * c[5]))

    # text belonging to one panel must not sit on a neighbour's plot area: the
    # panels are close together in a single-column layout, so a y-label or a
    # wide negative tick label can land on the plot beside it
    ax_index = {id(a): i for i, a in enumerate(axes)}
    intrude = []
    for k, b, _c, s, art in items:
        own = getattr(art, "axes", None)
        for ax in axes:
            if ax is own:
                continue
            ab = ax.bbox
            ix = min(b.x1, ab.x1) - max(b.x0, ab.x0)
            iy = min(b.y1, ab.y1) - max(b.y0, ab.y0)
            if ix > intrude_pt * sc and iy > intrude_pt * sc:
                intrude.append((k, s, b, ax_index[id(ax)], ix / sc, iy / sc))
    intrude.sort(key=lambda i: -(i[4] * i[5]))

    on_data = []
    for ax in axes:
        boxes = [(t.get_text(), t.get_window_extent(rend))
                 for t in ax.texts
                 if t.get_visible() and t.get_text().strip()]
        for line in ax.get_lines():
            xy = getattr(line, "get_xydata", lambda: None)()
            if xy is None or not len(xy):
                continue
            pts = ax.transData.transform(xy)
            keep = []
            for txt, bb in boxes:
                hit = sum(1 for px, py in pts
                          if bb.x0 - 4 <= px <= bb.x1 + 4
                          and bb.y0 - 4 <= py <= bb.y1 + 4)
                if hit:
                    on_data.append((txt.replace(chr(10), " ")[:34], hit, len(pts)))
                else:
                    keep.append((txt, bb))
            boxes = keep

    print("[layout audit %s] canvas %.1fx%.1f pt | text artists %d | "
          "outside canvas %d | text collisions %d | over a panel %d"
          % (layout, fb.width / sc, fb.height / sc, len(items),
             len(out_of_box), len(collisions), len(intrude)))
    for kind, b, snip in out_of_box:
        print("   OUTSIDE: %-13s %-28r %s" % (kind, snip, box(b)))
    for s1, b1, s2, b2, ix, iy in collisions[:16]:
        print("   COLLIDE: %-28r %s  <->  %-28r %s  (%.1f x %.1f pt)"
              % (s1, box(b1), s2, box(b2), ix, iy))
    for k, s, b, ai, ix, iy in intrude[:12]:
        print("   OVER-PANEL: %-13s %-24r %s on panel %d (%.1f x %.1f pt)"
              % (k, s, box(b), ai, ix, iy))
    if on_data:
        print("   TEXT-ON-DATA (informative):")
        for txt, hit, tot in sorted(set(on_data)):
            print("      %-36r %d of %d drawn points inside its box"
                  % (txt, hit, tot))
    text_out = [it for it in out_of_box if it[0] != "LEGEND"]
    if text_out or collisions or intrude:
        sys.exit("layout audit failed for %s: %d text box(es) outside the "
                 "canvas, %d text collision(s), %d text box(es) over another "
                 "panel's plot area"
                 % (layout, len(text_out), len(collisions), len(intrude)))
    return len(out_of_box), len(collisions), len(intrude)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=["1x2", "col1sbs"],
                    default="col1sbs",
                    help="col1sbs (default) is the single-column "
                         "manuscript layout; 1x2 is the earlier "
                         "full-width figure*")
    args = ap.parse_args()
    global LAYOUT, BASE_NAME
    LAYOUT = args.layout
    if LAYOUT != "col1sbs":
        # the legacy layout must not silently overwrite the artifact that the
        # manuscript includes
        BASE_NAME += "_" + LAYOUT
    res = load()
    notes = []
    bad = []
    for k, want in EXP_DELTA.items():
        got = [res[e][k]["delta_ratio"] for e, _ in EPS]
        if not all(abs(g - w) < 0.011 for g, w in zip(got, want)):
            bad.append((k, ["%.3f" % g for g in got], want))
    notes.append("self-check baseline delta_ratio medians vs manuscript: %s"
                 % ("OK" if not bad else "MISMATCH %s" % bad))
    notes.append("")
    notes.append("-- every plotted value --")
    notes.append("%-14s %-12s %8s %8s %8s" % ("method", "metric",
                                              *[e for e, _ in EPS]))
    for key, label, _c, _mk, _ls in STYLES:
        for f in ("delta_ratio", "2hop"):
            notes.append("%-14s %-12s %8.4f %8.4f %8.4f"
                         % (label, f, *[res[e][key][f] for e, _ in EPS]))
    notes.append("")
    c8 = max(res["0.8"][k]["2hop"] for k in CHAIN_KEYS)
    b8 = min(res["0.8"][k]["2hop"] for k in BASE_KEYS)
    notes.append("panel (b) at eps=0.8: Chain-NPM %.3f--%.3f vs baselines "
                 "%.3f--%.3f (%s)."
                 % (min(res["0.8"][k]["2hop"] for k in CHAIN_KEYS),
                    c8, b8, max(res["0.8"][k]["2hop"] for k in BASE_KEYS),
                    "Chain-NPM leads" if c8 < b8 else "Chain-NPM trails"))
    if bad:
        with open(COVERAGE_TXT, "w") as fh:
            fh.write("\n".join(notes) + "\n")
        raise SystemExit("refusing to draw: baseline deltas disagree with the "
                         "manuscript; see %s" % COVERAGE_TXT)

    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 8,
        "axes.labelsize": 8.8, "axes.titlesize": 8.4, "xtick.labelsize": 7.6,
        "ytick.labelsize": 7.6, "legend.fontsize": 7.4, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.4, "ytick.major.size": 2.4,
        "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    if LAYOUT != "1x2":
        # a 242 pt-wide figure: the panels are half as wide as in the
        # two-column layout, so the peripheral text steps down as well
        plt.rcParams.update({
            "font.size": 6.4, "axes.labelsize": 6.9,
            "axes.titlesize": 6.7, "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2, "legend.fontsize": 6.2,
            "xtick.major.size": 2.0, "ytick.major.size": 2.0,
        })
    if LAYOUT == "1x2":
        fig, axes = plt.subplots(1, 2, figsize=SIZE[LAYOUT],
                                 gridspec_kw=dict(left=0.092, right=0.995,
                                                  top=0.815,
                                                  bottom=0.165,
                                                  wspace=0.19))
    else:
        # single column: 241.6 pt of canvas holds two panels, so each panel is
        # ~87 pt wide instead of ~209 pt; the peripheral text steps down with
        # it (see the rcParams block above)
        fig, axes = plt.subplots(1, 2, figsize=SIZE[LAYOUT],
                                 gridspec_kw=dict(left=0.160, right=0.995,
                                                  top=0.745,
                                                  bottom=0.195,
                                                  wspace=0.33))
    xpos = [0, 1, 2]
    xt = [e for e, _ in EPS]

    # (a) Δ recovery
    ax = axes[0]
    runs = [res[e]["pertable"]["runs"] for e, _ in EPS]
    lo = [min(r["delta_ratio"] for r in rr) for rr in runs]
    hi = [max(r["delta_ratio"] for r in rr) for rr in runs]
    ax.fill_between(xpos, lo, hi, color="#bfbfbf", alpha=0.35, lw=0,
                    label="PrivMRF run range", zorder=1)
    for key, label, c, mk, ls in STYLES:
        med = [res[e][key]["delta_ratio"] for e, _ in EPS]
        rr = [[r["delta_ratio"] for r in res[e][key]["runs"]] for e, _ in EPS]
        ax.plot(xpos, med, color=c, ls=ls, lw=1.3, marker=mk, markersize=4.4,
                markerfacecolor=c, markeredgecolor="white",
                markeredgewidth=0.5, zorder=5)
        for x, rv in zip(xpos, rr):
            ax.plot([x, x], [min(rv), max(rv)], color=c, lw=1.0, alpha=0.35,
                    zorder=2, solid_capstyle="round")
    ax.axhline(0.0, color="#999999", lw=0.8, ls=":", zorder=0)
    ax.axhline(1.0, color="#999999", lw=0.8, ls="--", zorder=0)
    ax.set_ylim(-0.28, 1.22)
    ax.set_ylabel(r"$\Delta_{\mathrm{syn}}/\Delta_{\mathrm{real}}$",
                  labelpad=1.6)
    ax.set_title("(a) effect recovery",
                 pad=2.6 if LAYOUT == "1x2" else 2.0)

    # (b) 2-hop query RE
    ax = axes[1]
    for key, label, c, mk, ls in STYLES:
        med = [res[e][key]["2hop"] for e, _ in EPS]
        rr = [[r["2hop"] for r in res[e][key]["runs"]] for e, _ in EPS]
        ax.plot(xpos, med, color=c, ls=ls, lw=1.3, marker=mk, markersize=4.4,
                markerfacecolor=c, markeredgecolor="white",
                markeredgewidth=0.5, zorder=5)
        for x, rv in zip(xpos, rr):
            ax.plot([x, x], [min(rv), max(rv)], color=c, lw=1.0, alpha=0.35,
                    zorder=2, solid_capstyle="round")
    ax.set_ylim(0.0, 0.88)
    ax.set_ylabel("2-hop query RE", labelpad=1.6)
    ax.set_title("(b) query accuracy",
                 pad=2.6 if LAYOUT == "1x2" else 2.0)

    for ax in axes:
        ax.set_xticks(xpos)
        ax.set_xticklabels(xt)
        ax.set_xlabel(r"privacy budget $\varepsilon$", labelpad=1.6)
        ax.set_xlim(-0.22, 2.22)
        ax.grid(axis="y", color="#d9d9d9", lw=0.45)
        ax.set_axisbelow(True)

    handles = [Line2D([], [], color=c, marker=mk, ls=ls, lw=1.3,
                      markersize=4.4, markeredgecolor="white",
                      markeredgewidth=0.5, label=label)
               for _k, label, c, mk, ls in STYLES]
    if LAYOUT == "1x2":
        fig.legend(handles=handles, loc="upper center",
                   bbox_to_anchor=(0.5, 1.0), ncol=6, frameon=False,
                   handletextpad=0.3, columnspacing=0.8,
                   borderaxespad=0.0)
    else:
        # six entries do not fit one row across 242 pt: wrap to 3 x 2
        fig.legend(handles=handles, loc="upper center",
                   bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
                   handletextpad=0.3, columnspacing=0.7,
                   handlelength=1.3, borderaxespad=0.0)

    audit_layout(fig, axes, LAYOUT)
    fig.savefig(BASE_NAME + ".pdf")
    fig.savefig(BASE_NAME + ".png", dpi=300)
    with open(COVERAGE_TXT, "w") as fh:
        fh.write("\n".join(notes) + "\n")
    print("written:", BASE_NAME + ".pdf")
    print("written:", COVERAGE_TXT)
    for n in notes[:3]:
        print("#", n)
    print("#", notes[-1])


if __name__ == "__main__":
    main()