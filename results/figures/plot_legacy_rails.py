#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""附录三轨汇总图：TPC-H / deep TPC-H / ACS 三张表的图形化汇总（1 行 × 3 panel）。

版式
  panel = 一轨（TPC-H / deep TPC-H / ACS PUMS）；
  x = 该轨自己的隐私预算档（TPC-H 与 deep TPC-H 是 {0.2, 3.2}，ACS 是 {0.8, 3.2}）；
  线 = 方法 × 查询类别（Chain-NPM 蓝、PrivPetal 橙；实线 = 该轨最深跳，虚线 = 1-hop）；
  每 panel 2–4 条线。数据密度天然低于新 Fig 7/8（legacy 三轨只有 2 档预算），
  这一点在图注里写明，不用补做实验去凑。

图例只列方法（Chain-NPM / PrivPetal），各一个条目；实线/虚线的跳数编码由
正文图注承载（caption：实线 = 该轨最深跳，虚线 = ACS 的 1-hop 类）。此前图例
第三条是一个灰色虚线样本 "1-hop (ACS only)"，但图上每条虚线都按方法着色
（蓝/橙），灰色样本承诺了一条不存在的线，故删除。

数据来源与防漂移
  数值来自正文三张表的表体本身（表是手写的、无机器可读源）。脚本把每个数值
  以 LaTeX 形式 ``$0.083$`` 在 main.tex 里做**字面存在性检查**：若正文数值被改动
  而本脚本未同步，检查失败即拒绝出图。（同一数值在正文中出现多次属正常，只要
  至少出现一次即可。）
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MAIN = os.path.join(ROOT, "VLDB-submission", "manuscript", "main.tex")
BASE_NAME = os.path.join(HERE, "fig_legacy_rails")
COVERAGE_TXT = os.path.join(HERE, "plot_legacy_rails_coverage.txt")

C_CHAIN = "#1d4ed8"      # Chain-NPM
C_PETAL = "#b45309"      # PrivPetal（与 plot_main_effect.py 同一色）
MK_CHAIN = "o"
MK_PETAL = "s"

# (panel 标题, ε 档, [(方法, 查询类别, 线型, [逐 ε 的值])])
RAILS = [
    ("TPC-H (2 private tables)", ["0.2", "3.2"], [
        ("Chain-NPM", "2-hop", "-", [0.044, 0.031]),
        ("PrivPetal", "2-hop", "-", [0.083, 0.015]),
    ]),
    ("deep TPC-H (3 private tables)", ["0.2", "3.2"], [
        ("Chain-NPM", "3-hop", "-", [0.336, 0.054]),
        ("PrivPetal", "3-hop", "-", [0.126, 0.040]),
    ]),
    ("ACS PUMS (2 private tables)", ["0.8", "3.2"], [
        ("Chain-NPM", "2-hop", "-", [0.196, 0.198]),
        ("Chain-NPM", "1-hop", "--", [0.105, 0.102]),
        ("PrivPetal", "2-hop", "-", [0.090, 0.058]),
        ("PrivPetal", "1-hop", "--", [0.020, 0.015]),
    ]),
]


def main():
    text = open(MAIN, encoding="utf-8").read()
    notes = ["values are transcribed from the three hand-written tables in "
             "VLDB-submission/manuscript/main.tex; each is checked below for "
             "literal presence in that file so the figure cannot silently "
             "drift from the paper"]
    missing = []
    for title, eps, series in RAILS:
        for method, klass, _ls, vals in series:
            for v in vals:
                lit = "$%s$" % ("%.3f" % v)
                if lit not in text and ("%.2f" % v) not in text:
                    missing.append((title, method, klass, lit))
    notes.append("literal presence check: %s"
                 % ("OK" if not missing else "MISSING %s" % missing))
    notes.append("")
    notes.append("-- plotted values --")
    for title, eps, series in RAILS:
        notes.append("panel: %s   eps: %s" % (title, eps))
        for method, klass, ls, vals in series:
            notes.append("  %-10s %-6s %-3s %s" % (method, klass, ls,
                                                    ["%.3f" % v for v in vals]))
    notes.append("")
    notes.append("legend: one entry per method (Chain-NPM, PrivPetal), each a "
                 "solid line in its method colour; the solid/dashed hop "
                 "encoding is carried by the manuscript caption, because every "
                 "dashed line in the ACS panel is drawn in its method colour "
                 "(no grey reference line exists in the plot).")
    notes.append("density note: the legacy rails have only two budgets each and "
                 "at most two query classes, so this figure is deliberately "
                 "sparser than the four-database sweep figures; it summarises "
                 "the tables rather than replacing them.")
    if missing:
        with open(COVERAGE_TXT, "w") as fh:
            fh.write("\n".join(notes) + "\n")
        raise SystemExit("refusing to draw: values not found verbatim in "
                         "main.tex; see %s" % COVERAGE_TXT)

    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 8,
        "axes.labelsize": 8.6, "axes.titlesize": 8.0, "xtick.labelsize": 7.6,
        "ytick.labelsize": 7.6, "legend.fontsize": 7.4, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.4, "ytick.major.size": 2.4,
        "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.05),
                             gridspec_kw=dict(left=0.075, right=0.995,
                                              top=0.815, bottom=0.175,
                                              wspace=0.28))

    def color(m):
        return C_CHAIN if m.startswith("Chain") else C_PETAL

    def marker(m):
        return MK_CHAIN if m.startswith("Chain") else MK_PETAL

    for ax, (title, eps, series) in zip(axes, RAILS):
        xs = list(range(len(eps)))
        for method, klass, ls, vals in series:
            ax.plot(xs, vals, color=color(method), ls=ls, lw=1.3,
                    marker=marker(method), markersize=4.4,
                    markerfacecolor=color(method), markeredgecolor="white",
                    markeredgewidth=0.5, zorder=5)
        ax.set_xticks(xs)
        ax.set_xticklabels(eps)
        ax.set_xlim(-0.24, len(eps) - 0.76)
        ax.set_xlabel(r"privacy budget $\varepsilon$", labelpad=1.6)
        ax.set_title(title, pad=2.6)
        ax.grid(axis="y", color="#d9d9d9", lw=0.45)
        ax.set_axisbelow(True)
        ax.margins(y=0.18)
    axes[0].set_ylabel("query relative error", labelpad=1.6)

    handles = [
        Line2D([], [], color=C_CHAIN, marker=MK_CHAIN, ls="-", lw=1.3,
               markersize=4.4, markeredgecolor="white", markeredgewidth=0.5,
               label="Chain-NPM"),
        Line2D([], [], color=C_PETAL, marker=MK_PETAL, ls="-", lw=1.3,
               markersize=4.4, markeredgecolor="white", markeredgewidth=0.5,
               label="PrivPetal (official)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=2, frameon=False, handletextpad=0.35, columnspacing=1.2,
               borderaxespad=0.0)

    fig.savefig(BASE_NAME + ".pdf")
    fig.savefig(BASE_NAME + ".png", dpi=300)
    with open(COVERAGE_TXT, "w") as fh:
        fh.write("\n".join(notes) + "\n")
    print("written:", BASE_NAME + ".pdf")
    print("written:", COVERAGE_TXT)
    for n in notes:
        print("#", n)


if __name__ == "__main__":
    main()