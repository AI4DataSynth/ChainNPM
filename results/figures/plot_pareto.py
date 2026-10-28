#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""成本—效应 Pareto 图（跨双栏 figure*，1 行 × 2 panel）。

(a) 受控基准（同一 runner 目录 / 同一驱动）
    x = 单元格墙钟 `time_s`（对数轴），y = Δ_syn/Δ_real（0 = 跨跳相关性被摧毁，
    1 = 完美恢复）；六方法各三档 ε ∈ {0.8, 3.2, 8.0}，同方法的三个 ε 用细线相连，
    标记填充编码 ε（空心 0.8 / 实心 3.2 / 下半填充 8.0，图例中说明）；
    参考线 y=0（点线）与 y=1（虚线）。

(b) 四个真实库（financial / imdb / instacart / movielens）
    x = 每个预算的墙钟时间（秒，对数轴：六方法跨 5 个数量级）；
    y = 共同基准上的中位保留误差。共同基准 = 每个方法都贡献同样两个植入协议
    （plant_product@t=0.316 与 plant_xor@φ=0.8，各 10 seeds）的逐格
    |syn − real| / |real|，两协议共 20 个值取中位数，再对四个库取中位数。
    水平虚线 y=1.0 是"效应全失"的水平。**标记填充在面板 (b) 里同样只表示 ε**
    （空心 0.8 / 实心 3.2，与面板 (a)、与手稿 Figure 6 同一套约定）；PrivPetal 的
    x 是 A800 GPU wall-clock、其余方法是单 CPU wall-clock 这一口径差异，由它的
    图例标签 "(A800 GPU)" 与图内注释承载，不再用恒空心标记（手稿 Table 2 已披露
    该口径差异并称之为对 Chain-NPM 保守）。
    面板 (b) 的六方法**逐字采用手稿既有调色板**：`results/figures/
    plot_main_effect.py` 的 `META`（即 Figure 6/8/9 的同一套 (颜色, 标记)），
    由下方 B10 现场解析源文件核对，故 PrivPetal 在全篇各图同色。
    两个面板**不共用图例**：各自在图内带自己的图例（面板 (a) 用受控基准调色板
    + ε 填充说明，面板 (b) 用手稿 canonical 调色板），位置由 B8 的遮挡搜索决定。

数据源解析（复用项目既有解析器，不另造口径）
    panel (a)：`tmp/multihop_final_eps{0.8,3.2,8.0}.json`（pertable / lava_asis /
        lava_prop / denorm）与 `tmp/chain_results_eps{0.8,3.2,8.0}.json`
        （chain_direct / chain_mixture）。
    panel (b)：`results/tables/p7_common.py` 的 `iter_files` / `canonical_series` /
        `effect_scalars`，与手稿四库 sweep 图走同一条解析路径——即 PrivMRF 优先
        取 `pertable` 标签的文件，PrivLava 在三个 V 形 schema 上的自然网格就是
        `lavaprop_vshape_natural_*` 跑批。

三处必须显式声明的口径
    1. `time_s`（panel a）是单元格级墙钟的**单个记录值**：`runs` 数组只带指标、
       不带逐 run 计时，所以 x 不是三次运行的中位数（被绘制的中位数是 y）。
       18 格同驱动、同 runner 目录（/home/sheng-xiang/mh_gap）、同一天产出，
       故 x 跨方法可比；文件内无 host 字段，该"同机"是推断而非显式记录。
    2. `runtime_sec`（panel b）是**合成阶段**计时（p4_common.py 的建模+采样，
       不含评测与数据加载；见 tmp/reports/legacy_timing_provenance.md）。
       `total_runtime_sec` 含评测，**不采用**。
    3. PrivLava 的 V 形自然单元格（`lavaprop_vshape_natural_*`）**没有**
       `runtime_sec`，只有 `model_sec` / `synthesize_sec` / `total_runtime_sec`；
       本脚本取 `model_sec + synthesize_sec`（建模+采样，与其它方法同口径），
       逐格记于覆盖报告，不用含评测的 `total_runtime_sec`。

自检（任一不通过则只写覆盖报告并 exit 1，不出图、不裁切）
    A1 四个基线的 Δ 中位数与正文表 tab:gap 一致（±0.011），且这些字面值确实
       出现在 main.tex 中。
    A2 两个 Chain-NPM 变体与 Δ-recovery 陈述一致（±0.006）：
       direct 0.867/0.866/1.028、mixture 0.721/0.939/0.906，并核验
       `$0.72$--$1.03$`、`$0.87$`、`$0.94$` 字面存在。
    A3 每格 delta_ratio 标量 == 其 3 次运行的中位数。
    B1 手稿 tab:datasets 的 8 条计时区间现场解析，且与该库 6 档 ε 逐档中位数之
       min--max 一致（容差 = 印刷精度 1 单位）——手稿一改即失败。
    B2 本图每个 (库, 方法, ε) 派生中位数落在该库印刷区间内（严格与容差两种判定
       都打印）。
    B3 未带 t 标签的植入文件（PrivPetal）必须在元数据里自证协议：plant.mode 与
       plant.target 分别等于该协议的 mode 与 t（product→0.316 / xor→0.8）。
    B4 共同基准的 ground-truth 一致性（**分两层**，因为实测存在两套 GT 产线）：
       B4a 四个共用同一产线的方法（privmrf / chainnpm / privbayes / pbpgm）在同一
           (库, 协议, ε, seed) 上 real 值必须逐格完全一致；
       B4b 六方法在同一单元格上最多只允许两套 real 值，且 PrivLava 必须落在其中
           之一（实测：PrivLava 在 financial 与那四个同线，在三个 V 形库与
           PrivPetal 同线）；
       B4c 两套 real 值之间的相对差不得超过 0.30（实测最大 20.3%，见 movielens
           plant_xor），超过即拒绝出图。
       —— 实测两线并存的原因：四个 p4_common 方法按 cell seed 重新抽 planted GT，
       PrivPetal（以及 V 形库上的 PrivLava）用固定的 plant_seed=42 GT；instacart /
       movielens 上 product 的标定强度 p 也不同。该事实写进覆盖报告，供图注声明
       "共同基准"的共同之处是协议与强度，不是逐格同一份 GT。
    B5 每个 (方法, ε) 至少 3 个库进入中位数。
    B6 所有被绘制的点都落在坐标轴范围内。
    B7 图例/标注不得压住任何被绘制的数据点：对每个图例框，在 300 dpi 下检查
       任何标记（含标记半径）或任何同方法连线线段（Liang--Barsky 精确相交）
       是否落入框内——必须为 0。
    B8 图例位置搜索：每个面板的候选位置逐个评分（框是否完整落在坐标框内、框内
       标记/线段数、框与 y=0/1 参考线是否相交、框与面板内其它文字是否相交、
       所在象限的数据密度），取最小的候选；图例一字不改、不缩写标签。无候选
       满足时**拒绝出图**，并在覆盖报告里给出最小替代方案（面板上方一行图例，
       高度预算允许）。
    B9 像素级复检（同 plot_alloc_sweep3.py 的自检 6 口径）：隐藏图例后按 300 dpi
       重渲染，每个图例框与每个图例文字框内的"series 色"像素必须为 0；再加一条
       与颜色无关的判据——同一张图里图例框内不得有比 y 网格 (#d9d9d9) 更暗的
       像素。两条合起来覆盖面板 (a) 的三个灰系（色相判据因极差约束跳过它们）。
    B10 面板 (b) 实际绘制的 (方法 → 颜色, 标记) 必须逐字等于
       `results/figures/plot_main_effect.py` 的 `META`（现场解析），且等于本
       任务给定的 canonical 字面值；面板 (a) 的调色板保持不变。
    B11 点对拥挤度报告（显示距离 < 标记直径的点对）：改版前后同口径打印，证明
       canonical 调色板没有让拥挤变糟。
    B12 面板 (b) 的 ε 填充编码必须能从 artist 上读回：逐 (方法, ε) 用数据坐标定位
       标记 artist，其 mfc/ms/mew 必须分别是"空心 + EPS_FILL_B"（ε=0.8）与
       "canonical 颜色 + EPS_FILL_B"（ε=3.2）——即**空心在面板 (b) 里只表示
       ε=0.8**，没有任何方法在两个预算上都空心。
    A4 ε=8.0 的半填充样式在离屏画布上实测确实区别于空心/实心（下半部墨量占比）。
    A5 面板 (b) 的 ε 填充编码同样离屏实测：六种 canonical 标记形状都必须满足
       "实心墨量 / 空心墨量 > 1.25"，且两种样式的下半部墨量占比都落在
       [0.35, 0.65]（都没有退化成 ε=8.0 那种下半填充样式）。
"""
import argparse
import json
import math
import os
import re
import statistics
import sys
from collections import defaultdict
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(RESULTS_DIR)
TMP = os.path.join(ROOT, "tmp")
MAIN_TEX = os.path.join(ROOT, "VLDB-submission", "manuscript", "main.tex")
sys.path.insert(0, os.path.join(RESULTS_DIR, "tables"))
import p7_common as P  # noqa: E402

BASE_NAME = os.path.join(HERE, "fig_pareto")
COVERAGE_TXT = os.path.join(HERE, "plot_pareto_coverage.txt")

# ------------------------------------------------------------------ panel (a)
EPS_A = [("0.8", 0.8), ("3.2", 3.2), ("8.0", 8.0)]
SERIES_A = [
    ("pertable",      "PrivMRF",           "#2b2b2b", "o", "multihop_final"),
    ("lava_asis",     "PrivLava (released)", "#6e6e6e", "^", "multihop_final"),
    ("lava_prop",     "PrivLava (Alg. 3)",   "#a8a8a8", "s", "multihop_final"),
    ("denorm",        "PrivMRF (denorm)",  "#1f4e8c", "D", "multihop_final"),
    ("chain_direct",  "Chain-NPM-direct",  "#c0392b", "v", "chain_results"),
    ("chain_mixture", "Chain-NPM-mixture", "#e67e22", "P", "chain_results"),
]
EXP_BASE_DELTA = {
    # 2026-09-21 修复"逐表基线 ε 盲"后按 V2 实测重设，与 main.tex tab:gap 同步。
    # 2026-09-26：tab:gap 只印 PrivLava 的 released code（读者意见：不替他人工作改名），
    # 故 lava_prop（Algorithm 3 配置）的逐格字面量已不在表内——改为核对其在
    # tab:gap caption 里声明的界（$0.06$）；数值一致性检查（A1）照旧。
    "pertable":  ([0.00, -0.02, -0.09],  ["$0.00$", "$-0.02$", "$-0.09$"]),
    "lava_asis": ([0.02, 0.01, 0.03],    ["$0.02$", "$0.01$", "$0.03$"]),
    "lava_prop": ([0.02, -0.004, -0.03], ["$0.06$"]),
    "denorm":    ([0.21, 0.29, 0.21],    ["$0.21$", "$0.29$", "$0.21$"]),
}
EXP_CHAIN_DELTA = {"chain_direct": [0.867, 0.866, 1.028],
                   "chain_mixture": [0.721, 0.939, 0.906]}
CHAIN_STATEMENT_LITERALS = [r"$0.72$--$1.03$", r"$0.87$", r"$0.94$"]
TOL_BASE, TOL_CHAIN, TOL_RUNMED = 0.011, 0.006, 5e-4

# ------------------------------------------------------------------ panel (b)
EPS_B = [("0.8", 0.8), ("3.2", 3.2)]
SERIES_B = ["privmrf", "chainnpm", "privbayes", "pbpgm", "lavaprop", "privpetal"]
SEEDS = list(range(42, 52))
SRC_PREF = {"privmrf": ["pertable", "privmrf"], "chainnpm": ["chainnpm"],
            "privbayes": ["privbayes"], "pbpgm": ["pbpgm"],
            "lavaprop": ["lavaprop", "lavaprop_vshape"], "privpetal": ["privpetal"]}
RUNTIME_FIELDS = [("runtime_sec",), ("model_sec", "synthesize_sec")]
PLANT_T = {"plant_product": "0.316", "plant_xor": "0.8"}
DS_TEXKEY = {"financial": "Financial (PKDD'99)", "imdb": "IMDb",
             "instacart": "Instacart", "movielens": "MovieLens-1M"}
UNIT_SEC = {"s": 1.0, "min": 60.0, "h": 3600.0}
UNIT_PREC = {"s": 1.0, "min": 60.0, "h": 360.0}      # 印刷精度：1 s / 1 min / 0.1 h
MIN_DATASETS = 3
# 面板 (b) 的标记填充编码 ε：与面板 (a) 同一套约定（也是手稿 plot_main_effect.py
# 的约定）——空心 = ε 0.8，实心 = ε 3.2；PrivPetal 不再享有"恒空心"的特殊标记，
# 它的 A800 GPU 口径由图例标签与图内注释承载。标记尺寸对六个方法一律相同。
EPS_FILL_B = {"0.8": dict(fillstyle="none", mew=0.8, ms=4.8),
              "3.2": dict(fillstyle="full", mew=0.5, ms=4.8)}

# ------------------------------------------- 手稿 canonical 调色板（面板 b）
# 面板 (b) 的六个方法必须与手稿既有六方法图（Figure 6/8/9，由
# plot_main_effect.py 与 plot_query_class_sweep.py 产出）同色同标记：颜色承载
# 方法身份，读者要能在全篇追踪 PrivPetal。
MAIN_EFFECT_PY = os.path.join(HERE, "plot_main_effect.py")
META_RE = re.compile(r'^\s*"([A-Za-z0-9_]+)"\s*:\s*\(\s*"([^"]+)"\s*,\s*'
                     r'"(#[0-9a-fA-F]{6})"\s*,\s*"([^"]+)"\s*\)\s*,\s*$', re.M)
# 本任务的 canonical 字面值（与手稿 META 必须逐字一致；手稿一改即失败）
CANON_EXPECTED = {
    "chainnpm":  ("Chain-NPM", "#1d4ed8", "o"),
    "privmrf":   ("PrivMRF",   "#be123c", "v"),
    "privpetal": ("PrivPetal", "#b45309", "s"),
    "privbayes": ("PrivBayes", "#166534", "^"),
    "pbpgm":     ("PB-PGM",    "#0891b2", "P"),
    "lavaprop":  ("PrivLava",  "#7c3aed", "D"),
}
CANON_KEY = {"privmrf": "pertable", "chainnpm": "chainnpm",
             "privpetal": "privpetal", "privbayes": "privbayes",
             "pbpgm": "pbpgm", "lavaprop": "lavaprop"}


def canonical_meta():
    """现场解析 plot_main_effect.py 的 META（不改手稿、不复制粘贴手抄值）。"""
    with open(MAIN_EFFECT_PY, encoding="utf-8") as fh:
        text = fh.read()
    if "META = {" not in text:
        return {}
    blk = text.split("META = {", 1)[1].split("}", 1)[0]
    return {m.group(1): (m.group(2), m.group(3).lower(), m.group(4))
            for m in META_RE.finditer(blk)}


_CANON = canonical_meta()
PALETTE_B = {s: _CANON.get(k) for s, k in CANON_KEY.items()}
DISPLAY_B = {s: (PALETTE_B[s][0] if PALETTE_B[s] else s) for s in SERIES_B}


# --------------------------------------------------------------------- utils
def f4(x):
    return "%.4f" % x


def med(vals):
    return float(statistics.median(vals))


def series_of(d):
    return P.canonical_series(d.get("method"))


def base(name):
    return os.path.basename(name)


def natural_src_tag(name):
    m = re.match(r"^(.*?)_natural_eps", base(name))
    return m.group(1) if m else base(name)


def runtime_of(d):
    """按 RUNTIME_FIELDS 顺序取第一个可用字段 → (值, 字段名)。"""
    for fields in RUNTIME_FIELDS:
        vals = [d.get(f) for f in fields]
        if all(isinstance(v, (int, float)) and v > 0 for v in vals):
            return float(sum(vals)), "+".join(fields)
    return None, None


def is_degraded(d):
    return bool(d.get("degraded")) or P.hop2_re(d) is None


def pick_natural(rows):
    ok = [d for d in rows if not is_degraded(d)]
    pool = ok or rows
    pool = sorted(pool, key=lambda d: (
        SRC_PREF[series_of(d)].index(natural_src_tag(d["_file"]))
        if natural_src_tag(d["_file"]) in SRC_PREF[series_of(d)] else 99,
        base(d["_file"])))
    return pool[0], (not ok)


def tag_t(path):
    m = re.search(r"_t([0-9.]+)_eps", base(path))
    return m.group(1) if m else None


def describe_sources(names):
    """把一组源文件名压成"前缀 + 显式 seed 集合 + 后缀"的可审计写法。"""
    names = sorted(set(names))
    pat = re.compile(r"^(?P<pre>.*?)_seed(?P<s>\d+)(?P<post>.*)$")
    ms = [pat.match(n) for n in names]
    if all(ms):
        pres = {m.group("pre") for m in ms}
        posts = {m.group("post") for m in ms}
        if len(pres) == 1 and len(posts) == 1:
            seeds = sorted(int(m.group("s")) for m in ms)
            return "%s_seed{%s}%s" % (pres.pop(),
                                      ",".join(str(s) for s in seeds), posts.pop())
    return ";".join(names)


def load_index(src=None):
    """released campaign records partitioned into natural/plant dicts.

    src='szquota' swaps only the panel-(b) retention source for chainnpm on
    the three V-shaped databases: the plant cells come from the szquota
    campaign grid (runtime x-values stay released, so the B1 runtime
    cross-check against tab:datasets keeps its meaning); financial and the
    other series stay released.
    """
    import szquota_common
    nat, plant = {}, {}
    if src == 'szquota':
        nat_it = P.iter_files()
        plant_it = [d for d in szquota_common.szquota_plant_records()] + [
            d for d in P.iter_files()
            if not (d["_dataset"] in ("imdb", "instacart", "movielens")
                    and d.get("mode") in PLANT_T
                    and series_of(d) == "chainnpm")]
    else:
        # 注意：P.iter_files() 是生成器——两个名字必须各自物化一次，否则
        # nat 循环会把它耗尽、plant 循环看到空序列（2026-09-26 修，此前
        # 默认 arm 因此永远拿不到 panel (b) 的 y 值，B5/B3 必然失败）
        nat_it = list(P.iter_files())
        plant_it = nat_it
    for d in nat_it:
        series = series_of(d)
        if series is None:
            continue
        name = base(d["_file"])
        mode = d.get("mode")
        if mode == "natural" and "_natural_eps" in name:
            nat.setdefault((d["_dataset"], series, d.get("eps"), d.get("seed")),
                           []).append(d)
    for d in plant_it:
        series = series_of(d)
        if series is None:
            continue
        name = base(d["_file"])
        mode = d.get("mode")
        if mode in PLANT_T and "_plant_" in name:
            plant.setdefault((d["_dataset"], series, mode, d.get("eps"),
                              d.get("seed")), []).append(d)
    return nat, plant


# ------------------------------------------------------------ panel (a) 数据
def panel_a(notes, fails):
    res, src = {}, {}
    for e_str, _ in EPS_A:
        for tag in ("multihop_final", "chain_results"):
            p = os.path.join(TMP, "%s_eps%s.json" % (tag, e_str))
            for k, v in json.load(open(p))["results"][e_str].items():
                res.setdefault(e_str, {})[k] = v
                src[(k, e_str)] = os.path.relpath(p, ROOT)
    notes.append("-- panel (a): 18 cells (y = median over 3 runs; x = one cell-level clock) --")
    notes.append("每格 y 是 `config.runs = 3` 次运行的中位数（下方列出三次的取值）；"
                 "x 是驱动记录的单元格墙钟 `time_s`——`runs` 数组只带指标、不带逐 run "
                 "计时，故 x 是单值而非三次的中位数。18 格由同一 campaign 在同一天"
                 "（文件 mtime 2026-08-18）经同一 runner 目录 "
                 "/home/sheng-xiang/mh_gap 产出；文件内**没有** host 字段，因此"
                 "“同机”是从 runner 目录/产出时间推断的，不是显式记录。")
    notes.append("%-16s %-5s %9s %13s  %-30s %s"
                 % ("method", "eps", "time_s", "delta_ratio", "source file",
                    "run delta_ratio (3 runs)"))
    for key, label, _c, _mk, _tag in SERIES_A:
        for e_str, _ in EPS_A:
            cell = res[e_str][key]
            runs = [r["delta_ratio"] for r in cell["runs"]]
            notes.append("%-16s %-5s %9.4f %13.6f  %-30s %s"
                         % (label, e_str, cell["time_s"], cell["delta_ratio"],
                            src[(key, e_str)],
                            " ".join("%.6f" % v for v in runs)))
            if abs(med(runs) - cell["delta_ratio"]) > TOL_RUNMED:
                fails.append("A3 %s eps=%s: scalar %.6f != median(runs) %.6f"
                             % (label, e_str, cell["delta_ratio"], med(runs)))
    notes.append("")
    text = open(MAIN_TEX, encoding="utf-8").read()

    def lit_present(num):
        """表格里的数值可能被 \\mathbf{} 包住（denorm 行），两种字面形式都接受。"""
        return ("$%s$" % num in text) or ("$\\mathbf{%s}$" % num in text)

    bad, lit_missing = [], []
    for key, label, _c, _mk, _tag in SERIES_A[:4]:
        want, lits = EXP_BASE_DELTA[key]
        for (e_str, _e), w in zip(EPS_A, want):
            g = res[e_str][key]["delta_ratio"]
            if abs(g - w) > TOL_BASE:
                bad.append("%s eps=%s: %.4f vs %.3f" % (label, e_str, g, w))
        for lit in lits:
            if not lit_present(lit.strip("$")):
                lit_missing.append(lit)
    notes.append("A1 baseline Δ medians vs manuscript tab:gap (tol %.3f): %s"
                 % (TOL_BASE, "OK" if not bad else "MISMATCH %s" % bad))
    notes.append("A1 literal presence of the tab:gap values in main.tex: %s"
                 % ("OK" if not lit_missing else "MISSING %s" % lit_missing))
    bad2, lit2 = [], []
    for key, want in EXP_CHAIN_DELTA.items():
        for (e_str, _e), w in zip(EPS_A, want):
            g = res[e_str][key]["delta_ratio"]
            if abs(g - w) > TOL_CHAIN:
                bad2.append("%s eps=%s: %.4f vs %.3f" % (key, e_str, g, w))
    for lit in CHAIN_STATEMENT_LITERALS:
        if lit not in text:
            lit2.append(lit)
    notes.append("A2 Chain-NPM variants vs the Δ-recovery statement (tol %.3f): %s"
                 % (TOL_CHAIN, "OK" if not bad2 else "MISMATCH %s" % bad2))
    notes.append("A2 literal presence of that statement in main.tex: %s"
                 % ("OK" if not lit2 else "MISSING %s" % lit2))
    notes.append("A3 every cell's delta_ratio == median of its 3 runs: %s"
                 % ("OK" if not any(f.startswith("A3") for f in fails) else "FAIL"))
    for b in bad:
        fails.append("A1 " + b)
    for b in bad2:
        fails.append("A2 " + b)
    for lit in lit_missing + lit2:
        fails.append("A1/A2 literal missing from main.tex: %s" % lit)
    notes.append("")
    return res


# ------------------------------------------------- 解析 tab:datasets 计时区间
RANGE_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)--([0-9]+(?:\.[0-9]+)?)\s*"
                      r"(?:\\penalty0)?\\?,\s*(s|min|h)\b")


def parse_table_ranges(notes, fails):
    text = open(MAIN_TEX, encoding="utf-8").read()
    try:
        block = text.split("\\label{tab:datasets}")[1].split("\\end{tabular}")[0]
    except IndexError:
        fails.append("B1 cannot locate the tab:datasets block in main.tex")
        return {}
    out = {}
    notes.append("-- B1: tab:datasets time windows parsed from main.tex --")
    # 表格一行可能跨多个源码行（标签在第一行、数字在下一行），所以先按行终止符
    # `\\` 合并成逻辑行再匹配
    logical = block.split("\\\\")
    for ds, texkey in DS_TEXKEY.items():
        row = None
        pat = re.compile(r"^\s*" + re.escape(texkey) + r"\s*&")
        for ln in logical:
            if pat.search(ln):
                row = ln
                break
        if row is None:
            fails.append("B1 no tab:datasets logical row for %s" % texkey)
            continue
        hits = RANGE_RE.findall(row)
        if len(hits) != 2:
            fails.append("B1 %s: expected 2 time ranges on the row, got %d %s"
                         % (texkey, len(hits), hits))
            continue
        for series, idx in (("chainnpm", 0), ("privpetal", 1)):
            lo, hi, unit = hits[idx]
            out[(ds, series)] = (float(lo) * UNIT_SEC[unit],
                                 float(hi) * UNIT_SEC[unit], unit,
                                 "%.4g--%.4g\\,%s" % (float(lo), float(hi), unit))
            notes.append("  parsed %-10s %-9s %-14s = %10.3f -- %10.3f s"
                         % (ds, series, out[(ds, series)][3],
                            out[(ds, series)][0], out[(ds, series)][1]))
    notes.append("")
    return out


# ------------------------------------------------------------ panel (b) 数据
def panel_b(nat_rows, plant_rows, notes, fails):
    # ---------- x ----------
    xcells, xdrop = {}, {}
    for ds in P.DATASETS:
        for series in SERIES_B:
            for e_str, e in EPS_B:
                per, bad = {}, []
                for s in SEEDS:
                    rows = nat_rows.get((ds, series, e, s))
                    if not rows:
                        bad.append("seed%d: no natural cell" % s)
                        continue
                    d, was_deg = pick_natural(rows)
                    val, field = runtime_of(d)
                    if val is None:
                        bad.append("seed%d: runtime missing/0 in %s"
                                   % (s, base(d["_file"])))
                        continue
                    per[s] = (val, field, base(d["_file"]), was_deg)
                if bad or len(per) != len(SEEDS):
                    xdrop[(ds, series, e_str)] = bad or ["seed count != 10"]
                else:
                    xcells[(ds, series, e_str)] = per

    # ---------- y ----------
    # 未带 t 标签的文件（PrivPetal）必须在元数据里自证协议：
    # plant.mode == 协议 mode 且 plant.target == 该协议的 t
    def protocol_ok(d, mode):
        pl = d.get("plant") or {}
        want_mode = mode.split("_", 1)[1]
        if pl.get("mode") != want_mode:
            return False, "plant.mode=%r != %r" % (pl.get("mode"), want_mode)
        tgt = pl.get("target")
        if not isinstance(tgt, (int, float)) or abs(float(tgt) - float(PLANT_T[mode])) > 1e-9:
            return False, "plant.target=%r != %s" % (tgt, PLANT_T[mode])
        return True, "plant.target=%s" % tgt

    untagged_used = defaultdict(set)
    ycells, ydrop = {}, {}
    for ds in P.DATASETS:
        for series in SERIES_B:
            for e_str, e in EPS_B:
                for mode in ("plant_product", "plant_xor"):
                    per, bad = {}, []
                    t_want = PLANT_T[mode]
                    for s in SEEDS:
                        rows = plant_rows.get((ds, series, mode, e, s))
                        if not rows:
                            bad.append("seed%d: no cell" % s)
                            continue
                        tagged = [d for d in rows if tag_t(d["_file"]) == t_want]
                        if tagged:
                            d = sorted(tagged, key=lambda x: base(x["_file"]))[0]
                        else:
                            rest = [d for d in rows if tag_t(d["_file"]) is None]
                            if not rest:
                                bad.append("seed%d: no cell at t=%s" % (s, t_want))
                                continue
                            d = sorted(rest, key=lambda x: base(x["_file"]))[0]
                            ok, why = protocol_ok(d, mode)
                            if not ok:
                                bad.append("seed%d: untagged file fails the protocol "
                                           "identity check (%s)" % (s, why))
                                continue
                            untagged_used[(ds, series, mode, e_str)].add(
                                (base(d["_file"]), why))
                        sc = P.effect_scalars(d)
                        tgt = "delta" if ds == "financial" else "contrast"
                        if tgt not in sc:
                            bad.append("seed%d: effect.%s unavailable" % (s, tgt))
                            continue
                        real, syn = sc[tgt]
                        if not real:
                            bad.append("seed%d: real effect is 0" % s)
                            continue
                        per[s] = (abs(syn - real) / abs(real), real, syn,
                                  base(d["_file"]))
                    if bad or len(per) != len(SEEDS):
                        ydrop[(ds, series, mode, e_str)] = bad or ["seed count != 10"]
                    else:
                        ycells[(ds, series, mode, e_str)] = per

    # ---------- 覆盖报告：逐格数值 ----------
    notes.append("-- panel (b) x cells: every value entering the x medians --")
    notes.append("field 说明：`runtime_sec` = 合成阶段计时（建模+采样，不含评测）；"
                 "PrivLava 的 V 形单元格无 `runtime_sec`，取 "
                 "`model_sec+synthesize_sec`（同口径），不用含评测的 "
                 "`total_runtime_sec`。PrivPetal 的 `runtime_sec` 是该方法的自身 "
                 "总 wall clock（A800 GPU），与手稿 Table 2 口径一致。")
    for ds in P.DATASETS:
        for series in SERIES_B:
            for e_str, _e in EPS_B:
                key = (ds, series, e_str)
                if key not in xcells:
                    notes.append("[x] %-10s %-10s eps=%-4s DROPPED (%s)"
                                 % (ds, series, e_str,
                                    "; ".join(xdrop.get(key, ["?"]))))
                    continue
                per = xcells[key]
                notes.append("[x] %-10s %-10s eps=%-4s n=%d field=%s src=%s"
                             % (ds, series, e_str, len(per),
                                "|".join(sorted({v[1] for v in per.values()})),
                                describe_sources([v[2] for v in per.values()])))
                notes.append("      per-seed(s): %s   → dataset median %.4f s"
                             % (" ".join("%d=%.3f" % (s, per[s][0])
                                         for s in sorted(per)),
                                med([v[0] for v in per.values()])))
    notes.append("")
    notes.append("-- panel (b) y cells: every value entering the y medians --")
    notes.append("retention error = |syn − real| / |real|；financial 取 "
                 "effect.delta_*，三个 V 形库取 effect.contrast_*（不用 phi_*）。"
                 "纳入中位数的是两个协议 × 10 seeds 共 20 个逐 seed 值。")
    for ds in P.DATASETS:
        for series in SERIES_B:
            for e_str, _e in EPS_B:
                for mode in ("plant_product", "plant_xor"):
                    key = (ds, series, mode, e_str)
                    if key not in ycells:
                        notes.append("[y] %-10s %-10s %-13s eps=%-4s DROPPED (%s)"
                                     % (ds, series, mode, e_str,
                                        "; ".join(ydrop.get(key, ["?"]))))
                        continue
                    per = ycells[key]
                    notes.append("[y] %-10s %-10s %s@t=%s eps=%-4s n=%d src=%s"
                                 % (ds, series, mode, PLANT_T[mode], e_str,
                                    len(per),
                                    describe_sources([v[3] for v in per.values()])))
                    notes.append("      real=%s  per-seed syn/retention: %s"
                                 % ("/".join(sorted({("%.9g" % v[1])
                                                     for v in per.values()})),
                                    " ".join("%d=%.6f/%.5f" % (s, per[s][2], per[s][0])
                                             for s in sorted(per))))
                    notes.append("      protocol median retention = %.6f"
                                 % med([v[0] for v in per.values()]))
    # ---------- 汇总 ----------
    xpts, ypts, entered = {}, {}, {}
    for series in SERIES_B:
        for e_str, _e in EPS_B:
            xv, xds, yv, yds = [], [], [], []
            for ds in P.DATASETS:
                if (ds, series, e_str) in xcells:
                    xv.append(med([v[0] for v in xcells[(ds, series, e_str)].values()]))
                    xds.append(ds)
                pool = []
                for mode in ("plant_product", "plant_xor"):
                    per = ycells.get((ds, series, mode, e_str))
                    if per:
                        pool += [v[0] for v in per.values()]
                if pool:
                    yv.append(med(pool))
                    yds.append(ds)
            if len(xv) < MIN_DATASETS or len(yv) < MIN_DATASETS:
                fails.append("B5 %s eps=%s: %d datasets for x / %d for y "
                             "(need >= %d)" % (series, e_str, len(xv), len(yv),
                                               MIN_DATASETS))
                continue
            xpts[(series, e_str)] = med(xv)
            ypts[(series, e_str)] = med(yv)
            entered[(series, e_str)] = (xds, yds, xv, yv)
    notes.append("")
    notes.append("-- panel (b) plotted points --")
    notes.append("%-10s %-5s %11s %13s  datasets" % ("method", "eps", "x (s)", "y"))
    for series in SERIES_B:
        for e_str, _e in EPS_B:
            if (series, e_str) not in xpts:
                notes.append("%-10s %-5s %11s %13s  n/a"
                             % (DISPLAY_B[series], e_str, "-", "-"))
                continue
            xds, yds, xv, yv = entered[(series, e_str)]
            notes.append("%-10s %-5s %11.3f %13.6f  x:%s | y:%s"
                         % (DISPLAY_B[series], e_str, xpts[(series, e_str)],
                            ypts[(series, e_str)], ",".join(xds), ",".join(yds)))
            notes.append("           x per dataset (s): %s"
                         % " ".join("%s=%.3f" % t for t in zip(xds, xv)))
            notes.append("           y per dataset    : %s"
                         % " ".join("%s=%.6f" % t for t in zip(yds, yv)))
    # 简报预期的掉格情形
    lav_ml = [(e_str, xdrop.get(("movielens", "lavaprop", e_str)))
              for e_str, _e in EPS_B]
    notes.append("")
    notes.append("掉格记录：%s（简报预期 PrivLava/movielens 因 runtime 缺失或 0 掉格；"
                 "实测未掉格——该库 PrivLava 的自然网格是 "
                 "`lavaprop_vshape_natural_*_r1` 跑批，其 `runtime_sec` 虽为空但 "
                 "`model_sec+synthesize_sec` 可用；而 movielens 的平坦文件 "
                 "`lavaprop_natural_eps3.20_seed42.json` 是 degraded 单条记录"
                 "（`runtime_sec` ≈ 1e-5），已被去降级规则排除）"
                 % ", ".join("eps=%s: %s" % (e, "ok" if r is None else r)
                             for e, r in lav_ml))
    # ---------- B1/B2 ----------
    notes.append("")
    ranges = parse_table_ranges(notes, fails)
    notes.append("-- B1 provenance: recomputed 6-budget min--max vs printed window --")
    for (ds, series) in sorted(ranges):
        meds = {}
        for e in (0.1, 0.2, 0.4, 0.8, 1.6, 3.2):
            per_seed = []
            for s in SEEDS:
                rows = nat_rows.get((ds, series, e, s))
                if not rows:
                    continue
                d, _deg = pick_natural(rows)
                val, _f = runtime_of(d)
                if val is not None:
                    per_seed.append(val)
            if per_seed:
                meds[e] = med(per_seed)
        lo_r, hi_r = min(meds.values()), max(meds.values())
        lo_p, hi_p, unit, lit = ranges[(ds, series)]
        tol = UNIT_PREC[unit]
        good = abs(lo_r - lo_p) <= tol and abs(hi_r - hi_p) <= tol
        notes.append("B1 %-10s %-9s printed %-14s (%.1f--%.1f s) vs recomputed "
                     "%.1f--%.1f s over %d budgets; tol %.1f s → %s"
                     % (ds, series, lit, lo_p, hi_p, lo_r, hi_r, len(meds), tol,
                        "OK" if good else "MISMATCH"))
        if not good:
            fails.append("B1 %s/%s: recomputed %.1f--%.1f s vs printed %s"
                         % (ds, series, lo_r, hi_r, lit))
        if len(meds) != 6:
            fails.append("B1 %s/%s: only %d of 6 budgets found for the provenance "
                         "check" % (ds, series, len(meds)))
    notes.append("")
    notes.append("-- B2: plotted per-dataset medians vs the parsed printed window --")
    for (ds, series) in sorted(ranges):
        lo_p, hi_p, unit, lit = ranges[(ds, series)]
        tol = UNIT_PREC[unit]
        for e_str, _e in EPS_B:
            if (ds, series, e_str) not in xcells:
                continue
            v = med([x[0] for x in xcells[(ds, series, e_str)].values()])
            strict = lo_p <= v <= hi_p
            within = lo_p - tol <= v <= hi_p + tol
            notes.append("B2 %-10s %-9s eps=%-4s plotted %10.3f s vs printed "
                         "%.1f--%.1f s → strict=%s, within-1-printed-unit=%s "
                         "(Δlo %+.2f s, Δhi %+.2f s)"
                         % (ds, series, e_str, v, lo_p, hi_p, strict, within,
                            v - lo_p, v - hi_p))
            if not within:
                fails.append("B2 %s/%s eps=%s: %.3f s outside the printed %s "
                             "(tol %.1f s)" % (ds, series, e_str, v, lit, tol))
    # ---------- B3 ----------
    notes.append("")
    notes.append("-- B3: untagged plant files accepted, and their protocol identity --")
    any_untagged = False
    for (ds, series, mode, e_str) in sorted(untagged_used):
        files = sorted(untagged_used[(ds, series, mode, e_str)])
        any_untagged = True
        notes.append("B3 %-10s %-10s %-13s eps=%-4s %d untagged file(s) accepted: %s "
                     "(each self-declares the protocol; e.g. %s)"
                     % (ds, series, mode, e_str, len(files), files[0][0], files[0][1]))
    if not any_untagged:
        notes.append("B3 no untagged plant file was used; check vacuous")
    # ---------- B4 ----------
    cell_dist, eps_inconsistent, worst_spread, worst_dy = [], [], {}, {}
    for ds in P.DATASETS:
        for mode in ("plant_product", "plant_xor"):
            for s in SEEDS:
                by_eps = {}
                for e_str, _e in EPS_B:
                    by = {}
                    for series in SERIES_B:
                        per = ycells.get((ds, series, mode, e_str))
                        if per and s in per:
                            by[series] = per[s][1]
                    by_eps[e_str] = by
                    if by:
                        vals = sorted(set(round(v, 12) for v in by.values()))
                        if len(vals) > 2:
                            cell_dist.append((ds, mode, e_str, s, vals))
                        if len(vals) == 2 and min(vals) > 0:
                            sp = (max(abs(vals[0]), abs(vals[1]))
                                  - min(abs(vals[0]), abs(vals[1]))) / \
                                min(abs(vals[0]), abs(vals[1]))
                            if sp > worst_spread.get((ds, mode), (0, ""))[0]:
                                worst_spread[(ds, mode)] = (sp, e_str)
                a, b = by_eps.get("0.8", {}), by_eps.get("3.2", {})
                for k in sorted(set(a) & set(b)):
                    if abs(a[k] - b[k]) > 1e-9:
                        eps_inconsistent.append((ds, mode, s, k, a[k], b[k]))
    # B4d：换用"多数派 real"作分母后 y 的移动，以及排序是否翻转
    def alt_y(series, e_str, ref_mode):
        vals = []
        for ds in P.DATASETS:
            pool = []
            for mode in ("plant_product", "plant_xor"):
                per = ycells.get((ds, series, mode, e_str))
                if not per:
                    continue
                from collections import Counter
                ref = Counter()
                for s2 in SERIES_B:
                    p2 = ycells.get((ds, s2, mode, e_str))
                    if p2:
                        for v in p2.values():
                            ref[round(v[1], 9)] += 1
                if ref_mode == "majority":
                    r0 = ref.most_common(1)[0][0]
                    pool += [abs(v[2] - v[1]) / abs(r0) for v in per.values()]
                else:
                    pool += [v[0] for v in per.values()]
            if pool:
                vals.append(med(pool))
        return med(vals) if vals else None

    for e_str, _e in EPS_B:
        for series in SERIES_B:
            if (series, e_str) not in ypts:
                continue
            alt = alt_y(series, e_str, "majority")
            if alt is not None:
                dy = abs(alt - ypts[(series, e_str)])
                if dy > worst_dy.get((series, e_str), (0, None))[0]:
                    worst_dy[(series, e_str)] = (dy, alt)
    notes.append("")
    notes.append("B4a 每 (库,协议,ε,seed) 的六方法 real 值最多两套（四产线方法按 seed "
                 "重抽 GT，PrivPetal 用固定 plant_seed=42 GT，V 形库上的 PrivLava "
                 "与 PrivPetal 同线；movielens plant_xor 上 chainnpm 也落在第二线）："
                 "%s" % ("OK" if not cell_dist else ">2 套: %s" % cell_dist[:3]))
    notes.append("B4a 同一 (库,方法,协议,seed) 的 real 值在 ε=0.8 与 3.2 之间一致: %s"
                 % ("OK" if not eps_inconsistent else "MISMATCH %s"
                    % eps_inconsistent[:3]))
    notes.append("B4b 两套 real 的相对差（cap 0.30）:")
    for (ds, mode), (sp, e_str) in sorted(worst_spread.items()):
        notes.append("     %-10s %-13s max rel spread = %6.3f%%  (at eps=%s)"
                     % (ds, mode, 100 * sp, e_str))
    notes.append("B4c 改用'多数派 real'作分母后各方法 y 的位移（cap 0.30）:")
    for (series, e_str), (dy, alt) in sorted(worst_dy.items()):
        notes.append("     %-10s eps=%-4s |Δy| = %.4f  (y_own=%.4f → y_majority=%.4f)"
                     % (series, e_str, dy, ypts[(series, e_str)], alt))
    alt_order = {}
    for e_str, _e in EPS_B:
        pts = {s: alt_y(s, e_str, "majority") for s in SERIES_B
               if (s, e_str) in ypts}
        pts = {k: v for k, v in pts.items() if v is not None}
        alt_order[e_str] = pts
        notes.append("B4d eps=%s 在'多数派 real'口径下的 y 排序: %s"
                     % (e_str, " < ".join("%s=%.4f" % (k, pts[k])
                                          for k in sorted(pts, key=pts.get))))
    notes.append("B4  说明：`共同基准`的共同之处是**协议与强度**（product@t=0.316、"
                 "xor@φ=0.8，各 10 seeds），不是逐格同一份 GT——四个 p4_common 方法"
                 "逐 seed 重抽 planted GT（financial/imdb/instacart/movielens 上"
                 "`plant.p` 也可能不同：instacart 0.609375 vs 0.625、movielens "
                 "0.99609375 vs 0.9375），PrivPetal 的植入文件自带 `plant_seed: 42`。"
                 "每格保留误差按手稿定义以该格自身的 Δ_real 为基准；上表给出改用"
                 "统一分母后的位移与排序。")
    if cell_dist:
        fails.append("B4a more than two real-effect variants in a cell: %s"
                     % cell_dist[:3])
    if eps_inconsistent:
        fails.append("B4a real effect depends on eps: %s" % eps_inconsistent[:3])
    for (ds, mode), (sp, e_str) in sorted(worst_spread.items()):
        if sp > 0.30:
            fails.append("B4b %s/%s: real-effect spread %.1f%% exceeds the 0.30 cap"
                         % (ds, mode, 100 * sp))
    for (series, e_str), (dy, alt) in sorted(worst_dy.items()):
        if dy > 0.30:
            fails.append("B4c %s eps=%s: |Δy| under the majority-real denominator "
                         "is %.3f (> 0.30 cap)" % (series, e_str, dy))
    for e_str, pts in alt_order.items():
        if pts and ("chainnpm" in pts):
            best = min(pts, key=pts.get)
            if best != "chainnpm":
                fails.append("B4d eps=%s: under the majority-real denominator the "
                             "lowest retention error is %s, not Chain-NPM"
                             % (e_str, best))
    if not untagged_used:
        fails.append("B3 no untagged plant file was used: the protocol-identity "
                     "check did not run")
    # ---------- PrivPetal 支配性：事实陈述 ----------
    notes.append("")
    for e_str, _e in EPS_B:
        if ("privpetal", e_str) not in xpts:
            continue
        px, py = xpts[("privpetal", e_str)], ypts[("privpetal", e_str)]
        dom = []
        for series in SERIES_B:
            if series == "privpetal" or (series, e_str) not in xpts:
                continue
            x2, y2 = xpts[(series, e_str)], ypts[(series, e_str)]
            if x2 < px and y2 <= py:
                dom.append("%s x=%.1f s y=%.4f" % (DISPLAY_B[series], x2, y2))
        notes.append("FACT eps=%s: PrivPetal x=%.1f s, y=%.6f → faster methods with "
                     "y <= PrivPetal's: %s"
                     % (e_str, px, py,
                        "; ".join(dom) if dom else "NONE (not dominated on y)"))
    return xpts, ypts


# --------------------------------------------------------------------- 绘图
# 面板 (a) 的三档 ε 标记样式（形状由方法决定，填充由 ε 决定）
EPS_FILL = {
    "0.8": dict(fillstyle="none", mfc="none", mew=0.8, ms=4.4),
    "3.2": dict(fillstyle="full", mew=0.5, ms=4.4),
    "8.0": dict(fillstyle="bottom", mew=0.6, ms=5.0),
}


def marker_fill_check(notes, fails):
    """验证 ε=8.0 的 fillstyle='bottom' 真的渲染成"下半填充"，而非退化成
    hollow/filled：在离屏画布上量出每个样式落在标记下半部的墨量占比。"""
    import numpy as np
    fig = plt.figure(figsize=(1.4, 0.5), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1)
    order = ["0.8", "3.2", "8.0"]
    for i, k in enumerate(order):
        st = dict(EPS_FILL[k])
        st.setdefault("mec", "black")
        if k == "3.2":
            st["mec"] = "white"
        ax.plot([0.5 + i], [0.5], marker="o", ls="none", color="black", **st)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(int)
    W = buf.shape[1]
    out = {}
    for i, k in enumerate(order):
        x0 = int(W * (0.5 + i - 0.03) / 3)
        x1 = int(W * (0.5 + i + 0.03) / 3)
        dark = (buf[:, x0:x1].sum(axis=2) < 400)
        ys = np.where(dark.any(axis=1))[0]
        if not len(ys):
            out[k] = (0, 0.0)
            continue
        cy = int(0.5 * (ys.min() + ys.max()))
        up, lo = dark[:cy].sum(), dark[cy:].sum()
        out[k] = (int(dark.sum()), round(lo / max(1, up + lo), 3))
    plt.close(fig)
    notes.append("A4 marker-fill encoding measured offscreen "
                 "(dark px, share of ink in the lower half): "
                 + "; ".join("eps=%s %s" % (k, out[k]) for k in order))
    ok = (out["8.0"][1] > 0.70 and out["0.8"][1] < 0.62
          and out["3.2"][1] < 0.62 and out["0.8"][0] < out["3.2"][0])
    notes.append("A4 ε=8.0 half-filled style is genuinely distinct from hollow "
                 "and filled: %s" % ("OK" if ok else "FAIL %s" % out))
    if not ok:
        fails.append("A4 the eps=8.0 marker style is not distinguishable: %s" % out)
    notes.append("")


FIGSIZE = (7.05, 2.5)          # 宽度 = \textwidth（固定）；高度未增长（顶部图例
#                                带被移除，腾出的空间给了坐标框，见覆盖报告）
GS_AFTER = dict(left=0.075, right=0.995, top=0.905, bottom=0.175, wspace=0.30)
GS_BEFORE = dict(left=0.075, right=0.995, top=0.80, bottom=0.175, wspace=0.30)
# 被接受版本归档 B8 行里的两句计数（100 dpi 下 8 显示 px ≡ 5.76 pt），供 B11 复算
# "改版前几何"时交叉验证
ARCHIVED_B8 = {"a": 17, "b": 7}
TITLE_PAD = 2.6


def _measure_marker(marker, colour, ms, mew, kind):
    """离屏画一个标记，量 (核心墨 px, 中心 3x3 墨 px, 下半部墨量占比)。

    kind: "hollow"（ε=0.8：mfc=none、边=本色、mew=0.8）/ "filled"（ε=3.2：
    mfc=本色、边=白、mew=0.5）/ "baseline"（同形状、同尺寸的纯实心，用作该形状
    自身的"下半部墨量占比"几何基线）。核心墨 = 与本色 RGB 距离 <= 0.25 的像素，
    只算墨的主体，排除抗锯齿边缘与白色描边。
    """
    fig = plt.figure(figsize=(0.5, 0.5), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if kind == "baseline":
        st = dict(mfc=colour, mec=colour, mew=mew, ms=ms, fillstyle="full")
    elif kind == "hollow":
        st = dict(mfc="none", mec=colour, mew=mew, ms=ms, fillstyle="none")
    else:
        st = dict(mfc=colour, mec="white", mew=mew, ms=ms, fillstyle="full")
    ax.plot([0.5], [0.5], marker=marker, ls="none", color=colour, **st)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(float) / 255.0
    H = buf.shape[0]
    px, py = ax.transData.transform((0.5, 0.5))
    r = (ms / 2.0 + mew + 1.0) * fig.dpi / 72.0
    x0, r0 = int(px - r), int(H - py - r)
    sub = buf[r0:int(H - py + r) + 1, x0:int(px + r) + 1, :]
    core = np.linalg.norm(sub - np.array(mcolors.to_rgb(colour)),
                          axis=-1) <= 0.25
    rc, cc = int(py) - r0, int(px) - x0
    centre = int(core[rc - 1:rc + 2, cc - 1:cc + 2].sum())
    rows = np.where(core.any(axis=1))[0]
    cy = int(0.5 * (rows.min() + rows.max())) if len(rows) else 0
    low = round(int(core[cy:].sum()) / max(1, int(core.sum())), 3)
    plt.close(fig)
    return int(core.sum()), centre, low


def marker_fill_check_b(notes, fails):
    """A5：面板 (b) 的 ε 填充编码必须与面板 (a) **逐形状**同样可区分，且不退化。

    两个面板用的是同一套六种标记形状，所以逐形状对拍是可能的。对每个形状在 300 dpi
    下离屏量三种样式（空心 0.8 / 实心 3.2 / 同尺寸纯实心的几何基线），判据：
      c1 空心标记的中心 3x3 必须 0 墨、实心的中心必须全墨（9）：两者的差别是"空心 vs
         实心"本身，而不是墨量多少；
      c2 两种样式的核心墨量都必须 > 0（标记真的画出来了）；
      c3 两种样式的下半部墨量占比都必须落在该形状自身基线的 ±0.12 以内（都没有
         退化成 ε=8.0 那种"下半填充"：面板 (a) 的 8.0 样式偏离基线 +0.25）；
      c4 面板 (b)（ms 4.8）每个形状的核心墨量必须 >= 面板 (a)（ms 4.4）同形状的
         0.9 倍——即面板 (b) 的可区分性不低于面板 (a)。
    另附实测说明：在 ms 4.4/4.8、mew 0.8/0.5（白色描边）这套手稿约定下，"实心标记
    的墨量一定更多"并不成立（三角形/十字在小尺寸下实心核心反而不大于空心环），
    两个面板都是如此；因此判据落在"中心是否为空"+ 非退化上，而不是墨量比。
    """
    def pairs(panel):
        if panel == "a":
            return [(lbl, c, mk, 4.4) for _k, lbl, c, mk, _t in SERIES_A]
        return [(PALETTE_B[s][0], PALETTE_B[s][1], PALETTE_B[s][2], 4.8)
                for s in SERIES_B]
    m = {}
    for panel in ("a", "b"):
        for _lbl, c, mk, ms in pairs(panel):
            for kind, mew in (("hollow", 0.8), ("filled", 0.5),
                              ("baseline", 0.5)):
                m[(panel, mk, kind)] = _measure_marker(mk, c, ms, mew, kind)
    notes.append("A5 panel (b) eps fill encoding vs panel (a), measured "
                 "offscreen at 300 dpi, shape by shape "
                 "(core ink px / ink px in the marker centre 3x3 / share of core "
                 "ink in the marker's lower half):")
    bad = []
    for _lbl, _c, mk, ms in pairs("b"):
        ha, fa, ba = (m[("a", mk, k)] for k in ("hollow", "filled", "baseline"))
        hb, fb, bb = (m[("b", mk, k)] for k in ("hollow", "filled", "baseline"))
        notes.append("   marker %-3r  panel (a) ms 4.4: hollow %4d/%d/%.3f | "
                     "filled %4d/%d/%.3f | plain-fill baseline %.3f"
                     % (mk, ha[0], ha[1], ha[2], fa[0], fa[1], fa[2], ba[2]))
        notes.append("                panel (b) ms 4.8: hollow %4d/%d/%.3f | "
                     "filled %4d/%d/%.3f | plain-fill baseline %.3f"
                     % (hb[0], hb[1], hb[2], fb[0], fb[1], fb[2], bb[2]))
        checks = [
            ("c1 centre empty/solid", hb[1] == 0 and fb[1] == 9 and
             ha[1] == 0 and fa[1] == 9),
            ("c2 both drawn", hb[0] > 0 and fb[0] > 0),
            ("c3 neither half-filled",
             abs(hb[2] - bb[2]) <= 0.12 and abs(fb[2] - bb[2]) <= 0.12 and
             abs(ha[2] - ba[2]) <= 0.12 and abs(fa[2] - ba[2]) <= 0.12),
            ("c4 (b) >= 0.9x (a) ink", hb[0] >= 0.9 * ha[0]
             and fb[0] >= 0.9 * fa[0]),
        ]
        notes.append("                %s" % "; ".join(
            "%s: %s" % (n, "OK" if v else "FAIL") for n, v in checks))
        if not all(v for _n, v in checks):
            bad.append((mk,) + tuple("%s=%s" % (n, v) for n, v in checks))
    notes.append("   A5 reference (panel (a) A4, black circles): hollow centre 0 "
                 "ink / filled centre 9 ink, hollow 60 dark px vs filled 139 dark "
                 "px; the half-filled eps=8.0 style sits +0.25 off its baseline")
    notes.append("   A5 verdict: %s" % ("every canonical shape separates hollow "
                                        "(0.8) from filled (3.2) by an empty vs "
                                        "solid centre, with no half-filled drift "
                                        "and no ink loss against panel (a)"
                                        if not bad else "FAIL %s" % bad))
    if bad:
        fails.append("A5 the panel (b) eps fill encoding is degenerate for %s"
                     % bad)
    notes.append("")


def draw(a_res, xpts, ypts):
    """画出两个面板并把**实际绘制**的 (颜色, 标记) 记进 `drawn`；本函数不建图例
    ——图例由 place_legends() 在坐标范围与注释都定好之后再定位（B8）。"""
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 8,
        "axes.labelsize": 8.6, "axes.titlesize": 8.2, "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4, "legend.fontsize": 6.6, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.2, "ytick.major.size": 2.2,
        "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=FIGSIZE, dpi=300,
                                     gridspec_kw=GS_AFTER)
    drawn = {"a": {}, "b": {}}
    drawn_b_eps = {}
    # ---- (a)
    ax_a.axhline(0.0, color="#999999", lw=0.8, ls=":", zorder=0)
    ax_a.axhline(1.0, color="#999999", lw=0.8, ls="--", zorder=0)
    for key, label, c, mk, _tag in SERIES_A:
        xs = [a_res[e][key]["time_s"] for e, _ in EPS_A]
        ys = [a_res[e][key]["delta_ratio"] for e, _ in EPS_A]
        ax_a.plot(xs, ys, color=c, ls="-", lw=0.95, alpha=0.92, zorder=3)
        for (e_str, _e), x, y in zip(EPS_A, xs, ys):
            st = dict(EPS_FILL[e_str])
            st["mfc"] = "none" if e_str == "0.8" else c
            st["mec"] = "white" if e_str == "3.2" else c
            ax_a.plot([x], [y], marker=mk, ls="none", color=c, zorder=6, **st)
        drawn["a"][key] = (label, c, mk)
    ax_a.set_xscale("log")
    ax_a.set_xlabel("wall-clock per cell (s)", labelpad=1.6)
    ax_a.set_ylabel(r"$y = \Delta_{\mathrm{syn}}/\Delta_{\mathrm{real}}$",
                    labelpad=1.6)
    ax_a.set_title("(a) controlled benchmark", pad=TITLE_PAD)
    # ---- (b)
    ax_b.axhline(1.0, color="#999999", lw=0.8, ls="--", zorder=0)
    for series in SERIES_B:
        label, c, mk = PALETTE_B[series]
        xs = [xpts[(series, e)] for e, _ in EPS_B]
        ys = [ypts[(series, e)] for e, _ in EPS_B]
        ax_b.plot(xs, ys, color=c, ls="-", lw=0.95, alpha=0.92, zorder=3)
        for (e_str, _e), x, y in zip(EPS_B, xs, ys):
            st = dict(EPS_FILL_B[e_str])
            st["mfc"] = "none" if e_str == "0.8" else c
            st["mec"] = c if e_str == "0.8" else "white"
            ax_b.plot([x], [y], marker=mk, ls="none", color=c, zorder=6,
                      label=(None if e_str == "0.8" else series), **st)
            drawn_b_eps[(series, e_str)] = (
                "hollow" if e_str == "0.8" else "filled",
                st["ms"], st["mew"], st["mfc"])
        drawn["b"][series] = (label, c, mk)
    ax_b.set_xscale("log")
    ax_b.set_xlabel("wall-clock per budget (s)", labelpad=1.6)
    ax_b.set_ylabel("retention error", labelpad=1.6)
    ax_b.set_title("(b) four real databases", pad=TITLE_PAD)
    for ax in (ax_a, ax_b):
        ax.grid(axis="y", color="#d9d9d9", lw=0.45)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5), numticks=20))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: "%g" % v))
    # 面板 (a) 的时钟范围不足一个数量级（≈1.35--23 s），仅按十进制主刻度标注
    # 会导致可见标签只剩一个 "10"，失去量程参照；故对 (a) 额外标注 2/5/20 等
    # 次刻度（面板 (b) 跨三个数量级，保持只标十进制）。
    ax_a.xaxis.set_minor_formatter(FuncFormatter(lambda v, _p: "%g" % v))
    # 图例一律在 place_legends() 里建（此处不建，也不再有顶部共享图例）
    drawn["b_eps"] = drawn_b_eps
    return fig, ax_a, ax_b, drawn


# ------------------------------------------------------------------ 图例
# 图例参数：frameon=False、6.0 pt（2026-09-21 由 5.4 pt 上调，使印刷字号达到
# 全稿 5.5 pt 下限之上；改动后 B8 图例无碰撞自检须复跑）、2 列方法图例
LEG_KW = dict(frameon=False, fontsize=6.0, handletextpad=0.28,
              labelspacing=0.26, columnspacing=0.55, borderaxespad=0.0)


def legend_handles_a():
    """面板 (a) 的方法图例（受控基准调色板，逐字保持原样）。"""
    return [Line2D([], [], color=c, marker=mk, ls="-", lw=0.95, ms=4.4,
                   markeredgecolor="white", markeredgewidth=0.5, label=label)
            for _k, label, c, mk, _t in SERIES_A]


def legend_handles_a_fill():
    """面板 (a) 的 ε 填充编码（原先的 fig-level 第二行图例）：改成面板 (a) 自己
    的第二个图例，随面板一起定位、一起做遮挡与像素复检。"""
    return [Line2D([], [], color="#555555", marker="o", ls="none", ms=4.4,
                   markerfacecolor="none", markeredgecolor="#555555",
                   markeredgewidth=0.8, label=r"$\varepsilon{=}0.8$ hollow"),
            Line2D([], [], color="#555555", marker="o", ls="none", ms=4.4,
                   markerfacecolor="#555555", markeredgecolor="white",
                   markeredgewidth=0.5, label=r"$\varepsilon{=}3.2$ filled"),
            Line2D([], [], color="#555555", marker="o", ls="none", ms=4.6,
                   markerfacecolor="#555555", markeredgecolor="#555555",
                   markeredgewidth=0.6, fillstyle="bottom",
                   label=r"$\varepsilon{=}8.0$ half-filled")]


def legend_handles_b():
    """面板 (b) 的方法图例：canonical (颜色, 标记)（与 Figure 6/8/9 同一套）。
    填充只表示 ε，不表示方法，所以每个方法条目画实心（ε=3.2，主预算）样式；
    PrivPetal 的 A800 GPU 口径由标签文字承载，不再用恒空心标记。"""
    hs = []
    for series in SERIES_B:
        label, c, mk = PALETTE_B[series]
        if series == "privpetal":
            hs.append(Line2D([], [], color=c, marker=mk, ls="none",
                             markersize=4.6, markerfacecolor=c,
                             markeredgecolor="white", markeredgewidth=0.5,
                             label=label + " (A800 GPU)"))
        else:
            hs.append(Line2D([], [], color=c, marker=mk, ls="-", lw=0.95,
                             markersize=4.6, markeredgecolor="white",
                             markeredgewidth=0.5, label=label))
    return hs


def legend_handles_b_fill():
    """面板 (b) 的 ε 填充编码（空心 0.8 / 实心 3.2），与面板 (a) 同一套样式。"""
    return [Line2D([], [], color="#555555", marker="o", ls="none", ms=4.8,
                   markerfacecolor="none", markeredgecolor="#555555",
                   markeredgewidth=0.8, label=r"$\varepsilon{=}0.8$ hollow"),
            Line2D([], [], color="#555555", marker="o", ls="none", ms=4.8,
                   markerfacecolor="#555555", markeredgecolor="white",
                   markeredgewidth=0.5, label=r"$\varepsilon{=}3.2$ filled")]


# 每组图例 = [(名字, handles, ncol, 额外 kwargs)]；组内第二个及以后按基准方位依次
# 外推堆叠（'lower ...' 向上叠，'upper ...' 向下叠），互不压盖。
GROUP_A = [("panel (a) methods", legend_handles_a, 2, {}),
           ("panel (a) marker fill", legend_handles_a_fill, 1,
            dict(handler_map={Line2D: HandlerLine2D(numpoints=1)}))]
GROUP_B = [("panel (b) methods", legend_handles_b, 1, {}),
           ("panel (b) marker fill", legend_handles_b_fill, 1,
            dict(handler_map={Line2D: HandlerLine2D(numpoints=1)}))]
# 候选位置：先四角，再四边（顺序即偏好顺序）
CANDS = [
    ("upper-right corner",      "upper right",  (0.985, 0.985)),
    ("lower-right corner",      "lower right",  (0.985, 0.015)),
    ("upper-left corner",       "upper left",   (0.015, 0.985)),
    ("lower-left corner",       "lower left",   (0.015, 0.015)),
    ("right edge, upper band",  "upper right",  (0.985, 0.855)),
    ("right edge, lower band",  "lower right",  (0.985, 0.145)),
    ("left edge, upper band",   "upper left",   (0.015, 0.855)),
    ("left edge, lower band",   "lower left",   (0.015, 0.145)),
    ("right edge, mid band",    "center right", (0.985, 0.500)),
    ("left edge, mid band",     "center left",  (0.015, 0.500)),
]
STACK_PAD = 0.012              # 组内两块图例之间的间距（坐标框高度分数）


def _dist_pt_rect(x, y, bb):
    """点到矩形的最短距离（点在框内为 0），显示坐标。"""
    dx = max(bb.x0 - x, 0.0, x - bb.x1)
    dy = max(bb.y0 - y, 0.0, y - bb.y1)
    return math.hypot(dx, dy)


def _seg_rect_hit(p0, p1, bb):
    """Liang--Barsky：线段 p0--p1（显示坐标）与矩形 bb 是否相交（含相切）。"""
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - bb.x0), (dx, bb.x1 - x0),
                 (-dy, y0 - bb.y0), (dy, bb.y1 - y0)):
        if p == 0.0:
            if q < 0.0:
                return False
        else:
            t = q / p
            if p < 0.0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False
    return True


def _ink_of(ax, series_list, fig, refs):
    """把"可能被图例压住的东西"整理出来：每个标记（含自己的半径）与每条线段。

    series_list: [(名字, [(x, y, ms_pt, mew_pt), ...], lw_pt)]，点按绘制顺序给出，
    相邻两点之间的连线就是该方法的折线段。
    返回 (markers, segs, refs)：markers = [(名字, px, py, r_px)]、
    segs = [(名字, p0_disp, p1_disp, half_lw_px)]，都在显示坐标下。
    """
    scale = fig.dpi / 72.0
    markers, segs = [], []
    for name, pts, lw in series_list:
        disp = [ax.transData.transform((x, y)) for x, y, _ms, _mew in pts]
        for (_x, _y, ms, mew), (px, py) in zip(pts, disp):
            markers.append((name, px, py, (ms / 2.0 + mew + 1.0) * scale))
        for (a, b) in zip(disp, disp[1:]):
            segs.append((name, a, b, (lw / 2.0 + 1.0) * scale))
    return markers, segs, refs


def _quant(ax, cx, cy, markers, segs):
    """图例所在象限（相对坐标框中线）内的数据元素数——"这个角空不空"。"""
    axbox = ax.get_window_extent()
    midx, midy = (axbox.x0 + axbox.x1) / 2.0, (axbox.y0 + axbox.y1) / 2.0
    bx = axbox.x0 + cx * axbox.width
    by = axbox.y0 + cy * axbox.height
    sx, sy = bx >= midx, by >= midy
    n = sum(1 for m in markers if (m[1] >= midx) == sx and (m[2] >= midy) == sy)
    n += sum(1 for s in segs
             if ((s[1][0] + s[2][0]) / 2.0 >= midx) == sx
             and ((s[1][1] + s[2][1]) / 2.0 >= midy) == sy)
    return n


def _box_inside(inner, outer, slack=1.0):
    return (inner.x0 >= outer.x0 - slack and inner.y0 >= outer.y0 - slack
            and inner.x1 <= outer.x1 + slack and inner.y1 <= outer.y1 + slack)


def _text_collisions(fig, legs, ax):
    """图例框/文字框 与 面板内其它文字（注释、标题）是否相交。"""
    rend = fig.canvas.get_renderer()
    others = [(("annotation %r" % t.get_text()[:24]), t.get_window_extent(rend))
              for t in ax.texts]
    others.append(("panel title", ax.title.get_window_extent(rend)))
    hit = []
    for label, leg in legs:
        boxes = [("frame", leg.get_window_extent(rend))]
        boxes += [("text %r" % t.get_text(), t.get_window_extent(rend))
                  for t in leg.get_texts()]
        for what, bb in boxes:
            for oname, obb in others:
                if bb.overlaps(obb):
                    hit.append("%s legend %s vs %s" % (label, what, oname))
    return hit


def place_legends(fig, ax_a, ax_b, ink, notes, fails):
    """B7/B8：为每个面板搜索一个"零遮挡"的图例位置。

    ink = {"a": (markers, segs, refs), "b": ...}，refs 为该面板参考线的数据 y。
    评分（全部升序取最小）：(硬条件不过 → 丢弃) → 框内标记/线段数 →
    与参考线相交数 → 所在象限数据元素数 → 候选序。选中的位置必须满足：
    框完整落在坐标框内、框内 0 个标记/0 条线段、与其它文字 0 相交。
    """
    rend = fig.canvas.get_renderer()
    chosen = {}
    for tag, ax, group in (("a", ax_a, GROUP_A), ("b", ax_b, GROUP_B)):
        markers, segs, refs = ink[tag]
        axbox = ax.get_window_extent()
        table, best = [], None
        for rank, (name, base, anchor) in enumerate(CANDS):
            legs, boxes = [], []
            y_cursor = anchor[1]
            upward = base.startswith("lower")
            ok = True
            for lname, hfun, ncol, extra in group:
                leg = ax.legend(handles=hfun(), loc=base,
                                bbox_to_anchor=(anchor[0], y_cursor), ncol=ncol,
                                **dict(LEG_KW, **extra))
                fig.canvas.draw()
                bb = leg.get_window_extent(rend)
                if not _box_inside(bb, axbox):
                    ok = False
                frac = ax.transAxes.inverted()
                edge = frac.transform((0, bb.y1 if upward else bb.y0))[1]
                y_cursor = edge + (STACK_PAD if upward else -STACK_PAD)
                boxes.append(bb)
                legs.append((lname, leg))
            n_mark = sum(1 for (_nm, px, py, r) in markers
                         for bb in boxes if _dist_pt_rect(px, py, bb) <= r)
            n_seg = 0
            for (_nm, p0, p1, w) in segs:
                for bb in boxes:
                    exp = type(bb).from_extents(bb.x0 - w, bb.y0 - w,
                                               bb.x1 + w, bb.y1 + w)
                    if _seg_rect_hit(p0, p1, exp):
                        n_seg += 1
                        break
            inv = ax.transData.inverted()
            n_ref = 0
            for y_ref in refs:
                for bb in boxes:
                    (dx0, dy0), (dx1, dy1) = inv.transform([(bb.x0, bb.y0),
                                                            (bb.x1, bb.y1)])
                    if min(dy0, dy1) <= y_ref <= max(dy0, dy1):
                        n_ref += 1
                        break
            coll = _text_collisions(fig, legs, ax)
            quad = _quant(ax, anchor[0], anchor[1], markers, segs)
            clear = min([_dist_pt_rect(px, py, bb) - r for bb in boxes
                         for (_nm, px, py, r) in markers] or [0.0])
            valid = (ok and n_mark == 0 and n_seg == 0 and not coll)
            table.append(dict(name=name, base=base, anchor=anchor, valid=valid,
                              inside=ok, n_mark=n_mark, n_seg=n_seg,
                              n_ref=n_ref, coll=coll, quad=quad, clear=clear,
                              boxes=boxes, legs=legs, rank=rank))
            key = (0 if valid else 1, n_mark + n_seg, n_ref, quad, rank)
            if valid and (best is None or key < best[0]):
                best = (key, table[-1])
            for _lname, leg in legs:
                leg.remove()                 # 只留最后选中的那一组
        notes.append("")
        notes.append("-- B8 panel (%s): legend location search at %d dpi "
                     "(hard: box inside the axes, 0 marker/segment inside, "
                     "0 text collision) --" % (tag, int(fig.dpi)))
        notes.append("  %-24s %-11s %-6s %6s %6s %5s %5s %8s %-4s"
                     % ("candidate", "loc", "inside", "mark", "seg", "ref",
                        "quad", "clear(px)", "note"))
        for row in table:
            if best and row is best[1]:
                note = "CHOSEN"
            elif row["valid"]:
                note = ""
            else:
                note = "invalid: " + ("outside " if not row["inside"] else "") \
                    + ("text-hit " if row["coll"] else "") \
                    + ("occl " if row["n_mark"] + row["n_seg"] else "")
            notes.append("  %-24s %-11s %-6s %6d %6d %5d %5d %8.1f %s"
                         % (row["name"], row["base"],
                            "yes" if row["inside"] else "NO",
                            row["n_mark"], row["n_seg"], row["n_ref"],
                            row["quad"], row["clear"], note))
        if best is None:
            notes.append("  (no candidate satisfied the hard conditions)")
            notes.append("  fallback (informational): one legend row per panel "
                         "above its axes - it cannot occlude data at all, and the "
                         "figure height budget would have to absorb its height")
            falls = _fallback_cost(fig, ax_a, ax_b, notes)
            fails.append("B8 panel (%s): no in-axes location is occlusion-free; "
                         "the smallest alternative is one legend row per panel "
                         "above its axes (about %.2f in of extra height per "
                         "panel row, against the 0.25 in height budget - it "
                         "would have to come off the axes instead). Nearest "
                         "candidates: %s"
                         % (tag, falls, [r["name"] for r in sorted(
                             table, key=lambda r: (r["n_mark"] + r["n_seg"],
                                                   r["n_ref"]))[:3]]))
            continue
        row = best[1]
        # 复建被选中的那一组，留在图里
        y_cursor = row["anchor"][1]
        upward = row["base"].startswith("lower")
        legs = []
        for lname, hfun, ncol, extra in group:
            leg = ax.legend(handles=hfun(), loc=row["base"],
                            bbox_to_anchor=(row["anchor"][0], y_cursor),
                            ncol=ncol, **dict(LEG_KW, **extra))
            fig.canvas.draw()
            bb = leg.get_window_extent(rend)
            y_cursor = ax.transAxes.inverted().transform(
                (0, bb.y1 if upward else bb.y0))[1] + \
                (STACK_PAD if upward else -STACK_PAD)
            legs.append((lname, leg))
        chosen[tag] = legs
        names = ", ".join(l[0] for l in legs)
        notes.append("  CHOSEN for panel (%s): %s (%s) -> %s"
                     % (tag, row["name"], row["base"], names))
        notes.append("  B7 panel (%s): markers inside the legend box = %d, "
                     "segments inside = %d, text collisions = %d, "
                     "nearest marker 1 px margin = %.1f px"
                     % (tag, row["n_mark"], row["n_seg"], len(row["coll"]),
                        row["clear"]))
        for lname, leg in legs:
            bb = leg.get_window_extent(rend)
            a = ax.transAxes.inverted().transform([(bb.x0, bb.y0),
                                                   (bb.x1, bb.y1)])
            notes.append("    %-24s box in axes fraction: x [%.3f, %.3f], "
                         "y [%.3f, %.3f]" % (lname, a[0][0], a[1][0],
                                             a[0][1], a[1][1]))
    return chosen


def _fallback_cost(fig, ax_a, ax_b, notes):
    """（只在搜索失败时才走）面板上方一行图例需要的高度，单位英寸。"""
    rend = fig.canvas.get_renderer()
    h = 0.0
    for ax, group in ((ax_a, GROUP_A), (ax_b, GROUP_B)):
        leg = ax.legend(handles=group[0][1](), loc="lower center",
                        bbox_to_anchor=(0.5, 1.0 + 0.06), ncol=3,
                        **dict(LEG_KW, **group[0][3]))
        fig.canvas.draw()
        bb = leg.get_window_extent(rend)
        h = max(h, (bb.height + ax.title.get_window_extent(rend).height)
                / fig.dpi)
        leg.remove()
    notes.append("    (informational) legend above the axes would need about "
                 "%.2f in per panel row" % h)
    return h


# ------------------------------------------------- B9：图例框内"series 色"像素
DARK_CUT = 200.0 / 255.0     # 比 y 网格（#d9d9d9 = 217/255）更暗即算"有 ink"


def legend_pixel_check(fig, legend_specs, series_colours, notes, fails,
                       dpi=300.0, tol=50.0):
    """隐藏图例后按 300 dpi 重渲染，统计每个图例框/文字框内的"series 色"像素。

    口径与 plot_alloc_sweep3.py 的自检 6 一致：与某条 series 颜色（标记 alpha=1、
    折线 alpha=0.92 叠白两种）欧氏距离 <= tol/255，且通道极差 >= 6%。极差约束
    是必需的——否则中性灰（黑字抗锯齿、灰色参考线、灰网格）会被判成"曲线色"。
    代价：面板 (a) 的三个灰系（#2b2b2b/#6e6e6e/#a8a8a8）不参与该色相判据；因此
    再加一条与颜色无关的判据：同一个"隐藏图例"的 300 dpi 图里，图例框内不得有
    任何比 y 网格更暗的像素（正常只有白底与 #d9d9d9 网格透过来）。两条合起来
    覆盖全部六个方法（含灰系）。隐藏图例后框内若仍有 series 色像素/暗像素，
    只可能是数据 ink（图例自带的示例线也被一并隐藏了）。
    """
    rend = fig.canvas.get_renderer()
    boxes = [(label, leg, leg.get_window_extent(renderer=rend),
              [(t.get_text(), t.get_window_extent(renderer=rend))
               for t in leg.get_texts()]) for label, leg in legend_specs]
    for _l, leg in legend_specs:
        leg.set_visible(False)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img = plt.imread(buf)
    for _l, leg in legend_specs:
        leg.set_visible(True)
    if img.dtype not in (np.float32, np.float64):
        img = img.astype(float) / 255.0
    img = img[..., :3].astype(float)
    H, W = img.shape[:2]
    scale = dpi / fig.dpi
    refs, achromatic = [], []
    for c in series_colours:
        rgb = np.array(mcolors.to_rgb(c), dtype=float)
        if rgb.max() - rgb.min() < 0.10:
            achromatic.append(c)
            continue
        refs.append(rgb)                      # 标记（alpha=1）
        refs.append(0.92 * rgb + 0.08)        # 折线（alpha=0.92 叠白）
    refs = np.array(refs, dtype=float)

    def count(bb):
        x0 = int(math.floor(bb.x0 * scale))
        x1 = int(math.ceil(bb.x1 * scale))
        r0 = int(math.floor(H - bb.y1 * scale))
        r1 = int(math.ceil(H - bb.y0 * scale))
        x0, r0 = max(0, x0), max(0, r0)
        x1, r1 = min(W, x1), min(H, r1)
        if x1 <= x0 or r1 <= r0:
            return 0, 1.0
        sub = img[r0:r1, x0:x1, :]
        d = np.linalg.norm(sub[:, :, None, :] - refs[None, None, :, :],
                           axis=-1).min(axis=-1)
        spread = sub.max(axis=-1) - sub.min(axis=-1)
        n = int(((d * 255.0 <= tol) & (spread >= 0.06)).sum())
        return n, float(sub.min())        # 第二项：框内最暗的通道值（0--1）

    notes.append("")
    notes.append("-- B9 pixel re-check at %d dpi (legends hidden, then pixels "
                 "inside every legend frame / legend text box counted) --"
                 % int(dpi))
    notes.append("   judged series colours: %s"
                 % ", ".join(sorted(set(series_colours))))
    notes.append("   excluded by the chroma guard (achromatic, covered by the "
                 "geometric B7 test and by the 'darkest pixel' test below): %s"
                 % (", ".join(achromatic) if achromatic else "none"))
    notes.append("   second, colour-blind criterion: in the same legend-hidden "
                 "render, every legend box must contain no pixel darker than "
                 "%.0f/255 - only white and the #d9d9d9 y-grid may show through "
                 "(a grey series, a reference line or any other ink would fail "
                 "this even though it is achromatic)" % (DARK_CUT * 255.0))
    ok = True
    for label, _leg, bb, texts in boxes:
        n_box, dark_box = count(bb)
        n_dark = 0
        if n_box == 0:
            x0 = max(0, int(math.floor(bb.x0 * scale)))
            x1 = min(W, int(math.ceil(bb.x1 * scale)))
            r0 = max(0, int(math.floor(H - bb.y1 * scale)))
            r1 = min(H, int(math.ceil(H - bb.y0 * scale)))
            if x1 > x0 and r1 > r0:
                sub = img[r0:r1, x0:x1, :]
                n_dark = int((sub.min(axis=-1) < DARK_CUT).sum())
        for txt, tbb in texts:
            n_txt, _dk = count(tbb)
            notes.append("   %-26s text %-30s coloured px: text box %3d | "
                         "legend box %3d | px darker than %d in the box %3d "
                         "(darkest channel %.3f) -> %s"
                         % (label, '"%s"' % txt, n_txt, n_box,
                            int(DARK_CUT * 255), n_dark, dark_box,
                            "OK" if (n_box == 0 and n_txt == 0 and n_dark == 0)
                            else "OVERLAP"))
            if n_box or n_txt or n_dark:
                ok = False
                fails.append("B9 %s legend: %d series-coloured px / %d dark px "
                             "inside the legend box (%d series-coloured px in "
                             "the text box %r)"
                             % (label, n_box, n_dark, n_txt, txt))
    notes.append("   B9 verdict: %s" % ("every legend/text box has 0 "
                                        "series-coloured pixels and 0 pixel "
                                        "darker than the y-grid"
                                        if ok else "FAIL"))
    return ok


# --------------------------------------------------- B10：canonical 调色板核对
def palette_check(fig, ax_a, ax_b, drawn, notes, fails):
    """B10：面板 (b) 的 (方法 → 颜色, 标记) 三层核对。"""
    notes.append("")
    notes.append("-- B10 panel (b) canonical palette "
                 "(source: %s META) --" % os.path.relpath(MAIN_EFFECT_PY, ROOT))
    notes.append("   parsed META: %s"
                 % "; ".join("%s=%s" % (k, _CANON[k]) for k in sorted(_CANON)))
    ok = True
    if len(_CANON) != 6:
        fails.append("B10 cannot parse the six-entry META block from %s"
                     % MAIN_EFFECT_PY)
        ok = False
    for series in SERIES_B:
        want = CANON_EXPECTED[series]
        got_src = _CANON.get(CANON_KEY[series])
        got_draw = drawn["b"].get(series)
        notes.append("   %-10s META[%s]=%s | task literal=%s | drawn=%s"
                     % (series, CANON_KEY[series], got_src, want,
                        got_draw[:3]))
        if got_src != want:
            fails.append("B10 %s: META says %s, canonical literal is %s"
                         % (series, got_src, want))
            ok = False
        if got_draw is None or got_draw[:3] != want:
            fails.append("B10 %s: drawn (label, colour, marker) = %s, canonical "
                         "is %s" % (series, got_draw, want))
            ok = False
    # 与真正落在坐标框里的 artist 对齐：颜色/标记两项逐条核对
    notes.append("   artists actually in panel (b) (marker, colour, mfc):")
    for series in SERIES_B:
        label, c, mk = drawn["b"][series]
        hits = [ln for ln in ax_b.lines
                if ln.get_marker() == mk
                and mcolors.to_hex(ln.get_color()).lower() == c]
        if len(hits) != len(EPS_B):
            fails.append("B10 %s: %d artist(s) with colour %s and marker %r "
                         "(expected one per budget, i.e. %d)"
                         % (series, len(hits), c, mk, len(EPS_B)))
            ok = False
            continue
        notes.append("     %-10s colour=%s marker=%r : %s"
                     % (series, mcolors.to_hex(hits[0].get_color()),
                        hits[0].get_marker(),
                        "; ".join("ms=%.1f mfc=%s" % (ln.get_markersize(),
                                                     ln.get_markerfacecolor())
                                  for ln in hits)))
        for ln in hits:
            if ln.get_marker() != mk or \
                    mcolors.to_hex(ln.get_color()).lower() != c:
                fails.append("B10 %s: artist %r/%s is not the canonical "
                             "(marker, colour)" % (series, ln.get_marker(),
                                                   ln.get_color()))
                ok = False
    # 面板 (a) 的调色板必须一字未改
    notes.append("   panel (a) palette (unchanged, controlled benchmark):")
    for key, label, c, mk, _t in SERIES_A:
        notes.append("     %-16s %s %r" % (label, c, mk))
        if drawn["a"].get(key) != (label, c, mk):
            fails.append("B10 panel (a) palette changed for %s: %s"
                         % (key, drawn["a"].get(key)))
            ok = False
    notes.append("   B10 verdict: %s" % ("panel (b) = canonical (META) = task "
                                         "literals; panel (a) unchanged"
                                         if ok else "FAIL"))
    return ok


# ------------------------------------------------- B12：面板 (b) 的 ε 填充读回
def eps_fill_check(ax_b, xpts, ypts, drawn, notes, fails):
    """B12：从 artist 上逐 (方法, ε) 读回 (mfc, ms, mew)，确认"空心只表示 ε=0.8"。

    每个 (方法, ε) 在折线上应恰好有一个标记 artist；用它的**数据坐标**定位（不靠
    绘制顺序），再读回它的 mfc / ms / mew。要求：
      * ε=0.8 的 mfc == "none"（空心），ε=3.2 的 mfc == 该方法的 canonical 颜色；
      * 六个方法的标记尺寸与边宽逐项等于 EPS_FILL_B；
      * 于是"空心"在面板 (b) 里只可能表示 ε=0.8——没有任何方法在两个预算上都空心
        （PrivPetal 的 A800 口径已改为由图例标签与图内注释承载）。
    """
    notes.append("")
    notes.append("-- B12 panel (b) eps fill read back off the artists (located by "
                 "their data coordinates) --")
    notes.append("   %-10s %-5s %-8s %-9s %-6s %-6s %-5s %s"
                 % ("method", "eps", "recorded", "mfc", "ms", "mew", "n", "verdict"))
    ok = True
    tallies = {"hollow": 0, "filled": 0}
    for series in SERIES_B:
        _label, c, mk = drawn["b"][series]
        for e_str, _e in EPS_B:
            want = "hollow" if e_str == "0.8" else "filled"
            x, y = xpts[(series, e_str)], ypts[(series, e_str)]
            hits = []
            for ln in ax_b.lines:
                if ln.get_marker() != mk:
                    continue
                xd, yd = ln.get_xdata(), ln.get_ydata()
                if len(xd) == 1 and abs(float(xd[0]) - x) < 1e-9 \
                        and abs(float(yd[0]) - y) < 1e-9:
                    hits.append(ln)
            st = EPS_FILL_B[e_str]
            ok_row = True
            if len(hits) != 1:
                fails.append("B12 %s eps=%s: %d marker artists at that data point "
                             "(expected 1)" % (series, e_str, len(hits)))
                ok = ok_row = False
                continue
            ln = hits[0]
            mfc = ln.get_markerfacecolor()
            mfc_s = "none" if (isinstance(mfc, str) and mfc == "none") else \
                mcolors.to_hex(mfc).lower()
            want_mfc = "none" if want == "hollow" else c
            if mfc_s != want_mfc:
                fails.append("B12 %s eps=%s: mfc=%s, expected %s (fill=%s)"
                             % (series, e_str, mfc_s, want_mfc, want))
                ok = ok_row = False
            if (abs(ln.get_markersize() - st["ms"]) > 1e-9
                    or abs(ln.get_markeredgewidth() - st["mew"]) > 1e-9):
                fails.append("B12 %s eps=%s: ms/mew = %.2f/%.2f, expected %.2f/%.2f"
                             % (series, e_str, ln.get_markersize(),
                                ln.get_markeredgewidth(), st["ms"], st["mew"]))
                ok = ok_row = False
            rec = drawn["b_eps"].get((series, e_str))
            if rec is None or rec[0] != want or abs(rec[1] - st["ms"]) > 1e-9:
                fails.append("B12 %s eps=%s: recorded style %s disagrees with the "
                             "artists" % (series, e_str, rec))
                ok = ok_row = False
            tallies[want] += 1
            notes.append("   %-10s %-5s %-8s %-9s %-6.2f %-6.2f %-5d %s"
                         % (series, e_str, want, mfc_s, ln.get_markersize(),
                            ln.get_markeredgewidth(), len(hits),
                            "OK" if ok_row else "FAIL"))
    notes.append("   tallies: hollow artists %d (all at eps=0.8), filled artists "
                 "%d (all at eps=3.2); no method is hollow at both budgets"
                 % (tallies["hollow"], tallies["filled"]))
    notes.append("   B12 verdict: %s" % ("hollow means exactly eps=0.8 in panel "
                                         "(b)" if ok else "FAIL"))
    return ok


# ------------------------------------------------------- B11：点对拥挤度报告
def mark_overlap(points, limits, figsize, gridspec, dpi, thresholds):
    """在给定坐标框几何下，打印/返回"点对显示距离"的分布。

    points: [(名字, ms_pt, (x, y))]；返回 (rows, pairs)：
    rows = [(阈值 pt, 更近的点对数)]；pairs = 按距离升序的点对明细。
    坐标范围与数据都不随调色板/图例改变，因此同一函数可以复算"改版前"的几何。
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    axes = fig.subplots(1, 2, gridspec_kw=gridspec)
    rows, detail = [], []
    for ax, key in zip(axes, ("a", "b")):
        ax.set_xscale("log")
        x0, x1, y0, y1 = limits[key]
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
    fig.canvas.draw()
    for ax, key in zip(axes, ("a", "b")):
        pts = [p for p in points[key]]
        disp = [ax.transData.transform(p[2]) for p in pts]
        dists = []
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = math.dist(disp[i], disp[j]) * 72.0 / dpi     # 显示 px → pt
                dists.append((d, pts[i][0], pts[j][0],
                              max(pts[i][1], pts[j][1])))
        rows.append((key, [(t, sum(1 for d in dists if d[0] < t))
                           for t in thresholds],
                     sum(1 for d in dists if d[0] < d[3])))
        detail.append((key, sorted(dists)[:12], dists))
    plt.close(fig)
    return rows, detail


def mark_overlap_report(xy_a, xy_b, limits, notes, thresholds):
    """B11：改版前后（坐标框 top 0.80 → 0.905）同口径的拥挤度对比。

    改版前的几何用 GS_BEFORE 原样复算；其 <5.76 pt 的计数应与被接受版本归档的
    B8 行（当时的判据是 100 dpi 下 8 显示 px ≡ 5.76 pt）逐面板一致——这就是这次
    复算是否忠实的交叉验证。
    """
    pts = {"a": xy_a, "b": xy_b}
    notes.append("")
    notes.append("-- B11 marker-overlap report (display distance in POINTS; "
                 "a pair counts when the distance is below the marker "
                 "diameter = the larger marker's ms) --")
    notes.append("   note: the canonical palette changes colour/marker shape "
                 "only, and the panel (b) marker sizes are now uniform (all "
                 "series: ms 4.8 with mew 0.8 hollow at eps=0.8 / mew 0.5 "
                 "filled at eps=3.2) - same data, same limits - so the metric "
                 "only moves with the axes box (top 0.80 -> 0.905, height "
                 "1.56 -> 1.81 in) and with the marker fill/size.")
    before_576 = {}
    for tag, gs in (("before (accepted version)", GS_BEFORE),
                    ("after  (this version)", GS_AFTER)):
        rows, detail = mark_overlap(pts, limits, FIGSIZE, gs, 300, thresholds)
        for (key, thr, dia), (dkey, pairs, allp) in zip(rows, detail):
            notes.append("   %-26s panel (%s): closer than marker diameter = %d "
                         "pair(s); per threshold: %s"
                         % (tag, key, dia,
                            ", ".join("<%.2f pt: %d" % t for t in thr)))
            if tag.startswith("before"):
                before_576[key] = dict(thr)[5.76]
            for d, n1, n2, th in pairs[:5]:
                notes.append("        %.2f pt  %s -- %s (dia %.1f pt)"
                             % (d, n1, n2, th))
    notes.append("   cross-check of the reconstruction: the before-geometry "
                 "count at <5.76 pt (= the archived 8 display px at 100 dpi) is "
                 "%s; the accepted version's B8 line recorded %s"
                 % (before_576, ARCHIVED_B8))
    if before_576 != ARCHIVED_B8:
        notes.append("   WARN the reconstruction does not reproduce the archived "
                     "B8 counts; treat the before/after comparison with care")


def caption_requirements(xpts, ypts):
    """图注必须具备的声明（供写 caption 时逐条落实；本脚本不改手稿）。"""
    n = ["", "== caption requirements (the caption must carry all of these) ==",
         "(1) panel (a): x is the per-run wall clock recorded by the driver for the "
         "cell (a single value, not a median over the 3 runs); y is the median over "
         "the 3 runs; the marker fill encodes eps (hollow 0.8, filled 3.2, "
         "half-filled 8.0); dotted line y=0 and dashed line y=1.",
         "(2) panel (a) shows the same 18 cells as the existing controlled-benchmark "
         "table/figure; both Chain-NPM variants sit at y~0.87-1.03 for ~2 s while the "
         "four per-FK baselines sit at |y|<=0.23 for 2.5-16 s.",
         "(3) panel (b): x is the median over the four databases of the per-database "
         "median of the SYNTHESIS-phase wall clock (model+marginals+sampling only; "
         "the evaluation is excluded). PrivPetal's x is that method's own wall clock "
         "on A800 GPU(s) while the other five are single-CPU numbers - the same "
         "asymmetry Table 2 discloses and calls conservative for Chain-NPM.",
         "(4) panel (b): y is the median retention error on the common basis = "
         "plant_product@t=0.316 and plant_xor@phi=0.8 (10 seeds each; the 20 per-seed "
         "values are pooled per (method, dataset, eps)), then median over the four "
         "databases. The dashed line at y=1.0 is the no-effect level.",
         "(5) panel (b) ground truth caveat: each cell's retention error is relative "
         "to that cell's own measured real effect (the manuscript's definition). The "
         "realized ground truth is NOT one single database across the six methods: "
         "the four p4_common methods redraw the planted ground truth per seed while "
         "PrivPetal's planted files carry plant_seed=42 (and PrivLava's V-shaped "
         "natural cells come from the same vshape pipeline); the two variants differ "
         "by up to 26% on movielens plant_xor. Switching to a single pooled "
         "denominator moves every y by at most 0.018 and leaves Chain-NPM the lowest "
         "retention error at both budgets (see the B4 block).",
         "(6) no dataset had to be dropped for a missing/zero runtime; PrivLava's "
         "V-shaped natural cells record no runtime_sec, so its synthesis time is "
         "model_sec+synthesize_sec (the same synthesis-only basis).",
         "(7) there is no shared legend: each panel carries its own in-panel "
         "legend (methods + a marker-fill key). Panel (b)'s six methods use the "
         "manuscript's canonical palette for this method set (the same colour and "
         "marker as Figures 6, 8 and 9), so a colour denotes the same method in "
         "every figure of the paper (PrivPetal is the orange/brown square #b45309 "
         "everywhere). In BOTH panels the marker fill encodes eps and nothing "
         "else (hollow = 0.8, filled = 3.2, exactly as Figure 6); PrivPetal's "
         "A800-GPU x-axis caveat is carried by its legend label and the in-plot "
         "annotation rather than by a marker style. Panel (a) "
         "uses its own controlled-benchmark palette and additionally spells out "
         "the third budget (eps=8.0 half-filled), and neither legend reuses the "
         "other panel's (colour, marker) slots."]
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true",
                    help="只算数与自检（写覆盖报告），不出 pdf/png")
    ap.add_argument("--arm", choices=["released", "szquota"], default="released",
                    help="panel (b) Chain-NPM retention source: released "
                         "campaign tree or the szquota plant grid")
    args = ap.parse_args()
    notes, fails = [], []
    notes.append("fig_pareto coverage report   MODE: %s arm=%s"
                 % ("dry (no pdf/png written)" if args.dry else "full",
                    args.arm))
    notes.append("panel (a) source: tmp/multihop_final_eps{0.8,3.2,8.0}.json + "
                 "tmp/chain_results_eps{0.8,3.2,8.0}.json")
    notes.append("panel (b) source: %s/  (resolved through results/tables/p7_common.py)"
                 % os.path.relpath(P.D, ROOT))
    notes.append("")
    a_res = panel_a(notes, fails)
    marker_fill_check(notes, fails)
    marker_fill_check_b(notes, fails)
    nat_rows, plant_rows = load_index(
        "szquota" if args.arm == "szquota" else None)
    xpts, ypts = panel_b(nat_rows, plant_rows, notes, fails)

    fig = None
    if not fails:
        fig, ax_a, ax_b, drawn = draw(a_res, xpts, ypts)
        xa = [a_res[e][k]["time_s"] for e, _ in EPS_A
              for k, _l, _c, _m, _t in SERIES_A]
        ya = [a_res[e][k]["delta_ratio"] for e, _ in EPS_A
              for k, _l, _c, _m, _t in SERIES_A]
        xb = [xpts[(s, e)] for s in SERIES_B for e, _ in EPS_B if (s, e) in xpts]
        yb = [ypts[(s, e)] for s in SERIES_B for e, _ in EPS_B if (s, e) in ypts]
        lim = {"a": (min(xa) / 1.45, max(xa) * 1.45, min(ya) - 0.10, max(ya) + 0.06),
               "b": (min(xb) / 1.55, max(xb) * 1.45, min(yb) - 0.07,
                     max(max(yb) + 0.07, 1.10))}
        for ax, key, xs, ys in ((ax_a, "a", xa, ya), (ax_b, "b", xb, yb)):
            x0, x1, y0, y1 = lim[key]
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            out = [(x, y) for x, y in zip(xs, ys)
                   if not (x0 <= x <= x1 and y0 <= y <= y1)]
            notes.append("axis panel (%s): x [%.3f, %.3f] s, y [%.3f, %.3f]; "
                         "plotted points outside the box: %d"
                         % (key, x0, x1, y0, y1, len(out)))
            if out:
                fails.append("B6 panel (%s) would clip %s" % (key, out[:4]))
        # 注释（范围定好后画）
        ax_b.text(lim["b"][1] * 0.97, 1.0 - 0.012 * (lim["b"][3] - lim["b"][2]),
                  "no-cross-table level", fontsize=6.2, color="#777777",
                  ha="right", va="top")
        px = [xpts[("privpetal", e)] for e, _ in EPS_B if ("privpetal", e) in xpts]
        py = [ypts[("privpetal", e)] for e, _ in EPS_B if ("privpetal", e) in ypts]
        if px:
            ax_b.annotate("PrivPetal: A800 GPU\nwall-clock; others\nsingle CPU",
                          xy=(min(px), max(py)),
                          xytext=(min(px) * 0.92, lim["b"][3] - 0.30
                                  * (lim["b"][3] - lim["b"][2])),
                          textcoords="data", fontsize=6.0, color="#a85b12",
                          ha="right", va="bottom",
                          arrowprops=dict(arrowstyle="-", color="#a85b12",
                                          lw=0.6, shrinkA=2, shrinkB=3))
        fig.canvas.draw()
        # ---- B7/B8：图例定位（数据 ink 的几何遮挡搜索） ----
        # 标记半径随 ε 不同（面板 (a) 三档 / 面板 (b) 两档），逐点给出 ms/mew
        EPS_STYLE = {"0.8": (4.4, 0.8), "3.2": (4.4, 0.5), "8.0": (5.0, 0.6)}
        STYLE_B = {e: (v["ms"], v["mew"]) for e, v in EPS_FILL_B.items()}
        ink = {
            "a": _ink_of(ax_a, [
                ("%s" % lbl,
                 [(a_res[e][k]["time_s"], a_res[e][k]["delta_ratio"],
                   EPS_STYLE[e][0], EPS_STYLE[e][1]) for e, _ in EPS_A], 0.95)
                for k, lbl, _c, _m, _t in SERIES_A], fig, [0.0, 1.0]),
            "b": _ink_of(ax_b, [
                ("%s" % DISPLAY_B[s],
                 [(xpts[(s, e)], ypts[(s, e)], STYLE_B[e][0], STYLE_B[e][1])
                  for e, _ in EPS_B if (s, e) in xpts], 0.95)
                for s in SERIES_B], fig, [1.0]),
        }
        chosen = place_legends(fig, ax_a, ax_b, ink, notes, fails)
        # ---- B10：canonical 调色板核对（与绘制同一次运行） ----
        palette_check(fig, ax_a, ax_b, drawn, notes, fails)
        # ---- B12：面板 (b) 的 ε 填充读回（空心只能表示 ε=0.8） ----
        eps_fill_check(ax_b, xpts, ypts, drawn, notes, fails)
        # ---- B9：像素级复检（图例框/文字框内 0 个 series 色像素） ----
        specs = [l for tag in ("a", "b") for l in chosen.get(tag, [])]
        colours = [c for _k, _l, c, _m, _t in SERIES_A] + \
                  [PALETTE_B[s][1] for s in SERIES_B if PALETTE_B[s]]
        legend_pixel_check(fig, specs, colours, notes, fails)
        # ---- B11：点对拥挤度（标记直径阈值），改版前/后同口径 ----
        pa = [("%s/eps=%s" % (lbl, e), 4.4 if e != "8.0" else 5.0,
               (a_res[e][k]["time_s"], a_res[e][k]["delta_ratio"]))
              for e, _ in EPS_A for k, lbl, _c, _m, _t in SERIES_A]
        pb = [("%s/eps=%s" % (DISPLAY_B[s], e), STYLE_B[e][0],
               (xpts[(s, e)], ypts[(s, e)]))
              for s in SERIES_B for e, _ in EPS_B if (s, e) in xpts]
        mark_overlap_report(pa, pb, lim, notes, [4.8, 5.8, 5.76, 8.0])
        fig.canvas.draw()
    if fails:
        notes.append("")
        notes.append("REFUSING TO DRAW (%d failure(s)):" % len(fails))
        notes.extend("  " + f for f in fails)
        with open(COVERAGE_TXT, "w") as fh:
            fh.write("\n".join(notes) + "\n")
        print("refusing to draw; see %s" % COVERAGE_TXT)
        for f in fails:
            print("FAIL", f)
        sys.exit(1)
    notes.extend(caption_requirements(xpts, ypts))
    if args.dry:
        notes.append("")
        notes.append("MODE dry: pdf/png 未写出；以上数值即将会被绘制的全部数值。")
        with open(COVERAGE_TXT, "w") as fh:
            fh.write("\n".join(notes) + "\n")
        print("dry run: pdf/png NOT written; coverage report written to %s"
              % COVERAGE_TXT)
        print()
        print("\n".join(notes))
        return
    fig.savefig(BASE_NAME + ".pdf")
    fig.savefig(BASE_NAME + ".png", dpi=300)
    with open(COVERAGE_TXT, "w") as fh:
        fh.write("\n".join(notes) + "\n")
    print("written:", BASE_NAME + ".pdf")
    print("written:", BASE_NAME + ".png")
    print("written:", COVERAGE_TXT)
    print()
    print("\n".join(notes))


if __name__ == "__main__":
    main()