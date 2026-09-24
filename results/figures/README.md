# results/figures/ — 终版图表与生成脚本

本目录保存论文用的图（PDF + PNG）与它们的生成脚本，以及每个脚本自检后写出的
`plot_*_coverage.txt`（逐点数值 + 自检判定）。

## ⚠️ 数据来源是"双写"关系（2026-09-21 踩坑记录）

**绘图脚本读的是 `tmp/` 下的战役源头文件，不是 `results/` 归档。** 本目录的
`results/by_dataset/*`、`results/controlled/*_3run.json` 都是**归档镜像**。改数据时
两处都要更新，否则会出现"图没变、PDF 字节却变了"的假更新（PDF 的时间戳/元数据会让
`cmp` 报 CHANGED）。

已确认的双写对：

| 绘图脚本 | 实际读取（源头） | 归档镜像 |
|---|---|---|
| `plot_controlled_2panel.py` | `tmp/multihop_final_eps{0.8,3.2,8.0}.json`、`tmp/chain_results_eps*.json` | `results/controlled/multihop_final_eps*_3run.json`、`chain_results_eps*_3run.json` |
| `plot_main_effect.py` / `plot_main_crosshop.py` / `plot_query_class_sweep.py` / `plot_dataset_sweep.py` / `plot_pareto.py` | 六方法单元格：`results/by_dataset/<ds>/`（经 `p7_common.iter_files`）**与** plant 口径的 `tmp/results/<ds>/` | `results/by_dataset/`（partial：仅 natural） |
| `plot_alloc_sweep3.py` / `plot_legacy_rails.py` | `results/controlled/alloc_sweep_*`、legacy rails 的 tmp 归档 | `results/controlled/` |

**正确流程**：先更新源头（`tmp/...`）→ 再同步归档镜像（`results/...`）→ 再跑绘图脚本
→ `cmp` 前后 PDF 只能作为"文件是否被重写"的证据，**不能**作为"数据是否变了"的证据；
数据是否变了要看 `plot_*_coverage.txt` 里的逐点数值。

参考实现：`tmp/scripts/prov/install_controlled_v2.sh`（受控台架 V2 的源头+镜像双写）。

## 复现对照
- 受控台架 V2（逐表基线 ε 诚实 + 作者协议 H̃ 重连）与 pre-fix 的 ε 盲臂：
  `results/arms/controlled_epsblind/README.md`
- 星形 1.2ε 超支的修复与重跑：`results/arms/vshape_eps12/README.md`、`results/README.md`
  （2026-09-21 两节）