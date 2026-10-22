# DREAM 两轮策略审计：离线目标提高，但不能说在线调用减半

审计对象是已完成的 `comparison-20261023` 两轮学习：L0 产生 P1，L1 产生 P2。已核对两轮离线选择、P1 在 E1 与 P2 在 E2 的实际使用，以及最终报告中的质量与调用成本。审计过程没有新增模型调用，也没有修改运行代码。

P1 的 SHA-256 为 `dfbff3f432ac9a3f70c3f53689e96ad36cd7a6daa002b2234e9331122e8ec393`。它在官方离线目标上确实优于 incumbent，但这个收益来自低 β 下少探索一条分支，不是几何质量提高，也不是默认在线策略自行把调用数减半。

## 实际指标

E0 有 4 个已执行候选，开发集分数全部为 1。L0 使用这份固定 trace 做 replay，得到：

| 策略／设置 | 终止探测数 | 最佳分数 | Pareto AUC | reward |
|---|---:|---:|---:|---:|
| incumbent：固定 parallel-refine，所有 sweep β | 4 | 1 | 0 | -0.05 |
| P1：β=0、0.25 | 2 | 1 | — | — |
| P1：β=0.5、0.75、1 | 4 | 1 | — | — |
| P1：整个 β 策略族的 Pareto 汇总 | — | 1 | 0.5 | 0.45 |
| P1：实际默认 β=0.6 的单次 replay | 4 | 1 | 不单独汇总 | 不单独汇总 |

表中的 AUC 衡量不同 β 的**终止预算—质量点**构成的 Pareto 前沿，不是某次运行逐步进展曲线的面积。P1 前沿留下预算 2、质量 1 的点；相对 4 次穷尽预算，AUC 为 `(4-2)/4 = 0.5`。单 worker 的 penalty 为 1，权重为 0.05，因此 reward 为 `0.5-0.05 = 0.45`。incumbent 的终止预算为 4，所以 AUC 为 0、reward 为 -0.05。

这是当前离线目标下可验证的效率改善。不过，默认 β=0.6 的 replay 仍执行了 4 次。E1 被编排器统一限制为最多 2 次探测，实际执行的是同一分支的根节点与 refinement，二者开发集分数均为 1。这不能写成“学习后的策略把在线调用减半”。此外，策略学习本身消耗模型调用与 tokens，离线 AUC 不包含这部分成本。

## L1：同分选择 P2，不是第二次已证明的提升

L1 的 replay 池包含 E0 和 E1 两份 trace。P1 作为 incumbent 重新评估，P2 作为新 proposal 评估，两者均为 **AUC=0.25、reward=0.20**。官方选择规则在 reward 相同时选择较晚轮次，因此 P2 被推广。这证明新策略通过了同一池上的资格检查，不证明 P2 比 P1 更好。

| L1 的 trace | 实际已执行节点数 | P1 的前沿预算 / 最佳分数 / AUC | P2 的前沿预算 / 最佳分数 / AUC |
|---|---:|---|---|
| E0 (`iter000`) | 4 | 2 / 1 / 0.5 | 2 / 1 / 0.5 |
| E1 (`_current`) | 2 | 2 / 1 / 0 | 2 / 1 / 0 |
| 两份 trace 等权汇总 | — | AUC=0.25，reward=0.20 | AUC=0.25，reward=0.20 |

L0 的 reward=0.45 与 L1 的 0.20 不能直接解释成策略退化：评价池变了，同一个 P1 在新池上也是 0.20。新增 E1 trace 只有两个已观察结果；两次探测已用完该 trace 的可重放结果，所以其终止点 AUC 为 0。平均 AUC 为 `(0.5+0)/2=0.25`，减去相同的 0.05 penalty 后为 0.20。

P2 的默认 β 仍是 0.6。其默认 replay 在 E0 上使用 4 次、在 E1 上使用 2 次；后者受 replay 可观察范围限制，不能据此证明默认策略产生了新的提前停止能力。P2 修改了近期成功趋势、成功停滞计数及保留动作竞争等规则，但本次饱和结果没有证明这些修改带来收益。静态检查同样未发现隐藏结果读取或具体 trace/node ID 硬编码；网格仍为 `branch_count=2, refine_count=1`。

## 选择到实际使用的哈希

- P1：L0 选中哈希与 E1 的 `live_cycle_manifest.json` 完全一致，均为 `dfbff3f432ac9a3f70c3f53689e96ad36cd7a6daa002b2234e9331122e8ec393`。E1 manifest 同时记录默认 β=0.6、宿主 `max_steps=2`、实际 `probe_budget=2`。
- P2：L1 选中哈希与 E2 归档 manifest 的实际使用哈希完全一致，均为 `926ce20554997ea41e977f4e0b04cdf20945e54e88b3b7adf566bfdb79c38bbf`。E2 仍使用默认 β=0.6，宿主 `max_steps=2`、实际 `probe_budget=2`；selected-to-used 检查通过。

## 最终在线结果与成本

DREAM 分支实际产生 **6 次模型调用：2 次策略学习与 4 次在线求解器生成**。日志汇总为 **390,885 input tokens、28,560 output tokens**，包括两轮策略学习成本；共享 E0 的 4 次调用另计，没有重复计入 DREAM 分支。

E1 与 E2 的 4 个新求解器开发集分数全部为 1；两轮各自按开发集选出的冠军，以及累计冠军，开发集与 heldout 分数也均为 1。E0 已处于同样的质量上限，因此本次在线结果没有证明几何质量提高。得到验证的是两次策略选择、推广和后续实际执行，以及 P1 在冻结 E0 trace 上的 Pareto 目标改善；这不等于默认在线调用减少或计入学习成本后的总体收益。

## 策略代码检查

未发现文件读取、隐藏 trace/outcome 访问、`run_best` 查询，或针对具体 trace ID、`b0_a0` 等节点 ID 的硬编码。策略主要使用已观察前缀、合法动作及其公开元数据、baseline 和已完成历史；节点 ID 仅用于动作引用与确定性排序。这是对这份源代码的静态检查，不是通用安全证明。

低 β 的宽度调度只允许打开一条分支。P1 先探测该分支的根节点，再进行一次 refinement，随后停止；另一个根节点仍处于保留状态。两个已探测结果都是 1，因此能维持当前 trace 上的最佳质量。默认 β=0.6 的宽度调度允许两条分支，不能据低 β 的 Pareto 前沿推断其默认部署行为。

`plan_grid` 保持 2 branches × 2 attempts，即 `branch_count=2, refine_count=1`。代码显式限制 replay 支持范围；其历史调整需要 `observations` 字段，而当前 completed-cycle manifest 没有该字段，所以不会形成网格调整票数。当前 P1 没有发现导致 L1 因网格缩小而失去 replay 支持的问题。

E1 只执行完整网格中的部分节点也可以进入 L1：live trace 保留未执行 scaffold，replay 排除 `evaluated=False` 的节点，目标按实际已执行节点数归一化。未执行节点不是伪造的零分样本。

## 可以与不可以得出的结论

可以说：**P1 在同一份已观察 trace 上改善了官方 Pareto 目标，并且代码没有发现读取隐藏结果的捷径。**

尚不能说：默认在线策略已经节省一半调用、3D 质量提高、跨任务泛化成立，或扣除元学习成本后总体更高效。这个任务的 E0 已经饱和为 1 分；还需要区分默认 β 行为、宿主预算约束、完整调用成本和新在线结果。

## 证据位置

实验根目录相对本仓库为：

`workspace/comparison-20261023/`

以下路径均相对该实验根目录；原始运行目录由 Git 忽略，不是公开仓库中已上传的文件。采用归档文件，避免工作区的 `dream/policy/method.py` 被后续策略覆盖：

- L0 选择结果与推广哈希：`journal/dream_update1.json`。
- P1 源码：`dream/learning/iter000/heuristic_history/r0001_20261023_023405_dd7a1df1/dream_rsi/policy/discovered/method.py`。关键位置：第 92 行默认 β、第 98 行宽度调度、第 224 行宽度限制、第 355 行 `plan_grid`。
- P1 目标与默认行为：同一个 `r0001_20261023_023405_dd7a1df1` 目录下的 `proposal_results/beta_sweep.json`、`proposal_results/heuristic_eval.json`、`proposal_results/policy_execution_traces.jsonl`。
- incumbent 目标：`dream/learning/iter000/heuristic_history/r0000_20261023_023405_dd7a1df1/proposal_results/beta_sweep.json`。
- E0 网格与部署元数据：`dream/history/iter000/live_cycle_manifest.json`。
- E1 使用的 P1：`dream/history/iter001/live_cycle_manifest.json`、`journal/dream_online1.json`。
- L1 选择：`journal/dream_update2.json`。
- L1 incumbent P1：`dream/learning/iter001/heuristic_history/r0000_20261023_024023_1a1f5613/proposal_results/`。
- P2 源码与评估：`dream/learning/iter001/heuristic_history/r0001_20261023_024023_1a1f5613/dream_rsi/policy/discovered/method.py` 及同轮 `proposal_results/`。
- E2 使用的 P2：`dream/history/iter002/live_cycle_manifest.json`、`journal/dream_online2.json`。
- 最终质量、轮次与调用成本：`report.json` 的 `results`、`rounds.dream`、`usage.by_arm.dream`。

沿用的 DREAM 框架逻辑可在公开 [DREAM-RSI 3D fork](https://github.com/LjPeng1/dream-rsi-3d) 查看：`dream_rsi/learning/heuristic_search/eval.py` 的 `_summarize`、`_trace_auc`、`_pareto_summary`；`dream_rsi/learning/heuristic_search/workflow.py` 的 `_select_best_round`；`dream_rsi/env/trace_env.py` 的 `legal_actions`；`dream_rsi/learning/trace_pool.py` 的 `sync_workspace_pool_to_job`。
