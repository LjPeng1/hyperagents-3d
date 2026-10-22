# 首次 3D pilot 结果证据审计

审计日期：2026-10-23。运行目录：`workspace/comparison-20261023`。**hyper_adaptive、hyper_fixed、DREAM 均已完成两轮更新与两轮在线阶段，最终 heldout 报告已生成。** 本审计复核 `report.json`、全部 13 个当前阶段 journal、14 个 canonical 模型日志及源码文件；并无模型地重放了 Hyper 的实际 Meta 请求。没有额外模型请求，没有修改运行代码。

## 最终结果

| 对象 | dev combined score（6 场景） | heldout combined score（8 场景） | heldout strict success |
|---|---:|---:|---:|
| 初始 solver | 0.6738452381 | 0.6830505952 | 0/8 |
| 共享 E0 best | 1.0 | 1.0 | 8/8 |
| Hyper adaptive，第 1、2 轮 | 1.0 | 1.0 | 8/8 |
| Hyper fixed，第 1、2 轮 | 1.0 | 1.0 | 8/8 |
| DREAM，第 1、2 轮 | 1.0 | 1.0 | 8/8 |

各 arm 的 online best 和 cumulative best 都得到表中结果；所有被报告程序文件的实际 SHA 均与报告记录一致。**E0 在任何 Meta/policy 更新前已经达到开发与最终 heldout 满分，后续三条轨迹没有可测质量增益。** 这次试验验证了可执行链条，没有证明 RSI 改进有效，更不能用于排列两篇论文的方法优劣。heldout 是同一解析布局任务的额外实例，不是 Meta/policy 的跨任务迁移测试。

## 已确认的继承链

以下为 SHA-256 前 12 位，完整值保存在各 invocation 的 `started.json`、`input_bundle.json`、`completed.json` 与 `result.json`：

| 阶段 | 实际执行 Meta | 输出 Task | 输出 Meta | 实际请求哈希 |
|---|---|---|---|---|
| update1 | M0 `e9215559e7f1` | A1 `b1e693fe57ee` | M1 `64b3b53cbea6` | `5048f5ec6077` |
| update2 | **M1 `64b3b53cbea6`** | A2 `ab0e5235e44e` | M2 `bfa02d942a76` | `9f01627de389` |

检查了源码重算哈希，而不只相信 metadata。使用对应开发历史重新运行各次输入 Meta 的 worker，所得请求对象与已保存的 `executed_request.json` **完全相同**。因此可以确认：M1 确实作为 Python 代码执行，并参与产生 M2；不是仅将 M1 源码作为附件保存。M2 已生成并随 A2 继承，但当前仅两轮更新，**没有第三次 meta invocation 来证明 M2 自身后续执行效果**。

## 请求改变具有实际内容

M0 的实际指令为 45 字符的通用修改要求；M1 的实际指令为 77,010 字符，包含 API 约束、依据开发记录的更新规则，以及序列化证据。M1 实际做了开发记录整理、可比 benchmark 分组、失败场景摘要、满分时复用通用 solver 的指示，并传入 7 条 evidence records 和 `iterations_left=1`。M0/M1 与 M1/M2 的 AST 均不同，变化超出空白格式。

这证明了**改进程序改变了后续改进请求**，不证明请求变得更有效。宿主原本已经附加完整开发上下文；M1 再次将部分证据写入请求，会增加重复信息。记录中的总 prompt 从 95,752 bytes 增至 217,124 bytes，均未截断。固定 API 提示未在零调用修复中改变，详见 [CODE_AUDIT_ZH.md](CODE_AUDIT_ZH.md) 的协议修正记录。

## 四次 task 均复用了已有 solver

`online1_1`、`online1_2` 执行 A1；`online2_1`、`online2_2` 执行 A2。四者均记录 `model_episodes=0`、`request_hashes=[]`，没有模型日志目录；返回的 `result.json.program` **逐字相同于 E0 best 的源码文本**，文本 UTF-8 SHA 为 `b3d1aa5dcb2f…`。这是已评估程序复用，不是四次新的求解器发现。

文件哈希需区分换行：E0 原文件采用 LF，Windows `write_text` 保存的四个 `program.py` 采用 CRLF，所以后者文件字节 SHA 为 `e92d2be6010f…`。独立比较确认唯一差异为 246 个换行的 LF/CRLF，归一化后逐字相同。四个文件的实际字节哈希均符合对应 candidate 记录。

四次输出均由 evaluator 重新评估：相同的 6 个 dev 场景，`scene_sha256=5f8098ef5921…`，combined score、valid rate、strict success rate 均为 1.0，每个场景均 strict success。E0 best 已经达到这些分数，因此开发集增益为 **0**。最终 8 个 heldout 场景也全部通过，但 E0 已同样通过。记录标记 `analytic_aabb`、`blender_verified=False`，没有 Blender 视觉质量结论。

## 固定 Meta 对照也选择了相同复用行为

`hyper_fixed` 的两次 update 都实际执行初始 M0 `e9215559e7f1…`，输入和保留输出的 meta 源码均与 `initial_bundle.json` 逐字相同，`freeze_meta=true`。模型分别提议了 meta `830f87ddc66c…`、`416208fe79f8…`，但宿主没有继承它们。两次 `executed_request.json` 都仍是 45 字符的原始通用指令，请求哈希均为 `5048f5ec6077…`；宿主附加的当前 task 和开发历史仍随轮次变化，不能把请求哈希不变误读成完整模型输入不变。

两代 task 分别是 `6b75245e2e3c…`、`95ba89bf836b…`。四次 task invocation 都为 `model_episodes=0`、空请求哈希、无模型日志，返回的源码文本均等于 E0 best `b3d1aa5dcb2f…`，与 adaptive 的四次输出也相同。四次开发评估的三个总体指标均为 1.0，使用相同 dev 场景哈希。**fixed meta 不等于固定 task：该对照仍允许 M0 生成新的 task 代码，只禁止继承新 meta。**

因此，本次两代开发轨迹里，改变 meta 并未带来区别于 fixed 的 solver、分数或 task 调用节省。固定 meta 也学出了满分时直接复用的 task 行为；不能把 adaptive 的这种行为归因于“meta 自改进已经更强”。这仍是单次、已饱和任务上的观察，不证明所有场景中 meta 修改均无效。

## DREAM 的策略确实被选择并执行

对每轮分别计算被选择的历史 `method.py` 实际文件哈希，并与 `best_round.json`、update journal 的推广预期、online journal 的实际使用记录，以及归档的 `live_cycle_manifest.json` 比较，全部一致：

| 在线轮次 | 选中、推广和实际使用 policy SHA | 最终归档 | 本轮 online best solver SHA |
|---|---|---|---|
| E1（L0 后） | `dfbff3f432ac…` | `dream/history/iter001` | `51df18e11af0…` |
| E2（L1 后） | `926ce2055499…` | `dream/history/iter002` | `a78b4393fd4c…` |

最终 `dream/policy/method.py` 也等于第二轮策略。两轮都选择了新 policy（`incumbent=false`），确实在随后各两次在线生成中使用；E2 已归档，没有执行无后续在线用途的 L2。DREAM 产生了不同源码的 solver，但最终开发与 heldout 分数仍与 E0 相同。冻结 trace replay 上的 policy 选择不等于在线质量提升；这里没有观测到超过 E0 的分数收益。

## Heldout 反馈没有返回前面的生成阶段

审查了 14 个实际 `prompt.txt`，其文本 SHA 全部匹配对应 manifest。没有发现 heldout 场景 ID、`split=heldout` 记录、heldout 场景哈希 `d5ce70e9d7ae…` 或初始 heldout 分数出现在这些提示中。两次 DREAM 学习的文件清单只包含公开 API、策略源码及投影后的已观察反馈，未包含 `trace/`、`trace_pool/` 或 `raw_scores.json`；Hyper 上下文仍只读取开发记录。这里检查的是实际内容及白名单路径，不是只相信 `heldout_feedback_supplied=false` 标签。

代码路径要求所有 arm 在线阶段完成后，才在 `phase_report` 选择程序并调用 heldout evaluator；结果写入最终 report，不回流给模型。13 个当前 journal 全部 completed，最后生成阶段结束于 `2026-10-22 18:46:01.658 UTC`，report 写入时间为 `18:46:02.772 UTC`，与上述顺序一致。源码路径、实际提示和执行记录共同支持本次未发生 heldout 结果回流；它们不构成恶意候选无法读取本机文件的安全保证。

零调用契约修正前的失败 journal 另存于 `diagnostics/zero-call-contract/`，没有从诊断记录抹去。该步骤由无模型 worker 证实未调用模型后才重新开放；`amendments.json` 保留原/新 host 哈希、提示不变、重复付费调用为 0 的记录。故本次仍是公开修订过的工程 pilot，不能称为完全预注册的效果验证。

## 实际调用与 token

全程实际模型调用 **14 次 / 上限 22 次**，即 `4 shared E0 + 2 adaptive + 2 fixed + 6 DREAM`。14 个模型日志均有 usage，return code 均为 0，均无超时；当前没有失败或不确定的付费步骤。独立按 canonical 路径重新枚举并求和，和最终 report 完全一致：

| 成本归属 | 实际模型 episodes | input tokens | cached input tokens | output tokens |
|---|---:|---:|---:|---:|
| 共享 E0 | 4 | 125,092 | 13,312 | 15,699 |
| Hyper adaptive | 2 | 134,115 | 0 | 10,283 |
| Hyper fixed | 2 | 108,926 | 0 | 7,288 |
| DREAM（2 次学习 + 4 次在线） | 6 | 390,885 | 0 | 28,560 |
| 总计 | **14** | **759,018** | **13,312** | **61,830** |

总 reasoning output 报告值为 5,896；总 CLI 时长约 1,720.84 秒。cached input 是 input 中的缓存字段，不额外累加成新的输入 token。DREAM 这里的 6 次是真实模型调用，不包括 policy replay 的纯本地模拟次数。

canonical 计数路径为：`e0/explore_b*/attempt_*/agent_log/codex_json/run.json`（4）、两 Hyper arm 的 `calls/update*/logs/run.json`（各 2）、`dream/learning/iter*/logs/r*/run.json`（2）以及 `dream/history/iter001`、`iter002` 中的在线 `run.json`（各 2）。没有重计 DREAM 内复制的 E0、嵌套 history、trace_pool 或 baseline 日志。

下表只计 adaptive 自身，不把共享 E0 重复算入各 arm。token 是 CLI 原样报告的用量，未换算货币；reasoning 为单列报告字段，不额外加到 output 上。

| 调用 | 模型 episodes | input tokens | cached input | output tokens | reasoning output |
|---|---:|---:|---:|---:|---:|
| update1 | 1 | 49,375 | 0 | 4,144 | 516 |
| update2 | 1 | 84,740 | 0 | 6,139 | 221 |
| 四次 task 合计 | 0 | 0 | 0 | 0 | 0 |
| adaptive 合计 | **2** | **134,115** | **0** | **10,283** | **737** |

两个 meta episode 的 CLI 总时长为约 284.15 秒，包含 CLI 开销，不能当纯模型推理时间。共享 E0 另有 4 episodes、125,092 input tokens、15,699 output tokens，只在总账计一次。总实验调用上限仍是 22；不补用 adaptive 没有消耗的四次 task 调用额度。

fixed 也只有两次 meta episode：update1 为 49,367 input / 3,463 output，update2 为 59,559 input / 3,825 output。合计 **108,926 input / 7,288 output / 438 reasoning output tokens**，cached input 为 0；四次 task 没有模型调用。两组的实际模型 episode 数相同。adaptive 本次多报告 25,189 input 与 2,995 output tokens，没有观察到对应开发质量收益；不据此外推方法的普遍成本优势或劣势。

原始成本证据在 `hyper_adaptive/calls/update1/logs/{run,usage}.json` 与 `update2` 的同名文件；fixed 使用 `hyper_fixed` 下相同路径。可在仓库根目录用只读查询重算当前账本：

```python
import json
from pathlib import Path
from portable.experiment import collect_usage
usage = collect_usage(Path('workspace/comparison-20261023').resolve())
print(json.dumps({'episodes': usage['episodes'], 'by_arm': usage['by_arm'],
                  'tokens': usage['reported_token_totals']}, indent=2))
```

最终证据支持的结论是：**可执行的 Meta 继承和 DREAM 策略选择链条都已跑通；三条轨迹的开发与 heldout 结果均未超过 E0。** adaptive 相对 fixed 没有质量或调用数优势，报告 token 更多。Hyper 比 DREAM 少调用模型，来自本次接口下 TaskAgent 可直接复用已有 solver，而 DREAM 仍按固定在线预算生成四个候选；这不是等 token、等编辑对象或完整原论文环境下的方法胜负。单次、同任务、分数饱和且线性继承的 pilot，不能支持持续能力增长、跨领域迁移或普遍 RSI 结论。
