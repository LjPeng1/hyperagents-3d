# Portable HyperAgents × 3D 实验

这是对 HyperAgents 的受限移植，用于检查“改写后的改进器是否真的参与下一轮改进”，并与 DREAM 的探索策略改进进行小预算比较。它不是原论文 Docker/tool-agent 实验的完整复现。论文证据与因果解释边界见 [PAPER_AUDIT_ZH.md](PAPER_AUDIT_ZH.md)。

## 环境与版本

两个仓库应位于同一目录：

```text
research/
  dream-rsi-3d/
  hyperagents-3d/
```

此移植使用 sibling DREAM fork 的 Codex transport，基准提交为 `361706351ccfabfca9b1829c9ce44177923271f5`。HyperAgents 原始代码基于 `59a68f672dfb92c74aeb7e61535d776fb36e172d`。先核对实际版本；不要为满足示例而重置有本地修改的 checkout。

在 `hyperagents-3d` 根目录运行以下 PowerShell 命令。可以复用已准备好的 sibling 虚拟环境；仅在不存在时创建它。

```powershell
git -C ..\dream-rsi-3d rev-parse HEAD
if (-not (Test-Path ..\dream-rsi-3d\.venv\Scripts\python.exe)) {
    python -m venv ..\dream-rsi-3d\.venv
}
$taskPython = '..\dream-rsi-3d\.venv\Scripts\python.exe'
& $taskPython -m pip install -e ..\dream-rsi-3d pytest
```

本 portable 路径不需要安装原始 `requirements.txt` 的 Genesis、BALROG、CUDA 或 proofgrader 依赖。Codex CLI 需要已完成认证；模型与 timeout 通过共享 transport 的 `CODEX_MODEL`、`CODEX_EXEC_TIMEOUT_SEC` 等配置传入。未设置 `CODEX_MODEL` 表示使用现有 CLI 默认值，不等于证实某个具体模型版本；以调用日志为准。不要通过额外参数覆盖 transport 的输出 schema、输出路径或权限设置。

## 离线检查与运行入口

这些检查不调用模型：

```powershell
New-Item -ItemType Directory -Force workspace | Out-Null
& $taskPython -m pytest tests_portable -q --basetemp workspace\pytest-portable
& $taskPython -m portable.experiment --help
```

实验入口为 `python -m portable.experiment`，阶段为 `e0`、`hyper`、`dream`、`report`、`all`。`--output` 必须位于本仓库的 `workspace/` 内；首次运行使用空目录。`--timeout` 是每个模型 episode 的秒数，默认 300。下面的运行命令会调用模型；`report` 阶段本身只做离线检查与最终评估。

```powershell
$env:CODEX_MODEL = 'gpt-6-astra'
$env:CODEX_EXTRA_ARGS = '-c model_reasoning_effort=\"medium\"'
& $taskPython -m portable.experiment --phase all --output workspace/comparison --timeout 300
```

也可以按阶段执行，保持 output、timeout、模型配置和源码不变：

```powershell
& $taskPython -m portable.experiment --phase e0 --output workspace/comparison --timeout 300
& $taskPython -m portable.experiment --phase hyper --output workspace/comparison --timeout 300
& $taskPython -m portable.experiment --phase dream --output workspace/comparison --timeout 300
& $taskPython -m portable.experiment --phase report --output workspace/comparison --timeout 300
```

首次执行写入 `protocol.json`，冻结 host、初始 task/meta、六个任务文件和共享 transport 的哈希，以及模型、额外参数、timeout 与调用上限。后续配置或源码变化会拒绝复用该运行目录。完成的阶段可读取已有结果，不会再次调用模型。

完整预定预算上限为 **22 个模型 episodes**：

| 阶段 | 模型调用上限 |
|---|---:|
| 新鲜共享 E0 | 4 次任务程序生成 |
| Hyper adaptive | 2 次 meta 更新 + 4 次在线任务生成 = 6 |
| Hyper fixed-meta | 2 次 meta 更新调用 + 4 次在线任务生成 = 6 |
| DREAM | 2 次探索策略学习 + 4 次在线任务生成 = 6 |
| 合计 | 22 |

每个 arm 都进行两轮 update → online，online 每轮最多两个候选。共享 E0 只计费一次，三个 arm 使用同一份新鲜初始轨迹。固定 meta 组仍调用改进器并更新 task，只丢弃提议的 meta 改动。DREAM 学习使用冻结历史重放，随后在新在线生成中核对选中的策略哈希确实被使用。所有失败调用计入预算，不自动重试；episodes 相同不代表固定 token 或费用。

## 真正执行的版本链

每个版本保存两个完整源码文件：

```text
TaskAgent.forward(inputs) -> (reusable_solver_source, message_history)
MetaAgent.forward(repo_path, eval_path, iterations_left=None)
```

初始版本使用仓库原始 `task_agent.py` 与 `meta_agent.py`，通过兼容 shim 执行，未在进程中导入原始重依赖。TaskAgent 的结果是可复用于多个场景的 `solve(scene)` 程序，而不是一个场景的固定坐标。MetaAgent 的模型请求产生完整的子 TaskAgent、子 MetaAgent 源码；冻结 host 保存它们。随后真正装载并执行子版本，而不是只保存一份未被使用的“改进建议”。

执行记录必须同时保留父/子源码哈希、实际执行角色源码哈希、请求哈希和评估结果。MetaAgent 源码改动本身不证明效果：必须核实下一轮使用的 `executed_source_sha256` 与该子 meta 的哈希一致。固定 meta 对照保留初始 meta，其他比较条件由同一 host 管理。当前 pilot 对通过接口验证的子 bundle 无条件继承，不按 meta 质量择优；这是检查执行链的设计，不是原生 archive 搜索或已证实的更优 meta 选择。

原始 MetaAgent 当前只把 `repo_path` 用于自己的提示，没有自动使用 `eval_path` 或 `iterations_left`。portable host 会附加已观察到的开发集上下文，改写版本也可以调用 `portable_context.get_context()` 获取其副本。这个显式上下文桥接属于本移植的实现，不应描述为原生文件工具已运行。

## 调用与修改边界

每次角色执行最多请求 **一次**模型调用。worker 第一次执行到 `chat_with_agent` 时返回请求；host 调用模型后，将完整 history 重放给同一源码。请求哈希必须一致，第二次模型调用被拒绝。原始 meta 的 `tools_available='all'` 在此只作为兼容参数接收，不会启动 bash、编辑器或任何原生工具。

TaskAgent 也可以直接返回求解器源码与空 history，使用 **零次**模型调用；宿主仍执行开发集评测，并记录 `model_episodes=0`。这样的调用可能只复用旧源码，不代表产生了新解。Meta 更新目前须请求一次模型以输出两个子文件。首轮真实试验发现了宿主错误地拒绝零调用的契约问题；修复未改变模型提示，未重复任何付费请求，原失败与协议修正均保留在示例的 `diagnostics/` 和 `amendments.json` 中。

| 可变化 | 冻结 |
|---|---|
| `task_agent.py` 的完整实现 | 3D 数据分割和外部几何评分器 |
| `meta_agent.py` 的完整实现 | transport、模型选择、调用次数与超时 |
| 两文件内的普通 helper 函数、提示和数据 | host runner、选择/汇报规则、held-out 反馈边界 |

不允许添加共享 helper 文件、动态工具、持久化文件记忆或替换 harness。这个范围比原生 HyperAgents 更窄，便于将 task 与 meta 分开做交换/冻结比较。它也不包含原始 Docker 环境、自由文件编辑、完整 DGM archive 搜索或可改写的外层 parent selector。

允许导入 `json`、`math`、`re`、`statistics`、`collections`、`itertools`，以及以下 shim：

```python
from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent
from utils.common import extract_jsons
from portable_context import get_context
```

`AgentSystem` 提供 `self.model` 和 `self.log`。返回的消息使用 `text` 字段。禁止其他导入、文件/网络/进程访问、反射、`eval`/`exec`、通配导入、`global`、裸 `except`、`finally`、async、含 `yield` 的生成器函数、私有属性访问；`__init__` 和 `super().__init__()` 是允许的特例。生成器表达式可以使用。只允许对 `self` 的公共属性赋值，不允许改写模块属性。每个源文件上限 100,000 UTF-8 bytes。

**AST 检查和子进程隔离不是恶意代码安全沙箱。** 它们限制此实验的接口和常见越界行为，不构成对任意 Python 攻击的隔离证明。host 的外部评分和独立重新导入也必须保留。

## 3D 任务和证据边界

评估使用固定尺寸、固定对象 ID 的轴对齐 cuboid 布局。候选只输出每个对象的中心 XYZ；房间边界、地面/支撑接触、穿插和空间关系由外部几何计算核验。删除对象、缩小尺寸或自报分数不能改变宿主的输入和评分。

开发集反馈用于生成和选择：每轮与累计冠军均只按开发集 combined score 选，平分时选后出现的候选。held-out 分数在三个 arm 都完成后才用于最终报告，不能进入模型上下文或用于挑选候选；本地可读源码不等于具有安全隔离的秘密测试集。固定 seed 保证数据可复查，不保证 LLM 生成完全可复现。这里测试的是 solver 的未见场景表现，不是冻结 meta 后迁移到新任务的 Improvement@k 实验。

默认几何评估不逐次启动 Blender。最终 GLB/PNG 生成与 GLB 重新导入是另一层验证：它能证明导出对象数、尺寸和坐标与候选一致，不能把存在碰撞的布局变成成功，也不证明任意资产的视觉质量、拓扑或物理稳定性。

报告应区分：代码能运行、对象合法输出率、严格约束成功率、平均几何得分、改进器后续是否执行、改进器是否带来额外收益。单次 22-episode pilot 不能证明普遍有效或持续自加速 RSI；episodes 相等也不代表 token、推理量和成本相等。模型调用失败与无提升都属于结果，不能丢弃后只保留成功重试。

## 日志与恢复

每个角色调用保存 `started.json`、`input_bundle.json`、`executed_request.json`、`completed.json` 和 transport logs。每个付费阶段另有 `journal/*.json`。已开始、失败或状态不明确的阶段禁止直接重复执行，避免不确定状态下再次消耗模型调用。先检查日志，再决定是否需要一个明确记录的新运行；不要删除失败痕迹伪装成首次运行。

最终 `report.json` 包含选中候选的 dev/held-out 得分、两轮结果、meta 继承核验、DREAM 策略使用证据和限制。`usage.json` 分别列出共享 E0 与各 arm 的 episodes、已报告 token 和时长。缺失 usage 表示未知，不是零成本；CLI 时长包含调用开销，不能当作 GPU FLOPs 或精确计费金额。

此项目沿用 HyperAgents 的 CC BY-NC-SA 4.0 许可边界；其非商业限制与 Apache/MIT 风格的无此限制许可不同。
