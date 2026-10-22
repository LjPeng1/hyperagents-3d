# HyperAgents → 3D 可移植实现审计

审计日期：2026-10-23。上游基点：`59a68f672dfb92c74aeb7e61535d776fb36e172d`。上游为 [facebookresearch/HyperAgents](https://github.com/facebookresearch/HyperAgents)，fork 为 [LjPeng1/hyperagents-3d](https://github.com/LjPeng1/hyperagents-3d)。本文区分已读代码、离线测试和未完成验证；不把 mock 测试等同于真实模型结果。

## 当前判断

已检查 `portable/runtime.py`、`portable/worker.py`、`portable/experiment.py`、共享 3D 任务及相关测试，并追踪调用的 DREAM orchestration 和学习逻辑。**受限执行链条可以真实继承并运行 task/meta 代码；不是只把“修改后的 meta”保存在文件里。** 初次审查的 35 项 portable 离线测试通过；首次真实运行暴露了零调用契约矛盾，修正后 36 项通过，见下文补记。审计和这些测试没有调用模型。

对当前明确披露的有界机制试验，未发现阻断运行的实现错误。**但本版本采用无条件线性继承，省略官方 DGM-H 的 archive 父节点搜索，不能作为论文完整方法的性能复现。** 是否能执行一遍，与是否构成足以评价 RSI 的实验，是不同结论。

## 与官方实现的对应和差异

| 环节 | 官方实现 | 本次 portable 迁移 |
|---|---|---|
| 任务 agent | `TaskAgent.forward(inputs)` 接收任务，解析模型的 response | 保留原始类；桥接 response 为完整 `solve(scene)` 源码，再执行实际 TaskAgent 的后处理 |
| Meta agent | `MetaAgent.forward(...)` 用工具修改仓库 | 执行当前版本 MetaAgent 以生成请求；宿主要求模型返回两个完整源码字符串并保存 |
| 继承与搜索 | agent 修改形成 diff，从 archive 选择父节点并加载谱系 | bundle 保存 task/meta 源码；下一代无条件使用刚产生的合法 child，没有历史 agent 父节点选择 |
| 运行环境 | Docker、bash/editor 工具、完整依赖 | Python 子进程重放、限制导入与 AST、固定结构化输出；不执行模型工具 |
| 上下文 | agent 自己读取代码和评估文件 | 宿主附加完整授权上下文，附源码和提示哈希；超限失败，不静默截断 |
| 3D 评估 | 原仓库没有本任务 | 复制 DREAM 迁移版的同一 `blender_layout` 任务，比较同一解析几何指标 |
| 调用预算 | 改进过程可包含多轮工具/模型交互 | 每次 task invocation 允许零或一个模型 episode；meta update 必须产生一个，零调用修正见补记 |

官方依据：[task_agent.py](https://github.com/facebookresearch/HyperAgents/blob/59a68f672dfb92c74aeb7e61535d776fb36e172d/task_agent.py)、[meta_agent.py](https://github.com/facebookresearch/HyperAgents/blob/59a68f672dfb92c74aeb7e61535d776fb36e172d/meta_agent.py)、[generate_loop.py](https://github.com/facebookresearch/HyperAgents/blob/59a68f672dfb92c74aeb7e61535d776fb36e172d/generate_loop.py)、[run_meta_agent.py](https://github.com/facebookresearch/HyperAgents/blob/59a68f672dfb92c74aeb7e61535d776fb36e172d/run_meta_agent.py)。

这是 **HyperAgents 的受限 portable 线性继承变体**，不能称为论文原环境的完整复现。宿主固定提示显式说明任务目标、API 和上下文，比上游极简初始 meta 提供更多指导；agent 无法自由改变调用次数、使用任意工具、写外部记忆文件。可改进的主要对象是单次调用的请求构造、允许的确定性处理，以及继承源码中的逻辑/知识。solver 历史上下文不等于可选择父节点的 agent archive；这个差异会影响性能，必须在与 DREAM 的结果一起展示。

## 已核实的执行语义

1. **真正执行源码。** worker 校验源码后在独立命名空间定义类并调用 `forward`。源码不是仅作为 LLM prompt 的附件。runtime 还会把执行产生的请求交给固定 transport。
2. **重放不额外收费。** 需要模型时，第一次执行在调用处暂停；宿主只调用一次 `_structured_call`；第二次执行注入该响应并检查请求哈希一致。TaskAgent 也可以直接返回源码和空 history，此时没有模型调用或响应重放。worker 自身不连接模型。
3. **TaskAgent 接收的是代码。** 宿主将 `program` 包装进 `<json>{"response": ...}</json>`，原 TaskAgent 可解析。runtime 采用它实际返回的源码，离线测试验证后处理修改确实保留。
4. **改后的 meta 会执行。** `improve` 产生新 bundle；后续 `_invoke` 执行其 meta 源码。测试通过改变请求文字，验证下一代使用新实现和新源码哈希。
5. **固定 meta 语义正确。** `freeze_meta=True` 保留传入 bundle 的 meta，而丢弃提议的新 meta。runner 从初始 bundle 开始，每代检查 child 的 meta 不变；最终报告再次验证第二次更新执行的 meta 哈希。
6. **模型和额外调用受控。** 请求必须使用宿主指定的模型标识；尝试第二次调用会失败。结构化响应必须匹配固定字段，源码必须可解析，工具事件会被拒绝。
7. **付费调用不盲目重试。** 已有 `started.json` 的目录禁止重复 invocation；日志保留请求、源码哈希及 Codex usage。异常后的继续策略应由 runner 明确处理，不能通过换目录自动重复不确定调用。

## 实验解释上的必要限制

**一个 episode 不等于相同计算成本。** 输入上下文、生成长度以及缓存比例可能不同；总预算必须包含 meta 和 task 两类调用。runtime 的单次约束不保证整个实验只调用一次，也不保证跨 arm 的 token 数相同。

**宿主提示始终提供完整上下文。** meta 学会读取/整理反馈所带来的增益，可能与原论文的自行发现评估文件不同。若模型忽略执行请求而主要依赖固定宿主提示，修改 meta 的有效作用会减弱；应检查请求日志，不能仅凭源码发生变化就断言 meta 能力提升。

**Meta 的返回值不是最终补丁接口。** 当前契约以那一次模型输出中的两个源码字段作为 child，允许 `MetaAgent.forward` 返回 `None`。因此 meta 可以改调用前的逻辑，但模型输出后的任意 Python 后处理不会自动成为保存结果。报告中应保留这一限制。

**代码边界不等于安全沙箱。** agent AST 检查和子进程隔离都有明确的能力限制；几何 solver 由共享 evaluator 在独立进程执行。这不构成对恶意代码的操作系统级隔离。本审计没有因此宣称所有本机文件都对候选不可见。

**当前 3D 任务是解析布局基准。** 它检查边界、支撑、AABB 碰撞和空间关系，评估结果明确标记 `analytic_aabb` 与 `blender_verified=False`。这种任务适合低成本验证控制器，但成功不代表已证明复杂网格建模、外观质量或 Blender 原生操作能力。

## 测试记录

在共享 DREAM 虚拟环境中执行：

```powershell
python -m pytest tests_portable -q --basetemp workspace/review-pytest-full
```

结果：**35 passed**，11.66 秒。覆盖原始类的兼容性、响应桥接、后处理、meta 继承、固定 meta、单调用约束、重放一致性、非法导入/反射和非有限值拒绝；另验证六个 3D 任务文件与 DREAM checkout 的对应文件逐字节一致。完整 runner 的 mock 测试验证 22 次调用上限的生命周期、已完成阶段不重复、失败调用记账、拷贝日志去重、测试集最后评估、并列开发分数的选择规则、已评估程序哈希及策略选择到实际执行的哈希链。未做付费模型调用，未重新验证 Blender 渲染。

## Runner 审核结果

- **开发/测试隔离：通过当前路径审查。** `observed_context` 只序列化 problem、初始程序、最佳程序和开发历史；检查 `split == dev` 及程序哈希。没有递归读取 report/整个 run。`phase_report` 要求三 arm 的两轮在线阶段都已完成，再评 heldout；模型上下文不读取 report。`heldout_feedback_supplied=False` 标签不是证明，证明来自这些数据路径及测试。
- **候选选择：开发分数决定。** 并列时按候选列表顺序取后者，不读取 heldout。最终测试对象来自已保存的在线轮最佳和累计最佳；测试前复核程序文件哈希。
- **固定 meta / 继承：通过。** 两 arm 各自从初始 bundle 开始；adaptive 使用上代 child，fixed 每代保留初始 meta。模型请求的执行源码和 child 哈希有日志与断言。
- **DREAM 生命周期：通过静态追踪与 mock。** 使用新 E0；L0 后归档 E0，再 E1/L1/E2，不执行无后续在线用途的 L2；验证选中、推广、实际使用 policy 的哈希一致。
- **成本核算：通过 canonical 日志路径及 mock。** E0 至多四 episode，三个 arm 各两次更新和四次在线生成，总上限 22；E0 只记一次。使用各 arm 的真实日志目录，排除复制的 E0、嵌套 history 和 baseline。已记录的失败调用计数；没有 usage 的日志标为未知，不视为零 token。真实运行后仍需核对实际日志是否完整。
- **不重复不确定付费步骤：通过。** journal 先写 started，失败或中断步骤禁止再次执行；已完成阶段直接重用记录。protocol 冻结任务、host、transport、模型覆盖参数及调用上限，配置变化要求新输出目录。
- **原始方法保真度：有限。** 两次 meta 更新只够检查“修改的 meta 是否在下一代被执行”，不足以估计长期进步。线性继承且不按开发质量挑选 meta，DREAM 则按 replay 质量选择策略；这个差异不允许对两篇论文作优劣归因。

这些检查支持开展有界的真实可行性试验；单次试验不能支持方法优越性、meta 跨任务迁移或普遍 RSI 结论。若目标升级为 DGM-H 基线，应先恢复 agent archive 与父节点选择，再加入多次独立重复及共同 task 起点的 meta 迁移对照。

## 首次真实运行后的零调用契约修正

2026-10-23，真实 `hyper_adaptive/update1` 返回的 TaskAgent 检测到开发集已达到满分，于是直接返回上下文中的已评估通用 solver 和空 history。冻结提示中的原始 API 写的是每个 `forward` **at most ONCE**，但宿主实现要求恰好一次请求，因此 `hyper_adaptive_online1` 在进入模型 transport 前失败。这是实现与已提供契约的矛盾，初次 35 项 mock 测试未覆盖该分支；不能把它记为模型失败或额外模型调用。

独立复查使用保存的真实 child 和原开发历史调用无网络 worker，得到 `status=completed`、`request_hashes=[]`、空 history。返回程序 SHA-256 为 `b3d1aa5dcb2f252dad16d5dacd562fdf88a5f3d2e55c16d670c76adc685016ff`，与原 best 完全一致。原失败 invocation 目录只有 `started.json` 和 `input_bundle.json`，没有 `logs/`；对应 journal 显示失败发生在约 0.225 秒内。确认未调用模型依赖执行路径、worker 结果及文件证据，不能仅凭没有 usage 推断免费。

修正只允许 **TaskAgent 的已完成、零请求执行**返回 `(program_source, [])`；仍检查源码大小和 Python AST，记录 `model_episodes=0`，由原 evaluator 重新评估其输出。MetaAgent 仍必须请求一次模型；一请求上限、固定 meta 控制和两 Hyper arm 的规则相同，总上限仍为 22，不把省下的调用重新分配为额外尝试。零调用本身不是新的付费 episode，也不生成虚假的模型 usage。独立负例检查确认伪造 history、无效语法、超过 100 KB 的源码和零调用 meta 均被拒绝，且没有进入模型 transport。

为避免中途改变提示影响两组可比性，最终保留 `API_DESCRIPTION` 的开跑前原文。以 AST 读取 `git show ec82288:portable/runtime.py` 的字符串常量，确认与修复后字符串逐字相同，SHA-256 为 `bee79b81d1145e141208258c5ac7d9dd2eb04e78154dcf2acd926d726657e5f5`。修复改变宿主接受逻辑和记账，没有改变模型所见的 API 提示。

原失败 journal、invocation、原 protocol 和离线 worker 验证保留在 `workspace/comparison-20261023/diagnostics/zero-call-contract/`。`amendments.json` 记录修改理由、前后 runtime 哈希、仅重新开放的已证实未付费步骤、`repeated_paid_episodes=0`、提示不变和预算上限不变。已完成的付费 update1 直接复用，没有重付费生成 child。这个运行必须描述为有公开工程修正的 pilot，不能表述为完全预注册、从未修订的效果实验。

修复后的独立全套离线回归：**36 passed，12.02 秒**，路径 `workspace/review-zero-call-regression`。另有四项上述临时负例检查。审计者没有运行付费模型请求。当前复用行为仅说明在已有开发集满分 solver 时省下生成调用，不能视作新候选的独立发现、准确率提升或 meta 自改进有效性的证据；仍需报告实际调用量和相同源码的重复评估。
