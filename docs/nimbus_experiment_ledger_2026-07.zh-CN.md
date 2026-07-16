# Nimbus 实验总账（2026 年 7 月）

这份文档是中文导航账本，解决三个问题：

1. 我们每次实验到底想回答什么；
2. 得到了什么结果，证据是否可信；
3. 哪部分算法被保留、哪部分被否定、下一步为什么这样做。

完整英文实验细节、命令和产物路径仍以
[`v3_experiments_2026-07.md`](v3_experiments_2026-07.md) 为准；本账本负责把
时间线和因果关系讲清楚。当前代码结论对应仓库提交 `e0b6686`。

**数据保存边界：**repo 里提交的是实现、分析器、审计逻辑和文档，GPU 实验产生的
大体积 JSONL/日志仍保存在 `$MSCRATCH`，不在 git 中。第 6 节逐项登记 E0–E10 的
输入、原始输出、summary、decision log、marker、复算命令和缺失项。路径“已登记”
不等于本机已验证文件仍存在；正式引用前必须在 GPU 箱上核对文件、行数和 SHA256。

---

## 0. 先说人话：现在到底是什么结论？

不是“以前的算法全错了”，也不是“旧 V2 公式被 KV 替代了”。

Nimbus 的决策应拆成三个不同问题：

| 问题 | 当前答案 | 单位/作用 |
|---|---|---|
| 什么时候必须外发？ | 预测留下本地的请求是否会违反 TTFT SLO | 秒；触发条件 |
| 外发谁？ | 用成本除以请求的资源占用价值排序 | 相对分数；selector |
| 本地现在有多忙？ | sequence、prefill work、KV/状态池等部署特征 | predictor 输入，不是统一预算 |

旧 V2 公式仍然完整保留：

```text
old_v2_weight = P × (U / prefill_throughput + D × TPOT)
```

- `P`：完整 prompt tokens；
- `U`：尚未处理、未命中缓存的 prompt tokens；
- `D`：预计 decode tokens。

它目前回答的是“先踢谁”：

```text
selector_score = cloud_cost / old_v2_weight
```

score 小的先外发。它不是 TTFT 触发器，不与某个凭空构造的
token-second capacity 比较，也不是历史完整 0/1 knapsack DP。

当前代码默认仍是：

```text
kv_gap + cost_disp_current
```

但完整实验已经证明：它在本次 dense-32B 压力 cell 上不能保证 TTFT。下一版的
实验方向是：

```text
ttft_pred + cost_cachedisp_old + support-envelope fallback
```

这仍是实验候选，不是已经切换的生产默认。

---

## 1. 如何阅读证据等级

为了避免以后把旧 headline 当成最终结论，每个实验都标一个等级：

| 等级 | 含义 |
|---|---|
| **VALIDATED** | token、配置、marker、hash 和对照均通过审计，可用于当前结论 |
| **EXPLORATORY** | 数据真实，但重复数、顺序或对照不足，只能生成假设 |
| **MECHANISM ONLY** | 能说明系统机制，但不能证明 selector 或整体算法优越 |
| **SUPERSEDED** | 后续发现口径/载荷不一致，只保留作历史诊断，不能当最终证据 |

总原则：后面的审计结论覆盖前面的 headline 解读，但不会删除历史数字。

---

## 2. 实验时间线总览

| ID | 日期 | 实验目的 | 一句话结果 | 证据等级 |
|---|---|---|---|---|
| E0 | 7 月上旬 | 验证 router harness 本身没有引入偏差 | open-loop parity 1.003，queue neutrality 110 vs 111 ms | VALIDATED |
| E1 | 7 月 12 日前 | 同比例外发时，“踢谁”是否重要 | pre-v3 明显优于 random，但触发器严重欠踢 | EXPLORATORY |
| E2 | 7 月 12 日 | 在长 prompt rednote slice 上测试 KV v3 | 本地 TTFT 改善；悲观 combined 与 all-local 基本打平 | EXPLORATORY |
| E3 | 7 月 12–13 日 | 在 hybrid 35B hardest cell 上验证 v3 | 0 本地违约，但随后发现 gauge 不是 token-KV meter | MECHANISM ONLY |
| E4 | 7 月 13 日 | 探明 gauge 在 hybrid/dense 模型上的真实语义 | hybrid gauge≈sequence state；dense gauge≈token KV | VALIDATED |
| E5 | 7 月 13 日 | 在 dense 32B 上给 KV 触发器一次公平机会 | gauge 单位正确，但 KV 非绑定资源，仍系统性欠踢 | SUPERSEDED / MECHANISM ONLY |
| E6 | 7 月 14 日 | 修复 payload、同刻到达、profile 与审计合同 | 得到 token-aligned trace 和 shared-prefill TTFT predictor | VALIDATED |
| E7 | 7 月 14 日 | 在相同 TTFT stop rule 下初筛 selector | old V2 257 routes，current 266，newest 326 | EXPLORATORY |
| E8 | 7 月 15 日 | 检查 selector 结果能否跨顺序/重复稳定 | old V2 6/6 胜 newest；与 current 路由数等价但更便宜 | VALIDATED |
| E9 | 7 月 15 日 | 在完整 11,605 请求上做预注册泛化 gate | A/C 0 本地违约；K 严重失败；总 gate 因 B=39 而 FAIL | VALIDATED |
| E10 | 7 月 16 日 | 验证固定 OpenRouter provider 的首 token cancel/TTFT 路径 | DeepInfra burst 16/16 客户端断流；TTFT p50/p95/p99=467/929/1,167 ms，0/16 超 5s | MECHANISM ONLY / EXPLORATORY |
| E11 | 7 月 16 日 | 完整 11,605 请求真实云 live-hybrid A/C | 已冻结 A→C、5s observed TTFT 与完整性规则；尚未读取结果 | PRE-REGISTERED / NOT RUN |

---

## 3. 每次实验的目的、结果和真正含义

### E0 — Router/harness 基础正确性

**目的**

先证明后续差异不是 router 自己造成的排队、请求语义或统计误差。

**关键结果**

- 与可信 open-loop harness 的 TTFT p50 比值：`1.003`；
- queue neutrality：p50 `110 ms vs 111 ms`；
- 无压力时 Nimbus 0 kick，延迟与 all-local 一致；
- 当前离线测试：router 81 项 + evidence 36 项 = **117/117**。

**结论**

后续实验可以把差异归因到 workload/policy/server 条件，而不是框架基础偏差。

**证据**

- `router/test_*.py`
- `tools/test_analyze_ttft_*.py`
- `tools/test_ttft_matrix_evidence.py`

---

### E1 — Pre-v3：同样外发比例下，选择是否重要？

**决策问题**

如果 Nimbus 和 random 外发相同比例，请求选择能否明显改变 TTFT？

**设置**

`extreme_burst_1200`，pre-v3 old max-displacement policy，与 matched random
对照。

| Policy | 外发比例 | 本地 TTFT p50 | 本地 5s 违约 |
|---|---:|---:|---:|
| all-local | 0% | 324.8 s | 97.9% |
| pre-v3 Nimbus | 25.6% | 12.4 s | 94.8% |
| random matched | 约 25.6% | 108.7 s | 97.0% |

**当时得到的信号**

“踢谁”明显重要，但 pre-v3 只外发 25.6%，仍有 94.8% 本地违约，说明触发器
严重欠踢。

**后来发现的限制**

直接 ShareGPT replay 的 controller token metadata 与实际 HTTP payload 不一致，
同一时间到达的请求也常被逐条决策。因此它只能作为早期选择信号，不能证明具体
displacement selector 的因果优势。

此外，两份历史汇总对 random 外发比例的记录曾有轻微"冲突"：本 handoff 主表记
`25.6%`，旧 `router/README{,.zh-CN}.md` 记 `25.3%`。**2026-07-15 已依据箱上
原始 summary 裁决：两个数字都对**——25.6% 是 nimbus 的实际自选比例（=random
臂的 target），25.3% 是 random 臂 i.i.d. 抛硬币的实现值（详见 §6.4）。E1 三臂
raw 已重新定位并双站点镜像。

**证据等级：EXPLORATORY。**

---

### E2 — Rednote 长 prompt slice：KV-heavy workload 能否救 TTFT？

**目的**

专门挑长 prompt、看起来更可能 KV-bound 的生产 slice，测试 KV v3 是否能满足
5 秒 TTFT。

**设置**

- 80 个请求；
- prompt 大约 3k–7k tokens；
- 20× 时间压缩。

| 指标 | all-local | v3 Nimbus |
|---|---:|---:|
| 外发 | 0% | 56.3% |
| 本地 TTFT p50 / p95 | 10.74 / 26.7 s | 2.98 / 7.7 s |
| 本地违约 | 54/80 = 67.5% | 8/35 = 22.9% |
| 成本 | $0 | $0.072 |

**为什么不能只看 22.9%？**

外发请求使用的是零延迟 NullCloud。若按悲观口径把 45 个被踢请求也算违约，
Nimbus 是 `(8+45)/80 = 66.3%`，all-local 是 `67.5%`，几乎打平。

**结论**

本地 TTFT 确实改善，但“多外发”不自动等于端到端 SLO 更好。必须同时报告：

- 本地违约；
- 外发比例；
- 悲观 combined；
- 云成本。

**证据等级：EXPLORATORY。** slice 小，且不是最终 selector 证明。

---

### E3 — Hybrid 35B hardest cell：headline 很漂亮，但为什么？

**目的**

在完整 `extreme_burst_1200` 上测试当时的 KV v3。

| 指标 | all-local 历史锚点 | v3 Nimbus |
|---|---:|---:|
| 外发 | 0% | 3,931/11,605 = 33.9% |
| 本地 TTFT p50 | 324.8 s | 0.320 s |
| 本地违约 | 97.9% | 0/7,674 |
| 成本 | $0 | $1.79 |

**headline**

数字非常漂亮：0 本地违约。

**审计发现的异常**

踢出时客户端能重建的 token commitment：

- p50 `20,841` tokens；
- max `45,840` tokens；
- 只占声称容量 `216,512` 的最多 21%。

也就是说，真正的 token-KV 根本没有满，但 `kv_cache_usage_perc` 却接近满载。

**结论**

结果延迟是真实的，但“因为 token-KV 快满了所以正确踢出”的机制解释是错的。
需要 E4 gauge probe。

**证据等级：MECHANISM ONLY。** 不能用于证明 selector。

---

### E4 — Gauge probe：同一个指标名，在不同架构上不是同一种资源

**目的**

固定 token 量、改变并发数，弄清 `kv_cache_usage_perc` 到底在测什么。

**Hybrid Qwen3.6-35B-A3B**

| 负载 | 理论 in-flight token 占比 | gauge |
|---|---:|---:|
| 8 路小请求 | 0.8% | 19.12% |
| 32 路小请求 | 3.2% | 76.48% |
| 尝试 64 路 | 6.4% | 97.99%，实际约 41 路上限 |
| 2×8k prompt | 7.9% | 3.40% |

它主要像“每个 sequence 固定占一份 GDN state”的并发表，而不是 token-KV 表。

**Dense Qwen3-32B**

| 负载 | 实测 gauge |
|---|---:|
| 8 路小请求 | 1.50% |
| 32 路 | 5.95% |
| 64 路 | 11.89%，64 路全部运行 |

它才近似诚实的 token-KV meter。

**结论**

不能只看 metric 名字就把百分比乘以 token capacity。Gauge 必须做
deployment-specific probe 后才能进入模型：

- dense full-attention：可能代表 token KV；
- hybrid GDN：可能主要代表 per-sequence state；
- 其他部署：需要重新标定。

**证据等级：VALIDATED。**

**复现工具**：`tools/kv_gauge_probe.py`。

---

### E5 — Dense 32B：Gauge 单位终于正确，为何 KV trigger 仍失败？

**目的**

执行历史 Option B：换纯 full-attention dense 模型，让 gauge 确实表示 token KV，
再检查 KV v3。

**先确认中等 cell 是否有压力**

`burst_1200` all-local p50 只有 `227 ms`、0 违约，说明旧 harness 的 2,262.9 s
来自未截断生成，不再是当前口径锚点。

**极端 cell 结果**

| Arm | 外发 | 本地 TTFT p50 | 本地违约 | 悲观 combined | 成本 |
|---|---:|---:|---:|---:|---:|
| all-local | 0% | 620.3 s | 98.2% | 98.2% | $0 |
| random matched | 38.0% | 163.7 s | 96.1% | 97.6% | $1.82 |
| KV v3 | 38.6% | 6.40 s | 64.1% | 78.0% | $2.70 |

**关键机制**

Gauge 单位这次是对的，但引擎全程 sequence/compute-bound：

- `peak_inflight=128` 钉死；
- token KV 最高约 69%，从未成为绑定资源。

KV trigger 把等待队列缩到“刚好能塞进 free KV”便停止，留下约 6 秒等待，系统性
欠踢。

**结论**

“单位正确”还不够，trigger 必须对准会造成 TTFT 的绑定资源。这个结果直接推动
我们从 KV gap 转向 TTFT violation trigger。

**后来发现的限制**

这批直接 replay 同样存在 payload-token 与同刻决策问题，因此不能把 KV v3 与
random 的差异归因于 selector。

**证据等级：SUPERSEDED / MECHANISM ONLY。**

---

### E6 — 审计合同与 TTFT predictor：先把实验做对

**目的**

修复此前所有可能混淆 selector 结论的问题：

- scheduler tokens 与实际 payload tokens 对齐；
- 同一时间到达的请求形成完整候选集合；
- server、trace、profile、模型和 endpoint 全部绑定；
- completion marker 原子写入；
- 本地实际 prompt/decode usage 必须逐条精确。

**新设计分工**

```text
Trigger:  max predicted TTFT > SLO - guard
Selector: 比较 old V2 / current displacement / newest / random
Stop:     踢最短 prefix，直到所有 waiting survivors 预测安全
```

**第一次 profile 为什么被拒绝？**

旧 profiler 把 128 个 sequence slots 错当成 128 条独立 prefill lanes，得到
`7.792 s` guard，已经大于 5 秒 SLO。Runner 正确拒绝运行。

**修复**

Schema-v2 predictor 同时模拟：

- sequence slot releases；
- 一条 aggregate shared-prefill work lane；
- 已等待时间；
- decode residence；
- 独立 first-token overhead。

**第一份被接受的 schema-v2 profile（7 月 14 日 lifecycle）**

| 项目 | 数值 |
|---|---:|
| shared prefill throughput | 3,264.2498 tok/s |
| TPOT | 151.8412 ms |
| first-token overhead | 455.1077 ms |
| guard | 1,683 ms |
| held-out classifier | TP 192, FN 0, FP 22, TN 70 |
| profile SHA256 | `549dd0bf…e23f` |

**E8 使用的独立新 lifecycle profile（7 月 15 日）**

| 项目 | 数值 |
|---|---:|
| shared prefill throughput | 3,255.0044 tok/s |
| TPOT | 152.3152 ms |
| first-token overhead | 440.4170 ms |
| guard | 1,712 ms |
| held-out classifier | TP 193, FN 0, FP 21, TN 70 |
| profile SHA256 | `ab703ddc…c819` |

两次独立 lifecycle 的标定值接近，说明 schema-v2 的标定具有一定稳定性；它们
仍是两个不同 server lifecycle，不能把原始样本混在一起算成更多重复。

**结论**

从这里开始，后续 selector 比较才具备可审计的共同 stop rule。

**证据等级：VALIDATED。**

---

### E7 — 512-request 初筛：旧 V2、current、newest 谁更好？

**目的**

在完全相同的 TTFT trigger、SLO、guard、trace 和 server 下，只替换 selector。

| Selector | 外发 / 本地 | 本地违约 | 悲观 combined | 成本 |
|---|---:|---:|---:|---:|
| old V2 | 257 / 255 | 0 | 50.20% | $0.137257 |
| current displacement | 266 / 246 | 0 | 51.95% | $0.147803 |
| newest | 326 / 186 | 0 | 63.67% | $0.149767 |

**结论**

- TTFT trigger 在该 slice 上工作：留下本地的请求均满足 SLO；
- old V2 比 newest 少外发 69 个请求且成本更低；
- old V2 与 current 只差 9 个请求，一次固定顺序不能分胜负。

**证据等级：EXPLORATORY。** 需要重复和顺序平衡，因此进入 E8。

---

### E8 — 六 block 重复性与顺序敏感性

**目的**

判断 E7 是稳定 selector 信号，还是一次运行/顺序噪声。

**设计**

- 六个 512-request block；
- 共 24 arms；
- A=`old V2`，B=`newest`，C=`current`，R=`waiting_random`；
- 顺序覆盖 `ABC/BCA/CAB`，发现 A/C 符号变化后按预注册规则补
  `ACB/CBA/BAC`。

每个 arm 都是 512/512 成功、token/decode 精确、0 本地违约。

| 比较 | 结果 |
|---|---|
| A vs B | A 6/6 胜；平均少外发 50.17 个请求，改善 9.80 pp |
| A vs C 路由数 | 平均 A−C = −1.5；范围 −14…+2，属于运行/顺序等价带 |
| A vs C 成本 | A 6/6 更便宜；平均少 $0.010716，约 7.2% |

**结论**

- old V2 对 newest 的优势稳定；
- 不能声称 old V2 在路由数量上胜 current；
- 可以声称两者路由数等价，而 old V2 在六次里都更便宜。

**证据等级：VALIDATED。** 但仍只使用同一 512-request slice，因此进入 E9
完整 cell 泛化。

---

### E9 — 11,605-request 预注册完整泛化 gate

**目的**

在没有用于前面 selector 调优的完整 trace 上同时回答：

1. A/C 的 selector 信号能否泛化；
2. TTFT trigger 是否对所有 selector 都安全；
3. 默认 `kv_gap` 是否能作为 TTFT 安全 trigger；
4. all-local 是否确认负载确实有压力。

**预注册顺序**

```text
B newest → K kv_gap → C current → A old V2 → L all-local
```

**最终结果**

| Arm | Trigger + selector | 外发 / 本地 | 本地违约 | 悲观 combined | 成本 |
|---|---|---:|---:|---:|---:|
| A | TTFT + old V2 | 6,748 / 4,857 | **0** | 58.147% | **$3.344049** |
| B | TTFT + newest | 8,591 / 3,014 | **39 (1.294%)** | 74.364% | $3.643557 |
| C | TTFT + current | 6,729 / 4,876 | **0** | **57.984%** | $3.519140 |
| K | KV gap + current | 6,060 / 5,545 | **5,249 (94.662%)** | 97.449% | $3.477171 |
| L | all-local | 0 / 11,605 | **11,544 (99.474%)** | 99.474% | $0 |

**冻结 gates**

| Gate | 结果 | 含义 |
|---|---:|---|
| 五 marker / hash / token integrity | PASS | 证据完整 |
| all-local pressure anchor | PASS | 负载确实很重 |
| A/B/C 全部 0 本地违约 | **FAIL** | B 有 39 个违约 |
| A/B/C post-kick prediction ≤3.288 s | PASS | 内部 stop rule 记录满足边界 |
| A vs C 等价带 | PASS | A−C=+19，位于 ±363 行内 |
| A 成本 ≤1.05×C | PASS | A/C=0.950246 |
| A 至少领先 B 5 pp | PASS | 实际领先 16.217 pp |
| A Pareto 优于 K | PASS | 本地和悲观 combined 均严格更好 |

**为什么总 gate 是 FAIL？**

预注册规则要求 A、B、C 在相同 TTFT trigger 下都必须 0 本地违约。B 留下了 39
个违约，所以总 gate 失败。Analyzer exit code 1 表示“有效证据下的 outcome
failure”；exit code 2 才表示证据损坏。

**B 的 39 个违约说明什么？**

- 32/39 的本地 service TTFT 自身就超过 5 秒；其余 7 个是 queue+service 相加
  超线；
- 39/39 都发生在 client-visible active requests 为 126–128 时；
- planned `prompt + requested decode` commitment 全部超过 profile 的支持上界；
- B 本地请求 TPOT p50 是标定值的 1.259×，违约请求为 1.371×；
- 同期 engine-reported cache gauge 为 98.1–100%，但其语义依赖部署，只能作为
  相关上下文，不能直接叫“token-KV 满了”。

现有 decision log 没保存完整 snapshot IDs、每请求 prediction/score 和 in-flight
状态，因此不能独立重放 selector 因果链。

**A 与 C 到底谁赢？**

- 路由数只差 19，位于冻结等价带内：只能说“路由数量等价”；
- routed set 的 Jaccard 只有 0.679，共有 2,573 个请求去向不同：不能说
  “两个 selector 等价”；
- A 成本比 C 低 $0.175091，约 4.98%；
- 结合 E8，old V2 仍是主要实验 selector，但不授权默认切换。

**最终算法含义**

- 被否定：`kv_gap` 足以保证 TTFT 安全；
- 被保留：TTFT violation 作为触发目标；
- 被保留：旧 V2 token-seconds 作为 victim ordering 信号；
- 新问题：predictor 离开 calibration support envelope 时需要保守 fallback。

**证据等级：VALIDATED。** 总 gate 的“失败”本身就是有效结论。

### E10 — OpenRouter 首 token cancel 探针

**日期 / 代码来源**

- 日期：2026-07-16；
- 实现已由提交 `e0b6686` 固化；
- model：`qwen/qwen3-32b`；
- provider：只允许 DeepInfra，`allow_fallbacks=false`；
- 新模式：`--cloud-stop-after-first-token`。

**目的**

验证三件事：

1. 真实 OpenRouter 流式请求能否在第一个生成 token 后由客户端主动断开；
2. 这种请求能否诚实地计入 TTFT SLO，而不伪装成完整 E2E/TPOT 样本；
3. 本地与 OpenRouter 对 Qwen reasoning 的 SSE 表达不同，怎样定义同口径 TTFT。

**先发现并修掉的口径错误**

最初实现只把非空 `delta.content` 当首 token。但当前本地 vLLM 没开 reasoning
parser，Qwen 的 `<think>` token 也从 `content` 流出；OpenRouter 则把它们放在
`delta.reasoning`（或 `reasoning_content`）。所以只等正文会让两边 TTFT 不可比。

第一条 content-only smoke 的 261-token cap 全被 reasoning 用完，没有正文，最终
完整跑完而没有 cancel：客户端 E2E 7.373 s、client estimate `$0.0000958`；OpenRouter
generation record 的真实费用为 `$0.000094842`。因此 E10 把统一边界修成：

```text
first-generated-token TTFT
= 从请求发出到第一个非空 reasoning 或 content delta
```

结果行另存 `first_token_kind`；若后来真的出现正文，再单独存
`first_content_ttft_ms`，不能把两者混成一个指标。

**16-request burst 配置**

- 输入：heldout token-aligned trace 的前 16 行；
- materialized prompt 合计 7,658 tokens，decode cap 合计 4,501 tokens；
- trace arrival span 1 s；
- cloud concurrency 16；temperature 0；SLO 5 s；
- 未启用 OpenRouter response cache；
- 收到首个 reasoning/content token 后立即关闭流。

**结果**

| 指标 | 结果 |
|---|---:|
| 请求 / HTTP success / error | 16 / 16 / 0 |
| 客户端请求断流 / 完整完成 | 16 / 0 |
| first token 类型 | 16 reasoning / 0 content |
| TTFT min / p50 / p95 / p99 / max | 349 / 467 / 929 / 1,167 / 1,167 ms |
| 5 s TTFT 违约 | **0 / 16** |
| raw 中 cost pending | 16 / 16 |

注意：这里只能说 **16/16 客户端成功发起断流**，不能说 16/16 provider 已确认
取消。延迟查询 16 个 generation ID 时只找到 1 条；该条明确
`cancelled=true`，计 847 个 native prompt tokens、6 个 completion tokens，费用
`$0.000026`，专用 key 的累计 usage 也只增加了同样的 `$0.000026`。其余 15 条
generation record 当时为 not found，费用继续记 pending，不能擅自填 `$0`。

**真实 cloud 的新 headline 口径**

NullCloud 的 `pessimistic_combined` 只是假设上界。真实 cloud 应看：

```text
observed combined TTFT violation
= (local/cloud error、无首 token、或 arrival→first generated token > 5s) / N
= summary.overall.slo_violation_pct
```

并强制检查 `overall.slo_measured_n == overall.n`、`cloud.routed_only == 0`。历史 E7–E9
冻结的 pessimistic gates 不事后改写；只是以后 real-cloud 表不再拿那个字段做
headline。

**证据与等级**

- `$MSCRATCH/openrouter_ttft_cancel_20260716/`；
- burst16 raw SHA256：`e9bc6c8c72e98707ca08fd65b87a58523e6bff042ee1538a341e2453f1ccb9b5`；
- summary SHA256：`0a45bebed96e00fa19fcdd4ea81f379f33946cd02f627bf36b1c4b81980bc4d0`；
- billing audit SHA256：`e147071f5622a8b74a5c351ad3a525b5bddfa99c6c84c2deaeb60b2984033cab`。

**证据等级：MECHANISM ONLY / EXPLORATORY。** 固定 provider 的 transport、断流和
TTFT 语义已验证；n=16、单 provider、短 burst 不能代表完整云端分布，更不能说明
E2E、TPOT、正文质量、mid-stream reliability 或完整费用。

**临时实验假设**

2026-07-16 起，后续 TTFT 实验暂时假设“首 token cancel 不改变云端负载”。这是为
推进实验而采用的建模假设，**不是 E10 验证出的事实**。在此前提下，可以让 cancel
行进入 observed combined TTFT；需要完整服务语义的论文结论仍要补 full-drain
sensitivity leg。

**下一步顺序（后被 E11 决策覆盖）**

原计划依次跑 512 frozen-route shadow、512 live pilot、完整 11,605 A/C。Murphy
在 2026-07-16 明确选择跳过两个 512 步骤，直接进入完整 live-hybrid A/C；跳过的
便宜检查并没有被证明“没必要”，只是接受更高费用和配置出错风险来换取直接证据。

### E11 — 完整 11,605 请求 real-cloud live-hybrid A/C（预注册）

**状态：PRE-REGISTERED / NOT RUN。** 以下规则在启动正式 arm、读取正式结果前
写入 repo；结果出来后只能追加，不能回改 gate。

前置实现由 `4c18ae6` 固化：local/cloud 独立 `ignore_eos`、两段 cloud wait、
real-cloud matrix fingerprint/manifest 和 marker-v2 行级审计。83 个 router tests 与
41 个 tools tests 全绿。

**决策问题**

在同一个 TTFT 触发器、同一个本地部署和真实 DeepInfra 首-token TTFT 下，旧 V2
selector（A）与 current displacement selector（C）的 observed combined 5s TTFT、
本地安全性、外发比例和云端 gate wait 分别是多少？

**固定输入、arm 与执行顺序**

- 完整 token-aligned trace：11,605 行、1,199 s；SHA256
  `465ef070d2a4a399ad41142b9e40bd9c505599d05af2f9d4dd56f9eb02024c52`；
- manifest SHA256：
  `c5621d3e45f7b1e2ee49485f7948a267dde65d29b34a377247061f7cccc72f7a`；
- 新启动一个 Qwen3-32B no-prefix-cache lifecycle，并为它重新生成绑定当前
  PID、日志、代码 hash 的 schema-v2 profile；
- 冻结顺序 **A → C**（order-seed 标签 `20260716`）；A 为
  `ttft_pred:cost_cachedisp_old:0`，C 为
  `ttft_pred:cost_disp_current:0`；
- 两臂共用 trace、server lifecycle、profile、SLO=5s、temperature=0，臂间
  cooldown 20s；
- local：`local_ignore_eos=true`，prompt/decode 必须与 materialized token 精确一致；
- cloud：`qwen/qwen3-32b`，只允许 DeepInfra，fallback=false，不启用 response
  cache，`cloud_ignore_eos=false`，首个 reasoning/content token 后客户端断流，
  pre-first-token concurrency=16；
- 冻结 7 月 16 日 [OpenRouter DeepInfra 标价](https://openrouter.ai/qwen/qwen3-32b/pricing)：
  input `$0.08/M`、output `$0.28/M`，即两臂都使用 `in_price=0.08`、
  `out_price=0.28` 做 selector score 与费用估算；后续价格变化不回改本次 arm。

**冻结口径**

```text
cloud arrival→first-token TTFT
  = pre_route_queue_ms + cloud_gate_wait_ms + service_ttft_ms

primary = summary.overall.slo_violation_pct
```

错误、timeout、没有生成 token 就结束都计入 5s TTFT 违约分母；真实云结果不拿
`pessimistic_combined` 当 headline。必须同时报告 overall/local/cloud 违约与 TTFT、
route fraction、两段等待、HTTP/错误类型、requested/observed provider、已知与 pending
费用覆盖。两臂 cancel-mode OpenRouter 采用 `$1.20–$1.50` 的 operational budget
target；这是根据 E10 prompt-dominated billing 做的预算，不是 provider-side 硬上限。
Pending generation record 不得填成 `$0`。

**预注册 integrity gate**

1. 每臂恰好 11,605 个唯一 raw rows；marker 绑定 raw/summary/decisions、trace、
   manifest、profile、server PID/log 与不含密钥值的 cloud 配置；
2. `overall.slo_measured_n == overall.n == 11605`，且
   `cloud.routed_only == 0`；
3. applied victim IDs 与 cloud row IDs 完全一致；成功 local rows token exact；
4. 每条成功 cloud row 有有限、非负 arrival-to-first-token TTFT，且
   `stream_abort_requested=true`、`response_completed=false`；
5. 单请求 429/5xx/timeout 留在分母，不因结果难看而重跑删除。只有 hash/binding
   不符、系统性认证/配置拒绝、输出损坏或本地 server 死亡才中止并判实验无效。

**预注册 outcome 解读**

- retained-local 0 违约是安全目标，不是证据完整性的先决条件；
- 只有一个 A→C pair：能给 feasibility 和 effect estimate，不能直接声称已重复证明
  selector superiority；
- 若 `|A−C| < 1 percentage point`，结论必须写“未分胜负”，再跑 C→A；即使差值
  更大也必须标为有顺序限制的一次性结果；
- 不测完整响应 E2E、TPOT、答案质量、mid-stream reliability 或 full-response 成本；
- 暂时采用“首 token cancel 不改变上游 cloud load”的用户授权假设，但它仍未验证。

---

## 4. 算法是怎样一步步演化的？

```text
V0: FLOP weight
    问题：没有刻画 KV residence / TTFT 压力

V1/V2: token-seconds displacement
    贡献：提供“踢谁”的选择信号
    未解决：weight 应与什么在线 capacity 比？

Shipped v3: kv_gap trigger + current displacement selector
    问题 1：gauge 在 hybrid 上未必是 token KV
    问题 2：即使 gauge 是 token KV，KV 也未必是造成 TTFT 的绑定资源

实验 TTFT 版: ttft_pred trigger + 可替换 selector
    进展：A/C 在完整 cell 上 0 本地违约
    问题：B 使 predictor 离开支持域，selector-independent safety gate 失败

下一版目标: support-aware ttft_pred + old V2 selector
    support 内：按 TTFT 预测做最短 prefix spill
    support 外：保守 admission/spill fallback
```

这就是“为什么看起来从 V2 变成了 KV，又从 KV 变成 TTFT”的完整答案：

- V2 weight 一直属于 selection layer；
- 后来缺的是 online trigger/budget；
- 我们先尝试 KV gap，但实验发现它不能统一代表 TTFT 瓶颈；
- 所以 trigger 改成直接优化 TTFT，V2 weight 继续用于排序。

---

## 5. 当前未解决的欠账

下面任何一项没做完，都不能把实验候选直接改成默认：

1. **Support-envelope hardening**：以 sequence occupancy、shared-prefill work、
   workload mix 等标定 predictor；超出支持域时保守 fallback。
2. **完整 decision replay schema**：保存 snapshot IDs、waiting age、每请求
   prediction、selector score/order、in-flight release state。
3. **第二个完整 server lifecycle**：确认 E9 不是单次运行/顺序漂移。
4. **在线 decode estimate**：当前实验使用 trace/oracle decode 长度。
5. **Cache-aware trace**：当前 no-cache 实验不能验证 `U=P−cached_tokens` 项。
6. **真实 cloud latency**：E10 已验证固定 DeepInfra 的首-token transport/TTFT
   路径；E11 已预注册直接完整 11,605 A/C，两个 512 pilot 由用户明确跳过而非完成。
   Cancel 模式也不能替代 E2E/TPOT/mid-stream 的 full-drain sensitivity。
7. **负载/guard sweep 与 offline oracle**：确定性能前沿，并与历史 0/1 DP 或
   clairvoyant oracle 比较。

当前执行顺序（7 月 16 日决策后）：直接完成 E11 full A/C；随后
1 → 2 → 3 → 4/5 → 7。任何默认切换仍必须等 support-envelope hardening 完成。

---

## 6. 原始数据、结果文件与复算索引

这一节回答“表里的数字究竟在哪里”。它同时区分三件容易混淆的事：

1. **代码可复现**：repo 中存在 runner/analyzer；
2. **原始结果可追溯**：知道 GPU 箱上的具体目录和文件；
3. **证据可独立审计**：还有 manifest、hash、marker、server/profile 绑定。

只满足第 1 项，不等于历史数字还能被复算。当前 repo 本身没有提交任何 GPU
结果 JSONL；`$MSCRATCH` 下的数据需要单独保留或迁移到长期 artifact store。

### 6.1 一个完整 matrix arm 会产生什么

从 E6 开始，`experiments/run_ttft_selector_matrix.sh` 对每个 arm 使用统一文件组。
假设 stem 为 `<scenario>_<trigger>_<selector>_seed<seed>`：

| 文件 | 内容 | 用来回答什么 |
|---|---|---|
| `<stem>.jsonl` | 每请求原始行：endpoint、成功状态、arrival、queue/service/e2e、prompt/decode usage、成本 | 真实请求级结果是什么 |
| `<stem>.summary.json` | overall/local/cloud、TTFT 分位数、违约、成本、queue、token alignment、config | 文档中的汇总数字从哪里来 |
| `<stem>.decisions.jsonl` | 每次 Nimbus 决策、候选顺序、victims、pre/post prediction | 为什么踢这些请求；现 schema 仍不足以完整重放 |
| `<stem>.complete.json` | raw/summary/decision 的 SHA256、行数、arm 与 run fingerprint | 文件有没有被截断、换包或错配 |
| `matrix_manifest.txt` | commit、trace/profile/server/config hashes、arm 顺序 | 五个 arm 是否真的同口径 |
| `matrix_events.log` | arm 开始/结束时间与实际命令 | 实际按什么顺序运行 |
| `*.audit.json` / `*.audit.md` | analyzer 的机器可读/人可读审计 | 冻结 gate 如何判定 |

`summary.json` 不是原始数据，Markdown 表更不是原始数据。论文数字至少必须能追到
`summary → raw → marker → manifest/profile/trace` 这一条链。

### 6.2 当前数据完整度总览

| ID | 当前保存状态 | 能否从现有登记独立复算 | 主要缺口 |
|---|---|---|---|
| E0 | 代码在 repo；**neutrality/pacing/random 的 raw 已在箱上定位并镜像**（`router_step2/`、`router_final/`、`router_nimbus/`，见 §6.13） | **除 parity 锚点外均可复算** | parity（vs `vllm/run.py`）那一次的 raw 确实未保存；同-schema runner 已删 |
| E1 | **raw 三臂齐全**（`router_eb/`，已哈希并镜像）；25.3%/25.6% 冲突**已裁决**（见 §6.4） | **是** | 无 marker 合约（前合约时代实验） |
| E2 | 六个文件健在（mtime 2026-07-10 未动），**已哈希并镜像**；summary 另有本地抢救副本 | **是（机制级）** | 无 hash/marker 合约原生绑定；raw 从未被后期合约覆盖 |
| E3 | raw+summary 健在且**双站点哈希一致**（box == 本地 2026-07-12 scp 副本，`c3283c5a…`） | **是（机制级）** | 无后期 marker 合约 |
| E4 | probe 代码在 repo；**两次原始 stdout（phase 级）已抢救归档**；hybrid server log **原件健在已哈希**（`1e07e737…`）；dense server log 已被 07-14 启动覆盖 | **stdout 可核验（phase 级）** | 每 0.5s 采样行当时被过滤，永久丢失；dense server log 关键行只剩转录/文档二级证据 |
| E5 | **10 个文件精确 stem+哈希已登记并镜像**（§6.8） | **是（机制级）** | 前合约时代实验；server log 原件被覆盖（同 E4-dense） |
| E6 | trace/profile/anchor/smoke 均登记，**已镜像** | **是** | 第一份被拒 profile 的文件名未登记 |
| E7 | 三 arm 目录、marker fingerprint 均登记，**已镜像** | **是** | 单次固定顺序，只是探索证据 |
| E8 | 24 arms、profile、server log、analyzer 均登记，**已镜像** | **是** | 决策 schema 不能完整 replay selector |
| E9 | 五 arms、两份 audit、trace/profile/server 全绑定，**已镜像** | **是** | 同样缺完整 selector replay state |
| E10 | 代码在 `e0b6686`；burst16 raw/summary/billing audit 已登记并做 repo 外本地镜像 | **部分（TTFT 可复算）** | 15/16 generation metadata 未找到，完整费用 pending；只是单-provider probe |

**E0–E9 于 2026-07-15 在 GPU 箱上逐文件核验存在性与 SHA256，并完成双站点镜像**（全量
清单与镜像位置见 §6.13）。上表"已镜像" = 文件同时存在于 `$MSCRATCH` 与本地
Mac 镜像，且 232/232 哈希校验通过。E10 是 2026-07-16 单独新增的小型镜像，见
§6.14，不属于前述 232 文件清单。

### 6.3 E0 数据索引 — Router/harness 基础验证

- **结果出处**：`router/README.md` 与 `router/README.zh-CN.md` 的“Baseline 与
  验证记录”；本账本 E0 只是转录。
- **repo 代码**：`router/run.py`、`router/common.py`、`router/test_*.py` 与
  `tools/test_*.py`。
- **离线复算**：

```bash
python3 -m unittest \
  router.test_common router.test_run router.test_nimbus \
  tools.test_analyze_ttft_full_cell \
  tools.test_analyze_ttft_repeatability \
  tools.test_analyze_ttft_violation_context \
  tools.test_ttft_matrix_evidence
```

- **原 GPU 数据状态**：parity `1.003`、queue neutrality `110 vs 111 ms`、
  pressure pacing `756/756` 的原始 JSONL/summary 路径没有进入当前 artifact
  inventory；当时用于 parity 的同-schema runner 后来已删除。
- **因此能主张**：框架设计和离线证据测试当前可验证；历史 GPU 数字目前是
  summary-only anchor，正式投稿前应重跑并按 E6 之后的 marker 合约归档。

### 6.4 E1 数据索引 — Pre-v3 selection signal

- **结果出处**：`docs/v3_experiments_2026-07.md` §3.1，以及
  `router/README{,.zh-CN}.md` 的历史第 7 项。
- **raw/summary（2026-07-15 在箱上重新定位并镜像）**：`$MSCRATCH/router_eb/`
  三臂齐全 —— `all_local_eb1200.{jsonl,summary.json}`（2026-07-09）、
  `nimbus_eb1200.{jsonl,summary.json}`（2026-07-10）、
  `random_f256_eb1200.{jsonl,summary.json}`（2026-07-10）；运行日志
  `eb_leg.log`、`eb_nimbus.log`、`eb_random.log`。SHA256 见 §6.13 全量清单。
- **25.3%/25.6% 冲突已裁决（2026-07-15，依据 summary 原文）**：两个数字都对，
  指不同的量 ——
  - pre-v3 nimbus `actual_fraction = 0.25566…` → 主 handoff 的 **25.6%**（nimbus
    实际自选比例，也是 random 臂的 `target_fraction = 0.256` 的来源）；
  - random 臂 `actual_fraction = 0.25274…` → 旧 README 的 **25.3%**（i.i.d. 硬币
    在 0.256 目标下的实现值）。
  文档引用时应写 "random（target 0.256，实现 25.3%）"。headline 复核：random
  summary `ttft_p50_ms = 108,651.19`、`slo_violation_pct = 97.013` 与两份文档
  记载一致。
- **因此能主张**：raw 已恢复可复算；但该实验仍属 token-alignment 合约之前的
  直接 replay，只保留"同量级 shed 下 Nimbus p50 明显低于 random"的探索信号，
  不能用于精确 selector effect 或论文主结果。

### 6.5 E2 数据索引 — Rednote n=80

- **输入**：`$MSCRATCH/nimbus/data/rednote_slice.jsonl`。
- **运行脚本/日志**：`$MSCRATCH/v3_accept.sh`、`$MSCRATCH/v3_accept.log`。
- **结果目录**：`$MSCRATCH/router_v3/`。
- **两个 arm**：`v3_all_local_rednote.*` 与 `v3_nimbus_rednote.*`；至少应有
  请求级 `.jsonl` 和 `.summary.json`。
- **数字来源**：TTFT、违约、peak_inflight、cost 取两个 summary；
  `66.3%` 由 `(8 local violations + 45 routed) / 80` 手算。
- **审计限制**：这是 completion-marker 合约之前的实验，文档没有记录 trace/
  result hash；原始 runner 命令仅保存在 scratch 脚本中。

### 6.6 E3 数据索引 — Hybrid 35B full leg1

- **输入/实际命令**：`$MSCRATCH/v3_accept.sh` 的 leg 1；命令副本也在英文
  handoff §3.3。
- **raw**：`$MSCRATCH/router_v3/v3_nimbus_eb1200.jsonl`。
- **summary**：`$MSCRATCH/router_v3/v3_nimbus_eb1200.summary.json`。
- **headline 数字**：raw/summary 中的 routed/local、TTFT、违约、成本和 queue
  telemetry。
- **机制审计复算**：

```bash
python3 tools/analyze_eb1200.py \
  "$MSCRATCH/router_v3/v3_nimbus_eb1200.jsonl" 216512
```

该命令重建 kick timeline，并得到 kick 时 commitment p50 `20,841`、max
`45,840`。注意它只能用请求事件构造 token commitment **上界**；不是 engine
内部 KV occupancy 的直接采样。

### 6.7 E4 数据索引 — Gauge semantics probe

- **权威 probe 实现**：`tools/kv_gauge_probe.py`。
- **GPU 箱旧副本**：`$MSCRATCH/kv_gauge_probe.py`；repo 版本为准。
- **hybrid server 配置**：`$MSCRATCH/start_vllm_qwen36.sh`；相关启动输出在
  `$MSCRATCH/v3_accept.log`，但现有 inventory 没有把某个 log 与 probe run
  做 hash 绑定。
- **dense campaign 根目录**：`$MSCRATCH/router_v32b/`；dense probe 的独立
  stdout 文件名未登记。
- **重跑示例**：

```bash
python3 tools/kv_gauge_probe.py \
  http://127.0.0.1:8010 Qwen3.6-35B-A3B
```

- **当前证据边界（2026-07-15 更新）**：两次原始 probe stdout 已从会话工件中
  **抢救归档**（phase-summary 级；当时 grep 过滤掉了每 0.5s 采样行，采样行
  永久丢失）——`E4_hybrid_probe_stdout_2026-07-12.txt` 与
  `E4_dense_probe_stdout_2026-07-13.txt`，位置与哈希见 §6.13。
  **hybrid server log 原件健在**：`$MSCRATCH/router_step1/vllm_server.log`
  （pid 113957，07-11 23:35 启动，即 probe 所打的那个 server；SHA256
  `1e07e737…6ead`，含 `num_gpu_blocks_override=512`、`216,512 tokens`、
  `3.07x`、`enable_prefix_caching=False` 全部关键行），已镜像。
  **dense server log 原件已被 07-14 的再启动覆盖**（启动脚本用 `>` 截断），
  `112,064 tokens`/`2.74x` 关键行只剩会话转录与文档二级证据；教训见 §8 模板
  新增字段。论文前仍建议按新合约在 hybrid/dense 各重跑一次 probe。

### 6.8 E5 数据索引 — Dense 32B 历史 campaign

- **结果根目录**：`$MSCRATCH/router_v32b/`。
- **运行脚本**：`$MSCRATCH/v32b_accept.sh`、`$MSCRATCH/v32b_eb.sh`、
  `$MSCRATCH/v32b_rand.sh`。
- **server**：Qwen3-32B、FLASH_ATTN、max_num_seqs 128、KV 112,064；完整
  启动命令应从上述脚本和目录日志恢复。
- **结果来源（2026-07-15 已补全精确 stem 并镜像）**：`$MSCRATCH/router_v32b/`
  十个文件全部健在——
  `v32b_all_local_b1200.{jsonl,summary.json}`、`v32b_nimbus_b1200.{…}`、
  `v32b_all_local_eb1200.{…}`、`v32b_nimbus_eb1200.{…}`、
  `v32b_random_eb1200.{…}`；SHA256 见 §6.13 全量清单。运行日志
  `v32b_accept.log`、`v32b_eb.log`、`v32b_rand.log` 同步镜像（内含各 leg 的
  DONE 时间戳 marker：`V32B_ACCEPT_DONE 07-13 07:25:43`、
  `V32B_EB_DONE 07-13 08:32:52`、`V32B_RAND_DONE`）。
- **审计限制**：这是 token-alignment/complete-cohort 合约之前的实验；即使文件
  仍在，也只能用于机制诊断，不能恢复 selector 因果证明。server log 原件已被
  覆盖（见 §6.7）。

### 6.9 E6 数据索引 — Token-aligned trace 与 TTFT profile

- **根目录**：`$MSCRATCH/router_ttft_nocache_78846da/`。
- **512 输入与 provenance**：
  `heldout512_token_aligned.jsonl`、
  `heldout512_token_aligned.jsonl.manifest.json`。
- **完整 11,605 输入与 provenance**：
  `extreme_token_aligned.jsonl`、
  `extreme_token_aligned.jsonl.manifest.json`；trace SHA256
  `465ef070…24c52`，manifest SHA256 `c5621d3e…72f7a`。
- **7 月 14 accepted profile**：
  `ttft_profile_v2_bcda9d0.json`，SHA256 `549dd0bf…e23f`。
- **queue-pressure anchor**：`heldout512_all_local/`。
- **newest smoke**：`heldout512_ttft_smoke_bcda9d0/`。
- **同 lifecycle server log**：`vllm_qwen32b_nocache.log`。
- **生成/标定命令模板**：见 `router/README.zh-CN.md` 的“token 对齐的
  no-cache 实验前置条件”；实现分别是
  `tools/materialize_token_aligned_trace.py` 和 `tools/profile_ttft_batch.py`。
- **未登记项**：最初被 5s runner 拒绝、guard=`7,792 ms` 的 schema-v1
  profile 文件名没有写入 artifact inventory；诊断数字只保存在 handoff §5c。

### 6.10 E7 数据索引 — 512-request 三 selector 初筛

- **结果目录**：
  `$MSCRATCH/router_ttft_nocache_78846da/heldout512_selector_compare_bcda9d0/`。
- **共同输入/profile/server**：使用 E6 的 heldout512 trace、
  `ttft_profile_v2_bcda9d0.json` 和 `vllm_qwen32b_nocache.log`。
- **arms**：`cost_cachedisp_old`、`newest`、`cost_disp_current`，都使用
  `ttft_pred`；每个 arm 有 raw、summary、decisions、complete 四件套。
- **matrix 证据**：目录内 `matrix_manifest.txt`、`matrix_events.log`；三 marker
  共享 fingerprint `a0e9b148…23a1`。
- **表格来源**：routed/local、TTFT、local violations、cost 来自各
  `.summary.json`；marker 校验 raw/summary/decision hashes 和行数。
- **统计限制**：固定顺序只跑一次，不能把 E7 当作 old-v2 胜 current 的重复
  证据；这正是 E8 存在的原因。

### 6.11 E8 数据索引 — 六 block / 24 arms

- **campaign 根目录**：`$MSCRATCH/router_ttft_repeat_cac4a5d_20260715/`。
- **六个 block**：`block01_r0_abc/` … `block06_r5_bac/`；每个 block 含 R、
  A、B、C 四个 arm 的 raw/summary/decisions/complete，以及 manifest/events。
- **profile**：`ttft_profile_v2_cac4a5d_lifecycle1.json`，SHA256
  `ab703ddc…c819`。
- **server log**：`vllm_qwen32b_nocache.log`。
- **输入 trace**：E6 的 `heldout512_token_aligned.jsonl` 及 manifest。
- **复算命令**：

```bash
CAMPAIGN=$MSCRATCH/router_ttft_repeat_cac4a5d_20260715
python3 tools/analyze_ttft_repeatability.py "$CAMPAIGN"/block*
```

analyzer 以一个 512-request arm 为统计单位，验证 24 markers、72 个 raw/
summary/decision hash、token exactness 和 0 本地违约；不能把单个请求伪装成
24×512 个独立重复。

### 6.12 E9 数据索引 — 11,605-request 五臂 gate

- **结果根目录**：
  `$MSCRATCH/router_ttft_full_c6de62a_20260715/full11605_bkcal/`。
- **共同 trace**：E6 的 `extreme_token_aligned.jsonl` 及 manifest。
- **共同 profile/server**：E8 的 lifecycle-1 profile 和 server log。
- **arms/执行顺序**：B=`ttft_pred:newest:0` →
  K=`kv_gap:cost_disp_current:0` → C=`ttft_pred:cost_disp_current:0` →
  A=`ttft_pred:cost_cachedisp_old:0` → L=`anchor:all_local:0`。
- **每 arm 文件**：raw、summary、decisions（L 为 anchor，无 Nimbus decision
  语义）、complete；共同 `matrix_manifest.txt` 和 `matrix_events.log`。
- **冻结 gate 输出**：`full_cell_gate.audit.json`、
  `full_cell_gate.audit.md`。
- **B 违约诊断输出**：`B_violation_context.audit.json`、
  `B_violation_context.audit.md`。
- **复算命令**：

```bash
FULL=$MSCRATCH/router_ttft_full_c6de62a_20260715/full11605_bkcal
PROFILE=$MSCRATCH/router_ttft_repeat_cac4a5d_20260715/ttft_profile_v2_cac4a5d_lifecycle1.json
SERVER_LOG=$MSCRATCH/router_ttft_repeat_cac4a5d_20260715/vllm_qwen32b_nocache.log

python3 tools/analyze_ttft_full_cell.py "$FULL" \
  --json-out "$FULL/full_cell_gate.audit.json" \
  --markdown-out "$FULL/full_cell_gate.audit.md"

python3 tools/analyze_ttft_violation_context.py \
  --raw "$FULL/extreme_burst_1200_ttft_pred_newest_seed0.jsonl" \
  --decisions "$FULL/extreme_burst_1200_ttft_pred_newest_seed0.decisions.jsonl" \
  --summary "$FULL/extreme_burst_1200_ttft_pred_newest_seed0.summary.json" \
  --marker "$FULL/extreme_burst_1200_ttft_pred_newest_seed0.complete.json" \
  --manifest "$FULL/matrix_manifest.txt" --profile "$PROFILE" \
  --server-log "$SERVER_LOG" --server-log-arm-start "07-15 06:14:14" \
  --json-out "$FULL/B_violation_context.audit.json" \
  --markdown-out "$FULL/B_violation_context.audit.md"
```

`analyze_ttft_full_cell.py` 对这次数据应以 exit code `1` 结束：表示证据有效但
冻结 outcome gate 失败；exit code `2` 才表示缺文件、hash/token 不一致等无效
证据。不能为了 shell 显示绿色而把 exit 1 解释成“实验坏了”。

### 6.13 2026-07-15 箱上核验与双站点镜像记录

当日对 E0–E9 全部证据做了存在性核验、SHA256 清点和整体迁移：

- **箱上全量清单**：`$MSCRATCH/artifact_export_2026-07-15.SHA256SUMS`
  （232 个文件，覆盖 `router_eb/`、`router_v3/`、`router_v32b/`、
  `router_step2/`、`router_final/`、`router_nimbus/`、三个 `router_ttft_*`
  campaign 根目录、`nvml_shim/`、全部 accept/eb/rand 运行日志、
  hybrid `vllm_server.log` 原件与两份启动脚本）。
- **打包**：`$MSCRATCH/artifact_export_2026-07-15.tar.gz`，17,653,260 B，
  SHA256 `f0bbb305e34a741643eac7981f0cabee7bfb1f62e62de4e0f18947999a60911d`。
- **Mac 镜像**：`~/Desktop/Nimbus/artifacts/gpu_box_mirror_2026-07-15/`
  （在 repo 之外，不进 git）；解包后 **232/232 内层哈希校验通过**，241 MB。
- **会话工件抢救**：`~/Desktop/Nimbus/artifacts/salvage_2026-07-15/`
  （E2 summary 全文、E3 raw 的 07-12 本地副本——与箱上
  `c3283c5a…b23a` 哈希一致、两次 E4 probe stdout、当日原版分析/probe 脚本；
  内含 `PROVENANCE.md` 与 `SHA256SUMS.txt`）。
- **含义**：E0（除 parity 锚点）–E9 的原始证据现同时存在于两台机器且经哈希
  校验；`$MSCRATCH` 被清理不再导致"文档只剩路径"。真正的长期归档
  （对象存储/大文件仓）仍待定，但已不阻塞。

### 6.14 E10 数据索引 — OpenRouter TTFT cancel

- **远端结果根目录**：`$MSCRATCH/openrouter_ttft_cancel_20260716/`；
- **阶段**：`smoke1/`（错误的 content-only 口径）、`smoke2/`（修正后的单条
  cancel）、`smoke3_usage_delta/`（专用 key usage 差分）、`burst16/`；
- **代码**：`router/common.py`、`router/run.py`，实现由 `e0b6686` 固化；
- **burst16 raw**：`result.jsonl`，16 行，SHA256
  `e9bc6c8c72e98707ca08fd65b87a58523e6bff042ee1538a341e2453f1ccb9b5`；
- **burst16 summary**：`result.summary.json`，SHA256
  `0a45bebed96e00fa19fcdd4ea81f379f33946cd02f627bf36b1c4b81980bc4d0`；
- **事后计费审计**：`billing_audit.json`，SHA256
  `e147071f5622a8b74a5c351ad3a525b5bddfa99c6c84c2deaeb60b2984033cab`；
- **本地镜像**：workspace 的 repo-sibling
  `artifacts/openrouter_ttft_cancel_2026-07-16/`，不进 git；
- **复算**：TTFT 分位数和 `stream_abort_requested/response_completed/
  first_token_kind` 可直接从 burst16 raw 复算；费用只对查到的 generation 记录
  有证据，15/16 缺记录必须保留 pending。

---

## 7. 证据与复现入口

### Repo 内工具

| 工具 | 用途 |
|---|---|
| `tools/kv_gauge_probe.py` | 验证部署 gauge 随 token/并发如何变化 |
| `tools/analyze_eb1200.py` | 重建早期 leg-1 时间线与 kick 时资源上界 |
| `tools/materialize_token_aligned_trace.py` | 物化 token-aligned no-cache trace |
| `tools/profile_ttft_batch.py` | 生成绑定部署的 TTFT profile |
| `tools/analyze_ttft_repeatability.py` | 审计六 block / 24 arms 重复实验 |
| `tools/analyze_ttft_full_cell.py` | 验证五 marker 并计算冻结 gates |
| `tools/analyze_ttft_violation_context.py` | 诊断 B 的 39 个 TTFT 违约上下文 |
| `experiments/run_ttft_selector_matrix.sh` | 运行 server/trace/profile-bound matrix |

### 关键证据目录约定

不在 repo 提交机器名、用户名或绝对路径：

- `$MSCRATCH/router_ttft_repeat_cac4a5d_20260715/`：六 block campaign；
- `$MSCRATCH/router_ttft_full_c6de62a_20260715/full11605_bkcal/`：完整五臂
  matrix、markers、gate audit 和 B diagnosis；
- `$MSCRATCH/router_ttft_nocache_78846da/`：token-aligned trace 与早期 profile。
- `$MSCRATCH/openrouter_ttft_cancel_20260716/`：E10 real-cloud cancel probes。

完整 SHA、运行命令和 server 注意事项见
[`v3_experiments_2026-07.md`](v3_experiments_2026-07.md) 第 5g、8、9 节。

---

## 8. 以后每次实验必须追加的模板

以后新增实验，不再只记一个漂亮数字。复制下面模板追加到本账本：

```markdown
### EXX — <实验名>

**日期 / commit**
- 日期：
- clean detached commit：

**决策问题**
- 这个实验要决定什么？

**假设**
- 若假设成立，预期观察到什么？
- 什么结果会推翻它？

**固定配置**
- model / engine / GPU / cache mode：
- 设备健康状态与 workaround（哪块物理卡、GPU0 wedge 状态、NVML shim 是否
  生效、CUDA 序号是否因死卡重编号）：
- server log 文件名（必须含时间戳或 lifecycle 后缀——启动脚本用 `>` 截断，
  复用文件名会覆盖上一个 lifecycle 的原件，E4/E5 的 dense server log 即因此
  丢失）：
- trace hash / N / arrival span：
- profile hash / SLO / guard / seed：
- arms 与执行顺序：

**数据与产物**
- 输入 trace 路径或 artifact URI、SHA256、manifest：
- 结果根目录或长期 artifact URI：
- 每个 arm 的 raw / summary / decisions / marker 文件名：
- matrix manifest / events / server log / profile：
- analyzer 与完整复算命令：
- analyzer 预期 exit code：
- 文件存在性与 hash 最后核验日期：
- scratch 清理前迁移位置：

**预注册验收标准**
- integrity stops：
- outcome gates：

**结果**
- 成功/错误数：
- routed/local：
- TTFT p50/p95/p99：
- local/cloud/overall observed TTFT violation；若为 NullCloud 再单列
  `pessimistic_combined` assumed upper bound；cost 与 pending coverage：
- artifact marker/hash：
- 每个结果字段来自哪个 summary/audit key：

**结论**
- 支持了什么：
- 推翻了什么：
- 不能说明什么：

**下一步**
- 下一实验及其验收标准：
```

每次必须把“实验目的”“数据位置”“复算命令”和“不能说明什么”写全，否则这个
实验不能进入论文主张。仅写 `$MSCRATCH/some_dir/` 而没有文件名、hash 和 marker，
也不算完成归档。
