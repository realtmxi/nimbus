# Nimbus 当前算法与实验结论（截至 2026-07-21）

这是一份面向新读者的**当前真相页**：不需要聊天前情，就能知道 Nimbus 现在的
目标、算法、已经做过的实验、结果能证明什么，以及下一步为什么这样排。

冻结的实验合同、hash、命令和产物索引仍以
[`v3_experiments_2026-07.md`](v3_experiments_2026-07.md) 与
[`nimbus_experiment_ledger_2026-07.zh-CN.md`](nimbus_experiment_ledger_2026-07.zh-CN.md)
为准。历史 shipped-v3 的 KV 设计见
[`notion_algorithm_design_v3.zh-CN.md`](notion_algorithm_design_v3.zh-CN.md)；它不再代表
推荐方向。

阅读路径：只想知道现在算法，读 §0–3；看证据和边界，读 §6–8；准备继续实验，读 §9。

---

## 0. 一页结论

| 状态 | 当前事实 |
|---|---|
| Nimbus 兼容默认 | 选择 `--policy nimbus` 后，trigger/selector 默认为 `kv_gap + cost_disp_current`；已被证明不能在本次 dense-32B 压力 cell 上保证 TTFT |
| 已实现实验候选 | `ttft_pred + cost_cachedisp_old` |
| 下一版目标 | `support-aware ttft_pred + cost_cachedisp_old + online decode estimator` |
| 算法是否“推翻” | 没有。旧 V2 公式保留为“踢谁”的信号；被否定的是把 KV gap 当通用 TTFT trigger |
| selector 是否已有 winner | 没有。A 与 C 的一次真实 A→C ordered pair 仍按预注册规则判为 `unresolved` |
| 是否可切实验默认 | 还不可以。support fallback、在线输出长度估计、decision replay/消融和 selector 顺序鲁棒性尚未完成；生产上线另有更多 gate |

项目当前应优化的目标是：

```text
先让本地 + 云端的总体 TTFT 违约率满足预先冻结的 SLO 预算，
再最小化预期完整响应 API 成本；route count / 数据暴露是次级指标。
```

本轮实验把 TTFT 阈值固定为 5 秒，但生产可接受的总体违约率 `epsilon` 还没有正式
冻结。因此我们已经验证了算法方向，还没有闭合生产决策。

最重要的概念拆分是：

```text
系统状态 ──> Trigger：要不要外发、需要外发多少
                    │
                    v
              Selector：外发谁
                    │
                    v
              云端执行与总体 TTFT / 成本核算
```

- TTFT 是控制目标；
- sequence slots、prefill work、KV/状态池是 predictor 的部署特征；
- 旧 V2 token-seconds 是 selector 信号；
- 这三类量不能混成一个“统一 KV 背包预算”。

---

## 1. 先把名称说清楚

实验里反复出现的字母是 **arm 名称，不是算法版本号**：

| Arm | Trigger | Selector | 用途 |
|---|---|---|---|
| `L` | 无 | all-local | 压力锚点 |
| `K` | `kv_gap` | `cost_disp_current` | shipped-v3 / `--policy nimbus` 的默认 trigger/selector 兼容基线 |
| `A` | `ttft_pred` | `cost_cachedisp_old` | 当前主要实验候选；复用 exact old-V2 weight |
| `B` | `ttft_pred` | `newest` | naive selector / predictor 安全性压力测试 |
| `C` | `ttft_pred` | `cost_disp_current` | current-displacement 对照 |
| `R`（E8） | `ttft_pred` | `waiting_random` | 与 A/B/C 共用 trigger 的随机 victim-order 对照；普通 arrival-time `random` policy 另算 |

以后不应再用一个含混的“v3”同时指 shipped KV 版和 TTFT 实验版。本文统一使用：

- **shipped-v3**：`K = kv_gap + cost_disp_current`；
- **TTFT experimental candidate**：`A = ttft_pred + cost_cachedisp_old`；
- **next target**：在 A 上补 support fallback 与在线 `D_hat`。

---

## 2. 旧 V2 公式到底还在不在？

在，而且完整保留。

定义：

| 符号 | 含义 |
|---|---|
| `P` | historical old-V2 外层使用的完整 prompt length；本轮 no-cache 时可解释为 prompt KV，cache-aware 的 shared/marginal 语义尚未验证 |
| `U` | 本地尚未处理、未命中 prefix cache 的 prompt tokens |
| `D_hat` | 预计 decode tokens；当前实验实现仍使用 trace 中的 capped oracle `D` |
| `R_pre` | 当前部署标定的 shared-prefill throughput，tokens/s |
| `tau` | 当前部署标定的有效 TPOT，s/token |
| `b` | 单独标定的 first-token fixed overhead，s |
| `g` | TTFT guard，s |

旧 V2 weight 是：

```text
old_v2_weight(r)
  = P_r * (U_r / R_pre + D_hat_r * tau)                 [token*s]
```

它近似表达“这条请求占着 prompt KV 多久”。这里的 displacement 不是把某条 cache
真的搬走，而是**资源规模 × 驻留时间**：请求留在本地会占用多少 token-state、占多久。

当前 A selector 使用：

```text
expected_full_response_cloud_cost(r)
  = (P_r * input_price_per_M + D_hat_r * output_price_per_M) / 1e6

score_A(r)
  = expected_full_response_cloud_cost(r) / old_v2_weight(r)
```

按 `score_A` **升序**外发：花更少的钱、释放更多 token-seconds 的请求优先。

三个容易误解的点：

1. 旧 V2 没有被 KV 公式替代；它从“背包 weight/budget”位置移动到了 selector。
2. `old_v2_weight` 不与任何 token-second capacity 比较；线上不存在已验证的通用
   token-second 容量。
3. `cost_cachedisp_old` 是 greedy victim ordering，不是历史 0/1-knapsack DP，也不保证
   求出全局最便宜集合。

当前 C 对照使用：

```text
current_weight(r)
  = (U_r + D_hat_r) * (U_r / R_pre + D_hat_r * tau)

score_C(r)
  = expected_full_response_cloud_cost(r) / current_weight(r)
```

本轮关键实验全部是 no-cache，所以 `U=P`。因此它们**没有验证**原公式中
`U=P-cached_tokens` 的缓存项，也没有验证远端 provider cache 的折扣价格。真实
cache-aware cost 将来还需区分 remote cached/uncached input price。

---

## 3. 当前 TTFT 候选算法如何工作

权威实现位于 [`../router/nimbus.py`](../router/nimbus.py)。runner 在到达、完成和周期 tick
处调用 policy；完整同刻到达 cohort 会先进入外部 FIFO，再做一次决策。

### 3.1 同部署标定

在同一组 GPU、模型、engine、并发配置、prefix-cache 设置和 MTP/speculative mode 下标定：

- `max_inflight` / sequence slots；
- shared-prefill throughput `R_pre`；
- decode `tau`；
- first-token overhead `b`；
- guard `g`，并记录 calibration support cells/domain（runtime 的显式 envelope 判定仍待
  实现）。

这些参数不能跨 deployment 静默复用。E12 fresh retry 的绑定 profile 例如为：

```text
max_inflight = 128
R_pre        = 3247.990163 tokens/s
tau          = 151.213123 ms/token
b            = 372.804523 ms
g            = 1712 ms
SLO          = 5 s
```

该 profile 的 held-out classifier 为 421 条、FN=0；这只证明绑定 cell 的 5 秒分类，
不等于 predictor 在所有负载上都有零 false negative。

### 3.2 预测 waiting queue 的 TTFT

predictor 同时近似两个本地资源：

1. `max_inflight` 个 sequence slots；
2. 一条 shared-prefill compute lane。

对 waiting request `j`，令：

- `age_j` 为它已经等待的时间；
- `slot_ready_j` 为最早 sequence slot 的预计释放时间；
- `prefill_ready` 为 shared-prefill lane 完成前面工作的时间。

则代码中的核心近似为：

```text
prefill_start_j = max(slot_ready_j, prefill_ready)
prefill_done_j  = prefill_start_j + U_j / R_pre

predicted_TTFT_j
  = age_j + prefill_done_j + b

predicted_slot_release_j
  = prefill_done_j + b + max(D_hat_j - 1, 0) * tau
```

已在 decode 的请求按“剩余 decode tokens × `tau`”估计 slot release；已 admission、尚未
产生首 token 的请求，其完整 prefill work 也会进入 shared lane。128 个空 sequence
slots 因而不再被错误理解成 128 条独立 prefill lanes。

当前 predictor **不读取 KV gauge**。它会读取 in-flight 状态，但 post-kick 声明的范围
只是仍在 waiting queue 的 survivors；已经 in-flight 的请求不在这条预测保证里。

### 3.3 Trigger：何时、踢多少

```text
deadline = SLO - guard

on arrival / completion / periodic tick:
    pred = predict(waiting, inflight, sequence_slots, shared_prefill_lane)

    if max(pred) <= deadline:
        不外发
    else:
        ranked = 按 selector score 排序 waiting requests
        找最小 k，使删除 ranked[0:k] 后
        所有 waiting survivors 的 predicted_TTFT <= deadline
        把这 k 条路由到 cloud
```

实现不是逐条盲踢，而是对固定 ranking 二分搜索最短 prefix；删除更多 victims 不会让
剩余 FCFS waiting request 更晚。

### 3.4 当前 Nimbus 默认 K 的精确规则

为兼容保留的 shipped-v3 不是上述 TTFT trigger，而是：

```text
footprint(r) = U_r + D_hat_r

gap = max(0,
    sum(footprint(waiting))
  + sum(D_hat - generated_tokens for inflight)
  - KV_headroom)

release_target = gap + 0.05 * KV_headroom
```

`gap>0` 时，K 按 `score_C=cloud_cost/current_weight` 排序，累计踢出的 `footprint`
达到 `release_target` 后停止。`KV_headroom` 取 live gauge 换算值与 client-known in-flight
当前占用上界中的更保守者，代码等价于：

```text
KV_headroom = min(
    live_gauge_available_tokens,
    KV_capacity - sum(U + generated_tokens for inflight))
```

in-flight 的未来 decode 增长不塞进 headroom；它以
`sum(D_hat-generated_tokens)` 单独进入上面的 `gap`。

这套规则只有在部署 probe 与 token-dominated occupancy 一致、且 KV 确为当前 TTFT 绑定资源时，
才有正确物理含义。E9 的 dense cell 满足前一项、不满足后一项，所以 K 严重欠踢。

### 3.5 实验默认切换与生产上线是两道不同的门

阻止把 A 提升为 repo **实验默认**的事项：

1. **Oracle decode**：E12 的实验 `D` 来自历史 completion token 数再 cap 到 1,024。
   线上能知道 client 请求的 `max_tokens` cap，却不知道实际 completion 长度；需要 estimator
   消融，或明确接受“用 cap 做保守上界”的路由代价。
2. **Support fallback**：状态离开 calibration coverage 时还没有显式的保守
   admission/spill fallback。
3. **可重放性与因果证据**：decision log 尚不足以精确 replay；predictor 组件消融和
   selector 反序/平衡重复尚未完成。

即使上面完成，阻止**生产上线**的事项仍包括：

1. **Cloud feasibility gate**：trigger 只判断本地 waiting survivors；路由前不判断 provider
   TTFT、错误率、concurrency gate 或 provider 是否还能作为安全出口。
2. **目标预算未冻结**：5 秒阈值已定，但总体允许违约率 `epsilon` 尚未预注册。
3. **部署动态未覆盖**：`tau` 仍是静态标定值，尚未建模 batch、MTP acceptance 和负载变化；
   cache-aware 语义也未验证。
4. **完整服务语义未测**：首-token cancel 不覆盖 full-drain E2E、mid-stream reliability、
   答案质量和真实完整账单。
5. **仍是实验 harness**：`router/run.py` 是可审计 trace-replay runner，不是生产 ingress
   服务。

下一版目标才是：

```text
if current_state is outside calibrated support:
    conservative admission/spill fallback
elif predicted waiting TTFT is unsafe:
    route the minimum old-V2-ranked prefix
```

---

## 4. 为什么曾经“变成 KV”，现在又回到 TTFT

| 阶段 | Trigger / budget | Selector / weight | 审计后的结论 |
|---|---|---|---|
| V0 | TTFT violation loop | FLOP weight | 历史起点；没有刻画 residence |
| V1 | 未闭合 | `P * D` cache-displacement proxy | 建立“占多少 × 占多久”的直觉 |
| V2 | 没找到同单位在线 capacity | `P * (U/R_pre + D*tau)` | selector 信号保留，不能再当通用背包 weight |
| shipped-v3 | KV token gap | current displacement | 在特定 KV-bound 部署上单位可闭合；不能保证通用 TTFT |
| TTFT experimental | predicted TTFT | old/current/newest 可替换 | A/C 在关键 cell retained-local 为 0；仍有 support/oracle 缺口 |
| next target | support-aware predicted TTFT | old V2 + online `D_hat` | 尚未完成 |

当时采用 KV gap，是为了解答 V2 的开放问题：“weight 是 token-seconds 时，budget 应是
什么？”shipped-v3 把触发层改成 token footprint 与 token KV headroom 比较，让单位先
闭合。但实验后来发现：

- hybrid 模型的同名 gauge 主要像 per-sequence state pool，不是 token KV；
- dense 模型上的 phase evidence 与 token-dominated occupancy 一致，足以作为本次实验
  anchor，但不是精确 fit；该 workload 的绑定资源证据仍指向 sequence/prefill compute，
  KV 没先满；
- 所以“metric 单位正确”仍不等于“它是造成 TTFT 的绑定资源”。

结论不是“KV 不重要”，而是 KV 只能作为 deployment-profiled feature/fallback 信号，
不能定义一个跨架构的 TTFT trigger。

---

## 5. 两条 workload 为什么是 11,605 和 11,604

| Workload | 行数 | 保留了什么 | 牺牲了什么 | 回答的问题 |
|---|---:|---|---|---|
| synthetic token-aligned | 11,605 | 原 trace 的累计 token 长度和到达压力 | filler 不保留真实文本语义 | 调度机制、压力与 selector 泛化 |
| ShareGPT current-turn | 11,604 | 真正发送的 current-turn 文本、相同到达时序，并用 Qwen3-32B tokenizer 重计 token | 不再保留累计长上下文分布；一个 221k-token outlier 按冻结规则 drop | 真实 payload 语义和 cloud 外部有效性 |

两条腿的 estimand 不同，不能平均，也不能拿一条替代另一条。早期直接 replay 曾把“累计
conversation token metadata”配到“仅 current user turn 的 HTTP payload”上；E6 以后所有
selector 证据都要求 scheduler token 与实际 payload 精确对齐。

---

## 6. 实验总览

### 6.1 当前结论所依赖的主证据

| ID | 问题与设置 | 关键结果 | 能说明什么 | 不能说明什么 |
|---|---|---|---|---|
| E0 | router 与可信 open-loop harness 是否一致 | 历史 parity summary p50 ratio `1.003`；可复算 queue neutrality `110 vs 111 ms`；当前离线测试 `296/296` | harness 基础可信 | policy 本身有效；parity 原始 raw 已缺失，需视作历史 summary anchor |
| E4 | 同名 KV gauge 的语义是否跨模型一致 | hybrid：32 路小请求 phase max gauge `76.48%`，观察到 max running 约 41；2×8k phase max gauge `3.40%`。dense：64 路 gauge `11.89%` 且全运行 | hybrid 明显受 sequence state 主导；dense 更接近 token KV | 精确每-sequence 系数、硬并发上限，或 KV 是所有部署的 TTFT trigger |
| E8 | selector 信号是否跨顺序/重复稳定 | 6 blocks / 24 arms；old V2 对 newest `6/6` 胜，平均少路由 `50.17`；old/current 路由差均值 `-1.5`，old 成本 `6/6` 更低、平均约 `7.2%` | old V2 是稳定的选择信号；与 current 路由数等价 | old V2 已普遍胜 current |
| E9 | 11,605 synthetic full-cell 泛化 | A/C retained-local 均 `0`；K 为 `5,249/5,545` 违约；B 为 `39/3,014`；L 为 `11,544/11,605` | TTFT trigger 方向；KV trigger 在该 cell 失败；A/C 值得继续 | predictor 对所有 selector/support 状态都安全；cache/online-D 有效 |
| E12 local | 11,604 真实 current-turn、NullCloud | L `11,400/11,604` 违约；A/C retained-local 均 `0`；A/C 路由 `5,109/4,099` | 结果不是 synthetic filler 独有 | 全体用户真实 cloud TTFT |
| E12-live retry | 同一 current-turn trace，真实 OpenRouter/DeepInfra，首-token cancel，固定 A→C | A/C 路由 `5,097/4,057`；overall 5s 违约 `11/3`；retained-local 均 `0` | 一组真实 cloud hybrid pair 完整、两臂总体 TTFT 都低 | selector winner；完整响应 E2E/可靠性/账单 |
| E12 reprice | 对 live 实际 routed cohort 零出网复算 | A/C full-cap model `$0.49967044/$0.51352936` | 在 trace capped-D 模型下 A 更便宜 | 实际 full-drain provider bill |

E4 留存的是每个 phase 的独立 max-usage/max-running stdout，原始 0.5 秒时间序列已丢；
两项峰值不能假装来自同一时刻。因此架构方向结论成立，精确每-sequence 系数和“41 是
硬上限”仍需重跑 probe 才能主张。

成本只允许在同一实验、同一价格和同一 workload 内比较。E8/E9 使用 input
`$0.15/M`、output `$1.20/M`；E12 使用 DeepInfra `$0.08/M`/`$0.28/M`。E12 local 与
E12-live reprice 的 routed cohorts 也不同，所以两组约 `$0.50` 的数字不是重复测量或冲突。

### 6.2 E9：完整 synthetic cell 把 trigger 与 selector 分开

| Arm | Trigger + selector | 路由 / 本地 | retained-local 5s 违约 | NullCloud 建模成本 |
|---|---|---:|---:|---:|
| A | TTFT + old V2 | 6,748 / 4,857 | **0** | **$3.344049** |
| B | TTFT + newest | 8,591 / 3,014 | **39 (1.294%)** | $3.643557 |
| C | TTFT + current | 6,729 / 4,876 | **0** | $3.519140 |
| K | KV gap + current | 6,060 / 5,545 | **5,249 (94.662%)** | $3.477171 |
| L | all-local | 0 / 11,605 | **11,544 (99.474%)** | $0 |

这张表支持三个结论：

1. K 没有把该 cell 的 TTFT 绑定资源建模对；
2. A 与 C 都能在该次运行中保护 retained-local；
3. B 的 39 个违约在 dispatch 时，其 planned `prompt + requested decode` commitment
   全部超过 profile 观测最大值 98,304，也超过 declared KV proxy 112,656；这与离开
   标定覆盖一致，但 runtime 当时没有一个事前定义的 support-envelope classifier。
   因此不能宣称 predictor 已经 selector-independent safe。

A/C 路由数只差 19，位于冻结等价带内；A 建模成本低 4.98%。这是继续研究 A 的理由，
不是默认切换授权。

### 6.3 E12 local：真实 current-turn payload 的零出网 gate

| Arm | NullCloud 路由 | 留在本地 | retained-local 违约 | local TTFT p50 / p95 / p99 | full-cap 建模成本 |
|---|---:|---:|---:|---:|---:|
| L | 0 | 11,604 | **11,400** | 644,578 / 1,304,490 / 1,363,365 ms | $0 |
| A | 5,109 | 6,495 | **0** | 1,393 / 2,003 / 2,176 ms | $0.500634 |
| C | 4,099 | 7,505 | **0** | 1,490 / 2,165 / 2,490 ms | $0.51621328 |

A 多路由 1,010 条，却因为选中更便宜的 cohort，full-cap 建模成本低 3.018%。但 cloud
是 NullCloud，所以这里的“0 本地违约”绝不能写成“全部 11,604 用户 0 违约”。

### 6.4 E12-live：真实 cloud 的一次 A→C ordered pair

旧 lifecycle 只完成 canary+A，C 从未启动，已永久标记 `TERMINAL PARTIAL`；其
`$0.02670220` 仍计入授权预算，不能与另一个 lifecycle 的 C 拼接。

fresh retry 的有效 pair 为：

| 指标 | A：old V2 | C：current |
|---|---:|---:|
| total / success | 11,604 / 11,601 | 11,604 / 11,604 |
| local / cloud | 6,507 / 5,097 | 7,547 / 4,057 |
| overall TTFT p50 / p95 / p99 | 1,126.482 / 1,983.963 / 2,451.606 ms | 1,306.939 / 2,123.023 / 2,496.230 ms |
| overall 5s violation | 11 = **0.0947949%** | 3 = **0.0258532%** |
| retained-local violation | **0 / 6,507** | **0 / 7,547** |
| timeout / HTTP 429 | 3 / 0 | 0 / 0 |
| settled first-token spend | `$0.02720312` | `$0.01973248` |

C 少路由 1,040 条、少 8 个 overall 违约；A 的 TTFT p50/p95/p99 反而更快。违约率绝对
差只有 0.06894 个百分点，低于预注册的 1 pp 门槛，而且只有一次固定 A→C 顺序，所以
selector 结论仍是 **unresolved**。

TTFT 分位数只统计成功且有有限 TTFT 的 rows：A 的 3 条 timeout 不进分位数，但仍作为
失败进入 11/11,604 的 overall 违约分子。Fresh retry（含 canary）花费 `$0.04693796`；
加上旧 partial 后累计 `$0.07364016`，低于授权的 `$3` 上限。

这一 live lifecycle 没有同期 all-local arm；强压力锚点 `L=11,400/11,604` 来自较早的
current-turn local-gate lifecycle，不能包装成同 lifecycle 的 A/C-vs-L 因果估计。

对这两个实际 routed cohort 的零出网 full-cap 复算为：

| Arm | routed | prompt tokens | capped decode tokens | modeled full-cap cost |
|---|---:|---:|---:|---:|
| A | 5,097 | 1,013,293 | 1,495,025 | **$0.49967044** |
| C | 4,057 | 730,784 | 1,625,238 | **$0.51352936** |

C 虽少 1,040 个 API calls，却多选了 130,213 个 capped decode tokens，故该模型下比 A
贵 `$0.01385892`。这与 cancel 实验中 C 的实际 spend 更低不矛盾：两组数字回答的是
不同问题。

### 6.5 历史与机制证据

| ID | 结果 | 当前用途 |
|---|---|---|
| E1 pre-v3 | Nimbus 25.6% 外发、本地 p50 12.4s；matched random 实际 25.3%、p50 108.7s；all-local 324.8s | 说明“踢谁”有信号；token/payload 与同刻候选审计不足，不能证明具体 selector |
| E2 rednote n=80 | all-local 67.5% 违约；v3 retained-local 22.9%，若把 NullCloud routes 全算违约则 66.3% | 提醒本地改善不能代替总体 SLO |
| E3 hybrid full | v3 路由 3,931，retained-local 0；但踢出时可重建 token commitment max 仅 45,840/216,512 | 延迟结果真实，原“token KV 满”机制解释错误 |
| E5 dense full | KV v3 路由 38.6%，retained-local 64.1% 违约；peak inflight=128、KV gauge 最高约 69% | 即使 gauge 表现与 token-dominated occupancy 一致，KV 仍不是该 cell 的充分 TTFT trigger |
| E6 profile/audit | shared-prefill schema-v2，两次 lifecycle profile 接近，held-out 5s FN=0 | 后续 selector 对照具备共同 stop rule |
| E7 512 screen | old/current/newest 路由 257/266/326，均 retained-local 0 | 产生 E8 假设，单次不可定胜负 |
| E10 cloud probe | DeepInfra 16/16 首 generated token 后客户端断流；TTFT p50/p95/p99 467/929/1,167ms，0/16 超 5s | 验证 transport/TTFT 口径，不代表完整 cloud 分布或 provider 已全确认 cancel |
| E11 synthetic live | fresh profile 通过；formal A/C 在 bulk third-party data-export gate 前停止，0 请求/0 费用 | arms 未执行；不能当结果 |

---

## 7. 指标和成本口径

| 名称 | 它回答什么 | 不能怎样解读 |
|---|---|---|
| retained-local violation | predictor/selector 留在本地的人是否安全 | 不代表所有用户；被 route 的人不在分母 |
| `overall.slo_violation_pct` | real-cloud 本地+云端的主 TTFT 指标；失败/缺 TTFT 也算违约 | 不能只看成功请求分位数代替它 |
| `pessimistic_combined` | NullCloud 历史假设上界：把每条 cloud route 都算违约 | 不是 real-cloud headline，也不是真实云“悲观测量” |
| route fraction | API calls、原文暴露量和运营复杂度 | 不等于美元成本；请求 token mix 不同 |
| settled first-token spend | 首-token-cancel 实验的实际账户增量 | 不是生产完整响应成本 |
| modeled full-response capped cost | 假设每条 route 生成满 trace capped `D` 的反事实成本 | 不是实测 full-drain bill；provider 可能提前 EOS/tokenizer 不同 |
| full-drain bill / E2E | 完整返回的实际费用、可靠性和延迟 | 尚未由 E12-live 测量 |

E10/E12-live 的“first token”定义为第一个非空 reasoning **或** content delta，以匹配
绑定的本地 vLLM 流；它不一定是用户看到的第一个正文 token。
实验按用户授权采用了“客户端 cancel 不改变上游 cloud load”的临时假设；这不是 E12
证明出的 provider 行为。

项目的优化目标若是“满足 TTFT 后最小化完整响应美元成本”，本次 A 的 full-cap 模型
更好；若把最少 API calls/最少原文暴露设为第一目标，本次 C 更好。目标函数和允许违约
预算未冻结前，不能把其中一个偏好偷写成 selector winner。

---

## 8. 当前能说、不能说什么

### 已有证据支持

- 应继续用 TTFT violation 作为 trigger 方向，而不是回退到 universal KV gap。
- KV/状态池仍重要，但只能作为 deployment-profiled feature 或 support fallback 输入。
- old V2 没有失效；它是有实验证据的 victim-ordering signal。
- A/C 在 synthetic full cell、真实 current-turn local gate 都实现 retained-local 0 违约。
- 一组真实 A→C pair 中，两臂总体 5 秒 TTFT 违约率都低于 0.1%。
- route count 与美元成本可以反向：C 少路由，A 的 capped-D full-response 模型更便宜。

### 证据尚不支持

- old V2 已经普遍胜过 current selector；
- TTFT candidate 已可替换 repo 默认或直接生产部署；
- cached-token 项、provider cache price 或 prefix-cache displacement 已验证；
- trace/oracle `D` 可以被线上 predictor 无损替代；
- predictor 在 support envelope 外仍安全；
- 首-token cancel 代表完整响应 E2E、mid-stream reliability、答案质量或实际 full bill；
- MTP 对调度的影响可由一个跨 batch 的固定 TPOT 完整代表。

本轮 dense TTFT campaign 的 server 记录为 `speculative_config=None`；早期 hybrid
框架虽使用过 MTP，也不能据此声称当前候选已经对 MTP 鲁棒。

---

## 9. 下一步实验顺序

### P0：先做 predictor 消融，零出网

对 predictor 的因果比较固定 selector=`cost_cachedisp_old`、相同 workload/profile/server
lifecycle，只改变 trigger 模型。先在冻结的 512-request slice 上单独跑一次 L 和
K-default 作为非因果 anchors；再对下面六个 core arms 做 6×6 Latin-square 平衡顺序
screen。只有通过安全 gate 的 arms 才进入 11,604/11,605 full-cell 确认：

| Arm | 改动 | 回答的问题 |
|---|---|---|
| `K-old` | `kv_gap + cost_cachedisp_old` | 与 T 只差 trigger 的正交对照 |
| `T-full` | 完整 `ttft_pred` | 当前候选基线 |
| `T-no-decode` | predictor 不计 decode slot residence | decode/TPOT 压力贡献多大 |
| `T-no-shared-lane` | waiting requests 不共享 aggregate prefill lane，改为仅受各自 slot 约束；仍保留 age/inflight slot state | shared-prefill serialization 是否必要 |
| `T-no-guard` | `guard=0` | 安全余量贡献和过度路由代价 |
| `T-estD-pred-only` | 只在 predictor 中用在线 `D_hat`；selector、cost 与 footprint 仍冻结 oracle-D | estimator 对 trigger safety 的独立影响 |

非因果 anchors 为：`L=all-local` 压力锚点，以及
`K-default=kv_gap + cost_disp_current` 当前兼容默认。它们不进入上述 predictor-only
causal contrast。

实现消融时必须把 predictor 参数与 selector 参数解耦；例如 `T-no-decode` 不能顺手把
old-V2 selector 中的 `tau` 也清零，否则同时改了两个层。

验收顺序：

1. retained-local 5 秒违约 / false negative 是硬 gate；
2. 按 load、sequence occupancy、prefill backlog 分桶检查 support envelope；
3. 安全后再比较 full-cap cost、route fraction 和 decision latency；
4. 保存完整 decision replay：snapshot IDs、age、每请求 prediction、score/order、
   in-flight release state。

### P1：support-envelope 与 fallback

根据 P0 找到必须保留的 predictor 组件，显式判断 live state 是否在 calibration support
内；support 外走保守 admission/spill，而不是继续相信外推数值。

### P2：在线输出长度与 cache-aware 语义

先在 no-cache 条件下比较 oracle-D 与在线 `D_hat`。除 P0 的 predictor-only arm 外，还要
做一个 2×2：predictor 使用 oracle/estimated D × selector/cost 使用 oracle/estimated D。
其中全在线 arm 让同一 `D_hat` 同时进入 predictor、old-V2 selector 分母、cloud-cost
分子和所有依赖 D 的 resource estimate。之后才生成有 prefix-cache hit 的 workload，
验证 `U=P-cached_tokens` 和远端 cached-price 项；online-D 与 cache 不要在一个实验里
同时改变。

### P3：cloud feasibility gate

因为最终目标是 overall local+cloud TTFT，控制面必须在 route 前检查固定 provider 的近期
TTFT/error/concurrency 支持范围；不安全、过期或不可用时 fail closed 到另一条明确策略，
不能继续把 cloud 当无条件救生舱。先用 synthetic probe/历史聚合做 gate calibration，
再用小规模 real-cloud 验证；不需要直接重发全部敏感 current-turn workload。

### P4：反序/平衡 live selector 重复

在 predictor hardening 后再跑 fresh C→A 或 balanced blocks，估计顺序效应。任何新的
原始 ShareGPT 外发与费用都必须获得新的明确授权、同日价格快照和累计 budget gate；
现有 A→C 不授权自动追加 POST。

### P5：MTP 与 full-drain sensitivity

MTP 不应作为一个手写常数惩罚项塞进算法。应把同一 deployment 分成：

```text
MTP off / on  ×  static TPOT / batch-aware TPOT
```

重新 profile `TPOT_eff(mode, batch, acceptance, load)`，再看 predictor safety 和 routing/cost
前沿是否变化。最后用小样本 full-drain leg 测实际完整费用、E2E、mid-stream reliability
和答案质量；不需要再次发送全部 11,604 条才能回答 sensitivity。

---

## 10. 代码、证据与复现入口

| 入口 | 内容 |
|---|---|
| [`../router/nimbus.py`](../router/nimbus.py) | trigger、predictor、old/current selector 公式 |
| [`../router/run.py`](../router/run.py) | 外部 FIFO、同刻 cohort、dispatcher、decision log、CLI |
| [`../router/README.zh-CN.md`](../router/README.zh-CN.md) | runner 用法与指标语义 |
| [`v3_experiments_2026-07.md`](v3_experiments_2026-07.md) | 冻结英文 handoff、合同、hash、命令与产物 |
| [`nimbus_experiment_ledger_2026-07.zh-CN.md`](nimbus_experiment_ledger_2026-07.zh-CN.md) | E0–E12 中文时间线与原始数据索引 |
| [`../tools/kv_gauge_probe.py`](../tools/kv_gauge_probe.py) | hybrid/dense gauge 语义探针 |
| [`../tools/analyze_eb1200.py`](../tools/analyze_eb1200.py) | E3 token commitment 重建 |
| [`../tools/analyze_ttft_full_cell.py`](../tools/analyze_ttft_full_cell.py) | E9 五臂完整 gate |
| [`../tools/analyze_ttft_current_turn_gate.py`](../tools/analyze_ttft_current_turn_gate.py) | E12 local L/A/C 审计 |
| [`../tools/audit_e12_live.py`](../tools/audit_e12_live.py) | E12-live 最终完整性/预算审计 |
| [`../tools/analyze_e12_full_response_cost.py`](../tools/analyze_e12_full_response_cost.py) | 实际 routed cohort 的零出网 full-cap 复算 |

离线验证（无网络、无 GPU、不会发送 prompt）：

```bash
python3 -m unittest discover -s router -p 'test_*.py'
python3 -m unittest discover -s tools -p 'test_*.py'
```

本次文档整理所在 working tree 已执行 router `105/105` + tools/evidence `191/191`，合计
`296/296`。GPU 原始请求级证据位于受限 `$MSCRATCH`；不得把原文、机器名、用户名、
secret 或绝对 scratch 路径提交进 repo。
