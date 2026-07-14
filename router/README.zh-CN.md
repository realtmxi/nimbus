[English](README.md) | **简体中文**

# router/ — hybrid routing 框架

把 trace 里的请求按 policy 分发给本地 vLLM 和云 sink,记录延迟、成本和分流比例。
**单一入口**:`python -m router.run`。policy:三个 baseline(`all_local`/`all_cloud`/
`random`)+ **nimbus** 甩负载 policy(cache-displacement,见下)。


## 文件

| 文件 | 作用 |
|---|---|
| `run.py` | **唯一入口**:外部 FIFO + work-conserving dispatcher + KV 读数器 + CLI |
| `common.py` | 共享库:`one_request`/`load_trace`/`SCENARIOS`(逐行取自 `vllm/run.py` @ `dff1a81`)、`Endpoint`、`Policy`、`NullCloud`、计费、`summarize` |
| `nimbus.py` | Nimbus v3 baseline + 正交的 KV-gap / 预测 TTFT 触发器与 victim-selector 消融 |
| `test_run.py` / `test_common.py` / `test_nimbus.py` | 63 个单元测试,无需网络/aiohttp/GPU |

## 架构

```
到达 ──Policy(到达时决策)──cloud──> fake sink(默认)/ 真实云
        │local
        v
   [外部 FIFO] ──work-conserving dispatcher──> vLLM(内部队列≈空)
```

- **Policy**:`all_local` / `all_cloud` / `random --fraction f --seed s`——同一规则的
  p=0/1/f,i.i.d. 抛硬币,不读任何系统状态;同 seed 逐请求决策可复现
- **dispatcher** 只要有空位就放行(绝不无谓扣请求):`inflight < max_inflight`,
  与 server 的 `--max-num-seqs` 对齐 ⇒ vLLM 内部队列≈空
- 无压力 ⇒ 队列恒空 ⇒ 行为退化为 open-loop(≡ `vllm/run.py` 的跑法)
- 有压力 ⇒ 溢出堆在**我们的**队列里(带完整身份)——nimbus 甩负载 policy 的操作对象。
  为什么队列必须自己维护:引擎只暴露排队**计数**(vLLM `/metrics` 三个 gauge,已对
  v0.19 源码验证),不暴露排队者身份;选择性外包需要名单
- **KV 意识在 nimbus policy 里**,不在 dispatcher:kick 检查先于 dispatch,且甩负载决策计算期间 admission 冻结(正在完成的请求不可能把待踢者放进本地)

## 云 sink 两档

`--cloud null`(**默认**):fake request——只记"这条 route 去了 cloud" + trace 的
token 数/成本(`--max-tokens` 语义与真实 payload 完全一致:替换 trace 值),
不建模延迟;此类行带 `routed_only=true`,summary 自动排除在 SLO 统计外。

`--cloud real --cloud-url … --cloud-model … --cloud-api-key-env KEY`:真实流式调用,
`--cloud-max-concurrency`(默认 32)防 burst 下自打 429。等实验需要真实云延迟时再用。
(Jialu 有 14k 条真实 OpenRouter 测量在 GPU 机 `/scratch/jialu/initial_result/`
可估分布;实测 qwen3-32b TTFT p50≈10s——reasoning+排队,云并不"快"。)

## 用法(GPU 机;`python` 用带 aiohttp 的 env,从仓库根目录跑)

```bash
python -m router.run --data <trace.jsonl> --scenario burst_300 \
  --policy random --fraction 0.3 --seed 0 \
  --local-url http://127.0.0.1:8010/v1/chat/completions --local-model Qwen3.6-35B-A3B \
  --max-inflight 128        # 对齐 server 的 --max-num-seqs
```

输出:逐请求 JSONL(`ttft_ms = queue_delay_ms + service_ttft_ms`,从 trace 到达时刻算,
与 open-loop 口径可比)+ `.summary.json`(overall/local/cloud 三段 + queue 遥测)。
每段都带 `slo_measured_n`——排除 `routed_only` 后的 SLO 显式分母。
`pessimistic_combined` 另把每条 cloud route 都算作违约,避免 NullCloud 奖励
过度外包。计费:失败请求 $0;local 侧恒 $0。可用 `--decision-log FILE` 记录
每次应用/作废的 Nimbus 决策及 victim 顺序。

本地测试(无需网络/GPU):`python3 -m unittest router.test_common router.test_run router.test_nimbus`

## Baseline 与验证记录

baseline = **Jialu 的 `vllm/run.py`**(团队已验证的 open-loop 压测)。本包内曾有一个
同 schema 的 open-loop runner,完成 parity 锚定后按第一性原理删除。

以下均为 GPU 机实测,Qwen3.6-35B-A3B,`burst_300`,n=756:

| # | 检查 | 结果 |
|---|---|---|
| 1 | **parity 锚点**:open-loop all_local ≡ `vllm/run.py`(同 server 背靠背) | ✅ paired TTFT p50 ratio=1.003;尾部差异=先跑腿的 JIT warmup(用当时的同 schema runner 测得,已删) |
| 2 | **queue 中立性**:本框架 all_local ≈ open-loop | ✅ p50 110 vs 111ms、p99 233 vs 223ms、queue_delay max 2ms |
| 3 | 压力下 pacing:排队在 client 侧、引擎不淹没、无泄漏 | ✅ queue_delay p50=83.8s 而引擎 service TTFT p50=90ms(756/756 成功);机制现为纯并发闸门,同性质由 `max_inflight=1` 串行化单测覆盖 |
| 4 | random 端到端:比例 29.4%/目标 30%、计费重算精确相等、local 侧 $0 | ✅ |
| 5 | null cloud:`routed_only` 不进 SLO 统计、`--max-tokens` 与 payload 语义一致 | ✅ 单元测试 |
| 6 | **nimbus 中立性**:容量充足时 0 踢出,≡ all_local | ✅ p50 113 vs 112ms(旧策略下测得;无压力时新旧策略行为相同——都是 0 踢出;kick 先于 dispatch 后 tick 每次到达都会触发) |
| 7 | 历史 pre-v3 max-displacement 在 compute-bound `extreme_burst_1200` 上:nimbus 外包 25.6%、本地 p50 12.4s,random@25.3% 为 108.7s(all-local 325s),但三者本地 SLO 违约仍很高 | ⚠️ 只能说明选择信号,**不是 v3 验证**:该 workload 是 compute/slot-bound,明确超出 v3 的 KV-bound 范围 |

## 有意不做的事(边界即设计)

| 不做 | 为什么 | 何时回来 |
|---|---|---|
| 云延迟建模 | 路由决策不读任何云侧指标,fake sink 足以证明架构 | 画 cost-vs-SLO 图时 |
| sweep/画图驱动 | 第一个目标格子刚测出来 | 随 frontier 实验 |

(nimbus 和 KV 感知 admission 到 2026-07-09 为止都在这个表里——现已实现:
nimbus 的 kick 检查在 dispatch **之前**跑,`--policy nimbus` 下引擎饱和时
即使 slots 空闲,新到达也会被甩。)

## Nimbus v3 policy(`--policy nimbus --kv-capacity-tokens N`)

队列级甩负载,每次到达/完成时在本地 dispatch **之前**裁决。权威设计见
[`docs/notion_algorithm_design_v3.zh-CN.md`](../docs/notion_algorithm_design_v3.zh-CN.md)。
在线规则为:

```
footprint(req)    = local_prompt_tokens + expected_decode                 [tokens]
residence(req)    = local_prompt_tokens / prefill_tput + expected_decode × TPOT
displacement(req) = footprint(req) × residence(req)                      [token·s]

G = max(0, Σ footprint(waiting) + Σ remaining_decode(inflight) - K_headroom)
release_target = G + 0.05 × K_headroom

若 G > 0:
    按 cloud_cost(req) / displacement(req) 升序踢
    直到 Σ footprint(kicked) >= release_target
```

footprint 与 `/metrics` 实读的 headroom 单位一致,共同决定"放不放得下";
displacement 只进入踢出排序,cloud price 让排序具备成本意识。额外释放 5% headroom
避免紧接着再次触发。Nimbus 本地请求会启用 vLLM continuous usage stats,按精确累计
生成 token 数跟踪进度,不再把 MTP content chunk 当作单个 token。

本 policy 明确是 **KV-bound**。compute/slot-bound 过载需要另一套触发信号,不会
被静默当成 KV 压力。v3 在线不运行背包 solver;精确 cover-form DP 计划作为离线
评估参考,当前尚未实现。

## 实验性预测 TTFT 触发器

`--nimbus-trigger ttft_pred` 用已等待时间、每条在途请求的进度、本请求 prefill
和首个 decode step,在本地 slots 上模拟 FCFS admission;它不读取 KV gauge。
绝对时间参数必须按该部署的工作 batch 显式标定:

```bash
python -m router.run ... --policy nimbus \
  --nimbus-trigger ttft_pred --nimbus-selector cost_cachedisp_old \
  --prefill-tput 2000 --tpot-ms 103 --slo-s 5 \
  --ttft-guard-ms 300 --nimbus-tick-ms 250
```

selector 有 `newest`、`waiting_random`、`max_cachedisp_old`、
`cost_cachedisp_old`、`cost_disp_current`。两个 `*_cachedisp_old` 只复用原 v2
weight 公式做启发式排序,**不是**历史完整 0/1-knapsack 实现。当前 TTFT stop rule
是诊断口径:把留下的本地请求压到预测不违约,再由悲观 combined 指标判断这次外包
是否值得。decode 长度仍取 trace 上限(oracle);估计器与 combined-objective 消融是
后续项。可复现 driver 为 `experiments/run_ttft_selector_matrix.sh`。
