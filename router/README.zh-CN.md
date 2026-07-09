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
| `nimbus.py` | nimbus 甩负载 policy:等待集合超出真实 KV 余量时,踢 displacement 最大的请求(`--policy nimbus --kv-capacity-tokens N`);背包 solver 为成本感知消融保留 |
| `test_run.py` / `test_common.py` / `test_nimbus.py` | 52 个单元测试,无需网络/aiohttp/GPU |

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
计费:失败请求 $0;local 侧恒 $0。

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
| 7 | **nimbus 压力测试**(slots=4、KV 预算 3k):自选外包 29.0%,本地 SLO 违约 **0%**,同约束 all_local **91.1%**(p50 435ms vs 32.8s) | ⚠️ **旧($-背包)策略下测得**——仅作机制演示;修正后的 CacheDisp 策略待 GPU 复位后在 extreme_burst 格上复测 |

## 有意不做的事(边界即设计)

| 不做 | 为什么 | 何时回来 |
|---|---|---|
| 云延迟建模 | 路由决策不读任何云侧指标,fake sink 足以证明架构 | 画 cost-vs-SLO 图时 |
| sweep/画图驱动 | 第一个目标格子刚测出来 | 随 frontier 实验 |

(nimbus 和 KV 感知 admission 到 2026-07-09 为止都在这个表里——现已实现:
nimbus 的 kick 检查在 dispatch **之前**跑,`--policy nimbus` 下引擎饱和时
即使 slots 空闲,新到达也会被甩。)

## nimbus policy(`--policy nimbus --kv-capacity-tokens N`)

队列级甩负载,每次到达/完成时在本地 dispatch **之前**裁决。语义——CacheDisp
核心(2026-07-09 review 抓到实现漂移后由 Murphy 重申):

```
displacement(req) ≈ prompt_tokens × (prefill时间 + decode_tokens × TPOT)   [token·s]
while Σ footprint(waiting) > K_avail(/metrics):
    踢 displacement 最大的请求
```

踢的是"占最多缓存、占最久"的请求——API 成本**不进**踢出规则(它属于 frontier
对比)。成本感知的背包变体(min 云成本 s.t. 释放足够 displacement / max 释放
displacement s.t. 云预算)是计划中的消融,`solve_knapsack`/`value_usd` 为此保留。
剩余开放旋钮:触发条件(装不下 vs 队头等待逼近 SLO,即决策 2)。
