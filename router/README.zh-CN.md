[English](README.md) | **简体中文**

# router/ — hybrid routing 框架

把 trace 里的请求按 policy 分发给本地 vLLM 和云 sink,记录延迟、成本和分流比例。
**单一入口**:`python -m router.run`。nimbus knapsack 尚未接入——将来作为队列上的
policy 插件进来。


## 文件

| 文件 | 作用 |
|---|---|
| `run.py` | **唯一入口**:外部 FIFO + work-conserving dispatcher + CLI |
| `common.py` | 共享库:`one_request`/`load_trace`/`SCENARIOS`(逐行取自 `vllm/run.py` @ `dff1a81`)、`Endpoint`、`Policy`、`NullCloud`、计费、`summarize` |
| `test_run.py` / `test_common.py` | 31 个单元测试,无需网络/aiohttp/GPU |

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
- 有压力 ⇒ 溢出堆在**我们的**队列里(带完整身份)——未来 nimbus knapsack 的操作对象。
  为什么队列必须自己维护:引擎只暴露排队**计数**(vLLM `/metrics` 三个 gauge,已对
  v0.19 源码验证),不暴露排队者身份;选择性外包需要名单
- **KV 意识故意不在这里**:它属于 nimbus 算法本身(budget=KV),随算法一起进来

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

本地测试(无需网络/GPU):`python3 -m unittest router.test_common router.test_run`

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

## 有意不做的事(边界即设计)

| 不做 | 为什么 | 何时回来 |
|---|---|---|
| nimbus knapsack | 当前目标只是打通分发框架 | 下一步,作为队列上的 policy |
| KV 感知 admission | KV 属于 nimbus 算法(budget=KV) | 随 nimbus |
| 云延迟建模 | 路由决策不读任何云侧指标,fake sink 足以证明架构 | 画 cost-vs-SLO 图时 |
| sweep/画图驱动 | 还没有要扫的实验 | 随 nimbus |

## 扩展点:nimbus 怎么插

在到达时钩子(现有 `Policy.outsource`)之外,加一个**队列级钩子**:

```
policy.on_tick(queue, admission) -> [要踢去云的请求]     # 每次到达/完成时调用
```

knapsack 的 item 就是队列元素(身份、token 数、已等待时长齐全);被踢请求走现有
cloud sink,`queue_delay` 记到被踢时刻。开工前需定:budget 单位(tokens vs
token·s)、触发信号、KV 信号来源(客户端账本 vs `/metrics`)——前两个是算法口径,
open questions 记录在 Notion 算法文档里。
