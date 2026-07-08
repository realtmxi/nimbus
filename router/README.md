# router/ — 从零重写的 hybrid routing 框架

把 trace 里的请求按 policy 分发给本地 vLLM 和云 sink,记录延迟/成本/分流比例。
**单一入口** `python -m router.run`;baseline 对照 = Jialu 已验证的 `vllm/run.py`。
nimbus knapsack 尚未接入(将作为队列上的 policy 插件)。

## 文件

| 文件 | 作用 |
|---|---|
| `run.py` | **唯一入口**:外部 FIFO + work-conserving dispatcher + CLI |
| `common.py` | 共享库:`one_request`/`load_trace`/SCENARIOS(逐行取自 **`vllm/run.py` @ `dff1a81`**,即 Jialu 分支 mtp 版——她实际在跑、gpu1 上 md5 核对过的版本;main 上的旧版 payload 略有差异:有 temperature/top_p、无 stream_options)、Endpoint、Policy、NullCloud、计费、summarize |
| `test_run.py` / `test_common.py` | 31 个单元测试,无需网络/aiohttp |

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
- 有压力 ⇒ 溢出堆在**我们的**队列里(带完整身份)—— 未来 nimbus knapsack 的操作对象。
  为什么队列必须自己维护:引擎只暴露排队**计数**(vLLM `/metrics` 三个 gauge),
  不暴露排队者身份;选择性外包需要名单
- **KV 意识故意不在这里**:它属于 nimbus 算法(budget=KV),Step 3 随算法进来

## 云 sink 两档

`--cloud null`(**默认**):fake request——只记"这条 route 去了 cloud" + trace 的
token 数/成本(`--max-tokens` 语义与真实 payload 完全一致:替换 trace 值),
**不建模延迟、不参与 SLO 统计**(`routed_only` 标记,summary 自动排除)。

`--cloud real --cloud-url ... --cloud-model ... --cloud-api-key-env KEY`:真实流式调用,`--cloud-max-concurrency`(默认 32)防 burst 下自打 429。
等实验需要云侧延迟数字时再用(Jialu 有 14k 条真实 OpenRouter 测量在 gpu1
`/scratch/jialu/initial_result/` 可估分布;qwen3-32b 实测 TTFT p50≈10s——reasoning+排队,云并不"快")。

## 用法(gpu1;`python` 用带 aiohttp 的 env,从仓库根目录跑)

```bash
python -m router.run --data <trace.jsonl> --scenario burst_300 \
  --policy random --fraction 0.3 --seed 0 \
  --local-url http://127.0.0.1:8010/v1/chat/completions --local-model Qwen3.6-35B-A3B \
  --max-inflight 128        # 对齐 server 的 --max-num-seqs
```

输出:逐请求 JSONL(`ttft_ms = queue_delay_ms + service_ttft_ms`,从 trace 到达时刻算,
与 open-loop 口径可比)+ `.summary.json`(overall/local/cloud + queue 遥测段)。

本地测试(无需网络/GPU):`python3 -m unittest router.test_common router.test_run`

## Baseline 对照

**用 Jialu 的 `vllm/run.py`**(团队已验证的 open-loop 压测)。曾有一个同 schema 的
open-loop runner 在本包内,用它完成了 parity 锚定后按第一性原理删除(2026-07-08)。

## 验证记录(gpu1,Qwen3.6-35B-A3B,burst_300,n=756)

| # | 检查 | 结果 |
|---|------|------|
| 1 | **parity 锚点**:open-loop all_local ≡ `vllm/run.py`(同 server 背靠背) | ✅ paired TTFT p50 ratio=1.003;尾部差异=先跑腿的 JIT warmup(用当时的同 schema open-loop runner 测得,该 runner 已删) |
| 2 | **queue 中立性**:本框架 all_local ≈ open-loop | ✅ p50 110 vs 111ms、p99 233 vs 223ms、queue_delay max 2ms |
| 3 | 压力下 pacing:排队在 client 侧、引擎不淹没、无泄漏 | ✅ 当时用 KV 预算制造压力:queue_delay p50=83.8s 而 service TTFT p50=90ms(引擎全程健康);机制现为纯并发闸门,同性质由 `max_inflight=1` 串行化单测覆盖 |
| 4 | random 端到端:比例 29.4%/目标 30%、计费重算精确相等、local 侧 $0 | ✅ |
| 5 | null cloud:routed_only 不进 SLO 统计、`--max-tokens` 与 payload 语义一致 | ✅ 单元测试 |

## Step 3(未实现):nimbus knapsack 作为队列上的 policy

on-tick 钩子对等待队列跑 knapsack,踢出的请求走 cloud sink。开工前需定:
① budget 单位(tokens vs token·s)② 触发信号 ③ KV 信号来源(客户端账本 vs `/metrics`)
④ 被踢请求的 queue_delay 记账小改。
