# router/ — 从零重写的 hybrid routing 框架

逐步重建 local+cloud 路由,每一步都有上一步作参照物来验证正确性。
**当前状态:基础分发层完整**(open-loop + 队列两种架构、fake/真实两种云 sink、
三个 baseline policy)。nimbus knapsack 尚未接入(将作为队列上的 policy 插件)。

## 文件

| 文件 | 作用 |
|---|---|
| `run.py` | Step 1:open-loop router(无队列,到达即派发;`one_request`/`load_trace` 逐行取自已验证的 `vllm/run.py`) |
| `run_queued.py` | Step 2:外部 FIFO + work-conserving dispatcher(并发闸门;nimbus 未来插在这) |
| `test_run.py` / `test_run_queued.py` | 29 个单元测试,无需网络/aiohttp |

## 云 sink 两档

`--cloud null`(**默认**):fake request——只记"这条 route 去了 cloud" + trace 的
token 数/成本,**不建模任何延迟、不参与 SLO 统计**(`routed_only` 标记,summary 自动
排除)。routing 决策不依赖任何云侧指标,证明架构跑通用这档就够。

`--cloud real --cloud-url ... --cloud-api-key-env KEY`:真实流式调用,记实测
TTFT/成本。等实验真的需要云侧延迟数字时再用(注:Jialu 有一批真实 OpenRouter 测量在
gpu1 `/scratch/jialu/initial_result/`,届时可先用它估计分布,qwen3-32b 走 OpenRouter
的实测 TTFT p50≈10s——reasoning + 排队,云并不"快")。

## 用法(gpu1;`python` 用带 aiohttp 的 env,从仓库根目录跑)

```bash
# open-loop(Step 1;all_local 已与 Jialu 脚本背靠背验证:paired p50 ratio=1.003)
python -m router.run --data <trace.jsonl> --scenario burst_300 \
  --policy all_local --local-url http://127.0.0.1:8010/v1/chat/completions \
  --local-model Qwen3.6-35B-A3B

# 队列版(Step 2)。--max-inflight 对齐 server 的 --max-num-seqs
python -m router.run_queued --data <trace.jsonl> --scenario burst_300 \
  --policy random --fraction 0.3 --seed 0 \
  --local-url http://127.0.0.1:8010/v1/chat/completions --local-model Qwen3.6-35B-A3B \
  --max-inflight 128
# cloud 默认 null(fake);summary 里多一段 queue:{queue_delay_p50/p99/max, peak_inflight}
```

输出:逐请求 JSONL(队列版多 `queue_delay_ms`/`service_ttft_ms`,TTFT=两者之和,
与 open-loop 口径可比)+ `.summary.json`(overall/local/cloud + queue 段)。

## 队列版架构(Step 2)

```
到达 ──policy(到达时决策)──cloud──> fake sink(默认)/ 真实云
        │local
        v
   [外部 FIFO] ──work-conserving dispatcher──> vLLM(内部队列≈空)
```

- dispatcher 只要有空位就放行(绝不无谓扣请求):`inflight < max_inflight`,
  与 server 的 `--max-num-seqs` 对齐 ⇒ vLLM 内部队列≈空
- 无压力 ⇒ 队列恒空 ⇒ 行为退化为 open-loop(= 验收判据"queue 中立性")
- 有压力 ⇒ 溢出堆在**我们的**队列里 —— 未来 nimbus knapsack 的操作对象
- **KV 意识故意不在这里**:它属于 nimbus 算法本身(budget=KV),Step 3 随算法一起进来
  (曾以"客户端 KV 预留账本"实现并在 gpu1 验证过机制,2026-07-08 按第一性原理删除)

## 验证阶梯

| # | 检查 | 状态 |
|---|------|------|
| 1 | open-loop all_local ≡ Jialu `vllm/run.py`(同 server 背靠背,burst_300 n=756) | ✅ paired TTFT p50 ratio=1.003;尾部差异=先跑腿的 JIT warmup(慢请求全在前 5s) |
| 2 | random 机制:actual_fraction≈目标、双侧记账、计费重算精确相等 | ✅($0.067427 精确匹配) |
| 3 | null cloud:routed_only 不参与 SLO 统计、token 计数/成本按 trace 记 | ✅ 单元测试 |
| 4 | **queue 中立性**:queued all_local ≈ open-loop all_local(低压) | ✅ gpu1 实测(burst_300 n=756):p50 110 vs 111ms、p99 233 vs 223ms、queue_delay max 2ms |
| 5 | 压力下 pacing:排队在 client 侧、引擎不淹没、无泄漏 | ✅ gpu1 实测(当时用 KV 预算 4000 tok 制造压力):756/756 成功、queue_delay p50=83.8s 而 **service TTFT p50=90ms/max 421ms**(排队全在我们队列里,引擎全程健康)。该机制现已简化为纯并发闸门,同性质由 `max_inflight=1` 串行化单测覆盖 |
| 6 | random 端到端(gpu1,云侧当时为模拟 sink,现已简化为 null):比例 29.4%/目标 30%、计费重算精确相等、local 侧 $0 | ✅ gpu1 实测 |

## Step 3(未实现):nimbus knapsack 作为队列上的 policy

on-tick 钩子对等待队列跑 knapsack,踢出的请求走 cloud sink。进 Step 3 需要:
① budget 单位(tokens vs token·s,Murphy 拍板)② 触发信号(同上)
③ KV 信号来源(客户端预留账本 vs /metrics 真值)④ 被踢请求的 queue_delay 记账小改。
