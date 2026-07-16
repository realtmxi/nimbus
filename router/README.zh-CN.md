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
| `test_run.py` / `test_common.py` / `test_nimbus.py` | 75 个单元测试,无需网络/aiohttp/GPU |

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
- 同一 trace 时间戳的请求先组成一个完整 arrival cohort,policy 看完整批后才 dispatch;
  selector 不会再被拿去比较一串人为制造的单候选队列。

## 云 sink 两档

`--cloud null`(**默认**):fake request——只记"这条 route 去了 cloud" + trace 的
token 数/成本(`--max-tokens` 语义与真实 payload 完全一致:替换 trace 值),
不建模延迟;此类行带 `routed_only=true`,summary 自动排除在 SLO 统计外。

`--cloud real --cloud-url … --cloud-model … --cloud-api-key-env KEY`:真实流式调用,
`--cloud-max-concurrency`(默认 32)防 burst 下自打 429。等实验需要真实云延迟时再用。
(队友有 14k 条真实 OpenRouter 测量在 `$JSCRATCH/initial_result/`
可估分布;实测 qwen3-32b TTFT p50≈10s——reasoning+排队,云并不"快"。)

若只做 OpenRouter TTFT 探针,再加
`--cloud-provider deepinfra --cloud-no-fallbacks
--cloud-stop-after-first-token`。最后一个参数在首个非空 reasoning 或 content delta
后断流,与当前本地 vLLM 的首生成 token 边界一致。该模式只测 TTFT;没有完整
E2E/TPOT,拿不到末尾 usage 时成本保持 pending。普通 real-cloud 路径仍完整读到
`[DONE]`。

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
过度外包;它只是 NullCloud assumed upper bound,不是真实 cloud headline。真实云看
`overall.slo_*`,并检查 `overall.slo_measured_n == overall.n`、
`cloud.routed_only == 0`。计费:失败请求 $0;local 侧恒 $0。可用
`--decision-log FILE` 记录
每次应用/作废的 Nimbus 决策及 victim 顺序。逐请求结果还会并排记录
`scheduler_prompt_tokens` 与 endpoint 实报 `prompt_tokens`;summary 的
`token_alignment` 会把 trace/payload 单位错位显式暴露出来。

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

只有当 deployment gauge 探针已验证 token-KV 语义时，footprint 与
`/metrics` headroom 才共享同一单位，并共同决定"放不放得下"。metric 名称
本身不是证明：hybrid-model gauge 可能表示每序列状态。displacement 只进入
踢出排序，cloud price 让排序具备成本意识。额外释放 5% headroom
避免紧接着再次触发。Nimbus 本地请求会启用 vLLM continuous usage stats,按精确累计
生成 token 数跟踪进度,不再把 MTP content chunk 当作单个 token。

本 policy 明确是 **KV-bound**。compute/slot-bound 过载需要另一套触发信号,不会
被静默当成 KV 压力。v3 在线不运行背包 solver;精确 cover-form DP 计划作为离线
评估参考,当前尚未实现。

## 实验性预测 TTFT 触发器

`--nimbus-trigger ttft_pred` 同时模拟两个资源:本地 sequence slots 和一条共享
prefill compute lane。128 个空 slot 不代表 128 个并发 prompt 能同时拿到首 token。
模型计入已等待时间、精确 decode 进度、尚未首 token 的在途 prefill 全量工作、
等待队列的累计 prefill,以及单独拟合的固定首 token 开销;它不读取 KV gauge。
绝对时间参数必须在同一部署上显式标定:

```bash
python -m router.run ... --policy nimbus \
  --nimbus-trigger ttft_pred --nimbus-selector cost_cachedisp_old \
  --prefill-tput 3270 --tpot-ms 152 --first-token-overhead-ms 461 --slo-s 5 \
  --ttft-guard-ms 1685 --nimbus-tick-ms 250
```

上面的数字只展示参数彼此独立,实际必须读取同一 server 的 profile artifact。
`first-token-overhead-ms` 只进入 TTFT;`tpot-ms` 用于 decode slot residence,同时仍
进入原 v2 weight,不能把二者混成一个参数。

selector 有 `newest`、`waiting_random`、`max_cachedisp_old`、
`cost_cachedisp_old`、`cost_disp_current`。两个 `*_cachedisp_old` 只复用原 v2
weight 公式做启发式排序,**不是**历史完整 0/1-knapsack 实现。当前 TTFT stop rule
是诊断口径:只把留下的**等待队列 survivors**压到预测不违约;已经 in-flight 的
请求不在这个 post-kick 声明里,因此 decision log/summary 明确标作
`prediction_scope=waiting_only`。NullCloud 的悲观 combined 只是假设上界;真实
cloud 用实测 `overall.slo_violation_pct`。
decode 长度仍取 trace 上限(oracle);估计器与 combined-objective 消融是后续项。
可复现 driver 为 `experiments/run_ttft_selector_matrix.sh`;显式的
`anchor:all_local:0` arm 复用同一套绑定 manifest/marker 合约,但不会假装该
anchor 存在 Nimbus trigger 或 decision log。

**当前证据边界（2026-07-15）。** 在已完成的预注册 11,605-request dense-32B
no-cache full cell 中，`ttft_pred + cost_cachedisp_old` 与 `ttft_pred +
cost_disp_current` 留在本地的请求均为 0 违约；old V2 位于冻结的路由等价带内，
且成本低 4.98%。`ttft_pred + newest` 留下 39 个违约，这些违约都发生在
client-visible load 超出 profile 支持域时；`kv_gap + cost_disp_current` 则留下 5,249 个违约。
因此 `ttft_pred` 仍属实验性，还需要显式 support-envelope/resource fallback；
`kv_gap` 保留为代码默认是为了兼容，并不代表它已被证明能保证 TTFT 安全。
详见 [`../docs/v3_experiments_2026-07.md`](../docs/v3_experiments_2026-07.md)
第 5g 节。

### token 对齐的 no-cache 实验前置条件

主 ShareGPT/BurstGPT 文件的 `num_prefill_tokens` 是累计会话长度,但
`prompt_text` 只有当前 user turn。直接重放可能出现“调度器按 500 tokens 计算、
HTTP 实际不到 20 tokens”,不能拿来证明 displacement selector。应先用部署的
tokenizer 物化一份自洽 no-cache trace,以关闭 prefix cache 的 vLLM 运行,并在同一
部署上标定参数:

```bash
python tools/materialize_token_aligned_trace.py \
  --input <sharegpt-burstgpt.jsonl> --output <aligned.jsonl> \
  --scenario extreme_burst_1200 --tokenizer <model-path> \
  --salt dense32b-nocache-v1 \
  --max-prompt-tokens 32768 --max-decode-tokens 1024 \
  --max-context-tokens 40960 --overflow-policy error

python tools/profile_ttft_batch.py \
  --base-url http://127.0.0.1:8010 --model qwen3-32b \
  --tokenizer <model-path> --server-log <active-vllm.log> \
  --server-pid <recorded-pid> \
  --kv-capacity-tokens <startup-log-token-capacity> \
  --slo-s 5 \
  --output <profile.json>

DATA=<aligned.jsonl> BASE_URL=http://127.0.0.1:8010 MODEL=qwen3-32b \
  PROFILE=<profile.json> SERVER_LOG=<active-vllm.log> SERVER_PID=<recorded-pid> \
  OUT_DIR=<new-results-dir> experiments/run_ttft_selector_matrix.sh
```

这条腿按构造满足 `cached_tokens = 0`。cache-aware workload 必须另跑;不能把
这条腿的结果写成“已经验证旧公式里的 cached-token 项”。prompt/decode/context
cap 使它成为变换后的 synthetic workload,每次汇报都必须带 manifest 中受影响
行数。GPU 箱若 home quota 紧张,两个命令前都要把 `TMPDIR`、`HF_HOME`、
`XDG_CACHE_HOME` 指向 scratch。
