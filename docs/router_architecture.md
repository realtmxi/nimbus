# Router 架构说明

> 对象:`router/`(2026-07,[PR #3](https://github.com/realtmxi/nimbus/pull/3))。
> 本文讲**结构和为什么**;怎么跑、验证状态见 [router/README.md](../router/README.md)。

## 1. 这是什么

Nimbus 的 local/cloud 请求路由框架,从零重写。它取代的不是某个旧文件,而是一种工作方式:
旧 harness(`experiments/run_engine.py` 等)功能多但没人逐行验证过;`router/` 反过来——
**只造证明当前命题所需的最小物件,每一层都有上一层做参照物来验收**。

三条设计原则:

1. **可信源优先**:凡是能抄 Jialu 验证过的代码(`vllm/run.py`)的,逐行抄,不重新发明
2. **第一性原理**:不预建层。KV 感知、云延迟建模、nimbus 算法都被有意排除(§7)
3. **每层可验收**:新加一层,必须能和上一层对比出"没有引入行为差异"

当前能力:把 trace 里的请求按 policy(all_local / all_cloud / random)分发给本地 vLLM
和云 sink,记录延迟/成本/分流比例。仅此而已——这就是目标。

## 2. 目录结构

```
router/
├── run.py               590 行  Step 1:open-loop 路由器 + 全部共享构件
├── run_queued.py        279 行  Step 2:外部队列 + work-conserving dispatcher
├── test_run.py          295 行  ┐
├── test_run_queued.py   178 行  ┘ 29 个单元测试(stub session,无需网络/aiohttp/GPU)
├── README.md                    用法 + 验证阶梯(实测数字)
└── __init__.py                  空,使 router 成为包(python -m router.run_queued)
```

依赖:标准库 + aiohttp(仅真实发请求时;`--cloud null` 的纯云路径连 aiohttp 都不需要)。
**不 import `nimbus/`、`experiments/` 的任何东西**——和旧 harness 零耦合。

## 3. 两种运行架构

### Step 1:open-loop(`run.py`)——和 Jialu 压测同构

```
到达时刻一到 ──Policy──┬─local──> vLLM(排队发生在引擎内部)
                       └─cloud──> cloud sink
```

无队列、无闸门,请求到达即派发。存在的意义:**基准参照物**。它的 all_local 路径和
`vllm/run.py` 行为等价(同 open-loop、同 payload、同 TTFT 口径),所以它的数字可以直接
和 Jialu 的历史结果/复跑结果对齐——这是整个信任链的锚点。

### Step 2:queued(`run_queued.py`)——为可拓展性而生

```
到达 ──Policy(到达时决策)──cloud──> cloud sink
        │local
        v
   [外部 FIFO] ──dispatcher: inflight < max_inflight ──> vLLM(内部队列≈空)
```

dispatcher 是 **work-conserving** 的:有空位立刻放行,绝不无谓扣请求。
`--max-inflight` 对齐 server 的 `--max-num-seqs`,于是:

- 无压力 ⇒ 外部队列恒空 ⇒ 行为**退化为 open-loop**(可验收:queue 中立性)
- 有压力 ⇒ 溢出堆在**我们的**队列里(带完整身份),vLLM 内部几乎不排队

为什么必须自己维护队列:引擎只暴露排队**计数**(vLLM `/metrics` 的
`num_requests_waiting` 等三个 gauge,已对 v0.19 源码验证),不暴露排队者的**身份**
(真正的队列是 `scheduler.py` 里的 `self.waiting`,无任何 HTTP 端点)。将来的
shedding policy 要"挑着踢",挑人得有名单——名单只能在自己手里。

## 4. 请求的一生(queued 版)

1. **加载**:`load_trace` 按 `--scenario` 的 `arrived_at` 窗口切 BurstGPT JSONL,
   得到 `{request_id, relative_arrival_s, prompt, max_tokens, prompt_tokens, session_id}`
2. **到达**:replay 循环 sleep 到 `relative_arrival_s`,调 `policy.outsource(req)`
3. **分叉**:
   - cloud → `route_cloud`:null sink 立即记一条 `routed_only` 结果(或 real:真实流式调用)
   - local → 进 FIFO;`maybe_dispatch()` 在有空位时弹出队头,spawn `serve_local`
4. **本地服务**:`one_request` 发流式 chat-completion,测 `service_ttft`;完成后释放
   名额并再触发 `maybe_dispatch()`(完成事件驱动下一次放行)
5. **记账**:`ttft_ms = queue_delay_ms + service_ttft_ms`(从 trace 到达时刻算起,
   与 open-loop 口径可比);逐请求增量写 JSONL
6. **收尾**:`summarize` 出 overall/local/cloud 三段 + queue 遥测段

## 5. 五个核心抽象

| 抽象 | 位置 | 一句话 | 关键接口 |
|---|---|---|---|
| `Endpoint` | run.py | 一个 OpenAI 兼容端点(url/model/key/单价) | frozen dataclass |
| `Policy` | run.py | 到达时路由决策;三个 policy = 同一规则的 p=0/1/f | `outsource(req) -> bool` |
| `one_request` | run.py | 发一条流式请求并测 TTFT/TPOT/错误(**逐行取自 vllm/run.py**) | `await one_request(session, endpoint, req, due)` |
| `NullCloud` | run.py | fake 云 sink:只记 routed + token 计数,不建模延迟 | `serve(req, due) -> result dict` |
| `LocalAdmission` | run_queued.py | 本地并发闸门 + 峰值遥测 | `fits / reserve / release` |

## 6. 数据口径(读结果前必看)

- **结果行**(JSONL,每请求一条):`endpoint`(local/cloud)、`success`、`ttft_ms`、
  `tpot_ms`、`e2e_ms`、`prompt_tokens`/`completion_tokens`(来自 usage 或 trace)、
  `cost_usd`、错误三元组;queued 版另有 `queue_delay_ms`/`service_ttft_ms`
- **`routed_only=true`**(null sink 产物):此行**无延迟主张**,summary 自动把它排除在
  SLO 统计外,只计入数量与成本——避免"云侧 100% 违约"这种假数字
- **计费**:`cost = prompt×in_price + completion×out_price`(per Mtok);**失败请求 $0**;
  local 侧恒 $0(GPU 成本不在此计)
- **确定性**:同 `--seed` ⇒ random 的逐请求去向完全相同(跨 run 可复现、可配对比较)

## 7. 有意不做的事(边界即设计)

| 不做 | 为什么 | 何时回来 |
|---|---|---|
| nimbus knapsack | 当前目标只是打通框架 | Step 3,作为队列上的 policy |
| KV 感知 admission | KV 是 nimbus 算法的一部分(budget=KV),不是框架的 | Step 3(曾实现 KV 预留账本并在 gpu1 验证过机制,按第一性原理删除) |
| 云延迟建模(sim) | 路由决策不读任何云侧指标,fake sink 足以证明架构 | 画 cost-vs-SLO 图时(Jialu 的 14k 条真实 OpenRouter 测量在 gpu1 `/scratch/jialu/initial_result/` 可做校准源) |
| sweep/画图驱动 | 还没有要扫的实验 | 随 Step 3 |

## 8. nimbus 将来怎么插(Step 3 预告)

到达时钩子(现有 `Policy.outsource`)之外,加一个**队列级钩子**:

```
policy.on_tick(queue, admission) -> [要踢去云的请求]     # 每次到达/完成时调用
```

knapsack 的 item 就是队列元素(身份、token 数、已等待时长齐全);被踢请求走现有
cloud sink,`queue_delay` 记到被踢时刻(~3 行改动)。开工前需定四件事:
budget 单位(tokens vs token·s)、触发信号、KV 信号来源(客户端账本 vs `/metrics`)、
以及以上两个算法口径由 Murphy 拍板——详见 Notion《Algorithm Design》的 open questions。

## 9. 验证方法论(为什么可以信这套代码)

每层的验收都是**和已验证参照物的可复跑对比**,不是一次性演示:

```
Jialu vllm/run.py(团队已验证)
   └── Step 1 all_local 背靠背:paired TTFT p50 ratio = 1.003     ← 锚点
         └── Step 2 queued all_local:110 vs 111ms(queue 零开销)  ← queue 中立性
               └── Step 3 nimbus vs random/all_local/all_cloud     ← 未来:算法收益
```

外加 29 个单元测试锁行为规约(分流确定性、失败不计费、名额不泄漏、FIFO、
routed_only 不进 SLO),以及 CLI 冒烟(单测覆盖不到 replay 主循环——历史上真漏过一个
改名残留 NameError,教训:**每次改动后单测 + CLI 冒烟都要跑**)。
具体实测数字见 [router/README.md](../router/README.md) 的验证阶梯表。
