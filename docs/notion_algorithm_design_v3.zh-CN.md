[English](notion_algorithm_design_v3.md) | **简体中文**

# Nimbus 算法设计（v3，KV 受限）

> **状态（2026-07-15）：这是已发布的 baseline，不是当前实验的触发器。**
> 仓库默认仍是本文描述的 `kv_gap`。7 月 14–15 日 dense-32B 实验显式选择了
> 与 selector 正交的 `ttft_pred`：只有在标定模型预测等待请求将违反 TTFT
> SLO 时才外发，再在同一停止条件下比较不同的 victim ordering。六个重复性
> block 将 `ttft_pred + cost_cachedisp_old` 提升为首要实验候选，但没有替换
> 本文默认算法；current displacement 仍是必须保留的对照。详见
> [`v3_experiments_2026-07.md`](v3_experiments_2026-07.md) 第 5b–5g 节。
> 该腿关闭了 prefix cache，`cached_tokens=0`，因此不能验证旧 v2 公式中的
> cached-token 项。

**范围假设（开宗明义）：** 本地绑定资源是 KV 缓存。实验使用 KV 先饱和的负载（例如长 prompt 的生产切片）。算力/slot 受限的过载不在本设计范围内，在 limitations 中讨论。

---

## 第 1 部分 — 单个请求的四个量

全文运行示例：**一个请求带着 2,000-token 的 prompt 到达，我们预期它生成 300-token 的回复。**

### ① footprint — 它将占用多少 GPU 内存（单位：tokens）

```
footprint(r) = prompt tokens + expected output tokens = 2000 + 300 = 2300 tokens
```

为什么是相加：KV 缓存按 token 存状态。2,000 个 prompt token 一进入就占槽位；300 个生成 token 随产出逐个占槽。峰值时该请求持有 2,300 个 token-slot。

**这个量回答：「放得下吗？」**

数字从哪来：prompt 长度可直接计数；输出长度在线未知，必须估计（已知弱点，论文另述；实验先用 trace 真值，并明确标为 oracle）。

**前缀缓存说明：** 本基础版本假设无 prefix-cache hit。在 prefix-aware 模式下，将 prompt tokens 替换为该请求的 **uncached / 边际** KV tokens（它实际会新增的块）；远端（云厂商侧）cache hit 只影响 `cost`，从不影响本地容量。

### ② residence — 这块内存占用多久（单位：秒）

```
residence(r) = prompt processing time + generation time
             = 2000 / 20000  +  300 × 0.0095
             = 0.1 s + 2.85 s ≈ 3 s
```

两个阶段：先处理 prompt（GPU 大约 20,000 tokens/s——离线 profile 标定的常数），再逐个生成 token（约 9.5 ms/token——**在我们机器上测得**；注意每 token 时间取决于并发请求数，这正是 MTP/推测解码线程接入之处）。

**这个量回答：「占多久？」**

### ③ displacement — 多少 × 多久（单位：token·秒）

```
displacement(r) = footprint × residence = 2300 × 3 ≈ 6,900 token·seconds
```

直觉：请求对缓存的「伤害」不只是占多少，还有占多久。占 2,300 tokens 达 3 秒，与占 230 tokens 达 30 秒，伤害相当。**这里保留的是 V2 的 token·seconds / displacement 思想，但代数上并不是 exact V2 公式。** 令 `P` 为完整 prompt、`U` 为剩余 uncached prompt、`D` 为 decode，两条信号分别是：

```
exact old V2 = P × (U / prefill_tput + D × TPOT)
current v3   = (U + D) × (U / prefill_tput + D × TPOT)
```

仓库保留了两者供 selector 对照（`cost_cachedisp_old` 与默认的 `cost_disp_current`）。它们都**不**用来判断是否放得下；只用于**排序：先踢谁**。

（严格来说，真量是积分 `∫ KV_tokens(t) dt`；我们用峰值预留 × residence 作为其保守的在线代理。）

### ④ cost — 外发它要花多少钱（单位：美元）

```
cost(r) = 2000 × ($0.15 / 1M) + 300 × ($1.20 / 1M) ≈ $0.0007
```

数字从哪来：云 API 价目表（例如每百万输入 token $0.15，每百万输出 token $1.20）。

---

## 第 2 部分 — 系统的两个量

**K_headroom — 我们还可再承诺多少 GPU 内存（单位：tokens）。** 不是估计：服务引擎（vLLM）暴露 metrics 端点，我们每 0.25 s 轮询真实 KV 用量。例子：总容量 216,512 tokens，引擎内正在跑的请求占 166,000 → K_headroom = 50,512。（若使用安全上限 `K_safe`——例如总量的 90%——则 `K_headroom = K_safe − current_KV_usage`；默认 `K_headroom = K_avail`，即原始空闲量。）

**飞行中请求的 remaining_decode（单位：tokens）。** 正在跑的请求会持续增长——每生成一个 token 多占一个 KV 槽。其剩余增长为 `expected output tokens − tokens generated so far`。对于本地 vLLM 请求，实现会启用 `stream_options.continuous_usage_stats`，在每次流式更新中读取引擎给出的精确累计 completion-token 数。实现不会再把 content chunk 当成 token：MTP 下一个 content chunk 可以包含多个已接受 token。这笔已承诺的未来增长必须计入压力。若端点未返回 continuous usage，请求会显式失败，而不是静默退回有偏的 chunk 计数。

**队列** — 尚未送入 GPU 的请求。它在我们手里（引擎前的外部队列），因此我们知道每个等待请求的身份与大小。（这是必要的：引擎只暴露队列*计数*，从不暴露内部排队请求的身份——选择性外发需要名单。）

---

## 第 3 部分 — 算法（三步）

下面场景：**30 个请求在等待，footprint 之和 69,000 tokens；飞行中请求合计还有 8,000 tokens 的剩余输出；K_headroom = 50,512。**

### 第 1 步 — 要不要踢？（触发）

```
G = [ Σ_waiting footprint(r)  +  Σ_inflight remaining_decode(r)  −  K_headroom ]₊

  = [ 69,000 + 8,000 − 50,512 ]₊ = 26,488 tokens        ([x]₊ 表示 max(0, x))
```

G > 0 意味着：**在当前 KV headroom 下，这个等待集合无法全部安全留在本地**——已承诺需求（等待集合的 footprint + 运行中请求的剩余增长）超出空闲量 26,488 tokens。若 G ≤ 0，什么都不做。**无压力时算法不执行外发，并保持 all-local 的路由行为**（实验已验证：有空闲容量时零踢出、TTFT 一致）。Nimbus 仍需读取 KV 与生成进度，才能确认该条件。

飞行中项是**主公式的一部分**，不是可选细化——省略它会系统性低估压力，因为 K_headroom 是*当前*读数，尚未包含运行中请求的未来增长。

### 第 2 步 — 踢多少？

踢到被踢请求的 footprint 达到释放目标：

```
release_target = G + h · K_headroom          (h = 0.05)
               = 26,488 + 0.05 × 50,512 ≈ 29,014 tokens
```

不是刚好回到警戒线，而是额外留出当前 headroom 的 5%——像把水位排到略*低于*警戒线，而不是刚好到线，以避免下一个到达立刻再触发。

### 第 3 步 — 踢谁？（排序——核心）

对每个等待请求算性价比分数：

```
score(r) = cost(r) / displacement(r)
         = dollars charged ÷ cache·time released
```

**按 score 升序踢**——每花一美元买回最多的「cache·time」。比较两个请求：

|              | request A    | request B                          |
| ------------ | ------------ | ---------------------------------- |
| footprint    | 2,300 tokens | 2,300 tokens                       |
| residence    | 3 s          | 30 s（很长的生成）                 |
| displacement | 6,900        | 69,000                             |
| cost         | $0.0007      | $0.003                             |
| **score**    | 1.0×10⁻⁷     | **0.4×10⁻⁷ ← 先踢这个**            |

B 外发贵 4×，但对缓存时间的伤害大 10×——踢它更划算。**这就是 displacement 起作用的地方，也是它唯一起作用的地方**——它决定顺序，从不决定是否放得下。

按此顺序踢，直到盖住释放目标（本场景 29,014 tokens）。被踢请求去云 API，我们付它们的 cost。

（启用 MTP/batch 耦合时：每踢一个会缩小运行中的 batch，从而改变每个人的每 token 时间与 residence——因此每踢一次后重新估计并重排。静态近似下，一轮排序即可。）

**在线 vs 离线，精确表述：** 在线算法是覆盖形式的**密度贪心**（「释放 ≥ release_target tokens，优先买回最多 cache·time」）——在线不跑背包求解器。静态覆盖问题（「以最小总成本释放 ≥ G tokens」）有精确 DP 解；该 DP **计划作为离线评估 oracle**，但当前仓库尚未实现。

---

## 第 4 部分 — 为什么三个量绝不能混用

| 量           | 单位     | 精确回答一个问题 |
|--------------|----------|------------------|
| footprint    | tokens   | 放得下吗（与 K_headroom / 默认即 K_avail 比——**同单位，可比较**） |
| displacement | token·s  | 先踢谁（**仅排序，从不与容量比**——不存在「token·秒容量」这种东西。把它当背包权重正是早先的失败：单个 48k-prompt 请求「重」768k token·s，超过整个 450k 预算，而 GPU 物理上还能轻松放下——导致严重过度甩负载，违约 70% vs all-local 的 0%） |
| cost         | dollars  | 踢它要花多少（score 的分子） |

---

## 第 5 部分 — 两个诚实脚注（给论文；不改变算法）

1. 输出长度在线未知。量 ①③④ 都依赖对它的估计。实验必须包含「oracle 长度 vs 估计长度」对比。
2. 被踢请求也可能违约延迟 SLO（测得的云端 TTFT：我们通道上中位数 ≈ 10 s——云不是快速逃生舱）。总账必须也计入它们的违约，不只是本地侧。

---

## 附录 — 相对上一版算法设计改了什么

| 之前 | 现在 | 为什么 |
|---|---|---|
| budget = `total_weight − min_weight`（每轮强制 ≥1 次踢出） | budget/gap = 真实容量：`G = [Σ_waiting footprint + Σ_inflight remaining_decode − K_headroom]₊`，K_headroom 默认即引擎实时读取的 K_avail | 旧 budget 不是容量，只是「至少踢一个」的技巧；背包在其下退化 |
| 背包 **weight = token·秒**（V2 displacement） | **weight = tokens**（footprint）；displacement 移到排序位置 | token·秒没有可比较的容量（上面的 48k-prompt 反例）；tokens 与 K_headroom 同单位 |
| 在线解 0/1 DP，每轮踢一个 | 按 `cost/displacement` 贪心直到盖住 `release_target`；精确 cover-form DP 计划作为离线 oracle | 旧 budget 下 DP 已退化为密度选择；贪心 O(n log n)、在线可负担，计划中的 DP 将作为评估上界 |
| gap 只计等待队列 | gap 主公式计入飞行中请求的剩余输出增长 | 运行中请求持续消耗 KV；忽略会低估压力 |
| （隐含）只计本地侧违约 | 两侧都计违约（被踢请求也算） | 测得的云延迟本身就超过 SLO；外发保护的是队列*其余*请求，不是被踢的那个 |

V2 的开放问题——「若 weight 是 token·秒，budget 是什么？」——通过消解解决：weight 与 budget 现在都活在 tokens 里；保留下来的是 token·秒的**排序思想**。Exact old-V2 公式单独保存在 `classic_cachedisp_token_s`，已发布默认值使用另一条 `(uncached_prompt + decode) × residence` proxy；两者优劣要靠实验比较，不能视为同一公式。
