[English](notion_algorithm_design_v3.md) | **简体中文**

# Nimbus 算法设计（v3，KV 受限）

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

直觉：请求对缓存的「伤害」不只是占多少，还有占多久。占 2,300 tokens 达 3 秒，与占 230 tokens 达 30 秒，伤害相当。**这就是 V2 公式——论文的核心想法。** 它**不**用来判断是否放得下（那是 footprint 的事）；它只用于**排序：先踢谁**。

### ④ cost — 外发它要花多少钱（单位：美元）

```
cost(r) = 2000 × ($0.15 / 1M) + 300 × ($1.20 / 1M) ≈ $0.0007
```

数字从哪来：云 API 价目表（例如每百万输入 token $0.15，每百万输出 token $1.20）。

---

## 第 2 部分 — 系统的两个量

**K_avail — 此刻有多少 GPU 内存空闲（单位：tokens）。** 不是估计：服务引擎（vLLM）暴露 metrics 端点，我们每 0.25 s 轮询真实读数。例子：总容量 216,512 tokens，引擎内正在跑的请求占 166,000 → K_avail = 50,512。

**队列** — 尚未送入 GPU 的请求。它在我们手里（引擎前的外部队列），因此我们知道每个等待请求的身份与大小。（这是必要的：引擎只暴露队列*计数*，从不暴露内部排队请求的身份——选择性外发需要名单。）

---

## 第 3 部分 — 算法（三步）

下面场景：**30 个请求在等待；它们的 footprint 之和为 69,000 tokens；K_avail = 50,512。**

### 第 1 步 — 要不要踢？（触发）

```
gap G = total footprint of the queue − K_avail = 69,000 − 50,512 = 18,488 tokens
```

G > 0 意味着：即便 GPU 跑完当前一切，这个队列也放不下——超出容量 18,488 tokens。若 G ≤ 0，什么都不做。**无压力时算法零开销，行为与全本地完全一致**（实验已验证：有空闲容量时零踢出，TTFT 与 all-local 基线相同）。

细化：正在跑的请求还在增长（每生成一个 token 多占一个槽），所以 G 还应计入飞行中请求的*剩余*预期输出——否则 gap 会被系统性低估。

### 第 2 步 — 踢多少？

踢到被踢请求的 footprint 之和 ≥ G，再加一点余量（例如 5%），这样紧接着的下一个到达不会立刻再触发——像把水位排到略*低于*警戒线，而不是刚好到线，以避免振荡。

### 第 3 步 — 踢谁？（排序——核心）

对每个等待请求算性价比分数：

```
score(r) = cost(r) / displacement(r)
         = dollars charged ÷ cache·time released
```

**按 score 升序踢**——每花一美元买回最多的「cache·time」。比较两个请求：

|                | request A | request B |
|----------------|-----------|-----------|
| footprint      | 2,300 tokens | 2,300 tokens |
| residence      | 3 s       | 30 s（很长的生成） |
| displacement   | 6,900     | 69,000 |
| cost           | $0.0007   | $0.003 |
| **score**      | 1.0×10⁻⁷  | **0.4×10⁻⁷ ← 先踢这个** |

B 外发贵 4×，但对缓存时间的伤害大 10×——踢它更划算。**这就是 displacement 起作用的地方，也是它唯一起作用的地方**——它决定顺序，从不决定是否放得下。

按此顺序踢，直到盖住 18,488-token 的 gap（加余量）。被踢请求去云 API，我们付它们的 cost。

（启用 MTP/batch 耦合时：每踢一个会缩小运行中的 batch，从而改变每个人的每 token 时间与 residence——因此每踢一次后重新估计并重排。静态近似下，一轮排序即可。）

**离线参考：** 同一选择问题——「以最小总成本释放 ≥ G tokens」——可用动态规划精确求解。我们**不**在线跑它；它在评估中作为离线上界，衡量贪心排序离最优有多远。

---

## 第 4 部分 — 为什么三个量绝不能混用

| 量     | 单位     | 精确回答一个问题 |
|--------------|----------|------------------------------|
| footprint    | tokens   | 放得下吗（与 K_avail 比——**同单位，可比较**） |
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
| budget = `total_weight − min_weight`（每轮强制 ≥1 次踢出） | budget/gap = 真实容量：`G = Σ footprint − K_avail`，K_avail 从引擎实时读取 | 旧 budget 不是容量，只是「至少踢一个」的技巧；背包在其下退化 |
| 背包 **weight = token·秒**（V2 displacement） | **weight = tokens**（footprint）；displacement 移到排序位置 | token·秒没有可比较的容量（上面的 48k-prompt 反例）；tokens 与 K_avail 同单位 |
| 在线解 0/1 DP，每轮踢一个 | 按 `cost/displacement` 贪心直到盖住 gap；DP 保留为离线 oracle | 旧 budget 下 DP 已退化为密度选择；贪心 O(n log n)、在线可负担，DP 现作评估上界 |
| gap 只计等待队列 | gap 也计入飞行中请求的剩余输出增长 | 运行中请求持续消耗 KV；忽略会低估压力 |
| （隐含）只计本地侧违约 | 两侧都计违约（被踢请求也算） | 测得的云延迟本身就超过 SLO；外发保护的是队列*其余*请求，不是被踢的那个 |

V2 的开放问题——「若 weight 是 token·秒，budget 是什么？」——通过消解解决：weight 与 budget 现在都活在 tokens 里；token·秒公式（V2）作为排序信号完整保留，而这正是论文的核心主张。
