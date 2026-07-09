[English](exp_knapsack_vs_sorting_design.md) | **简体中文**

# 实验：背包 vs 贪心排序

## 动机

Nimbus 用 0/1 背包求解器，从等待队列中决定外发哪些请求。背包在权重预算（缓存置换）约束下，最大化本地保留请求的总价值（节省的 API 成本）。

一个自然问题是：**背包求解器是否必要，还是按 value/weight 比值排序后贪心选取就能得到同样结果？**

贪心排序是 O(n log n)，实现极简。背包 DP 是 O(n * W)，更复杂。若二者决策完全相同，就应采用更简单的那个。

## 背景：Nimbus 如何使用背包

检测到 TTFT 违约时，Nimbus 运行迭代循环：

```
while TTFT_violation_exists():
    items = [knapsack_item(r) for r in waiting_queue]
    budget = total_weight - 1        # force at least 1 outsource
    keep_set, outsource_set = knapsack.solve(items, budget)
    outsource(outsource_set[0])      # kick 1 request
    remove from queue
    recheck violations
```

每个请求变成一个背包物品：
- **weight** = `prefill_tokens * decode_tokens`（缓存置换）
- **value** = 外发时的 API 成本（本地保留所能节省的）

预算设为 `total_weight - 1`，保证每轮至少外发一个请求。

## 实验设计

### 两个求解器（已在 `routing/outsourcing/knapsack.py` 实现）

| 求解器 | 策略 | 工作方式 |
|--------|----------|-------------|
| **dp_scaled** | 0/1 背包 DP | 通过动态规划在预算内找全局最优子集。缩放权重以保持 DP 表可解。 |
| **fractional** | 贪心排序 | 按 value/weight 比值降序排序，从顶部贪心填满预算。即经典分数背包 / 贪心方法。 |

### 输入：来自真实 Trace 的队列快照

从 ShareGPT+BurstGPT trace（200K 请求）中提取队列快照，模拟「某一时刻等待队列长什么样」：

- 按到达时间遍历 trace
- 每个快照取连续 `queue_size` 个请求的窗口
- 扫描队列大小：10、20、50、100、150、200
- 每个队列大小 20 个快照

### 测试 1：单次决策

对每个快照，两个求解器各做一次决策，`budget = total_weight - 1`：

```
dp_keep, dp_outsource = dp_scaled.solve(items, budget)
gr_keep, gr_outsource = fractional.solve(items, budget)
```

我们测量：
- **精确一致率**：两个求解器是否外发同一个请求？
- **Jaccard 相似度**：外发集合的重叠
- **价值差距**：|dp_value_kept - greedy_value_kept| / total_value
- **求解延迟**：每次求解的墙钟时间

### 测试 2：迭代外发

模拟 Nimbus 实际循环——一次外发一个请求：

```
for each solver independently:
    while removed_weight < target_fraction * original_total_weight:
        solve with budget = total_weight - 1
        remove the outsourced request from the item list
        repeat
```

目标比例扫描：10%、25%、50%。

我们测量：
- **逐步一致率**：每一轮两者是否选中同一请求？
- **累计价值差距**：全部迭代后，DP 多保留了多少价值？

### 为什么两个测试都重要

单次决策回答：「做一次决策时，有没有差别？」  
迭代回答：「决策会累积时，差距会不会放大？」

若背包在单次决策中只略优，但在迭代模式下差距会放大，则对 Nimbus（本身就是迭代运行）仍然重要。

## 结果

### 单次决策

| 队列大小 | 一致率 | 平均价值差距 | 最大价值差距 | DP 时间 | 贪心时间 |
|-----------|-----------|---------------|--------------|---------|-------------|
| 10 | 0% | 11.8% | 20.7% | 9ms | 0.008ms |
| 20 | 0% | 7.3% | 12.7% | 20ms | 0.019ms |
| 50 | 0% | 3.7% | 8.0% | 48ms | 0.03ms |
| 100 | 0% | 1.8% | 4.0% | 105ms | 0.06ms |
| 150 | 0% | 0.8% | 1.4% | 150ms | 0.08ms |
| 200 | 5% | 0.4% | 0.8% | 201ms | 0.11ms |

### 迭代（节选）

| 队列大小 | 目标 | 逐步一致率 | 平均价值差距 |
|-----------|--------|-------------------|---------------|
| 10 | 25% | 0% | 21.4% |
| 20 | 25% | 0% | 33.4% |
| 50 | 25% | 0% | 24.8% |
| 50 | 50% | 0% | 39.1% |
| 100 | 25% | 0% | 20.3% |
| 200 | 25% | 0.8% | 3.8% |

## 分析

### 为什么几乎从不一致？

因为 LLM 请求负载的 **weight-to-value 比值高度异质**。一个 `prefill * decode` 很大（高 weight）的请求可能 API 成本很低（低 value），反之亦然。贪心按比值排序，挑「单个看起来高效」的物品，却错过全局打包更好的组合。

经典例子：
```
Item A: weight=1000, value=50  (ratio=0.05)  <- large displacement, cheap
Item B: weight=100,  value=40  (ratio=0.40)  <- small displacement, medium
Item C: weight=80,   value=35  (ratio=0.44)  <- small displacement, medium
Budget = 1179

Greedy: keep C+B (value=75), outsource A
DP:     keep A+C (value=85), outsource B
DP wins by 13%.
```

贪心被 A 的低比值骗了，但 A 其实是最值得单独保留的物品，因为它几乎独自填满预算。

### 为什么迭代模式下差距会放大？

每一轮贪心都做略次优的选择。「错误」请求留在队列里，扭曲后续决策。经过 10–20 轮，误差累积，导致 20–45% 的累计价值差距。

### 贪心快 1000 倍

DP：每次求解 10–200ms。贪心：0.01–0.1ms。对 Nimbus 的在线决策循环（每次违约检查都会跑，可能每秒多次），DP 延迟可接受但不可忽略。

## 结论

**背包（dp_scaled）是合理的。** 价值差距有意义：
- 单次决策（小队列）4–12%
- 迭代模式 20–45%

对这类负载，贪心排序**不等价**于背包。LLM 请求特征的异质性（prefill/decode 分布重尾且弱相关）恰好构成 0/1 背包优于贪心的条件。

**建议**：生产环境继续用 dp_scaled。10–200ms 的求解时间对 Nimbus 的违约检查节奏是可接受的。

## 产物

- 脚本：`scripts/analysis/exp_knapsack_vs_sorting.py`
- 数据：`docs/images/knapsack_vs_sorting.json`
