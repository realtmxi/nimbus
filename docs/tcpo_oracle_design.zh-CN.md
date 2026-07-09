[English](tcpo_oracle_design.md) | **简体中文**

# TCPO：面向 Nimbus 的 Trace 全知压力 Oracle

## 1. 为什么需要离线 Oracle

Nimbus v1 用 `prefill_tokens * decode_tokens`（缓存置换）作为背包权重，决定外发哪些请求。在 15% 外发比例下，相对基于 FLOP 的方法可获得 11.7× TTFT 提升。

但一个自然问题仍在：**CacheDisp 离最优外发决策还有多远？** 没有离线最优基线，就无法回答。

### 现有工作在做什么

近期论文（Jaillet et al. 2025、Ao et al. 2025、Sorted-F 2025）把带 KV 缓存约束的 LLM 推理离线最优调度形式化了。但它们都假设**单一本地集群**，并优化批调度顺序。没有人把**外发到外部 API**当作决策变量。

### 我们需要什么

一个离线 oracle，满足：
- 拥有完整未来信息（全部请求到达、token 数、前缀重叠）
- 做与 Nimbus **同类**的决策：逐请求 local vs remote
- **不**改动本地服务栈（不重调度、不改缓存策略）
- 能扩展到真实生产 trace（28K 请求）
- 能对照真实 SGLang 回放做验证

## 2. 关键洞察：双稳态简化了问题

双稳态发现意味着系统只有两个状态：
- **健康态**：TTFT ~ 100ms，前缀缓存得以保留，队列短
- **崩溃态**：TTFT ~ 100,000ms，前缀缓存被驱逐，队列爆炸

两者之间没有平滑退化。因此：

> **我们不需要预测精确 TTFT。只需保证系统永不越过悬崖阈值。**

这把 oracle 从「预测每一种分配下的延迟」（不可行）降为「在每个时刻把内存压力压在阈值以下」（经典优化问题）。

## 3. 问题形式化

### 时间背包问题（TKP）

我们的问题直接对应经典的 **Temporal Knapsack Problem**（Bartholdi 1980，Caprara et al. 2013），也等价于 **Unsplittable Flow on a Path**（Grandoni et al. STOC 2022，存在 PTAS）。

### 设定

给定 N 个请求的 trace，每个请求 i 有：
- `arrival_i`：到达时间
- `b_i`：block_hash_ids（其 KV 缓存 block 序列）
- `N_i = |b_i|`：总 block 数
- `decode_tokens_i`：输出 token 数
- `c_i`：外发时的 API 成本

给定系统参数：
- `C_safe`：KV 缓存压力阈值（悬崖以下）
- `T_prefill`：prefill 吞吐（tokens/sec，已 profile）
- `TPOT_healthy`：健康态下每输出 token 时间（已 profile）

### 决策变量

```
Y = (y_1, y_2, ..., y_N)
y_i in {0, 1}
y_i = 1 表示请求 i 外发到 API
y_i = 0 表示请求 i 本地服务
```

### 逐请求压力剖面

对每个本地服务的请求 i（y_i = 0），计算：

**边际内存占用**（因前缀共享而依赖 Y）：
```
h_i(Y) = 前缀缓存命中长度（来自先前请求的已缓存 block）
m_i(Y) = N_i - h_i(Y)          （本请求新插入的 block）
```

**时间**（来自已 profile 的吞吐，**不是**崩溃态跑出来的）：
```
s_i = arrival_i                                         （压力开始）
e_i = arrival_i + m_i * block_size / T_prefill          （prefill 完成）
      + decode_tokens_i * TPOT_healthy                  （decode 完成，压力结束）
```

**对时间桶 k 的压力贡献**：
```
q_{ik}(Y) = m_i(Y)    若桶 k 与 [s_i, e_i) 重叠
          = 0          否则
```

### 目标与约束

```
minimize    sum_i  c_i * y_i                       （总 API 成本）

subject to  for all time buckets k:
            sum_i  q_{ik}(Y) * (1 - y_i)  <=  C_safe
                                                   （保持在悬崖以下）

            y_i in {0, 1}
```

用自然语言说：找成本最小的外发请求集合，使得本地 GPU 的 KV 缓存压力在任意时刻都不超过安全阈值。

### 为什么 `q_{ik}` 依赖 Y（前缀耦合）

若请求 A 被外发，其 block 永不插入本地缓存。同会话后续请求 B、C 会失去前缀命中：
- `h_B` 下降（可匹配的已缓存 block 更少）
- `m_B` 上升（需插入的新 block 更多）
- `q_{Bk}` 变大（压力矩形更大）

该耦合**仅发生在会话内**（已验证：我们 trace 中的 block_hash_ids 按会话命名空间隔离，跨会话重叠 = 0）。

## 4. 算法：不动点 TCPO

因为 `q_{ik}` 依赖 Y，不能一次解完 ILP。我们用不动点迭代：

```
Algorithm: Fixed-Point TCPO

Input:  trace, T_prefill, TPOT_healthy, C_safe
Output: outsourcing assignment Y*

1. Initialize Y^(0) = 0  (all requests local)

2. For t = 0, 1, 2, ..., T_max:

   a. REPLAY: Run logical radix tracker with assignment Y^(t)
      - Process requests in arrival order
      - Maintain LRU block cache (same policy as SGLang RadixAttention)
      - For outsourced requests (y_i = 1): skip, do not insert blocks
      - For local requests (y_i = 0):
          * Match prefix -> compute h_i, m_i
          * Insert new blocks at prefill_done time (not arrival)
          * Record (s_i, e_i, m_i)

   b. BUILD: Construct sparse pressure profile
      - For each local request, record active time interval and m_i
      - Index by time bucket for constraint building

   c. SOLVE: ILP with PuLP/CBC
      - min  sum c_i * y_i
      - s.t. per-bucket pressure constraints (sparse)
      - Extract new assignment Y^(t+1)

   d. CHECK: If Y^(t+1) == Y^(t) or change < 1%, stop.

3. Return Y^(T)
```

### 收敛性

这是 best-response 动态。经验上预期 2–5 次迭代收敛，因为：
- 外发更多请求只会增大同会话请求的 m_i
- 依赖图按会话划分（会话间不相交）
- 会话通常很小（典型 2–5 个请求）

安全措施：上限 T_max = 5 次迭代，取最后一次分配。

### 一致性检查（是否真的需要不动点？）

在投入迭代前先测量：在全本地（Y=0）与 CacheDisp-15% 分配下分别跑 tracker，比较剩余本地请求的 m_i。若平均 delta_m < 5%，则一次求解的 TCPO 就够用。

## 5. 计算 `m_i`：逻辑 Radix Tracker

这是核心计算。它**不是**模拟器——只跟踪哪些 block 在缓存中、哪些不在。

```python
class BlockRadixCache:
    """Logical LRU block cache mimicking SGLang RadixAttention."""

    def __init__(self, capacity_blocks):
        self.capacity = capacity_blocks
        self.cache = {}  # block_hash -> last_access_time

    def match_and_insert(self, block_hash_ids, insert_time):
        """Match prefix, insert new blocks at insert_time.

        Returns (h_i, m_i): prefix hit length and marginal new blocks.
        """
        # (a) Count consecutive prefix hits
        h_i = 0
        for bh in block_hash_ids:
            if bh in self.cache:
                self.cache[bh] = insert_time  # LRU touch
                h_i += 1
            else:
                break

        # (b) Insert new blocks (evict LRU if full)
        new_blocks = block_hash_ids[h_i:]
        for bh in new_blocks:
            if len(self.cache) >= self.capacity:
                lru_key = min(self.cache, key=self.cache.get)
                del self.cache[lru_key]
            self.cache[bh] = insert_time

        m_i = len(new_blocks)
        return h_i, m_i
```

约 30 行 Python。无 GPU、无时序模拟、无 batch 建模。

### 重要细节：在 prefill_done 时插入，而非 arrival

在 SGLang 中，KV block 只有在 prefill **完成之后**才会进入 radix 树并可复用。因此：

```
insert_time = arrival_time + m_i * block_size / T_prefill
```

不是 `arrival_time`。若用到达时间，会对紧挨着到达的请求高估前缀命中。

## 6. 标定

### TPOT_healthy

来源：knee sweep 实验中的健康态比例（例如 CacheDisp 在 25%）。
从 `metrics.csv` 提取：
```
TPOT_healthy = 1 / mean(gen_throughput) over stable window
```
不要用 f00（崩溃态）或 batch=1 的 profile。

### C_safe

来源：knee sweep 实验。
- 对每个外发比例，记录 `metrics.csv` 中的峰值 `token_usage`
- 对每个比例，检查 TTFT p50 是否 < 1 秒（健康）
- `C_safe = 0.9 * max(peak_token_usage where fraction is healthy)`

这是从真实实验标定的系统参数，不是崩溃 trace 里的数字。

## 7. 验证

### 第 1 层：下界（LB）

从全本地压力剖面，计算必须移除的最小压力：
```
LB = sum over all buckets k:  max(0, pressure_k - C_safe)
```
任何策略至少要外发这么多 memory-time。

### 第 2 层：真实回放（主验证）

把 TCPO 的分配 Y* 作为新策略喂给 `run_offload_strategies.py`：
```python
class TCPOStrategy(OffloadStrategy):
    def should_outsource(self, req, kv_pressure):
        return self.precomputed_Y[req.idx] == 1
```
在真实 SGLang 上跑。测量 TTFT、成本、外发比例。与同比例的 CacheDisp 对比。

### 第 3 层：反事实回放 Oracle（CRO）

在 2–3 个采样过载窗口上，用真实 SGLang 回放作评估器，对外发子集做局部搜索。验证 TCPO 的选择接近回放搜索最优（Jaccard > 0.7）。

## 8. 我们声称什么（以及不声称什么）

### 我们声称

TCPO 是**在回放标定的 KV 压力模型下的路由上界**。在约化模型内（由双稳态合理化的容量约束 + 前缀感知边际压力）它是最优的。

### 我们不声称

TCPO 是真实系统的全局最优。约化模型忽略了：
- 批处理争用（较轻，约 1.65× TPOT，已体现在 TPOT_healthy 中）
- 悬崖之外的队列动态（我们只建模悬崖 vs 非悬崖）
- 跨会话前缀共享（当前 trace 中为零）

### 论文叙事

> CacheDisp（在线启发式）是 TCPO（离线 oracle）所操作的压力剖面的**标量松弛**。CacheDisp 用 `prefill_tokens * decode_tokens` 作为每个请求压力矩形的单数字摘要。TCPO 使用完整的时间展开剖面。若二者差距很小，则 CacheDisp 接近最优。

## 9. 相关工作

| 论文 | 做什么 | 与 TCPO 的差异 |
|-------|-------------|---------------------|
| Jaillet et al. 2025 | 单集群 LLM 调度的 IP，40–60 请求 | 无外发、无前缀感知、不可扩展 |
| Ao et al. WAIT 2025 | 单集群吞吐的流体模型 | 无外发、无前缀感知 |
| Sorted-F 2025 | 调度的 NP-hard + 常数近似 | 单集群，延迟目标 |
| INFERCEPT ICML'24 | 工具调用 KV 的 (unused_mem * unused_time) | 不同场景（工具暂停，非外发） |
| Hybrid LLM ICLR'24 | 质量驱动的本地 vs 云路由 | 无资源建模（KV、内存、队列） |
| TKP (Bartholdi'80, Caprara'13) | 经典时间背包 | 我们加入前缀感知的 m_i(Y) 耦合 |
| UFP PTAS (Grandoni STOC'22) | 存在 (1+eps) 近似 | 我们依托的理论工具 |

TCPO 是**首个把外发作为一等决策变量、并做前缀缓存感知压力建模的混合 LLM 推理离线最优形式化**。
