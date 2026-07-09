[English](nimbus_v2_pitch.md) | **简体中文**

# Nimbus v2：混合 LLM 推理中的缓存置换（Cache Displacement）

**目标会议**：EuroSys  
**作者**：Murphy, Yiyan Zhai, Yiyu Liu, Juncheng Yang  
**日期**：2026 年 4 月

---

## 一句话论点

混合 LLM 推理系统之所以外发了错误的请求，是因为它们在优化算力（FLOPs），而真正的瓶颈是**缓存置换**（内存 × 时间）。

---

## 问题

本地部署 LLM 便宜，但能力有限。当突发流量超过本地容量时，部分请求必须外发到云端 API。问题是：**该外发哪些请求？**

现有系统（包括我们的 Nimbus v1）用 **FLOP 代价** 作为外发权重：优先外发算力最贵的请求，以释放最多 GPU 周期。

**这是错的。** 在带前缀缓存（prefix-caching）的 LLM 服务中，主导代价不是算力，而是**缓存置换**——一个请求会在多大程度上削弱系统用缓存服务后续请求的能力。

---

## 关键洞察：缓存未命中 ≫ 批处理惩罚（152×）

我们在 Qwen2.5-7B（RTX PRO 6000 Blackwell）上测量了两种惩罚：

| 惩罚 | 量级 | 含义 |
|---|---|---|
| **缓存未命中**（前缀缓存被驱逐） | **251× TTFT** | 丢失缓存是灾难性的 |
| **批处理争用**（更大 batch） | **1.65× TPOT** | 共享 GPU 的影响较轻 |
| **比值** | **152×** | 保护缓存比节省算力重要 152 倍 |

因此正确的外发目标是：**保护前缀缓存**，而不是节省算力。

---

## 双稳态：缓存崩溃是悬崖，不是缓坡

带前缀缓存的 LLM 服务呈现双稳态——两个稳定状态之间存在尖锐、带滞后的相变：

- **健康态**：高缓存命中 → 服务快 → 队列短 → 内存压力低 → 缓存得以保留 → 高缓存命中（自我强化）
- **饱和态**：缓存被驱逐 → 服务变慢 → 队列堆积 → 内存压力升高 → 更多驱逐 → 缓存殆尽（自我强化）

**实验证据**（Qwen2.5-7B，ShareGPT+BurstGPT trace，28K 请求）：

| 指标 | 数值 |
|---|---|
| 相变 | TTFT 退化 251× |
| 滞后间隙 | 相同外发比例下 470× |
| 策略不变性 | 饱和后所有路由策略等价 |
| 会话一致性 | 比 oracle 请求级准入好 37× |

这不是渐进退化，而是反馈驱动的悬崖。一旦跌落，没有剧烈手段就爬不回来。

**经典类比**：Denning 的 thrashing（1968）——虚拟内存中工作集与缺页之间的同一类反馈环。

---

## 缓存置换 = prefill × decode

请求 r 在 GPU 中占用的 KV 内存 KV_memory(r) ∝ prefill_tokens，并在整个 decode 阶段持续 decode_tokens 步。总资源占用为：

```
cache_displacement(r) = ∫₀ᵈᵉᶜᵒᵈᵉ KV_memory(r) dt
                      = KV_per_token × prefill_tokens × decode_tokens
```

这是**内存-时间乘积**——资源占用对时间的积分。乘法来自积分本身，不是任意设计选择。

**经典类比**：Denning 的页面驻留时间（1968）、TCP 带宽时延积、作业调度中的曲线下面积。

---

## 基于 FLOP 会选错请求

FLOP 权重由 prefill²（二次注意力）主导。它会外发「长输入 / 短输出」请求（算力大，缓存置换小）。

缓存置换则外发高「内存 × 时间」请求（大 KV 被长时间持有）。

在真实编程 agent trace（Rednote，10K 请求）上：

| 指标 | 数值 |
|---|---|
| 输入/输出相关性 | 0.071（接近零——长输入 ≠ 长输出） |
| Top-10% 选择重叠 | **11%**（89% 的请求不同！） |
| 5% 外发时的 decode 缓解 | CacheDisp 比 FLOP 多 **6.3×** |

两种策略在生产负载上几乎**正交**。

### 为什么是乘，而不是加 / max / 其他？——六路消融

| 权重函数 | 移除的 CacheDisp | 排名 |
|---|---|---|
| **prefill × decode（本文）** | **262M** | **#1** |
| 仅 decode | 201M | #2 |
| prefill² + decode² | 140M | #3 |
| prefill + decode | 139M | #4 |
| FLOPs（Nimbus v1） | 132M | #5 |
| 仅 prefill | 117M | #6 |

乘法同时捕获两个维度（内存占用 × 持有时间）。加法会被较大的那一维主导，丢掉另一维。

---

## 端到端结果

**设置**：Qwen2.5-7B-Instruct，RTX PRO 6000 Blackwell 96GB，ShareGPT+BurstGPT trace（28K 请求），在线 SGLang 服务，trace 以 5× 加速回放。

### 按外发比例的 TTFT p50（ms）

| 比例 | Session-aware | FLOP-based | **CacheDisp** | Oracle（按大小） | **CD/FLOP** |
|---|---|---|---|---|---|
| 0% | 143,395 | 103,736 | 99,395 | 106,101 | 1.0× |
| **15%** | 35,344 | **20,572** | **1,756** | 48,466 | **11.7×** |
| 20% | 19,372 | 466 | **277** | 1,474 | 1.7× |
| 25% | 5,324 | 161 | **108** | 225 | 1.5× |
| 30% | 416 | 96 | 72 | 103 | 1.3× |
| 50% | 68 | 58 | 58 | 59 | 1.0× |

**核心数字**：在 15% 外发时，CacheDisp 达到 1.8s TTFT，而 FLOP 仍为 20.6s——仅因选对了外发请求，就获得 **11.7×** 提升。

### 机制证据

在关键的 15% 比例处：

| 指标 | CacheDisp | FLOP-based | Session-aware |
|---|---|---|---|
| KV 利用率 p90 | **12.1%** | 19.5% | 27.1% |
| 最大队列深度 | **18** | 1,482 | 2,683 |

CacheDisp 把内存压力压得足够低，从而保住前缀缓存。FLOP-based 仍会触发队列堆积（峰值 1,482）→ 缓存驱逐 → 双稳态。

### 意外发现：Oracle（按 prefill 大小）在 15% 时最差

Oracle 外发最大 KV 请求（长 prefill，但短 decode）。这能短暂释放内存，却不能减少 decode 批占用时间。CacheDisp 外发最高内存×时间乘积——同时释放两个维度。

### 美元效率

要满足 TTFT p99 < 5s 的 SLO：
- FLOP-based 需要约 30% 外发
- CacheDisp 需要约 25% 外发
- **在同等 SLO 下，CacheDisp 节省约 17% API 成本**

---

## 代码改了什么

`routing/outsourcing/decision.py` 里一行：

```python
# 之前（Nimbus v1 — 基于 FLOP）：
weight = int(prefill_flops + 0.6 * decode_flops)

# 之后（Nimbus v2 — 缓存置换）：
weight = int(req.prefill_tokens * req.estimated_decode_tokens)
```

Nimbus 系统其余部分（TTFT 预测器、迭代背包、前缀缓存感知）保持不变。

### 输出 token 估计不是问题

请求到达时 decode_tokens 未知。但排序鲁棒性分析表明：

| 估计噪声 | 与真实排序的 Spearman ρ | Top-10% 重叠 |
|---|---|---|
| 1.5× 噪声 | 0.999 | 91.6% |
| 3× 噪声 | 0.992 | 80.4% |
| 5× 噪声 | 0.984 | 76.6% |
| **基于 FLOP** | **0.932** | **53.1%** |

即便 5× 噪声的估计，排序也优于精确 FLOPs。实践中可用客户端指定的 `max_tokens`，或会话历史均值。

---

## 论文贡献

| # | 贡献 | 类型 | 关键数字 |
|---|---|---|---|
| C1 | 缓存未命中 ≫ 批处理惩罚（152×） | 测量 | 152× 比值 |
| C2 | 缓存置换指标（内存-时间乘积） | 理论 + 消融 | 六路消融第 1 |
| C3 | 基于 FLOP 会选错请求 | 关键发现 | 89% 不同，端到端差距 11.7× |
| C4 | 前缀缓存服务中的双稳态 | 形式化 | 251× TTFT，470× 滞后 |
| C5 | Nimbus v2 系统 | 系统 | 15% 时 11.7×，节省 17% 成本 |
| C6 | FreeInference + 生产 trace | 基础设施 | 55K LOC，3 条 trace |

---

## 相关工作定位

- **Mooncake、Preble、SGLang Router**：缓存感知路由——决定发到**哪里**。我们决定外发**什么**。正交。
- **vLLM、SGLang、FlashAttention**：服务引擎——提升本地吞吐。我们决定本地不够时怎么办。
- **Metastable failures（OSDI'22）**：一般框架。我们展示 LLM 服务中的一个具体实例，并用于外发决策。
- **Splitwise、DistServe**：Prefill/decode 分离。正交轴。

---

## 实验状态

- [x] 双稳态测量（251×，470× 滞后）— 完成
- [x] 六路权重函数消融 — 完成
- [x] 3 条 trace 上的 DI 反转 — 完成
- [x] 端到端 knee sweep（4 策略 × 7 比例）— 完成
- [x] 三面板机制图（TTFT + KV 利用率 + 队列深度）— 完成
- [x] 输出估计鲁棒性分析 — 完成
- [ ] 近 knee 重复实验以画误差棒 — 运行中（约 9h）
- [ ] Rednote trace 验证（可选，预期差距更大）
- [ ] Nimbus v2 集成测试（背包 + TTFT 预测器）

---

## 关键图

- `docs/images/cachedisp_knee_7b_3panel.pdf` — 主图（TTFT + 机制）
- `docs/images/cachedisp_knee_7b.pdf` — 双面板（TTFT p50 + p99）

## 数据

- Knee sweep 结果：gpu1 `logs/cachedisp_knee1/`
- 重复实验：gpu1 `logs/cachedisp_repeats/`（运行中）
- 双稳态数据：gpu1 `logs/phase2_7b_rednote/`
