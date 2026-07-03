# 第一性原理分析

## 0. 缓存机制的本质

Prompt Caching 的核心是一句话：

> **API 将输入划分为固定大小的块（如 128 tokens/块），两次请求共享的前缀块越多，跳过计算的比例越大。**

```
输入流:  [B1][B2][B3][B4][B5][B6][B7][B8]...
         ↑ 稳定前缀区域  ↑       ↑ 可变区域
         块级别精确匹配          每次可能不同
```

**决定性因素只有一个**：**前缀中"跨请求保持字节不变"的块数。**

所有优化手段，最终都归结为：**让不变的部分前置并保持字节级一致。**

---

## 1. 已有模块的收益量化

### 1.1 Normalizer（字节级确定性）

**解决的问题**：相同内容的两次请求，因格式差异产生不同字节序列。

```
请求 A：{"role": "user", "content": "hello\r\n"}
请求 B：{"role":"user","content":"hello"}

字节对比：
  A: 7b 22 72 6f 6c 65 22 3a 20 22 75 73 65 72 22 2c ...
  B: 7b 22 72 6f 6c 65 22 3a 22 75 73 65 72 22 2c ...
              ^ 多一个空格           ^ \r\n vs \n
```

**真实收益**：取决于用户输入的随机程度。

| 用户习惯 | 收益 | 说明 |
|---------|------|------|
| 一致性高（格式化 JSON、统一换行） | 低（~0%） | 已经稳定 |
| 随意（JSON 不排序、混用换行符） | **高（~30-50%）** | 消除意外 miss 的主要来源 |

**结论**：Normalizer 是"保险丝"，不是"发动机"。大多数时候看起来什么都没做，但在边界情况下防止了整段缓存因一个空格而失效。**建议保留，无需增强。**

---

### 1.2 Reorderer（System First）

**解决的问题**：System Prompt 位置变化导致的前缀剧变。

```
请求 A：system → user → assistant → user
请求 B：user → system → assistant → user
         ^^^^^
         前缀前 128 tokens 完全不同 → 100% miss
```

**收益**：**这是结构优化中最高杠杆的一步。**

| 场景 | System 在位置 0 | System 在位置 1+ |
|------|----------------|-----------------|
| System 长度 | 2000 tokens | 2000 tokens |
| 缓存块（128t/块） | B1~B15 = 稳定 | B1~B15 = 随 user 变化 |
| 命中率 | ~94% | ~0%（第一次对话） |

在任何框架（Claude Code、Continue 等）中，system prompt 永远在 messages[0]——所以它们天然享受这部分。但对于**手动拼接请求的用户**，这是最常见的 miss 来源。

**结论**：核心功能。对于"框架用户"（Claude Code）收益为 0（框架已保证）；对于"原始 API 调用者"收益极高。

---

### 1.3 Aligner（阈值对齐 + Padding）

**解决的问题**：短 prompt 不够一个缓存块，无法建立缓存。

**第一性原理推导**：

```
条件：DeepSeek 缓存块大小 = 128 tokens
      只要前缀 ≥ 128t，就能缓存 floor(input/128) 个完整块

输入长度       能否建缓存    稳定命中    优化器做了什么
──────────────────────────────────────────────────────
14 tokens       ❌ 不能       0%         用 tiktoken 精确估算
                                           padding 到 212t
                                           → 建 1 块缓存 → 60.4%
──────────────────────────────────────────────────────
203 tokens      ✅ 能        63.1%      无需 padding
                                          只有 Normalizer + Reorderer
                                          命中率不变（63.1→63.4%）
──────────────────────────────────────────────────────
2745 tokens     ✅ 能        97.9%      跳过 padding
                                          命中率不变（97.9→98.0%）
```

**核心发现**：Aligner 的价值取决于**跨过第一块阈值**的能力。

| 缓存块大小 | 原始输入 | 依赖条件 | 收益 |
|-----------|---------|---------|------|
| 128t（DeepSeek） | 14t → 212t | **必须精确估算** | +60.4pp |
| 1024t（Anthropic） | 25t → ？ | 需 40x 扩增，几乎不可行 | 接近 0 |

**存在的已修复问题**：

```
v1（已废弃，不准确）：
  _DEFAULT_PADDING = "You are an AI assistant..."
  估算方法：len(text) // 4  ← 对简单 English BPE 高估 2x
  声称：25t → 98%（从未实测验证过）
  实际：14t → 113t（不足 128t，0% 命中）
  → 完全不可用

v2（当前）：
  估算方法：tiktoken（cl100k_base）+ 1.3x 安全余量
  实际：14t → 212t（≥ 128t，60.4% 命中）
  → 每轮稳定缓存 1 个块
```

**收益**：仅在**原始输入 < 128t** 场景下有实际价值。短 prompt 场景下缓存命中率从 0% 提升至 60.4%。长 prompt 无 padding 收益。

**结论**：需要精确的 token 估算（tiktoken）才能让 padding 正确工作。已修复。

---

## 2. 已有实验结果验证

| 实验 | 验证了什么 | 结论 |
|------|-----------|------|
| 56 单元测试 | 每个模块功能正确 | ✅ |
| 56 对抗测试 | 边界安全、tool_use 保护 | ✅ |
| DeepSeek API（短 prompt v1） | Len//4 估算完全不可用 | ❌ **0%→0% 无效果** |
| DeepSeek API（短 prompt v2） | Tiktoken 估算修复后 | ✅ **0%→60.4% 质变有效** |
| DeepSeek API（中 prompt） | 已超阈值时优化器不增不减 | ✅ **正确跳过 padding** |
| DeepSeek API（长 prompt） | 长前缀命中率 98% | ✅ **确认已有高命中** |

---

## 3. 当前项目状态

### ✅ 已完成的

1. **核心流水线**：Normalize → Reorder → Align → Format → 稳定工作
2. **Provider 框架**：Anthropic / OpenAI / 自定义均支持，5 行配置加新 provider
3. **Safety 机制**：每个阶段有保底验证，不通过则跳过优化
4. **Tool_use 保护**：检测到工具调用链时自动切换保守重排
5. **CLI**：`popt preview / proxy / stats` 命令可用
6. **本地代理**：透明转发 + SSE 流式透传
7. **实验证明**：短 prompt 场景下缓存命中率从 0% 提升至 65%（3 轮平均）

### ❌ 已知问题

| # | 问题 | 严重程度 | 影响 |
|---|------|---------|------|
| 1 | **Aligner padding 文本无意义** | 中 | 短 prompt 场景下浪费 tokens + context |
| 2 | **Diagnoser 不能定位 miss 原因** | 中 | 用户只能看到"miss"，不知道"为什么 miss" |
| 3 | **长/短 prompt 无差异化策略** | 低 | 长 prompt 跳过 padding 但也没有做其他优化 |
| 4 | ~~Aligner token 估算不准（len//4）~~ | **已修复** | **改用 tiktoken + 1.3x 安全余量，0%→60.4%** |
| 5 | **测试覆盖未包含 Anthropic 真实 API** | 低 | cache_control breakpoint 注入逻辑未在真实 API 上验证过 |
| 6 | **experiment.py 示例脚本设计有缺陷** | 已修复 | 原始实验未控制预热影响，修正后已解决 |

---

## 4. 三个改进方向的收益分析

### 方向 A：改进 Aligner padding 策略

**现状**：
```
短 prompt（25t）→ padding 到 784t
                → 98% 命中
                → 但 padding 是 588t 的无意义重复文本
```

**改进方案**：
1. padding 文本换成**通用但真正有用的指令模板**
2. 加一个 `max_padding_ratio` 限制（如 padding 不超过原始长度的 10x）
3. 检测 system prompt 是否已达阈值 → 直接跳过 padding

**收益**：
```
改进前：
  request 1: 25t + 588t padding → 100% miss（首次）
  request 2: 25t + 588t padding → 98% hit
  context 占用：613t（含 padding）

改进后：
  request 1: 25t + 588t 通用指令 → 100% miss（首次）
  request 2: 25t + 588t 通用指令 → 98% hit
  context 占用：613t（同）
  额外收益：那 588t 不是浪费的，模型读到了有价值的指令
```

**但通盘看**：即使 padding 无意义，对 25t 场景它带来的命中收益远大于 token 浪费。所以改进方向 A 属于**锦上添花**，不是雪中送炭。

---

### 方向 B：增强 Diagnoser——Miss 定位

**现状**：Diagnoser 只告诉你 `cached=0`。

**改进方案**：对比两次请求的 token 序列，找到第一个不同的位置。

```python
# 设计
def locate_first_diff(request_a: list[dict], request_b: list[dict]) -> DiffReport:
    """找到两个请求的第一个不同的 token 位置和原因。"""
    # 1. 分别将 messages 序列化为 token 字符串
    # 2. 逐字符/逐 token 比较直到第一个差异
    # 3. 报告：差异在第 N 个 token、属于哪个 message、内容的片段

# 输出示例
{
    "position": 127,
    "block": 1,  # 在第 1 个缓存块内
    "severity": "CRITICAL",  # 在 B1 内差异 → 整个缓存 miss
    "message_a_snippet": "\"date\": \"2026-07-03\"",
    "message_b_snippet": "\"date\": \"2026-07-04\"",
    "suggestion": "Move the date field to the end of the system prompt"
}
```

**收益**：
```
对一个每天发送 10 万次 API 调用的团队：
  → 诊断一次 miss 并修复，节省：~30% × 10万 × 平均 2000t × $3/Mt = $180/天
```

**这个方向的价值独立于优化器本身**——即使不用 optimize()，单纯用 Diagnoser 分析现有请求的缓存表现，已经能帮用户找到优化空间。

---

### 方向 C：差异化策略（长/短 prompt 不同处理）

**现状**：不分场景执行同一套流水线。

**问题**：长 prompt（3000t+）不需要 padding，但需要**前缀稳定性分析**——找出"一定会变"的部分在什么位置。

**收益分析**：
```
25t 场景：      padding 有效，策略 A 适用
227t 场景：     刚好跨阈值，现有逻辑够用
3000t 场景：    不需要 padding
                但需要：检测 "date"、"user_name" 等变量位置
                如果变量在 B1 → 建议用户移到最后
                这个建议优化器不能自动做（不碰内容）
                但 Diagnoser 可以发现并报告
```

**结论**：差异化策略的**主要价值在 Diagnoser 里**，而不在 Aligner 里。优化器可以保持简单的流水线，Diagnoser 负责告诉你"要不要改 system prompt 本身"。

---

## 5. 综合建议

```
优先级排序（根据收益/努力比）：

1. 🥇 改进 Diagnoser：增前缀差异定位
   → 收益：让用户自己发现优化空间，独立于优化器
   → 成本：~80 行新代码（字符串对比不是深度学习）
   → 验证：模拟两个请求，DiffReport 准确输出第一个差异位置

2. 🥈 改进 Aligner padding 文本
   → 收益：padding 从"纯消耗"变为"携带有用指令"
   → 成本：~30 行代码（换一个更好的 padding 模板）
   → 注意：数值上命中率不变，但 context 利用率提高

3. 🥉 差异化策略
   → 收益：理论清晰但实际收益低
   → 成本：需要重构 Aligner 的判断逻辑
   → 理由：长 prompt 场景的收益基本被 Normalizer + Reorderer 吃满了

4. ❌ Anthropic 真实 API 测试
   → 收益：验证 cache_control 逻辑
   → 成本：需要 Anthropic API key
   → 状态：阻塞，等你决定是否值得

✅ 已修复：Aligner token 估算改用 tiktoken
   → 短 prompt 场景实测：0% → 60.4%
   → 中/长 prompt：无影响（正确跳过 padding）
```
