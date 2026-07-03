# popt — Prompt Structure Optimizer

**popt** 是一个轻量级的 prompt 结构优化器。它通过重组消息结构来最大化 LLM API 的 [Prompt Caching](https://api-docs.deepseek.com/guides/kv_cache) 命中率——不改语义，不调用 LLM。

```bash
# 安装（需要 Python 3.10+）
pip install popt

# 诊断两个请求为什么缓存表现不同
popt diagnose request_a.json request_b.json

# 预览优化效果
popt preview request.json --provider deepseek

# 启动本地代理（透明优化所有请求）
popt proxy --port 9999
```

---

## 为什么需要 popt？

LLM API 的 Prompt Caching 基于一个简单原理：**两次请求的前缀字节完全一致的部分越多，跳过计算的比例越大**。

```
请求 1: [system=A] [user=X] [assistant=...] [user=Y]  → 建缓存
请求 2: [system=A] [user=X] [assistant=...] [user=Z]  → 命中 75%
         ^^^^^^^^  ^^^^^^^  ^^^^^^^^^^^^^^
         前缀三块完全一致，直接从缓存读
```

但实际中，**相同的语义内容可能因格式差异产生不同的字节序列**，导致缓存 miss：

| 原因 | 示例 |
|------|------|
| JSON 字段顺序不同 | `{"role":"user","content":"hi"}` vs `{"content":"hi","role":"user"}` |
| 换行符不统一 | `\n` vs `\r\n` |
| 消息顺序不稳定 | system 在第 1 位 vs 第 3 位 |
| 输入太短 | 10 tokens 不够一个缓存块（128t） |

**popt 解决这些问题**：标准化 → 稳定排序 → 对齐阈值 → 格式化输出。

---

## 安装

```bash
pip install popt
```

可选依赖（强烈建议安装，大幅提升 token 估算精度）：

```bash
pip install tiktoken
```

---

## 快速开始

### Python API

```python
from popt import optimize, preview

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
]

# 预览优化效果
report = preview(messages, provider="deepseek")
print(report)
# {
#   'message_count': {'before': 2, 'after': 2, 'separators_added': 0},
#   'estimated_tokens': {'before': 18, 'after': 289},
#   'meets_threshold': True,
# }

# 执行优化
optimized = optimize(messages, provider="deepseek")
# 直接用于 API 调用
```

### 诊断两个请求的差异

```bash
popt diagnose req_a.json req_b.json -p deepseek
```

输出：

```
  -- Shape comparison --
    X System                 6128abab    67617710    ← system prompt 变了
    = Role sequence          386b504e    386b504e    ← 角色顺序未变

  -- First difference --
    Position:   ~token 0
    Block:      0  !! CRITICAL        ← 在第 0 块差异，整段缓存失效
    Message:    [0] role=system
    Snippet A:  Today is 2026-07-03.
    Snippet B:  Today is 2026-07-04.

    Suggestion: 如果差异是变量（日期、用户名），建议移到消息末尾
```

### 透明代理

```bash
popt proxy --port 9999
# 然后将 API 客户端指向本地代理：
#   export DEEPSEEK_BASE_URL=http://localhost:9999/v1
# 或直接配置 SDK 的 base_url
```

---

## CLI 命令

| 命令 | 作用 |
|------|------|
| `popt preview <file> [-p <provider>]` | 预览优化效果，显示 prefix shape hash |
| `popt diagnose <a.json> <b.json> [-p <provider>]` | 对比两个请求，定位缓存 miss 的根因 |
| `popt proxy [--port PORT]` | 启动透明代理 |
| `popt stats [<file>]` | 从日志文件分析缓存命中率 |

### popt preview

```bash
popt preview request.json -p deepseek --verbose
popt preview request.json -p deepseek --show   # 同时显示优化后的消息
```

### popt diagnose

```bash
popt diagnose request_a.json request_b.json
popt diagnose request_a.json request_b.json -p deepseek --verbose
```

---

## 支持的 Provider

| Provider | 缓存机制 | 阈值 | 状态 |
|----------|---------|------|------|
| DeepSeek | 自动前缀匹配，128t 每块 | 128t | ✅ 实测验证 |
| Anthropic | `cache_control` breakpoint | 1024t | ⚠️ 代码已实现，未实测 |
| OpenAI | 自动前缀匹配 | 1025t | ⚠️ 代码已实现，未实测 |
| 自定义 | `register_provider(name, ...)` | 配置 | ✅ 5 行代码 |

---

## 架构

```
用户请求                     popt 代理 / API
    │                            │
    ▼                            ▼
输入消息 ──→ Normalizer ──→ Reorderer ──→ Aligner ──→ Formatter ──→ 输出
                │              │            │              │
                ▼              ▼            ▼              ▼
            标准化 whitespace  system 前置  检查并填充到    provider 特定
            规范化 JSON       同角色按     缓存阈值        格式，剥离内部标记
                               hash 排序   注入 cache_
                                           control 标记
```

### 各模块职责

| 模块 | 作用 | 对缓存的贡献 |
|------|------|-------------|
| **Normalizer** | `\r\n`→`\n`、JSON 键排序、BOM 清除 | 消除格式差异导致的意外 miss |
| **Reorderer** | system 永远在前，同角色按确定顺序排列 | 保证前缀在不同请求间稳定 |
| **Aligner** | 短 prompt 填充到阈值，加 `cache_control` | **唯一能创造缓存价值的模块** |
| **Formatter** | 剥离内部标记，输出 provider 格式 | 兼容不同 API 的入参要求 |

### 安全机制

每个阶段都有 SafetyCheck 保底——如果优化改变了语义内容，回退到原始输入。检测项：

- 原始内容集是否为优化结果的子集
- System prompt 是否被修改
- 消息角色是否被篡改

---

## 实验验证

在 DeepSeek v4 flash 上的实际测试数据：

| 场景 | 原始 | 优化后 | 说明 |
|------|------|--------|------|
| **短 prompt (~14t)** | 0.0% | **60.4%** | 填充到 128t 阈值，缓存 1 块 |
| **中 prompt (~203t)** | 63.1% | 63.4% | 已达阈值，不做多余操作 |
| **长 prompt (~2745t)** | 97.9% | 98.0% | 正常缓存，跳过 padding |

运行实验：

```bash
python examples/experiment_deepseek.py
```

（需要 `DEEPSEEK_API_KEY` 环境变量）

---

## 开发

```bash
# 安装依赖
pip install tiktoken pytest

# 运行测试
python -m pytest

# 特定测试
python -m pytest tests/test_diagnoser.py -v
```

---

## 项目状态

```
技术验证（核心流水线 + 诊断）  →  80%
实测验证（多 provider）         →  30%
产品化（文档、包发布）           →  10%
消息修复（代理层防御）           →   0%
```

当前聚焦：完善诊断能力和产品形态。

---

## License

MIT
