# popt — Prompt Structure Optimizer

**popt** 是一个轻量级的 prompt HTTP 代理，自动优化消息结构来最大化 LLM API 的 [Prompt Caching](https://api-docs.deepseek.com/guides/kv_cache) 命中率。

```bash
pip install poptimize

# 一行命令：启动代理 + 运行 AI 工具，自动优化所有请求
popt run -- claude
popt run -- python my_agent.py
popt run --model gpt-4o -- python script.py
popt run --model deepseek-v4-flash -- node bot.js
```

**核心能力**：popt 是一个 **HTTP 透明代理**，插在任何 AI 框架和 API 之间。内置 15+ 服务商预设（DeepSeek、OpenAI、Anthropic、Groq、Together、Codex…），一条命令切换模型。不需要改 SDK、不需要装插件。

> 📖 **完整操作文档**：[docs/usage.md](docs/usage.md) — 从安装到实战的全部步骤

---

## 快速开始

### 🥇 `popt run` — 一条命令通吃所有框架

```bash
# 自动检测 ANTHROPIC_BASE_URL / OPENAI_BASE_URL 环境变量
popt run -- claude
popt run -- python my_script.py

# 指定模型（自动配 upstream + provider）
popt run --model deepseek-v4-flash -- claude
popt run --model gpt-4o -- python my_script.py
popt run --model grok-beta -- python script.py  # xAI Grok
popt run --model codex-* -- node bot.js          # GitHub Copilot
popt run --model llama-3-70b -- python bot.py    # Groq

# 或手动指定
popt run --upstream https://api.deepseek.com/anthropic --provider deepseek -- python script.py
```

`popt run` 会自动：
1. 从 `--model`、环境变量、`.poptimerc` 文件三级检测上游
2. 内置 15+ 服务商模型注册表，模型名自动匹配正确的 API 地址
3. 在随机端口启动优化代理
4. 设置子进程的环境变量指向代理
5. 运行你的命令
6. 命令结束后自动关闭代理

**自动检测优先级**：

```
upstream:  --upstream > --model > $POPT_UPSTREAM > $ANTHROPIC_BASE_URL > .poptimerc > 注册表推断
provider:  --provider > --model > $POPT_PROVIDER > 从URL推断 > .poptimerc > 默认
```

### 🥈 `popt proxy` — 长驻代理

```bash
# 启动代理（自动从环境变量读取配置）
popt proxy

# 或手动指定
popt proxy --port 9999 --upstream https://api.deepseek.com/anthropic --provider deepseek
```

然后在另一个终端正常使用你的 AI 工具：

```bash
# Claude Code (已配置 ANTHROPIC_BASE_URL)
claude

# OpenAI SDK (已配置 OPENAI_BASE_URL)
python my_script.py

# 任何 HTTP 客户端
curl http://localhost:9999/v1/chat/completions ...
```

### 🥉 Python API — 代码集成

```python
from optimizer import optimize

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
]

optimized = optimize(messages, provider="deepseek")
```

---

## 支持的 Provider 与缓存机制

| Provider | 缓存机制 | 阈值 | API 格式 | 状态 |
|----------|---------|------|---------|------|
| **DeepSeek** | 自动前缀匹配，128t 每块 | 128t | Anthropic | ✅ 实测验证 |
| **Anthropic** | `cache_control` breakpoint | 1024t | Anthropic | ⚠️ 代码已实现 |
| **OpenAI** | 自动前缀匹配 | 1025t | OpenAI | ⚠️ 代码已实现 |
| **Groq** | N/A | 128t | OpenAI | 🟢 预设就绪 |
| **Together AI** | N/A | 128t | OpenAI | 🟢 预设就绪 |
| **Mistral** | Prompt Caching (beta) | 128t | OpenAI | 🟢 预设就绪 |
| **Fireworks AI** | Prompt Caching | 128t | OpenAI | 🟢 预设就绪 |
| **xAI (Grok)** | N/A | 128t | OpenAI | 🟢 预设就绪 |
| **Perplexity (Sonar)** | N/A | 128t | OpenAI | 🟢 预设就绪 |
| **GitHub Copilot / Codex** | N/A | 128t | OpenAI | 🟢 预设就绪 |
| **OpenRouter** | 透传（取决于上游） | 128t | OpenAI | 🟢 预设就绪 |
| **Azure OpenAI** | 同 OpenAI | 1025t | OpenAI | 🟢 预设就绪 |
| **Google Gemini** | Context Caching | 128t | OpenAI | 🟢 预设就绪 |
| **自定义** | `register_provider()` | 自定义 | 任意 | ✅ 5 行代码 |

## 跨框架接入

popt 是 HTTP 代理，任何能配置 API 地址的框架都能接入。

| 框架 | 接入方式 | 需要改代码？ |
|------|---------|:----------:|
| **Claude Code** | 设置 `ANTHROPIC_BASE_URL` → `popt run -- claude` | ❌ 0 行 |
| **OpenAI Python SDK** | 设置 `OPENAI_BASE_URL` → `popt run -- python script.py` | ❌ 0 行 |
| **Anthropic Python SDK** | 设置 `ANTHROPIC_BASE_URL` → `popt run -- python script.py` | ❌ 0 行 |
| **LangChain / LlamaIndex** | 设置环境变量 → `popt run` | ❌ 0 行 |
| **Node.js / curl / 任意 HTTP** | 改 `base_url` 指向代理 | 1 行 |

### 环境变量自动检测

```bash
# 以下变量任意一个被设置，popt 会自动识别并配置优化参数
export ANTHROPIC_BASE_URL="https://api.deepseek.com/anthropic"
export OPENAI_BASE_URL="https://my-openai-proxy.example.com/v1"
export POPT_UPSTREAM="https://custom.api.com"
export POPT_PROVIDER="deepseek"
export POPT_MODEL="deepseek-v4-flash"   # 通过模型名自动配 upstream + provider
```

### `.poptimerc` 项目配置文件

在项目根目录放一个 `.poptimerc`（TOML 格式），替代重复输入参数：

```toml
[project]
model = "deepseek-v4-flash"
# provider 和 upstream 可选 — 会自动从 model 推断

[proxy]
port = 9999
host = "127.0.0.1"
```

搜索路径（高优先级覆盖低）：
1. `$POPT_CONFIG` 环境变量指定路径
2. `./.poptimerc`（当前目录）
3. 逐级向上（直到驱动器根）
4. `~/.poptimerc`（用户目录）

**完整检测链**：
```
CLI 参数 > 环境变量 > 项目 .poptimerc > 上级目录 > ~/.poptimerc > 注册表推断 > 内置默认值
```

---

## CLI 命令

| 命令 | 作用 |
|------|------|
| `popt run -- <cmd>` | 启动代理 + 运行命令（自动检测上游，推荐） |
| `popt run --model NAME -- <cmd>` | 指定模型，自动配 upstream + provider |
| `popt proxy [--port PORT] [--model NAME]` | 启动长驻代理 |
| `popt preview <file> [-p <provider>]` | 预览优化效果 |
| `popt diagnose <a.json> <b.json> [-p <provider>]` | 对比两个请求，定位缓存 miss 根因 |
| `popt stats [<file>]` | 分析日志中的缓存命中率 |

### popt preview

```bash
popt preview request.json -p deepseek --show
```

输出优化前后的消息数、token 估计、缓存阈值对比。

### popt diagnose

```bash
popt diagnose request_a.json request_b.json -p deepseek
```

逐块对比两个请求，显示首个差异位置、严重程度和修复建议。

---

## 安装

```bash
pip install poptimize
```

可选依赖（提升 token 估算精度）：

```bash
pip install tiktoken
```

### 从源码安装

```bash
git clone https://github.com/vansye/prompt-cache-optimizer
cd prompt-cache-optimizer
pip install -e .
```

---

## 完整工作示例

以 Claude Code + DeepSeek 为例：

```bash
# 方式 A: 用模型名（推荐）
popt run --model deepseek-v4-flash -- claude

# 方式 B: 或用环境变量
$env:ANTHROPIC_BASE_URL = 'https://api.deepseek.com/anthropic'
popt run -- claude

# 输出:
#   popt run -- proxy on :57128
#   =============================================
#   [OK] Upstream: https://api.deepseek.com/anthropic
#   [OK] Provider: deepseek (128t cache blocks, anthropic API)
#   -> ANTHROPIC_BASE_URL = http://127.0.0.1:57128
#   Model: deepseek-v4-flash
#   Command: claude
#   =============================================
#
# Claude Code 启动后，所有 API 请求自动经过优化代理
```

换成 OpenAI：

```bash
popt run --model gpt-4o -- python my_script.py
# 自动：upstream=https://api.openai.com, provider=openai, OPENAI_BASE_URL 指向代理
```

换成 Groq：

```bash
popt run --model llama-3-70b -- python my_agent.py
# 自动：upstream=https://api.groq.com/openai/v1, provider=groq
```

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

每个阶段都有 SafetyCheck 保底——如果优化改变了语义内容，回退到原始输入。

---

## 实验验证

在 DeepSeek v4 flash 上的实际测试数据：

| 场景 | 原始 | 优化后 | 说明 |
|------|------|--------|------|
| **短 prompt (~14t)** | 0.0% | **60.4%** | 填充到 128t 阈值，缓存 1 块 |
| **中 prompt (~203t)** | 63.1% | 63.4% | 已达阈值，不做多余操作 |
| **长 prompt (~2745t)** | 97.9% | 98.0% | 正常缓存，跳过 padding |

---

## License

MIT
