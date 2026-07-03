# popt 操作手册

popt 是一个 **HTTP 透明代理**，插在 AI 框架和模型 API 之间，自动优化请求消息的结构来最大化 Prompt Caching 命中率。

**不需要改代码**。任何 AI 框架——Claude Code、OpenAI SDK、Anthropic SDK、Codex、LangChain、LlamaIndex、Hermes……只要能配 API 地址或环境变量，就能接 popt。

---

## 目录

1. [安装](#1-安装)
2. [原理：popt 做了什么](#2-原理popt-做了什么)
3. [三层使用方式](#3-三层使用方式)
4. [`popt run` 详细操作](#4-popt-run-详细操作)
5. [`popt proxy` 详细操作](#5-popt-proxy-详细操作)
6. [模型切换](#6-模型切换)
7. [配置文件 .poptimerc](#7-配置文件-poptimerc)
8. [Python API](#8-python-api)
9. [实战场景](#9-实战场景)
10. [验证优化是否生效](#10-验证优化是否生效)
11. [命令参考](#11-命令参考)
12. [常见问题（FAQ）](#12-常见问题faq)

---

## 1. 安装

### 1.1 从 PyPI 安装

```bash
# 全局安装（推荐用 uv）
uv tool install poptimize

# 或用 pip
pip install poptimize
```

### 1.2 从源码安装（开发）

```bash
git clone https://github.com/vansye/prompt-cache-optimizer
cd prompt-cache-optimizer
pip install -e .
```

### 1.3 验证安装

```bash
popt run --help
```

输出应包含 `--model`、`--upstream`、`--provider` 等选项：

```
usage: popt run [-h] [--upstream UPSTREAM] [--provider PROVIDER] [--model MODEL] ...

positional arguments:
  cmd_args              Command to run. Prefix with -- to separate from popt args.

options:
  --model, -m MODEL     Model name (e.g. deepseek-v4-flash, gpt-4o).
                        Auto-configures upstream and provider from the built-in registry.
```

> **Windows 注意**：如果 `popt` 命令找不到，有以下几种方案：
> - 用 `uv tool list` 查看 `poptimize` 是否已安装
> - 把 `%USERPROFILE%\.local\bin` 或 Python `Scripts` 目录加到 PATH
> - 直接 `python -m cli.main` 代替 `popt`

---

## 2. 原理：popt 做了什么

### 2.1 整体流程

```
               popt 代理（:57128）
┌─────────────────────────────────────────────────┐
│  请求进来 → Normalizer → Reorderer → Aligner    │
│             → Formatter → 转发到上游 API        │
└─────────────────────────────────────────────────┘
     ▲                                     │
     │                                     ▼
你的 AI 工具                       真正的 API 服务商
(Claude / OpenAI SDK / Hermes)    (DeepSeek / OpenAI / Groq / …)
```

### 2.2 每个模块的作用

| 模块 | 做什么 | 对缓存的贡献 |
|------|--------|-------------|
| **Normalizer** | 统一换行符 `\r\n`→`\n`、JSON 键排序、清除 BOM | **消除格式差异**导致的意外 miss |
| **Reorderer** | system 消息永远排最前、同角色按确定顺序排列 | **保证前缀稳定**，不同请求间不变 |
| **Aligner** | 短 prompt 填充 Padding 到缓存阈值，加 `cache_control` 标记 | **唯一能创造缓存价值**的模块 |
| **Formatter** | 剥离内部标记、输出 provider 所需格式 | **适配不同 API**（OpenAI / Anthropic） |

每个阶段都有 **SafetyCheck** 保底——如果优化改变了语义内容，回退到原始输入。

### 2.3 什么是 Prompt Caching

LLM API 会缓存请求的前缀。如果下一个请求的前缀和上一个相同，就直接复用计算结果，**不重新算**。这就是缓存命中。

popt 的作用就是保证相同的前缀在多次请求间**完全一致**：

- 相同的 system prompt → 排序一致
- 相同的角色顺序 → 顺序一致  
- 短 prompt 填充到缓存阈值 → 前缀块稳定

---

## 3. 三层使用方式

| 方式 | 一句话 | 适用场景 |
|------|--------|---------|
| **`popt run`** | 启动代理 → 运行命令 → 命令结束自动关闭 | 一次性使用（推荐新手） |
| **`popt proxy`** | 后台长驻，多个工具共享一个代理 | 长期运行的服务 |
| **Python API** | 代码里直接调 `optimize()` | 自动化流程、自定义管道 |

---

## 4. `popt run` 详细操作

### 4.1 最简单的用法（自动检测环境变量）

如果你已经设了 `ANTHROPIC_BASE_URL` 或 `OPENAI_BASE_URL`：

```bash
$env:ANTHROPIC_BASE_URL = 'https://api.deepseek.com/anthropic'
popt run -- claude
```

popt 自动检测到 `ANTHROPIC_BASE_URL`，用它作为上游，同时推断出 provider=deepseek。

实际运行输出：

```
  popt run -- proxy on :57128
  =============================================
  [OK] Upstream: https://api.deepseek.com/anthropic
  [OK] Provider: deepseek (128t cache blocks, anthropic API)
  -> ANTHROPIC_BASE_URL = http://127.0.0.1:57128
  Command: claude
  =============================================
```

然后 Claude Code 启动，所有 API 请求自动经过优化代理。Claude Code 退出后代理自动关闭。

### 4.2 `--model` 参数（推荐）

**一条命令配好所有东西：**

```bash
# DeepSeek → 自动配 upstream + provider + api 格式
popt run --model deepseek-v4-flash -- claude

# OpenAI → 自动切
popt run --model gpt-4o -- python my_script.py

# Groq 上的 deepseek-r1
popt run --model deepseek-r1-671b -- python script.py

# xAI Grok
popt run --model grok-beta -- node bot.js

# GitHub Copilot Codex
popt run --model codex-gpt-4 -- node bot.js
```

用 `--model` 时，输出类似：

```
  popt run -- proxy on :58321
  =============================================
  [OK] Upstream: https://api.deepseek.com/anthropic
  [OK] Provider: deepseek (128t cache blocks, anthropic API)
  -> ANTHROPIC_BASE_URL = http://127.0.0.1:58321
  Model: deepseek-v4-flash
  Command: claude
  =============================================
```

`--model` 的内置注册表目前已涵盖 15+ 服务商。模型名匹配支持 glob 通配符（`deepseek-*`、`gpt-*`、`grok-*` 等）。匹配规则：

1. **精确匹配优先**（`gpt-4o` 精确命中）
2. **通配取最长 pattern**（`deepseek-r1-*` 比 `deepseek-*` 更精确，优先匹配）

### 4.3 手动指定（完全控制）

```bash
popt run --upstream https://api.deepseek.com --provider openai -- python bot.py
popt run -u https://custom.api.com --provider my-provider -- python test.py
```

### 4.4 搭配前置 `--` 分隔符

`--` 后的所有内容都视为子进程的命令，不会被 popt 解析：

```bash
popt run --model gpt-4o -- python -u my_script.py --verbose --flag value
```

### 4.5 检测优先级

```
upstream:  --upstream > --model 注册表 > $POPT_UPSTREAM > $ANTHROPIC_BASE_URL
           > $OPENAI_BASE_URL > .poptimerc > 空（代理模式不转发）

provider:  --provider > --model 注册表 > $POPT_PROVIDER > 从 URL 推断
           > .poptimerc > 默认 openai

api_format: model_info.api_format > provider_config.api_format > "openai"
```

---

## 5. `popt proxy` 详细操作

### 5.1 启动代理

```bash
# 仅启动（从环境变量自动检测上游）
popt proxy

# 指定端口
popt proxy --port 8888

# 指定模型
popt proxy --port 8888 --model deepseek-v4-flash

# 完全手动
popt proxy --port 9999 --upstream https://api.deepseek.com --provider openai
```

### 5.2 配置 AI 工具指向代理

代理启动后，在另一个终端设置对应的环境变量：

**如果 popt 用的是 Anthropic 格式（deepseek、anthropic）：**

```powershell
$env:ANTHROPIC_BASE_URL = 'http://127.0.0.1:9999'
```

**如果 popt 用的是 OpenAI 格式（openai、groq、xai 等）：**

```powershell
$env:OPENAI_BASE_URL = 'http://127.0.0.1:9999/v1'
```

然后正常使用你的工具即可：

```bash
claude                    # Claude Code
python my_script.py       # 任何用 OpenAI SDK 的脚本
```

### 5.3 停止代理

按 `Ctrl+C`。

### 5.4 popt proxy 完整输出示例

```
  popt proxy running on http://127.0.0.1:18002
  =============================================
  [OK] Upstream: https://api.deepseek.com
  [OK] Provider: openai (1025t cache blocks, openai API)

  Point your AI client to this proxy:
    $env:OPENAI_BASE_URL    = 'http://127.0.0.1:18002/v1'

  Then run your AI tool normally:
    claude
    python your_script.py
  =============================================
  Press Ctrl+C to stop
```

---

## 6. 模型切换

### 6.1 一条命令换模型

```bash
# DeepSeek
popt run --model deepseek-v4-flash -- python script.py
# → 自动配 upstream=api.deepseek.com/anthropic, provider=deepseek, 格式=anthropic

# OpenAI
popt run --model gpt-4o -- python script.py
# → 自动配 upstream=api.openai.com, provider=openai, 格式=openai

# Groq
popt run --model deepseek-r1-671b -- python script.py
# → 自动配 upstream=api.groq.com/openai/v1, provider=groq, 格式=openai

# xAI
popt run --model grok-beta -- python script.py
# → 自动配 upstream=api.x.ai, provider=xai, 格式=openai
```

### 6.2 环境变量指定模型

```bash
$env:POPT_MODEL = 'deepseek-v4-flash'
popt run -- claude
# → 等效于 popt run --model deepseek-v4-flash -- claude
```

### 6.3 切换场景示例

假设你原来用 DeepSeek 跑 Claude Code：

```bash
popt run --model deepseek-v4-flash -- claude
```

想换到 GPT-4o：

```bash
popt run --model gpt-4o -- claude
```

三处改变自动完成：

| 配置项 | DeepSeek | → GPT-4o |
|--------|----------|----------|
| upstream | `api.deepseek.com/anthropic` | `api.openai.com` |
| provider | `deepseek` | `openai` |
| 设给子进程的 env var | `ANTHROPIC_BASE_URL` | `OPENAI_BASE_URL` |

---

## 7. 配置文件 .poptimerc

### 7.1 为什么需要它

每次运行都输入 `--upstream ... --provider ...` 很啰嗦。放一个 `.poptimerc` 在项目根目录，popt 自动加载。

### 7.2 格式（TOML）

```toml
[project]
model = "deepseek-v4-flash"
# provider 和 upstream 可选 — 从 model 自动推断

[proxy]
port = 9999
host = "127.0.0.1"
```

`[project]` 下的字段：

| 字段 | 作用 | 可选？ |
|------|------|--------|
| `model` | 模型名（从注册表自动配） | 可选，但推荐 |
| `provider` | 优化策略名 | 可选 |
| `upstream` | API 上游地址 | 可选 |

`[proxy]` 下的字段：

| 字段 | 默认值 | 作用 |
|------|--------|------|
| `host` | `"127.0.0.1"` | 代理监听地址 |
| `port` | `9999` | 代理监听端口 |

### 7.3 搜索路径（高优先级覆盖低）

1. **`$POPT_CONFIG`** 环境变量指定的文件路径
2. **`./.poptimerc`**（当前目录）
3. 逐级向上找（直到驱动器根）
4. **`~/.poptimerc`**（用户 home 目录）

多个文件合并，高优先级覆盖低优先级同名字段。

### 7.4 创建步骤

```bash
# 进入项目目录
cd /path/to/my-project

# 创建 .poptimerc
cat > .poptimerc << 'EOF'
[project]
model = "deepseek-v4-flash"

[proxy]
port = 9999
EOF

# 验证加载
cd /path/to/my-project && popt proxy
```

### 7.5 完整示例

```toml
# .poptimerc — popt 项目配置
[project]
model = "deepseek-v4-flash"
# provider 和 upstream 不写 → 自动从 model 推断

[proxy]
port = 18002
host = "127.0.0.1"

# popt 会忽略未知 section，可以放自己的注释
[my_notes]
description = "开发环境配置"
```

---

## 8. Python API

### 8.1 基本用法

```python
from optimizer import optimize

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
]

# 优化消息结构
result = optimize(messages, provider="deepseek")

# result 中的消息已被重排、对齐到缓存阈值
print(len(result))  # >= 2
```

### 8.2 优化预览（不实际调 API）

```python
from optimizer import preview

messages = [{"role": "user", "content": "Hello!"}]
report = preview(messages, provider="deepseek")

print(report["message_count"]["before"])    # 1
print(report["message_count"]["after"])     # >= 1
print(report["estimated_tokens"]["before"]) # ~3
print(report["estimated_tokens"]["after"])  # ~36（填充到 128t 阈值）
print(report["cache_threshold"])            # 128
print(report["meets_threshold"])            # True（已填充到阈值）
```

### 8.3 注册自定义 provider

```python
from optimizer.config import register_provider, get_config

# 注册一个自定义服务商的配置
register_provider(
    "my-provider",
    cache_threshold=128,
    api_format="openai",
    supports_breakpoints=False,
    default_model="my-model",
)

# 现在 optimize 可以用它
from optimizer import optimize
result = optimize(messages, provider="my-provider")

# 查询配置
cfg = get_config("my-provider")
print(cfg.cache_threshold)  # 128
```

---

## 9. 实战场景

### 场景 1：快速验证网络连通性

```bash
popt run --model deepseek-v4-flash -- echo "connectivity test"
```

输出：

```
  popt run -- proxy on :52431
  =============================================
  [OK] Upstream: https://api.deepseek.com/anthropic
  [OK] Provider: deepseek (128t cache blocks, anthropic API)
  -> ANTHROPIC_BASE_URL = http://127.0.0.1:52431
  Model: deepseek-v4-flash
  Command: echo connectivity test
  =============================================

connectivity test
```

如果能看到这个输出，说明 popt 安装正确、模型注册表工作正常。

### 场景 2：Claude Code + DeepSeek

**方法 A（推荐）：指定模型名**

```bash
popt run --model deepseek-v4-flash -- claude
```

**方法 B：设环境变量**

```bash
# Windows PowerShell
$env:ANTHROPIC_BASE_URL = 'https://api.deepseek.com/anthropic'
popt run -- claude

# 或 Linux/macOS
export ANTHROPIC_BASE_URL='https://api.deepseek.com/anthropic'
popt run -- claude
```

Claude Code 启动后，所有 API 请求走 popt 代理，自动优化。

### 场景 3：OpenAI SDK 脚本

```python
# my_script.py — 不需要改任何代码
from openai import OpenAI
client = OpenAI(api_key="sk-...")
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.choices[0].message.content)
```

运行时，popt 自动设置 `OPENAI_BASE_URL` 指向代理：

```bash
popt run --model gpt-4o -- python my_script.py
```

### 场景 4：Hermes（长期运行框架）

**[Hermes](https://github.com/sshterm/hermes)** 是一个 AI 框架。这是完整的接入步骤。

**步骤 1：在 Hermes 项目根目录放 `.poptimerc`**

```toml
[project]
upstream = "https://api.deepseek.com"
provider = "openai"

[proxy]
port = 18002
host = "127.0.0.1"
```

> 注：Hermes 用的是 **DeepSeek 的 OpenAI 兼容接口**（`/v1/chat/completions`），所以 `provider = "openai"` 才能正确匹配请求格式。

**步骤 2：确认 Hermes 的配置**

检查 `S:\Coding\.hermes\config.yaml`，确保含有：

```yaml
providers:
  deepseek:
    base_url: http://127.0.0.1:18002  # 指向 popt 代理
```

如果原来指向其他地址，改成这个。

**步骤 3：启动 popt 代理**

```bash
cd S:/Coding/.hermes
popt proxy
```

输出：

```
  popt proxy running on http://127.0.0.1:18002
  =============================================
  [OK] Upstream: https://api.deepseek.com
  [OK] Provider: openai (1025t cache blocks, openai API)

  Point your AI client to this proxy:
    $env:OPENAI_BASE_URL    = 'http://127.0.0.1:18002/v1'

  Then run your AI tool normally:
    claude
    python your_script.py
  =============================================
  Press Ctrl+C to stop
```

**步骤 4：启动 Hermes**

在另一个终端照常启动 Hermes。所有请求自动经过 popt 代理。

**步骤 5：验证代理在工作**

观察 popt 代理终端的日志输出：

```
INFO | OPTIMIZE | openai | 2->2 msgs | ~150->~280 tokens
INFO | FORWARD  | openai | 200 | 1523 bytes | 2.34s
```

说明请求已经过了优化和转发。

### 场景 5：预览优化效果（不调 API）

```bash
# 准备一个请求 JSON
echo '[{"role":"system","content":"You are helpful."},{"role":"user","content":"Hello"}]' > request.json

# 预览优化
popt preview request.json -p deepseek --show
```

输出：

```
  popt preview - deepseek
  ========================================
  Messages:    2 -> 2
  Role order:  system -> user
               system -> user
  Est. tokens: ~9 -> ~38
  Cache threshold: 128 tokens
  Meets threshold: YES
  ----------------------------------------
  Prefix shape hashes:
    system:           5a8f1b2c
    role_sequence:    9d4e7f1a
    content_prefix:   3b6c9e2d
    full_prefix:      1a2b3c4d

  Optimized messages:
  [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello"}
  ]
```

token 从 ~9 填充到 ~38（接近 128t 阈值的第一块）。

### 场景 6：诊断缓存为什么没命中

当你发现两次请求缓存没命中时：

```bash
# 保存两次请求
popt diagnose request_a.json request_b.json -p deepseek
```

输出：

```
  popt diagnose - deepseek
  ==================================================
  Request A: request_a.json (2 messages)
  Request B: request_b.json (2 messages)
  Cache block: 128 tokens  |  Threshold: 128

  -- Shape comparison --
    = system             5a8f1b2c  5a8f1b2c
    X role sequence      9d4e7f1a  a1b2c3d4
    X content prefix     3b6c9e2d  7f8a9b0c
    Changed: role_sequence, content_prefix

  -- First difference --
    Position:   ~token 52
    Block:      0  !! MAJOR
    Message:    [1] role=user
    Field:      content
    Snippet A:  "Tell me about Python"
    Snippet B:  "Tell me about JavaScript"

    Suggestion: These two requests diverge early in the user message.
                The cache prefix breaks at token ~52 (block 0).
```

### 场景 7：分析日志统计缓存命中率

```bash
# 如果记录了日志
popt stats proxy.log
```

输出：

```
  popt stats -- 156 requests
  ========================================
  Total input tokens:   285,432
  Total cache created:  142,716
  Total cache read:     89,544
  Total output tokens:  31,204
  ────────────────────────────────────────
  Hit ratio:            62.7%
  Savings ratio:        31.4%
  Est. cost saved:      $0.0842
```

### 场景 8：代码里集成 optimize

```python
from optimizer import optimize
from optimizer.config import get_config

# 你的消息
messages = [
    {"role": "system", "content": "You are a coding assistant."},
    {"role": "user", "content": "Write a Python function to sort a list."},
]

# 优化
optimized = optimize(messages, provider="deepseek")

# 查看优化后信息
cfg = get_config("deepseek")
print(f"Threshold: {cfg.cache_threshold}t")
print(f"Messages: {len(messages)} → {len(optimized)}")

# 传到 API
import os
from openai import OpenAI
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=optimized,
)
```

### 场景 9：代理日志解读

popt 在运行时会输出三类日志行：

```
# 请求经过优化
INFO | OPTIMIZE | deepseek | 3->3 msgs | ~45->~78 tokens
#            provider   消息数变化    token 估算变化

# 正常转发（非流式）
INFO | FORWARD  | deepseek | 200 | 523 bytes | 1.23s
#            provider   HTTP 状态   响应大小   耗时

# 流式响应
INFO | STREAM   | deepseek | 127.0.0.1 | 15.2s
#            provider   客户端 IP    总耗时

# 上游错误
WARNING | UPSTREAM ERROR | openai | 401 | 0.45s
```

---

## 10. 验证优化是否生效

### 10.1 看代理日志

popt 启动后，每次请求都会打印优化日志。看到 `OPTIMIZE` 行就说明请求被优化了。

### 10.2 检查代理端口是否在监听

```bash
netstat -ano | findstr :18002
```

输出类似：

```
  TCP    127.0.0.1:18002    0.0.0.0:0    LISTENING    12345
```

如果在 LISTENING，说明代理在运行。

如果端口被其他程序占用：

```bash
# 查看谁占了端口
netstat -ano | findstr :18002
# 最后一列是 PID，去任务管理器查

# 或换一个端口
popt proxy --port 18003
```

### 10.3 用 stats 命令分析命中率

```bash
# 如果启用了日志记录
popt stats proxy.log
```

### 10.4 手动发送请求测试

```bash
# 向代理发一个请求，看是否正常转发
$body = @{
    model = "deepseek-chat"
    messages = @(
        @{role = "user"; content = "ping"}
    )
} | ConvertTo-Json

curl -X POST http://127.0.0.1:18002/v1/chat/completions `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer $env:DEEPSEEK_API_KEY" `
  -d $body
```

如果返回正常响应，说明代理运行正常。

---

## 11. 命令参考

### 11.1 命令总表

| 命令 | 作用 | 示例 |
|------|------|------|
| `popt run` | 启动代理 → 跑命令 → 自动关闭 | `popt run --model gpt-4o -- python test.py` |
| `popt proxy` | 启动长驻代理 | `popt proxy --port 9999` |
| `popt preview` | 预览优化效果（不调 API） | `popt preview request.json -p deepseek` |
| `popt diagnose` | 诊断两个请求的缓存差异 | `popt diagnose a.json b.json -p deepseek` |
| `popt stats` | 分析日志中的缓存命中率 | `popt stats proxy.log` |

### 11.2 run 参数

```
popt run [-h] [-u UPSTREAM] [--provider PROVIDER] [-m MODEL] [cmd_args ...]

--model, -m MODEL      模型名（推荐）。自动配 upstream + provider
--upstream, -u URL     上游 API 地址
--provider PROVIDER    优化策略（deepseek / anthropic / openai / ...）
cmd_args               要运行的命令（-- 之后的所有内容）
```

### 11.3 proxy 参数

```
popt proxy [-h] [--host HOST] [-p PORT] [-u UPSTREAM] [--provider PROVIDER] [-m MODEL]

--host HOST            绑定地址（默认 127.0.0.1）
--port, -p PORT        绑定端口（默认 9999）
--model, -m MODEL      模型名
--upstream, -u URL     上游 API 地址
--provider PROVIDER    优化策略
```

### 11.4 preview 参数

```
popt preview [-h] [--provider PROVIDER] [--verbose] [--show] file

file                   请求 JSON 文件
--provider, -p         服务商（默认 deepseek）
--verbose, -v          详细输出
--show, -s             显示优化后的消息体
```

### 11.5 diagnose 参数

```
popt diagnose [-h] [--provider PROVIDER] [--verbose] file_a file_b

file_a                 请求 A（建立缓存的请求）
file_b                 请求 B（没命中的请求）
--provider, -p         服务商（默认 deepseek）
--verbose, -v          详细差异信息
```

### 11.6 stats 参数

```
popt stats [-h] [file]

file                   JSON Lines 日志文件（可选，默认读 stdin）
```

---

## 12. 常见问题（FAQ）

### Q：popt 和框架的关系是什么？

**零侵入**。popt 是一个 HTTP 代理，不需要改框架的任何代码。框架照常用自己的 SDK，popt 在中间拦截 HTTP 请求，优化完后转发到真正的 API。

```
框架 SDK → HTTP 请求 → popt 代理（优化）→ 真实 API
                    ← 响应 ←
```

### Q：支持哪些框架？

| 框架 | 接入方式 |
|------|---------|
| **Claude Code** | `popt run --model xxx -- claude`（自动设 `ANTHROPIC_BASE_URL`） |
| **OpenAI Python SDK** | `popt run --model xxx -- python script.py`（自动设 `OPENAI_BASE_URL`） |
| **Anthropic Python SDK** | `popt run --model xxx -- python script.py`（自动设 `ANTHROPIC_BASE_URL`） |
| **LangChain / LlamaIndex** | 设环境变量 → `popt run` 自动接管 |
| **Hermes** | 改 `base_url` 指向 popt 端口 |
| **Node.js / curl / 任意 HTTP 客户端** | 改 `base_url` 指向代理地址 |

### Q：如何添加一个新的服务商？

**方法 A：已有 `providers.json` 条目**

如果该服务商已内置（15+），直接 `--model xxx`。

**方法 B：运行时注册**

```python
from optimizer.config import register_provider
register_provider("my-new-provider", cache_threshold=128, api_format="openai")
```

**方法 C：修改 providers.json**

编辑 `optimizer/providers.json`，按格式添加条目：

```json
{
  "name": "my-provider",
  "api_format": "openai",
  "base_url": "https://api.myprovider.com/v1",
  "model_patterns": ["my-model-*", "my-other-*"],
  "config": {
    "cache_threshold": 128,
    "default_model": "my-model-default"
  }
}
```

### Q：支持哪些模型？

内置 15+ 服务商的模型匹配：

| 服务商 | 模型名示例 | 自动配好的配置 |
|--------|-----------|--------------|
| DeepSeek | `deepseek-v4-flash`、`deepseek-chat` | anthropic 格式、128t 块 |
| OpenAI | `gpt-4o`、`gpt-4-turbo`、`o1-preview` | openai 格式、1025t 块 |
| Anthropic | `claude-sonnet-5`、`claude-opus-4-8` | anthropic 格式、1024t 块、cache_control |
| Groq | `llama-3.3-70b`、`deepseek-r1-671b`、`mixtral-8x7b` | openai 格式、128t 块 |
| Together AI | `together-*` | openai 格式、128t 块 |
| Mistral | `mistral-large`、`mistral-7b` | openai 格式、128t 块 |
| xAI | `grok-beta`、`grok-2-latest` | openai 格式、128t 块 |
| Perplexity | `sonar-pro`、`pplx-*` | openai 格式、128t 块 |
| GitHub Copilot | `codex-*`、`copilot-*` | openai 格式、128t 块 |
| OpenRouter | `openrouter/*` | openai 格式、128t 块 |
| Google Gemini | `gemini-*` | openai 格式、128t 块 |

不支持的模型名 → 自动回退到 openai 默认配置。

### Q：换了模型后需要改什么？

**只需要改 `--model` 参数**：

```bash
# DeepSeek → OpenAI
popt run --model gpt-4o -- claude

# OpenAI → xAI Grok
popt run --model grok-beta -- python script.py

# GPU → Groq 上的 Llama
popt run --model llama-3.3-70b-versatile -- python script.py
```

不用手动查 URL、不用改 provider 名、不用改代码。

### Q：popt 和直接调 API 相比，会多消耗 token 吗？

会少量增加（Padding 填充）。但对短 prompt 来说，增加的 token 远小于缓存命中的节省：

| prompt 长度 | Padding 增加 | 缓存命中后节省 | 净效果 |
|-------------|-------------|---------------|-------|
| 14t → 128t | ~114t | 首次 0，后续复用 ~60% | **正收益** |
| 203t | 0 | 63% | **正收益** |
| 2745t | 0 | 98% | **正收益** |

### Q：报错 "command not found: popt"？

```bash
# 检查是否安装了
uv tool list | grep poptimize
# 或
pip show poptimize

# 检查版本
popt run --help
# 如果版本 < 0.3.0，没有 --model 功能

# 升级
uv tool upgrade poptimize
# 或
pip install --upgrade poptimize

# 如果还是找不到，Windows 上试试
python -m cli.main run --help
```

### Q：Windows 上 Python 子进程找不到命令？

popt 自动使用 `shell=True`，支持 `.cmd`/`.bat` 文件：

```bash
popt run -- claude  # claude.cmd 会被正确解析
```

如果仍有问题，用绝对路径：

```bash
popt run -- "%USERPROFILE%\.local\bin\claude.cmd"
```

### Q：报错 "Address already in use" / 端口被占用？

```bash
# 查看占用端口的进程
netstat -ano | findstr :9999
# 最后一列是 PID → 去任务管理器结束

# 或直接换端口
popt proxy --port 12345
# popt run 自动选空闲端口，不需要手动换
```

### Q：Windows 上 Unicode 显示乱码？

终端编码问题：

```powershell
$env:PYTHONIOENCODING = 'utf-8'
chcp 65001
```

### Q：代理日志没输出？

可能是日志级别设置。popt 默认 INFO 级别以上才输出。如果没看到 `OPTIMIZE` / `FORWARD` 行：

1. 确认代理确实收到了请求——检查代理端口是否在 LISTENING
2. 确认客户端正确配置了 `BASE_URL` 指向代理
3. 确认 API key 正确，请求能到达上游

### Q：怎么在 Docker 里用 popt？

```dockerfile
FROM python:3.11-slim
RUN pip install poptimize

# 启动代理 + 运行你的程序
CMD popt run --model deepseek-v4-flash -- python /app/my_agent.py
```

### Q：怎么关掉/跳过优化？

如果某个请求不需要优化，直接发到上游 API（不经过代理）就行。或者：

```python
from optimizer import optimize
# 只要不调 optimize，原始消息不会被修改
```
