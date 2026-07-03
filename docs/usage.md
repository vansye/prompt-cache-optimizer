# popt 操作手册

popt 是一个 HTTP 透明代理，插在 AI 框架和模型 API 之间，自动优化消息结构，最大化 Prompt Caching 命中率。

**不需要改代码**。任何 AI 框架——Claude Code、OpenAI SDK、Codex、LangChain、Hermes……只要能配 API 地址，就能接 popt。

---

## 目录

1. [安装](#1-安装)
2. [核心概念](#2-核心概念)
3. [`popt run` —— 一条命令通吃所有框架](#3-popt-run--一条命令通吃所有框架)
4. [`popt proxy` —— 长驻代理模式](#4-popt-proxy--长驻代理模式)
5. [模型切换（--model）](#5-模型切换--model)
6. [配置文件 .poptimerc](#6-配置文件-poptimerc)
7. [实际场景](#7-实际场景)
8. [验证是否生效](#8-验证是否生效)
9. [命令参考](#9-命令参考)
10. [常见问题](#10-常见问题)

---

## 1. 安装

```bash
# pip 安装
pip install poptimize

# 或 uv（推荐，更快）
uv tool install poptimize
```

验证安装：

```bash
popt run --help
# 输出应包含 --model 选项
```

> **注意**：Windows 上如果 `popt` 命令找不到，检查 `Scripts` 目录是否在 PATH 中，或直接用 `python -m cli.main`。

---

## 2. 核心概念

popt 只有一张图：

```
你的 AI 工具                popt 代理                 模型 API
(Claude / Hermes      →    localhost:端口      →     DeepSeek / OpenAI
 / Codex / 任何框架)        优化消息结构                / Groq / ...
```

**3 层接入方式，从易到难：**

| 方式 | 特点 | 适用场景 |
|------|------|---------|
| `popt run -- <cmd>` | 启动代理 → 运行命令 → 自动关闭 | 一条命令跑完就结束（推荐） |
| `popt proxy` | 后台长驻，多个工具共用 | Hermes、持续服务 |
| Python API | `from optimizer import optimize` | 代码集成、自动化流程 |

---

## 3. `popt run` —— 一条命令通吃所有框架

### 3.1 自动检测（最简单）

如果已经设好了 `ANTHROPIC_BASE_URL` 或 `OPENAI_BASE_URL` 环境变量：

```bash
# 直接跑，自动检测上游
popt run -- claude
popt run -- python my_agent.py
popt run -- node bot.js
```

### 3.2 指定模型（推荐）

```bash
popt run --model deepseek-v4-flash -- claude
popt run --model gpt-4o -- python my_script.py
popt run --model grok-beta -- python script.py
popt run --model deepseek-r1-671b -- python test.py
```

`--model` 自动完成所有配置：

| 模型名 | 自动配好 upstream | provider | 格式 |
|--------|-------------------|----------|------|
| `deepseek-v4-flash` | `api.deepseek.com/anthropic` | deepseek | anthropic |
| `gpt-4o` | `api.openai.com` | openai | openai |
| `deepseek-r1-671b` | `api.groq.com/openai/v1` | groq | openai |
| `grok-beta` | `api.x.ai` | xai | openai |
| `codex-*` | `api.githubcopilot.com` | github-copilot | openai |

内置 15+ 服务商，全部通过模型名自动匹配。

### 3.3 手动指定（完全控制）

```bash
popt run --upstream https://api.deepseek.com --provider deepseek -- claude
```

### 3.4 检测优先级

```
upstream:  --upstream > --model > $POPT_UPSTREAM > $ANTHROPIC_BASE_URL > .poptimerc > 注册表
provider:  --provider > --model > $POPT_PROVIDER > 从 URL 推断 > .poptimerc > 默认
```

---

## 4. `popt proxy` —— 长驻代理模式

适合长期运行的服务（Hermes、持续运行的 agent）：

```bash
# 自动检测
popt proxy

# 指定端口和模型
popt proxy --port 8888 --model gpt-4o

# 手动指定上游
popt proxy --port 9999 --upstream https://api.deepseek.com --provider openai
```

然后在另一个终端正常用你的 AI 工具：

```bash
# 设置环境变量指向代理
$env:OPENAI_BASE_URL = 'http://127.0.0.1:9999/v1'
# 或 $env:ANTHROPIC_BASE_URL = 'http://127.0.0.1:9999'

# 正常运行
python my_script.py
```

按 `Ctrl+C` 停止代理。

---

## 5. 模型切换（--model）

`--model` 是 popt 最方便的功能。一条命令换模型：

```bash
# 换到 DeepSeek
popt run --model deepseek-v4-flash -- claude

# 换到 OpenAI
popt run --model gpt-4o -- python script.py

# 换到 Groq
popt run --model deepseek-r1-671b -- python script.py

# 换到 xAI Grok
popt run --model grok-beta -- python script.py

# 换到 GitHub Copilot / Codex
popt run --model codex-gpt-4 -- node bot.js
```

### 环境变量也可以指定模型

```bash
$env:POPT_MODEL = 'deepseek-v4-flash'
popt run -- claude    # 自动读 POPT_MODEL
```

---

## 6. 配置文件 .poptimerc

在每个项目根目录放 `.poptimerc`，避免重复输入参数。

### 格式（TOML）

```toml
[project]
model = "deepseek-v4-flash"
# provider 和 upstream 可选 — 从 model 自动推断

[proxy]
port = 9999
host = "127.0.0.1"
```

### 搜索路径（高优先级覆盖低）

1. `$POPT_CONFIG` 环境变量指定的路径
2. `./.poptimerc`（当前目录）
3. 逐级向上找 `.poptimerc`（直到驱动器根）
4. `~/.poptimerc`（用户目录）

### 典型示例

**搭配 Hermes（DeepSeek OpenAI 接口）：**

```toml
[project]
upstream = "https://api.deepseek.com"
provider = "openai"

[proxy]
port = 18002
host = "127.0.0.1"
```

然后：

```bash
cd /path/to/hermes
popt proxy
# 代理在 18002 端口，Hermes 零改动直接使用
```

---

## 7. 实际场景

### 场景 1：调试 / 测试某个模型

```bash
# 快速验证模型能不能通
popt run --model deepseek-v4-flash -- echo "connectivity test"

# 预览优化效果（不实际调 API）
popt preview request.json -p deepseek --show
```

### 场景 2：跑 Claude Code + DeepSeek

```bash
# 方法 A：指定模型名
popt run --model deepseek-v4-flash -- claude

# 方法 B：或设环境变量
$env:ANTHROPIC_BASE_URL = 'https://api.deepseek.com/anthropic'
popt run -- claude
```

### 场景 3：跑 Python OpenAI SDK 脚本

```bash
export OPENAI_API_KEY="sk-..."
popt run --model gpt-4o -- python my_script.py
```

脚本里不需要改任何代码——popt 自动设 `OPENAI_BASE_URL` 指向代理。

### 场景 4：长期运行 Hermes + DeepSeek

放 `.poptimerc` 在 Hermes 根目录：

```toml
[project]
upstream = "https://api.deepseek.com"
provider = "openai"

[proxy]
port = 18002
```

```bash
cd S:/Coding/.hermes
popt proxy
```

Hermes 已配好 `base_url: http://127.0.0.1:18002`，直接启动 Hermes 即可。

### 场景 5：代码里集成

```python
from optimizer import optimize

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
]

optimized = optimize(messages, provider="deepseek")
# → 消息已重排、对齐到缓存阈值
```

### 场景 6：诊断缓存为什么没命中

```bash
popt diagnose request_a.json request_b.json -p deepseek
# → 定位首个差异位置、严重程度、修复建议
```

---

## 8. 验证是否生效

### 看代理日志

`popt run` / `popt proxy` 运行时会打印优化日志：

```
INFO | OPTIMIZE | deepseek | 3->3 msgs | ~45->~78 tokens
INFO | FORWARD  | deepseek | 200 | 523 bytes | 1.23s
```

- `OPTIMIZE`：请求经过优化
- `FORWARD`：正常转发
- `STREAM`：流式响应

### 对比缓存命中率

```bash
popt stats log.jsonl
# → 显示命中率、节省比例、节省金额
```

### 检查代理端口

```bash
netstat -ano | findstr :18002
# → TCP 127.0.0.1:18002    LISTENING
#   如果端口被占用，换一个
```

---

## 9. 命令参考

| 命令 | 作用 |
|------|------|
| `popt run -- <cmd>` | 启动代理 → 跑命令 → 自动关闭 |
| `popt run --model NAME -- <cmd>` | 指定模型跑命令（推荐） |
| `popt proxy [--port PORT]` | 启动长驻代理 |
| `popt preview <file> [-p provider]` | 预览优化效果（不调 API） |
| `popt diagnose <a.json> <b.json> [-p provider]` | 诊断两个请求的缓存差异 |
| `popt stats [<file>]` | 分析日志中的缓存命中率 |

### 全局选项

```
--model, -m NAME     模型名（自动配 upstream + provider）
--upstream, -u URL   上游 API 地址
--provider NAME      优化策略（deepseek / anthropic / openai）
```

---

## 10. 常见问题

### Q：popt 和框架的关系？

**零侵入**。popt 是 HTTP 代理，不需要改框架代码。框架照常用自己 SDK，popt 在中间拦截请求做优化，转发到真正的 API。

### Q：支持哪些框架？

| 框架 | 接入方式 |
|------|---------|
| Claude Code | `popt run -- model xxx -- claude` |
| OpenAI SDK | popt 自动设 `OPENAI_BASE_URL` |
| Anthropic SDK | popt 自动设 `ANTHROPIC_BASE_URL` |
| LangChain / LlamaIndex | 设环境变量 → popt 自动接管 |
| Hermes | 改 `base_url` 指向 popt 代理端口 |
| Node.js / curl / 任意 HTTP | 改 `base_url` 指向代理 |

### Q：支持哪些模型和 API？

内置 15+ 服务商：DeepSeek、OpenAI、Anthropic、Groq、Together AI、Mistral、Fireworks AI、xAI (Grok)、Perplexity、GitHub Copilot (Codex)、OpenRouter、Azure OpenAI、Google Gemini。均支持 `--model` 自动匹配。

### Q：换了模型后需要改什么？

只需改 `--model` 参数：

```bash
# DeepSeek → OpenAI
popt run --model gpt-4o -- claude

# OpenAI → Groq
popt run --model deepseek-r1-671b -- python script.py
```

### Q：缓存优化到底省什么？

对 DeepSeek v4 flash，实测数据：

| prompt 长度 | 无优化 | 优化后 | 省什么 |
|------------|-------|-------|-------|
| 短 (~14t) | 0% 命中 | **60.4%** 命中 | 重复请求省 60% token |
| 长 (~2745t) | 97.9% 命中 | 98.0% 命中 | 接近满命中 |

核心机制：**前缀缓存**。popt 保证相同前缀在多次请求间完全一致，最大化 API 侧缓存命中。

### Q：报错 "command not found: popt"？

```bash
# uv 安装的检查
uv tool list | grep poptimize

# pip 安装的检查
pip show poptimize

# 如果版本 < 0.3.0，升级
uv tool upgrade poptimize
# 或
pip install --upgrade poptimize
```

### Q：报错端口被占用？

换个端口：

```bash
popt proxy --port 12345
# 或
popt run --model xxx -- claude   # run 模式自动选空闲端口
```

### Q：Windows 上 Unicode 显示乱码？

终端切换到 UTF-8：

```bash
$env:PYTHONIOENCODING = 'utf-8'
chcp 65001
```
