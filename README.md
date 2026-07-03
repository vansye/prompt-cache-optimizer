# popt — Prompt Structure Optimizer

**popt** 是一个轻量级的 HTTP 透明代理，插在 AI 框架和模型 API 之间，自动优化消息结构来最大化 [Prompt Caching](https://api-docs.deepseek.com/guides/kv_cache) 命中率。

**不需要改代码。** 任何 AI 框架——Claude Code、OpenAI SDK、Anthropic SDK、Codex、LangChain、Hermes……只要能配 API 地址或环境变量，就能接 popt。

```bash
pip install poptimize

# 一条命令启动代理 + 运行 AI 工具，所有请求自动优化
popt run -- claude
popt run -- python my_agent.py
popt run --model gpt-4o -- python script.py
popt run --model deepseek-v4-flash -- node bot.js
```

> 📖 **完整操作手册**：[docs/usage.md](docs/usage.md) — 安装检查 → 模型切换 → Hermes 接入 → 验证优化 → 故障排查

---

## 快速开始

### `popt run` — 一条命令通吃

```bash
# 自动检测 ANTHROPIC_BASE_URL / OPENAI_BASE_URL 环境变量
popt run -- claude
popt run -- python my_script.py

# 指定模型（自动配 upstream + provider + API 格式）
popt run --model deepseek-v4-flash -- claude
popt run --model gpt-4o -- python my_script.py

# 手动指定（完全控制）
popt run --upstream https://api.deepseek.com --provider deepseek -- python script.py
```

`--model` 一键完成所有配置：

| 模型名 | upstream | provider | 格式 |
|--------|----------|----------|------|
| `deepseek-v4-flash` | `api.deepseek.com/anthropic` | deepseek | anthropic |
| `gpt-4o` | `api.openai.com` | openai | openai |
| `deepseek-r1-671b` | `api.groq.com/openai/v1` | groq | openai |
| `grok-beta` | `api.x.ai` | xai | openai |

### `popt proxy` — 长驻代理

```bash
popt proxy --port 9999 --model deepseek-v4-flash
```

另一个终端设 `ANTHROPIC_BASE_URL=http://127.0.0.1:9999`，正常使用你的工具。

### Python API — 代码集成

```python
from optimizer import optimize
messages = [{"role": "user", "content": "Hello!"}]
optimized = optimize(messages, provider="deepseek")
```

---

## 如何工作

```
输入消息 → Normalizer → Reorderer → Aligner → Formatter → 到上游 API
           (标准化)    (排序固定)  (填充阈值) (适配格式)
```

每个阶段有 SafetyCheck 保底——优化改变语义则回退原始输入。

## 实验验证（DeepSeek v4 flash）

| 场景 | 未优化 | 优化后 |
|------|--------|--------|
| 短 prompt (~14t) | 0% 命中 | **60.4%** 命中 |
| 中 prompt (~203t) | 63.1% | 63.4% |
| 长 prompt (~2745t) | 97.9% | 98.0% |

---

## 安装

```bash
pip install poptimize
# 或
uv tool install poptimize
```

可选：`pip install tiktoken`（提升 token 估算精度）

## License

MIT
