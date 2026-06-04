# vibe

一个基于 LangGraph 的编码 Agent，支持对话式交互和工具调用。

## 安装

```bash
uv sync
```

## 配置

首次运行前需配置模型和 API Key，支持以下方式（优先级从高到低）：

1. **CLI 参数**：`-m <模型名>`、`--base-url <地址>`
2. **配置文件**：在配置目录下的 `settings.json` 中设置 `model`、`base_url`、`api_key`
3. **环境变量**：`OPENAI_API_KEY`（fallback）

配置示例：

```json
{
  "model": "gpt-4o",
  "api_key": "sk-..."
}
```

## 使用

### TUI 交互模式（默认）

```bash
vibe "写一个快速排序"
```

进入交互式终端后可多轮对话，按 `Ctrl+C` 退出。

### 单次输出模式

```bash
vibe "列出当前目录下的 Python 文件" --print
```

### 续接会话

```bash
vibe -s <session_id> "继续刚才的任务"
```

### 指定模型

```bash
vibe -m claude-sonnet-4-6 "解释这段代码"
```

## 项目结构

```
src/vibe/
├── agent/          # LangGraph ReAct 循环：状态图、对话管理
├── tools/          # 工具集合：bash、read、write、edit 等
├── session/        # 会话持久化（JSONL）
├── tui.py          # 终端交互界面
├── cli.py          # 命令行入口
└── config.py       # 配置加载
```

## 开发

```bash
uv run pytest      # 运行测试
uv run ruff check .  # 代码检查
uv run ruff format . # 格式化
```
