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

### Harbor 测试逻辑

```bash
uv run pytest tests/test_harbor_agent.py
```

Harbor 适配层通过 `VibeInstalledAgent` 暴露给 Harbor，测试用 `RecordingAgent` 记录 root/agent 执行命令，用 `FakeEnvironment` 记录上传行为，因此不需要真实 Harbor 环境即可验证集成契约。

测试覆盖三类行为：

1. **导入契约**：`vibe.harbor_agent` 和 `src.vibe.harbor_agent` 都能加载 `VibeInstalledAgent`，且 `name()` 返回 `vibe`。
2. **安装流程**：`install()` 会清理远端安装目录和虚拟环境，把本地源码上传到 `/installed-agent/vibe`，创建 `/opt/vibe-venv`，执行 editable install，并用 `/opt/vibe-venv/bin/vibe --help` 校验 CLI 可用。
3. **运行流程**：`run()` 以 agent 身份执行 `/opt/vibe-venv/bin/vibe --print -- <instruction>`，输出通过 `tee` 写入 `/logs/agent/vibe.txt`，并且不会通过 `cwd` 参数覆盖工作目录。
