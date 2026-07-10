<p align="center">
  <img src="https://count.getloli.com/get/@astrbot_plugin_ideal_reasoning?theme=gelbooru-h" alt="访问计数" />
</p>

# Astrbot 多厂商推理强度切换插件

最近老是刷到claude code开启Ultracode装逼的视频，我也想装逼，于是这个插件诞生了。

通过指令修改当前会话所用聊天LLM的推理参数。修改会持久化到 AstrBot 模型提供商配置并实时重载，无需重启。

## 配置

安装并重载插件后，在 AstrBot 插件配置页的“Provider 厂商分组”中，将每个聊天 Provider 加入其实际模型厂商分组：

| 配置值 | 厂商/模型 |
| --- | --- |
| `openai` | OpenAI GPT-5.5 / 5.6 / o3 系列 |
| `anthropic` | Anthropic Claude 4 / Opus 4.8 系列 |
| `gemini_3` | Google Gemini 3.0 / 3.1 Pro 系列 |
| `gemini_25` | Google Gemini 2.5 系列 |
| `deepseek` | DeepSeek V4 Pro 系列 |
| `xai` | xAI Grok 3 Mini / 4.X 系列 |

同一厂商可添加多个 Provider。插件会用当前会话的 Provider ID 查找对应分组；因此通过 AstrBot `/provider` 切换模型后，后续 `/reasoning`、`/reasoning_options`、`/reasoning_session` 和 LLM 工具都会自动改用对应厂商的参数格式与可选值。

如果当前 Provider 没有加入任何分组，插件才会使用顶层 `vendor` 的全局兜底值。`/reasoning_vendor` 用于查看或修改此兜底值，不会覆盖已经配置的 Provider 分组。

## 指令

| 指令 | 权限 | 作用 |
| --- | --- | --- |
| `/reasoning <值>` | 管理员 | 修改当前 Provider 的推理参数 |
| `/reasoning` | 管理员 | 显示当前厂商可选值 |
| `/reasoning_options` | 所有用户 | 查询当前厂商可选值、当前 Provider 和当前值 |
| `/reasoning_levels` | 所有用户 | `/reasoning_options` 的别名 |
| `/reasoning_session [值]` | 管理员 | 查询或设置当前会话临时值；`off/reset/clear` 清除 |
| `/reasoning_vendor [厂商]` | 管理员 | 查询或切换未分组 Provider 使用的全局兜底厂商；切换时清除全部会话临时设置 |

## LLM 函数工具

插件可向 AstrBot 注册两个工具：

- `get_reasoning_options`：只读查询当前厂商、可选值和会话临时状态，默认启用。
- `set_reasoning_level`：让 LLM 临时调整当前会话后续请求的推理强度，默认关闭。

`set_reasoning_level` 不调用 Provider 管理器的更新或重载方法，也不写磁盘配置。它以 `unified_msg_origin` 隔离状态，并在实际调用 Provider 时临时替换该实例的请求配置，请求完成后立即恢复。共享 Provider 的替换过程有异步锁保护，避免并发会话串值。

函数工具在一次 Agent 工具循环中调用后，设置会从紧接着的下一次模型请求生效。每次实际模型请求消费一次有效次数，用完后自动清除。

### 工具安全配置

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `enable_query_tool` | `true` | 注册只读查询工具 |
| `enable_set_tool` | `false` | 注册会话级设置工具 |
| `set_tool_admin_only` | `true` | 调用时主动检查消息事件管理员身份 |
| `allow_group_chat_tools` | `false` | 是否允许群聊自动设置 |
| `llm_tool_max_level` | `high` | LLM 自动设置的最高字符串等级 |
| `max_thinking_budget` | `8192` | Gemini 2.5 自动设置的最大正整数预算 |
| `session_effective_requests` | `3` | 临时设置有效请求次数，限制为 1～20 |

管理员 `/reasoning` 仍然修改 Provider 全局持久化配置；`/reasoning_session` 和 LLM 工具只修改当前会话状态。优先级为：

1. 当前会话临时设置
2. Provider 全局配置
3. 厂商或模型默认值

## 厂商参数

### OpenAI

- 请求参数：`reasoning.effort`
- 可选值：`none`、`minimal`、`low`、`medium`、`high`、`xhigh`
- `off` 映射为 `none`

### Anthropic

- 请求参数：`effort`
- 可选值：`low`、`medium`、`high`、`xhigh`、`max`
- `off` 删除该参数，恢复厂商/模型默认行为

### Gemini 3.0 / 3.1

- 请求参数：`thinkingLevel`
- 可选值：`minimal`、`low`、`medium`、`high`
- `xhigh`、`max` 映射为 `high`
- 使用 AstrBot 原生 Google GenAI Provider 时，插件会写入 `gm_thinking_config.level`

### Gemini 2.5

- 请求参数：`thinkingBudget`
- 可选值：`0`（禁用）、`-1`（动态）或任意正整数 token 数
- 便捷输入：`off` 映射为 `0`，`dynamic` 映射为 `-1`
- 使用 AstrBot 原生 Google GenAI Provider 时，插件会写入 `gm_thinking_config.budget`

### DeepSeek

- 请求参数：`reasoning_effort`
- 可选值：`high`、`max`
- `low`、`medium` 映射为 `high`
- `xhigh` 映射为 `max`
- `off` 删除该参数
- 启用推理后，部分后端会忽略 `temperature` 等采样参数

### xAI

- 请求参数：`reasoning_effort`
- 可选值：`low`、`medium`、`high`
- `off` 删除该参数

## 请求体兼容

- AstrBot 4.16 的 OpenAI 兼容 Provider 使用 `custom_extra_body` 字典。
- 插件兼容已有的 `custom_request_body` 字段以及 JSON 字符串格式。
- 修改时会保留请求体内其他自定义参数，并清理之前由本插件写入的其他厂商推理参数。
- 原生 Gemini Provider 使用 AstrBot 自带的 `gm_thinking_config`，不会错误地把参数塞进无效的额外请求体。

## 注意事项

- 修改将直接修改模型提供商的自定义请求体参数，所有使用同一 Provider 的会话都会受到影响。
- 上述 Provider 级影响仅针对管理员 `/reasoning`；函数工具和 `/reasoning_session` 按会话隔离。
- 必须在插件配置中选择与当前 Provider 实际模型相匹配的厂商。
- 模型或中转后端若不支持对应参数，请求可能被后端拒绝。
- 修改函数工具注册开关后需要重载插件，使工具注册状态生效。
