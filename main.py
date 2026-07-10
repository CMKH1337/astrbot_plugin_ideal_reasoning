import asyncio
import copy
import json
from contextvars import ContextVar
from dataclasses import dataclass
from types import MethodType
from typing import Any, Awaitable, Callable

from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

from astrbot.api import AstrBotConfig, FunctionTool, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_context import AstrAgentContext


EXTRA_BODY_KEYS = ("custom_extra_body", "custom_request_body")
MANAGED_BODY_KEYS = (
    "reasoning",
    "effort",
    "thinkingLevel",
    "thinkingBudget",
    "reasoning_effort",
)


@dataclass(frozen=True)
class VendorSpec:
    display_name: str
    parameter_name: str
    value_type: str
    options: tuple[str, ...]
    default_description: str
    note: str = ""


@dataclass
class SessionReasoningState:
    vendor: str
    requested_value: str
    value: str | int | None
    remaining_requests: int


ACTIVE_REASONING: ContextVar[
    tuple[str, str, str | int | None] | None
] = ContextVar(
    "reasoning_switch_active",
    default=None,
)


@pydantic_dataclass
class GetReasoningOptionsTool(FunctionTool[AstrAgentContext]):
    name: str = "get_reasoning_options"
    description: str = "查询当前会话可用的推理强度、映射规则和临时设置。"
    parameters: dict = Field(default_factory=lambda: {"type": "object", "properties": {}})
    callback: Callable[[AstrMessageEvent], Awaitable[str]] | None = Field(
        default=None,
        exclude=True,
    )

    async def call(
        self,
        context: ContextWrapper[AstrAgentContext],
        **kwargs,
    ) -> str:
        if self.callback is None:
            return "推理选项工具尚未初始化。"
        return await self.callback(context.context.event)


@pydantic_dataclass
class SetReasoningLevelTool(FunctionTool[AstrAgentContext]):
    name: str = "set_reasoning_level"
    description: str = (
        "为当前会话的后续模型请求临时设置推理强度。"
        "仅在任务确实需要改变推理深度时调用；设置不会修改全局 Provider 配置。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "level": {
                    "type": "string",
                    "description": (
                        "推理强度。先调用 get_reasoning_options 查询可选值；"
                        "Gemini 2.5 也可传整数预算、dynamic 或 off。"
                    ),
                },
            },
            "required": ["level"],
        },
    )
    callback: Callable[[AstrMessageEvent, str], Awaitable[str]] | None = Field(
        default=None,
        exclude=True,
    )

    async def call(
        self,
        context: ContextWrapper[AstrAgentContext],
        **kwargs,
    ) -> str:
        if self.callback is None:
            return "推理设置工具尚未初始化。"
        return await self.callback(context.context.event, str(kwargs.get("level", "")))


VENDOR_SPECS = {
    "openai": VendorSpec(
        "OpenAI (GPT-5.5 / 5.6 / o3)",
        "reasoning.effort",
        "String",
        ("none", "minimal", "low", "medium", "high", "xhigh"),
        "因模型而异（如 GPT-5.5 默认为 medium）",
    ),
    "anthropic": VendorSpec(
        "Anthropic (Claude 4 / Opus 4.8)",
        "effort",
        "String",
        ("low", "medium", "high", "xhigh", "max"),
        "因模型而异（如 Opus 4.8 默认为 high）",
        "同时影响思考深度、回答长度和工具调用意愿。",
    ),
    "gemini_3": VendorSpec(
        "Google Gemini 3.0 / 3.1 Pro",
        "thinkingLevel",
        "String",
        ("minimal", "low", "medium", "high"),
        "因模型而异（如 3.1 Pro 默认为 high）",
        "输入 xhigh 或 max 时会映射为 high。",
    ),
    "gemini_25": VendorSpec(
        "Google Gemini 2.5",
        "thinkingBudget",
        "Integer",
        ("0", "-1", "<正整数 token 数>"),
        "-1（动态）",
        "0 表示禁用，-1 表示动态，也可输入具体 token 数量。",
    ),
    "deepseek": VendorSpec(
        "DeepSeek V4 Pro",
        "reasoning_effort",
        "String",
        ("high", "max"),
        "high",
        "low/medium 映射为 high，xhigh 映射为 max；启用后 temperature 等参数可能不生效。",
    ),
    "xai": VendorSpec(
        "xAI (Grok 3 Mini / 4.X)",
        "reasoning_effort",
        "String",
        ("low", "medium", "high"),
        "厂商未明确指定",
    ),
}


@register(
    "astrbot_plugin_reasoning_switch",
    "CMKH",
    "通过指令切换不同厂商模型的推理强度。",
    "v1.2.0",
)
class ReasoningSwitchPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.session_states: dict[str, SessionReasoningState] = {}
        self._wrapped_providers: dict[str, Any] = {}
        self._provider_locks: dict[str, asyncio.Lock] = {}

        if self.config.get("enable_query_tool", True):
            self.context.add_llm_tools(
                GetReasoningOptionsTool(callback=self._tool_get_options),
            )
        if self.config.get("enable_set_tool", False):
            self.context.add_llm_tools(
                SetReasoningLevelTool(callback=self._tool_set_level),
            )

    async def terminate(self) -> None:
        """恢复被包装的 Provider 方法并清理会话状态。"""
        for provider_id, provider in list(self._wrapped_providers.items()):
            originals = getattr(provider, "_reasoning_switch_originals", None)
            if isinstance(originals, dict):
                for method_name, original in originals.items():
                    setattr(provider, method_name, original)
                delattr(provider, "_reasoning_switch_originals")
            logger.debug(f"[reasoning_switch] 已恢复 Provider {provider_id} 请求方法")
        self._wrapped_providers.clear()
        self._provider_locks.clear()
        self.session_states.clear()
        tool_manager = self.context.get_llm_tool_manager()
        for tool_name in ("get_reasoning_options", "set_reasoning_level"):
            if tool_manager.get_func(tool_name) is not None:
                tool_manager.remove_func(tool_name)

    @filter.on_llm_request(priority=1000)
    async def on_llm_request(
        self,
        event: AstrMessageEvent,
        request: ProviderRequest,
    ) -> None:
        """为当前请求激活会话级推理参数，不修改 Provider 持久化配置。"""
        state = self.session_states.get(event.unified_msg_origin)
        if state is None:
            ACTIVE_REASONING.set(None)
            return

        try:
            provider_id = await self.context.get_current_chat_provider_id(
                umo=event.unified_msg_origin,
            )
            provider = self.context.get_using_provider(event.unified_msg_origin)
            if provider is None:
                raise ValueError("当前会话没有可用的聊天 Provider")
            self._ensure_provider_wrapped(provider_id, provider)
            ACTIVE_REASONING.set(
                (event.unified_msg_origin, state.vendor, state.value),
            )
            logger.info(
                f"[reasoning_switch] 会话 {event.unified_msg_origin} 已准备应用 "
                f"{state.vendor}={state.value}",
            )
        except Exception as exc:  # noqa: BLE001
            ACTIVE_REASONING.set(None)
            logger.error(f"[reasoning_switch] 激活会话级推理参数失败: {exc}")

    def _ensure_provider_wrapped(self, provider_id: str, provider: Any) -> None:
        if getattr(provider, "_reasoning_switch_originals", None) is not None:
            self._wrapped_providers[provider_id] = provider
            return

        originals: dict[str, Callable[..., Any]] = {}
        provider_lock = self._provider_locks.setdefault(provider_id, asyncio.Lock())
        for method_name in ("text_chat", "text_chat_stream"):
            original = getattr(provider, method_name, None)
            if original is None:
                continue
            originals[method_name] = original

            if method_name == "text_chat_stream":
                async def stream_wrapper(
                    _provider,
                    *args,
                    __original=original,
                    **kwargs,
                ):
                    async with provider_lock:
                        with self._temporary_provider_config(_provider):
                            try:
                                async for item in __original(*args, **kwargs):
                                    yield item
                            finally:
                                self._consume_active_state()

                setattr(provider, method_name, MethodType(stream_wrapper, provider))
            else:
                async def chat_wrapper(
                    _provider,
                    *args,
                    __original=original,
                    **kwargs,
                ):
                    async with provider_lock:
                        with self._temporary_provider_config(_provider):
                            try:
                                return await __original(*args, **kwargs)
                            finally:
                                self._consume_active_state()

                setattr(provider, method_name, MethodType(chat_wrapper, provider))

        setattr(provider, "_reasoning_switch_originals", originals)
        self._wrapped_providers[provider_id] = provider

    class _TemporaryProviderConfig:
        def __init__(self, plugin: "ReasoningSwitchPlugin", provider: Any):
            self.plugin = plugin
            self.provider = provider
            self.original_config: dict[str, Any] | None = None

        def __enter__(self):
            active = ACTIVE_REASONING.get()
            if active is None:
                return self
            _, vendor, value = active
            self.original_config = self.provider.provider_config
            temporary_config = copy.deepcopy(self.original_config)
            self.plugin._apply_runtime_value(temporary_config, vendor, value)
            self.provider.provider_config = temporary_config
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            if self.original_config is not None:
                self.provider.provider_config = self.original_config
            return False

    def _temporary_provider_config(self, provider: Any) -> "_TemporaryProviderConfig":
        return self._TemporaryProviderConfig(self, provider)

    def _consume_active_state(self) -> None:
        active = ACTIVE_REASONING.get()
        if active is None:
            return
        umo, vendor, value = active
        state = self.session_states.get(umo)
        if state is not None and state.vendor == vendor and state.value == value:
            state.remaining_requests -= 1
            logger.debug(
                f"[reasoning_switch] 会话 {umo} 已消费一次临时设置，"
                f"剩余 {state.remaining_requests} 次",
            )
            if state.remaining_requests <= 0:
                self.session_states.pop(umo, None)
        next_state = self.session_states.get(umo)
        if next_state is None:
            ACTIVE_REASONING.set(None)
        else:
            ACTIVE_REASONING.set((umo, next_state.vendor, next_state.value))

    def _apply_runtime_value(
        self,
        provider_config: dict[str, Any],
        vendor: str,
        value: str | int | None,
    ) -> None:
        provider_type = provider_config.get("type", "")
        if vendor in {"gemini_3", "gemini_25"} and provider_type == "googlegenai_chat_completion":
            thinking_config = copy.deepcopy(provider_config.get("gm_thinking_config", {}))
            if not isinstance(thinking_config, dict):
                thinking_config = {}
            config_key = "level" if vendor == "gemini_3" else "budget"
            if value is None:
                thinking_config.pop(config_key, None)
            else:
                thinking_config[config_key] = value
            provider_config["gm_thinking_config"] = thinking_config
            return

        body_key = self._get_extra_body_key(provider_config)
        original_body = provider_config.get(body_key)
        request_body = self._parse_request_body(original_body, body_key)
        for key in MANAGED_BODY_KEYS:
            request_body.pop(key, None)
        if value is not None:
            if vendor == "openai":
                request_body["reasoning"] = {"effort": value}
            elif vendor == "anthropic":
                request_body["effort"] = value
            elif vendor == "gemini_3":
                request_body["thinkingLevel"] = value
            elif vendor == "gemini_25":
                request_body["thinkingBudget"] = value
            else:
                request_body["reasoning_effort"] = value
        provider_config[body_key] = self._serialize_request_body(
            request_body,
            original_body,
        )

    async def _tool_get_options(self, event: AstrMessageEvent) -> str:
        vendor = await self._get_event_vendor(event)
        state_text = self._format_session_state(event.unified_msg_origin)
        return self._format_options(vendor) + "\n" + state_text

    async def _tool_set_level(self, event: AstrMessageEvent, level: str) -> str:
        if not self.config.get("enable_set_tool", False):
            return "会话推理设置工具已被管理员禁用。"
        if self.config.get("set_tool_admin_only", True) and not event.is_admin():
            return "权限不足：管理员配置为仅允许管理员会话使用此工具。"
        if not self.config.get("allow_group_chat_tools", False) and not event.is_private_chat():
            return "当前不允许在群聊中使用会话推理设置工具。"

        normalized_level = level.strip().lower()
        if not normalized_level:
            return "level 不能为空。请先调用 get_reasoning_options 查询可选值。"
        if normalized_level in {"reset", "clear"}:
            self.session_states.pop(event.unified_msg_origin, None)
            ACTIVE_REASONING.set(None)
            return "已清除当前会话的临时推理设置。"

        vendor = await self._get_event_vendor(event)
        try:
            mapped_value = self._normalize_value(vendor, normalized_level)
            self._validate_tool_limit(vendor, mapped_value)
        except ValueError as exc:
            return f"设置失败：{exc}\n{self._format_options(vendor)}"

        turns = max(1, min(int(self.config.get("session_effective_requests", 3)), 20))
        existing = self.session_states.get(event.unified_msg_origin)
        if (
            existing is not None
            and existing.vendor == vendor
            and existing.value == mapped_value
        ):
            return (
                f"当前会话已设置为 {mapped_value}，"
                f"还剩 {existing.remaining_requests} 次模型请求有效。"
            )

        self.session_states[event.unified_msg_origin] = SessionReasoningState(
            vendor=vendor,
            requested_value=normalized_level,
            value=mapped_value,
            remaining_requests=turns,
        )
        provider_id = await self.context.get_current_chat_provider_id(
            umo=event.unified_msg_origin,
        )
        provider = self.context.get_using_provider(event.unified_msg_origin)
        if provider is None:
            self.session_states.pop(event.unified_msg_origin, None)
            return "设置失败：当前会话没有可用的聊天 Provider。"
        self._ensure_provider_wrapped(provider_id, provider)
        ACTIVE_REASONING.set(
            (event.unified_msg_origin, vendor, mapped_value),
        )
        return (
            f"已为当前会话设置 {VENDOR_SPECS[vendor].display_name} "
            f"推理值 {mapped_value}，从下一次模型请求开始生效，共 {turns} 次。"
            "该设置不会修改全局 Provider 配置。"
        )

    def _validate_tool_limit(self, vendor: str, value: str | int | None) -> None:
        if value is None:
            return
        if vendor == "gemini_25":
            budget = int(value)
            max_budget = max(0, int(self.config.get("max_thinking_budget", 8192)))
            if budget > max_budget:
                raise ValueError(f"thinkingBudget 超过工具上限 {max_budget}")
            return

        order = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
        max_level = str(self.config.get("llm_tool_max_level", "high") or "high").lower()
        if max_level not in order:
            max_level = "high"
        actual = str(value).lower()
        if actual in order and order.index(actual) > order.index(max_level):
            raise ValueError(f"推理强度 {actual} 超过工具上限 {max_level}")

    def _format_session_state(self, umo: str) -> str:
        state = self.session_states.get(umo)
        if state is None:
            return "当前会话临时设置：无"
        return (
            f"当前会话临时设置：{state.vendor}={state.value} "
            f"（用户请求 {state.requested_value}，剩余 {state.remaining_requests} 次）"
        )

    @filter.command("reasoning_session")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def reasoning_session(self, event: AstrMessageEvent, level: str = ""):
        """查询、设置或清除当前会话临时推理参数。"""
        normalized_level = level.strip().lower()
        if not normalized_level:
            yield event.plain_result(self._format_session_state(event.unified_msg_origin))
            return
        if normalized_level in {"off", "reset", "clear"}:
            self.session_states.pop(event.unified_msg_origin, None)
            ACTIVE_REASONING.set(None)
            yield event.plain_result("已清除当前会话的临时推理设置。")
            return

        vendor = await self._get_event_vendor(event)
        try:
            mapped_value = self._normalize_value(vendor, normalized_level)
            turns = max(1, min(int(self.config.get("session_effective_requests", 3)), 20))
        except (TypeError, ValueError) as exc:
            yield event.plain_result(f"设置失败：{exc}\n\n{self._format_options(vendor)}")
            return
        self.session_states[event.unified_msg_origin] = SessionReasoningState(
            vendor=vendor,
            requested_value=normalized_level,
            value=mapped_value,
            remaining_requests=turns,
        )
        yield event.plain_result(
            f"当前会话已临时设置为 {mapped_value}，后续 {turns} 次模型请求生效。",
        )

    @filter.command("reasoning")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def reasoning(self, event: AstrMessageEvent, level: str = ""):
        """按插件配置的厂商切换当前 Provider 的推理强度。"""
        vendor = await self._get_event_vendor(event)
        spec = VENDOR_SPECS[vendor]
        normalized_level = level.strip().lower()
        if not normalized_level:
            yield event.plain_result(self._format_options(vendor))
            return

        try:
            mapped_value = self._normalize_value(vendor, normalized_level)
            provider_id = await self.context.get_current_chat_provider_id(
                umo=event.unified_msg_origin,
            )
            provider_config = self._get_provider_config(provider_id)
            effective_config = self.context.provider_manager.get_merged_provider_config(
                provider_config,
            )
            applied_parameter = self._apply_vendor_value(
                provider_config,
                effective_config,
                vendor,
                mapped_value,
            )
            await self.context.provider_manager.update_provider(
                provider_id,
                provider_config,
            )
        except ValueError as exc:
            yield event.plain_result(f"切换失败：{exc}\n\n{self._format_options(vendor)}")
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"[reasoning_switch] 切换 Provider 推理模式失败: {exc}")
            yield event.plain_result("切换失败，请查看 AstrBot 日志。")
            return

        display_value = "关闭（删除参数）" if mapped_value is None else str(mapped_value)
        mapping_text = ""
        if mapped_value is not None and str(mapped_value).lower() != normalized_level:
            mapping_text = f"（由 {normalized_level} 映射）"
        logger.info(
            f"[reasoning_switch] Provider {provider_id} 的 {applied_parameter} "
            f"已设置为 {display_value}",
        )
        yield event.plain_result(
            f"已切换 {spec.display_name} 推理模式\n"
            f"Provider：{provider_id}\n"
            f"参数：{applied_parameter}\n"
            f"值：{display_value}{mapping_text}\n"
            "注意：该 Provider 的所有会话都会生效。",
        )

    @filter.command("reasoning_options", alias=["reasoning_levels"])
    async def reasoning_options(self, event: AstrMessageEvent):
        """查询当前配置厂商支持的推理强度选项和当前值。"""
        vendor = await self._get_event_vendor(event)
        current_text = ""
        try:
            provider_id = await self.context.get_current_chat_provider_id(
                umo=event.unified_msg_origin,
            )
            provider_config = self._get_effective_provider_config(provider_id)
            parameter, current_value = self._get_current_value(provider_config, vendor)
            rendered_value = "未显式设置（使用厂商/模型默认值）"
            if current_value is not None:
                rendered_value = str(current_value)
            current_text = (
                f"\n当前 Provider：{provider_id}"
                f"\n当前参数：{parameter}"
                f"\n当前值：{rendered_value}"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[reasoning_switch] 查询当前推理配置失败: {exc}")
            current_text = f"\n当前值读取失败：{exc}"

        yield event.plain_result(self._format_options(vendor) + current_text)

    @filter.command("reasoning_vendor")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def reasoning_vendor(self, event: AstrMessageEvent, vendor: str = ""):
        """查询或切换推理参数适配厂商。"""
        normalized_vendor = vendor.strip().lower()
        if not normalized_vendor:
            yield event.plain_result(
                "全局兜底厂商："
                f"{self._get_vendor()}\n"
                f"可选厂商：{' / '.join(VENDOR_SPECS)}\n"
                "已分组 Provider 会自动识别厂商。\n"
                "切换兜底值：/reasoning_vendor <厂商>",
            )
            return
        if normalized_vendor not in VENDOR_SPECS:
            yield event.plain_result(
                f"不支持的厂商：{normalized_vendor}\n"
                f"可选厂商：{' / '.join(VENDOR_SPECS)}",
            )
            return

        self.config["vendor"] = normalized_vendor
        self.config.save_config()
        self.session_states.clear()
        ACTIVE_REASONING.set(None)
        yield event.plain_result(
            f"已切换推理参数适配厂商为：{VENDOR_SPECS[normalized_vendor].display_name}\n"
            "已清除全部会话临时设置。",
        )

    def _get_vendor(self) -> str:
        vendor = str(self.config.get("vendor", "openai") or "openai").lower()
        if vendor not in VENDOR_SPECS:
            logger.warning(f"[reasoning_switch] 未知厂商 {vendor}，已回退到 openai")
            return "openai"
        return vendor

    async def _get_event_vendor(self, event: AstrMessageEvent) -> str:
        provider_id = await self.context.get_current_chat_provider_id(
            umo=event.unified_msg_origin,
        )
        return self._get_vendor_for_provider(provider_id)

    def _get_vendor_for_provider(self, provider_id: str) -> str:
        provider_map = self.config.get("provider_vendor_map", {})
        if isinstance(provider_map, list):
            for mapping in provider_map:
                if not isinstance(mapping, dict):
                    continue
                vendor = mapping.get("__template_key")
                mapped_provider_id = str(mapping.get("provider_id", "") or "")
                if vendor in VENDOR_SPECS and mapped_provider_id == provider_id:
                    return vendor
        return self._get_vendor()

    def _get_provider_config(self, provider_id: str) -> dict[str, Any]:
        for provider_config in self.context.provider_manager.providers_config:
            if provider_config.get("id") == provider_id:
                return copy.deepcopy(provider_config)
        raise ValueError(f"未找到当前 Provider 配置：{provider_id}")

    def _get_effective_provider_config(self, provider_id: str) -> dict[str, Any]:
        provider_config = self._get_provider_config(provider_id)
        return self.context.provider_manager.get_merged_provider_config(provider_config)

    @staticmethod
    def _normalize_value(vendor: str, value: str) -> str | int | None:
        if value == "off":
            if vendor == "openai":
                return "none"
            if vendor == "gemini_25":
                return 0
            return None

        if vendor == "gemini_25":
            if value == "dynamic":
                return -1
            try:
                budget = int(value)
            except ValueError as exc:
                raise ValueError("Gemini 2.5 需要 0、-1、dynamic 或正整数 token 数") from exc
            if budget < -1:
                raise ValueError("thinkingBudget 不能小于 -1")
            return budget

        if vendor == "gemini_3" and value in {"xhigh", "max"}:
            return "high"
        if vendor == "deepseek":
            if value in {"low", "medium"}:
                return "high"
            if value == "xhigh":
                return "max"

        if value not in VENDOR_SPECS[vendor].options:
            options = " / ".join(VENDOR_SPECS[vendor].options)
            raise ValueError(f"不支持 {value}，可选值：{options} / off")
        return value

    def _apply_vendor_value(
        self,
        provider_config: dict[str, Any],
        effective_config: dict[str, Any],
        vendor: str,
        value: str | int | None,
    ) -> str:
        provider_type = effective_config.get("type", "")
        if vendor in {"gemini_3", "gemini_25"} and provider_type == "googlegenai_chat_completion":
            thinking_config = copy.deepcopy(effective_config.get("gm_thinking_config", {}))
            if not isinstance(thinking_config, dict):
                thinking_config = {}
            config_key = "level" if vendor == "gemini_3" else "budget"
            if value is None:
                thinking_config.pop(config_key, None)
            else:
                thinking_config[config_key] = value
            provider_config["gm_thinking_config"] = thinking_config
            return f"gm_thinking_config.{config_key}"

        body_key = self._get_extra_body_key(effective_config)
        original_body = provider_config.get(body_key)
        effective_body = effective_config.get(body_key)
        request_body = self._parse_request_body(effective_body, body_key)
        for key in MANAGED_BODY_KEYS:
            request_body.pop(key, None)

        if value is not None:
            if vendor == "openai":
                request_body["reasoning"] = {"effort": value}
            elif vendor == "anthropic":
                request_body["effort"] = value
            elif vendor == "gemini_3":
                request_body["thinkingLevel"] = value
            elif vendor == "gemini_25":
                request_body["thinkingBudget"] = value
            else:
                request_body["reasoning_effort"] = value

        provider_config[body_key] = self._serialize_request_body(
            request_body,
            original_body if original_body is not None else effective_body,
        )
        return VENDOR_SPECS[vendor].parameter_name

    def _get_current_value(
        self,
        provider_config: dict[str, Any],
        vendor: str,
    ) -> tuple[str, Any]:
        provider_type = provider_config.get("type", "")
        if vendor in {"gemini_3", "gemini_25"} and provider_type == "googlegenai_chat_completion":
            config_key = "level" if vendor == "gemini_3" else "budget"
            thinking_config = provider_config.get("gm_thinking_config", {})
            value = thinking_config.get(config_key) if isinstance(thinking_config, dict) else None
            return f"gm_thinking_config.{config_key}", value

        body_key = self._get_extra_body_key(provider_config)
        body = self._parse_request_body(provider_config.get(body_key), body_key)
        if vendor == "openai":
            reasoning = body.get("reasoning")
            value = reasoning.get("effort") if isinstance(reasoning, dict) else None
        else:
            value = body.get(VENDOR_SPECS[vendor].parameter_name)
        return VENDOR_SPECS[vendor].parameter_name, value

    @staticmethod
    def _format_options(vendor: str) -> str:
        spec = VENDOR_SPECS[vendor]
        options = " / ".join(spec.options)
        note = f"\n说明：{spec.note}" if spec.note else ""
        return (
            f"当前适配厂商：{spec.display_name}\n"
            f"参数：{spec.parameter_name}\n"
            f"可选值：{options}\n"
            f"默认值：{spec.default_description}"
            f"{note}\n"
            "切换指令：/reasoning <值>"
        )

    @staticmethod
    def _get_extra_body_key(provider_config: dict[str, Any]) -> str:
        for key in EXTRA_BODY_KEYS:
            if key in provider_config:
                return key
        return "custom_extra_body"

    @staticmethod
    def _parse_request_body(value: Any, field_name: str) -> dict[str, Any]:
        if value in (None, ""):
            return {}
        if isinstance(value, dict):
            return copy.deepcopy(value)
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{field_name} 不是合法 JSON") from exc
            if isinstance(parsed, dict):
                return parsed
        raise ValueError(f"{field_name} 必须是 JSON 对象")

    @staticmethod
    def _serialize_request_body(body: dict[str, Any], original_value: Any) -> Any:
        if isinstance(original_value, str):
            return json.dumps(body, ensure_ascii=False)
        return body
