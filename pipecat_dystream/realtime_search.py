"""Low-overhead routing for provider-native real-time web search."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from typing import Any
from zoneinfo import ZoneInfo


_MODE_ALIASES = {
    "router": "smart",
    "provider_auto": "auto",
}
_VALID_MODES = frozenset({"off", "smart", "auto", "always"})
_VALID_STRATEGIES = frozenset({"turbo", "max", "agent"})

_EXPLICIT_SEARCH_RE = re.compile(
    r"(?:帮我|请|麻烦)?(?:联网|上网|在线)?(?:查一下|查查|查询|搜索|搜一下|检索)"
    r"|(?:联网|上网)(?:看|查|搜索|检索)?"
    r"|\b(?:search|look\s+up|browse|web\s+search)\b",
    re.IGNORECASE,
)

# These subjects are inherently time-sensitive.  The list is intentionally
# broad enough for public real-time information, but excludes conversational
# words such as "现在" by themselves so "你现在在干嘛" stays on the fast path.
_LIVE_SUBJECT_RE = re.compile(
    r"天气|气温|降雨|下雨|空气质量|台风|地震|预警|"
    r"新闻|热搜|头条|突发|"
    r"股价|股票|行情|金价|银价|油价|汇率|币价|期货|指数|市值|"
    r"比分|赛程|战绩|积分榜|排名|票房|"
    r"航班|机票|列车|高铁|火车票|路况|拥堵|堵车|"
    r"财报|公告|政策|法规|利率|库存|现任|在任|"
    r"\b(?:weather|forecast|news|headline|stock|price|market|exchange\s+rate|"
    r"score|schedule|flight|traffic)\b",
    re.IGNORECASE,
)

_FRESHNESS_RE = re.compile(
    r"现在|目前|当前|今天|今日|今晚|明天|本周|本月|最近|刚刚|最新|实时|截至|"
    r"\b(?:now|current|currently|today|tonight|tomorrow|latest|recent|live)\b",
    re.IGNORECASE,
)

_FRESH_FACT_RE = re.compile(
    r"多少|多少钱|谁是|是谁|几点|什么时候|哪一|哪些|有没有|"
    r"发生|发布|上线|更新|版本|模型|结果|进展|状态|数据|消息|价格|"
    r"CEO|首席执行官|董事长|总统|总理|负责人|"
    r"\b(?:who|what|when|where|how\s+much|released?|updated?|version|model|status)\b",
    re.IGNORECASE,
)

_LOCAL_CONTEXT_RE = re.compile(
    r"(?:我们|咱们|本项目|这个项目|当前系统|这套系统|你)"
    r".{0,16}(?:模型|版本|架构|代码|方案|服务|程序|配置)",
    re.IGNORECASE,
)

_STATIC_EXPLANATION_RE = re.compile(
    r"解释|介绍|科普|原理|定义|概念|什么是|是什么|有何区别|有什么区别|为何|为什么|"
    r"\b(?:explain|define|definition|difference|how\s+does|why)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RealtimeSearchDecision:
    """One request's routing decision."""

    enabled: bool
    forced: bool
    reason: str


def normalize_realtime_search_mode(mode: str) -> str:
    normalized = _MODE_ALIASES.get(mode.strip().lower(), mode.strip().lower())
    if normalized not in _VALID_MODES:
        choices = ", ".join(sorted(_VALID_MODES))
        raise ValueError(f"realtime_search_mode must be one of: {choices}")
    return normalized


def normalize_realtime_search_strategy(strategy: str) -> str:
    normalized = strategy.strip().lower()
    if normalized not in _VALID_STRATEGIES:
        choices = ", ".join(sorted(_VALID_STRATEGIES))
        raise ValueError(f"realtime_search_strategy must be one of: {choices}")
    return normalized


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return " ".join(filter(None, (_content_text(item) for item in content)))
    if isinstance(content, dict):
        for key in ("text", "content", "transcript"):
            if key in content:
                return _content_text(content[key])
        return ""
    for attribute in ("text", "content", "transcript"):
        value = getattr(content, attribute, None)
        if value is not None:
            return _content_text(value)
    return ""


def last_user_text(context: Any) -> str:
    """Return the newest user text without serializing the whole chat history."""

    for message in reversed(list(context.get_messages())):
        if isinstance(message, dict):
            role = str(message.get("role", ""))
            content = message.get("content", "")
        else:
            role = str(getattr(message, "role", ""))
            content = getattr(message, "content", "")
        if role.lower() == "user":
            return _content_text(content).strip()
    return ""


class RealtimeSearchPolicy:
    """Route only freshness-sensitive turns to DashScope web search.

    ``smart`` keeps ordinary chat byte-for-byte on the existing request path.
    ``auto`` enables provider-side search selection for every non-empty turn.
    ``always`` additionally forces a search.  ``off`` is the rollback default.
    """

    def __init__(self, mode: str = "off"):
        self._mode = normalize_realtime_search_mode(mode)

    @property
    def mode(self) -> str:
        return self._mode

    def decide(self, text: str) -> RealtimeSearchDecision:
        query = text.strip()
        if self._mode == "off" or not query:
            return RealtimeSearchDecision(False, False, "off_or_empty")
        if self._mode == "always":
            return RealtimeSearchDecision(True, True, "mode_always")
        if self._mode == "auto":
            return RealtimeSearchDecision(True, False, "provider_auto")
        if _EXPLICIT_SEARCH_RE.search(query):
            return RealtimeSearchDecision(True, True, "explicit_search")
        if _LOCAL_CONTEXT_RE.search(query):
            return RealtimeSearchDecision(False, False, "local_context")
        if _LIVE_SUBJECT_RE.search(query):
            if _STATIC_EXPLANATION_RE.search(query) and not _FRESHNESS_RE.search(query):
                return RealtimeSearchDecision(False, False, "static_explanation")
            return RealtimeSearchDecision(True, True, "live_subject")
        if _FRESHNESS_RE.search(query) and _FRESH_FACT_RE.search(query):
            return RealtimeSearchDecision(True, True, "fresh_fact")
        return RealtimeSearchDecision(False, False, "ordinary_chat")


def add_dashscope_search_params(
    params: dict[str, Any],
    decision: RealtimeSearchDecision,
    *,
    strategy: str = "turbo",
    current_time: datetime | None = None,
) -> dict[str, Any]:
    """Copy request parameters and add DashScope's non-standard search fields."""

    if not decision.enabled:
        return params

    result = dict(params)
    existing_body = result.get("extra_body")
    if existing_body is None:
        body: dict[str, Any] = {}
    elif isinstance(existing_body, dict):
        body = dict(existing_body)
    else:
        raise ValueError("extra_body must be a mapping when real-time search is enabled")

    body["enable_search"] = True
    existing_options = body.get("search_options")
    if existing_options is None:
        search_options: dict[str, Any] = {}
    elif isinstance(existing_options, dict):
        search_options = dict(existing_options)
    else:
        raise ValueError("extra_body.search_options must be a mapping")
    search_options["search_strategy"] = normalize_realtime_search_strategy(strategy)
    if decision.forced:
        search_options["forced_search"] = True
    body["search_options"] = search_options
    result["extra_body"] = body

    now = current_time or datetime.now(ZoneInfo("Asia/Shanghai"))
    if now.tzinfo is None:
        now = now.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    else:
        now = now.astimezone(ZoneInfo("Asia/Shanghai"))
    clock_instruction = (
        "实时检索时间基准：当前北京时间为"
        f"{now:%Y-%m-%d %H:%M}（UTC+08:00）。"
        "凡涉及今天、现在或最新，必须按这个时间检索和核对日期；"
        "第一句先给核心结论且不超过二十个汉字；如需补充，最多再加一句。"
        "不要输出Markdown、网址或引用编号。"
    )
    messages = list(result.get("messages") or ())
    if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
        first = dict(messages[0])
        first["content"] = f"{first.get('content', '')}\n{clock_instruction}".strip()
        messages[0] = first
    else:
        messages.insert(0, {"role": "system", "content": clock_instruction})
    result["messages"] = messages
    return result
