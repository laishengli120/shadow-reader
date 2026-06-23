from __future__ import annotations

import abc
import base64
import io
import json
import logging
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import NamedTuple
import urllib.request

import requests
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)
import asyncio
import tempfile

from pydub import AudioSegment

def _run_async(coro):
    """在当前线程中运行一个协程，兼容已有事件循环的情况。"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 如果在异步上下文中（如 pytest-asyncio），用 nest_asyncio
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        return loop.run_until_complete(coro)
    except RuntimeError:
        # 当前线程没有事件循环，新建一个
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
            asyncio.set_event_loop(None)
# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("shadow_reader")


# ═══════════════════════════════════════════════════════════════════
# 数据结构：VoiceOption
# ═══════════════════════════════════════════════════════════════════

class VoiceOption(NamedTuple):
    value: str        # 传给 API 的实际参数值
    label: str        # 前端展示名
    lang:  str        # "zh" | "en" | "zh/en" — 帮助前端按语言分组


# ═══════════════════════════════════════════════════════════════════
# 抽象基类
# ═══════════════════════════════════════════════════════════════════

class TTSProvider(abc.ABC):
    """所有 TTS 服务商的统一接口。子类实现 tts() 和 voices 属性。"""

    #: 该 provider 支持的音色列表
    voices: list[VoiceOption] = []
    supports_native_rate = False
    audio_format = "mp3"
    max_workers = 8

    @property
    def voice_options(self) -> list[VoiceOption]:
        return self.voices

    @property
    def allowed_voice_values(self) -> frozenset[str]:
        return frozenset(v.value for v in self.voice_options)

    @abc.abstractmethod
    def tts(self, text: str, voice: str, rate: float = 1.0) -> bytes:
        """合成单句，返回 audio_format 指定格式的字节。"""

    def validate_voice(self, voice: str) -> bool:
        return voice in self.allowed_voice_values


# ═══════════════════════════════════════════════════════════════════
# Provider 1: OpenAI
# ═══════════════════════════════════════════════════════════════════

class OpenAIProvider(TTSProvider):
    """
    官方 OpenAI TTS。
    凭证：api_key（sk-... 格式）
    文档：https://platform.openai.com/docs/api-reference/audio/createSpeech
    """

    voices = [
        VoiceOption("alloy",   "Alloy — 中性",          "en"),
        VoiceOption("echo",    "Echo — 男声",            "en"),
        VoiceOption("fable",   "Fable — 表现力强",       "en"),
        VoiceOption("nova",    "Nova — 女声",            "en"),
        VoiceOption("onyx",    "Onyx — 低沉男声",        "en"),
        VoiceOption("shimmer", "Shimmer — 清亮女声",     "en"),
    ]

    def __init__(self, api_key: str) -> None:
        self._client = OpenAI(api_key=api_key)

    def tts(self, text: str, voice: str, rate: float = 1.0) -> bytes:
        resp = self._client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
        )
        return resp.content


# ═══════════════════════════════════════════════════════════════════
# Provider 2: 硅基流动 (SiliconFlow)
# ═══════════════════════════════════════════════════════════════════

class SiliconFlowProvider(TTSProvider):
    """
    硅基流动 TTS，兼容 OpenAI 格式，只需替换 base_url。
    凭证：api_key（硅基流动控制台获取）
    文档：https://docs.siliconflow.cn/cn/api-reference/audio/speech
    音色前缀固定为 FunAudioLLM/CosyVoice2-0.5B:xxx
    """

    BASE_URL      = "https://api.siliconflow.cn/v1"
    DEFAULT_MODEL = "FunAudioLLM/CosyVoice2-0.5B"

    voices = [
        VoiceOption("FunAudioLLM/CosyVoice2-0.5B:alex",     "Alex — 男声 (英文)",     "en"),
        VoiceOption("FunAudioLLM/CosyVoice2-0.5B:anna",     "Anna — 女声 (英文)",     "en"),
        VoiceOption("FunAudioLLM/CosyVoice2-0.5B:bella",    "Bella — 女声 (英文)",    "en"),
        VoiceOption("FunAudioLLM/CosyVoice2-0.5B:benjamin", "Benjamin — 男声 (英文)", "en"),
        VoiceOption("FunAudioLLM/CosyVoice2-0.5B:claire",   "Claire — 女声 (中文)",   "zh"),
        VoiceOption("FunAudioLLM/CosyVoice2-0.5B:david",    "David — 男声 (中文)",    "zh"),
        VoiceOption("FunAudioLLM/CosyVoice2-0.5B:diana",    "Diana — 女声 (中英双语)", "zh/en"),
    ]

    def __init__(self, api_key: str) -> None:
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.BASE_URL,
        )

    def tts(self, text: str, voice: str, rate: float = 1.0) -> bytes:
        resp = self._client.audio.speech.create(
            model=self.DEFAULT_MODEL,
            voice=voice,
            input=text,
        )
        return resp.content


# ═══════════════════════════════════════════════════════════════════
# Provider 3: 阿里云 DashScope (千问 TTS)
# ═══════════════════════════════════════════════════════════════════

class DashScopeProvider(TTSProvider):
    """
    阿里云百炼 千问3-TTS-Flash。
    凭证：api_key（百炼控制台 sk-xxx 格式）
    文档：https://help.aliyun.com/zh/model-studio/qwen-tts
    调用方式：dashscope.MultiModalConversation.call()
              响应中 output.audio.url 是下载链接（24h 有效）
    依赖：pip install dashscope
    """

    MODEL = "qwen3-tts-flash"

    voices = [
        VoiceOption("Cherry",   "Cherry — 女声 (中文)",    "zh"),
        VoiceOption("Ethan",    "Ethan — 男声 (中文)",     "zh"),
        VoiceOption("Serena",   "Serena — 女声 (英文)",    "en"),
        VoiceOption("Dylan",    "Dylan — 男声 (英文)",     "en"),
        VoiceOption("Zhiyu",    "Zhiyu — 女声 (中英双语)", "zh/en"),
    ]

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def tts(self, text: str, voice: str, rate: float = 1.0) -> bytes:
        try:
            import dashscope
        except ImportError as exc:
            raise RuntimeError(
                "dashscope 未安装，请执行: pip install dashscope"
            ) from exc

        # 根据音色推断语言类型
        _lang_map = {
            "Cherry": "Chinese", "Ethan": "Chinese",
            "Serena": "English", "Dylan": "English",
            "Zhiyu":  "Chinese",
        }
        lang = _lang_map.get(voice, "Chinese")

        response = dashscope.MultiModalConversation.call(
            model=self.MODEL,
            api_key=self._api_key,
            text=text,
            voice=voice,
            language_type=lang,
            stream=False,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"DashScope 错误 {response.status_code}: {response.message}"
            )

        # 响应中包含音频 URL，需下载
        audio_url = (
            response.output
            .get("audio", {})
            .get("url")
        )
        if not audio_url:
            raise RuntimeError("DashScope 响应中未找到音频 URL")

        dl = requests.get(audio_url, timeout=30)
        dl.raise_for_status()
        return dl.content


# ═══════════════════════════════════════════════════════════════════
# Provider 4: 火山引擎 (ByteDance Volcengine)
# ═══════════════════════════════════════════════════════════════════

class VolcengineProvider(TTSProvider):
    """
    火山引擎豆包语音合成（HTTP 非流式接口）。
    凭证格式（JSON 字符串）：
        {"appid": "xxx", "token": "xxx", "cluster": "volcano_tts"}
    文档：https://www.volcengine.com/docs/6561/1096680
    鉴权：Authorization: Bearer;{token}
    音频在响应 JSON 的 data 字段（base64 MP3）
    """

    API_URL = "https://openspeech.bytedance.com/api/v1/tts"
    supports_native_rate = True

    voices = [
        VoiceOption("BV701_streaming", "通用女声",              "zh"),
        VoiceOption("BV700_streaming", "通用男声",              "zh"),
        VoiceOption("BV034_streaming", "知性女声",              "zh"),
        VoiceOption("BV102_streaming", "磁性男声",              "zh"),
        VoiceOption("BV001_streaming", "标准女声",              "zh"),
        VoiceOption("BV002_streaming", "标准男声",              "zh"),
        VoiceOption("BV503_streaming", "EMO情感女声",           "zh"),
        VoiceOption("BV056_streaming", "清澈女声（英文）",      "en"),
        VoiceOption("BV703_streaming", "活力男声（英文）",      "en"),
    ]

    def __init__(self, api_key: str) -> None:
        # api_key 是 JSON 字符串，包含 appid / token / cluster
        try:
            cred = json.loads(api_key)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "火山引擎凭证格式错误，需提供 JSON：\n"
                '{"appid": "xxx", "token": "xxx", "cluster": "volcano_tts"}'
            ) from exc

        self._appid   = cred.get("appid", "")
        self._token   = cred.get("token", "")
        self._cluster = cred.get("cluster", "volcano_tts")

        if not self._appid or not self._token:
            raise ValueError("火山引擎凭证缺少 appid 或 token")

    def tts(self, text: str, voice: str, rate: float = 1.0) -> bytes:
        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer;{self._token}",
        }
        body = {
            "app": {
                "appid":   self._appid,
                "token":   self._token,     # API 要求在 body 中也传递
                "cluster": self._cluster,
            },
            "user": {"uid": "shadow_reader"},
            "audio": {
                "voice_type":   voice,
                "encoding":     "mp3",
                "speed_ratio":  rate,
                "volume_ratio": 1.0,
                "pitch_ratio":  1.0,
            },
            "request": {
                "reqid":         str(uuid.uuid4()),
                "text":          text,
                "text_type":     "plain",
                "operation":     "query",
                "with_frontend": 1,
                "frontend_type": "unitTson",
            },
        }

        resp = requests.post(
            self.API_URL,
            json=body,
            headers=headers,
            timeout=30,
        )

        if resp.status_code != 200:
            raise RuntimeError(
                f"火山引擎 HTTP {resp.status_code}: {resp.text[:200]}"
            )

        data = resp.json()
        code = data.get("code")

        # 火山引擎业务错误码：3000 = 成功
        if code != 3000:
            msg = data.get("message", "未知错误")
            # 常见错误码映射
            _err = {
                40001: "文本为空或不含有效字符",
                40002: "文本超出长度限制",
                40101: "鉴权失败，请检查 appid 和 token",
                40103: "账户余额不足",
                50000: "服务端内部错误",
            }
            raise RuntimeError(
                f"火山引擎合成失败 [{code}]: {_err.get(code, msg)}"
            )

        audio_b64 = data.get("data")
        if not audio_b64:
            raise RuntimeError("火山引擎响应中未找到音频数据")

        return base64.b64decode(audio_b64)


# ═══════════════════════════════════════════════════════════════════
# Provider 5: Microsoft Azure TTS (REST API)
# ═══════════════════════════════════════════════════════════════════

class AzureProvider(TTSProvider):
    """
    Microsoft Azure 认知服务 TTS（REST 接口，无需 SDK）。
    凭证格式（JSON 字符串）：
        {"key": "xxx", "region": "eastus"}
    文档：https://learn.microsoft.com/azure/ai-services/speech-service/rest-text-to-speech
    请求体：SSML XML
    鉴权：Ocp-Apim-Subscription-Key header（直接用 key，无需 token 兑换）
    """

    voices = [
        VoiceOption("en-US-JennyNeural",        "Jenny — 美式英文女声",    "en"),
        VoiceOption("en-US-GuyNeural",           "Guy — 美式英文男声",     "en"),
        VoiceOption("en-US-AriaNeural",          "Aria — 美式英文女声",    "en"),
        VoiceOption("en-GB-SoniaNeural",         "Sonia — 英式英文女声",   "en"),
        VoiceOption("en-GB-RyanNeural",          "Ryan — 英式英文男声",    "en"),
        VoiceOption("zh-CN-XiaoxiaoNeural",      "晓晓 — 中文普通话女声", "zh"),
        VoiceOption("zh-CN-YunxiNeural",         "云希 — 中文普通话男声", "zh"),
        VoiceOption("zh-CN-XiaoyiNeural",        "晓伊 — 中文普通话女声", "zh"),
        VoiceOption("zh-TW-HsiaoChenNeural",     "曉臻 — 繁體中文女聲",  "zh"),
    ]
    supports_native_rate = True

    # SSML 模板：根据音色名推断语言
    _SSML = (
        "<speak version='1.0' xml:lang='{lang}'>"
        "<voice xml:lang='{lang}' name='{voice}'>"
        "<prosody rate='{rate_pct}'>{text}</prosody>"
        "</voice>"
        "</speak>"
    )

    def __init__(self, api_key: str) -> None:
        try:
            cred = json.loads(api_key)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Azure 凭证格式错误，需提供 JSON：\n"
                '{"key": "xxx", "region": "eastus"}'
            ) from exc

        self._key    = cred.get("key", "")
        self._region = cred.get("region", "eastus")

        if not self._key or not self._region:
            raise ValueError("Azure 凭证缺少 key 或 region")

    @staticmethod
    def _voice_to_lang(voice_name: str) -> str:
        """从音色名提取语言标签，例如 en-US-JennyNeural → en-US"""
        parts = voice_name.split("-")
        return f"{parts[0]}-{parts[1]}" if len(parts) >= 2 else "en-US"

    @staticmethod
    def _rate_to_ssml(rate: float) -> str:
        return f"{int(round((rate - 1.0) * 100)):+d}%"

    def tts(self, text: str, voice: str, rate: float = 1.0) -> bytes:
        lang = self._voice_to_lang(voice)
        ssml = self._SSML.format(
            lang=lang,
            voice=voice,
            rate_pct=self._rate_to_ssml(rate),
            text=text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"),
        )

        url = (
            f"https://{self._region}.tts.speech.microsoft.com"
            "/cognitiveservices/v1"
        )
        headers = {
            "Ocp-Apim-Subscription-Key": self._key,
            "Content-Type":              "application/ssml+xml",
            "X-Microsoft-OutputFormat":  "audio-24khz-48kbitrate-mono-mp3",
            "User-Agent":                "ShadowReader/1.0",
        }

        resp = requests.post(url, data=ssml.encode("utf-8"), headers=headers, timeout=30)

        if resp.status_code == 401:
            raise RuntimeError("Azure 鉴权失败，请检查 key 和 region。")
        if resp.status_code == 400:
            raise RuntimeError(f"Azure 请求格式错误：{resp.text[:200]}")
        if resp.status_code != 200:
            raise RuntimeError(
                f"Azure HTTP {resp.status_code}: {resp.text[:200]}"
            )

        return resp.content
# ═══════════════════════════════════════════════════════════════════
# Provider: edge-tts（推荐）
# 调用微软 Edge 在线 TTS，Neural 级别音质，免费，无需任何 Key
# ═══════════════════════════════════════════════════════════════════

class EdgeTTSProvider(TTSProvider):
    """
    微软 Edge TTS（非官方，调用 Edge 浏览器同款服务）
    - 无需 API Key
    - Neural 神经网络语音，音质接近付费服务
    - 需要联网
    - 依赖: pip install edge-tts

    注意：edge-tts 是异步库，在同步 Flask 中需用 _run_async() 桥接。
    """
    def __init__(self, api_key: str = "") -> None:
        pass  # 免费服务商不需要凭证
    supports_native_rate = True
    # 精选常用中英文音色（完整列表有 400+ 个，可用 edge-tts --list-voices 查看）
    voices = [
        # 中文
        VoiceOption("zh-CN-XiaoxiaoNeural",  "晓晓 — 中文女声（亲切）",   "zh"),
        VoiceOption("zh-CN-YunxiNeural",     "云希 — 中文男声",           "zh"),
        VoiceOption("zh-CN-XiaoyiNeural",    "晓伊 — 中文女声（活泼）",   "zh"),
        VoiceOption("zh-CN-YunjianNeural",   "云健 — 中文男声（运动）",   "zh"),
        VoiceOption("zh-CN-XiaochenNeural",  "晓辰 — 中文女声（自然）",   "zh"),
        VoiceOption("zh-TW-HsiaoChenNeural", "曉臻 — 繁體中文女聲",      "zh"),
        VoiceOption("zh-TW-YunJheNeural",    "雲哲 — 繁體中文男聲",      "zh"),
        # 英文（美式）
        VoiceOption("en-US-JennyNeural",     "Jenny — 美式英文女声",      "en"),
        VoiceOption("en-US-GuyNeural",       "Guy — 美式英文男声",        "en"),
        VoiceOption("en-US-AriaNeural",      "Aria — 美式英文女声",       "en"),
        VoiceOption("en-US-DavisNeural",     "Davis — 美式英文男声",      "en"),
        VoiceOption("en-US-TonyNeural",      "Tony — 美式英文男声（活力）","en"),
        # 英文（英式）
        VoiceOption("en-GB-SoniaNeural",     "Sonia — 英式英文女声",      "en"),
        VoiceOption("en-GB-RyanNeural",      "Ryan — 英式英文男声",       "en"),
        VoiceOption("en-GB-LibbyNeural",     "Libby — 英式英文女声",      "en"),
        # 英文（澳式）
        VoiceOption("en-AU-NatashaNeural",   "Natasha — 澳式英文女声",    "en"),
        VoiceOption("en-AU-WilliamNeural",   "William — 澳式英文男声",    "en"),
    ]

    @staticmethod
    def _rate_to_edge(rate: float) -> str:
        return f"{int(round((rate - 1.0) * 100)):+d}%"

    def tts(self, text: str, voice: str, rate: float = 1.0) -> bytes:
        try:
            import edge_tts
        except ImportError as exc:
            raise RuntimeError(
                "edge-tts 未安装，请执行: pip install edge-tts"
            ) from exc

        async def _synthesize() -> bytes:
            try:
                communicate = edge_tts.Communicate(
                    text=text,
                    voice=voice,
                    rate=self._rate_to_edge(rate),
                )
            except TypeError:
                communicate = edge_tts.Communicate(text=text, voice=voice)
            chunks: list[bytes] = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    chunks.append(chunk["data"])
            if not chunks:
                raise RuntimeError("edge-tts 未返回任何音频数据")
            return b"".join(chunks)

        try:
            return _run_async(_synthesize())
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(str(exc)) from exc


# ═══════════════════════════════════════════════════════════════════
# Provider: gTTS（Google Text-to-Speech）
# ═══════════════════════════════════════════════════════════════════

class GTTSProvider(TTSProvider):
    """
    Google Translate TTS（非官方接口）
    - 无需 API Key
    - 音质中等，语调自然
    - 需要联网
    - 无法选择不同音色（每种语言只有一个声音）
    - 仅适合个人/学习用途
    - 依赖: pip install gtts
    """
    def __init__(self, api_key: str = "") -> None:
        self._chunk_workers = _clamp_int(
            os.environ.get("GTTS_CHUNK_WORKERS"),
            default=4,
            low=1,
            high=16,
        )
        self._timeout = _clamp_float(
            os.environ.get("GTTS_TIMEOUT"),
            default=10.0,
            low=1.0,
            high=60.0,
        )

    max_workers = 12

    # gTTS 没有多音色概念，用「语言」模拟音色选择
    voices = [
        VoiceOption("en",    "English (默认)",         "en"),
        VoiceOption("en-gb", "English — British",      "en"),
        VoiceOption("en-au", "English — Australian",   "en"),
        VoiceOption("en-ca", "English — Canadian",     "en"),
        VoiceOption("en-in", "English — Indian",       "en"),
        VoiceOption("zh-CN", "中文（普通话）",           "zh"),
        VoiceOption("zh-TW", "中文（繁體）",             "zh"),
        VoiceOption("yue",   "粤语",                   "zh"),
    ]

    # gTTS 的 lang 参数与上面 value 的映射（部分 value 需要拆分）
    _LANG_MAP = {
        "en":    ("en", None),
        "en-gb": ("en", "co.uk"),
        "en-au": ("en", "com.au"),
        "en-ca": ("en", "ca"),
        "en-in": ("en", "co.in"),
        "zh-CN": ("zh-CN", None),
        "zh-TW": ("zh-TW", None),
        "yue":   ("yue", None),
    }

    @staticmethod
    def _send_prepared_request(tts_obj, prepared_request) -> bytes:
        from gtts.tts import gTTSError

        try:
            requests.packages.urllib3.disable_warnings(
                requests.packages.urllib3.exceptions.InsecureRequestWarning
            )
        except Exception:
            pass

        try:
            with requests.Session() as session:
                response = session.send(
                    request=prepared_request,
                    verify=False,
                    proxies=urllib.request.getproxies(),
                    timeout=tts_obj.timeout,
                )
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            raise gTTSError(tts=tts_obj, response=response) from exc
        except requests.exceptions.RequestException as exc:
            raise gTTSError(tts=tts_obj) from exc

        chunks: list[bytes] = []
        for line in response.iter_lines(chunk_size=1024):
            decoded_line = line.decode("utf-8")
            if "jQ1olc" not in decoded_line:
                continue
            audio_search = re.search(r'jQ1olc","\[\\"(.*)\\"]', decoded_line)
            if not audio_search:
                raise gTTSError(tts=tts_obj, response=response)
            chunks.append(base64.b64decode(audio_search.group(1).encode("ascii")))

        if not chunks:
            raise gTTSError(tts=tts_obj, response=response)
        return b"".join(chunks)

    def tts(self, text: str, voice: str, rate: float = 1.0) -> bytes:
        try:
            from gtts import gTTS
        except ImportError as exc:
            raise RuntimeError(
                "gTTS 未安装，请执行: pip install gtts"
            ) from exc

        lang, tld = self._LANG_MAP.get(voice, ("en", None))
        kwargs = {
            "text": text,
            "lang": lang,
            "slow": False,
            "timeout": self._timeout,
        }
        if tld:
            kwargs["tld"] = tld

        tts_obj = gTTS(**kwargs)
        prepared_requests = tts_obj._prepare_requests()
        if not prepared_requests:
            raise RuntimeError("gTTS 未生成有效请求。")

        try:
            worker_count = max(1, min(len(prepared_requests), self._chunk_workers))
            results: list[bytes | None] = [None] * len(prepared_requests)
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                futures = {
                    pool.submit(
                        self._send_prepared_request,
                        tts_obj,
                        prepared_request,
                    ): index
                    for index, prepared_request in enumerate(prepared_requests)
                }
                for future in as_completed(futures):
                    results[futures[future]] = future.result()
        except Exception as exc:
            raise RuntimeError(str(exc)) from exc

        return b"".join(result for result in results if result is not None)


# ═══════════════════════════════════════════════════════════════════
# Provider: pyttsx3（完全离线）
# ═══════════════════════════════════════════════════════════════════

class Pyttsx3Provider(TTSProvider):
    """
    pyttsx3 离线 TTS，调用系统自带语音引擎
    - 完全离线，无需网络
    - 音质较差（系统级 TTS）
    - 音色取决于操作系统已安装的语音包
    - Windows: SAPI5（自带多种语音）
    - macOS: NSSpeechSynthesizer（自带 Alex、Samantha 等）
    - Linux: espeak-ng（需手动安装: sudo apt install espeak espeak-ng）
    - 依赖: pip install pyttsx3

    实现说明：
    pyttsx3 的 save_to_file() 只支持写到文件路径，不支持 BytesIO。
    这里用 tempfile 写临时文件，读回后删除，避免磁盘残留。
    """
    def __init__(self, api_key: str = "") -> None:
        pass  # 免费服务商不需要凭证
    supports_native_rate = True
    max_workers = 1

    # 音色列表在运行时从系统动态获取，这里提供通用默认值
    # 用户可通过 /providers 接口查看实际可用音色
    voices = [
        VoiceOption("__default__",  "系统默认音色",        "zh/en"),
        VoiceOption("__female__",   "系统女声（如可用）",   "zh/en"),
        VoiceOption("__male__",     "系统男声（如可用）",   "zh/en"),
    ]

    @classmethod
    def get_system_voices(cls) -> list[VoiceOption]:
        """
        动态获取当前系统已安装的语音列表。
        供 /providers 接口调用，返回真实可用音色。
        """
        try:
            import pyttsx3
            engine = pyttsx3.init()
            sys_voices = engine.getProperty("voices")
            engine.stop()
            result = []
            for v in sys_voices:
                name  = v.name or v.id
                # 简单判断语言
                vid   = v.id.lower()
                lang  = "zh" if ("zh" in vid or "chinese" in vid or "mandarin" in vid) \
                        else "en"
                result.append(VoiceOption(v.id, name, lang))
            return result if result else cls.voices
        except Exception:
            return cls.voices

    @property
    def voice_options(self) -> list[VoiceOption]:
        values: dict[str, VoiceOption] = {}
        for voice in self.get_system_voices() + self.voices:
            values[voice.value] = voice
        return list(values.values())

    def _resolve_voice_id(self, voice: str) -> str | None:
        """将虚拟音色名（__female__ 等）解析为系统实际的 voice.id。"""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            sys_voices = engine.getProperty("voices") or []
            engine.stop()
        except Exception:
            return None

        if voice == "__default__" or not sys_voices:
            return None  # 不设置，使用系统默认

        if voice == "__female__":
            for v in sys_voices:
                if "female" in (v.gender or "").lower() \
                        or "zira" in v.id.lower() \
                        or "samantha" in v.id.lower():
                    return v.id
            return sys_voices[0].id

        if voice == "__male__":
            for v in sys_voices:
                if "male" in (v.gender or "").lower() \
                        or "david" in v.id.lower() \
                        or "alex" in v.id.lower():
                    return v.id
            return sys_voices[-1].id

        # voice 是真实系统 id（从 get_system_voices() 获取的）
        return voice

    def tts(self, text: str, voice: str, rate: float = 1.0) -> bytes:
        try:
            import pyttsx3
        except ImportError as exc:
            raise RuntimeError(
                "pyttsx3 未安装，请执行: pip install pyttsx3\n"
                "Linux 还需要: sudo apt install espeak espeak-ng"
            ) from exc

        voice_id = self._resolve_voice_id(voice)

        # pyttsx3 不支持 BytesIO，必须写到临时文件
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            engine = pyttsx3.init()
            if voice_id:
                engine.setProperty("voice", voice_id)
            engine.setProperty("rate", int(150 * rate))    # 语速（wpm），默认 200 偏快
            engine.setProperty("volume", 1.0)
            engine.save_to_file(text, tmp_path)
            engine.runAndWait()
            engine.stop()

            # 读取临时文件并用 pydub 标准化为 MP3
            audio = AudioSegment.from_file(tmp_path)
            buf = io.BytesIO()
            audio.export(buf, format="mp3")
            return buf.getvalue()

        finally:
            # 无论成功与否都删除临时文件
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ═══════════════════════════════════════════════════════════════════
# Provider Registry
# ═══════════════════════════════════════════════════════════════════

class ProviderRegistry:
    """
    根据 provider_name + api_key 实例化对应的 TTSProvider。
    每个请求独立实例化，避免 key 跨请求泄漏。
    """

    _factories: dict[str, type[TTSProvider]] = {
        "openai":       OpenAIProvider,
        "siliconflow":  SiliconFlowProvider,
        "dashscope":    DashScopeProvider,
        "volcengine":   VolcengineProvider,
        "azure":        AzureProvider,
        "edge-tts":     EdgeTTSProvider,    # 推荐，Neural 级别，免费
        "gtts":         GTTSProvider,       # Google，免费，音色单一
        "pyttsx3":      Pyttsx3Provider,    # 完全离线，音质较差
    }

    @classmethod
    def get(cls, name: str, api_key: str) -> TTSProvider:
        factory = cls._factories.get(name)
        if factory is None:
            raise ValueError(
                f"未知服务商 '{name}'，可选: {sorted(cls._factories)}"
            )
        return factory(api_key)

    @classmethod
    def names(cls) -> list[str]:
        return sorted(cls._factories.keys())

    @classmethod
    def voices_map(cls) -> dict:
        """返回所有服务商的音色信息，供 /providers 接口使用。"""
        result = {}
        for name, factory in cls._factories.items():
            result[name] = [
                {"value": v.value, "label": v.label, "lang": v.lang}
                for v in factory.voices
            ]
        return result


# ═══════════════════════════════════════════════════════════════════
# Flask 应用
# ═══════════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app, resources={r"/generate": {"origins": os.environ.get("CORS_ORIGINS", "*")}})

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["10 per minute"],
    storage_uri="memory://",
)

MAX_SENTENCES = 50
MAX_INTERVAL  = 10.0
MIN_INTERVAL  = 0.0
MAX_SPEECH_RATE = 1.5
MIN_SPEECH_RATE = 0.6
DEFAULT_SPEECH_RATE = 1.0

_CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
_LATIN_RE = re.compile(r"[A-Za-z]")


@contextmanager
def _bytes_io():
    buf = io.BytesIO()
    try:
        yield buf
    finally:
        buf.close()


def _clamp_float(value, default: float, low: float, high: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


def _clamp_int(value, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


def _char_language(char: str) -> str | None:
    if _CJK_RE.match(char):
        return "zh"
    if _LATIN_RE.match(char):
        return "en"
    return None


def _split_text_by_language(text: str) -> list[tuple[str, str]]:
    """
    Split a line into zh/en runs while keeping neutral punctuation near the
    preceding spoken run. Timing still stays at the original line level.
    """
    parts: list[tuple[str, str]] = []
    current_lang: str | None = None
    buf: list[str] = []
    prefix: list[str] = []

    for char in text:
        char_lang = _char_language(char)
        if char_lang is None:
            if current_lang is None:
                prefix.append(char)
            else:
                buf.append(char)
            continue

        if current_lang is None:
            current_lang = char_lang
            buf.extend(prefix)
            prefix.clear()
            buf.append(char)
            continue

        if char_lang == current_lang:
            buf.append(char)
            continue

        parts.append(("".join(buf), current_lang))
        current_lang = char_lang
        buf = [char]

    if current_lang is None:
        spoken = "".join(prefix).strip()
        return [(spoken, "en")] if spoken else []

    parts.append(("".join(buf), current_lang))
    return [(part, part_lang) for part, part_lang in parts if part.strip()]


def _default_voice_for_lang(
    voice_options: list[VoiceOption],
    target_lang: str,
    fallback: str | None = None,
) -> str:
    valid = {voice.value for voice in voice_options}
    if fallback and fallback in valid:
        return fallback

    for voice in voice_options:
        if voice.lang == target_lang or voice.lang == "zh/en":
            return voice.value

    return voice_options[0].value if voice_options else ""


def _adjust_audio_rate(segment: AudioSegment, rate: float) -> AudioSegment:
    if abs(rate - 1.0) < 0.01:
        return segment

    with _bytes_io() as out:
        segment.export(
            out,
            format="mp3",
            parameters=["-filter:a", f"atempo={rate:.3f}"],
        )
        out.seek(0)
        return AudioSegment.from_file(out, format="mp3")


def _audio_bytes_to_segment(
    audio_bytes: bytes,
    audio_format: str = "mp3",
) -> AudioSegment:
    with _bytes_io() as fp:
        fp.write(audio_bytes)
        fp.seek(0)
        return AudioSegment.from_file(fp, format=audio_format)


class _SynthesisTask(NamedTuple):
    line_index: int
    part_index: int
    text: str
    voice: str


class _SynthesisResult(NamedTuple):
    line_index: int
    part_index: int
    segment: AudioSegment
    tts_seconds: float
    decode_seconds: float


class _SynthesisMetrics(NamedTuple):
    worker_count: int
    part_count: int
    tts_wall_seconds: float
    tts_call_seconds: float
    decode_seconds: float
    audio_processing_seconds: float


def _synthesize_part(
    provider: TTSProvider,
    task: _SynthesisTask,
    speech_rate: float,
) -> _SynthesisResult:
    tts_started = time.perf_counter()
    audio_bytes = provider.tts(task.text, task.voice, speech_rate)
    tts_seconds = time.perf_counter() - tts_started

    decode_started = time.perf_counter()
    segment = _audio_bytes_to_segment(audio_bytes, provider.audio_format)
    decode_seconds = time.perf_counter() - decode_started

    return _SynthesisResult(
        task.line_index,
        task.part_index,
        segment,
        tts_seconds,
        decode_seconds,
    )


def _synthesize_lines(
    provider: TTSProvider,
    lines: list[str],
    voice_zh: str,
    voice_en: str,
    speech_rate: float,
) -> tuple[list[AudioSegment], _SynthesisMetrics]:
    """Synthesize all language runs globally, then rebuild lines in order."""
    tasks: list[_SynthesisTask] = []
    part_counts: list[int] = []

    for line_index, line in enumerate(lines):
        parts = _split_text_by_language(line)
        part_counts.append(len(parts))
        for part_index, (part, part_lang) in enumerate(parts):
            voice = voice_zh if part_lang == "zh" else voice_en
            tasks.append(_SynthesisTask(line_index, part_index, part, voice))

    worker_count = max(1, min(len(tasks), provider.max_workers))
    parts_by_line: list[list[AudioSegment | None]] = [
        [None] * count for count in part_counts
    ]
    results: list[_SynthesisResult] = []

    tts_wall_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = [
            pool.submit(_synthesize_part, provider, task, speech_rate)
            for task in tasks
        ]
        for future in as_completed(futures):
            result = future.result()
            parts_by_line[result.line_index][result.part_index] = result.segment
            results.append(result)
    tts_wall_seconds = time.perf_counter() - tts_wall_started

    processing_started = time.perf_counter()
    line_segments: list[AudioSegment] = []
    for line_parts in parts_by_line:
        combined = AudioSegment.empty()
        for part in line_parts:
            if part is None:
                raise RuntimeError("语音片段生成不完整。")
            combined += part

        if not provider.supports_native_rate:
            combined = _adjust_audio_rate(combined, speech_rate)
        line_segments.append(combined)
    audio_processing_seconds = time.perf_counter() - processing_started

    return line_segments, _SynthesisMetrics(
        worker_count=worker_count,
        part_count=len(tasks),
        tts_wall_seconds=tts_wall_seconds,
        tts_call_seconds=sum(result.tts_seconds for result in results),
        decode_seconds=sum(result.decode_seconds for result in results),
        audio_processing_seconds=audio_processing_seconds,
    )


def _openai_error_response(exc: APIStatusError) -> tuple[dict, int]:
    if isinstance(exc, AuthenticationError):
        return {"error": "API Key 无效或已过期，请检查设置。"}, 401
    if isinstance(exc, RateLimitError):
        return {"error": "API 配额已用尽或请求过于频繁，请稍后重试。"}, 429
    status = getattr(exc, "status_code", 500)
    return {"error": f"服务商返回错误（HTTP {status}），请稍后重试。"}, 502


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/providers", methods=["GET"])
def list_providers():
    """
    返回可用服务商列表及音色，前端用此动态渲染下拉菜单。
    同时返回各 provider 需要的凭证格式说明，方便前端展示提示。
    """
    credential_hints = {
        "openai":      {"format": "string",  "placeholder": "sk-...",
                        "hint": "OpenAI API Key"},
        "siliconflow": {"format": "string",  "placeholder": "sk-...",
                        "hint": "硅基流动 API Key"},
        "dashscope":   {"format": "string",  "placeholder": "sk-...",
                        "hint": "阿里云百炼 API Key"},
        "volcengine":  {"format": "json",
                        "placeholder": '{"appid":"xxx","token":"xxx","cluster":"volcano_tts"}',
                        "hint": "火山引擎凭证 JSON，包含 appid、token、cluster"},
        "azure":       {"format": "json",
                        "placeholder": '{"key":"xxx","region":"eastus"}',
                        "hint": "Azure 凭证 JSON，包含 key 和 region"},
        "edge-tts":    {"format": "none", "placeholder": "",
                 "hint": "无需 API Key，调用微软 Edge TTS 服务"},
        "gtts":        {"format": "none", "placeholder": "",
                 "hint": "无需 API Key，调用 Google Translate TTS"},
        "pyttsx3":     {"format": "none", "placeholder": "",
                 "hint": "完全离线，调用系统自带语音引擎"},
    }
    voices_map = ProviderRegistry.voices_map()
    return jsonify({
        "providers": ProviderRegistry.names(),
        "voices":    voices_map,
        "credential_hints": credential_hints,
    })


FREE_PROVIDERS = {"edge-tts", "gtts", "pyttsx3"}

def register_free_routes(app):
    """将免费服务商相关路由注册到 Flask app。"""
 
    @app.route("/system-voices", methods=["GET"])
    def system_voices():
        """
        返回 pyttsx3 在当前系统上实际可用的语音列表。
        前端在用户选择 pyttsx3 时调用此接口动态填充音色下拉菜单。
        """
        voices = Pyttsx3Provider.get_system_voices()
        return jsonify({
            "voices": [
                {"value": v.value, "label": v.label, "lang": v.lang}
                for v in voices
            ]
        })

register_free_routes(app)

@app.route("/generate", methods=["POST"])
@limiter.limit("10 per minute")
def generate_audio():
    request_started = time.perf_counter()
    # ── 1. 解析与校验 ─────────────────────────────────────────────
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "请求体必须是 JSON 格式。"}), 400

    api_key       = (data.get("api_key")  or "").strip()
    text          = (data.get("text")     or "").strip()
    provider_name = (data.get("provider") or "openai").strip().lower()
    voice         = (data.get("voice")    or "").strip()
    voice_zh      = (data.get("voice_zh") or "").strip()
    voice_en      = (data.get("voice_en") or "").strip()
 
    interval = _clamp_float(data.get("interval", 2.0), 2.0, MIN_INTERVAL, MAX_INTERVAL)
    speech_rate = _clamp_float(
        data.get("speech_rate", DEFAULT_SPEECH_RATE),
        DEFAULT_SPEECH_RATE,
        MIN_SPEECH_RATE,
        MAX_SPEECH_RATE,
    )
 
    if provider_name not in ProviderRegistry.names():
        return jsonify({
            "error": f"不支持的服务商 '{provider_name}'，可选: {ProviderRegistry.names()}"
        }), 400
 
    # 免费服务商不需要 API Key 校验
    is_free = provider_name in FREE_PROVIDERS
    if not is_free and len(api_key) < 5:
        return jsonify({"error": "请提供有效的 API 凭证。"}), 400
 
    try:
        # 免费服务商传空字符串，Provider.__init__ 不使用 api_key 参数
        provider = ProviderRegistry.get(provider_name, api_key)
    except (ValueError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 400

    voice_options = provider.voice_options
    valid_values = {v.value for v in voice_options}
    requested_voices = {
        "voice": voice,
        "voice_zh": voice_zh,
        "voice_en": voice_en,
    }
    invalid_voices = {
        key: value
        for key, value in requested_voices.items()
        if value and value not in valid_values
    }
    if invalid_voices:
        valid = [v.value for v in voice_options]
        return jsonify({
            "error": f"不支持的音色 {invalid_voices}，{provider_name} 可用音色: {valid}"
        }), 400

    voice_zh = _default_voice_for_lang(
        voice_options,
        "zh",
        fallback=voice_zh or voice or voice_en,
    )
    voice_en = _default_voice_for_lang(
        voice_options,
        "en",
        fallback=voice_en or voice or voice_zh,
    )
    if not voice_zh or not voice_en:
        return jsonify({"error": f"{provider_name} 暂无可用音色。"}), 400

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return jsonify({"error": "文本内容不能为空。"}), 400
    if len(lines) > MAX_SENTENCES:
        return jsonify({
            "error": f"句子数超出上限（{MAX_SENTENCES} 句），请分批处理。"
        }), 400

    # ── 2. 并发 TTS ──────────────────────────────────────────────
    try:
        line_segments, synthesis_metrics = _synthesize_lines(
            provider,
            lines,
            voice_zh,
            voice_en,
            speech_rate,
        )

    except AuthenticationError as exc:
        body, code = _openai_error_response(exc)
        return jsonify(body), code
    except RateLimitError as exc:
        body, code = _openai_error_response(exc)
        return jsonify(body), code
    except APIStatusError as exc:
        logger.error("OpenAI API error: status=%s", exc.status_code)
        body, code = _openai_error_response(exc)
        return jsonify(body), code
    except APIConnectionError:
        logger.error("Cannot reach provider: %s", provider_name)
        return jsonify({"error": f"无法连接 {provider_name}，请检查网络。"}), 503
    except RuntimeError as exc:
        # 火山引擎 / Azure / DashScope 的业务错误
        logger.error("Provider error [%s]: %s", provider_name, exc)
        return jsonify({"error": str(exc)}), 502
    except Exception as exc:
        logger.exception("Unexpected TTS error: provider=%s", provider_name)
        return jsonify({"error": str(exc)}), 500

    # ── 3. 拼接音频 ───────────────────────────────────────────────
    final_assembly_started = time.perf_counter()
    silence  = AudioSegment.silent(duration=int(interval * 1000))
    combined = AudioSegment.empty()
    timings: list[dict] = []
    current_ms = 0

    for idx, (line, segment) in enumerate(zip(lines, line_segments)):
        duration_ms = len(segment)
        timings.append({
            "index":      idx,
            "text":       line,
            "start_time": current_ms / 1000.0,
            "end_time":   (current_ms + duration_ms) / 1000.0,
        })
        combined   += segment
        current_ms += duration_ms

        if idx < len(lines) - 1:
            combined   += silence
            current_ms += len(silence)

    final_assembly_seconds = time.perf_counter() - final_assembly_started
    audio_processing_seconds = (
        synthesis_metrics.audio_processing_seconds + final_assembly_seconds
    )

    # ── 4. 导出并返回 ────────────────────────────────────────────
    encoding_started = time.perf_counter()
    with _bytes_io() as out:
        combined.export(out, format="mp3")
        final_audio = out.getvalue()
    encoding_seconds = time.perf_counter() - encoding_started

    base64_started = time.perf_counter()
    audio_b64 = base64.b64encode(final_audio).decode()
    base64_seconds = time.perf_counter() - base64_started
    total_seconds = time.perf_counter() - request_started

    logger.info(
        "Generated: provider=%s sentences=%d parts=%d workers=%d "
        "interval=%.1fs rate=%.1fx duration=%.1fs "
        "tts_wall=%.3fs tts_calls_total=%.3fs decode_total=%.3fs "
        "audio_processing=%.3fs encode=%.3fs base64=%.3fs total=%.3fs",
        provider_name,
        len(lines),
        synthesis_metrics.part_count,
        synthesis_metrics.worker_count,
        interval,
        speech_rate,
        current_ms / 1000.0,
        synthesis_metrics.tts_wall_seconds,
        synthesis_metrics.tts_call_seconds,
        synthesis_metrics.decode_seconds,
        audio_processing_seconds,
        encoding_seconds,
        base64_seconds,
        total_seconds,
    )
    return jsonify({"audio_base64": audio_b64, "timings": timings})


# ─────────────────────────────────────────────
# Error handlers
# ─────────────────────────────────────────────

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": "请求过于频繁，请稍后再试。"}), 429

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "接口不存在。"}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "不支持该 HTTP 方法。"}), 405


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true",
    )
