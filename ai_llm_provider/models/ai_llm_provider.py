from __future__ import annotations

import ipaddress
import json
import logging
import math
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

import requests as http_requests

from odoo import _, api, fields, models
from odoo.exceptions import UserError, ValidationError

# SSRF: block private/reserved hosts (except ollama which runs locally).
_SSRF_BLOCKED_HOSTS = {"metadata.google.internal"}
_MAX_STREAM_DURATION = 300  # 5 minutes max for any streaming response

_logger = logging.getLogger(__name__)

PROVIDER_SELECTION = [
    ("openai", "OpenAI"),
    ("anthropic", "Anthropic"),
    ("google", "Google Gemini"),
    ("openrouter", "OpenRouter"),
    ("ollama", "Ollama (Local)"),
    ("groq", "Groq"),
    ("mistral", "Mistral"),
    ("elevenlabs", "ElevenLabs"),
    ("custom", "Custom"),
]


class AiLlmProvider(models.Model):
    _name = "ai.llm.provider"
    _description = "AI LLM Provider"
    _order = "sequence, id"

    name = fields.Char(string="Name", required=True)
    sequence = fields.Integer(string="Sequence", default=10)
    code = fields.Selection(
        selection=PROVIDER_SELECTION,
        string="Provider",
        required=True,
    )
    base_url = fields.Char(string="Base URL", required=True)
    api_key = fields.Char(string="API Key", groups="base.group_system")
    is_active = fields.Boolean(string="Active", default=True)
    custom_headers = fields.Text(
        string="Custom Headers (JSON)",
        groups="base.group_system",
        help="Additional HTTP headers as JSON object, e.g. {\"X-Custom\": \"value\"}",
    )
    model_ids = fields.One2many(
        comodel_name="ai.llm.model",
        inverse_name="provider_id",
        string="Models",
    )

    @api.constrains("base_url", "code")
    def _check_base_url_format(self):
        for record in self:
            url = (record.base_url or "").strip()
            if not url:
                continue
            if not url.startswith(("http://", "https://")):
                raise ValidationError(
                    _("Base URL must start with http:// or https://. Got: %s") % url
                )
            # SSRF protection: block private IPs for non-local providers
            if record.code in ("ollama",):
                continue
            parsed = urlparse(url)
            hostname = (parsed.hostname or "").lower()
            if hostname in _SSRF_BLOCKED_HOSTS:
                raise ValidationError(
                    _("Base URL '%s' points to a blocked internal host.") % url
                )
            if hostname in ("localhost", "127.0.0.1", "::1") or hostname.endswith(".local"):
                raise ValidationError(
                    _("Local URLs are only allowed for Ollama providers. "
                      "Got: %s") % url
                )
            try:
                addr = ipaddress.ip_address(hostname)
                if addr.is_private or addr.is_loopback or addr.is_link_local:
                    raise ValidationError(
                        _("Private/internal IP addresses are not allowed "
                          "(SSRF risk). Got: %s") % url
                    )
            except ValueError:
                pass  # Not an IP literal — hostname is OK

    def _assert_api_key(self):
        """Raise UserError if API key is required but missing (skips Ollama)."""
        self.ensure_one()
        provider = self.sudo()
        if not provider.api_key and provider.code not in ("ollama",):
            raise UserError(
                _("API key is required for provider '%s'. "
                  "Configure it under Settings → AI → LLM Providers.")
                % self.name
            )

    def _get_headers(self):
        """Build HTTP headers for API requests."""
        self.ensure_one()
        # Read as sudo — API keys are admin-only fields.
        provider = self.sudo()
        headers = {
            "Content-Type": "application/json",
        }
        if provider.api_key:
            is_elevenlabs = (
                provider.code == "elevenlabs"
                or "elevenlabs.io" in (provider.base_url or "")
            )
            if is_elevenlabs:
                headers["xi-api-key"] = provider.api_key
            else:
                headers["Authorization"] = f"Bearer {provider.api_key}"
        if provider.custom_headers:
            try:
                custom = json.loads(provider.custom_headers)
                if isinstance(custom, dict):
                    if "Authorization" in custom and "Authorization" in headers:
                        _logger.info(
                            "[LLM] Provider %s custom_headers overrides Authorization header",
                            self.name,
                        )
                    headers.update(custom)
            except (json.JSONDecodeError, TypeError):
                _logger.warning(
                    "[LLM] Provider %s has invalid custom_headers JSON", self.name
                )
        return headers

    def _build_chat_completion_payload(
        self,
        model_code,
        messages,
        tools=None,
        tool_choice="auto",
        stream=False,
        extra_payload=None,
    ):
        payload = {
            "model": model_code,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        if stream:
            payload["stream"] = True
            # OpenAI-compatible providers may include usage in the final stream chunk.
            payload["stream_options"] = {"include_usage": True}
        if isinstance(extra_payload, dict):
            for key, value in extra_payload.items():
                if key in ("model", "messages", "tools", "tool_choice", "stream"):
                    continue
                payload[key] = value
        return payload

    def _extract_http_error_body(self, response):
        body = ""
        if response is None:
            return body
        try:
            parsed = response.json()
            if isinstance(parsed, dict):
                error_data = parsed.get("error")
                if isinstance(error_data, dict):
                    body = error_data.get("message", "") or json.dumps(error_data)[:500]
                elif isinstance(error_data, str):
                    body = error_data
            if not body:
                body = response.text[:500]
        except Exception:
            body = response.text[:500]
        return body

    def _parse_json_response(self, response, *, operation):
        try:
            return response.json()
        except ValueError as exc:
            body = (getattr(response, "text", "") or "")[:500]
            _logger.error(
                "[LLM] %s returned invalid JSON for %s: %s",
                self.name,
                operation,
                body or exc,
            )
            raise UserError(
                f"{self.name} returned an invalid JSON response for {operation}."
            )

    def _retryable_status_code(self, status_code):
        return int(status_code or 0) in {408, 409, 425, 429, 500, 502, 503, 504}

    def _parse_retry_after(self, response):
        """Parse the Retry-After header into seconds. Returns 0 if missing/invalid."""
        if response is None:
            return 0
        header_value = (response.headers.get("Retry-After") or "").strip()
        if header_value.isdigit():
            return max(0, int(header_value))
        if header_value:
            try:
                retry_at = parsedate_to_datetime(header_value)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                return math.ceil(max(0, (retry_at - datetime.now(timezone.utc)).total_seconds()))
            except (TypeError, ValueError, OverflowError):
                pass
        return 0

    def _retry_after_seconds(self, response, attempt):
        provider_delay = self._parse_retry_after(response)
        # For rate limits (429) with long Retry-After, don't waste retries
        # with too-short delays — the caller should handle the error instead.
        # For transient server errors (5xx), cap at 3s to avoid blocking workers.
        if provider_delay > 10:
            return provider_delay  # Let the caller decide whether to wait
        if provider_delay > 0:
            return min(5, provider_delay)
        # Default: 1s, 2s, 3s — capped to avoid starving Odoo workers
        return min(3, max(1, attempt + 1))

    def _post_with_retries(
        self,
        url,
        *,
        headers,
        timeout,
        json_payload=None,
        data=None,
        files=None,
        stream=False,
    ):
        """POST with up to 2 retries on transient errors. Sleeps capped at 3s."""
        response = None
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response = http_requests.post(
                    url,
                    headers=headers,
                    json=json_payload,
                    data=data,
                    files=files,
                    timeout=timeout,
                    stream=stream,
                )
            except (
                http_requests.exceptions.ConnectionError,
                http_requests.exceptions.Timeout,
            ) as exc:
                if attempt == max_attempts - 1:
                    raise
                wait_seconds = self._retry_after_seconds(None, attempt)
                _logger.warning(
                    "[LLM] %s transient %s from %s, retrying in %ss (%d/%d)",
                    self.name,
                    exc.__class__.__name__,
                    url,
                    wait_seconds,
                    attempt + 1,
                    max_attempts,
                )
                if wait_seconds > 0:
                    time.sleep(wait_seconds)
                continue
            if response.ok:
                return response
            if not self._retryable_status_code(response.status_code) or attempt == max_attempts - 1:
                if stream:
                    try:
                        response.content
                    except Exception:  # pragma: no cover - best-effort error buffering
                        pass
                try:
                    response.raise_for_status()
                finally:
                    response.close()
            wait_seconds = self._retry_after_seconds(response, attempt)
            # If provider requests a long wait (rate limit), fail immediately
            # with a clear message instead of blocking the Odoo worker.
            if wait_seconds > 10:
                status_code = response.status_code
                body = self._extract_http_error_body(response)
                response.close()
                raise UserError(
                    f"{self.name} is rate-limited (HTTP {status_code}). "
                    f"The provider suggests waiting {wait_seconds} seconds before retrying. "
                    f"Details: {body or 'rate limit exceeded'}"
                )
            _logger.warning(
                "[LLM] %s transient HTTP %s from %s, retrying in %ss (%d/%d)",
                self.name,
                response.status_code,
                url,
                wait_seconds,
                attempt + 1,
                max_attempts,
            )
            response.close()
            if wait_seconds > 0:
                time.sleep(wait_seconds)
        return response

    def _normalize_stream_content_piece(self, piece):
        if isinstance(piece, str):
            return piece
        if isinstance(piece, list):
            normalized_parts = []
            for entry in piece:
                if isinstance(entry, str):
                    normalized_parts.append(entry)
                    continue
                if not isinstance(entry, dict):
                    continue
                entry_type = str(entry.get("type") or "").lower()
                if entry_type in ("reasoning", "thinking", "analysis", "reasoning_text", "reasoning_content"):
                    continue
                if isinstance(entry.get("text"), str):
                    normalized_parts.append(entry["text"])
                    continue
                if isinstance(entry.get("content"), str):
                    normalized_parts.append(entry["content"])
            return "".join(normalized_parts)
        return ""

    def _normalize_stream_reasoning_piece(self, piece):
        if isinstance(piece, str):
            # Plain strings in OpenAI-compatible deltas are regular content,
            # not explicit reasoning streams.
            return ""
        if isinstance(piece, list):
            normalized_parts = []
            for entry in piece:
                if isinstance(entry, str):
                    continue
                if not isinstance(entry, dict):
                    continue
                entry_type = str(entry.get("type") or "").lower()
                if entry_type not in ("reasoning", "thinking", "analysis", "reasoning_text", "reasoning_content"):
                    continue
                if isinstance(entry.get("text"), str):
                    normalized_parts.append(entry["text"])
                    continue
                if isinstance(entry.get("content"), str):
                    normalized_parts.append(entry["content"])
                    continue
                if isinstance(entry.get("reasoning"), str):
                    normalized_parts.append(entry["reasoning"])
            return "".join(normalized_parts)
        return ""

    def _normalize_message_content(self, content):
        if isinstance(content, str):
            return content
        normalized = self._normalize_stream_content_piece(content)
        if normalized:
            return normalized
        if content in (None, False, ""):
            return ""
        if isinstance(content, dict):
            if isinstance(content.get("text"), str):
                return content["text"]
            if isinstance(content.get("content"), str):
                return content["content"]
        return str(content)

    def _chat_completion_stream_inner(
        self, model_code, messages, tools=None, tool_choice="auto", timeout=60,
        extra_payload=None, extra_headers=None
    ):
        """Inner stream generator. Use chat_completion_stream() instead."""
        self.ensure_one()
        self._assert_api_key()
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = self._get_headers()
        if isinstance(extra_headers, dict):
            headers.update(extra_headers)
        payload = self._build_chat_completion_payload(
            model_code=model_code,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            stream=True,
            extra_payload=extra_payload,
        )

        _logger.info(
            "[LLM] %s stream request to %s model=%s messages=%d",
            self.name,
            url,
            model_code,
            len(messages),
        )
        start = time.time()
        stream_deadline = start + _MAX_STREAM_DURATION

        content_parts = []
        tool_calls_by_index = {}
        finish_reason = None
        usage = {}
        response_id = None
        response_model = model_code
        saw_sse_data = False
        saw_valid_sse_chunk = False
        non_sse_lines = []
        _response_ref = None  # Track response for guaranteed cleanup

        try:
            with self._post_with_retries(
                url,
                headers=headers,
                json_payload=payload,
                timeout=timeout,
                stream=True,
            ) as response:
                _response_ref = response
                response.raise_for_status()
                # Force UTF-8 — requests defaults to ISO-8859-1 when charset is omitted.
                response.encoding = 'utf-8'

                for raw_line in response.iter_lines(decode_unicode=True):
                    # Hard stream duration limit — prevents indefinite blocking
                    # on slow/malicious providers that trickle data.
                    if time.time() > stream_deadline:
                        _logger.warning(
                            "[LLM] %s stream exceeded %ds limit, aborting",
                            self.name, _MAX_STREAM_DURATION,
                        )
                        raise UserError(
                            _("Streaming response from %s exceeded the %d-second limit.")
                            % (self.name, _MAX_STREAM_DURATION)
                        )
                    if not raw_line:
                        continue
                    line = raw_line.strip()

                    if not line.startswith("data:"):
                        non_sse_lines.append(line)
                        continue

                    saw_sse_data = True
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                    except Exception:
                        _logger.debug("[LLM] %s stream chunk parse failed: %s", self.name, data[:200])
                        continue
                    saw_valid_sse_chunk = True

                    if isinstance(chunk, dict) and isinstance(chunk.get("error"), dict):
                        error_message = chunk["error"].get("message") or "Provider stream returned an error."
                        raise UserError(f"{self.name} returned an error: {error_message}")

                    if isinstance(chunk, dict):
                        if chunk.get("id"):
                            response_id = chunk["id"]
                        if chunk.get("model"):
                            response_model = chunk["model"]
                        if isinstance(chunk.get("usage"), dict):
                            usage = chunk["usage"]

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice.get("finish_reason")

                    delta = choice.get("delta") or {}
                    reasoning_piece = self._normalize_stream_reasoning_piece(
                        delta.get("reasoning")
                        or delta.get("reasoning_content")
                        or delta.get("thinking")
                        or delta.get("analysis")
                    )
                    if not reasoning_piece:
                        reasoning_piece = self._normalize_stream_reasoning_piece(delta.get("content"))
                    if reasoning_piece:
                        yield {"type": "reasoning_delta", "delta": reasoning_piece}
                    content_piece = self._normalize_stream_content_piece(delta.get("content"))
                    if content_piece:
                        content_parts.append(content_piece)
                        yield {"type": "content_delta", "delta": content_piece}

                    for tool_chunk in delta.get("tool_calls") or []:
                        idx = tool_chunk.get("index", 0)
                        # If a new id arrives for an existing index, it's a new tool call.
                        # Gemini sometimes sends multiple calls with index=0.
                        chunk_id = tool_chunk.get("id")
                        if chunk_id and idx in tool_calls_by_index and tool_calls_by_index[idx].get("id") and tool_calls_by_index[idx]["id"] != chunk_id:
                            idx = max(tool_calls_by_index.keys()) + 1
                        merged = tool_calls_by_index.setdefault(idx, {
                            "id": chunk_id,
                            "type": tool_chunk.get("type", "function"),
                            "function": {"name": "", "arguments": ""},
                        })
                        if chunk_id:
                            merged["id"] = chunk_id
                        if tool_chunk.get("type"):
                            merged["type"] = tool_chunk["type"]
                        # Preserve extra fields (e.g. thought_signature for Gemini 3)
                        for key in tool_chunk:
                            if key not in ("index", "id", "type", "function"):
                                merged[key] = tool_chunk[key]

                        function_data = tool_chunk.get("function") or {}
                        if function_data.get("name"):
                            # Name arrives complete — set, don't append
                            merged["function"]["name"] = function_data["name"]
                        if function_data.get("arguments"):
                            # Arguments stream in chunks — append
                            merged["function"]["arguments"] = (
                                merged["function"].get("arguments", "") + function_data["arguments"]
                            )
                        # Preserve extra function-level fields (e.g. thought_signature)
                        for key in function_data:
                            if key not in ("name", "arguments"):
                                merged["function"][key] = function_data[key]

            if not saw_sse_data and non_sse_lines:
                try:
                    fallback_result = json.loads("\n".join(non_sse_lines))
                    if isinstance(fallback_result, dict) and isinstance(
                        fallback_result.get("error"), dict
                    ):
                        error_message = (
                            fallback_result["error"].get("message")
                            or "Provider stream returned an error."
                        )
                        raise UserError(
                            f"{self.name} returned an error: {error_message}"
                        )
                    if isinstance(fallback_result, dict) and fallback_result.get("choices"):
                        yield {"type": "result", "result": fallback_result}
                        return
                    raise UserError(
                        f"{self.name} returned an invalid non-stream response for a streamed request."
                    )
                except UserError:
                    raise
                except Exception as exc:
                    body = "\n".join(non_sse_lines)[:500]
                    _logger.error(
                        "[LLM] %s stream fallback parse failed: %s",
                        self.name,
                        body or exc,
                    )
                    raise UserError(
                        f"{self.name} returned an invalid non-stream response for a streamed request."
                    ) from exc
            if not saw_sse_data:
                raise UserError(
                    f"{self.name} returned an empty response for a streamed request."
                )
            if not saw_valid_sse_chunk:
                raise UserError(
                    f"{self.name} returned an invalid streamed response."
                )

        except http_requests.exceptions.Timeout:
            elapsed = time.time() - start
            _logger.error(
                "[LLM] %s stream timeout after %.1fs calling %s", self.name, elapsed, url
            )
            raise UserError(
                f"Request to {self.name} timed out after {timeout}s. "
                "Please try again or increase the timeout."
            )
        except http_requests.exceptions.ConnectionError as exc:
            _logger.error("[LLM] %s stream connection error: %s", self.name, exc)
            raise UserError(
                f"Could not connect to {self.name} at {self.base_url}. "
                "Please verify the URL and your network connection."
            )
        except http_requests.exceptions.HTTPError as exc:
            elapsed = time.time() - start
            body = self._extract_http_error_body(exc.response)
            _logger.error(
                "[LLM] %s stream HTTP %s after %.1fs: %s",
                self.name,
                exc.response.status_code if exc.response is not None else "???",
                elapsed,
                body,
            )
            raise UserError(
                f"{self.name} returned an error: {body or exc}"
            )

        message = {
            "role": "assistant",
            "content": "".join(content_parts),
        }
        if tool_calls_by_index:
            message["tool_calls"] = [
                tool_calls_by_index[idx] for idx in sorted(tool_calls_by_index)
            ]

        result = {
            "id": response_id,
            "model": response_model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
        }
        if usage:
            result["usage"] = usage

        elapsed = time.time() - start
        _logger.info("[LLM] %s stream responded in %.2fs", self.name, elapsed)
        yield {"type": "result", "result": result}

    def chat_completion_stream(
        self, model_code, messages, tools=None, tool_choice="auto", timeout=60,
        extra_payload=None, extra_headers=None
    ):
        """Wrapper that guarantees HTTP cleanup if the caller abandons the generator."""
        gen = self._chat_completion_stream_inner(
            model_code, messages, tools=tools, tool_choice=tool_choice,
            timeout=timeout, extra_payload=extra_payload,
            extra_headers=extra_headers,
        )
        try:
            yield from gen
        finally:
            gen.close()

    def chat_completion(
        self, model_code, messages, tools=None, tool_choice="auto", timeout=60,
        extra_payload=None, extra_headers=None
    ):
        """Non-streaming chat completion. Returns the full JSON response dict."""
        self.ensure_one()
        self._assert_api_key()
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = self._get_headers()
        if isinstance(extra_headers, dict):
            headers.update(extra_headers)

        payload = self._build_chat_completion_payload(
            model_code=model_code,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            stream=False,
            extra_payload=extra_payload,
        )

        _logger.info(
            "[LLM] %s request to %s model=%s messages=%d",
            self.name,
            url,
            model_code,
            len(messages),
        )
        start = time.time()
        response = None

        try:
            response = self._post_with_retries(
                url,
                headers=headers,
                json_payload=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            result = self._parse_json_response(response, operation="chat completion")
        except http_requests.exceptions.Timeout:
            elapsed = time.time() - start
            _logger.error(
                "[LLM] %s timeout after %.1fs calling %s", self.name, elapsed, url
            )
            raise UserError(
                f"Request to {self.name} timed out after {timeout}s. "
                "Please try again or increase the timeout."
            )
        except http_requests.exceptions.ConnectionError as exc:
            _logger.error("[LLM] %s connection error: %s", self.name, exc)
            raise UserError(
                f"Could not connect to {self.name} at {self.base_url}. "
                "Please verify the URL and your network connection."
            )
        except http_requests.exceptions.HTTPError as exc:
            elapsed = time.time() - start
            body = self._extract_http_error_body(exc.response)
            _logger.error(
                "[LLM] %s HTTP %s after %.1fs: %s",
                self.name,
                exc.response.status_code if exc.response is not None else "???",
                elapsed,
                body,
            )
            raise UserError(
                f"{self.name} returned an error: {body or exc}"
            )
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception:  # pragma: no cover - defensive cleanup
                    pass

        elapsed = time.time() - start
        _logger.info("[LLM] %s responded in %.2fs", self.name, elapsed)
        return result

    def text_to_speech(self, model, text, voice_id=None, output_format="mp3_44100_128", timeout=30):
        """Generate speech audio from text. Returns raw audio bytes."""
        self.ensure_one()
        self._assert_api_key()
        provider = self.sudo()
        resolved_voice_id = voice_id or model.voice_id
        if not resolved_voice_id:
            raise UserError("No voice ID configured for TTS model %s." % model.display_name)

        base = (provider.base_url or "").rstrip("/")
        url = f"{base}/text-to-speech/{resolved_voice_id}"

        # Use standard provider auth headers.
        headers = self._get_headers()
        headers["Accept"] = "audio/mpeg"

        voice_settings = {}
        if model.voice_settings_json:
            try:
                voice_settings = json.loads(model.voice_settings_json)
            except (json.JSONDecodeError, TypeError):
                pass

        payload = {
            "text": text,
            "model_id": model.model_id,
        }
        if voice_settings:
            payload["voice_settings"] = voice_settings
        if output_format:
            url = f"{url}?output_format={output_format}"

        _logger.info("[TTS] %s request to %s voice=%s len=%d", self.name, url, resolved_voice_id, len(text))
        start = time.time()
        response = None

        try:
            response = self._post_with_retries(
                url,
                headers=headers,
                json_payload=payload,
                timeout=timeout,
            )
            response.raise_for_status()
        except http_requests.exceptions.Timeout:
            raise UserError(f"TTS request to {self.name} timed out after {timeout}s.")
        except http_requests.exceptions.ConnectionError as exc:
            raise UserError(f"Could not connect to TTS provider {self.name}: {exc}")
        except http_requests.exceptions.HTTPError as exc:
            body = self._extract_http_error_body(exc.response)
            raise UserError(f"TTS error from {self.name}: {body or exc}")
        finally:
            if response is not None:
                try:
                    audio_content = response.content
                except Exception:
                    audio_content = None
                try:
                    response.close()
                except Exception:  # pragma: no cover - defensive cleanup
                    pass
            else:
                audio_content = None

        elapsed = time.time() - start
        _logger.info("[TTS] %s responded in %.2fs, %d bytes", self.name, elapsed, len(audio_content or b""))
        return audio_content or b""

    def transcribe_audio(self, audio_bytes, filename='audio.webm', timeout=30):
        """Transcribe audio via OpenAI-compatible Whisper endpoint. Returns text."""
        self.ensure_one()
        provider = self.sudo()
        if provider.code not in ('openai', 'openrouter'):
            raise UserError("Audio transcription is only supported for OpenAI-compatible providers.")
        if not provider.api_key:
            raise UserError("API key is required for transcription.")

        base = (provider.base_url or 'https://api.openai.com/v1').rstrip('/')
        url = f"{base}/audio/transcriptions"

        # Detect mime type from filename extension
        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else 'webm'
        mime_map = {'webm': 'audio/webm', 'mp4': 'audio/mp4', 'wav': 'audio/wav', 'mp3': 'audio/mpeg', 'ogg': 'audio/ogg', 'm4a': 'audio/mp4'}
        mime = mime_map.get(ext, 'audio/webm')

        headers = {'Authorization': f'Bearer {provider.api_key}'}
        files = {'file': (filename, audio_bytes, mime)}
        data = {'model': 'whisper-1', 'response_format': 'json'}
        response = None

        try:
            response = self._post_with_retries(
                url,
                headers=headers,
                files=files,
                data=data,
                timeout=timeout,
            )
            response.raise_for_status()
        except http_requests.exceptions.Timeout:
            raise UserError(f"Transcription request to {self.name} timed out after {timeout}s.")
        except http_requests.exceptions.ConnectionError as exc:
            raise UserError(f"Could not connect to transcription provider {self.name}: {exc}")
        except http_requests.exceptions.HTTPError as exc:
            body = self._extract_http_error_body(exc.response)
            raise UserError(f"Transcription error from {self.name}: {body or exc}")
        try:
            result = self._parse_json_response(response, operation="audio transcription")
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception:  # pragma: no cover - defensive cleanup
                    pass
        return result.get('text', '')

    def _get_test_connection_model(self):
        """Return the best chat model for a connectivity check, or None."""
        self.ensure_one()
        chat_models = self.sudo().model_ids.filtered(
            lambda m: m.active and m.purpose == "chat"
        )
        if not chat_models:
            return None
        return chat_models.sorted(
            lambda m: (
                not bool(m.is_default),
                m.sequence,
                m.id,
            )
        )[:1]

    def _test_tts_connection(self):
        """Test a TTS provider by generating a short audio clip."""
        self.ensure_one()
        tts_model = self.sudo().model_ids.filtered(
            lambda m: m.active and m.purpose == "tts"
        )[:1]
        if not tts_model:
            return None
        audio = self.text_to_speech(tts_model, "Hello.", timeout=15)
        if audio and len(audio) > 100:
            return f"Audio generated ({len(audio)} bytes)"
        return "Audio response was empty or too small."

    def action_test_connection(self):
        """Test the provider connection with a simple request."""
        self.ensure_one()
        all_models = self.sudo().model_ids.filtered(lambda m: m.active)
        if not all_models:
            raise UserError(
                "Please add at least one model and save before testing."
            )

        # Try chat first
        chat_model = self._get_test_connection_model()
        if chat_model:
            messages = [{"role": "user", "content": "Say hello in one word."}]
            result = self.chat_completion(chat_model.model_id, messages, timeout=30)
            content = self._normalize_message_content(
                result.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "No response content")
            )
            return {
                "type": "ir.actions.client",
                "tag": "display_notification",
                "params": {
                    "title": "Connection Successful",
                    "message": f"Response: {(content or 'No response content')[:200]}",
                    "type": "success",
                    "sticky": False,
                },
            }

        # No chat models — try TTS
        tts_result = self._test_tts_connection()
        if tts_result:
            return {
                "type": "ir.actions.client",
                "tag": "display_notification",
                "params": {
                    "title": "Connection Successful",
                    "message": tts_result,
                    "type": "success",
                    "sticky": False,
                },
            }

        # Only non-testable models (embedding, image)
        purposes = ", ".join(set(all_models.mapped("purpose")))
        raise UserError(
            f"This provider only has {purposes} models. "
            "Connection testing requires a chat or TTS model."
        )
