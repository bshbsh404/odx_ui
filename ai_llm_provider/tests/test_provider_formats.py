"""Tests for provider-specific response formats and error handling.

Covers:
- OpenAI chat completion format parsing
- Anthropic-compatible response normalization
- Provider-specific error body extraction
- Streaming SSE parsing edge cases
- Rate limit (429) fail-fast behavior
- TTS provider-agnostic auth
- Custom headers override warning
- Health / latency tracking fields
"""
from __future__ import annotations

import json
from unittest.mock import Mock, patch, MagicMock

from odoo.exceptions import UserError
from odoo.tests.common import TransactionCase, tagged


def _make_response(status_code=200, json_body=None, text="", headers=None):
    resp = Mock()
    resp.ok = 200 <= status_code < 400
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.text = text or json.dumps(json_body or {})
    if json_body is not None:
        resp.json.return_value = json_body
    else:
        resp.json.side_effect = ValueError("No JSON")
    resp.raise_for_status = Mock()
    if status_code >= 400:
        import requests
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            f"{status_code} Error", response=resp
        )
    return resp


def _make_sse_response(chunks, status_code=200):
    """Build a mock streaming response that yields SSE lines."""
    resp = Mock()
    resp.ok = True
    resp.status_code = status_code
    resp.headers = {"content-type": "text/event-stream"}
    resp.encoding = "utf-8"
    resp.raise_for_status = Mock()

    lines = []
    for chunk in chunks:
        lines.append(f"data: {json.dumps(chunk)}")
        lines.append("")
    lines.append("data: [DONE]")

    resp.iter_lines = Mock(return_value=iter(lines))
    resp.__enter__ = Mock(return_value=resp)
    resp.__exit__ = Mock(return_value=False)
    return resp


@tagged("ai_llm_provider", "post_install", "-at_install")
class TestProviderErrorFormats(TransactionCase):
    """Test error body extraction across different provider response formats."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.provider = cls.env["ai.llm.provider"].create({
            "name": "Test Provider",
            "code": "openai",
            "base_url": "https://api.test.invalid/v1",
            "api_key": "test-key-123",
        })

    def test_openai_error_format(self):
        """OpenAI returns {"error": {"message": "...", "type": "..."}}"""
        resp = _make_response(429, json_body={
            "error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}
        })
        body = self.provider._extract_http_error_body(resp)
        self.assertEqual(body, "Rate limit exceeded")

    def test_anthropic_error_format(self):
        """Anthropic proxies may return {"error": {"message": "..."}} or {"error": "string"}"""
        resp = _make_response(400, json_body={"error": "invalid_api_key"})
        body = self.provider._extract_http_error_body(resp)
        self.assertEqual(body, "invalid_api_key")

    def test_plain_text_error(self):
        """Some providers return plain text errors."""
        resp = _make_response(500, text="Internal Server Error")
        body = self.provider._extract_http_error_body(resp)
        self.assertEqual(body, "Internal Server Error")

    def test_empty_error_response(self):
        """Handle empty error response gracefully."""
        resp = _make_response(502, text="")
        body = self.provider._extract_http_error_body(resp)
        self.assertEqual(body, "")

    def test_none_response(self):
        """Handle None response."""
        body = self.provider._extract_http_error_body(None)
        self.assertEqual(body, "")


@tagged("ai_llm_provider", "post_install", "-at_install")
class TestChatCompletionFormat(TransactionCase):
    """Test chat completion request/response format handling."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.provider = cls.env["ai.llm.provider"].create({
            "name": "Format Provider",
            "code": "openai",
            "base_url": "https://api.test.invalid/v1",
            "api_key": "test-key",
        })

    def test_payload_includes_tools_when_provided(self):
        tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
        payload = self.provider._build_chat_completion_payload(
            "gpt-4o", [{"role": "user", "content": "hi"}],
            tools=tools, tool_choice="auto",
        )
        self.assertEqual(payload["tools"], tools)
        self.assertEqual(payload["tool_choice"], "auto")

    def test_payload_excludes_tools_when_none(self):
        payload = self.provider._build_chat_completion_payload(
            "gpt-4o", [{"role": "user", "content": "hi"}],
        )
        self.assertNotIn("tools", payload)
        self.assertNotIn("tool_choice", payload)

    def test_stream_payload_includes_usage_option(self):
        payload = self.provider._build_chat_completion_payload(
            "gpt-4o", [{"role": "user", "content": "hi"}], stream=True,
        )
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["stream_options"], {"include_usage": True})

    def test_extra_payload_does_not_override_core_fields(self):
        payload = self.provider._build_chat_completion_payload(
            "gpt-4o", [{"role": "user", "content": "hi"}],
            extra_payload={"model": "HIJACKED", "temperature": 0.5},
        )
        self.assertEqual(payload["model"], "gpt-4o")  # Not overridden
        self.assertEqual(payload["temperature"], 0.5)  # Extra field added


@tagged("ai_llm_provider", "post_install", "-at_install")
class TestStreamingParsing(TransactionCase):
    """Test SSE streaming edge cases."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.provider = cls.env["ai.llm.provider"].create({
            "name": "Stream Provider",
            "code": "openai",
            "base_url": "https://api.test.invalid/v1",
            "api_key": "test-key",
        })

    def test_stream_reconstructs_openai_result_format(self):
        chunks = [
            {"id": "chatcmpl-123", "model": "gpt-4o", "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
            ]},
            {"id": "chatcmpl-123", "choices": [
                {"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}
            ]},
            {"id": "chatcmpl-123", "choices": [
                {"index": 0, "delta": {}, "finish_reason": "stop"}
            ]},
            {"id": "chatcmpl-123", "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}},
        ]
        sse_resp = _make_sse_response(chunks)

        with patch(
            "odoo.addons.ai_llm_provider.models.ai_llm_provider.AiLlmProvider._post_with_retries",
            return_value=sse_resp,
        ):
            results = list(self.provider.chat_completion_stream("gpt-4o", [{"role": "user", "content": "hi"}]))

        # Should yield content deltas and a final result
        result_events = [r for r in results if r.get("type") == "result"]
        self.assertEqual(len(result_events), 1)

        result = result_events[0]["result"]
        self.assertEqual(result["id"], "chatcmpl-123")
        self.assertEqual(result["choices"][0]["message"]["content"], "Hello")
        self.assertEqual(result["choices"][0]["finish_reason"], "stop")
        self.assertEqual(result["usage"]["total_tokens"], 6)

    def test_stream_handles_mid_stream_error(self):
        chunks = [
            {"id": "chatcmpl-123", "choices": [
                {"index": 0, "delta": {"content": "Hel"}, "finish_reason": None}
            ]},
            {"error": {"message": "Context length exceeded", "type": "invalid_request_error"}},
        ]
        sse_resp = _make_sse_response(chunks)

        with patch(
            "odoo.addons.ai_llm_provider.models.ai_llm_provider.AiLlmProvider._post_with_retries",
            return_value=sse_resp,
        ):
            with self.assertRaises(UserError) as ctx:
                list(self.provider.chat_completion_stream("gpt-4o", [{"role": "user", "content": "hi"}]))

        self.assertIn("Context length exceeded", str(ctx.exception))


@tagged("ai_llm_provider", "post_install", "-at_install")
class TestRateLimitHandling(TransactionCase):
    """Test 429 rate limit fail-fast behavior (Run 52 fix)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.provider = cls.env["ai.llm.provider"].create({
            "name": "RateLimit Provider",
            "code": "openai",
            "base_url": "https://api.test.invalid/v1",
            "api_key": "test-key",
        })

    def test_long_retry_after_fails_immediately(self):
        """When Retry-After > 10s, should fail fast with clear message."""
        resp = _make_response(429, json_body={
            "error": {"message": "Rate limit exceeded"}
        }, headers={"Retry-After": "30"})
        resp.ok = False

        with patch(
            "odoo.addons.ai_llm_provider.models.ai_llm_provider.http_requests.post",
            return_value=resp,
        ), patch(
            "odoo.addons.ai_llm_provider.models.ai_llm_provider.time.sleep"
        ) as sleep_mock:
            with self.assertRaises(UserError) as ctx:
                self.provider._post_with_retries(
                    "https://api.test.invalid/v1/chat/completions",
                    headers={"Authorization": "Bearer test"},
                    timeout=10,
                    json_payload={"model": "gpt-4o"},
                )

        self.assertIn("rate-limited", str(ctx.exception).lower())
        self.assertIn("30 seconds", str(ctx.exception))
        sleep_mock.assert_not_called()  # Should NOT have slept

    def test_short_retry_after_retries_normally(self):
        """When Retry-After ≤ 10s, should retry with the delay."""
        resp_429 = _make_response(429, headers={"Retry-After": "2"})
        resp_429.ok = False
        resp_ok = _make_response(200, json_body={"choices": []})

        with patch(
            "odoo.addons.ai_llm_provider.models.ai_llm_provider.http_requests.post",
            side_effect=[resp_429, resp_ok],
        ), patch(
            "odoo.addons.ai_llm_provider.models.ai_llm_provider.time.sleep"
        ) as sleep_mock:
            result = self.provider._post_with_retries(
                "https://api.test.invalid/v1/chat/completions",
                headers={"Authorization": "Bearer test"},
                timeout=10,
                json_payload={"model": "gpt-4o"},
            )

        self.assertIs(result, resp_ok)
        sleep_mock.assert_called_once_with(2)


@tagged("ai_llm_provider", "post_install", "-at_install")
class TestCustomHeaders(TransactionCase):
    """Test custom_headers behavior and override warning."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.provider = cls.env["ai.llm.provider"].sudo().create({
            "name": "Custom Header Provider",
            "code": "custom",
            "base_url": "https://api.test.invalid/v1",
            "api_key": "bearer-key",
        })

    def test_custom_headers_merged(self):
        self.provider.custom_headers = '{"X-Custom": "value"}'
        headers = self.provider._get_headers()
        self.assertEqual(headers["X-Custom"], "value")
        self.assertEqual(headers["Authorization"], "Bearer bearer-key")

    def test_custom_headers_can_override_auth(self):
        self.provider.custom_headers = '{"Authorization": "Basic abc123"}'
        headers = self.provider._get_headers()
        self.assertEqual(headers["Authorization"], "Basic abc123")

    def test_invalid_custom_headers_ignored(self):
        self.provider.custom_headers = "not json"
        headers = self.provider._get_headers()
        self.assertIn("Authorization", headers)  # Still has Bearer

    def test_elevenlabs_style_api_key(self):
        """ElevenLabs uses xi-api-key instead of Authorization."""
        self.provider.api_key = False
        self.provider.custom_headers = '{"xi-api-key": "eleven-key"}'
        headers = self.provider._get_headers()
        self.assertEqual(headers["xi-api-key"], "eleven-key")
        self.assertNotIn("Authorization", headers)


@tagged("ai_llm_provider", "post_install", "-at_install")
class TestTTSProviderAgnostic(TransactionCase):
    """Test TTS uses provider-agnostic auth (Run 56 fix)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.provider = cls.env["ai.llm.provider"].sudo().create({
            "name": "TTS Provider",
            "code": "custom",
            "base_url": "https://tts.test.invalid",
            "api_key": "tts-key",
        })
        cls.model = cls.env["ai.llm.model"].create({
            "name": "TTS Model",
            "model_id": "eleven_multilingual_v2",
            "provider_id": cls.provider.id,
            "purpose": "tts",
            "voice_id": "voice-abc",
        })

    def test_tts_uses_get_headers_not_hardcoded(self):
        """TTS should use _get_headers() for auth, not hardcoded xi-api-key."""
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.content = b"fake-audio-data"
        mock_resp.close = Mock()

        with patch.object(
            self.provider.__class__, "_post_with_retries", return_value=mock_resp
        ) as post_mock:
            mock_resp.raise_for_status = Mock()
            self.provider.text_to_speech(self.model, "Hello")

        call_kwargs = post_mock.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers", {})
        # Should use Bearer auth from _get_headers, not xi-api-key
        self.assertIn("Authorization", headers)
        self.assertEqual(headers["Authorization"], "Bearer tts-key")
        self.assertNotIn("xi-api-key", headers)


@tagged("ai_llm_provider", "post_install", "-at_install")
class TestProviderModel(TransactionCase):
    """Test provider and model data integrity."""

    def test_provider_selection_values(self):
        """All provider codes should be valid."""
        from odoo.addons.ai_llm_provider.models.ai_llm_provider import PROVIDER_SELECTION
        codes = [code for code, _label in PROVIDER_SELECTION]
        self.assertIn("openai", codes)
        self.assertIn("anthropic", codes)
        self.assertIn("google", codes)
        self.assertIn("ollama", codes)
        self.assertIn("groq", codes)
        self.assertIn("mistral", codes)
        self.assertIn("custom", codes)

    def test_model_purpose_selection(self):
        """All purpose values should be valid."""
        from odoo.addons.ai_llm_provider.models.ai_llm_model import PURPOSE_SELECTION
        purposes = [p for p, _label in PURPOSE_SELECTION]
        self.assertIn("chat", purposes)
        self.assertIn("tts", purposes)
        self.assertIn("embedding", purposes)
        self.assertIn("image", purposes)
