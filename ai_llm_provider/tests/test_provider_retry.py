from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from unittest.mock import Mock, PropertyMock, call, patch

import requests as http_requests

from odoo.exceptions import UserError, ValidationError
from odoo.tests.common import TransactionCase, tagged


@tagged("ai_llm_provider", "post_install", "-at_install")
class TestAiLlmProviderRetry(TransactionCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.provider = cls.env["ai.llm.provider"].create(
            {
                "name": "Retry Provider",
                "code": "openai",
                "base_url": "https://example.invalid/v1",
            }
        )

    def test_post_with_retries_retries_connection_errors_before_success(self):
        success_response = Mock()
        success_response.ok = True

        with patch(
            "odoo.addons.ai_llm_provider.models.ai_llm_provider.http_requests.post",
            side_effect=[
                http_requests.exceptions.ConnectionError("temporary failure"),
                success_response,
            ],
        ) as post_mock, patch(
            "odoo.addons.ai_llm_provider.models.ai_llm_provider.time.sleep"
        ) as sleep_mock:
            response = self.provider._post_with_retries(
                "https://example.invalid/v1/chat/completions",
                headers={"Authorization": "Bearer token"},
                timeout=10,
                json_payload={"model": "gpt-test"},
            )

        self.assertIs(response, success_response)
        self.assertEqual(post_mock.call_count, 2)
        sleep_mock.assert_called_once_with(1)

    def test_post_with_retries_reraises_timeout_after_retry_budget(self):
        with patch(
            "odoo.addons.ai_llm_provider.models.ai_llm_provider.http_requests.post",
            side_effect=http_requests.exceptions.Timeout("still failing"),
        ) as post_mock, patch(
            "odoo.addons.ai_llm_provider.models.ai_llm_provider.time.sleep"
        ) as sleep_mock:
            with self.assertRaises(http_requests.exceptions.Timeout):
                self.provider._post_with_retries(
                    "https://example.invalid/v1/chat/completions",
                    headers={"Authorization": "Bearer token"},
                    timeout=10,
                    json_payload={"model": "gpt-test"},
                )

        self.assertEqual(post_mock.call_count, 3)
        self.assertEqual(sleep_mock.call_args_list, [call(1), call(2)])

    def test_retry_after_seconds_supports_http_date_header(self):
        response = Mock()
        response.headers = {
            "Retry-After": format_datetime(
                datetime.now(timezone.utc) + timedelta(seconds=10),
                usegmt=True,
            )
        }

        self.assertEqual(self.provider._retry_after_seconds(response, 0), 3)

    def test_retry_after_seconds_allows_zero_second_header(self):
        response = Mock()
        response.headers = {"Retry-After": "0"}

        self.assertEqual(self.provider._retry_after_seconds(response, 0), 0)

    def test_retry_after_seconds_allows_past_http_date_header(self):
        response = Mock()
        response.headers = {
            "Retry-After": format_datetime(
                datetime.now(timezone.utc) - timedelta(seconds=10),
                usegmt=True,
            )
        }

        self.assertEqual(self.provider._retry_after_seconds(response, 0), 0)

    def test_post_with_retries_skips_sleep_for_zero_retry_after(self):
        retryable_response = Mock()
        retryable_response.ok = False
        retryable_response.status_code = 429
        retryable_response.headers = {"Retry-After": "0"}

        success_response = Mock()
        success_response.ok = True

        with patch(
            "odoo.addons.ai_llm_provider.models.ai_llm_provider.http_requests.post",
            side_effect=[retryable_response, success_response],
        ) as post_mock, patch(
            "odoo.addons.ai_llm_provider.models.ai_llm_provider.time.sleep"
        ) as sleep_mock:
            response = self.provider._post_with_retries(
                "https://example.invalid/v1/chat/completions",
                headers={"Authorization": "Bearer token"},
                timeout=10,
                json_payload={"model": "gpt-test"},
            )

        self.assertIs(response, success_response)
        self.assertEqual(post_mock.call_count, 2)
        retryable_response.close.assert_called_once()
        sleep_mock.assert_not_called()

    def test_post_with_retries_closes_non_retryable_error_response(self):
        response = Mock()
        response.ok = False
        response.status_code = 400
        response.raise_for_status.side_effect = http_requests.exceptions.HTTPError("bad request")

        with patch(
            "odoo.addons.ai_llm_provider.models.ai_llm_provider.http_requests.post",
            return_value=response,
        ):
            with self.assertRaises(http_requests.exceptions.HTTPError):
                self.provider._post_with_retries(
                    "https://example.invalid/v1/chat/completions",
                    headers={"Authorization": "Bearer token"},
                    timeout=10,
                    json_payload={"model": "gpt-test"},
                )

        response.close.assert_called_once()

    def test_post_with_retries_closes_final_retryable_error_response(self):
        response = Mock()
        response.ok = False
        response.status_code = 503
        response.headers = {}
        response.raise_for_status.side_effect = http_requests.exceptions.HTTPError("unavailable")

        with patch(
            "odoo.addons.ai_llm_provider.models.ai_llm_provider.http_requests.post",
            return_value=response,
        ) as post_mock, patch(
            "odoo.addons.ai_llm_provider.models.ai_llm_provider.time.sleep"
        ) as sleep_mock:
            with self.assertRaises(http_requests.exceptions.HTTPError):
                self.provider._post_with_retries(
                    "https://example.invalid/v1/chat/completions",
                    headers={"Authorization": "Bearer token"},
                    timeout=10,
                    json_payload={"model": "gpt-test"},
                )

        self.assertEqual(post_mock.call_count, 3)
        self.assertEqual(response.close.call_count, 3)
        self.assertEqual(sleep_mock.call_args_list, [call(1), call(2)])

    def test_post_with_retries_buffers_stream_error_body_before_close(self):
        response = Mock()
        response.ok = False
        response.status_code = 503
        response.headers = {}
        response.raise_for_status.side_effect = http_requests.exceptions.HTTPError(
            "unavailable",
            response=response,
        )

        with patch.object(type(response), "content", new_callable=PropertyMock) as content_mock:
            content_mock.return_value = b'{"error":{"message":"upstream unavailable"}}'
            with patch(
                "odoo.addons.ai_llm_provider.models.ai_llm_provider.http_requests.post",
                return_value=response,
            ) as post_mock, patch(
                "odoo.addons.ai_llm_provider.models.ai_llm_provider.time.sleep"
            ) as sleep_mock:
                with self.assertRaises(http_requests.exceptions.HTTPError):
                    self.provider._post_with_retries(
                        "https://example.invalid/v1/chat/completions",
                        headers={"Authorization": "Bearer token"},
                        timeout=10,
                        json_payload={"model": "gpt-test"},
                        stream=True,
                    )

        self.assertEqual(post_mock.call_count, 3)
        self.assertEqual(content_mock.call_count, 3)
        self.assertEqual(response.close.call_count, 3)
        self.assertEqual(sleep_mock.call_args_list, [call(1), call(2)])

    def test_inactive_default_model_does_not_block_new_active_default(self):
        provider = self.env["ai.llm.provider"].create(
            {
                "name": "Default Provider",
                "code": "openai",
                "base_url": "https://example.invalid/v1",
            }
        )
        self.env["ai.llm.model"].create(
            {
                "name": "Inactive Default",
                "model_id": "gpt-inactive",
                "provider_id": provider.id,
                "purpose": "chat",
                "is_default": True,
                "active": False,
            }
        )

        active_default = self.env["ai.llm.model"].create(
            {
                "name": "Active Default",
                "model_id": "gpt-active",
                "provider_id": provider.id,
                "purpose": "chat",
                "is_default": True,
                "active": True,
            }
        )

        self.assertEqual(
            self.env["ai.llm.model"].get_default_model(purpose="chat"),
            active_default,
        )

    def test_second_active_default_still_fails(self):
        provider = self.env["ai.llm.provider"].create(
            {
                "name": "Strict Provider",
                "code": "openai",
                "base_url": "https://example.invalid/v1",
            }
        )
        self.env["ai.llm.model"].create(
            {
                "name": "Primary Default",
                "model_id": "gpt-primary",
                "provider_id": provider.id,
                "purpose": "chat",
                "is_default": True,
            }
        )

        with self.assertRaises(ValidationError):
            self.env["ai.llm.model"].create(
                {
                    "name": "Second Default",
                    "model_id": "gpt-secondary",
                    "provider_id": provider.id,
                    "purpose": "chat",
                    "is_default": True,
                }
            )

    def test_action_test_connection_normalizes_structured_content(self):
        model = self.env["ai.llm.model"].create(
            {
                "name": "Structured Chat Model",
                "model_id": "gpt-structured",
                "provider_id": self.provider.id,
                "purpose": "chat",
                "active": True,
            }
        )

        with patch.object(
            type(self.provider),
            "chat_completion",
            return_value={
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "Hello"},
                                {"type": "reasoning", "text": "internal"},
                                {"type": "text", "text": " world"},
                            ]
                        }
                    }
                ]
            },
        ):
            action = self.provider.action_test_connection()

        self.assertEqual(model.provider_id, self.provider)
        self.assertEqual(action["params"]["type"], "success")
        self.assertEqual(action["params"]["message"], "Response: Hello world")

    def test_chat_completion_rejects_invalid_json_body(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.side_effect = ValueError("bad json")
        response.text = "<html>upstream proxy error</html>"

        with patch.object(
            type(self.provider),
            "_post_with_retries",
            return_value=response,
        ):
            with self.assertRaises(UserError):
                self.provider.chat_completion(
                    "gpt-test",
                    [{"role": "user", "content": "hello"}],
                    timeout=10,
                )

    def test_transcribe_audio_rejects_invalid_json_body(self):
        provider = self.env["ai.llm.provider"].create(
            {
                "name": "OpenAI Retry Provider",
                "code": "openai",
                "base_url": "https://example.invalid/v1",
                "api_key": "secret",
            }
        )
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.side_effect = ValueError("bad json")
        response.text = "not-json"

        with patch.object(
            type(provider),
            "_post_with_retries",
            return_value=response,
        ):
            with self.assertRaises(UserError):
                provider.transcribe_audio(b"audio-bytes", filename="audio.webm", timeout=10)

    def test_chat_completion_closes_success_response(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"total_tokens": 1},
        }

        with patch.object(
            type(self.provider),
            "_post_with_retries",
            return_value=response,
        ):
            result = self.provider.chat_completion(
                "gpt-test",
                [{"role": "user", "content": "hello"}],
                timeout=10,
            )

        self.assertEqual(result["choices"][0]["message"]["content"], "ok")
        response.close.assert_called_once()

    def test_text_to_speech_closes_success_response(self):
        provider = self.env["ai.llm.provider"].create(
            {
                "name": "TTS Provider",
                "code": "elevenlabs",
                "base_url": "https://example.invalid/v1",
                "api_key": "secret",
            }
        )
        model = self.env["ai.llm.model"].create(
            {
                "name": "TTS Model",
                "model_id": "voice-test",
                "provider_id": provider.id,
                "purpose": "tts",
                "voice_id": "voice-123",
            }
        )
        response = Mock()
        response.raise_for_status.return_value = None
        response.content = b"audio-bytes"

        with patch.object(
            type(provider),
            "_post_with_retries",
            return_value=response,
        ):
            result = provider.text_to_speech(model, "hello world", timeout=10)

        self.assertEqual(result, b"audio-bytes")
        response.close.assert_called_once()

    def test_transcribe_audio_closes_success_response(self):
        provider = self.env["ai.llm.provider"].create(
            {
                "name": "OpenAI Success Provider",
                "code": "openai",
                "base_url": "https://example.invalid/v1",
                "api_key": "secret",
            }
        )
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"text": "transcribed"}

        with patch.object(
            type(provider),
            "_post_with_retries",
            return_value=response,
        ):
            result = provider.transcribe_audio(
                b"audio-bytes",
                filename="audio.webm",
                timeout=10,
            )

        self.assertEqual(result, "transcribed")
        response.close.assert_called_once()

    def test_chat_completion_stream_raises_on_non_sse_json_error_body(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.iter_lines.return_value = ['{"error":{"message":"stream denied"}}']

        with patch.object(
            type(self.provider),
            "_post_with_retries",
            return_value=nullcontext(response),
        ):
            with self.assertRaises(UserError) as exc:
                list(
                    self.provider.chat_completion_stream(
                        "gpt-test",
                        [{"role": "user", "content": "hello"}],
                        timeout=10,
                    )
                )

        self.assertIn("stream denied", str(exc.exception))

    def test_chat_completion_stream_rejects_empty_stream_body(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.iter_lines.return_value = []

        with patch.object(
            type(self.provider),
            "_post_with_retries",
            return_value=nullcontext(response),
        ):
            with self.assertRaises(UserError) as exc:
                list(
                    self.provider.chat_completion_stream(
                        "gpt-test",
                        [{"role": "user", "content": "hello"}],
                        timeout=10,
                    )
                )

        self.assertIn("empty response", str(exc.exception))

    def test_chat_completion_stream_rejects_malformed_sse_chunks(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.iter_lines.return_value = [
            "data: {not-json}",
            "data: still-not-json",
        ]

        with patch.object(
            type(self.provider),
            "_post_with_retries",
            return_value=nullcontext(response),
        ):
            with self.assertRaises(UserError) as exc:
                list(
                    self.provider.chat_completion_stream(
                        "gpt-test",
                        [{"role": "user", "content": "hello"}],
                        timeout=10,
                    )
                )

        self.assertIn("invalid streamed response", str(exc.exception))
