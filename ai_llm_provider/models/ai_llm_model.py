from __future__ import annotations

import json

from odoo import _, api, fields, models
from odoo.exceptions import ValidationError


PURPOSE_SELECTION = [
    ("chat", "Chat Completion"),
    ("tts", "Text-to-Speech"),
    ("embedding", "Embedding"),
    ("image", "Image Generation"),
]


class AiLlmModel(models.Model):
    _name = "ai.llm.model"
    _description = "AI LLM Model"
    _order = "sequence, id"

    name = fields.Char(string="Name", required=True)
    model_id = fields.Char(
        string="Model ID",
        required=True,
        help="API identifier, e.g. gpt-4o, claude-sonnet-4-20250514, eleven_multilingual_v2",
    )
    provider_id = fields.Many2one(
        comodel_name="ai.llm.provider",
        string="Provider",
        required=True,
        ondelete="cascade",
    )
    purpose = fields.Selection(
        PURPOSE_SELECTION,
        string="Purpose",
        default="chat",
        required=True,
        help="What this model is used for: chat completion, text-to-speech, embedding, or image generation.",
    )
    voice_id = fields.Char(
        string="Voice ID",
        help="Voice identifier for TTS models (e.g. ElevenLabs voice ID).",
    )
    voice_settings_json = fields.Text(
        string="Voice Settings (JSON)",
        help='TTS voice parameters as JSON, e.g. {"stability": 0.5, "similarity_boost": 0.75}',
    )
    sequence = fields.Integer(string="Sequence", default=10)
    is_default = fields.Boolean(
        string="Default Model",
        default=False,
        help="When checked, this model will be used as the default for its purpose "
        "(chat, TTS, embedding, image). Only one default per purpose is allowed.",
    )
    max_tokens = fields.Integer(string="Max Tokens", default=4096)
    supports_tools = fields.Boolean(
        string="Supports Tools",
        default=True,
        help="Check if this model supports function/tool calling "
        "(e.g. GPT-4, Claude 3+, Gemini Pro).",
    )
    supports_vision = fields.Boolean(
        string="Supports Vision",
        default=False,
        help="Check if this model can process images in messages "
        "(e.g. GPT-4o, Claude 3 Sonnet/Opus).",
    )
    active = fields.Boolean(string="Active", default=True)

    def init(self):
        """Partial unique index: one default per purpose at the database level."""
        self.env.cr.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS ai_llm_model_one_default_per_purpose
            ON ai_llm_model (purpose)
            WHERE is_default = true AND active = true
        """)

    @api.constrains("is_default", "purpose")
    def _check_single_default(self):
        """Ensure only one active default model exists per purpose."""
        for record in self:
            if record.is_default and record.active and record.provider_id.is_active:
                existing = self.search([
                    ("is_default", "=", True),
                    ("purpose", "=", record.purpose),
                    ("active", "=", True),
                    ("provider_id.is_active", "=", True),
                    ("id", "!=", record.id),
                ], limit=1)
                if existing:
                    purpose_label = dict(PURPOSE_SELECTION).get(record.purpose, record.purpose)
                    raise ValidationError(
                        "Only one %(purpose)s model can be set as the default. "
                        "'%(model)s' is already the default model."
                        % {
                            "purpose": purpose_label.lower(),
                            "model": existing.display_name,
                        }
                    )

    @api.constrains("purpose", "voice_id")
    def _check_tts_voice_id(self):
        """Voice ID is only required at call time, not at save time.
        Some TTS providers (e.g. OpenAI) pass voice as a payload param."""
        pass

    @api.constrains("voice_settings_json")
    def _check_voice_settings_json(self):
        for record in self:
            if record.voice_settings_json:
                try:
                    settings = json.loads(record.voice_settings_json)
                    if not isinstance(settings, dict):
                        raise ValidationError(
                            _("Voice settings must be a JSON object, e.g. "
                              '{\"stability\": 0.5, \"similarity_boost\": 0.75}')
                        )
                except (json.JSONDecodeError, TypeError) as exc:
                    raise ValidationError(
                        _("Invalid JSON in voice settings: %s") % exc
                    )

    @api.model
    def get_default_model(self, purpose="chat"):
        """Return the default active model for a purpose, or False if none set."""
        domain = [
            ("is_default", "=", True),
            ("active", "=", True),
            ("provider_id.is_active", "=", True),
        ]
        if purpose:
            domain.append(("purpose", "=", purpose))
        return self.search(domain, limit=1) or False

    def name_get(self):
        """Return 'Provider / Model' format."""
        result = []
        for record in self:
            display_name = f"{record.provider_id.name} / {record.name}"
            result.append((record.id, display_name))
        return result
