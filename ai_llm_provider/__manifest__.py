{
    "name": "ODX LLM",
    "summary": "Multi-provider LLM integration for Odoo 19",
    "description": """
Connect to OpenAI, Anthropic, Google Gemini, Groq, Mistral, Ollama, and
custom providers through a unified OpenAI-compatible API.

Features:
- Chat completion with streaming (SSE)
- Tool/function calling with normalization
- Text-to-speech and audio transcription
- Rate limit handling and retry with backoff
- SSRF protection and custom headers
    """,
    "version": "19.0.1.0.0",
    "category": "Technical",
    "author": "Bashir Hassan",
    "website": "https://www.odxbuilder.com/",
    "support": "support@odxbuilder.com",
    "license": "LGPL-3",
    "images": ["static/description/multiple providers.jpeg"],
    "depends": ["base", "base_setup"],
    "data": [
        "security/ai_llm_provider_security.xml",
        "security/ir.model.access.csv",
        "views/ai_llm_provider_views.xml",
        "views/ai_llm_model_views.xml",
        "views/menus.xml",
        "views/res_config_settings_views.xml",
        "data/provider_data.xml",
    ],
    "demo": [
        "data/demo_data.xml",
    ],
    "installable": True,
    "application": False,
}
