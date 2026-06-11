"""
world-simiulator.config

Centralised settings for the world-simiulator testbed.

Loading order (pydantic-settings resolves in this priority, highest first):
  1. Actual environment variables  (e.g. injected by K8s)
  2. .env file pointed to by AI_ENV_FILE  (local dev)
  3. Default values defined below

Usage
─────
  from config import Settings, build_llm_registry, LLM_ROLE_CONFIG, models

  settings = Settings()
  settings.apply_langsmith()

  registry = build_llm_registry(settings, models, LLM_ROLE_CONFIG)
  llm = registry.get("classifier")
  result = llm.invoke(messages)

Settings is constructed once at the composition root and threaded down
into anything that needs it. There is deliberately no module-level
cached singleton — it makes tests harder (cache_clear footguns), breaks
in multi-process deployments, and hides the dependency from callers.

Deployment modes
────────────────
  Local dev  — set AI_ENV_FILE=/path/to/.env  (or export vars directly)
  K8s        — leave AI_ENV_FILE unset; inject vars via ConfigMap / Secret
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


# ── LangSmith project identity ────────────────────────────────────────────────
#
# Defined in code, not in the environment. This is intentional: the project
# name is a property of the application, not of the deployment. It is always
# forced into the environment by apply_langsmith() regardless of what .env
# or shell variables say.
LANGSMITH_PROJECT = "oceans-simulator"


# ── Settings ──────────────────────────────────────────────────────────────────


class Settings(BaseSettings):
    # ── Database ───────────────────────────────────────────────────────────────
    postgres_url: str = "postgresql://localhost:5432/oceans?options=-csearch_path%3Ddatasets,public"
    # ── Image catalog on disk ───────────────────────────────────────────────────
    # Where the downloaded fluke JPEGs live, laid out <individual_id>/<sighting_id>.jpg.
    # Lives OUTSIDE the repo so blobs never land in git. Override via IMAGE_ROOT.
    image_root: Path = Path.home() / "Source/DATA/oceans/images"
    # ── Embedder identity ───────────────────────────────────────────────────────
    # Selects which model produces vectors AND the tag stamped on every catalog row;
    # must be a key in image_embedder.EMBEDDERS. Flip it (e.g. EMBEDDER_VER=bioclip-v1
    # in .env) to A/B a different model without editing code, then re-run embed_images.
    # embedder_ver: str = "clip-vitb32-v1"
    embedder_ver: str = "clip-hf-vitb32-v1"

    # ── LLM credentials ───────────────────────────────────────────────────────
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    hf_token: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434"

    # ── AWS Bedrock ───────────────────────────────────────────────────────────
    # Bedrock auth is the standard AWS credential chain (env vars, shared
    # config/credentials file, or an IAM role) — not a single API key.
    # region/profile are optional overrides; leave unset to use boto3 defaults.
    aws_region: str | None = None
    aws_profile: str | None = None

    # ── LangSmith tracing ─────────────────────────────────────────────────────
    # Field names match the modern LANGSMITH_* env-var convention.
    # Project name is NOT here — see LANGSMITH_PROJECT constant above.
    langsmith_api_key: str = ""
    langsmith_tracing: bool = False
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    model_config = SettingsConfigDict(
        # env_file is intentionally NOT set at class-definition time —
        # the composition root passes ``_env_file=os.getenv("AI_ENV_FILE")``
        # at construction so the lookup happens per-instance. Tests can
        # construct Settings() without any .env interference.
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def apply_langsmith(self) -> None:
        """Write LangSmith settings into os.environ so LangGraph picks them up.

        Sets both LANGSMITH_* (modern SDK) and LANGCHAIN_* (older LangGraph
        tracing) so all library versions see the values.

        - Credentials and endpoint: written only if not already in the environment
          (real env takes priority over .env).
        - Project name: always forced to the LANGSMITH_PROJECT constant —
          code owns this value, not the environment.
        """
        tracing_value = "true" if self.langsmith_tracing else ""
        # Credentials/config: environment wins if already set
        for key, value in {
            "LANGSMITH_API_KEY": self.langsmith_api_key,
            "LANGSMITH_TRACING": tracing_value,
            "LANGSMITH_ENDPOINT": self.langsmith_endpoint,
            "LANGCHAIN_API_KEY": self.langsmith_api_key,
            "LANGCHAIN_TRACING_V2": tracing_value,
            "LANGCHAIN_ENDPOINT": self.langsmith_endpoint,
        }.items():
            if value and not os.environ.get(key):
                os.environ[key] = value
        # Project: code always wins
        os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT
        os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT

    def apply_hf(self) -> None:
        """Export the Hugging Face token into os.environ as HF_TOKEN.

        huggingface_hub / transformers auto-detect the HF_TOKEN environment
        variable, but values loaded from AI_ENV_FILE live only on this
        Settings instance — they are never written to the real environment.
        This bridges that gap. Real env wins if HF_TOKEN is already set.
        """
        if self.hf_token and not os.environ.get("HF_TOKEN"):
            os.environ["HF_TOKEN"] = self.hf_token.get_secret_value()


def get_settings() -> Settings:
    """Build a ``Settings`` instance, honouring ``AI_ENV_FILE``.

    If the ``AI_ENV_FILE`` environment variable is set, its value is
    passed as ``_env_file`` so pydantic-settings reads that file.
    Otherwise a plain ``Settings()`` is returned (env vars only).
    """
    env_file = os.getenv("AI_ENV_FILE")
    return Settings(_env_file=env_file) if env_file else Settings()
