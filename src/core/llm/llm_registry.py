"""
world-simiulator.config

Centralised settings for the world-simiulator testbed.

Loading order (pydantic-settings resolves in this priority, highest first):
  1. Actual environment variables  (e.g. injected by K8s)
  2. .env file pointed to by AI_ENV_FILE  (local dev)
  3. Default values defined below

Usage
─────
  from core.config import Settings
  from core.llm.llm_registry import build_llm_registry, LLM_ROLE_CONFIG

  settings = Settings()
  settings.apply_langsmith()

  registry = build_llm_registry(settings, LLM_ROLE_CONFIG)
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

import dataclasses
import logging
from enum import Enum
from typing import Any

from pydantic import SecretStr

from core.config import Settings
from core.llm.token_callback import TokenUsageCallback

logger = logging.getLogger(__name__)


# ── Provider / label enums ────────────────────────────────────────────────────


class LLMProvider(Enum):
    STUB = "STUB"
    OPENAI = "OPENAI"
    ANTHROPIC = "ANTHROPIC"
    OLLAMA = "OLLAMA"
    BEDROCK = "BEDROCK"


class LLMLabel(Enum):
    HAIKU = "haiku"
    SONNET = "sonnet"
    OPUS = "opus"
    GPT_MINI = "gpt-mini"
    GPT = "gpt"
    OLLAMA_LLAMA3 = "ollama-llama3"
    BEDROCK_SONNET = "bedrock-sonnet"
    BEDROCK_HAIKU = "bedrock-haiku"
    STUB = "STUB"


# ── Model config ──────────────────────────────────────────────────────────────


@dataclasses.dataclass
class LLMModel:
    model: str
    provider: LLMProvider
    # Settings attribute holding the API key. Only meaningful for
    # single-key providers (OpenAI/Anthropic). None for Ollama (endpoint)
    # and Bedrock (AWS credential chain). All catalog entries below use
    # keyword args, so field order is free to change.
    key_label: str | None = None
    api_key: SecretStr | None = None
    # USD per million tokens (public list price). None for providers where
    # pricing is regional/variable (Bedrock) or free (Ollama).
    price_per_1m_input: float | None = None
    price_per_1m_output: float | None = None


# ── Available model definitions ───────────────────────────────────────────────
# Maps LLMLabel → LLMModel. Add new models here as needed.
# key_label must match a field name on Settings.

models: dict[LLMLabel, LLMModel | None] = {
    # Anthropic — prices are public list rates as of 2026-05 (per million tokens)
    LLMLabel.HAIKU: LLMModel(
        key_label="anthropic_api_key",
        provider=LLMProvider.ANTHROPIC,
        model="claude-haiku-4-5-20251001",
        price_per_1m_input=0.80,
        price_per_1m_output=4.00,
    ),
    LLMLabel.SONNET: LLMModel(
        key_label="anthropic_api_key",
        provider=LLMProvider.ANTHROPIC,
        model="claude-sonnet-4-6",
        price_per_1m_input=3.00,
        price_per_1m_output=15.00,
    ),
    LLMLabel.OPUS: LLMModel(
        key_label="anthropic_api_key",
        provider=LLMProvider.ANTHROPIC,
        model="claude-opus-4-7",
        price_per_1m_input=15.00,
        price_per_1m_output=75.00,
    ),
    # OpenAI — prices are public list rates as of 2026-05 (per million tokens)
    LLMLabel.GPT_MINI: LLMModel(
        key_label="openai_api_key",
        provider=LLMProvider.OPENAI,
        model="gpt-4o-mini",
        price_per_1m_input=0.15,
        price_per_1m_output=0.60,
    ),
    LLMLabel.GPT: LLMModel(
        key_label="openai_api_key",
        provider=LLMProvider.OPENAI,
        model="gpt-4o",
        price_per_1m_input=2.50,
        price_per_1m_output=10.00,
    ),
    # Ollama (local)
    LLMLabel.OLLAMA_LLAMA3: LLMModel(
        key_label="ollama_base_url",
        provider=LLMProvider.OLLAMA,
        model="llama3.2:latest",
    ),
    # AWS Bedrock — Claude via the Converse API. No key_label: auth is the
    # AWS credential chain (see Settings.aws_region / aws_profile).
    # NOTE: depending on region/account these may need an inference-profile
    # ID/ARN (e.g. "us.anthropic.claude-3-5-sonnet-...") rather than the
    # bare on-demand model ID below.
    LLMLabel.BEDROCK_SONNET: LLMModel(
        provider=LLMProvider.BEDROCK,
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    ),
    LLMLabel.BEDROCK_HAIKU: LLMModel(
        provider=LLMProvider.BEDROCK,
        model="anthropic.claude-3-5-haiku-20241022-v1:0",
    ),
    # Stub — no LLM
    LLMLabel.STUB: None,
}


# ── Role → model label mapping ────────────────────────────────────────────────
# Maps agents role → LLMLabel. This is the SINGLE source of truth: the
# composition root (main.py) builds the registry from this dict. Change a
# label here to swap the model for that role everywhere.
#
# Only roles that are actually consumed via llm_registry.get(<role>) belong
# here — listing a role nothing requests is just a lie waiting to mislead.
# Consumers today:
#   - "classifier"        : cluster agents  (agents/cluster/nodes.py)
#   - "logistics"         : logistics ReAct loop, Phase 1 (agents/logistics/nodes.py)
#   - "logistics_extract" : logistics structured extraction, Phase 2 (same file)
#
# Phases 1 and 2 are deliberately separate roles so the structured-output
# pass can use a different model than the tool-calling loop without touching
# code — see make_extract_plan_node. They point at the same label for now.


# ── LLM Registry ──────────────────────────────────────────────────────────────


class LLMRegistry:
    """
    Role-based catalog of LangChain chat models, built just-in-time.

    Built once at startup and threaded into graph builders. Nodes request
    a model by role without knowing which provider or model was configured.

    Lazy construction
    ─────────────────
    Chat models are NOT constructed up front. Each role holds a factory;
    the model is built on first ``get(role)`` and cached thereafter. This
    means a process only needs credentials for the providers it actually
    uses — e.g. an eval that calls one Anthropic role won't fail because no
    OpenAI key is set for a role it never touches.

    Trade-off: credential/config errors surface on first ``get`` rather than
    at startup. Call ``warmup()`` at a composition root to opt back into
    fail-fast validation (e.g. a server that wants bad keys caught at boot).

    Callbacks and providers are eager (they're cheap), so ``usage_report()``
    lists every configured role and ``make_system_message()`` works without
    building the model.

    Usage:
        registry = build_llm_registry(settings, LLM_ROLE_CONFIG)
        llm = registry.get("classifier")   # built here, on demand
        result = llm.invoke(messages)
    """

    def __init__(
        self,
        factories: dict[str, Any],
        callbacks: dict[str, TokenUsageCallback] | None = None,
        providers: dict[str, LLMProvider] | None = None,
    ) -> None:
        self._factories = factories          # role → callable() -> chat model
        self._clients: dict[str, Any] = {}   # role → built model (lazy cache)
        self._callbacks = callbacks or {}
        self._providers = providers or {}

    def get(self, role: str, default: Any = None) -> Any:
        if role in self._clients:
            return self._clients[role]
        factory = self._factories.get(role)
        if factory is not None:
            client = factory()
            self._clients[role] = client
            return client
        if default is not None:
            return default
        raise KeyError(f"No LLM registered for role {role!r}.")

    def warmup(self, *roles: str) -> None:
        """Eagerly build the given roles (or all configured roles if none
        are named), so credential/config errors surface now rather than on
        first use. Idempotent — already-built roles are skipped by get()."""
        for role in (roles or tuple(self._factories)):
            self.get(role)

    @property
    def roles(self) -> list[str]:
        return sorted(self._factories)

    def make_system_message(self, role: str, text: str):
        """Build a SystemMessage with provider-appropriate prompt-caching markup.

        Anthropic: wraps text in a content block with cache_control so the
        system prompt is cached after the first call — cache hits cost ~10%
        of normal input price.

        OpenAI / others: returns a plain SystemMessage. OpenAI applies
        automatic prefix caching server-side with no client markup needed.
        """
        from langchain_core.messages import SystemMessage

        if self._providers.get(role) == LLMProvider.ANTHROPIC:
            return SystemMessage(content=[{
                "type": "text",
                "text": text,
                "cache_control": {"type": "ephemeral"},
            }])
        return SystemMessage(text)

    def reset_usage(self) -> None:
        """Reset all per-role token counters to zero.

        Call this before each eval run so usage_report() reflects only the
        tokens consumed by that run, not the accumulated session total.
        """
        for cb in self._callbacks.values():
            cb.reset()

    def usage_report(self) -> list[dict]:
        """Return per-role token usage totals since the last reset_usage() call."""
        return [cb.report() for cb in self._callbacks.values()]


def _resolve_provider_kwargs(model_cfg: LLMModel, settings: Settings) -> dict[str, Any]:
    """Resolve provider-specific construction kwargs from Settings.

    This is the credential seam. It is the only place that knows how a
    provider authenticates, so the role-based registry and every node
    above it stay provider-agnostic:

      - OpenAI / Anthropic : a single API key from a Settings attribute.
      - Ollama             : a base URL (no credential).
      - Bedrock            : the AWS credential chain — region/profile are
                             optional overrides; omitting them lets boto3
                             fall back to env / shared config / IAM role.

    Returns kwargs ready to splat into the LangChain chat-model ctor.
    """
    provider = model_cfg.provider

    if provider in (LLMProvider.OPENAI, LLMProvider.ANTHROPIC):
        raw = getattr(settings, model_cfg.key_label, None) if model_cfg.key_label else None
        api_key = raw.get_secret_value() if isinstance(raw, SecretStr) else raw
        return {"api_key": api_key or None}

    if provider == LLMProvider.OLLAMA:
        return {"base_url": settings.ollama_base_url}

    if provider == LLMProvider.BEDROCK:
        kwargs: dict[str, Any] = {}
        if settings.aws_region:
            kwargs["region_name"] = settings.aws_region
        if settings.aws_profile:
            kwargs["credentials_profile_name"] = settings.aws_profile
        return kwargs

    raise ValueError(f"Unknown provider: {provider}")


def _build_chat_model(
    model_cfg: LLMModel,
    provider_kwargs: dict[str, Any],
    callback: TokenUsageCallback | None = None,
) -> Any:
    """Instantiate a LangChain chat model from a resolved LLMModel.

    ``provider_kwargs`` comes from :func:`_resolve_provider_kwargs` — this
    function only knows which LangChain class maps to which provider, not
    how that provider authenticates.
    """
    callbacks = [callback] if callback is not None else []

    if model_cfg.provider == LLMProvider.OPENAI:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_cfg.model, temperature=0, callbacks=callbacks, **provider_kwargs
        )
    elif model_cfg.provider == LLMProvider.ANTHROPIC:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model_name=model_cfg.model, temperature=0, callbacks=callbacks, **provider_kwargs
        )
    elif model_cfg.provider == LLMProvider.OLLAMA:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model_cfg.model, temperature=0, callbacks=callbacks, **provider_kwargs
        )
    elif model_cfg.provider == LLMProvider.BEDROCK:
        try:
            from langchain_aws import ChatBedrockConverse
        except ImportError as exc:  # pragma: no cover - exercised only when Bedrock is selected
            raise RuntimeError(
                "Bedrock provider selected but 'langchain-aws' is not installed. "
                "Add langchain-aws to enable Bedrock — no other code change is needed."
            ) from exc

        return ChatBedrockConverse(
            model=model_cfg.model, temperature=0, callbacks=callbacks, **provider_kwargs
        )
    raise ValueError(f"Unknown provider: {model_cfg.provider}")


def build_llm_registry(
    settings: Settings,
    role_config: dict[str, LLMLabel],
    model_catalog: dict[LLMLabel, LLMModel | None] = models,
) -> LLMRegistry:
    """
    Build an LLMRegistry from settings + role config (+ an overridable catalog).

    Parameters
    ----------
    settings:     Loaded Settings (carries API keys and ollama_base_url).
    role_config:  role name → LLMLabel mapping (the app's selection of which
                  roles it wants and which model backs each).
    model_catalog: LLMLabel → LLMModel menu. Defaults to this module's `models`
                  catalog — callers only pass it to inject a fake in tests.

    STUB roles are skipped — registry.get() will raise KeyError if all
    roles are stubs and there is no fallback.
    """
    factories: dict[str, Any] = {}
    callbacks: dict[str, TokenUsageCallback] = {}
    providers: dict[str, LLMProvider] = {}

    for role, label in role_config.items():
        model_cfg = model_catalog.get(label)
        if model_cfg is None:
            logger.info("Skipping role %r — STUB label", role)
            continue

        # Credential resolution is cheap and side-effect-free, so it stays
        # eager. The expensive/validating step — constructing the LangChain
        # chat model (which imports the provider SDK and checks the key) —
        # is deferred into the factory and runs on first get(role).
        provider_kwargs = _resolve_provider_kwargs(model_cfg, settings)
        callback = TokenUsageCallback(
            role,
            price_per_1m_input=model_cfg.price_per_1m_input,
            price_per_1m_output=model_cfg.price_per_1m_output,
        )

        def _factory(model_cfg=model_cfg, provider_kwargs=provider_kwargs, callback=callback, role=role):
            logger.info(
                "Building LLM for role %r → %s (%s)",
                role,
                model_cfg.model,
                model_cfg.provider.value,
            )
            return _build_chat_model(model_cfg, provider_kwargs, callback)

        factories[role] = _factory
        callbacks[role] = callback
        providers[role] = model_cfg.provider
        logger.info("Registered LLM factory for role %r → %s (%s)", role, model_cfg.model, model_cfg.provider.value)

    return LLMRegistry(factories, callbacks, providers)
