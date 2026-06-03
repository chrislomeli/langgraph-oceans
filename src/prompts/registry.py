"""
world-simiulator.prompts.registry

The :class:`PromptRegistry` — a Jinja2-backed loader that renders versioned
prompt templates and resolves Pydantic model schemas referenced by name
inside templates.

Construction model
──────────────────
Build one registry at the composition root and pass it to the graph
builders that need it. The registry knows nothing about the rest of
the project — agents register their own Pydantic models against it,
which keeps prompts/ from importing agents/ (the dependency arrow
must always go agent → prompt, never the reverse).

Templates layout
────────────────
``templates/<prompt_name>/<version>/`` holds:
    prompt.j2     — Jinja2 template
    manifest.yaml — required_vars + description

Versions are sorted lexicographically — use zero-padded names if you
expect more than nine versions (``v01``, ``v02``, …, ``v10``).

Failure modes
─────────────
All raise :class:`exceptions.PromptError`. The registry never silently
falls back to a default template — missing templates, missing variables,
and unknown schema-filter names all fail loudly so they surface in CI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel

from exceptions import PromptError

DEFAULT_TEMPLATES_DIR = Path(__file__).parent / "templates"


class PromptRegistry:
    """Loads and renders Jinja2 prompt templates from a templates directory.

    Build once at the composition root, register the Pydantic models that
    templates reference via ``{{ "Name" | schema }}``, and pass the
    registry to graph builders.

    Example
    ───────
        registry = PromptRegistry()
        registry.register_model(AnomalyFinding)
        text = registry.render("evaluate", {"cluster_id": "north", ...})
    """

    def __init__(self, templates_dir: Path | None = None) -> None:
        self._templates_dir = templates_dir or DEFAULT_TEMPLATES_DIR
        self._models: dict[str, type[BaseModel]] = {}
        self._env = self._build_env()

    # ── Model registration ───────────────────────────────────────────
    def register_models(self, *models: type[BaseModel]) -> None:
        """Register one or more Pydantic models for the ``schema`` Jinja filter.

        Usage::

            registry.register_models(AnomalyFinding, ClassifyOutput)
        """
        for model in models:
            self._models[model.__name__] = model

    def register_model(self, model: type[BaseModel]) -> None:
        """Make a Pydantic model resolvable by the ``schema`` Jinja filter.

        The filter is invoked in templates as ``{{ "ModelName" | schema }}``
        and returns the model's JSON schema as an indented string. Register
        every model that any prompt template needs to reference.
        """
        self._models[model.__name__] = model

    # ── Rendering ────────────────────────────────────────────────────

    def latest_version(self, prompt_name: str) -> str:
        """Return the highest version directory name for a prompt."""
        prompt_dir = self._templates_dir / prompt_name
        if not prompt_dir.is_dir():
            raise PromptError(f"No prompt named {prompt_name!r} in {self._templates_dir}")
        versions = sorted(
            d.name for d in prompt_dir.iterdir() if d.is_dir() and (d / "manifest.yaml").exists()
        )
        if not versions:
            raise PromptError(f"No versioned templates found for prompt {prompt_name!r}")
        return versions[-1]

    def render(
        self,
        prompt_name: str,
        context: dict[str, Any],
        *,
        version: str | None = None,
    ) -> str:
        """Render a prompt template and return the completed string.

        Parameters
        ──────────
        prompt_name : Key matching a subdirectory of templates/.
        context     : Dict of variables passed to the template.
        version     : Explicit version (e.g. ``"v1"``). Defaults to latest.

        Raises
        ──────
        PromptError : Unknown prompt, missing manifest, missing required
                      variable, or unknown schema-filter model.
        """
        resolved_version = version or self.latest_version(prompt_name)
        manifest = self._load_manifest(prompt_name, resolved_version)

        required = manifest.get("required_vars", [])
        missing = [v for v in required if v not in context]
        if missing:
            raise PromptError(
                f"Prompt {prompt_name!r}/{resolved_version} missing required vars: {missing}"
            )

        template_path = f"{prompt_name}/{resolved_version}/prompt.j2"
        template = self._env.get_template(template_path)
        return template.render(**context)

    # ── Internals ────────────────────────────────────────────────────

    def _build_env(self) -> Environment:
        env = Environment(
            loader=FileSystemLoader(str(self._templates_dir)),
            undefined=StrictUndefined,  # raise on missing vars, never silently blank
            trim_blocks=True,  # strip newline after block tags
            lstrip_blocks=True,  # strip leading whitespace before block tags
        )
        env.filters["schema"] = self._schema_filter
        return env

    def _schema_filter(self, model_name: str) -> str:
        """Resolve a registered model name to its JSON schema string."""
        model_cls = self._models.get(model_name)
        if model_cls is None:
            raise PromptError(
                f"schema filter: unknown model {model_name!r}. Registered: {sorted(self._models)}"
            )
        return json.dumps(model_cls.model_json_schema(), indent=2)

    def _load_manifest(self, prompt_name: str, version: str) -> dict[str, Any]:
        manifest_path = self._templates_dir / prompt_name / version / "manifest.yaml"
        if not manifest_path.exists():
            raise PromptError(f"Missing manifest: {manifest_path}")
        with open(manifest_path) as f:
            return yaml.safe_load(f) or {}
