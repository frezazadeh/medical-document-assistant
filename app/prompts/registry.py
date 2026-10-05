"""Prompts live in YAML files next to this module, not inside Python strings.

Each file has a name, a version and a Jinja template. The version ends up in
every trace, so when an answer looks wrong we know which prompt produced it.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml
from jinja2 import Environment, StrictUndefined

TEMPLATES_DIR = Path(__file__).parent / "templates"

# StrictUndefined: a missing variable is a bug, fail instead of rendering an empty string.
_env = Environment(undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    description: str
    template: str

    @property
    def tag(self) -> str:
        return f"{self.name}@{self.version}"

    def render(self, **variables) -> str:
        return _env.from_string(self.template).render(**variables).strip()


class PromptRegistry:
    def __init__(self, directory: Path = TEMPLATES_DIR):
        self._prompts: dict[str, PromptTemplate] = {}
        for path in sorted(directory.glob("*.yaml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            prompt = PromptTemplate(
                name=raw["name"],
                version=str(raw["version"]),
                description=raw.get("description", ""),
                template=raw["template"],
            )
            self._prompts[prompt.name] = prompt

    def get(self, name: str) -> PromptTemplate:
        try:
            return self._prompts[name]
        except KeyError:
            known = ", ".join(sorted(self._prompts))
            raise KeyError(f"Unknown prompt '{name}'. Available: {known}") from None

    def names(self) -> list[str]:
        return sorted(self._prompts)
