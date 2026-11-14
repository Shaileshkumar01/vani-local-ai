"""Base abstractions for Vāṇī PC automation tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    """Result returned by a tool execution."""

    success: bool
    message: str
    spoken_response: str | None = None
    data: dict[str, Any] | None = None


class BaseTool(ABC):
    """Abstract base class for all local PC tools."""

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with the provided arguments."""
        raise NotImplementedError

