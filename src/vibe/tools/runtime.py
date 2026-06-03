import inspect
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, PrivateAttr, ValidationError, create_model

MAX_INLINE_OUTPUT_CHARS = 30_000


class Tool(BaseTool):
    _func: Callable[..., str] = PrivateAttr()

    def __init__(self, func: Callable[..., str]) -> None:
        super().__init__(
            name=func.__name__,
            description=inspect.getdoc(func) or "",
            args_schema=_args_schema(func),
            handle_validation_error=_validation_error_message,
        )
        self._func = func

    @property
    def func(self) -> Callable[..., str]:
        return self._func

    @property
    def signature(self) -> inspect.Signature:
        return inspect.signature(self.func)

    def _run(self, **kwargs: Any) -> str:
        try:
            result = self.func(**kwargs)
        except ValidationError as error:
            return _validation_error_message(error)
        except ValueError as error:
            return f"Error: {error}"

        return _deliver_output(self.name, result)


def tool(func: Callable[..., str]) -> Tool:
    return Tool(func)


def _args_schema(func: Callable[..., str]) -> type[BaseModel]:
    fields = {}
    for parameter in inspect.signature(func).parameters.values():
        annotation = parameter.annotation
        if annotation is inspect.Signature.empty:
            annotation = Any
        default = ... if parameter.default is inspect.Signature.empty else parameter.default
        fields[parameter.name] = (annotation, default)

    return create_model(f"{func.__name__.title()}Input", **fields)


def _validation_error_message(error: ValidationError) -> str:
    issue = error.errors()[0]
    field = ".".join(str(part) for part in issue["loc"])
    context = issue.get("ctx") or {}
    if issue["type"] == "greater_than_equal":
        return f"Error: {field} must be >= {context['ge']}"
    if issue["type"] == "less_than_equal":
        return f"Error: {field} must be <= {context['le']}"
    if issue["type"] == "string_too_short":
        return f"Error: {field} must not be empty"
    return f"Error: {field} {issue['msg']}"


def _deliver_output(tool_name: str, result: str) -> str:
    if len(result) <= MAX_INLINE_OUTPUT_CHARS:
        return result

    output_dir = Path(".vibe") / "tmp"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"tool-output-{tool_name}-{uuid.uuid4().hex[:8]}.txt"
    output_path.write_text(result, encoding="utf-8", errors="replace")
    return (
        f"Tool output is {len(result)} chars, saved to '{output_path}'. "
        "Use read with offset and limit, or bash with grep, to inspect the saved output."
    )
