import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, PrivateAttr, create_model


@dataclass
class ToolContext:
    name: str
    raw_args: dict[str, Any]
    args: dict[str, Any]
    result: Any = None
    hooks: list["Hook"] = field(default_factory=list)


class Hook:
    def prepare(self, ctx: ToolContext) -> None:
        pass

    def after(self, ctx: ToolContext) -> None:
        pass

    def on_error(self, ctx: ToolContext, error: Exception) -> str | None:
        return None


class RuntimeTool(BaseTool):
    _func: Callable[..., str] = PrivateAttr()
    _hooks: tuple[Hook, ...] = PrivateAttr()

    def __init__(self, func: Callable[..., str], hooks: tuple[Hook, ...]) -> None:
        super().__init__(name=func.__name__, description=inspect.getdoc(func) or "", args_schema=_args_schema(func))
        self._func = func
        self._hooks = hooks

    @property
    def func(self) -> Callable[..., str]:
        return self._func

    @property
    def hooks(self) -> tuple[Hook, ...]:
        return self._hooks

    @property
    def signature(self) -> inspect.Signature:
        return inspect.signature(self.func)

    @property
    def external_signature(self) -> inspect.Signature:
        parameters = [
            parameter.replace(annotation=_external_annotation(parameter.annotation))
            for parameter in self.signature.parameters.values()
        ]
        return self.signature.replace(parameters=parameters)

    def _run(self, **kwargs: Any) -> str:
        bound = self.signature.bind(**kwargs)
        bound.apply_defaults()
        ctx = ToolContext(name=self.name, raw_args=dict(kwargs), args=dict(bound.arguments))
        ctx.hooks.extend(self.hooks)
        try:
            self._run_prepare_hooks(ctx)
            ctx.result = self.func(**ctx.args)
            for hook in reversed(ctx.hooks):
                hook.after(ctx)
            return ctx.result
        except Exception as error:
            for hook in reversed(ctx.hooks):
                handled = hook.on_error(ctx, error)
                if handled is not None:
                    return handled
            raise

    def _run_prepare_hooks(self, ctx: ToolContext) -> None:
        index = 0
        while index < len(ctx.hooks):
            ctx.hooks[index].prepare(ctx)
            index += 1


def _args_schema(func: Callable[..., str]) -> type[BaseModel]:
    fields = {}
    for parameter in inspect.signature(func).parameters.values():
        annotation = _external_annotation(parameter.annotation)
        if annotation is inspect.Signature.empty:
            annotation = Any
        default = ... if parameter.default is inspect.Signature.empty else parameter.default
        fields[parameter.name] = (annotation, default)

    return create_model(f"{func.__name__.title()}Input", **fields)


def _external_annotation(annotation: Any) -> Any:
    return getattr(annotation, "__tool_external_annotation__", annotation)
