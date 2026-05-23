# schema.py ----------------------------------------------------
import inspect
import typing
from typing import Any, Callable, Dict, get_args, get_origin, get_type_hints

# Python type -> JSON Schema "type" keyword
_PRIMITIVES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _schema_for_annotation(annotation):
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}

    if annotation in _PRIMITIVES:
        return {"type": _PRIMITIVES[annotation]}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is typing.Union:  # Optional[X] / Union[X, None] -> schema for X
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _schema_for_annotation(non_none[0])
        return {}  # genuine multi-type union -> leave unconstrained

    if origin in (list, set, tuple):
        item = args[0] if args else Any
        return {"type": "array", "items": _schema_for_annotation(item)}

    if origin is dict:
        return {"type": "object"}

    return {}  # unknown / custom type -> unconstrained


def _parse_docstring_params(doc):
    # any "name: description" line is treated as a param description
    descriptions = {}
    for raw in (doc or "").splitlines():
        line = raw.strip()
        if ":" not in line:
            continue
        head, _, tail = line.partition(":")
        head = head.strip()
        if head and " " not in head and tail.strip():
            descriptions[head] = tail.strip()
    return descriptions


def derive_parameters(fn: Callable) -> Dict[str, Any]:
    """JSON-Schema 'object' for fn's params; no-default params are required."""
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}
    doc_params = _parse_docstring_params(inspect.getdoc(fn))

    properties = {}
    required = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if param.kind in (  # skip *args / **kwargs
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        prop = _schema_for_annotation(hints.get(name, param.annotation))
        if name in doc_params:
            prop = {**prop, "description": doc_params[name]}
        properties[name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(name)

    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def summary_from_docstring(fn: Callable) -> str:
    """First non-empty line of the docstring (default tool description)."""
    for line in (inspect.getdoc(fn) or "").splitlines():
        if line.strip():
            return line.strip()
    return ""
