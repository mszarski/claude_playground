"""Parameters: named, ranged values that evaluate lazily.

Arithmetic on a :class:`Param` builds an expression tree instead of a
number, so derived parameters (``hairline = stem * (1 - contrast)``) can be
declared once at module level and re-evaluated for every compile. Values are
supplied by the compiler through :func:`evaluating`; outside of it every
parameter evaluates to its default.
"""
from __future__ import annotations

import contextvars
import math
import operator
from contextlib import contextmanager
from typing import Any, Callable

_values: contextvars.ContextVar[dict[str, float] | None] = contextvars.ContextVar(
    "ductus_param_values", default=None
)


class Expr:
    """A lazily evaluated numeric expression."""

    def _eval(self, env: dict[str, float] | None) -> float:  # pragma: no cover
        raise NotImplementedError

    def eval(self) -> float:
        return float(self._eval(_values.get()))

    def __float__(self) -> float:
        return self.eval()

    def __int__(self) -> int:
        return int(self.eval())

    def __round__(self, n=None):
        return round(self.eval(), n)

    def __format__(self, spec):
        return format(self.eval(), spec)

    # arithmetic builds trees
    def _bin(self, other, op, sym, swap=False):
        a, b = (other, self) if swap else (self, other)
        return BinOp(op, sym, a, b)

    def __add__(self, o): return self._bin(o, operator.add, "+")
    def __radd__(self, o): return self._bin(o, operator.add, "+", True)
    def __sub__(self, o): return self._bin(o, operator.sub, "-")
    def __rsub__(self, o): return self._bin(o, operator.sub, "-", True)
    def __mul__(self, o): return self._bin(o, operator.mul, "*")
    def __rmul__(self, o): return self._bin(o, operator.mul, "*", True)
    def __truediv__(self, o): return self._bin(o, operator.truediv, "/")
    def __rtruediv__(self, o): return self._bin(o, operator.truediv, "/", True)
    def __pow__(self, o): return self._bin(o, operator.pow, "**")
    def __rpow__(self, o): return self._bin(o, operator.pow, "**", True)
    def __neg__(self): return Func(operator.neg, "-", self)
    def __abs__(self): return Func(abs, "abs", self)

    # comparisons evaluate eagerly (structure must not depend on them; see
    # principle 4 -- use them for values, never to decide point counts)
    def __lt__(self, o): return self.eval() < val(o)
    def __le__(self, o): return self.eval() <= val(o)
    def __gt__(self, o): return self.eval() > val(o)
    def __ge__(self, o): return self.eval() >= val(o)


class Param(Expr):
    """A named parameter with a default and an optional range."""

    def __init__(self, default: float, range: tuple[float, float] | None = None,
                 name: str | None = None):
        self.default = float(default)
        self.range = (float(range[0]), float(range[1])) if range else None
        self.name = name
        if self.range and not (self.range[0] <= self.default <= self.range[1]):
            raise ValueError(f"default {default} outside range {range}")

    def _eval(self, env):
        if env is not None and self.name in env:
            return env[self.name]
        return self.default

    def __repr__(self):
        return f"param({self.name or '?'}={self.default:g})"


class BinOp(Expr):
    def __init__(self, op: Callable, sym: str, a: Any, b: Any):
        self.op, self.sym, self.a, self.b = op, sym, a, b

    def _eval(self, env):
        return self.op(_ev(self.a, env), _ev(self.b, env))

    def __repr__(self):
        return f"({self.a!r} {self.sym} {self.b!r})"


class Func(Expr):
    def __init__(self, fn: Callable, label: str, *args: Any):
        self.fn, self.label, self.args = fn, label, args

    def _eval(self, env):
        return self.fn(*(_ev(a, env) for a in self.args))

    def __repr__(self):
        return f"{self.label}({', '.join(map(repr, self.args))})"


def _ev(x: Any, env) -> float:
    if isinstance(x, Expr):
        return x._eval(env)
    return x


def param(default: float, range: tuple[float, float] | None = None,
          name: str | None = None) -> Param:
    """Declare a parameter. Unnamed params are named after their variable
    when the font module is loaded."""
    return Param(default, range, name)


def val(x: Any) -> float:
    """Resolve a number, expression or zero-arg callable to a float."""
    if isinstance(x, Expr):
        return x.eval()
    if callable(x):
        return val(x())
    return float(x)


def pt(p: Any) -> tuple[float, float]:
    """Resolve a 2-tuple of numbers/expressions."""
    return (val(p[0]), val(p[1]))


def lazy_min(*a): return Func(min, "min", *a)
def lazy_max(*a): return Func(max, "max", *a)
def lazy_sqrt(a): return Func(math.sqrt, "sqrt", a)


@contextmanager
def evaluating(values: dict[str, float]):
    """Evaluate all parameters with ``values`` (name -> value)."""
    token = _values.set(dict(values))
    try:
        yield
    finally:
        _values.reset(token)


def current_values() -> dict[str, float] | None:
    return _values.get()
