"""
Minimal S-expression parser for KiCad symbol libraries and Specctra session files.

Atoms come back as strings (quoted strings have their quotes removed); lists
come back as Python lists.
"""

import re

_TOKEN = re.compile(r'\s*(?:(\()|(\))|"((?:[^"\\]|\\.)*)"|([^\s()"]+))')


def parse(text):
    """Parse every top-level expression in `text` and return them as a list."""
    stack = [[]]
    pos = 0
    n = len(text)
    while pos < n:
        m = _TOKEN.match(text, pos)
        if not m:
            if text[pos:].strip() == "":
                break
            raise ValueError(f"unparseable S-expression at offset {pos}")
        pos = m.end()
        if m.group(1):
            stack.append([])
        elif m.group(2):
            if len(stack) == 1:
                raise ValueError(f"unbalanced ')' at offset {pos}")
            done = stack.pop()
            stack[-1].append(done)
        elif m.group(3) is not None:
            stack[-1].append(m.group(3).replace('\\"', '"'))
        elif m.group(4) is not None:
            stack[-1].append(m.group(4))
    if len(stack) != 1:
        raise ValueError("unbalanced '(' at end of input")
    return stack[0]


def head(expr):
    """The first atom of a list expression, or None."""
    if isinstance(expr, list) and expr and isinstance(expr[0], str):
        return expr[0]
    return None


def children(expr, name):
    """Direct child lists of `expr` whose head is `name`."""
    return [c for c in expr if isinstance(c, list) and head(c) == name]


def child(expr, name):
    """The first direct child list of `expr` with head `name`, or None."""
    for c in expr:
        if isinstance(c, list) and head(c) == name:
            return c
    return None
