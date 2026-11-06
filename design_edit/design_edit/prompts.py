"""Chat formatting shared by data building, training and evaluation."""

from __future__ import annotations

from .schema import Node, serialize

SYSTEM_PROMPT = """You edit UI designs. A design is a tree of nodes, one per line, indented two spaces per level:
  <type>#<id> "<name>" key=value ...
Types: frame (container; layout=row|col auto-layout with gap, pad, align, justify), text, rect, image.
Colors are hex. Sizes are px. opacity is 0..1.

Reply ONLY with a patch, one operation per line, changing as little as possible:
  set <id> key=value ...
  unset <id> key ...
  delete <id>
  move <id> <parent-id> <index>
  insert <parent-id> <index> <type>#<new-id> "<name>" key=value ...
New nodes use ids after the largest existing id. Insert a parent before its children."""


def user_message(doc: Node, instruction: str) -> str:
    return f"Design:\n{serialize(doc)}\n\nRequest: {instruction}"


def prompt_messages(doc: Node, instruction: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message(doc, instruction)}]
