"""Provide functionality to parse ``CONTRIBUTORS.md`` & output HTML."""

import html
import re
from operator import itemgetter
from pathlib import Path
from types import MappingProxyType

CONTRIBUTORS_MD = Path(__file__).parent.parent.parent / "CONTRIBUTORS.md"

# Map known contributions to emojis (extend as needed)
EMOJI_MAP = MappingProxyType({
    "answering questions": '<span title="Answering questions">💬</span>',
    "bug reports": '<span title="Bug reports">🐛</span>',
    "code": '<span title="Code">💻</span>',
    "dissemination": '<span title="Dissemination">📢</span>',
    "documentation": '<span title="Documentation">📚</span>',
    "fixes": '<span title="Fixes">🛠️</span>',
    "ideas": '<span title="Ideas">💡</span>',
    "infrastructure": '<span title="Infrastructure">🧱</span>',
    "maintenance": '<span title="Maintenance">🚧</span>',
    "pr reviews": '<span title="PR reviews">👀</span>',
    "research": '<span title="Research">🔬</span>',
    "testing": '<span title="Testing">⚙️</span>',
    "tutorials": '<span title="Tutorials">🎓</span>',
})

# Markers in CONTRIBUTORS.md
HTML_START = "<!-- HTML START -->"
HTML_END = "<!-- HTML END -->"
TABLE_START = "<!-- TABLE START -->"
TABLE_END = "<!-- TABLE END -->"


type Entry = tuple[str, tuple[str, ...], tuple[str, str]]


def parse_contributors_table(content: str) -> tuple[Entry, ...]:
    """Find manual entries and parse the table.

    Not very robust on purpose, expects a precise format. See example
    below for output specification.

    Example
    -------
    Sample output::

        (
            ("alice", ("code", "documentation", "fixes"), ("Smith", "Alice")),
            ("bob", ("fixes",), ("Lee", "Bob")),
            ("noname", (), ("", "")),
        )

    """
    pattern = f"{re.escape(TABLE_START)}\n(.*)\n{re.escape(TABLE_END)}\n$"
    entries_str = re.search(pattern, content, re.DOTALL).group(1).split("\n")[2:]
    entries_tpl = tuple(
        tuple(map(str.strip, entry.split("|")[1:-1])) for entry in entries_str
    )
    return tuple(
        (
            username,
            tuple(map(str.strip, contrib_str.split(","))) if contrib_str else (),
            tuple(map(str.strip, itemgetter(0, 2)(name_str.partition(",")))),
        )
        for username, contrib_str, name_str in entries_tpl
    )


LAST_CHAR = chr(0x10FFFF)


def sort_key(entry: Entry) -> tuple[str, str, str, int]:
    """Generate sort key for given entry."""
    username, contributions, (lastname, firstname) = entry
    if not username:
        msg = f"Contributor entries require at least a username. Invalid entry: {entry}"
        raise ValueError(msg)

    return (
        lastname or LAST_CHAR,
        firstname or LAST_CHAR,
        username,
        -len(contributions),
    )


def generate_html(entries: tuple[Entry, ...]) -> str:
    """Generate HTML content corresponding to parsed entries."""
    html_parts = []
    html_parts.append(
        '<div style="display: flex; flex-wrap: wrap; justify-content: flex-start;">',
    )

    for username_raw, contributions, (lastname, firstname) in entries:
        username = html.escape(username_raw)
        emojis = " ".join(EMOJI_MAP[contrib] for contrib in contributions)
        name = html.escape(f"{firstname} {lastname}".strip())
        avatar_url = f"https://github.com/{username}.png"

        entry_html = f"""
<div align="center" style="width: 16.66%; padding: 0.8%;">
  <img src="{avatar_url}" width="100px" alt="@{username}" style="border-radius: 5%;">
  <p>
    {f"<strong>{name}</strong><br/>" if name else ""}
    <a href="https://github.com/{username}" style="font-family: monospace;
       font-size: 0.9em;">
      @{username}
    </a><br/>
    {emojis}
  </p>
</div>
"""
        html_parts.append(entry_html.strip())

    html_parts.append("</div>")
    return "\n".join(html_parts)


def update_contributors_md(path: Path) -> None:
    """Parse ``CONTRIBUTORS.md`` bottom table & update HTML content."""
    content = path.read_text(encoding="utf-8")
    entries = parse_contributors_table(content)
    html = generate_html(sorted(entries, key=sort_key))

    pattern = f"^(.*{re.escape(HTML_START)}\n).*(\n{re.escape(HTML_END)}.*)$"
    new_content = html.join(re.search(pattern, content, re.DOTALL).group(1, 2))

    if new_content != content:
        path.write_text(new_content, encoding="utf-8")


if __name__ == "__main__":
    update_contributors_md(CONTRIBUTORS_MD)
