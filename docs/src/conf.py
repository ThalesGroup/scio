"""For ruff :)."""

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import html
import importlib
import inspect
import os
import re
from collections.abc import Mapping, Sequence
from enum import EnumType
from functools import partial
from itertools import zip_longest
from operator import itemgetter
from os.path import dirname, relpath
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, cast

import torch
from paramclasses import IMPL, MISSING, isparamclass
from sphinx.application import Sphinx
from sphinx.util import logging
from sphinx_gallery.sorting import ExplicitOrder  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from re import Match

    from sphinx.ext.autodoc import _AutodocObjType  # noqa: TC004 (unclear...)

import scio

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = "scio"
author = "Élie Goudout"
project_copyright = "2024–%Y, THALES"  # noqa: RUF001 ("EN" dash)
release = scio.__version__
version = release if release == "unknown" else ".".join(scio.__version__.split(".")[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx.ext.linkcode",
    "sphinx_gallery.gen_gallery",
    "sphinxcontrib.katex",
    "myst_parser",
    "sphinx_design",
    "sphinxcontrib.images",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
suppress_warnings = ["config.cache"]  # Unpicklable ``autosummary_context``
logger = logging.getLogger(__name__)

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_show_sourcelink = False
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/ThalesGroup/scio",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/scio-pypi",
            "icon": "_static/pypi_logo.png",
            "type": "local",
        },
    ],
}

# -- Options for LaTeX output -------------------------------------------------
latex_engine = "lualatex"

# -- Options for "sphinxcontrib.bibtex" --------------------------------------
bibtex_bibfiles = ["refs.bib"]

# -- Options for "sphinx.ext.autodoc" ----------------------------------------
autodoc_use_type_comments = False
autodoc_typehints = "none"  # At least until github.com/sphinx-doc/sphinx/issues/13489


# -- Options for "sphinx.ext.autosummary" ------------------------------------
def classdoc_opts(
    *,
    meth: Literal["yes", "no", "summary"] = "yes",
    attr: Literal["yes", "no", "summary"] = "yes",
    exclude: tuple[str, ...] = (),
    include: tuple[str, ...] = (),
) -> dict[str, object]:
    """Define class doc options with dict look-up (for jinja)."""
    return {
        "meth": meth,
        "attr": attr,
        "exclude": exclude,
        "include": include,
    }


# The following defines which methods and attributes are shown or not in
# classes docs. See :func:`classfoc_opts` and :func:`get_classdoc_opts`
# **ORDER MATTERS** (see :func:`get_classdoc_opts`)
config_classdoc_opts = {
    "scio.eval.ROC": classdoc_opts(exclude=("__init__",), attr="summary"),
    "scio.eval.BaseDiscriminativePower": classdoc_opts(include=("__call__",)),
    "scio.eval.*": classdoc_opts(exclude=("from_roc",), attr="no"),
    "scio.recorder.Recorder": classdoc_opts(exclude=("__init__", "training")),
    "scio.scores.classification.jtla.JTLATestMultinomial": classdoc_opts(attr="no"),
    "scio.scores.utils.Index": classdoc_opts(exclude=("__init__",), attr="summary"),
    "scio.scores.BaseScore": classdoc_opts(
        attr="summary",
        exclude=("act_norm",),
        include=("__call__", "__repr__"),
    ),
    "scio.scores.BaseScore*": classdoc_opts(attr="no"),
    "scio.scores.classification.Template*": classdoc_opts(attr="no"),
    "scio.scores.*": classdoc_opts(exclude=("calibrate", "get_conformity"), attr="no"),
    "scio.utils.ScoreTimer": classdoc_opts(
        attr="yes",
        exclude=("__init__",),
        include=("__call__",),
    ),
    "scio.utils.misc.ScoreTimerStat": classdoc_opts(meth="no", attr="no"),
}


def get_classdoc_opts(fullname: str) -> dict[str, object]:
    """Find the first config matching with the fullname.

    Arguments
    ---------
    fullname: ``str``
        A class fullname, for example, ``"scio.scores.utils.Index"``.

    Returns
    -------
    value: ``dict[str, object]``
        First value from ``config_classdoc_opts`` for which the key
        matches with ``fullname``, or default ``classdoc_opts()`` if
        none. Matching here means either:

        - equality if there is no wildcard in the key;
        - equality when replacing a unique wildcard with a non-dotted
          name.

    Example
    -------
    Fullname ``"a.b.c"`` matches with keys ``"a.b.c"`` and ``"a.*.c"``,
    but not with ``"a.*"``.

    """
    for key, value in config_classdoc_opts.items():
        start, wildcard, end = key.partition("*")
        if not wildcard and fullname == start:
            return value
        if (
            wildcard
            and fullname.startswith(start)
            and fullname.endswith(end)
            and "." not in fullname[len(start) : len(fullname) - len(end)]
        ):
            return value

    return classdoc_opts()


def get_api_warning(fullname: str) -> str | None:
    """For (nonbase)score classes, get warning against using the api.

    Arguments
    ---------
    fullname: ``str``
        A class fullname, for example, ``"scio.scores.KNN"``.

    Returns
    -------
    warning: ``str | None``
        The warning if ``fullname`` is identified a a (nonbase)score,
        ``None`` otherwise.

    Example
    -------
    Generates a warning for ``"scio.scores.KNN"`` but not for
    ``"scio.scores.BaseScoreClassif"`` or ``"scio.scores.utils.Index"``.

    """
    prefix = "scio.scores."
    base_identifier = "BaseScore"
    warning = (
        "**Below this point**, the documentation is meant for development purposes "
        "only. Manual use of any listed member is **highly discouraged**. For usage, "
        "see :doc:`/auto_tutorials/inferring_with_confidence`."
    )

    if not fullname.startswith(prefix):
        return None

    name = fullname[len(prefix) :]
    if name.startswith(base_identifier) or "." in name:
        return None

    return warning


autosummary_context = {
    "get_classdoc_opts": get_classdoc_opts,
    "get_api_warning": get_api_warning,
}

# -- Options for "sphinx.ext.napoleon" ---------------------------------------
napoleon_use_rtype = False
napoleon_use_admonition_for_notes = True
napoleon_custom_sections = [
    ("Fields", "params_style"),  # For ``TypedDict``
]

# -- Options for "sphinx_copybutton" -----------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# -- Options for "sphinx.ext.linkcode" ---------------------------------------
github_url = "https://github.com/ThalesGroup/scio"
ref = (  # Commit hash for PR builds, else checked out branch
    os.environ["READTHEDOCS_GIT_COMMIT_HASH"]
    if os.environ.get("READTHEDOCS_VERSION_TYPE", "") == "external"
    else os.environ.get("READTHEDOCS_GIT_IDENTIFIER", "")
)
root_uri = f"{github_url}/blob/{ref}" if ref else Path(__file__).parents[2]


def linkcode_resolve(
    domain: str,
    info: Mapping[str, str],
) -> str | None:
    """
    Resolve a link to source in the Github repo.

    Based on the NumPy version.
    """
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = importlib.import_module(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    fn = None
    lineno = None

    try:
        fn = inspect.getsourcefile(obj)
    except Exception:  # noqa: BLE001
        fn = None
    if not fn:
        return None

    # Ignore re-exports as their source files are not within the repo
    module = inspect.getmodule(obj)
    if module is not None and not module.__name__.startswith("scio"):
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
        lineno_final = lineno + len(source) - 1
    except Exception:  # noqa: BLE001
        lineno_final = None

    fn = relpath(fn, start=dirname(scio.__file__))  # noqa: PTH120

    linespec = f"#L{lineno}-L{lineno_final}" if lineno else ""

    return (
        f"{root_uri}/scio/{fn}{linespec}"
        if ref
        else (cast("Path", root_uri) / "scio" / fn).as_uri()
    )


# -- Options for "sphinx_gallery.gen_gallery" --------------------------------
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "true")  # No download logs
# Monkeypatch for github.com/ThalesGroup/scio/issues/11
torch.hub.load = partial(torch.hub.load, skip_validation=True)

tutorials_order = [
    "inferring_with_confidence.py",
    "visualizing_and_evaluating_ood_detection_algorithms.py",
    "implementing_your_own_ood_detection_algorithm.py",
    "implementing_your_own_discriminative_power_metric.py",
    "diving_inside_neural_networks.py",
]

sphinx_gallery_conf = {
    "examples_dirs": ["tutorials"],
    "gallery_dirs": ["auto_tutorials"],
    "filename_pattern": "",
    "within_subsection_order": ExplicitOrder(tutorials_order),
}

# -- Options for "sphinxcontrib.images" --------------------------------------
images_config = {
    "override_image_directive": True,
    "download": False,
    "cache_path": ".cache",
}

# -- Callbacks for sphinx events (custom rendering) --------------------------
# 1. auto_contributors: Create pretty HTML contributors page by parsing
#    ``CONTRIBUTORS.md``
# 2. process_enum_docstring: Nice members table for documented enums
# 3. process_paramclass_docstring: Complete paramclass docstring for inherited
#    parameters

# 1. auto_contributors
EMOJI_SPAN = '<span class="contrib-emoji" legend="{}">{}</span>'
EMOJI_MAP = MappingProxyType({
    "answering questions": ("Answering questions", "💬"),
    "bug reports": ("Bug reports", "🐛"),
    "code": ("Code", "💻"),
    "dissemination": ("Dissemination", "📢"),
    "documentation": ("Documentation", "📚"),
    "fixes": ("Fixes", "🛠️"),
    "ideas": ("Ideas", "💡"),
    "infrastructure": ("Infrastructure", "🧱"),
    "maintenance": ("Maintenance", "🚧"),
    "pr reviews": ("PR reviews", "👀"),
    "research": ("Research", "🔬"),
    "testing": ("Testing", "⚙️"),
    "tutorials": ("Tutorials", "🎓"),
})

# Markers in CONTRIBUTORS.md
TABLE_START = "<!-- TABLE START -->"
TABLE_END = "<!-- TABLE END -->"


type Entry = tuple[str, tuple[str, ...], tuple[str, str]]


def parse_contributors_table(path: Path) -> tuple[Entry, ...]:
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
    content = path.read_text(encoding="utf-8")
    pattern = f"{re.escape(TABLE_START)}\n(.*)\n{re.escape(TABLE_END)}\n$"
    match = cast("Match[str]", re.search(pattern, content, re.DOTALL))
    entries_str = match[1].split("\n")[2:]  # Drop header lines
    entries_tpl = tuple(
        tuple(map(str.strip, entry.split("|")[1:-1])) for entry in entries_str
    )
    return tuple(
        (  # type: ignore[misc]  # ``itemgetter(0, 2)`` not understood
            username,
            tuple(map(str.strip, contrib_str.split(","))) if contrib_str else (),
            tuple(map(str.strip, itemgetter(0, 2)(name_str.partition(",")))),
        )
        for username, contrib_str, name_str in entries_tpl
    )


def generate_html(entries: Sequence[Entry], contributors_md_uri: str | Path) -> str:
    """Generate HTML content corresponding to parsed entries."""
    lines = []
    lines.append(
        '<div style="display: flex; flex-wrap: wrap; justify-content: flex-start;">',
    )

    for username_raw, contributions, (lastname, firstname) in entries:
        username = html.escape(username_raw)
        emojis = " ".join(EMOJI_SPAN.format(*EMOJI_MAP[c]) for c in contributions)
        name = html.escape(f"{firstname} {lastname}".strip())
        avatar_url = f"https://github.com/{username}.png"

        lines.extend(
            f"""
  <div align="center" style="width: 16.66%; padding: 0.7%;">
    <img src="{avatar_url}" width="100%" alt="@{username}" style="border-radius: 5%;">
    <p>
      {f"<strong>{name}</strong><br/>" if name else ""}
      <a href="https://github.com/{username}" style="font-family: monospace;
         font-size: 0.9em;">
        @{username}
      </a><br/>
      {emojis}
    </p>
  </div>
""".split("\n")[1:],
        )

    auto_gen_info = (
        '<i>This HTML content was automatically generated by parsing <a href="'
        f'{contributors_md_uri}"><code>CONTRIBUTING.md</code></a>.</i>'
    )
    lines.extend(["</div>", "", auto_gen_info, ""])
    return "\n".join(lines)


LAST_CHAR = chr(0x10FFFF)


def sort_key(entry: Entry) -> tuple[str, str, str, int]:
    """Generate sort key for given entry."""
    username, contributions, (lastname, firstname) = entry
    if not username:
        msg = f"Contributor entries require at least a username. Invalid entry: {entry}"
        raise ValueError(msg)

    return (
        lastname.title() or LAST_CHAR,
        firstname.title() or LAST_CHAR,
        username,
        -len(contributions),
    )


def auto_contributors(app: Sphinx) -> None:
    """Parse ``CONTRIBUTORS.md`` generate ``auto_contributors.html``."""
    start = perf_counter()
    auto_contributors_html = app.srcdir / "auto_contributors.html"
    contributors_md = app.srcdir.resolve().parents[1] / "CONTRIBUTORS.md"
    contributors_md_uri = (
        f"{root_uri}/CONTRIBUTORS.md"
        if ref
        else (cast("Path", root_uri) / "CONTRIBUTORS.md").as_uri()
    )
    logger.info(
        """[auto-contributors] Generating auto_contributors.html
    Parsing: %s
    Target: %s""",
        contributors_md,
        auto_contributors_html,
    )

    # Parse ``CONTRIBUTORS.md`` and generate html content
    entries = parse_contributors_table(contributors_md)
    html = generate_html(sorted(entries, key=sort_key), contributors_md_uri)

    # Generate ``auto_contributors.html``
    auto_contributors_html.write_text(html, encoding="utf-8")
    end = perf_counter()
    logger.info("[auto-contributors] Done in %ss", f"{end - start:.3f}")


# 2. process_enum_docstring
ATTRIBUTE_DIRECTIVE = ".. attribute:: "
INDENT = "   "
TYPESEP = "\n" + INDENT + ":type: "
MEMBERS = "Member"
TYPE = "Type"
VALUE = "Value"
DESC = "Description"
TABLESEP = "="


def parse_attributes(
    prepended_lines: tuple[str, ...],
) -> list[tuple[list[str], list[str], list[str]]]:
    """From attribute block, parse every name, description and type.

    First line is always blank (hence "prepended"), ``prepended_lines``
    is attribute block post :class:`NumpyDocstring`.

    Return example
    --------------
    ``(["DIFF"], ["str"], ["First line of desc,", "second final line."])``.

    """
    parsed = []
    joined = "\n".join(prepended_lines)
    attributes = joined.split("\n" + ATTRIBUTE_DIRECTIVE)[1:]
    for attribute in attributes:
        pre, type_hint, post = attribute.partition(TYPESEP)

        name, *desc_lines_indented = pre.split("\n")
        desc_lines = [line[len(INDENT) :] for line in desc_lines_indented[1:-1]]
        typ = post.strip(" \n") if type_hint else ""

        parsed.append(([name], [typ], desc_lines))

    return parsed


def make_table(enum: EnumType, prepended_lines: tuple[str, ...]) -> list[str]:
    """Make table from attribute block lines and values.

    Arguments
    ---------
    prepended_lines: list[str]
        Post :class:`NumpyDocstring` lines from on attributes block.
        Starts with a prepended empty line. Example::

            .. attribute:: attr1

               Description.

               :type: int

            .. attribute:: attr2

    enum: type
        Enum class corresponding to the docstring.

    Returns
    -------
    lines: list[str]
        Corresponding table, for example::

            ====== ========== ================================
            Member Value      Description
            ====== ========== ================================
            attr1  ``'val1'`` Description.
            attr2  ``'val2'``
            ====== ========== ================================

    """
    get_width = lambda iterable: max(map(len, iterable))  # noqa: E731 (lambda expression)
    headers = ([MEMBERS], [VALUE], [DESC])
    parsed = parse_attributes(prepended_lines)

    # Replace type info (don't care) with class value
    content = []
    for (attr,), _, desc_lines in parsed:
        attr_final = f":attr:`{attr}`"
        val_final = f":data:`{getattr(enum, attr).value!r}`"
        content.append(([attr_final], val_final.split("\n"), desc_lines))

    widths = tuple(
        max(map(get_width, column_elts))
        for column_elts in zip(headers, *content, strict=True)
    )
    hrule = tuple([TABLESEP * width] for width in widths)

    def make_lines(elts: tuple[list[str], ...]) -> list[str]:
        """Generate reST lines for one line or rule of the table."""
        return [
            " ".join(
                f"{elt:{width}}" for elt, width in zip(line_elts, widths, strict=False)
            )
            for line_elts in zip_longest(*elts, fillvalue="")
        ]

    lines = []
    for elts in (hrule, headers, hrule, *content, hrule):
        lines.extend(make_lines(elts))

    return [*lines, "\n"]


def transform_attribute_section_into_members_table(
    enum: EnumType,
    lines: list[str],
) -> None:
    """Transform attributes into members table.

    Treats every ``.. attribute::`` statement as a member.
    """
    new = []
    offset = 0
    while offset < len(lines):
        first = lines[offset]
        if not first.startswith(ATTRIBUTE_DIRECTIVE):
            new.append(first)
            offset += 1
            continue

        num = 1
        for line in lines[offset + 1 :]:
            if line.strip() and not line.startswith((INDENT, ATTRIBUTE_DIRECTIVE)):
                break

            num += 1

        prepended_lines = ("", *lines[offset : offset + num])
        new.extend(make_table(enum, prepended_lines))
        offset += num

    lines[:] = new


def process_enum_docstring(
    _app: Sphinx,
    what: _AutodocObjType,
    _name: str,
    obj: object,
    _options: dict[str, bool],
    lines: list[str],
) -> None:
    """For Enum, compact clean table instead of Attribute section."""
    if what == "class" and isinstance(obj, EnumType):
        transform_attribute_section_into_members_table(obj, lines)


# 3. process_paramclass_docstring
RUBRIC_DIRECTIVE = ".. rubric:: "


def get_undoc_params(obj: type, lines: list[str]) -> tuple[set[str], int]:
    """Get undocumented params and where to insert new ones.

    Insert should happend after the last parameter found if any. Else,
    right before the first found section if any. Else after the end.
    """
    param_regex = re.compile(r"^:(param|arg)\s+([\w*]+):")
    param_group = 2

    documented = []
    first_rubric = None
    offset = insert_here = 0
    while offset < len(lines):
        first = lines[offset]
        if not (match := param_regex.match(first)):
            if first.startswith(RUBRIC_DIRECTIVE) and first_rubric is None:
                first_rubric = offset
            offset += 1
            continue

        # Documented parameter found
        documented.append(match[param_group])

        # Handle multiline field description
        field_indent = " " * (len(match[0]) + 1)

        offset += 1
        insert_here = offset
        for line in lines[offset:]:
            if line.strip() and not line.startswith(field_indent):
                break

            offset += 1
            insert_here = offset

    undocumented = set(getattr(obj, IMPL).annotations) - set(documented)

    # If no parameter was found, insert as first rubric
    if not documented:
        insert_here = len(lines) if first_rubric is None else first_rubric

    return undocumented, insert_here


def get_undoc_param_owner(obj: type, attr: str) -> type:
    """For an undocumented parameter, get the first definer.

    We consider changing the annotation as defining a new parameter.
    """
    null = object()
    annotation = getattr(obj, IMPL).annotations[attr]

    owner = obj
    for parent in obj.__mro__[1:]:
        if annotation != getattr(parent, IMPL).annotations.get(attr, null):
            return owner
        owner = parent

    msg = f"No last parent found for parameter '{attr}'. This should never happen"
    raise ValueError(msg)


def add_undocumented_parameters(obj: type, lines: list[str]) -> None:
    """Add reference to parent class for undocumented parameters."""
    undoc_params, insert_here = get_undoc_params(obj, lines)
    if not undoc_params:
        return

    undoc_owners = {attr: get_undoc_param_owner(obj, attr) for attr in undoc_params}
    if required := {attr for attr, owner in undoc_owners.items() if owner is obj}:
        msg = (
            "Missing documentation for the following new parameters in "
            f"'{obj.__qualname__}': {required}"
        )
        raise ValueError(msg)

    mro = obj.__mro__

    def order(attr_owner: tuple[str, type]) -> tuple[int, str]:
        """Order for doc, sorting by reverse mro then alphabetical."""
        attr, owner = attr_owner
        return mro.index(owner), attr

    undoc_sorted = sorted(undoc_owners.items(), key=order)

    to_insert = []
    for attr, owner in undoc_sorted:
        default = getattr(obj, attr, MISSING)

        line = f":param {attr}: See :class:`~{owner.__module__}.{owner.__qualname__}`."
        if default is not getattr(owner, attr, MISSING):
            line += f" Now defaults to ``{default!r}``."

        to_insert.append(line)

    lines[:] = lines[:insert_here] + to_insert + [""] + lines[insert_here:]


def process_paramclass_docstring(
    _app: Sphinx,
    what: _AutodocObjType,
    _name: str,
    obj: object,
    _options: dict[str, bool],
    lines: list[str],
) -> None:
    """For paramclasses, complete doc for inherited parameters."""
    if what == "class":
        cls = cast("type", obj)
        if isparamclass(cls):
            add_undocumented_parameters(cls, lines)


# -- Connect callbacks to events ---------------------------------------------
def setup(app: Sphinx) -> None:
    """Connect our custom callbacks."""
    app.connect("builder-inited", auto_contributors)
    app.connect("autodoc-process-docstring", process_enum_docstring)
    app.connect("autodoc-process-docstring", process_paramclass_docstring)
