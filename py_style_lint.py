#!/usr/bin/env python3
#
# Single-file toolkit for AI-generated Python code. It holds the prompt prefix
# to paste before a codegen request, the rule registry, and the AST linter.
# Standard library only, so it can be copied anywhere or handed to an agent
# as-is. It is meant to be run on one file at a time.
#
# Usage:
#   python code_style_lint.py path/to/file.py
#   python code_style_lint.py path/to/file.py --python-version 3.12
#   python code_style_lint.py path/to/file.py --allow-suppression-comments
#   python code_style_lint.py path/to/file.py --warnings-as-errors
#   python code_style_lint.py --print-prompt
#   python code_style_lint.py --list-rules
#
# Exit code is 1 when any error-severity rule fires (or any warning when
# --warnings-as-errors is set), otherwise 0.
#

from __future__ import annotations

import argparse
import ast
import builtins
import io
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path


PROMPT_PREFIX = """\
# Code generation rules

You are writing Python code for a codebase with strict, opinionated conventions.
Follow every rule below. Where a rule conflicts with "best practice" you were
trained on, follow the rule below instead.

## Environment
- Python version: 3.11
- Pinned libraries: Pandas 3.x, NumPy 2.x, Requests 2.x, Matplotlib 3.x, OpenCV-Python 5.x,
  PyTorch 2.x, Scikit-learn 1.9.x, Plotly 7.x, Transformers 5.x
- Prefer the Python standard library. Only use a third-party library if it is
  pinned above; if you think you need one that is not listed, ask first.

## Error handling
- Do not use `try`/`except`/`finally` (including `except*`) at all.
- If you are convinced a specific line cannot be handled any other way, write
  the line plainly and put this comment block directly above it:

  #
  # TRY-EXCEPT-CANDIDATE: why this line needs exception handling
  #

- Do not work around the ban. All of these are forbidden too:
  `contextlib.suppress`, defining `__exit__`/`__aexit__`,
  `asyncio.gather(..., return_exceptions=True)`, assigning `sys.excepthook`
  or `threading.excepthook`, and `exec`/`eval`/`compile`/`__import__`.
- `raise` is allowed.

## Design philosophy
- YAGNI and KISS. Do not build for hypothetical future requirements.
- Long functions are fine. Do not split a function just to make it shorter;
  split it only if it does two genuinely different jobs.
- Prefer simpler code even if it is more verbose than a clever one-liner.
- No nested comprehensions: no comprehension inside another comprehension,
  and no comprehension with more than one `for`. Use a plain loop instead.
- No long oneliners.
- No ; to put multiple statements on one line.
- Use strings standard library instead of regexes unless needed.


## Functions
- Type all parameters and the return value of every function. Do not
  annotate local variables.
- No docstrings.
- IMPORTANT: every function should depend on as little as possible (few
  imported names, few calls to other project functions), and as little as
  possible should depend on it. Prefer passing explicit arguments over
  reaching for more collaborators.
- IMPORTANT: functions may accept loose/flexible inputs, but must return one
  narrow, consistent type. No `Optional`, `Union`, `X | None`, `Any` or
  `object` in return types, and never return a value on some paths and
  `None` on others.
- No mutable default arguments (`def f(x: list = [])` is forbidden).
- When calling a function, never pass an argument whose value equals that
  parameter's default. Pass an argument only to change its default.
  Forbidden examples: `open(path, "r")`, `print(text, end="\\n")`,
  `sorted(items, reverse=False)`, `settings.get(key, None)`,
  `range(0, count)`, or `scale(value, factor=1)` when `factor` defaults to 1.
- Avoid `*args` / `**kwargs` unless the function is genuinely variadic
  (e.g. a decorator or thin wrapper).
- A function must not read variables from an outer scope (module-level
  variables or an enclosing function's locals). Pass every value in as a
  parameter so there are no hidden dependencies. Imports, other functions,
  classes and UPPER_CASE module constants may be referenced.
- A function with more than 4 parameters (not counting a leading `self`/`cls`)
  lists them vertically: one parameter per line, none on the `def` line.

## Classes
- Prefer `@dataclass` for any structured data instead of dicts or tuples.
- Do not write any other kind of class unless an external library or
  framework requires it (e.g. subclassing a framework base class), and say
  so in a comment when you do. Do not create classes with `type(...)`.

## Naming
- Use long, descriptive names for everything: functions, variables,
  parameters, modules. No single-letter names except loop counters
  (`i`, `j`, `k`).
- Module-level constants are UPPER_CASE; avoid unexplained magic numbers or
  strings inline.
- Avoid redeclaring built in names like match, case, list, ...


## Comments
- For every code block comment on the "why" (intent, reasoning, non-obvious constraints) or
  on a concept, never to narrate what the code does line by line.
- Every why/conceptual comment must use this exact block form, with a bare
  `#` line above and below the text:

  #
  # comment text here
  #

## Imports and paths
- No wildcard imports (`from x import *`).
- Use `pathlib.Path` for anything involving file paths. Never use `os.path`.

## Output
- Use `print` for output. Do not use the `logging` module unless asked.
- Do not add too many prints.
- Use only f-strings for formatting. Do not use `str.format()` or `%` formatting.
- The only way to write a multiline string is with triple quotes. Do not
  concatenate string literals with `+` and do not rely on implicit
  concatenation of adjacent string literals.

## Checking your work
- After writing or changing a file, run:
  `python code_style_lint.py <file> --python-version <PYTHON_VERSION>`
- Fix every error and re-run until it exits 0. Read the warnings and fix
  them unless you have a concrete reason not to; if so, say why.
- Never add `# noqa` or any other suppression comment. If you believe a rule
  cannot be followed, stop and explain instead of suppressing it.

"""


@dataclass
class RuleDefinition:
    code: str
    name: str
    severity: str
    description: str


@dataclass
class Violation:
    line_number: int
    code: str
    message: str


@dataclass
class ParameterDefault:
    parameter_name: str
    parameter_position: int
    default_source: str
    minimum_positional_count: int


@dataclass
class KnownLibraryDefault:
    callable_name: str
    parameter_default: ParameterDefault


@dataclass
class FunctionScope:
    bound_names: set[str]
    non_variable_names: set[str]
    loaded_names: list[ast.Name]
    nested_functions: list[ast.FunctionDef | ast.AsyncFunctionDef]


@dataclass
class SuppressionComments:
    codes_by_line_number: dict[int, set[str]]
    file_wide_codes: set[str]
    comment_line_numbers: list[int]


RULES = (
    RuleDefinition("E001", "no-try-except", "error", "try/except/finally and except* are banned."),
    RuleDefinition("E002", "no-docstrings", "error", "No docstrings on modules, classes, or functions."),
    RuleDefinition(
        "E003",
        "classes-must-be-dataclasses",
        "error",
        "A class with no base class (other than object) must be a @dataclass. "
        "Classes with a base class are assumed to be required by a framework. "
        "Creating classes with type(name, bases, namespace) is banned.",
    ),
    RuleDefinition(
        "E004",
        "no-nested-comprehensions",
        "error",
        "No comprehension inside another comprehension, and no comprehension with more than one 'for'.",
    ),
    RuleDefinition("E005", "no-wildcard-imports", "error", "'from x import *' is banned."),
    RuleDefinition("E006", "use-pathlib", "error", "os.path (and posixpath/ntpath) is banned; use pathlib.Path."),
    RuleDefinition(
        "E007",
        "annotate-function-signatures",
        "error",
        "Every parameter (except a leading self/cls) and every return value must be annotated.",
    ),
    RuleDefinition("E008", "no-mutable-defaults", "error", "Defaults may not be [], {}, set literals or list()/dict()/set()."),
    RuleDefinition(
        "E009",
        "no-exception-swallowing-dodges",
        "error",
        "Bans contextlib.suppress, defining __exit__/__aexit__, return_exceptions=True, "
        "and assigning excepthook/unraisablehook.",
    ),
    RuleDefinition(
        "E010",
        "no-suppression-comments",
        "error",
        "noqa comments are reported as errors and ignored, unless --allow-suppression-comments is passed.",
    ),
    RuleDefinition(
        "E011",
        "no-dynamic-code",
        "error",
        "exec/eval/compile/__import__ are banned (they hide code from this linter).",
    ),
    RuleDefinition(
        "E012",
        "no-restated-defaults",
        "error",
        "A call to a function defined in the same file passes an argument (positional or keyword) "
        "equal to that parameter's default; omit it.",
    ),
    RuleDefinition("E013", "no-percent-formatting", "error", "Use f-strings, not '%' formatting."),
    RuleDefinition("E014", "no-str-format", "error", "Use f-strings, not str.format()."),
    RuleDefinition("E015", "no-semicolons", "error", "Multiple statements on one line with ';' are banned."),
    RuleDefinition(
        "E016",
        "no-external-scope-reads",
        "error",
        "A function reads a module-level variable or an enclosing function's local instead of "
        "receiving it as a parameter. Imports, functions, classes and UPPER_CASE constants are allowed.",
    ),
    RuleDefinition(
        "E017",
        "no-string-literal-concatenation",
        "error",
        "String literals joined with '+' or by implicit adjacency are banned; use one triple-quoted string.",
    ),
    RuleDefinition(
        "E018",
        "vertical-long-parameter-lists",
        "error",
        "A function with more than 4 parameters (not counting a leading self/cls) must list one "
        "parameter per line, none on the 'def' line.",
    ),
    RuleDefinition(
        "W006",
        "no-restated-library-defaults",
        "warning",
        "A call to a common builtin/stdlib function or method passes an argument equal to its known "
        "default (built-in table: open, print, range, enumerate, sorted, .sort, .get, int, .split, "
        ".strip, zip, .mkdir, subprocess.run). A warning because method names like .get are matched "
        "without knowing the object's type.",
    ),
    RuleDefinition(
        "W001",
        "dependency-imbalance",
        "warning",
        "A function whose count of dependencies (project functions it references plus imported names "
        "it uses), or of dependents (project functions referencing it), is far above the other "
        "functions in the file. A function named 'main' is exempt from the dependencies side, "
        "since an entry point is expected to wire everything together.",
    ),
    RuleDefinition("W002", "no-local-annotations", "warning", "Local variables should not be annotated."),
    RuleDefinition(
        "W003",
        "narrow-outputs",
        "warning",
        "Return annotation uses Optional/Union/Any/object/'|', or the function returns a value on "
        "some paths and None on others.",
    ),
    RuleDefinition("W004", "avoid-variadic-parameters", "warning", "*args/**kwargs only for genuinely variadic functions."),
    RuleDefinition("W005", "upper-case-constants", "warning", "Module-level names bound to a literal should be UPPER_CASE."),
    RuleDefinition("W007", "no-logging", "warning", "Use print, not the logging module, unless asked."),
    RuleDefinition("W008", "avoid-regex", "warning", "Prefer str methods over the re module unless a regex is needed."),
    RuleDefinition("W009", "no-single-letter-names", "warning", "No single-letter names except i, j, k."),
    RuleDefinition("W010", "no-builtin-shadowing", "warning", "Do not rebind built-in names or soft keywords (list, id, match, case, ...)."),
    RuleDefinition("P001", "long-descriptive-names", "prompt-only", "Use long, descriptive names."),
    RuleDefinition("P002", "why-comment-block-format", "prompt-only", "Why/concept comments use the '#' / '# text' / '#' block."),
    RuleDefinition("P003", "prefer-standard-library", "prompt-only", "Prefer the standard library; only pinned third-party libs."),
    RuleDefinition("P004", "yagni-kiss", "prompt-only", "YAGNI, KISS, simpler-but-longer code, long functions allowed."),
    RuleDefinition("P005", "no-long-oneliners", "prompt-only", "No long one-liners; use several plain statements."),
    RuleDefinition("P006", "few-prints", "prompt-only", "Do not add too many prints."),
)

SEVERITY_BY_CODE = {rule.code: rule.severity for rule in RULES}


FUNCTION_NODE_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)
SCOPE_NODE_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
COMPREHENSION_NODE_TYPES = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
TRY_NODE_TYPES = tuple(getattr(ast, type_name) for type_name in ("Try", "TryStar") if hasattr(ast, type_name))
BANNED_PATH_MODULES = ("os.path", "posixpath", "ntpath", "genericpath")
EXCEPTION_HOOK_ATTRIBUTES = ("excepthook", "unraisablehook")
MUTABLE_FACTORY_NAMES = ("list", "dict", "set")
BUILTIN_NAMES = {name for name in dir(builtins) if not name.startswith("_")} | {"match", "case"}
ALLOWED_SINGLE_LETTER_NAMES = ("i", "j", "k")
STATEMENT_LIST_FIELDS = ("body", "orelse", "finalbody")
BANNED_DYNAMIC_CODE_FUNCTIONS = ("exec", "eval", "compile", "__import__")
WIDE_TYPE_NAMES = ("Optional", "Union", "Any", "object")
WIDE_TYPE_STRING_MARKERS = ("Optional", "Union", "Any", "object", "|")
MINIMUM_FUNCTIONS_FOR_DEPENDENCY_CHECK = 4
DEPENDENCY_OUTLIER_RATIO = 3.0
DEPENDENCY_OUTLIER_MINIMUM_GAP = 4
ENTRY_POINT_FUNCTION_NAME = "main"
KEYWORD_ONLY_POSITION = -1
MAXIMUM_INLINE_PARAMETER_COUNT = 4
FSTRING_START_TOKEN_TYPE = getattr(tokenize, "FSTRING_START", -1)
FSTRING_END_TOKEN_TYPE = getattr(tokenize, "FSTRING_END", -1)

#
# A callable name starting with "." matches any method with that name, whatever
# the object is. A parameter position of KEYWORD_ONLY_POSITION means the
# argument is only checked when passed by keyword. minimum_positional_count
# covers range(), where the first positional is only "start" when at least two
# positionals are given.
#
KNOWN_LIBRARY_DEFAULTS = (
    KnownLibraryDefault("open", ParameterDefault("mode", 1, "'r'", 0)),
    KnownLibraryDefault("print", ParameterDefault("sep", KEYWORD_ONLY_POSITION, "' '", 0)),
    KnownLibraryDefault("print", ParameterDefault("end", KEYWORD_ONLY_POSITION, "'\\n'", 0)),
    KnownLibraryDefault("print", ParameterDefault("flush", KEYWORD_ONLY_POSITION, "False", 0)),
    KnownLibraryDefault("range", ParameterDefault("start", 0, "0", 2)),
    KnownLibraryDefault("range", ParameterDefault("step", 2, "1", 3)),
    KnownLibraryDefault("enumerate", ParameterDefault("start", 1, "0", 0)),
    KnownLibraryDefault("sorted", ParameterDefault("key", KEYWORD_ONLY_POSITION, "None", 0)),
    KnownLibraryDefault("sorted", ParameterDefault("reverse", KEYWORD_ONLY_POSITION, "False", 0)),
    KnownLibraryDefault(".sort", ParameterDefault("key", KEYWORD_ONLY_POSITION, "None", 0)),
    KnownLibraryDefault(".sort", ParameterDefault("reverse", KEYWORD_ONLY_POSITION, "False", 0)),
    KnownLibraryDefault(".get", ParameterDefault("default", 1, "None", 0)),
    KnownLibraryDefault("int", ParameterDefault("base", 1, "10", 0)),
    KnownLibraryDefault(".split", ParameterDefault("sep", 0, "None", 0)),
    KnownLibraryDefault(".split", ParameterDefault("maxsplit", 1, "-1", 0)),
    KnownLibraryDefault(".strip", ParameterDefault("chars", 0, "None", 0)),
    KnownLibraryDefault(".lstrip", ParameterDefault("chars", 0, "None", 0)),
    KnownLibraryDefault(".rstrip", ParameterDefault("chars", 0, "None", 0)),
    KnownLibraryDefault("zip", ParameterDefault("strict", KEYWORD_ONLY_POSITION, "False", 0)),
    KnownLibraryDefault(".mkdir", ParameterDefault("parents", KEYWORD_ONLY_POSITION, "False", 0)),
    KnownLibraryDefault(".mkdir", ParameterDefault("exist_ok", KEYWORD_ONLY_POSITION, "False", 0)),
    KnownLibraryDefault("subprocess.run", ParameterDefault("check", KEYWORD_ONLY_POSITION, "False", 0)),
)


def find_docstring_line_number(body: list[ast.stmt]) -> int:
    if not body:
        return 0
    first_statement = body[0]
    is_string_expression = (
        isinstance(first_statement, ast.Expr)
        and isinstance(first_statement.value, ast.Constant)
        and isinstance(first_statement.value.value, str)
    )
    if is_string_expression:
        return first_statement.lineno
    return 0


def collect_own_body_nodes(function_node: ast.AST) -> list[ast.AST]:
    #
    # Nested functions, lambdas and classes are their own scope: their returns
    # and annotations belong to them, not to the enclosing function.
    #
    collected_nodes = []
    pending_nodes = list(ast.iter_child_nodes(function_node))
    while pending_nodes:
        current_node = pending_nodes.pop()
        collected_nodes.append(current_node)
        if isinstance(current_node, SCOPE_NODE_TYPES):
            continue
        pending_nodes.extend(ast.iter_child_nodes(current_node))
    return collected_nodes


def check_no_try_except(tree: ast.Module) -> list[Violation]:
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, TRY_NODE_TYPES):
            violations.append(
                Violation(
                    node.lineno,
                    "E001",
                    "try/except is banned; write the line plainly with a TRY-EXCEPT-CANDIDATE comment block above it",
                )
            )
    return violations


def check_no_docstrings(tree: ast.Module) -> list[Violation]:
    violations = []
    module_docstring_line = find_docstring_line_number(tree.body)
    if module_docstring_line:
        violations.append(Violation(module_docstring_line, "E002", "module docstring is not allowed"))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        docstring_line = find_docstring_line_number(node.body)
        if docstring_line:
            violations.append(Violation(docstring_line, "E002", f"docstring on '{node.name}' is not allowed"))
    return violations


def collect_decorator_names(class_node: ast.ClassDef) -> set[str]:
    decorator_names = set()
    for decorator in class_node.decorator_list:
        decorator_target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(decorator_target, ast.Name):
            decorator_names.add(decorator_target.id)
        elif isinstance(decorator_target, ast.Attribute):
            decorator_names.add(decorator_target.attr)
    return decorator_names


def check_classes_are_dataclasses(tree: ast.Module) -> list[Violation]:
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            has_real_base_class = False
            for base_class in node.bases:
                if not (isinstance(base_class, ast.Name) and base_class.id == "object"):
                    has_real_base_class = True
            if not has_real_base_class and "dataclass" not in collect_decorator_names(node):
                violations.append(
                    Violation(node.lineno, "E003", f"class '{node.name}' must be a @dataclass (no framework base class)")
                )
        is_dynamic_class_creation = (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "type"
            and len(node.args) == 3
        )
        if is_dynamic_class_creation:
            violations.append(Violation(node.lineno, "E003", "creating a class with type(...) is banned"))
    return violations


def contains_comprehension(node: ast.AST) -> bool:
    for child_node in ast.walk(node):
        if isinstance(child_node, COMPREHENSION_NODE_TYPES):
            return True
    return False


def check_no_nested_comprehensions(tree: ast.Module) -> list[Violation]:
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, COMPREHENSION_NODE_TYPES):
            continue
        if len(node.generators) > 1:
            violations.append(Violation(node.lineno, "E004", "comprehension with more than one 'for'; use a plain loop"))
            continue
        if isinstance(node, ast.DictComp):
            inner_parts = [node.key, node.value]
        else:
            inner_parts = [node.elt]
        only_generator = node.generators[0]
        inner_parts.append(only_generator.iter)
        inner_parts.extend(only_generator.ifs)
        for inner_part in inner_parts:
            if contains_comprehension(inner_part):
                violations.append(Violation(node.lineno, "E004", "nested comprehension; use a plain loop"))
                break
    return violations


def check_no_wildcard_imports(tree: ast.Module) -> list[Violation]:
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        for imported_alias in node.names:
            if imported_alias.name == "*":
                violations.append(Violation(node.lineno, "E005", f"wildcard import from '{node.module}' is banned"))
    return violations


def check_use_pathlib(tree: ast.Module) -> list[Violation]:
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported_alias in node.names:
                if imported_alias.name in BANNED_PATH_MODULES:
                    violations.append(Violation(node.lineno, "E006", f"import '{imported_alias.name}' is banned; use pathlib"))
        elif isinstance(node, ast.ImportFrom):
            if node.module in BANNED_PATH_MODULES:
                violations.append(Violation(node.lineno, "E006", f"import from '{node.module}' is banned; use pathlib"))
            elif node.module == "os":
                for imported_alias in node.names:
                    if imported_alias.name == "path":
                        violations.append(Violation(node.lineno, "E006", "'from os import path' is banned; use pathlib"))
        elif isinstance(node, ast.Attribute):
            if node.attr == "path" and isinstance(node.value, ast.Name) and node.value.id == "os":
                violations.append(Violation(node.lineno, "E006", "'os.path' is banned; use pathlib"))
    return violations


def collect_unannotated_parameter_names(function_node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    missing_names = []
    positional_parameters = list(function_node.args.posonlyargs) + list(function_node.args.args)
    for parameter_index, parameter in enumerate(positional_parameters):
        if parameter_index == 0 and parameter.arg in ("self", "cls"):
            continue
        if parameter.annotation is None:
            missing_names.append(parameter.arg)
    for parameter in function_node.args.kwonlyargs:
        if parameter.annotation is None:
            missing_names.append(parameter.arg)
    if function_node.args.vararg is not None and function_node.args.vararg.annotation is None:
        missing_names.append("*" + function_node.args.vararg.arg)
    if function_node.args.kwarg is not None and function_node.args.kwarg.annotation is None:
        missing_names.append("**" + function_node.args.kwarg.arg)
    return missing_names


def check_function_signatures_annotated(tree: ast.Module) -> list[Violation]:
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, FUNCTION_NODE_TYPES):
            continue
        for parameter_name in collect_unannotated_parameter_names(node):
            violations.append(
                Violation(node.lineno, "E007", f"parameter '{parameter_name}' of '{node.name}' needs a type annotation")
            )
        if node.returns is None:
            violations.append(Violation(node.lineno, "E007", f"'{node.name}' needs a return type annotation"))
    return violations


def check_no_mutable_defaults(tree: ast.Module) -> list[Violation]:
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, FUNCTION_NODE_TYPES):
            continue
        for default_value in list(node.args.defaults) + list(node.args.kw_defaults):
            if default_value is None:
                continue
            is_mutable_literal = isinstance(default_value, (ast.List, ast.Dict, ast.Set))
            is_mutable_factory_call = (
                isinstance(default_value, ast.Call)
                and isinstance(default_value.func, ast.Name)
                and default_value.func.id in MUTABLE_FACTORY_NAMES
            )
            if is_mutable_literal or is_mutable_factory_call:
                violations.append(
                    Violation(default_value.lineno, "E008", f"mutable default in '{node.name}'; default to None instead")
                )
    return violations


def collect_assignment_targets(node: ast.AST) -> list[ast.expr]:
    if isinstance(node, ast.Assign):
        return list(node.targets)
    if isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        return [node.target]
    return []


def check_no_exception_swallowing_dodges(tree: ast.Module) -> list[Violation]:
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "contextlib":
            for imported_alias in node.names:
                if imported_alias.name == "suppress":
                    violations.append(Violation(node.lineno, "E009", "contextlib.suppress is banned"))
        if isinstance(node, ast.Attribute) and node.attr == "suppress":
            if isinstance(node.value, ast.Name) and node.value.id == "contextlib":
                violations.append(Violation(node.lineno, "E009", "contextlib.suppress is banned"))
        if isinstance(node, FUNCTION_NODE_TYPES) and node.name in ("__exit__", "__aexit__"):
            violations.append(Violation(node.lineno, "E009", f"defining {node.name} is banned (it can swallow exceptions)"))
        if isinstance(node, ast.keyword) and node.arg == "return_exceptions":
            if isinstance(node.value, ast.Constant) and node.value.value is True:
                violations.append(Violation(node.lineno, "E009", "return_exceptions=True is banned"))
        for assignment_target in collect_assignment_targets(node):
            if isinstance(assignment_target, ast.Attribute) and assignment_target.attr in EXCEPTION_HOOK_ATTRIBUTES:
                violations.append(Violation(node.lineno, "E009", f"assigning {assignment_target.attr} is banned"))
    return violations


def check_no_dynamic_code(tree: ast.Module) -> list[Violation]:
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in BANNED_DYNAMIC_CODE_FUNCTIONS:
                violations.append(Violation(node.lineno, "E011", f"{node.func.id}() is banned"))
    return violations


def normalize_expression_source(expression_source: str) -> str:
    return ast.dump(ast.parse(expression_source, mode="eval").body)


def collect_project_parameter_defaults(tree: ast.Module) -> dict[str, list[ParameterDefault]]:
    #
    # Defaults are stored as source text and compared after re-parsing, so
    # `None`, `-1`, `"r"` or a named constant all compare structurally rather
    # than by how they happen to be spelled.
    #
    defaults_by_function_name = {}
    for node in ast.walk(tree):
        if not isinstance(node, FUNCTION_NODE_TYPES):
            continue
        positional_parameters = list(node.args.posonlyargs) + list(node.args.args)
        first_defaulted_position = len(positional_parameters) - len(node.args.defaults)
        parameter_defaults = []
        for default_offset, default_node in enumerate(node.args.defaults):
            parameter_position = first_defaulted_position + default_offset
            parameter_defaults.append(
                ParameterDefault(
                    positional_parameters[parameter_position].arg, parameter_position, ast.unparse(default_node), 0
                )
            )
        for keyword_parameter, default_node in zip(node.args.kwonlyargs, node.args.kw_defaults):
            if default_node is not None:
                parameter_defaults.append(
                    ParameterDefault(keyword_parameter.arg, KEYWORD_ONLY_POSITION, ast.unparse(default_node), 0)
                )
        defaults_by_function_name[node.name] = parameter_defaults
    return defaults_by_function_name


def call_restates_default(call_node: ast.Call, parameter_default: ParameterDefault) -> bool:
    normalized_default = normalize_expression_source(parameter_default.default_source)
    for keyword_argument in call_node.keywords:
        if keyword_argument.arg == parameter_default.parameter_name:
            return ast.dump(keyword_argument.value) == normalized_default
    parameter_position = parameter_default.parameter_position
    if parameter_position == KEYWORD_ONLY_POSITION or len(call_node.args) <= parameter_position:
        return False
    if len(call_node.args) < parameter_default.minimum_positional_count:
        return False
    for positional_argument in call_node.args[: parameter_position + 1]:
        if isinstance(positional_argument, ast.Starred):
            return False
    return ast.dump(call_node.args[parameter_position]) == normalized_default


def describe_called_function(call_node: ast.Call) -> str:
    if isinstance(call_node.func, ast.Name):
        return call_node.func.id
    if isinstance(call_node.func, ast.Attribute) and isinstance(call_node.func.value, ast.Name):
        return f"{call_node.func.value.id}.{call_node.func.attr}"
    if isinstance(call_node.func, ast.Attribute):
        return "." + call_node.func.attr
    return ""


def library_default_applies(known_library_default: KnownLibraryDefault, call_node: ast.Call) -> bool:
    callable_name = known_library_default.callable_name
    if callable_name.startswith("."):
        return isinstance(call_node.func, ast.Attribute) and call_node.func.attr == callable_name[1:]
    return describe_called_function(call_node) == callable_name


def check_no_restated_defaults(tree: ast.Module) -> list[Violation]:
    violations = []
    defaults_by_function_name = collect_project_parameter_defaults(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        called_name = describe_called_function(node)
        if isinstance(node.func, ast.Name) and called_name in defaults_by_function_name:
            for parameter_default in defaults_by_function_name[called_name]:
                if call_restates_default(node, parameter_default):
                    violations.append(
                        Violation(
                            node.lineno,
                            "E012",
                            f"argument '{parameter_default.parameter_name}' passed to '{called_name}' "
                            "equals its default; omit it",
                        )
                    )
            continue
        for known_library_default in KNOWN_LIBRARY_DEFAULTS:
            if not library_default_applies(known_library_default, node):
                continue
            if call_restates_default(node, known_library_default.parameter_default):
                violations.append(
                    Violation(
                        node.lineno,
                        "W006",
                        f"argument '{known_library_default.parameter_default.parameter_name}' passed to "
                        f"'{called_name}' equals its default; omit it",
                    )
                )
    return violations


def collect_imported_names(tree: ast.Module) -> set[str]:
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported_alias in node.names:
                imported_names.add(imported_alias.asname or imported_alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for imported_alias in node.names:
                if imported_alias.name != "*":
                    imported_names.add(imported_alias.asname or imported_alias.name)
    return imported_names


def collect_referenced_names(function_node: ast.AST) -> set[str]:
    referenced_names = set()
    for node in ast.walk(function_node):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            referenced_names.add(node.id)
    return referenced_names


def find_dependency_outliers(
    count_by_function_name: dict[str, int],
    line_number_by_function_name: dict[str, int],
    problem_description: str,
) -> list[Violation]:
    #
    # Each function is compared to the average of the *other* functions, so a
    # single hub function cannot inflate the baseline it is judged against.
    #
    violations = []
    total_count = sum(count_by_function_name.values())
    other_function_count = len(count_by_function_name) - 1
    for function_name, function_count in count_by_function_name.items():
        average_of_others = (total_count - function_count) / other_function_count
        outlier_threshold = max(
            average_of_others * DEPENDENCY_OUTLIER_RATIO,
            average_of_others + DEPENDENCY_OUTLIER_MINIMUM_GAP,
        )
        if function_count > outlier_threshold:
            violations.append(
                Violation(
                    line_number_by_function_name[function_name],
                    "W001",
                    f"'{function_name}' {problem_description} {function_count} "
                    f"(other functions average {average_of_others:.1f})",
                )
            )
    return violations


def check_dependency_imbalance(tree: ast.Module) -> list[Violation]:
    function_nodes = [node for node in ast.walk(tree) if isinstance(node, FUNCTION_NODE_TYPES)]
    if len(function_nodes) < MINIMUM_FUNCTIONS_FOR_DEPENDENCY_CHECK:
        return []
    project_function_names = {function_node.name for function_node in function_nodes}
    imported_names = collect_imported_names(tree)
    line_number_by_function_name = {}
    dependency_count_by_function_name = {}
    dependent_count_by_function_name = dict.fromkeys(project_function_names, 0)
    for function_node in function_nodes:
        referenced_names = collect_referenced_names(function_node)
        referenced_project_functions = (referenced_names & project_function_names) - {function_node.name}
        used_imported_names = referenced_names & imported_names
        line_number_by_function_name[function_node.name] = function_node.lineno
        dependency_count_by_function_name[function_node.name] = len(referenced_project_functions) + len(
            used_imported_names
        )
        for referenced_function_name in referenced_project_functions:
            dependent_count_by_function_name[referenced_function_name] += 1
    #
    # An entry point exists to wire everything together, so being a hub is its
    # job; judging it would only teach the AI to hide the wiring elsewhere.
    #
    if ENTRY_POINT_FUNCTION_NAME in dependency_count_by_function_name:
        del dependency_count_by_function_name[ENTRY_POINT_FUNCTION_NAME]
    violations = find_dependency_outliers(
        dependency_count_by_function_name, line_number_by_function_name, "has many dependencies:"
    )
    violations.extend(
        find_dependency_outliers(
            dependent_count_by_function_name, line_number_by_function_name, "has many dependents:"
        )
    )
    return violations


def check_no_local_annotations(tree: ast.Module) -> list[Violation]:
    violations = []
    for function_node in ast.walk(tree):
        if not isinstance(function_node, FUNCTION_NODE_TYPES):
            continue
        for body_node in collect_own_body_nodes(function_node):
            if isinstance(body_node, ast.AnnAssign) and isinstance(body_node.target, ast.Name):
                violations.append(
                    Violation(body_node.lineno, "W002", f"local '{body_node.target.id}' should not be annotated")
                )
    return violations


def annotation_is_wide(annotation: ast.expr) -> bool:
    for node in ast.walk(annotation):
        if isinstance(node, ast.Name) and node.id in WIDE_TYPE_NAMES:
            return True
        if isinstance(node, ast.Attribute) and node.attr in WIDE_TYPE_NAMES:
            return True
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return True
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for type_marker in WIDE_TYPE_STRING_MARKERS:
                if type_marker in node.value:
                    return True
    return False


def has_mixed_return_values(function_node: ast.AST) -> bool:
    body_nodes = collect_own_body_nodes(function_node)
    returns_with_value = 0
    returns_without_value = 0
    for body_node in body_nodes:
        if isinstance(body_node, (ast.Yield, ast.YieldFrom)):
            return False
        if not isinstance(body_node, ast.Return):
            continue
        is_empty_return = body_node.value is None or (
            isinstance(body_node.value, ast.Constant) and body_node.value.value is None
        )
        if is_empty_return:
            returns_without_value += 1
        else:
            returns_with_value += 1
    return returns_with_value > 0 and returns_without_value > 0


def check_narrow_outputs(tree: ast.Module) -> list[Violation]:
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, FUNCTION_NODE_TYPES):
            continue
        if node.returns is not None and annotation_is_wide(node.returns):
            violations.append(Violation(node.lineno, "W003", f"'{node.name}' has a wide return type; return one narrow type"))
        if has_mixed_return_values(node):
            violations.append(Violation(node.lineno, "W003", f"'{node.name}' returns a value on some paths and None on others"))
    return violations


def check_avoid_variadic_parameters(tree: ast.Module) -> list[Violation]:
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, FUNCTION_NODE_TYPES):
            continue
        if node.args.vararg is not None or node.args.kwarg is not None:
            violations.append(Violation(node.lineno, "W004", f"'{node.name}' uses *args/**kwargs; is it truly variadic?"))
    return violations


def check_upper_case_constants(tree: ast.Module) -> list[Violation]:
    violations = []
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        for assignment_target in node.targets:
            if not isinstance(assignment_target, ast.Name):
                continue
            target_name = assignment_target.id
            if target_name != target_name.upper() and not target_name.startswith("__"):
                violations.append(Violation(node.lineno, "W005", f"module-level constant '{target_name}' should be UPPER_CASE"))
    return violations


def check_no_string_formatting(tree: ast.Module) -> list[Violation]:
    violations = []
    for node in ast.walk(tree):
        is_percent_formatting = (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Mod)
            and isinstance(node.left, ast.Constant)
            and isinstance(node.left.value, str)
        )
        if is_percent_formatting:
            violations.append(Violation(node.lineno, "E013", "'%' formatting is banned; use an f-string"))
        is_str_format_call = (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "format"
            and isinstance(node.func.value, ast.Constant)
            and isinstance(node.func.value.value, str)
        )
        if is_str_format_call:
            violations.append(Violation(node.lineno, "E014", "str.format() is banned; use an f-string"))
    return violations


def check_no_semicolons(tree: ast.Module) -> list[Violation]:
    #
    # Two consecutive statements in one block that start on the same line can
    # only have been joined with ';'.
    #
    violations = []
    for node in ast.walk(tree):
        for field_name in STATEMENT_LIST_FIELDS:
            statements = getattr(node, field_name, None)
            if not isinstance(statements, list):
                continue
            for previous_statement, current_statement in zip(statements, statements[1:]):
                if isinstance(current_statement, ast.stmt) and current_statement.lineno == previous_statement.end_lineno:
                    violations.append(Violation(current_statement.lineno, "E015", "';' between statements is banned"))
    return violations


def collect_parameter_nodes(arguments: ast.arguments) -> list[ast.arg]:
    parameter_nodes = list(arguments.posonlyargs) + list(arguments.args) + list(arguments.kwonlyargs)
    if arguments.vararg is not None:
        parameter_nodes.append(arguments.vararg)
    if arguments.kwarg is not None:
        parameter_nodes.append(arguments.kwarg)
    return parameter_nodes


def read_function_scope(function_node: ast.FunctionDef | ast.AsyncFunctionDef) -> FunctionScope:
    #
    # Nested functions and classes are not entered: they are their own scope,
    # checked separately. Lambdas are entered because they share the reads of
    # the function that contains them.
    #
    scope = FunctionScope({parameter.arg for parameter in collect_parameter_nodes(function_node.args)}, set(), [], [])
    pending_nodes = list(function_node.body)
    while pending_nodes:
        current_node = pending_nodes.pop()
        if isinstance(current_node, FUNCTION_NODE_TYPES):
            scope.non_variable_names.add(current_node.name)
            scope.nested_functions.append(current_node)
            continue
        if isinstance(current_node, ast.ClassDef):
            scope.non_variable_names.add(current_node.name)
            continue
        if isinstance(current_node, ast.Lambda):
            scope.bound_names.update(parameter.arg for parameter in collect_parameter_nodes(current_node.args))
        elif isinstance(current_node, ast.Name) and isinstance(current_node.ctx, ast.Store):
            scope.bound_names.add(current_node.id)
        elif isinstance(current_node, ast.Name) and isinstance(current_node.ctx, ast.Load):
            scope.loaded_names.append(current_node)
        elif isinstance(current_node, ast.alias):
            scope.non_variable_names.add(current_node.asname or current_node.name.split(".")[0])
        elif isinstance(current_node, ast.ExceptHandler) and current_node.name:
            scope.bound_names.add(current_node.name)
        elif isinstance(current_node, (ast.MatchAs, ast.MatchStar)) and current_node.name:
            scope.bound_names.add(current_node.name)
        pending_nodes.extend(ast.iter_child_nodes(current_node))
    return scope


def collect_module_variable_names(tree: ast.Module) -> set[str]:
    module_variable_names = set()
    for node in tree.body:
        for assignment_target in collect_assignment_targets(node):
            for target_node in ast.walk(assignment_target):
                if isinstance(target_node, ast.Name) and target_node.id != target_node.id.upper():
                    module_variable_names.add(target_node.id)
    return module_variable_names


def find_external_reads(
    function_node: ast.FunctionDef | ast.AsyncFunctionDef,
    enclosing_variable_names: set[str],
    module_variable_names: set[str],
) -> list[Violation]:
    violations = []
    scope = read_function_scope(function_node)
    external_names = enclosing_variable_names | module_variable_names
    for loaded_name in scope.loaded_names:
        if loaded_name.id in scope.bound_names or loaded_name.id in scope.non_variable_names:
            continue
        if loaded_name.id in external_names:
            violations.append(
                Violation(
                    loaded_name.lineno,
                    "E016",
                    f"'{function_node.name}' reads '{loaded_name.id}' from an outer scope; pass it as a parameter",
                )
            )
    nested_enclosing_names = enclosing_variable_names | scope.bound_names
    for nested_function in scope.nested_functions:
        violations.extend(find_external_reads(nested_function, nested_enclosing_names, module_variable_names))
    return violations


def collect_outermost_functions(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    outermost_functions = []
    pending_nodes = [tree]
    while pending_nodes:
        current_node = pending_nodes.pop()
        for child_node in ast.iter_child_nodes(current_node):
            if isinstance(child_node, FUNCTION_NODE_TYPES):
                outermost_functions.append(child_node)
            else:
                pending_nodes.append(child_node)
    return outermost_functions


def check_no_external_scope_reads(tree: ast.Module) -> list[Violation]:
    violations = []
    module_variable_names = collect_module_variable_names(tree)
    for function_node in collect_outermost_functions(tree):
        violations.extend(find_external_reads(function_node, set(), module_variable_names))
    return violations


def is_string_literal_expression(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str)
    if isinstance(node, ast.JoinedStr):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return is_string_literal_expression(node.left) and is_string_literal_expression(node.right)
    return False


def check_no_string_literal_addition(tree: ast.Module) -> list[Violation]:
    violations = []
    for node in ast.walk(tree):
        is_literal_addition = (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Add)
            and is_string_literal_expression(node.left)
            and is_string_literal_expression(node.right)
        )
        if is_literal_addition:
            violations.append(Violation(node.lineno, "E017", "string literals joined with '+'; use one triple-quoted string"))
    return violations


def check_no_implicit_string_concatenation(source_text: str) -> list[Violation]:
    #
    # The AST merges adjacent literals into one node, so only the token stream
    # still shows that two string literals were written next to each other.
    #
    violations = []
    previous_token_ended_string = False
    for token in tokenize.generate_tokens(io.StringIO(source_text).readline):
        if token.type in (tokenize.NL, tokenize.COMMENT):
            continue
        starts_string = token.type in (tokenize.STRING, FSTRING_START_TOKEN_TYPE)
        if starts_string and previous_token_ended_string:
            violations.append(
                Violation(token.start[0], "E017", "adjacent string literals concatenated implicitly; use one triple-quoted string")
            )
        previous_token_ended_string = token.type in (tokenize.STRING, FSTRING_END_TOKEN_TYPE)
    return violations


def check_long_parameter_lists_vertical(tree: ast.Module) -> list[Violation]:
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, FUNCTION_NODE_TYPES):
            continue
        parameter_nodes = collect_parameter_nodes(node.args)
        counted_parameter_count = len(parameter_nodes)
        if parameter_nodes and parameter_nodes[0].arg in ("self", "cls"):
            counted_parameter_count -= 1
        if counted_parameter_count <= MAXIMUM_INLINE_PARAMETER_COUNT:
            continue
        parameter_line_numbers = [parameter.lineno for parameter in parameter_nodes]
        has_shared_line = len(set(parameter_line_numbers)) != len(parameter_line_numbers)
        if has_shared_line or min(parameter_line_numbers) == node.lineno:
            violations.append(
                Violation(
                    node.lineno,
                    "E018",
                    f"'{node.name}' has {counted_parameter_count} parameters; put each on its own line",
                )
            )
    return violations


def check_no_logging_or_regex(tree: ast.Module) -> list[Violation]:
    violations = []
    for node in ast.walk(tree):
        imported_modules = []
        if isinstance(node, ast.Import):
            imported_modules = [imported_alias.name.split(".")[0] for imported_alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules = [node.module.split(".")[0]]
        if "logging" in imported_modules:
            violations.append(Violation(node.lineno, "W007", "logging module used; use print unless asked"))
        if "re" in imported_modules:
            violations.append(Violation(node.lineno, "W008", "re module used; prefer str methods unless a regex is needed"))
    return violations


def collect_declared_names(tree: ast.Module) -> list[tuple[int, str]]:
    declared_names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            declared_names.append((node.lineno, node.id))
        elif isinstance(node, ast.arg):
            declared_names.append((node.lineno, node.arg))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            declared_names.append((node.lineno, node.name))
    return declared_names


def check_no_single_letter_names(tree: ast.Module) -> list[Violation]:
    violations = []
    for line_number, declared_name in collect_declared_names(tree):
        if len(declared_name) == 1 and declared_name.isalpha() and declared_name not in ALLOWED_SINGLE_LETTER_NAMES:
            violations.append(Violation(line_number, "W009", f"single-letter name '{declared_name}'; use a descriptive name"))
    return violations


def check_no_builtin_shadowing(tree: ast.Module) -> list[Violation]:
    violations = []
    for line_number, declared_name in collect_declared_names(tree):
        if declared_name in BUILTIN_NAMES:
            violations.append(Violation(line_number, "W010", f"'{declared_name}' shadows a built-in name"))
    return violations


ALL_CHECKS = (
    check_no_try_except,
    check_no_docstrings,
    check_classes_are_dataclasses,
    check_no_nested_comprehensions,
    check_no_wildcard_imports,
    check_use_pathlib,
    check_function_signatures_annotated,
    check_no_mutable_defaults,
    check_no_exception_swallowing_dodges,
    check_no_dynamic_code,
    check_no_restated_defaults,
    check_dependency_imbalance,
    check_no_local_annotations,
    check_narrow_outputs,
    check_avoid_variadic_parameters,
    check_upper_case_constants,
    check_no_string_formatting,
    check_no_semicolons,
    check_no_external_scope_reads,
    check_no_string_literal_addition,
    check_long_parameter_lists_vertical,
    check_no_logging_or_regex,
    check_no_single_letter_names,
    check_no_builtin_shadowing,
)

ALL_SOURCE_CHECKS = (check_no_implicit_string_concatenation,)


def is_suppression_comment(comment_text: str) -> bool:
    lowered_text = comment_text.lower()
    comment_body = lowered_text.lstrip("#").strip()
    return comment_body.startswith("noqa") or "# noqa" in lowered_text or "#noqa" in lowered_text


def parse_suppressed_codes(comment_text: str) -> set[str]:
    noqa_position = comment_text.lower().index("noqa")
    remainder_text = comment_text[noqa_position + len("noqa"):].strip()
    if not remainder_text.startswith(":"):
        return {"ALL"}
    return {code_token.upper() for code_token in remainder_text[1:].replace(",", " ").split()}


def read_suppression_comments(source_text: str) -> SuppressionComments:
    #
    # Tokenizing (instead of scanning raw lines) means text that merely looks
    # like a suppression comment inside a string literal is not treated as one.
    #
    codes_by_line_number = {}
    file_wide_codes = set()
    comment_line_numbers = []
    source_lines = source_text.splitlines()
    for token in tokenize.generate_tokens(io.StringIO(source_text).readline):
        if token.type != tokenize.COMMENT or not is_suppression_comment(token.string):
            continue
        line_number = token.start[0]
        comment_line_numbers.append(line_number)
        suppressed_codes = parse_suppressed_codes(token.string)
        if source_lines[line_number - 1].strip().startswith("#"):
            file_wide_codes.update(suppressed_codes)
        else:
            codes_by_line_number[line_number] = suppressed_codes
    return SuppressionComments(codes_by_line_number, file_wide_codes, comment_line_numbers)


def is_suppressed(suppression_comments: SuppressionComments, violation: Violation) -> bool:
    line_codes = suppression_comments.codes_by_line_number.get(violation.line_number, set())
    all_applicable_codes = line_codes | suppression_comments.file_wide_codes
    return violation.code in all_applicable_codes or "ALL" in all_applicable_codes


def violation_sort_key(violation: Violation) -> tuple[int, str]:
    return (violation.line_number, violation.code)


def lint_source(source_text: str, python_version: tuple[int, int], allow_suppression_comments: bool) -> list[Violation]:
    #
    # TRY-EXCEPT-CANDIDATE: ast.parse raises SyntaxError on invalid code. Letting
    # it crash is acceptable: Python prints the file position and exits non-zero.
    #
    tree = ast.parse(source_text, feature_version=python_version)
    suppression_comments = read_suppression_comments(source_text)
    violations = []
    for check_function in ALL_CHECKS:
        violations.extend(check_function(tree))
    for source_check_function in ALL_SOURCE_CHECKS:
        violations.extend(source_check_function(source_text))
    if not allow_suppression_comments:
        for comment_line_number in suppression_comments.comment_line_numbers:
            violations.append(
                Violation(comment_line_number, "E010", "suppression comments are not allowed; fix the code instead")
            )
        return sorted(violations, key=violation_sort_key)
    kept_violations = [violation for violation in violations if not is_suppressed(suppression_comments, violation)]
    return sorted(kept_violations, key=violation_sort_key)


def parse_python_version(version_text: str) -> tuple[int, int]:
    if not version_text:
        return (sys.version_info.major, sys.version_info.minor)
    major_text, minor_text = version_text.split(".")[:2]
    return (int(major_text), int(minor_text))


def format_rule_list() -> str:
    output_lines = []
    for rule in RULES:
        output_lines.append(f"{rule.code}  {rule.severity:<12} {rule.name}")
        output_lines.append(f"      {rule.description}")
    return "\n".join(output_lines)


def lint_file_and_report(
    file_path: Path, python_version: tuple[int, int], allow_suppression_comments: bool, warnings_as_errors: bool
) -> int:
    violations = lint_source(file_path.read_text(), python_version, allow_suppression_comments)
    error_count = 0
    warning_count = 0
    for violation in violations:
        severity = SEVERITY_BY_CODE[violation.code]
        print(f"{file_path}:{violation.line_number}: [{violation.code}] {severity}: {violation.message}")
        if severity == "error":
            error_count += 1
        else:
            warning_count += 1
    print(f"\n{error_count} error(s), {warning_count} warning(s) in {file_path}")
    if error_count > 0 or (warnings_as_errors and warning_count > 0):
        return 1
    return 0


def build_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description="AST linter for the AI code-generation style rules.")
    argument_parser.add_argument("file_path", nargs="?", default="", help="single .py file to lint")
    argument_parser.add_argument("--python-version", default="", help="target version, e.g. 3.12 (default: running Python)")
    argument_parser.add_argument(
        "--allow-suppression-comments", action="store_true", help="honor noqa comments instead of reporting them"
    )
    argument_parser.add_argument("--warnings-as-errors", action="store_true", help="make warnings fail the run too")
    argument_parser.add_argument("--print-prompt", action="store_true", help="print the prompt prefix and exit")
    argument_parser.add_argument("--list-rules", action="store_true", help="print every rule and exit")
    return argument_parser


def main() -> int:
    argument_parser = build_argument_parser()
    arguments = argument_parser.parse_args()
    if arguments.print_prompt:
        print(PROMPT_PREFIX)
        return 0
    if arguments.list_rules:
        print(format_rule_list())
        return 0
    if not arguments.file_path:
        argument_parser.error("a file path is required unless --print-prompt or --list-rules is used")
    return lint_file_and_report(
        Path(arguments.file_path),
        parse_python_version(arguments.python_version),
        arguments.allow_suppression_comments,
        arguments.warnings_as_errors,
    )


if __name__ == "__main__":
    sys.exit(main())
