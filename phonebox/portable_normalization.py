"""Compile saved letter preprocessing into a standard-library program."""

import re
import unicodedata


class PortableNormalizationError(ValueError):
    """Raised when saved preprocessing cannot be reproduced portably."""


def compile_metadata_preprocessing(metadata):
    """Compile a present snapshot; reserve ``None`` for legacy metadata."""
    if "letter_preprocessing" not in metadata:
        return None
    return compile_letter_preprocessing(metadata["letter_preprocessing"])


_NORMAL_FORMS = {"NFC", "NFD", "NFKC", "NFKD"}
_UNICODE_SCALAR_RE = re.compile(r"\\u([0-9A-Fa-f]{4})")
_SHIPPED_FILTERS = {
    "[^-.'[:l:]] remove": "-.'",
    "[^-.[:l:]] remove": "-.",
}


def _exact_scalar(value):
    """Return one literal scalar from the narrow supported ICU syntax."""
    escaped = _UNICODE_SCALAR_RE.fullmatch(value)
    if escaped:
        scalar = chr(int(escaped.group(1), 16))
        return scalar if not 0xD800 <= ord(scalar) <= 0xDFFF else None
    if len(value) == 1 and value.isalpha():
        return value
    return None


def _rule_statements(rules):
    uncommented = "\n".join(line.split("#", 1)[0] for line in rules.splitlines())
    return [
        statement.strip() for statement in uncommented.split(";") if statement.strip()
    ]


def _compile_rules(rules, label):
    operations: list[dict[str, object]] = []
    unsupported = []
    replacements: dict[str, str] = {}

    def flush_replacements():
        if replacements:
            operations.append({"op": "map_chars", "map": dict(replacements)})
            replacements.clear()

    for statement in _rule_statements(rules or ""):
        if statement.startswith("::"):
            flush_replacements()
            directive = statement[2:].strip()
            if directive in _NORMAL_FORMS:
                operations.append({"op": "normalize", "form": directive})
            elif directive == "Any-Lower":
                operations.append({"op": "lower"})
            elif directive == "Null":
                continue
            elif directive.lower() == "[:m:] remove":
                operations.append({"op": "remove_mark_characters"})
            else:
                keep = _SHIPPED_FILTERS.get(directive.lower())
                if keep is not None:
                    operations.append(
                        {"op": "filter", "categories": ["L"], "keep": keep}
                    )
                else:
                    unsupported.append(f"{label}: {statement}")
            continue

        if statement.count(">") == 1:
            source, replacement = (part.strip() for part in statement.split(">", 1))
            source_scalar = _exact_scalar(source)
            replacement_scalar = _exact_scalar(replacement)
            if source_scalar is not None and replacement_scalar is not None:
                replacements[source_scalar] = replacement_scalar
                continue
        unsupported.append(f"{label}: {statement}")
    flush_replacements()
    return operations, unsupported


def compile_letter_preprocessing(snapshot):
    """Compile a version-1 raw preprocessing snapshot or raise on unsupported rules."""
    if not isinstance(snapshot, dict):
        raise PortableNormalizationError("letter_preprocessing must be an object")
    if snapshot.get("version") != 1:
        raise PortableNormalizationError(
            f"unsupported letter_preprocessing version: {snapshot.get('version')!r}"
        )

    source = snapshot.get("source")
    if not isinstance(source, dict):
        raise PortableNormalizationError(
            "letter_preprocessing.source must be an object"
        )
    for key in ("norm_rules", "g2p_rules"):
        if key not in source:
            raise PortableNormalizationError(
                f"letter_preprocessing.source.{key} is required"
            )

    operations: list[dict[str, object]] = []
    join_char = snapshot.get("join_char")
    if not isinstance(join_char, str):
        raise PortableNormalizationError(
            "letter_preprocessing.join_char must be a string"
        )
    if join_char:
        operations.append({"op": "replace", "source": join_char, "replacement": ""})

    cased = snapshot.get("cased")
    remove_accents = snapshot.get("remove_accents")
    filter_non_letters = snapshot.get("filter_non_letters")
    if not all(
        isinstance(value, bool) for value in (cased, remove_accents, filter_non_letters)
    ):
        raise PortableNormalizationError(
            "letter_preprocessing case/accent/filter flags must be booleans"
        )
    if not cased:
        operations.append({"op": "scalar_lower"})
    if remove_accents:
        operations.append({"op": "remove_accents"})
    if filter_non_letters:
        operations.append({"op": "normalize", "form": "NFD"})
        operations.append(
            {"op": "filter", "categories": ["L", "Mn", "Mc"], "keep": "-.'"}
        )
        operations.append({"op": "normalize", "form": "NFC"})

    unsupported = []
    for key in ("norm_rules", "g2p_rules"):
        rules = source.get(key)
        if rules is not None and not isinstance(rules, str):
            raise PortableNormalizationError(
                f"letter_preprocessing.source.{key} must be a string or null"
            )
        compiled, rejected = _compile_rules(rules, key)
        operations.extend(compiled)
        unsupported.extend(rejected)

    rewrites = snapshot.get("spelling_rewrites")
    if not isinstance(rewrites, dict) or not all(
        isinstance(key, str) and len(key) == 1 and isinstance(value, str)
        for key, value in rewrites.items()
    ):
        raise PortableNormalizationError(
            "letter_preprocessing.spelling_rewrites must map characters to strings"
        )
    if rewrites:
        operations.append({"op": "map_chars", "map": dict(rewrites)})

    joins = snapshot.get("letter_joins")
    if not isinstance(joins, list) or not all(isinstance(item, str) for item in joins):
        raise PortableNormalizationError(
            "letter_preprocessing.letter_joins must be a list of strings"
        )
    if unsupported:
        details = "; ".join(unsupported)
        raise PortableNormalizationError(
            f"letter preprocessing uses unsupported ICU rules: {details}"
        )
    return {
        "version": 1,
        "operations": operations,
        "join_char": join_char,
        "letter_joins": list(joins),
    }


def _apply_operations(text, operations):
    for operation in operations:
        op = operation.get("op")
        if op == "normalize":
            form = operation.get("form")
            if form not in _NORMAL_FORMS:
                raise PortableNormalizationError(
                    f"unsupported normalization form: {form!r}"
                )
            text = unicodedata.normalize(form, text)
        elif op == "replace":
            text = text.replace(operation["source"], operation["replacement"])
        elif op == "scalar_lower":
            text = "".join(character.lower() for character in text)
        elif op == "lower":
            text = text.lower()
        elif op == "remove_accents":
            text = unicodedata.normalize("NFD", text)
            text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
            text = unicodedata.normalize("NFC", text)
        elif op == "remove_mark_characters":
            text = "".join(
                ch for ch in text if not unicodedata.category(ch).startswith("M")
            )
        elif op == "filter":
            categories = tuple(operation["categories"])
            keep = operation["keep"]
            text = "".join(
                ch
                for ch in text
                if unicodedata.category(ch).startswith(categories) or ch in keep
            )
        elif op == "map_chars":
            mapping = operation["map"]
            text = "".join(mapping.get(ch, ch) for ch in text)
        else:
            raise PortableNormalizationError(f"unsupported portable operation: {op!r}")
    return text


def _join_letters(letters, joins, join_char):
    patterns = sorted((item.split() for item in joins), key=len, reverse=True)
    output = []
    index = 0
    while index < len(letters):
        match = next(
            (
                pattern
                for pattern in patterns
                if letters[index : index + len(pattern)] == pattern
            ),
            None,
        )
        if match:
            output.append(join_char.join(match))
            index += len(match)
        else:
            output.append(letters[index])
            index += 1
    return output


def apply_portable_preprocessing(text, program):
    """Apply a validated portable program and return cooked letter tokens."""
    if not isinstance(program, dict) or program.get("version") != 1:
        raise PortableNormalizationError("invalid portable preprocessing program")
    cooked = _apply_operations(text, program.get("operations", []))
    return _join_letters(
        list(cooked), program.get("letter_joins", []), program.get("join_char", "")
    )
