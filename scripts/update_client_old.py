from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


MODULE_PATTERN = re.compile(r"(?m)^\s*(\d+):\s*\(([^)]*)\)\s*=>\s*\{")
IDENT_START = set("_$") | {chr(code) for code in range(ord("A"), ord("Z") + 1)} | {
    chr(code) for code in range(ord("a"), ord("z") + 1)
}
IDENT_PART = IDENT_START | {str(num) for num in range(10)}
MULTI_PUNCT = (
    ">>>=",
    "&&=",
    "||=",
    "??=",
    "===",
    "!==",
    ">>>",
    "<<=",
    ">>=",
    "...",
    "=>",
    "?.",
    "??",
    "&&",
    "||",
    "==",
    "!=",
    ">=",
    "<=",
    "++",
    "--",
    "<<",
    ">>",
    "**",
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
    "&=",
    "|=",
    "^=",
)
OPEN_TO_CLOSE = {"(": ")", "[": "]", "{": "}"}
CLOSE_TO_OPEN = {value: key for key, value in OPEN_TO_CLOSE.items()}


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    start: int
    end: int


@dataclass(frozen=True)
class ModuleInfo:
    module_id: str
    body_start: int
    body_end: int
    code: str


@dataclass
class EventRange:
    name: str
    start: int
    end: int


@dataclass
class CallEntry:
    source: str
    module_id: str
    hash_value: str
    start: int
    end: int
    line: int
    column: int
    event_name: Optional[str]
    arg_count: int
    call_index_in_module: int = -1
    call_index_in_event: int = -1
    args_fingerprint: str = ""
    context_fingerprint: str = ""
    raw_signature: str = ""

    def to_output(self) -> Dict[str, object]:
        return {
            "source": self.source,
            "module_id": self.module_id,
            "hash": self.hash_value,
            "line": self.line,
            "column": self.column,
            "event_name": self.event_name,
            "arg_count": self.arg_count,
            "call_index_in_module": self.call_index_in_module,
            "call_index_in_event": self.call_index_in_event,
            "args_fingerprint": self.args_fingerprint,
            "context_fingerprint": self.context_fingerprint,
            "raw_signature": self.raw_signature,
        }


def sha1_short(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def skip_string(text: str, index: int, quote: str) -> int:
    index += 1
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == quote:
            return index + 1
        index += 1
    return index


def skip_line_comment(text: str, index: int) -> int:
    index += 2
    while index < len(text) and text[index] not in "\r\n":
        index += 1
    return index


def skip_block_comment(text: str, index: int) -> int:
    index += 2
    while index + 1 < len(text):
        if text[index] == "*" and text[index + 1] == "/":
            return index + 2
        index += 1
    return index


def find_matching_char(text: str, open_index: int) -> int:
    open_char = text[open_index]
    close_char = OPEN_TO_CLOSE[open_char]
    depth = 1
    index = open_index + 1

    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""

        if char in ("'", '"'):
            index = skip_string(text, index, char)
            continue
        if char == "`":
            index = skip_template(text, index)
            continue
        if char == "/" and next_char == "/":
            index = skip_line_comment(text, index)
            continue
        if char == "/" and next_char == "*":
            index = skip_block_comment(text, index)
            continue
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return index
        index += 1

    raise ValueError(f"Unmatched bracket {open_char!r} at index {open_index}")


def skip_template(text: str, index: int) -> int:
    index += 1
    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""
        if char == "\\":
            index += 2
            continue
        if char == "`":
            return index + 1
        if char == "$" and next_char == "{":
            close_index = find_matching_char(text, index + 1)
            index = close_index + 1
            continue
        index += 1
    return index


def compute_line_starts(text: str) -> List[int]:
    starts = [0]
    for match in re.finditer(r"\n", text):
        starts.append(match.end())
    return starts


def line_col_from_offset(line_starts: Sequence[int], offset: int) -> Tuple[int, int]:
    line_index = bisect.bisect_right(line_starts, offset) - 1
    line_start = line_starts[line_index]
    return line_index + 1, offset - line_start + 1


def public_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


def extract_modules(text: str) -> List[ModuleInfo]:
    marker = "exports.modules = {"
    marker_index = text.find(marker)
    if marker_index == -1:
        raise ValueError("Could not find exports.modules in bundle")

    brace_start = text.find("{", marker_index)
    if brace_start == -1:
        raise ValueError("Could not find opening brace for exports.modules")

    brace_end = find_matching_char(text, brace_start)
    modules: List[ModuleInfo] = []
    cursor = brace_start + 1

    while cursor < brace_end:
        match = MODULE_PATTERN.search(text, cursor, brace_end)
        if not match:
            break
        body_start = match.end() - 1
        body_end = find_matching_char(text, body_start)
        code = text[body_start + 1 : body_end]
        modules.append(
            ModuleInfo(
                module_id=match.group(1),
                body_start=body_start + 1,
                body_end=body_end,
                code=code,
            )
        )
        cursor = body_end + 1

    return modules


def tokenize(code: str) -> List[Token]:
    tokens: List[Token] = []
    index = 0
    length = len(code)

    while index < length:
        char = code[index]
        next_char = code[index + 1] if index + 1 < length else ""

        if char.isspace():
            index += 1
            continue

        if char == "/" and next_char == "/":
            index = skip_line_comment(code, index)
            continue

        if char == "/" and next_char == "*":
            index = skip_block_comment(code, index)
            continue

        if char in ("'", '"'):
            end = skip_string(code, index, char)
            tokens.append(Token("string", code[index + 1 : end - 1], index, end))
            index = end
            continue

        if char == "`":
            end = skip_template(code, index)
            tokens.append(Token("template", code[index:end], index, end))
            index = end
            continue

        if char == "#" and index + 1 < length and code[index + 1] in IDENT_START:
            start = index
            index += 1
            while index < length and code[index] in IDENT_PART:
                index += 1
            tokens.append(Token("identifier", code[start:index], start, index))
            continue

        if char in IDENT_START:
            start = index
            index += 1
            while index < length and code[index] in IDENT_PART:
                index += 1
            tokens.append(Token("identifier", code[start:index], start, index))
            continue

        if char.isdigit():
            start = index
            index += 1
            while index < length and (
                code[index].isalnum() or code[index] in "._xXn"
            ):
                index += 1
            tokens.append(Token("number", code[start:index], start, index))
            continue

        matched = None
        for punct in MULTI_PUNCT:
            if code.startswith(punct, index):
                matched = punct
                break
        if matched is not None:
            end = index + len(matched)
            tokens.append(Token("punct", matched, index, end))
            index = end
            continue

        tokens.append(Token("punct", char, index, index + 1))
        index += 1

    return tokens


def find_matching_token(tokens: Sequence[Token], open_index: int) -> int:
    open_value = tokens[open_index].value
    close_value = OPEN_TO_CLOSE[open_value]
    depth = 1
    index = open_index + 1

    while index < len(tokens):
        value = tokens[index].value
        if value == open_value:
            depth += 1
        elif value == close_value:
            depth -= 1
            if depth == 0:
                return index
        index += 1

    raise ValueError(f"Unmatched token {open_value!r} at token index {open_index}")


def find_top_level_comma(
    tokens: Sequence[Token], start_index: int, end_index: int
) -> Optional[int]:
    paren_depth = 0
    brace_depth = 0
    bracket_depth = 0

    for index in range(start_index, end_index):
        value = tokens[index].value
        if value == "(":
            paren_depth += 1
        elif value == ")":
            paren_depth -= 1
        elif value == "{":
            brace_depth += 1
        elif value == "}":
            brace_depth -= 1
        elif value == "[":
            bracket_depth += 1
        elif value == "]":
            bracket_depth -= 1
        elif value == "," and paren_depth == brace_depth == bracket_depth == 0:
            return index
    return None


def split_top_level_args(
    tokens: Sequence[Token], start_index: int, end_index: int
) -> List[Tuple[int, int]]:
    args: List[Tuple[int, int]] = []
    paren_depth = 0
    brace_depth = 0
    bracket_depth = 0
    arg_start = start_index

    for index in range(start_index, end_index):
        value = tokens[index].value
        if value == "(":
            paren_depth += 1
        elif value == ")":
            paren_depth -= 1
        elif value == "{":
            brace_depth += 1
        elif value == "}":
            brace_depth -= 1
        elif value == "[":
            bracket_depth += 1
        elif value == "]":
            bracket_depth -= 1
        elif (
            value == ","
            and paren_depth == 0
            and brace_depth == 0
            and bracket_depth == 0
        ):
            args.append((arg_start, index))
            arg_start = index + 1

    if arg_start < end_index:
        args.append((arg_start, end_index))

    return [entry for entry in args if entry[0] < entry[1]]


def normalize_token_slice(
    tokens: Sequence[Token], start_index: int, end_index: int
) -> str:
    pieces: List[str] = []
    for index in range(start_index, end_index):
        token = tokens[index]
        if token.kind == "string":
            if (
                index >= 2
                and tokens[index - 1].value == "("
                and tokens[index - 2].value == "callRemote"
            ):
                pieces.append('"__CALLREMOTE_ID__"')
            else:
                pieces.append(json.dumps(token.value, ensure_ascii=False))
        elif token.kind == "template":
            pieces.append("`__TEMPLATE__`")
        else:
            pieces.append(token.value)
    return " ".join(pieces)


def extract_event_ranges(tokens: Sequence[Token]) -> List[EventRange]:
    events: List[EventRange] = []

    for index in range(len(tokens) - 6):
        if (
            tokens[index].value == "mp"
            and tokens[index + 1].value == "."
            and tokens[index + 2].value == "events"
            and tokens[index + 3].value == "."
            and tokens[index + 4].value == "add"
            and tokens[index + 5].value == "("
            and tokens[index + 6].kind == "string"
        ):
            call_open = index + 5
            try:
                call_close = find_matching_token(tokens, call_open)
            except ValueError:
                continue

            event_name = tokens[index + 6].value
            comma_index = find_top_level_comma(tokens, call_open + 1, call_close)
            if comma_index is None:
                continue

            body_start = None
            for cursor in range(comma_index + 1, call_close):
                if tokens[cursor].value == "=>" and cursor + 1 < call_close:
                    if tokens[cursor + 1].value == "{":
                        body_start = cursor + 1
                        break
                if tokens[cursor].value == "function":
                    for inner in range(cursor + 1, call_close):
                        if tokens[inner].value == "{":
                            body_start = inner
                            break
                    if body_start is not None:
                        break

            if body_start is None:
                continue

            try:
                body_end = find_matching_token(tokens, body_start)
            except ValueError:
                continue

            events.append(
                EventRange(
                    name=event_name,
                    start=tokens[body_start].start,
                    end=tokens[body_end].end,
                )
            )

    events.sort(key=lambda entry: (entry.start, entry.end - entry.start))
    return events


def find_innermost_event(event_ranges: Sequence[EventRange], offset: int) -> Optional[str]:
    best_name = None
    best_size = None
    for event in event_ranges:
        if event.start <= offset < event.end:
            size = event.end - event.start
            if best_size is None or size < best_size:
                best_size = size
                best_name = event.name
    return best_name


def extract_calls(
    source_name: str,
    module: ModuleInfo,
    tokens: Sequence[Token],
    line_starts: Sequence[int],
) -> List[CallEntry]:
    event_ranges = extract_event_ranges(tokens)
    entries: List[CallEntry] = []

    for index in range(len(tokens) - 6):
        if (
            tokens[index].value == "mp"
            and tokens[index + 1].value == "."
            and tokens[index + 2].value == "events"
            and tokens[index + 3].value == "."
            and tokens[index + 4].value == "callRemote"
            and tokens[index + 5].value == "("
            and tokens[index + 6].kind == "string"
        ):
            call_open = index + 5
            try:
                call_close = find_matching_token(tokens, call_open)
            except ValueError:
                continue

            args = split_top_level_args(tokens, call_open + 1, call_close)
            if not args:
                continue

            hash_token = tokens[args[0][0]]
            if hash_token.kind != "string":
                continue

            start_abs = module.body_start + tokens[index].start
            line, column = line_col_from_offset(line_starts, start_abs)
            arg_count = max(0, len(args) - 1)
            args_norm = "|".join(
                normalize_token_slice(tokens, arg_start, arg_end)
                for arg_start, arg_end in args[1:]
            )
            window_start = max(0, index - 18)
            window_end = min(len(tokens), call_close + 19)
            context_norm = normalize_token_slice(tokens, window_start, window_end)
            event_name = find_innermost_event(event_ranges, tokens[index].start)

            entries.append(
                CallEntry(
                    source=source_name,
                    module_id=module.module_id,
                    hash_value=hash_token.value,
                    start=start_abs,
                    end=module.body_start + tokens[call_close].end,
                    line=line,
                    column=column,
                    event_name=event_name,
                    arg_count=arg_count,
                    args_fingerprint=sha1_short(args_norm),
                    context_fingerprint=sha1_short(context_norm),
                    raw_signature=args_norm[:400],
                )
            )

    entries.sort(key=lambda entry: entry.start)
    for call_index, entry in enumerate(entries):
        entry.call_index_in_module = call_index

    grouped_by_event: Dict[Optional[str], List[CallEntry]] = {}
    for entry in entries:
        grouped_by_event.setdefault(entry.event_name, []).append(entry)

    for group in grouped_by_event.values():
        group.sort(key=lambda entry: entry.start)
        for call_index, entry in enumerate(group):
            entry.call_index_in_event = call_index

    return entries


def collect_bundle_calls(path: Path, source_name: str) -> List[CallEntry]:
    text = path.read_text(encoding="utf-8")
    line_starts = compute_line_starts(text)
    calls: List[CallEntry] = []

    for module in extract_modules(text):
        tokens = tokenize(module.code)
        calls.extend(extract_calls(source_name, module, tokens, line_starts))

    return calls


def unique_match_pass(
    old_entries: Dict[int, CallEntry],
    new_entries: Dict[int, CallEntry],
    unmatched_old: set[int],
    unmatched_new: set[int],
    matches: Dict[int, Tuple[int, str]],
    method_name: str,
    key_builder,
) -> None:
    old_groups: Dict[Tuple[object, ...], List[int]] = {}
    new_groups: Dict[Tuple[object, ...], List[int]] = {}

    for entry_id in unmatched_old:
        old_groups.setdefault(key_builder(old_entries[entry_id]), []).append(entry_id)
    for entry_id in unmatched_new:
        new_groups.setdefault(key_builder(new_entries[entry_id]), []).append(entry_id)

    for key, old_group in old_groups.items():
        new_group = new_groups.get(key)
        if not new_group or len(old_group) != 1 or len(new_group) != 1:
            continue
        old_id = old_group[0]
        new_id = new_group[0]
        matches[old_id] = (new_id, method_name)
        unmatched_old.remove(old_id)
        unmatched_new.remove(new_id)


def order_fallback(
    old_entries: Dict[int, CallEntry],
    new_entries: Dict[int, CallEntry],
    unmatched_old: set[int],
    unmatched_new: set[int],
    matches: Dict[int, Tuple[int, str]],
) -> None:
    old_groups: Dict[Tuple[str, Optional[str]], List[int]] = {}
    new_groups: Dict[Tuple[str, Optional[str]], List[int]] = {}

    for entry_id in unmatched_old:
        entry = old_entries[entry_id]
        old_groups.setdefault((entry.module_id, entry.event_name), []).append(entry_id)
    for entry_id in unmatched_new:
        entry = new_entries[entry_id]
        new_groups.setdefault((entry.module_id, entry.event_name), []).append(entry_id)

    for key, old_group in old_groups.items():
        new_group = new_groups.get(key)
        if not new_group or len(old_group) != len(new_group):
            continue

        old_group.sort(key=lambda entry_id: old_entries[entry_id].start)
        new_group.sort(key=lambda entry_id: new_entries[entry_id].start)

        old_arg_counts = [old_entries[entry_id].arg_count for entry_id in old_group]
        new_arg_counts = [new_entries[entry_id].arg_count for entry_id in new_group]
        if old_arg_counts != new_arg_counts:
            continue

        for old_id, new_id in zip(old_group, new_group):
            matches[old_id] = (new_id, "order_fallback")
            unmatched_old.remove(old_id)
            unmatched_new.remove(new_id)


def match_calls(
    old_calls: List[CallEntry], new_calls: List[CallEntry]
) -> Tuple[List[Dict[str, object]], List[CallEntry], List[CallEntry]]:
    old_entries = {index: entry for index, entry in enumerate(old_calls)}
    new_entries = {index: entry for index, entry in enumerate(new_calls)}
    unmatched_old = set(old_entries)
    unmatched_new = set(new_entries)
    matches: Dict[int, Tuple[int, str]] = {}

    unique_match_pass(
        old_entries,
        new_entries,
        unmatched_old,
        unmatched_new,
        matches,
        "exact_context",
        lambda entry: (
            entry.module_id,
            entry.event_name,
            entry.arg_count,
            entry.args_fingerprint,
            entry.context_fingerprint,
        ),
    )
    unique_match_pass(
        old_entries,
        new_entries,
        unmatched_old,
        unmatched_new,
        matches,
        "event_position",
        lambda entry: (
            entry.module_id,
            entry.event_name,
            entry.call_index_in_event,
            entry.arg_count,
            entry.args_fingerprint,
        ),
    )
    unique_match_pass(
        old_entries,
        new_entries,
        unmatched_old,
        unmatched_new,
        matches,
        "module_position",
        lambda entry: (
            entry.module_id,
            entry.call_index_in_module,
            entry.arg_count,
            entry.args_fingerprint,
        ),
    )
    unique_match_pass(
        old_entries,
        new_entries,
        unmatched_old,
        unmatched_new,
        matches,
        "module_event_order",
        lambda entry: (
            entry.module_id,
            entry.event_name,
            entry.call_index_in_event,
            entry.arg_count,
        ),
    )
    order_fallback(old_entries, new_entries, unmatched_old, unmatched_new, matches)

    matched_rows: List[Dict[str, object]] = []
    for old_id, (new_id, match_method) in sorted(
        matches.items(), key=lambda item: old_entries[item[0]].start
    ):
        old_entry = old_entries[old_id]
        new_entry = new_entries[new_id]
        matched_rows.append(
            {
                "old_hash": old_entry.hash_value,
                "new_hash": new_entry.hash_value,
                "module_id": old_entry.module_id,
                "event_name": old_entry.event_name,
                "arg_count": old_entry.arg_count,
                "old_line": old_entry.line,
                "new_line": new_entry.line,
                "old_column": old_entry.column,
                "new_column": new_entry.column,
                "old_call_index_in_module": old_entry.call_index_in_module,
                "new_call_index_in_module": new_entry.call_index_in_module,
                "old_call_index_in_event": old_entry.call_index_in_event,
                "new_call_index_in_event": new_entry.call_index_in_event,
                "match_method": match_method,
                "signature": {
                    "args_fingerprint": old_entry.args_fingerprint,
                    "context_fingerprint": old_entry.context_fingerprint,
                    "raw_signature": old_entry.raw_signature,
                },
            }
        )

    old_unmatched = [old_entries[entry_id] for entry_id in sorted(unmatched_old)]
    new_unmatched = [new_entries[entry_id] for entry_id in sorted(unmatched_new)]
    return matched_rows, old_unmatched, new_unmatched


def generate_mapping(old_path: Path, new_path: Path, output_path: Path) -> Dict[str, object]:
    old_calls = collect_bundle_calls(old_path, "old")
    new_calls = collect_bundle_calls(new_path, "new")
    matched_rows, old_unmatched, new_unmatched = match_calls(old_calls, new_calls)

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "old_file": public_path(old_path),
        "new_file": public_path(new_path),
        "summary": {
            "old_calls_total": len(old_calls),
            "new_calls_total": len(new_calls),
            "matched_total": len(matched_rows),
            "old_unmatched_total": len(old_unmatched),
            "new_unmatched_total": len(new_unmatched),
        },
        "matches": matched_rows,
        "old_unmatched": [entry.to_output() for entry in old_unmatched],
        "new_unmatched": [entry.to_output() for entry in new_unmatched],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def build_hash_mapping(payload: Dict[str, object]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for row in payload["matches"]:
        old_hash = row["old_hash"]
        new_hash = row["new_hash"]
        if old_hash != new_hash:
            mapping[old_hash] = new_hash
    return mapping


def collect_remote_call_targets(tokens: Sequence[Token]) -> List[Tuple[str, int]]:
    targets: List[Tuple[str, int]] = []

    for index in range(len(tokens) - 4):
        is_call_remote = (
            index + 5 < len(tokens)
            and tokens[index].value == "mp"
            and tokens[index + 1].value == "."
            and tokens[index + 2].value == "events"
            and tokens[index + 3].value == "."
            and tokens[index + 4].value == "callRemote"
            and tokens[index + 5].value == "("
        )
        is_emit_server = (
            index + 3 < len(tokens)
            and tokens[index].value == "alt"
            and tokens[index + 1].value == "."
            and tokens[index + 2].value == "emitServer"
            and tokens[index + 3].value == "("
        )

        if not is_call_remote and not is_emit_server:
            continue

        open_index = index + 5 if is_call_remote else index + 3
        try:
            close_index = find_matching_token(tokens, open_index)
        except ValueError:
            continue

        args = split_top_level_args(tokens, open_index + 1, close_index)
        if not args:
            continue

        arg_start, arg_end = args[0]
        if arg_end - arg_start != 1:
            continue

        first_arg = tokens[arg_start]
        if first_arg.kind in ("string", "number"):
            targets.append(("literal", arg_start))
        elif first_arg.kind == "identifier":
            targets.append(("identifier", arg_start))

    return targets


def update_script_file(script_path: Path, hash_mapping: Dict[str, str]) -> Dict[str, object]:
    script_text = script_path.read_text(encoding="utf-8")
    tokens = tokenize(script_text)
    line_starts = compute_line_starts(script_text)
    replacements: Dict[int, Dict[str, object]] = {}
    referenced_identifiers: set[str] = set()
    old_script_path = script_path.with_name(f"{script_path.stem}-old{script_path.suffix}")
    new_script_path = script_path.with_name(f"{script_path.stem}-new{script_path.suffix}")

    def queue_replacement(token_index: int, replacement_reason: str) -> None:
        token = tokens[token_index]
        new_hash = hash_mapping.get(token.value)
        if new_hash is None:
            return

        if token.kind == "string":
            quote = script_text[token.start]
            new_text = f"{quote}{new_hash}{quote}"
        elif token.kind == "number":
            new_text = new_hash
        else:
            return

        line, column = line_col_from_offset(line_starts, token.start)
        replacements[token.start] = {
            "start": token.start,
            "end": token.end,
            "old_value": token.value,
            "new_value": new_hash,
            "line": line,
            "column": column,
            "reason": replacement_reason,
            "new_text": new_text,
        }

    for target_type, token_index in collect_remote_call_targets(tokens):
        token = tokens[token_index]
        if target_type == "literal":
            queue_replacement(token_index, "direct_call_argument")
        elif target_type == "identifier":
            referenced_identifiers.add(token.value)

    for index in range(len(tokens) - 3):
        if (
            tokens[index].value in ("const", "let", "var")
            and tokens[index + 1].kind == "identifier"
            and tokens[index + 1].value in referenced_identifiers
            and tokens[index + 2].value == "="
            and tokens[index + 3].kind in ("string", "number")
        ):
            queue_replacement(index + 3, f"variable_initializer:{tokens[index + 1].value}")

        if (
            tokens[index].kind == "identifier"
            and tokens[index].value in referenced_identifiers
            and tokens[index + 1].value == "="
            and tokens[index + 2].kind in ("string", "number")
            and (index == 0 or tokens[index - 1].value not in (".", ":"))
        ):
            queue_replacement(index + 2, f"variable_assignment:{tokens[index].value}")

    updated_text = script_text
    if replacements:
        ordered_replacements = sorted(
            replacements.values(), key=lambda entry: entry["start"], reverse=True
        )
        for replacement in ordered_replacements:
            updated_text = (
                updated_text[: replacement["start"]]
                + replacement["new_text"]
                + updated_text[replacement["end"] :]
            )

    old_script_path.write_text(script_text, encoding="utf-8")
    new_script_path.write_text(updated_text, encoding="utf-8")
    return {
        "script_path": public_path(script_path),
        "old_script_path": public_path(old_script_path),
        "new_script_path": public_path(new_script_path),
        "updated": bool(replacements),
        "replacements": list(sorted(replacements.values(), key=lambda entry: entry["start"])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map old callRemote hashes to new ones by structural parsing."
    )
    parser.add_argument("--old", default="client-old.js", help="Path to the old bundle")
    parser.add_argument("--new", default="client-new.js", help="Path to the new bundle")
    parser.add_argument(
        "--out",
        default="callremote-map.json",
        help="Where to write the mapping JSON",
    )
    parser.add_argument(
        "--update-script",
        help="Optional JS file to update by replacing old ids in callRemote/emitServer usages",
    )
    return parser.parse_args()


def print_summary(payload: Dict[str, object], output_path: Path) -> None:
    summary = payload["summary"]
    print(
        "[callremote-mapper]",
        f"written={output_path}",
        f"old_total={summary['old_calls_total']}",
        f"new_total={summary['new_calls_total']}",
        f"matched={summary['matched_total']}",
        f"old_unmatched={summary['old_unmatched_total']}",
        f"new_unmatched={summary['new_unmatched_total']}",
    )


def main() -> int:
    args = parse_args()
    old_path = Path(args.old).resolve()
    new_path = Path(args.new).resolve()
    output_path = Path(args.out).resolve()

    if not old_path.exists():
        print(f"Old bundle not found: {old_path}", file=sys.stderr)
        return 1
    if not new_path.exists():
        print(f"New bundle not found: {new_path}", file=sys.stderr)
        return 1

    try:
        payload = generate_mapping(old_path, new_path, output_path)
        print_summary(payload, output_path)

        if args.update_script:
            script_path = Path(args.update_script).resolve()
            if not script_path.exists():
                print(f"Script to update not found: {script_path}", file=sys.stderr)
                return 1

            update_result = update_script_file(script_path, build_hash_mapping(payload))
            print(
                "[callremote-mapper]",
                f"script={update_result['script_path']}",
                f"old_copy={update_result['old_script_path']}",
                f"new_copy={update_result['new_script_path']}",
                f"updated={update_result['updated']}",
                f"replacements={len(update_result['replacements'])}",
            )

        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:  # pragma: no cover
        print(f"Failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
