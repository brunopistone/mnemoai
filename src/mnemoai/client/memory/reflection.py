"""Local, evidence-linked observations and shared credential redaction."""

import hashlib
import json
import re

from mnemoai.client.memory.playbook_records import PlaybookEntry
from mnemoai.utils.shell_syntax import simple_command_tokens

_SENSITIVE = re.compile(r"api.?key|password|secret|token|authorization", re.I)
_TOKEN = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|pypi-[A-Za-z0-9_-]{16,}|AKIA[A-Z0-9]{16})\b")


def redact(value):
    if isinstance(value, dict):
        return {str(k): "[redacted]" if _SENSITIVE.search(str(k)) else redact(v)
                for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        value = _TOKEN.sub("[redacted]", value)
        return re.sub(
            r"""(?i)(["']?(?:password|secret|api[_-]?key|authorization|access_token)"""
            r"""["']?\s*[:=]\s*)(?:"[^"]*"|'[^']*'|(?:Bearer|Basic)\s+[^\s,;]+|[^\s,;]+)""",
            r'\1"[redacted]"', value,
        )
    return value


def evidence_item(name, args, result, call_id, outcome, source):
    text = result if isinstance(result, str) else json.dumps(result, default=str)
    try:
        preview = json.dumps(redact(json.loads(text)), ensure_ascii=True)
    except (ValueError, TypeError):
        preview = redact(text)
    local_args = {key: value for key, value in args.items()
                  if key in {"path", "file_path", "command", "mode", "working_directory"}}
    match = args.get("old_string", args.get("old_str"))
    if isinstance(match, str):
        local_args["match_chars"] = len(match)
    if "replace_all" in args:
        local_args["replace_all"] = args["replace_all"]
    return {
        "tool": name, "args": json.dumps(redact(args), ensure_ascii=True)[:600],
        "result": preview[:1200], "truncated": len(preview) > 1200,
        "outcome": outcome,
        "operation": operation_key(name, args, result),
        "local_args": redact(local_args),
        "ref": {
            **source, "tool_call_id": call_id, "tool": name, "outcome": outcome,
            "result_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "args_sha256": hashlib.sha256(
                json.dumps(args, sort_keys=True, default=str).encode()
            ).hexdigest(),
        },
    }


def operation_key(name, args, result):
    """Only pair retries whose tool and explicit target/command agree."""
    target = args.get("file_path") or args.get("path")
    if name == "execute_bash":
        if not args.get("command"):
            return None
        try:
            data = json.loads(result) if isinstance(result, str) else result
        except (ValueError, TypeError):
            data = {}
        cwd = data.get("cwd", "") if isinstance(data, dict) else ""
        target = (args.get("command"), args.get("working_directory") or cwd)
    if not target:
        return None
    return hashlib.sha256(json.dumps([name, target], sort_keys=True).encode()).hexdigest()


def _arguments(item):
    if isinstance(item.get("local_args"), dict):
        return item["local_args"]
    try:
        value = json.loads(item["args"])
        return value if isinstance(value, dict) else {}
    except (ValueError, TypeError):
        return {}


def _is_check(item):
    if item["tool"] != "execute_bash":
        return False
    tokens = simple_command_tokens(_arguments(item).get("command", ""))
    if not tokens:
        return False
    program = tokens[0].rsplit("/", 1)[-1]
    if program in {"pytest", "mypy", "pyright"}:
        return True
    if program == "ruff":
        return "check" in tokens or "--check" in tokens
    if program.startswith("python"):
        return (
            len(tokens) >= 3 and tokens[1] == "-m"
            and tokens[2] in {"pytest", "unittest", "compileall"}
        ) or any(t.rsplit("/", 1)[-1].startswith("test_") and t.endswith(".py") for t in tokens[1:])
    return False


def _describe(item):
    # Never persist file bodies, replacement text or arbitrary stdout as advice.
    args = {key: value for key, value in _arguments(item).items()
            if key in {"path", "file_path", "command", "mode", "working_directory",
                       "match_chars", "replace_all"}}
    return f"{item['tool']} {json.dumps(args, ensure_ascii=True, sort_keys=True)[:240]}"


def extract_local(evidence, task, *, scope):
    """Record bounded facts, not invented causes, instructions or success claims."""
    failures, candidates = {}, []
    for item in evidence:
        key = item.get("operation")
        if item["outcome"] == "failure":
            candidates.append([item])
            if key:
                failures[key] = candidates[-1]
        elif key in failures:
            pair = failures.pop(key)
            pair.append(item)
        elif _is_check(item):
            candidates.append([item])

    # Failures/recoveries get priority over repetitive successful checks.
    candidates.sort(key=lambda items: (items[0]["outcome"] != "failure", -len(items)))
    entries, seen = [], set()
    context = " ".join(str(redact(task)).split())[:200]
    for items in candidates:
        first = items[0]
        if first["outcome"] == "failure":
            kind = first.get("failure_type") or "tool error"
            strategy = f"Observed {_describe(first)} failed ({kind})."
            if len(items) > 1:
                strategy += f" Later {_describe(items[-1])} on the same operation succeeded."
        else:
            strategy = f"Observed check {_describe(first)} returned success."
        qualification = " Historical tool outcome only; not proof of cause or general correctness."
        strategy = str(redact(strategy))[:800 - len(qualification)] + qualification
        identity = (context, strategy)
        if identity in seen:
            continue
        seen.add(identity)
        entries.append(PlaybookEntry(
            context=context or f"Using {first['tool']}", strategy=strategy,
            source="Locally recorded tool outcomes", confidence=0.5,
            outcome=first["outcome"], tools=sorted({item["tool"] for item in items}),
            source_refs=[item["ref"] for item in items], scope=scope, provenance="observed",
        ))
        if len(entries) == 3:
            break
    return entries
