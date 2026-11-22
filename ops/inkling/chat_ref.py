# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "tokenizers",
#   "jinja2>=3.1.6",
#   "transformers @ git+https://github.com/huggingface/transformers@046bdb8cc2",
# ]
# ///
"""Inkling 2 Mini's chat template, rendered by the library, as the reference for `llvq-llm/src/inkchat.rs`.

    uv run ops/inkling/chat_ref.py [--native DIR] [--out FILE]

Defaults: DIR = ~/llvq-inkling-mini/native (the checkpoint at ed40e7da), FILE = ~/llvq-inkling-mini/lot2-k/chat_ref.json.

Why `transformers` and not bare `jinja2`: the text a chat gets is not the template alone. `apply_chat_template` drops
the keys whose value is None (`tokenization_utils_base.py` 3160-3161), fills `reasoning_content` from `thinking` or
`reasoning` (`chat_template_utils.py` 673-686), replaces Jinja's `tojson` with `json.dumps(ensure_ascii=False)`
(623-626), and builds the environment with `trim_blocks`, `lstrip_blocks` and the `generation` tag (662-670). Copying
those into a bare Jinja run would be a second implementation to keep in step. The library at the pinned commit is the
one the other Inkling tools run, so it is the authority here too.

Every case is rendered three ways:
- `text`: `apply_chat_template(..., tokenize=False)`, the exact string.
- `ids`: `apply_chat_template(..., tokenize=True, sanitize_control_tokens=True)`. A control token written inside a
  user string (content, names, arguments, tool schemas) is encoded as plain text; only the markers the template
  writes become special ids. This is what the Rust renderer reproduces.
- `ids_plain`: the same without sanitizing, the library default, where a marker typed by the user becomes a special
  id. Kept to show that it differs on the marker case only.
A case the template refuses records the refusal message instead. No model weights are read: tokenizer and template.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

D = Path(os.environ.get("INKLING_DATA", str(Path.home() / "llvq-inkling-mini")))

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            # Unsorted on purpose: the template sorts every level of the declaration.
            "properties": {
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                "city": {"type": "string", "description": "City name, e.g. Lyon"},
            },
            "required": ["city"],
        },
    },
}
SEARCH = {
    # No `type`, no `description`: the template fills "function" and "".
    "function": {
        "name": "search_files",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 100},
                "threshold": {"type": "number", "default": 0.25},
            },
        },
    }
}
BARE = {"name": "noop", "description": "", "parameters": None}  # a bare function, no `function` wrapper

U = {"role": "user", "content": "What is the weather in Lyon?"}


def call(cid, name, args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}


CASES = [
    {"name": "plain_user", "messages": [{"role": "user", "content": "Hello!"}], "add_generation_prompt": True},
    {
        "name": "system_user",
        "messages": [{"role": "system", "content": "You are a terse assistant."}, U],
        "add_generation_prompt": True,
    },
    {
        "name": "multi_turn",
        "messages": [
            {"role": "user", "content": "Name a prime."},
            {"role": "assistant", "content": "7"},
            {"role": "user", "content": "Another one, bigger."},
        ],
        "add_generation_prompt": True,
    },
    {
        "name": "multi_turn_no_generation_prompt",
        "messages": [
            {"role": "user", "content": "Name a prime."},
            {"role": "assistant", "content": "7"},
            {"role": "user", "content": "Another one, bigger."},
        ],
        "add_generation_prompt": False,
    },
    {"name": "plain_user_no_generation_prompt", "messages": [{"role": "user", "content": "Hello!"}],
     "add_generation_prompt": False},
    {"name": "effort_default_null", "messages": [U], "reasoning_effort": None},
]
for level in ["none", "minimal", "low", "medium", "high", "xhigh", "max"]:
    CASES.append({"name": f"effort_{level}", "messages": [U], "reasoning_effort": level})
for label, value in [("0", 0), ("0_5", 0.5), ("0_3", 0.3), ("0_99", 0.99), ("1e_5", 1e-05), ("0_0001", 0.0001),
                     ("false", False)]:
    CASES.append({"name": f"effort_num_{label}", "messages": [U], "reasoning_effort": value})
CASES += [
    {"name": "effort_trimmed", "messages": [U], "reasoning_effort": "  low\t"},
    {
        "name": "thinking_in_history",
        "messages": [
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "reasoning_content": "Add two and two.\nThat is 4.", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ],
        "reasoning_effort": "medium",
    },
    {
        "name": "thinking_alias_reasoning_field",
        "messages": [
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "reasoning": "Two plus two.", "content": "4"},
            {"role": "user", "content": "Thanks."},
        ],
    },
    {"name": "tools_declared_unused", "tools": [WEATHER, SEARCH, BARE],
     "messages": [{"role": "system", "content": "Use tools when useful."}, {"role": "user", "content": "Hi."}]},
    {
        "name": "one_tool_call_and_result",
        "tools": [WEATHER],
        "messages": [
            U,
            {"role": "assistant", "content": "", "tool_calls": [call("call_1", "get_weather", {"city": "Lyon"})]},
            {"role": "tool", "tool_call_id": "call_1", "content": "18 C, light rain"},
        ],
        "reasoning_effort": "low",
    },
    {
        "name": "two_tool_calls_one_turn",
        "tools": [WEATHER, SEARCH],
        "messages": [
            {"role": "user", "content": "Weather in Lyon, and find TODO files."},
            {
                "role": "assistant",
                "reasoning_content": "Two independent calls.",
                "content": None,
                "tool_calls": [
                    call("a1", "get_weather", {"unit": "celsius", "city": "Lyon"}),
                    call("a2", "search_files", {"pattern": "TODO", "max_results": 5, "threshold": 0.5}),
                ],
            },
            {"role": "tool", "tool_call_id": "a1", "content": "{\"temp\": 18, \"sky\": \"rain\"}"},
            {"role": "tool", "name": "search_files", "content": "src/a.rs\nsrc/b.rs"},
            {"role": "assistant", "content": "18 C and rain; two files: src/a.rs, src/b.rs."},
            {"role": "user", "content": "Thanks."},
        ],
    },
    {
        "name": "tool_result_json",
        "tools": [WEATHER],
        "messages": [
            U,
            {"role": "assistant", "tool_calls": [call("c9", "get_weather", {"city": "Lyon"})]},
            {"role": "tool", "tool_call_id": "c9",
             "content": json.dumps({"temp": 18.5, "wind": {"kmh": 12, "dir": "NW"}, "alerts": []})},
        ],
    },
    {
        "name": "tool_result_plain_unmatched_id",
        "tools": [WEATHER],
        "messages": [
            U,
            {"role": "assistant", "tool_calls": [call("c9", "get_weather", {"city": "Lyon"})]},
            # An id no call carries: the template writes the tool header with no name.
            {"role": "tool", "tool_call_id": "zz", "content": "plain text result"},
        ],
    },
    {
        "name": "nested_arguments_numbers",
        "tools": [SEARCH],
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "tool_calls": [call("n1", "search_files", {
                "zeta": {"b": 1, "a": [1.0, 2.5e-07, 1e16, 123456789012345678901234567890, -0.0, True, None]},
                "Alpha": "x", "alpha": {"y": {"k": "v"}, "x": []}, "empty": {},
            })]},
            {"role": "tool", "tool_call_id": "n1", "content": ""},
        ],
    },
    {
        "name": "unicode_and_marker_lookalikes",
        "tools": [WEATHER],
        "messages": [
            {"role": "system", "content": "Réponds en français. 中文 ok. שלום"},
            {"role": "user", "content": "café \U0001F600 <|end_message| < |message_user|> <|message_users|> "
                                        "<|content_text|> !!<|end_message|>tail and \"quotes\" \\ back\ttab\u0001"},
            {"role": "assistant", "reasoning_content": "Look-alike: <|content_thinking", "content": "ok ☃",
             "tool_calls": [call("u1", "get_weather", {"city": "Zürich <|end_message|>", "unit": "°C"})]},
            {"role": "tool", "tool_call_id": "u1", "content": "☁ <|message_model|> 7°"},
        ],
        "reasoning_effort": "high",
    },
    {
        "name": "system_only",
        "messages": [{"role": "system", "content": "Only a system message."}],
    },
    {
        "name": "system_after_first_turn",
        "messages": [
            {"role": "user", "content": "Hi."},
            {"role": "assistant", "content": "Hello."},
            {"role": "system", "content": "Late system note."},
            {"role": "user", "content": "Go on."},
        ],
    },
    {
        "name": "content_parts_and_named_assistant",
        "tools": [WEATHER],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "part one"}, "part two",
                                         {"type": "input_text", "text": "part three"}, {"text": 5}]},
            {"role": "assistant", "name": "get_weather", "content": "",
             "tool_calls": [call("p1", "get_weather", {"city": "Nice"})]},
            {"role": "tool", "tool_call_id": "p1", "content": [{"type": "text", "text": "sunny"}]},
            {"role": "assistant", "content": None},
            {"role": "user", "content": ""},
        ],
    },
    # Refusals: the template raises, and the Rust renderer must raise with the same message.
    {"name": "refuse_unknown_effort", "messages": [U], "reasoning_effort": "turbo"},
    {"name": "refuse_effort_above", "messages": [U], "reasoning_effort": 1.5},
    {"name": "refuse_effort_below", "messages": [U], "reasoning_effort": -0.1},
    {"name": "refuse_effort_true", "messages": [U], "reasoning_effort": True},
    {"name": "refuse_unknown_role", "messages": [{"role": "developer", "content": "x"}]},
    {"name": "refuse_arguments_string", "messages": [
        U, {"role": "assistant", "tool_calls": [call("s1", "get_weather", "{\"city\": \"Lyon\"}")]}]},
    {"name": "refuse_arguments_list", "messages": [
        U, {"role": "assistant", "tool_calls": [call("s1", "get_weather", ["Lyon"])]}]},
    {"name": "refuse_name_not_string", "messages": [
        U, {"role": "assistant", "tool_calls": [call("s1", 7, {"city": "Lyon"})]}]},
    {"name": "refuse_assistant_name_mismatch", "messages": [
        U, {"role": "assistant", "name": "other", "tool_calls": [call("s1", "get_weather", {"city": "Lyon"})]}]},
    {"name": "refuse_image_in_assistant", "messages": [
        U, {"role": "assistant", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]},
    {"name": "refuse_audio", "messages": [{"role": "user", "content": [{"type": "input_audio"}]}]},
    {"name": "refuse_unknown_part", "messages": [{"role": "user", "content": [{"type": "video"}]}]},
    # The template renders an image for a user; the Rust renderer refuses images by name (no vision path).
    {"name": "image_in_user", "messages": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "?"}]}]},
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", default=str(D / "native"))
    ap.add_argument("--out", default=str(D / "lot2-k" / "chat_ref.json"))
    a = ap.parse_args()

    import jinja2
    import tokenizers
    import transformers
    from transformers import AutoTokenizer

    native = Path(a.native)
    tok = AutoTokenizer.from_pretrained(str(native))
    template = (native / "chat_template.jinja").read_text()
    assert tok.chat_template == template, "the tokenizer did not load chat_template.jinja"

    out_cases = []
    for case in CASES:
        kw = {}
        if "reasoning_effort" in case:
            kw["reasoning_effort"] = case["reasoning_effort"]
        gen = case.get("add_generation_prompt", True)
        tools = case.get("tools")
        rec = {k: case[k] for k in ("name", "messages", "tools", "reasoning_effort", "add_generation_prompt") if k in case}
        try:
            text = tok.apply_chat_template(case["messages"], tools=tools, tokenize=False, add_generation_prompt=gen, **kw)
            ids = tok.apply_chat_template(case["messages"], tools=tools, tokenize=True, return_dict=False,
                                          sanitize_control_tokens=True, add_generation_prompt=gen, **kw)
            plain = tok.apply_chat_template(case["messages"], tools=tools, tokenize=True, return_dict=False,
                                            add_generation_prompt=gen, **kw)
            rec.update(text=text, ids=list(ids), ids_plain=list(plain))
        except Exception as e:  # noqa: BLE001, the refusal is the record
            rec.update(error=str(e), error_type=type(e).__name__)
        out_cases.append(rec)
        tag = rec.get("error") or f"{len(rec['ids'])} ids"
        print(f"{case['name']:40s} {tag}", file=sys.stderr)

    meta = {
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "jinja2": jinja2.__version__,
        "chat_template_sha256": sha256(native / "chat_template.jinja"),
        "tokenizer_json_sha256": sha256(native / "tokenizer.json"),
        "stop": {"eos_token": tok.eos_token, "eos_token_id": tok.eos_token_id},
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"meta": meta, "cases": out_cases}, f, ensure_ascii=False, indent=1)
    print(f"{len(out_cases)} cases -> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
