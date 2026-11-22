# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "jinja2",
#   "tokenizers",
#   "transformers @ git+https://github.com/huggingface/transformers@046bdb8cc2",
# ]
# ///
"""The reference prompts of Qwen3's official chat template, with tools, for `llvq-runtime`.

    uv run ops/qwen3_tools_reference.py <Qwen3 snapshot dir> llvq-runtime/tests/fixtures/qwen3-tools-reference.json

The snapshot is the tokenizer of `Qwen/Qwen3-4B` as the Hub cache holds it (it is read, never written). Each case
is rendered by `apply_chat_template` twice: once with OpenAI's form of the tool calls (`arguments` a JSON string,
which the template writes verbatim) and once with Ollama's (`arguments` an object, which it writes through
`tojson`). Both texts must agree, and both are written with the ids of the whole prompt. The Rust renderer is
checked against the text byte for byte (`tools::tests`), and against the ids when the tokenizer is on the machine.
"""

import hashlib
import json
import pathlib
import sys

import transformers
from transformers import AutoTokenizer

READ_FILE = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a UTF-8 text file and return its contents (\"café\" & <tags> kept).",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the workspace"},
                "max_bytes": {"type": "integer", "minimum": 1, "maximum": 65536},
                "ratio": {"type": "number", "minimum": 0.5},
            },
            "required": ["path"],
        },
    },
}
LIST_DIR = {
    "type": "function",
    "function": {
        "name": "list_dir",
        "description": "List the entries of a directory.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    },
}


def call(i, name, args):
    return {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": args}}


def cases():
    """(name, tools, messages with object arguments)."""
    yield "single_turn_no_system", [READ_FILE, LIST_DIR], [{"role": "user", "content": "What is in notes.txt?"}]
    yield "two_calls_two_results", [READ_FILE, LIST_DIR], [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "What is in notes.txt?"},
        {"role": "assistant", "content": "", "tool_calls": [call(0, "list_dir", {"path": "."})]},
        {"role": "tool", "tool_call_id": "call_0", "content": "notes.txt\nsrc/"},
        {"role": "assistant", "content": "Two things to read.", "tool_calls": [
            call(1, "read_file", {"path": "notes.txt", "max_bytes": 4096}),
            call(2, "list_dir", {"path": "src"}),
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": "buy milk"},
        {"role": "tool", "tool_call_id": "call_2", "content": "main.rs"},
    ]
    yield "second_query_after_tools", [LIST_DIR], [
        {"role": "user", "content": "List the root."},
        {"role": "assistant", "content": "", "tool_calls": [call(0, "list_dir", {"path": "/"})]},
        {"role": "tool", "tool_call_id": "call_0", "content": "etc\nusr"},
        {"role": "assistant", "content": "The root holds etc and usr."},
        {"role": "user", "content": "And /usr?"},
    ]
    yield "two_calls_without_text", [READ_FILE, LIST_DIR], [
        {"role": "user", "content": "Read both."},
        {"role": "assistant", "content": "", "tool_calls": [
            call(0, "read_file", {"path": "a.txt"}),
            call(1, "read_file", {"path": "b.txt", "ratio": 0.75}),
        ]},
        {"role": "tool", "tool_call_id": "call_0", "content": "A"},
        {"role": "tool", "tool_call_id": "call_1", "content": "B"},
    ]
    yield "history_without_tools_list", [], [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "", "tool_calls": [call(0, "list_dir", {"path": "."})]},
        {"role": "tool", "tool_call_id": "call_0", "content": "a"},
    ]


def as_openai(messages):
    out = []
    for m in messages:
        m = json.loads(json.dumps(m))
        for c in m.get("tool_calls", []):
            c["function"]["arguments"] = json.dumps(c["function"]["arguments"], ensure_ascii=False)
        out.append(m)
    return out


def main():
    snap, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    tok = AutoTokenizer.from_pretrained(snap)
    rows = []
    for name, tools, msgs in cases():
        texts = []
        for form in (msgs, as_openai(msgs)):
            texts.append(tok.apply_chat_template(form, tools=tools or None, tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False))
        assert texts[0] == texts[1], f"{name}: the object and the string forms of the arguments differ"
        ids = tok(texts[0], add_special_tokens=False)["input_ids"]
        rows.append({"name": name, "tools": tools, "messages": as_openai(msgs), "text": texts[0], "ids": ids})
    cfg = snap / "tokenizer_config.json"
    doc = {
        "source": "Qwen/Qwen3-4B tokenizer_config.json chat_template, rendered by transformers apply_chat_template",
        "snapshot": snap.name,
        "tokenizer_config_sha256": hashlib.sha256(cfg.read_bytes()).hexdigest(),
        "transformers": transformers.__version__ + " @ git 046bdb8cc2",
        "enable_thinking": False,
        "cases": rows,
    }
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    print(f"{out}: {len(rows)} cases")


if __name__ == "__main__":
    main()
