# /// script
# requires-python = ">=3.10"
# dependencies = ["ollama==0.6.1", "openai==2.8.1"]
# ///
"""The official ollama and openai Python clients, unmodified, against llvq.

Gates R2 and R3 of docs/plan-runtime.md: each client streams a chat, and the
streamed text equals the one-object answer. Run against any llvq server:

    uv run ops/llvq_runtime_clients.py 127.0.0.1:11435 qwen3:4b

or against the tiny CPU model of the tests (`cargo run -p llvq-runtime
--example tiny_server` prints its address; model `tiny:test`).
"""

import sys
from importlib.metadata import version

import ollama
import openai

host, model = sys.argv[1], sys.argv[2]
words = "hello how are you" if model == "tiny:test" else "Name three primary colours."
messages = [{"role": "user", "content": words}]
opts = {"temperature": 0, "num_predict": 24}

c = ollama.Client(host=f"http://{host}")
assert any(m.model == model for m in c.list().models), f"{model} not listed"
streamed, last = "", None
for part in c.chat(model=model, messages=messages, stream=True, options=opts):
    streamed += part.message.content or ""
    last = part
assert last is not None and last.done, "the ollama stream did not end with done"
whole = c.chat(model=model, messages=messages, options=opts)
assert whole.message.content == streamed, (whole.message.content, streamed)
assert last.eval_count == whole.eval_count
gen = "".join(p.response for p in c.generate(model=model, prompt=words, stream=True, options=opts))
assert gen == streamed, "the templated generate is the chat's user turn"
print(f"ollama {version('ollama')}: chat and generate stream "
      f"{last.eval_count} tokens, done_reason={last.done_reason}")

o = openai.OpenAI(base_url=f"http://{host}/v1", api_key="unused")
assert any(m.id == model for m in o.models.list().data)
chunks = list(o.chat.completions.create(model=model, messages=messages, stream=True, temperature=0, max_tokens=24,
                                        stream_options={"include_usage": True}))
text = "".join(ch.choices[0].delta.content or "" for ch in chunks if ch.choices)
finish = [ch.choices[0].finish_reason for ch in chunks if ch.choices and ch.choices[0].finish_reason]
one = o.chat.completions.create(model=model, messages=messages, temperature=0, max_tokens=24)
assert one.choices[0].message.content == text == streamed, (one.choices[0].message.content, text, streamed)
assert finish == [one.choices[0].finish_reason]
assert chunks[-1].usage is not None and chunks[-1].usage.completion_tokens == one.usage.completion_tokens
print(f"openai {version('openai')}: chat.completions streams {one.usage.completion_tokens} tokens, "
      f"finish_reason={finish[0]}, same text as the ollama client")
