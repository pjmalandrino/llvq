# Fixtures

`ollama-openapi.yaml` is `docs/openapi.yaml` of `ollama/ollama` at commit
`6362e0a30887bdd4f69e07a94e8a513383576204` (branch `main`, 2026-11-06), unmodified.
`ollama-LICENSE` is that repository's MIT license, which covers it. `tests/openapi.rs` reads it
to check that every Ollama route `llvq` serves exists there with the same method and the same
required request fields.
