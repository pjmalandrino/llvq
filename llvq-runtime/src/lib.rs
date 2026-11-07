//! `llvq`: pull, run and serve the sealed files with Ollama's commands and
//! API, and OpenAI's chat format. The plan is `docs/plan-runtime.md`.
//!
//! * [`library`]: model names, the compiled-in library, the build's backend.
//! * [`store`]: `LLVQ_HOME`, blobs and manifests, the checked pull.
//! * [`chat`]: messages to ids, and the text streamed back.
//! * [`engine`]: one thread per loaded model, the queue, the keep-alive.
//! * [`api`]: the HTTP routes and the OpenAPI document.
//! * [`client`]: what the command line sends to a running server.

#![forbid(unsafe_code)]

pub mod chat;
pub mod engine;
pub mod library;
pub mod store;
pub mod api;
pub mod openai;
pub mod client;
