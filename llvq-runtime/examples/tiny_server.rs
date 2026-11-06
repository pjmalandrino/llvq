//! The server with the tiny deterministic Qwen3 of the tests, for the client
//! checks of `ops/llvq_runtime_clients.py`: the official `ollama` and
//! `openai` Python packages, unmodified, against llvq's routes. CPU only, a
//! few megabytes, no sealed file.
//!
//! ```text
//! cargo run -p llvq-runtime --example tiny_server     # prints the address
//! ```

#[path = "../tests/common/mod.rs"]
mod common;

fn main() {
    let host = common::serve(common::store("example"), "http://127.0.0.1:9");
    println!("{host}");
    loop {
        std::thread::park();
    }
}
