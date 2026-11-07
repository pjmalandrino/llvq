//! The OpenAPI 3.1 document: generated from the handlers, committed as
//! `openapi.json`, and checked against Ollama's own document at a pinned
//! commit (`tests/fixtures/ollama-openapi.yaml`, MIT, notice beside it).
//!
//! `LLVQ_UPDATE_OPENAPI=1 cargo test -p llvq-runtime --test openapi` rewrites
//! the committed copy after a deliberate change.

use serde_json::Value;
use std::collections::BTreeSet;

/// Ollama's `main` at the time of the plan (2026-11-06).
const OLLAMA_COMMIT: &str = "6362e0a30887bdd4f69e07a94e8a513383576204";

fn ours() -> Value {
    serde_json::to_value(llvq_runtime::api::openapi()).unwrap()
}

fn ollama() -> Value {
    let y: serde_yaml::Value = serde_yaml::from_str(include_str!("fixtures/ollama-openapi.yaml")).unwrap();
    serde_json::to_value(y).unwrap()
}

const METHODS: [&str; 5] = ["get", "post", "delete", "head", "put"];

/// `(path, method)` for every operation of a document.
fn operations(doc: &Value) -> BTreeSet<(String, String)> {
    let mut out = BTreeSet::new();
    for (p, ops) in doc["paths"].as_object().unwrap() {
        for m in METHODS {
            if ops.get(m).is_some() {
                out.insert((p.clone(), m.to_string()));
            }
        }
    }
    out
}

/// The required fields of an operation's JSON body, following one `$ref`.
fn required(doc: &Value, path: &str, method: &str) -> Option<BTreeSet<String>> {
    let schema = doc["paths"][path][method].get("requestBody")?["content"].get("application/json")?["schema"].clone();
    let schema = match schema.get("$ref").and_then(Value::as_str) {
        Some(r) => doc["components"]["schemas"][r.rsplit('/').next().unwrap()].clone(),
        None => schema,
    };
    Some(
        schema
            .get("required")
            .and_then(Value::as_array)
            .map(|a| a.iter().map(|x| x.as_str().unwrap().to_string()).collect())
            .unwrap_or_default(),
    )
}

#[test]
fn the_committed_document_is_the_generated_one() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/openapi.json");
    let generated = serde_json::to_string_pretty(&ours()).unwrap() + "\n";
    if std::env::var("LLVQ_UPDATE_OPENAPI").as_deref() == Ok("1") {
        std::fs::write(path, &generated).unwrap();
        return;
    }
    let committed = std::fs::read_to_string(path)
        .unwrap_or_else(|_| panic!("{path} is missing: LLVQ_UPDATE_OPENAPI=1 writes it"));
    assert!(
        committed == generated,
        "openapi.json differs from the document the handlers generate; if the change is deliberate, \
         rerun with LLVQ_UPDATE_OPENAPI=1 and commit the file"
    );
}

#[test]
fn the_document_is_openapi_3_1_and_names_every_served_route() {
    let doc = ours();
    assert!(doc["openapi"].as_str().unwrap().starts_with("3.1"));
    let ops = operations(&doc);
    for (p, m) in [
        ("/api/version", "get"),
        ("/api/tags", "get"),
        ("/api/ps", "get"),
        ("/api/show", "post"),
        ("/api/pull", "post"),
        ("/api/delete", "delete"),
        ("/api/copy", "post"),
        ("/api/chat", "post"),
        ("/api/generate", "post"),
        ("/v1/models", "get"),
        ("/v1/models/{model}", "get"),
        ("/v1/chat/completions", "post"),
        ("/v1/completions", "post"),
        ("/openapi.json", "get"),
        ("/", "get"),
    ] {
        assert!(ops.contains(&(p.to_string(), m.to_string())), "{} {p} is not in the document", m.to_uppercase());
    }
}

/// Routes we answer that Ollama's document does not list: the legacy
/// embeddings route, which Ollama still serves, the OpenAI routes, which its
/// compatibility page describes outside `openapi.yaml`, and our own document
/// and chat page.
fn outside_ollamas_document(path: &str) -> bool {
    path == "/api/embeddings" || path.starts_with("/v1/") || path == "/openapi.json" || path == "/"
}

#[test]
fn every_ollama_route_we_answer_is_in_ollamas_document_with_its_required_fields() {
    let (ours, theirs) = (ours(), ollama());
    let theirs_ops = operations(&theirs);
    let mut checked = 0;
    for (p, m) in operations(&ours) {
        if outside_ollamas_document(&p) {
            continue;
        }
        assert!(
            theirs_ops.contains(&(p.clone(), m.clone())),
            "{} {p} is not in Ollama's openapi.yaml at {OLLAMA_COMMIT}",
            m.to_uppercase()
        );
        let not_served = ours["paths"][&p][&m]["tags"].as_array().is_some_and(|t| t.iter().any(|x| x == "not served"));
        if not_served {
            continue;
        }
        assert_eq!(
            required(&ours, &p, &m),
            required(&theirs, &p, &m),
            "{} {p}: required request fields differ from Ollama's",
            m.to_uppercase()
        );
        checked += 1;
    }
    assert_eq!(checked, 9, "the nine served Ollama operations");
}

/// A route added with a plain `.route()` would be served and missing from
/// the document. Every route goes through `routes!` and its `#[utoipa::path]`.
#[test]
fn every_route_goes_through_its_annotation() {
    for (f, src) in [("api.rs", include_str!("../src/api.rs")), ("openai.rs", include_str!("../src/openai.rs"))] {
        for bypass in [".route(", ".route_service(", ".nest(", ".nest_service(", ".fallback("] {
            assert!(
                !src.contains(bypass),
                "{f}: {bypass} bypasses the OpenAPI document; use .routes(routes!(handler)) with #[utoipa::path]"
            );
        }
    }
}
