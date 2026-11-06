//! What the command line sends to a running server, over HTTP, as Ollama's
//! own client does. `LLVQ_HOST` names the server.

use serde_json::Value;
use std::io::BufRead;
use std::time::Duration;

/// The default address, decision D1: 11434 is Ollama's, and both servers may
/// run on one machine.
pub const DEFAULT_HOST: &str = "127.0.0.1:11435";

/// `LLVQ_HOST` as `host:port`, without a scheme. `0.0.0.0` binds every
/// interface for `serve` and is reached as 127.0.0.1 by the client.
pub fn host() -> String {
    let h = std::env::var("LLVQ_HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string());
    let h = h.trim().trim_start_matches("http://").trim_end_matches('/').to_string();
    match h.contains(':') {
        true => h,
        false => format!("{h}:11435"),
    }
}

pub struct Client {
    base: String,
    agent: ureq::Agent,
}

/// An error the server answered with, or a connection that failed.
#[derive(Debug)]
pub enum ClientError {
    Status(u16, String),
    Transport(String),
}

impl std::fmt::Display for ClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClientError::Status(s, m) => write!(f, "{m} (HTTP {s})"),
            ClientError::Transport(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for ClientError {}

fn convert(e: ureq::Error) -> ClientError {
    match e {
        ureq::Error::Status(code, resp) => {
            let text = resp.into_string().unwrap_or_default();
            let msg = serde_json::from_str::<Value>(&text)
                .ok()
                .and_then(|v| {
                    v.get("error").and_then(|e| e.as_str().map(str::to_string).or_else(|| {
                        e.get("message").and_then(Value::as_str).map(str::to_string)
                    }))
                })
                .unwrap_or(text);
            ClientError::Status(code, msg)
        }
        ureq::Error::Transport(t) => ClientError::Transport(t.to_string()),
    }
}

impl Client {
    pub fn new(host: &str) -> Self {
        let connect = host.replace("0.0.0.0", "127.0.0.1");
        Client {
            base: format!("http://{connect}"),
            agent: ureq::AgentBuilder::new().timeout_connect(Duration::from_secs(2)).build(),
        }
    }

    pub fn from_env() -> Self {
        Self::new(&host())
    }

    pub fn base(&self) -> &str {
        &self.base
    }

    /// Whether a server answers `/api/version`.
    pub fn alive(&self) -> bool {
        self.agent
            .get(&format!("{}/api/version", self.base))
            .timeout(Duration::from_secs(2))
            .call()
            .is_ok()
    }

    pub fn get(&self, path: &str) -> Result<Value, ClientError> {
        let r = self.agent.get(&format!("{}{path}", self.base)).call().map_err(convert)?;
        r.into_json().map_err(|e| ClientError::Transport(e.to_string()))
    }

    /// A request with a JSON body; the reply's body, empty or JSON.
    pub fn send(&self, method: &str, path: &str, body: &Value) -> Result<Value, ClientError> {
        let r = self.agent.request(method, &format!("{}{path}", self.base)).send_json(body.clone()).map_err(convert)?;
        let text = r.into_string().map_err(|e| ClientError::Transport(e.to_string()))?;
        match text.trim().is_empty() {
            true => Ok(Value::Null),
            false => serde_json::from_str(&text).map_err(|e| ClientError::Transport(e.to_string())),
        }
    }

    /// POST an NDJSON route and hand each line to `on_line`. A line that
    /// carries `error` ends the stream with that error.
    pub fn stream(
        &self,
        path: &str,
        body: &Value,
        on_line: &mut dyn FnMut(&Value) -> anyhow::Result<()>,
    ) -> anyhow::Result<()> {
        let r = self.agent.post(&format!("{}{path}", self.base)).send_json(body.clone()).map_err(convert)?;
        let reader = std::io::BufReader::new(r.into_reader());
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let v: Value = serde_json::from_str(&line)?;
            if let Some(e) = v.get("error").and_then(Value::as_str) {
                anyhow::bail!("{e}");
            }
            on_line(&v)?;
        }
        Ok(())
    }
}
