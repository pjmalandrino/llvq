//! `llvq`: pull, run and serve the sealed files with Ollama's command set.
//!
//! ```text
//! llvq serve                       start the server on LLVQ_HOST (127.0.0.1:11435)
//! llvq pull qwen3:4b               download, check size and sha256, register
//! llvq pull qwen3:4b --from FILE   register a local copy, same checks, no network
//! llvq run qwen3:4b                interactive chat; /bye leaves, /clear forgets
//! llvq run qwen3:4b "a prompt"     one answer, streamed, then exit; stdin is read when piped
//! llvq ls | ps | show | stop | rm | cp
//! ```
//!
//! `llvq serve` also serves a chat page at `/`, on the same routes.
//!
//! The client commands talk HTTP to `LLVQ_HOST`, as Ollama's do. `ls`, `rm`,
//! `cp`, `show` and `pull` work on the store directly when no server answers.
//! `run` starts a server in the background when none answers (decision D7),
//! logging to `LLVQ_HOME/logs/server.log`.

#![forbid(unsafe_code)]

use clap::{Parser, Subcommand};
use llvq_runtime::client::{Client, ClientError};
use llvq_runtime::library::{self, Backend, ModelName};
use llvq_runtime::store::{PullEvent, Store};
use serde_json::{json, Value};
use std::io::{IsTerminal, Read, Write};
use std::sync::Arc;
use std::time::Duration;

#[derive(Parser)]
#[command(name = "llvq", about = "Run the sealed LLVQ files with Ollama's commands and API", disable_version_flag = true)]
struct Cli {
    /// Print the version and the commit.
    #[arg(short = 'v', long = "version")]
    version: bool,
    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Start the server on LLVQ_HOST.
    Serve,
    /// Download a model of the library, check its size and sha256, register it.
    Pull {
        model: String,
        /// Register a local copy of the sealed file instead, with the same checks.
        #[arg(long)]
        from: Option<std::path::PathBuf>,
    },
    /// Chat with a model, or answer one prompt and exit.
    Run {
        model: String,
        /// The prompt; without it, an interactive chat (or stdin, when piped).
        prompt: Vec<String>,
        /// Let the model reason out loud before it answers.
        #[arg(long)]
        think: bool,
        /// Print the counts and timings of each answer.
        #[arg(long)]
        verbose: bool,
        /// Sampling temperature; 0 is greedy. Default: the server's.
        #[arg(long)]
        temperature: Option<f32>,
        /// How long the model stays loaded after the last answer, e.g. 5m.
        #[arg(long)]
        keepalive: Option<String>,
    },
    /// List the models in the store.
    #[command(alias = "list")]
    Ls,
    /// List the loaded models.
    Ps,
    /// Show a model's manifest and served configs.
    Show { model: String },
    /// Unload a model.
    Stop { model: String },
    /// Remove models from the store.
    Rm { models: Vec<String> },
    /// Copy a model to a second name.
    Cp { source: String, destination: String },
}

fn name(s: &str) -> anyhow::Result<ModelName> {
    ModelName::parse(s).map_err(anyhow::Error::msg)
}

fn human(bytes: u64) -> String {
    let b = bytes as f64;
    match b {
        b if b >= 1e9 => format!("{:.1} GB", b / 1e9),
        b if b >= 1e6 => format!("{:.0} MB", b / 1e6),
        b if b >= 1e3 => format!("{:.0} KB", b / 1e3),
        _ => format!("{bytes} B"),
    }
}

fn ago(secs: u64) -> String {
    let d = llvq_runtime::store::now_secs().saturating_sub(secs);
    match d {
        0..=59 => "seconds ago".into(),
        60..=3599 => format!("{} minutes ago", d / 60),
        3600..=86_399 => format!("{} hours ago", d / 3600),
        _ => format!("{} days ago", d / 86_400),
    }
}

/// One progress line on stderr, rewritten in place.
fn progress(e: &PullEvent) {
    match e {
        PullEvent::Status(s) => eprintln!("\r\x1b[K{s}"),
        PullEvent::Progress { digest, total, completed } => {
            let pct = match *total {
                0 => 100.0,
                t => *completed as f64 * 100.0 / t as f64,
            };
            eprint!(
                "\r\x1b[Kpulling {}… {pct:3.0}% {} / {}",
                &digest[7..digest.len().min(19)],
                human(*completed),
                human(*total)
            );
            let _ = std::io::stderr().flush();
        }
    }
}

fn status_to_event(v: &Value) -> PullEvent {
    match (v.get("digest").and_then(Value::as_str), v.get("total").and_then(Value::as_u64)) {
        (Some(d), Some(t)) => PullEvent::Progress {
            digest: d.to_string(),
            total: t,
            completed: v.get("completed").and_then(Value::as_u64).unwrap_or(0),
        },
        _ => PullEvent::Status(v.get("status").and_then(Value::as_str).unwrap_or("").to_string()),
    }
}

fn entry(n: &ModelName) -> anyhow::Result<&'static library::Entry> {
    library::find(n).ok_or_else(|| {
        let known: Vec<String> = library::entries().iter().map(|e| e.model_name().to_string()).collect();
        anyhow::anyhow!("{n} is not in the llvq library, which holds {}", known.join(", "))
    })
}

fn pull(client: &Client, model: &str, from: Option<std::path::PathBuf>) -> anyhow::Result<()> {
    let n = name(model)?;
    let e = entry(&n)?;
    if let Some(file) = from {
        Store::from_env()?.import(&n, e, &file, &mut |ev| progress(&ev))?;
        return Ok(());
    }
    if client.alive() {
        return client.stream("/api/pull", &json!({ "model": n.to_string() }), &mut |v| {
            progress(&status_to_event(v));
            Ok(())
        });
    }
    Store::from_env()?.pull(&n, e, &llvq_runtime::store::hub_endpoint(), &mut |ev| progress(&ev))?;
    Ok(())
}

fn ls(client: &Client) -> anyhow::Result<()> {
    let rows: Vec<(String, String, u64, String)> = match client.alive() {
        true => client
            .get("/api/tags")?
            .get("models")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
            .iter()
            .map(|m| {
                let s = |k: &str| m.get(k).and_then(Value::as_str).unwrap_or("").to_string();
                let size = m.get("size").and_then(Value::as_u64).unwrap_or(0);
                // `2026-11-06T18:00:19.000000000Z` as `2026-11-06 18:00 UTC`.
                let at = s("modified_at");
                let when = match at.len() >= 16 {
                    true => format!("{} {} UTC", &at[..10], &at[11..16]),
                    false => at,
                };
                (s("name"), s("digest"), size, when)
            })
            .collect(),
        false => Store::from_env()?
            .list()?
            .into_iter()
            .map(|(n, m)| (n.to_string(), m.sha256.clone(), m.size, ago(m.modified_at)))
            .collect(),
    };
    println!("{:<24} {:<14} {:<10} MODIFIED", "NAME", "ID", "SIZE");
    for (n, d, size, when) in rows {
        println!("{:<24} {:<14} {:<10} {when}", n, &d[..d.len().min(12)], human(size));
    }
    Ok(())
}

fn ps(client: &Client) -> anyhow::Result<()> {
    anyhow::ensure!(client.alive(), "no llvq server answers at {}; start one with `llvq serve`", client.base());
    let v = client.get("/api/ps")?;
    println!("{:<20} {:<10} {:<10} {:<10} {:<34} {:<6} UNTIL", "NAME", "SIZE", "STATE", "BACKEND", "CONFIG", "QUEUE");
    for m in v.get("models").and_then(Value::as_array).cloned().unwrap_or_default() {
        let s = |k: &str| m.get(k).and_then(Value::as_str).unwrap_or("").to_string();
        let l = m.get("llvq").cloned().unwrap_or(Value::Null);
        let ls = |k: &str| l.get(k).and_then(Value::as_str).unwrap_or("-").to_string();
        println!(
            "{:<20} {:<10} {:<10} {:<10} {:<34} {:<6} {}",
            s("name"),
            human(m.get("size").and_then(Value::as_u64).unwrap_or(0)),
            ls("state"),
            ls("backend"),
            ls("config"),
            l.get("queue").and_then(Value::as_u64).unwrap_or(0),
            s("expires_at")
        );
        if let Some(e) = l.get("error").and_then(Value::as_str) {
            println!("  {e}");
        }
    }
    Ok(())
}

fn show(client: &Client, model: &str) -> anyhow::Result<()> {
    let n = name(model)?;
    let v = match client.alive() {
        true => client.send("POST", "/api/show", &json!({ "model": n.to_string() }))?,
        false => {
            let m = Store::from_env()?.get(&n)?.ok_or_else(|| anyhow::anyhow!("model '{n}' not found"))?;
            let served = library::served_configs(&m.configs, &m.sha256);
            json!({ "details": { "family": m.family, "parameter_size": m.parameter_size,
                                  "quantization_level": format!("Tetra {} b/param", m.bits_per_param) },
                    "llvq": { "repo": m.repo, "revision": m.revision, "file": m.file, "sha256": m.sha256,
                              "config_cuda": served.cuda, "config_metal": served.metal,
                              "config_cpu": served.cpu,
                              "backend": Backend::of_build().label() } })
        }
    };
    let d = &v["details"];
    let l = &v["llvq"];
    let s = |x: &Value| x.as_str().unwrap_or("-").to_string();
    println!("  Model");
    println!("    architecture        {}", s(&d["family"]));
    println!("    parameters          {}", s(&d["parameter_size"]));
    println!("    quantization        {}", s(&d["quantization_level"]));
    println!("  Source");
    println!("    repository          {}", s(&l["repo"]));
    println!("    revision            {}", s(&l["revision"]));
    println!("    file                {}", s(&l["file"]));
    println!("    sha256              {}", s(&l["sha256"]));
    println!("  Served");
    println!("    config on cuda      {}", s(&l["config_cuda"]));
    println!("    config on metal     {}", s(&l["config_metal"]));
    println!("    config on cpu       {}", s(&l["config_cpu"]));
    println!("    this build serves   {}", s(&l["backend"]));
    Ok(())
}

fn stop(client: &Client, model: &str) -> anyhow::Result<()> {
    let n = name(model)?;
    anyhow::ensure!(client.alive(), "no llvq server answers at {}", client.base());
    client.send("POST", "/api/generate", &json!({ "model": n.to_string(), "keep_alive": 0, "stream": false }))?;
    Ok(())
}

fn rm(client: &Client, models: &[String]) -> anyhow::Result<()> {
    let alive = client.alive();
    for m in models {
        let n = name(m)?;
        let found = match alive {
            true => match client.send("DELETE", "/api/delete", &json!({ "model": n.to_string() })) {
                Ok(_) => true,
                Err(ClientError::Status(404, _)) => false,
                Err(e) => return Err(e.into()),
            },
            false => Store::from_env()?.remove(&n)?,
        };
        anyhow::ensure!(found, "model '{n}' not found");
        println!("deleted '{n}'");
    }
    Ok(())
}

fn cp(client: &Client, src: &str, dst: &str) -> anyhow::Result<()> {
    let (a, b) = (name(src)?, name(dst)?);
    let found = match client.alive() {
        true => match client.send("POST", "/api/copy", &json!({ "source": a.to_string(), "destination": b.to_string() })) {
            Ok(_) => true,
            Err(ClientError::Status(404, _)) => false,
            Err(e) => return Err(e.into()),
        },
        false => Store::from_env()?.copy(&a, &b)?,
    };
    anyhow::ensure!(found, "model '{a}' not found");
    println!("copied '{a}' to '{b}'");
    Ok(())
}

/// Decision D7: a `run` with no server starts one in the background.
fn ensure_server(client: &Client) -> anyhow::Result<()> {
    if client.alive() {
        return Ok(());
    }
    let store = Store::from_env()?;
    let logs = store.root().join("logs");
    std::fs::create_dir_all(&logs)?;
    let log = logs.join("server.log");
    let out = std::fs::OpenOptions::new().create(true).append(true).open(&log)?;
    eprintln!("starting llvq serve in the background (log: {})", log.display());
    std::process::Command::new(std::env::current_exe()?)
        .arg("serve")
        .stdin(std::process::Stdio::null())
        .stdout(out.try_clone()?)
        .stderr(out)
        .spawn()?;
    for _ in 0..100 {
        if client.alive() {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    anyhow::bail!("the server did not answer within 10 s; see {}", log.display())
}

struct RunOpts {
    think: bool,
    verbose: bool,
    temperature: Option<f32>,
    keepalive: Option<String>,
}

/// One chat turn through `/api/chat`, streamed to stdout. Returns the answer.
fn turn(client: &Client, model: &str, history: &[Value], o: &RunOpts) -> anyhow::Result<String> {
    let mut body = json!({ "model": model, "messages": history, "stream": true, "think": o.think });
    if let Some(t) = o.temperature {
        body["options"] = json!({ "temperature": t });
    }
    if let Some(k) = &o.keepalive {
        body["keep_alive"] = json!(k);
    }
    let mut answer = String::new();
    let mut thinking = false;
    let mut out = std::io::stdout();
    client.stream("/api/chat", &body, &mut |v| {
        let m = &v["message"];
        if let Some(t) = m.get("thinking").and_then(Value::as_str).filter(|t| !t.is_empty()) {
            if !thinking {
                writeln!(out, "\x1b[2mThinking...")?;
                thinking = true;
            }
            write!(out, "{t}")?;
        }
        if let Some(c) = m.get("content").and_then(Value::as_str).filter(|c| !c.is_empty()) {
            if thinking {
                write!(out, "\n...done thinking.\x1b[0m\n\n")?;
                thinking = false;
            }
            write!(out, "{c}")?;
            answer.push_str(c);
        }
        out.flush()?;
        if v.get("done").and_then(Value::as_bool) == Some(true) {
            if thinking {
                write!(out, "\x1b[0m")?;
            }
            writeln!(out)?;
            if let Some(w) = v.get("warnings").and_then(Value::as_array) {
                for w in w {
                    eprintln!("warning: {}", w.as_str().unwrap_or(""));
                }
            }
            if o.verbose {
                let n = |k: &str| v.get(k).and_then(Value::as_u64).unwrap_or(0);
                let rate = |c: u64, ns: u64| c as f64 / (ns as f64 / 1e9).max(1e-9);
                eprintln!(
                    "\nprompt: {} tokens ({} cached) in {:.2} s\nanswer: {} tokens in {:.2} s, {:.1} tok/s \
                     (this client's rate, not a measurement)\nend: {}",
                    n("prompt_eval_count"),
                    n("prompt_eval_cached_count"),
                    n("prompt_eval_duration") as f64 / 1e9,
                    n("eval_count"),
                    n("eval_duration") as f64 / 1e9,
                    rate(n("eval_count"), n("eval_duration")),
                    v.get("done_reason").and_then(Value::as_str).unwrap_or("")
                );
            }
        }
        Ok(())
    })?;
    Ok(answer)
}

fn run(client: &Client, model: &str, prompt: Vec<String>, o: RunOpts) -> anyhow::Result<()> {
    let n = name(model)?;
    ensure_server(client)?;
    let model = n.to_string();
    if let Err(ClientError::Status(404, _)) = client.send("POST", "/api/show", &json!({ "model": model })) {
        entry(&n)?;
        eprintln!("{model} is not in the store; pulling it");
        client.stream("/api/pull", &json!({ "model": model }), &mut |v| {
            progress(&status_to_event(v));
            Ok(())
        })?;
    }
    let mut text = prompt.join(" ");
    let stdin = std::io::stdin();
    if !stdin.is_terminal() {
        let mut piped = String::new();
        stdin.lock().read_to_string(&mut piped)?;
        if !piped.trim().is_empty() {
            text = match text.is_empty() {
                true => piped,
                false => format!("{text}\n\n{piped}"),
            };
        }
    }
    if !text.trim().is_empty() {
        turn(client, &model, &[json!({ "role": "user", "content": text })], &o)?;
        return Ok(());
    }
    eprint!("loading {model}… ");
    let t0 = std::time::Instant::now();
    let mut load = json!({ "model": model, "stream": false });
    if let Some(k) = &o.keepalive {
        load["keep_alive"] = json!(k);
    }
    client.send("POST", "/api/generate", &load)?;
    eprintln!("{:.1} s", t0.elapsed().as_secs_f64());
    eprintln!("/bye leaves, /clear forgets the conversation\n");
    let mut history: Vec<Value> = Vec::new();
    let mut line = String::new();
    loop {
        print!(">>> ");
        std::io::stdout().flush()?;
        line.clear();
        if stdin.read_line(&mut line)? == 0 {
            println!();
            return Ok(());
        }
        let msg = line.trim();
        match msg {
            "" => continue,
            "/bye" | "/exit" => return Ok(()),
            "/clear" | "/reset" => {
                history.clear();
                eprintln!("conversation cleared\n");
                continue;
            }
            "/?" | "/help" => {
                eprintln!("/bye leaves, /clear forgets the conversation\n");
                continue;
            }
            _ => {}
        }
        history.push(json!({ "role": "user", "content": msg }));
        match turn(client, &model, &history, &o) {
            Ok(a) => {
                history.push(json!({ "role": "assistant", "content": a }));
                println!();
            }
            Err(e) => {
                history.pop();
                eprintln!("error: {e:#}\n");
            }
        }
    }
}

fn serve() -> anyhow::Result<()> {
    let host = llvq_runtime::client::host();
    let store = Arc::new(Store::from_env()?);
    let models = llvq_runtime::engine::Models::from_env(store.clone(), Arc::new(llvq_runtime::engine::DeviceLoader))?;
    let state = llvq_runtime::api::AppState::new(Arc::new(models), llvq_runtime::store::hub_endpoint());
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(&host)
            .await
            .map_err(|e| anyhow::anyhow!("binding {host}: {e}"))?;
        eprintln!(
            "llvq {} ({}) serving on http://{host}, backend {}, store {}",
            env!("CARGO_PKG_VERSION"),
            env!("LLVQ_BUILD_COMMIT"),
            Backend::of_build().label(),
            store.root().display()
        );
        eprintln!("chat in the browser: http://{}/", host.replace("0.0.0.0", "127.0.0.1"));
        llvq_runtime::api::serve(listener, state).await
    })
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    if cli.version {
        println!("llvq version {} ({})", env!("CARGO_PKG_VERSION"), env!("LLVQ_BUILD_COMMIT"));
        return Ok(());
    }
    let client = Client::from_env();
    match cli.cmd {
        None => {
            use clap::CommandFactory;
            Cli::command().print_help()?;
            Ok(())
        }
        Some(Cmd::Serve) => serve(),
        Some(Cmd::Pull { model, from }) => pull(&client, &model, from),
        Some(Cmd::Run { model, prompt, think, verbose, temperature, keepalive }) => {
            run(&client, &model, prompt, RunOpts { think, verbose, temperature, keepalive })
        }
        Some(Cmd::Ls) => ls(&client),
        Some(Cmd::Ps) => ps(&client),
        Some(Cmd::Show { model }) => show(&client, &model),
        Some(Cmd::Stop { model }) => stop(&client, &model),
        Some(Cmd::Rm { models }) => rm(&client, &models),
        Some(Cmd::Cp { source, destination }) => cp(&client, &source, &destination),
    }
}
