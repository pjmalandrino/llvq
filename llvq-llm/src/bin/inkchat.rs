//! `inkchat`: a conversation with Inkling 2 Mini in this engine, its experts resident in lattice codes, loaded as
//! `inkrun` loads them. One line of input is one user message.
//!
//! ```text
//! inkchat --config D/textonly/config.json --checkpoint D/textonly/model.safetensors --layers D/layers \
//!         --tokenizer D/native/tokenizer.json --device metal|cuda|cpu \
//!         [--tables <dir of embed.q4g64.llvqraw and unembed.q4g64.llvqraw>] \
//!         [--glue candle|fused] [--attn candle|fused|fused+scores] [--require-layers 24] \
//!         [--effort none] [--system <text>] [--max-new 2048] [--max-context 4096] \
//!         [--temp 0] [--top-p 0.95] [--seed 1] [--thinking show|hide]
//! ```
//!
//! ## What it is, and what it is not
//!
//! A demonstrator, to read the model's answers by eye. Nothing here is measured: the numbers of this model come
//! from `inkrun` and `inkgsm`, which hold their protocol. The rate printed after a turn is this loop's rate, with
//! a parser and a terminal in every step.
//!
//! ## The conversation
//!
//! - The prompt is the checkpoint's template, through `llvq_llm::inkchat`. The first message is rendered with the
//!   reasoning effort (`--effort`, `none` by default, the setting of the GSM8K runs; the template's own default is
//!   0.9) and `--system` when given. The template has no generation prompt: the model writes its own
//!   `<|message_model|>`.
//! - The cache is kept across turns, and it holds the model's own ids: a later message feeds the end of sampling of
//!   the turn before it and `InkChat::follow_up` of the message, nothing else. The model's past thinking stays in
//!   the context as it wrote it.
//! - A turn ends on `<|content_model_end_sampling|>`, or after `--max-new` tokens. A turn cut inside a text or a
//!   thinking part is closed as the template would close it, and the conversation goes on. A turn cut anywhere
//!   else, and a turn the template's parser refuses, drop the conversation: the cache would hold a turn the
//!   template never writes.
//! - `--temp 0` (default) is greedy, the first maximum of the f32 logits, as `inkrun` and `inkgsm` take it. Above
//!   zero the sampler is `llvq_llm::repl::Sampler` (temperature, then nucleus `--top-p`), seeded by `--seed`.
//!   The checkpoint's `generation_config.json` samples at temperature 1.0.
//! - `--tables` reads the two vocabulary tables of the sealed file instead of the checkpoint's. On Metal the head
//!   runs on the int4 table (`inkling::head`) and the embedding rows are decoded on the host; elsewhere both are
//!   decoded dense, as `inkgsm --tables` does.
//! - `--max-context` refuses a message that could carry the conversation past that many positions. The model has
//!   no position limit of its own that a chat reaches; the cache grows with every position.
//!
//! Commands: `/reset` drops the conversation, `/effort <level>` sets the effort and drops the conversation (the
//! effort is written once, at its start), `/bye` or the end of input leaves.

use std::collections::HashMap;
use std::io::{BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, bail, ensure, Context, Result};
use candle_core::{DType, Device, Tensor};

use llvq_llm::inkchat::{effort_text, mark, Delta, InkChat, Json, TurnParser};
use llvq_llm::inkling::{tables, AttnKind, GlueKind, Inkling, LoadOptions};
use llvq_llm::lm::CausalLm;
use llvq_llm::repl::Sampler;

const DEFAULT_MAX_NEW: usize = 2048;
const DEFAULT_MAX_CONTEXT: usize = 4096;

fn usage() -> ! {
    eprintln!(
        "usage: inkchat --config <config.json> --checkpoint <model.safetensors> --layers <D/layers> \
         --tokenizer <tokenizer.json> --device metal|cuda|cpu [--tables <dir>] [--glue candle|fused] \
         [--attn candle|fused|fused+scores] [--require-layers N] [--effort none] [--system <text>] \
         [--max-new {DEFAULT_MAX_NEW}] [--max-context {DEFAULT_MAX_CONTEXT}] [--temp 0] [--top-p 0.95] [--seed 1] \
         [--thinking show|hide]"
    );
    std::process::exit(2)
}

struct Args {
    config: PathBuf,
    checkpoint: PathBuf,
    layers: PathBuf,
    tokenizer: PathBuf,
    device: String,
    tables: Option<PathBuf>,
    glue: GlueKind,
    attn: AttnKind,
    require_layers: usize,
    effort: String,
    system: Option<String>,
    max_new: usize,
    max_context: usize,
    temp: f32,
    top_p: f32,
    seed: u64,
    show_thinking: bool,
}

fn parse() -> Result<Args> {
    let mut a: HashMap<String, String> = HashMap::new();
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let Some(name) = k.strip_prefix("--") else { usage() };
        let v = it.next().unwrap_or_else(|| usage());
        ensure!(a.insert(name.to_string(), v).is_none(), "--{name} given twice");
    }
    let mut take = |k: &str| a.remove(k);
    let need = |v: Option<String>, k: &str| v.ok_or_else(|| anyhow!("--{k} is required"));
    fn num<T: std::str::FromStr>(v: Option<String>, k: &str, default: T) -> Result<T> {
        match v {
            None => Ok(default),
            Some(s) => s.trim().parse().map_err(|_| anyhow!("--{k} {s:?} is not a number")),
        }
    }
    let args = Args {
        config: need(take("config"), "config")?.into(),
        checkpoint: need(take("checkpoint"), "checkpoint")?.into(),
        layers: need(take("layers"), "layers")?.into(),
        tokenizer: need(take("tokenizer"), "tokenizer")?.into(),
        device: need(take("device"), "device")?,
        tables: take("tables").map(PathBuf::from),
        glue: take("glue").as_deref().unwrap_or("candle").parse()?,
        attn: take("attn").as_deref().unwrap_or("candle").parse()?,
        require_layers: num(take("require-layers"), "require-layers", 0)?,
        effort: take("effort").unwrap_or_else(|| "none".into()),
        system: take("system"),
        max_new: num(take("max-new"), "max-new", DEFAULT_MAX_NEW)?,
        max_context: num(take("max-context"), "max-context", DEFAULT_MAX_CONTEXT)?,
        temp: num(take("temp"), "temp", 0.0)?,
        top_p: num(take("top-p"), "top-p", 0.95)?,
        seed: num(take("seed"), "seed", 1)?,
        show_thinking: match take("thinking").as_deref() {
            None | Some("show") => true,
            Some("hide") => false,
            Some(o) => bail!("--thinking {o}: show or hide"),
        },
    };
    if let Some(k) = a.keys().next() {
        bail!("unknown option --{k}");
    }
    ensure!(args.max_new > 0, "--max-new 0 generates nothing");
    ensure!(args.temp >= 0.0 && args.temp.is_finite(), "--temp {}: zero or more", args.temp);
    ensure!(args.top_p > 0.0 && args.top_p <= 1.0, "--top-p {}: in (0, 1]", args.top_p);
    Ok(args)
}

fn device(name: &str) -> Result<Device> {
    Ok(match name {
        "cpu" => Device::Cpu,
        "metal" => Device::new_metal(0).context("Metal device (built with --features metal?)")?,
        "cuda" => Device::new_cuda(0).context("CUDA device (built with --features cuda?)")?,
        o => bail!("--device {o}: cpu, metal or cuda"),
    })
}

/// The effort as the template reads it: a number, or one of its names. Refused here when the template refuses it.
fn effort_json(effort: &str) -> Result<Json> {
    let j = match effort.parse::<f64>() {
        Ok(_) => Json::parse(effort)?,
        Err(_) => Json::str(effort),
    };
    effort_text(Some(&j)).with_context(|| format!("effort {effort}"))?;
    Ok(j)
}

#[allow(unused_variables, clippy::match_single_binding)]
fn make_resident(model: &mut Inkling, layers: &Path, dev: &Device) -> Result<()> {
    match dev {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        Device::Metal(_) => {
            let rt = llvq_llm::inkling::resident_runtime(dev, llvq_llm::inkling::DEFAULT_PREFILL_ROWS)?;
            model.make_resident(layers, rt)?;
        }
        #[cfg(all(target_os = "linux", feature = "cuda"))]
        Device::Cuda(_) => {
            let rt = llvq_llm::inkling::resident_cuda::served_runtime(dev, &model.cfg)?;
            llvq_llm::inkling::resident_cuda::make_resident(model, layers, rt)?;
        }
        _ => eprintln!("inkchat: no resident path on this device; every layer keeps its native experts"),
    }
    Ok(())
}

fn load(args: &Args, dev: &Device) -> Result<Inkling> {
    // Where the int4 head has a kernel: the table stays packed on the device.
    let on_metal = matches!(dev, Device::Metal(_) | Device::Cuda(_));
    // On Metal the sealed file's head stays int4 on the device; its file is named at load.
    let int4_head = match &args.tables {
        Some(d) if on_metal => Some(d.join(tables::UNEMBED_FILE)),
        _ => None,
    };
    let t = Instant::now();
    let mut model = Inkling::load_with_attn(&args.config, &args.checkpoint, dev, &LoadOptions { int4_head }, args.attn)?;
    dev.synchronize()?;
    eprintln!("inkchat: dense weights on {} in {:.1} s", args.device, t.elapsed().as_secs_f64());
    let t = Instant::now();
    make_resident(&mut model, &args.layers, dev)?;
    dev.synchronize()?;
    let resident = model.substituted_layers();
    eprintln!("inkchat: {} layers resident in {:.1} s", resident.len(), t.elapsed().as_secs_f64());
    ensure!(
        resident.len() >= args.require_layers,
        "{} layers resident, --require-layers {}",
        resident.len(),
        args.require_layers
    );
    if let Some(d) = &args.tables {
        if on_metal {
            let (vocab, hidden) = (model.cfg.vocab_size, model.cfg.hidden_size);
            let rows = tables::decode_table(&d.join(tables::EMBED_FILE), tables::EMBED_NAME, vocab, hidden)?;
            model.embed_table = Some(tables::EmbedTable::from_rows(rows, vocab, hidden)?);
        } else {
            let t = tables::load(d, &model.cfg, dev)?;
            model.embed_table = Some(t.embed);
            model.out_proj = Some(Box::new(t.head));
        }
        eprintln!("inkchat: the two vocabulary tables of the sealed file from {}", d.display());
    }
    model.set_glue(args.glue)?;
    model.take_launches();
    Ok(model)
}

/// The next token from one row of logits: the first maximum at temperature zero, the sampler's draw above it.
fn pick(logits: &Tensor, sampler: &mut Sampler) -> Result<u32> {
    let row = logits.flatten_all()?.to_dtype(DType::F32)?;
    if sampler.temp > 0.0 {
        return sampler.pick(&row);
    }
    let v = row.to_vec1::<f32>()?;
    let (mut i1, mut b1) = (0usize, f32::NEG_INFINITY);
    for (i, &x) in v.iter().enumerate() {
        if x > b1 {
            b1 = x;
            i1 = i;
        }
    }
    ensure!(b1.is_finite(), "logits without a finite maximum");
    Ok(i1 as u32)
}

/// How a turn cut by the cap is closed: the ids the template would write after `said` to end the turn, or `None`
/// when the cut fell where the template has no way to close (a part header, a tool call's JSON).
fn close_cut_turn(chat: &InkChat, said: &[u32]) -> Option<Vec<u32>> {
    let tok = chat.tokenizer();
    let id = |m: &str| tok.token_to_id(m);
    let end_message = id(mark::END_MESSAGE)?;
    let model = id(mark::MODEL)?;
    let bodies = [id(mark::TEXT)?, id(mark::THINKING)?];
    let others = [id(mark::INVOKE_JSON)?, id(mark::INVOKE_TEXT)?, id(mark::XML)?, id(mark::TOOL_ERROR)?, id(mark::IMAGE)?];
    let last = said.iter().rev().find(|t| **t == end_message || **t == model || bodies.contains(t) || others.contains(t))?;
    if *last == end_message {
        Some(vec![chat.end_sampling])
    } else if bodies.contains(last) {
        Some(vec![end_message, chat.end_sampling])
    } else {
        None
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Shown {
    Nothing,
    Thinking,
    Text,
}

/// Writes the deltas of a turn: thinking dimmed (or between brackets when the output is not a terminal), text
/// plain, a tool call on its own line.
struct Screen {
    color: bool,
    show_thinking: bool,
    last: Shown,
}

impl Screen {
    fn switch(&mut self, to: Shown, out: &mut impl Write) -> std::io::Result<()> {
        if self.last == to {
            return Ok(());
        }
        if self.last == Shown::Thinking && self.show_thinking {
            write!(out, "{}", if self.color { "\x1b[0m\n\n" } else { "\n[/thinking]\n" })?;
        } else if self.last == Shown::Text {
            writeln!(out)?;
        }
        if to == Shown::Thinking && self.show_thinking {
            write!(out, "{}", if self.color { "\x1b[2m" } else { "[thinking]\n" })?;
        }
        self.last = to;
        Ok(())
    }

    fn show(&mut self, deltas: &[Delta], out: &mut impl Write) -> std::io::Result<()> {
        for d in deltas {
            match d {
                Delta::Thinking(t) => {
                    self.switch(Shown::Thinking, out)?;
                    if self.show_thinking {
                        write!(out, "{t}")?;
                    }
                }
                Delta::Text(t) => {
                    self.switch(Shown::Text, out)?;
                    write!(out, "{t}")?;
                }
                Delta::ToolCall(c) => {
                    self.switch(Shown::Nothing, out)?;
                    writeln!(out, "[tool call {} {}]", c.name, c.arguments.dumps(false, false))?;
                }
                Delta::End => {}
            }
        }
        out.flush()
    }
}

enum Stop {
    /// The model ended its turn.
    End,
    /// `--max-new` tokens, the last one not yet in the cache.
    Cap,
    /// The parser refused the turn.
    Refused(anyhow::Error),
}

fn user(text: &str) -> Json {
    Json::Object(vec![("role".into(), Json::str("user")), ("content".into(), Json::str(text))])
}

fn main() -> Result<()> {
    let args = parse()?;
    let chat = InkChat::new(
        tokenizers::Tokenizer::from_file(&args.tokenizer).map_err(|e| anyhow!("{}: {e}", args.tokenizer.display()))?,
    )?;
    let mut effort = effort_json(&args.effort)?;
    let dev = device(&args.device)?;
    let model = load(&args, &dev)?;

    let mut sampler = Sampler { temp: args.temp, top_p: args.top_p, state: args.seed };
    let stdout = std::io::stdout();
    let color = stdout.is_terminal();
    println!(
        "  Inkling 2 Mini, {} resident layers, glue {}, attention {}",
        model.substituted_layers().len(),
        model.glue().name(),
        model.attn().name()
    );
    println!(
        "  effort {}, temp {:.2}, top_p {:.2}, at most {} new tokens a turn, {} positions",
        effort_text(Some(&effort))?,
        args.temp,
        args.top_p,
        args.max_new,
        args.max_context
    );
    println!("  /reset drops the conversation, /effort <level> sets the effort and drops it, /bye leaves\n");

    let mut cache = model.new_cache();
    // What the cache must take before the next message: the end of the turn before it.
    let mut owed: Vec<u32> = Vec::new();
    let stdin = std::io::stdin();
    let mut input = stdin.lock();
    loop {
        print!("> ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        if input.read_line(&mut line)? == 0 {
            println!();
            return Ok(());
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "/bye" {
            return Ok(());
        }
        if line == "/reset" {
            println!("  conversation dropped, {} positions released\n", cache.len());
            cache = model.new_cache();
            owed.clear();
            continue;
        }
        if let Some(level) = line.strip_prefix("/effort") {
            match effort_json(level.trim()) {
                Ok(e) => {
                    effort = e;
                    println!("  effort {}, conversation dropped\n", effort_text(Some(&effort))?);
                    cache = model.new_cache();
                    owed.clear();
                }
                Err(e) => println!("  {e:#}\n"),
            }
            continue;
        }

        let opening = cache.is_empty() && owed.is_empty();
        let ids = if opening {
            let mut msgs = Vec::new();
            if let Some(s) = &args.system {
                msgs.push(Json::Object(vec![("role".into(), Json::str("system")), ("content".into(), Json::str(s))]));
            }
            msgs.push(user(line));
            chat.render(&msgs, &[], Some(&effort)).map(|r| r.ids)
        } else {
            chat.follow_up(&user(line)).map(|tail| owed.iter().copied().chain(tail).collect())
        };
        let ids: Vec<u32> = match ids {
            Ok(ids) => ids,
            Err(e) => {
                println!("  {e:#}\n");
                continue;
            }
        };
        // The answer's worst case, and the two ids that can close a cut turn.
        if cache.len() + ids.len() + args.max_new + 2 > args.max_context {
            println!(
                "  this turn could reach {} positions of the {} allowed, and {} are held. /reset to start over.\n",
                cache.len() + ids.len() + args.max_new + 2,
                args.max_context,
                cache.len()
            );
            continue;
        }

        let t_prompt = Instant::now();
        let mut logits = CausalLm::prefill(&model, &ids, &mut cache)?;
        owed.clear();
        let mut next = pick(&logits, &mut sampler)?;
        let prompt_s = t_prompt.elapsed().as_secs_f64();

        let t_turn = Instant::now();
        let mut parser = TurnParser::new(&chat);
        let mut screen = Screen { color, show_thinking: args.show_thinking, last: Shown::Nothing };
        let mut said: Vec<u32> = Vec::new();
        let mut out = stdout.lock();
        let stop = loop {
            if next == chat.end_sampling {
                break match parser.push_ids(&[next]) {
                    Ok(d) => {
                        screen.show(&d, &mut out)?;
                        Stop::End
                    }
                    Err(e) => Stop::Refused(e),
                };
            }
            said.push(next);
            match parser.push_ids(&[next]) {
                Ok(d) => screen.show(&d, &mut out)?,
                Err(e) => break Stop::Refused(e),
            }
            if said.len() >= args.max_new {
                break Stop::Cap;
            }
            logits = CausalLm::step(&model, next, cache.len(), &mut cache)?;
            next = pick(&logits, &mut sampler)?;
        };
        screen.switch(Shown::Nothing, &mut out)?;
        drop(out);
        let secs = t_turn.elapsed().as_secs_f64();

        let mut dropped = false;
        match stop {
            Stop::End => owed.push(chat.end_sampling),
            Stop::Cap => match close_cut_turn(&chat, &said) {
                Some(close) => {
                    println!("\n  [cut at {} tokens; the turn is closed there]", args.max_new);
                    owed.push(*said.last().expect("the cap is at least one token"));
                    owed.extend(close);
                }
                None => {
                    println!("\n  [cut at {} tokens inside a part the template cannot close]", args.max_new);
                    dropped = true;
                }
            },
            Stop::Refused(e) => {
                println!("\n  [the turn broke the template's format: {e:#}]");
                dropped = true;
            }
        }
        let n = said.len();
        println!(
            "\n  {n} tokens in {secs:.1} s ({:.1} tok/s) after a {}-token prompt read in {prompt_s:.1} s; \
             {} positions held. This loop's rate, not a measurement",
            n as f64 / secs.max(1e-9),
            ids.len(),
            cache.len() + owed.len()
        );
        if dropped {
            println!("  conversation dropped\n");
            cache = model.new_cache();
            owed.clear();
        } else {
            println!();
        }
    }
}
