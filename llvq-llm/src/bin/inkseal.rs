//! `inkseal`: the Inkling 2 Mini seal (lot 1, item 2). See `llvq_llm::inkseal`.
//!
//! ```text
//! inkseal seal --out X.llvq [--layer-dir D/layers/L06 ...] [--records f.llvqrec ...] [--wh-dir DIR | --wh-root ROOT]
//!              [--raw t.llvqraw ...] [--native model.safetensors --sidecar X.sidecar.safetensors]
//!              [--blob name=path ...] [--partial] [--text-only]
//! inkseal verify X.llvq [--layer-dir ...] [--wh-dir|--wh-root ...] [--raw t.llvqraw ...] [--sidecar S] [--native N]
//! inkseal flip IN OUT (--record K | --raw-index K) [--bit B]
//! ```
//!
//! `seal` passes every record through byte for byte; `verify` decodes every
//! lattice record of the sealed file and compares each expert record of the
//! given layers with `w13_hat.f32` / `w2_hat.f32` bit for bit, and, given a
//! sidecar and the native file, compares every sidecar tensor with the native
//! bytes; `flip` writes a copy with one bit of one record's code stream
//! flipped, the negative control of `verify`. Threads from `LLVQ_THREADS`
//! (default 2).

use anyhow::{bail, ensure, Context, Result};
use llvq_llm::inkseal::{self, SealInputs};
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = args.first() else { bail!("usage: inkseal seal|verify|flip ...") };
    let mut it = args[1..].iter();
    let mut pos: Vec<String> = Vec::new();
    let mut inp = SealInputs::default();
    let (mut record, mut raw_index, mut bit) = (None, None, 0u64);
    while let Some(a) = it.next() {
        let mut val = || it.next().cloned().with_context(|| format!("{a} needs a value"));
        match a.as_str() {
            "--out" => inp.out = PathBuf::from(val()?),
            "--layer-dir" => inp.layer_dirs.push(PathBuf::from(val()?)),
            "--records" => inp.record_files.push(PathBuf::from(val()?)),
            "--wh-dir" => inp.wh_dirs.push(PathBuf::from(val()?)),
            "--wh-root" => inp.wh_dirs.extend(inkseal::wh_dirs_under(&PathBuf::from(val()?))?),
            "--raw" => inp.raw_files.push(PathBuf::from(val()?)),
            "--native" => inp.native = Some(PathBuf::from(val()?)),
            "--sidecar" => inp.sidecar = Some(PathBuf::from(val()?)),
            "--blob" => {
                let v = val()?;
                let (n, p) = v.split_once('=').context("--blob takes name=path")?;
                inp.blobs.push((n.to_string(), PathBuf::from(p)));
            }
            "--partial" => inp.partial = true,
            "--text-only" => inp.text_only = true,
            "--record" => record = Some(val()?.parse::<usize>()?),
            "--raw-index" => raw_index = Some(val()?.parse::<usize>()?),
            "--bit" => bit = val()?.parse::<u64>()?,
            _ if a.starts_with("--") => bail!("unknown flag {a}"),
            _ => pos.push(a.clone()),
        }
    }
    match cmd.as_str() {
        "seal" => {
            ensure!(pos.is_empty(), "seal takes no positional argument");
            ensure!(!inp.out.as_os_str().is_empty(), "seal needs --out");
            let t0 = std::time::Instant::now();
            let r = inkseal::seal(&inp)?;
            println!(
                "sealed {}: {} records ({} expert, {} int4), payload {} bits, {} raw tensors, {} bytes",
                inp.out.display(),
                r.records,
                r.expert_records,
                r.int4_records,
                r.payload_bits,
                r.raws,
                r.file_bytes
            );
            if let Some(s) = &inp.sidecar {
                println!("sidecar {}: {} tensors, {} bytes", s.display(), r.sidecar_tensors, r.sidecar_bytes);
            }
            println!(
                "omitted: {} tensors, {} native elements (an NVFP4 byte holds two weights){}",
                r.omitted_tensors.len(),
                r.omitted_elements,
                if inp.partial { " (partial seal)" } else { "" }
            );
            println!("seal: {:.1} s", t0.elapsed().as_secs_f64());
            Ok(())
        }
        "verify" => {
            let [sealed] = pos.as_slice() else { bail!("verify takes the sealed file") };
            let t0 = std::time::Instant::now();
            let r = inkseal::verify_all(&inkseal::VerifyInputs {
                sealed: PathBuf::from(sealed),
                layer_dirs: inp.layer_dirs.clone(),
                wh_dirs: inp.wh_dirs.clone(),
                raw_files: inp.raw_files.clone(),
                sidecar: inp.sidecar.clone(),
                native: inp.native.clone(),
                threads: inkseal::threads()?,
            })?;
            println!("verify {sealed}: {} records, {} lattice decoded, {} decode failures", r.records, r.lattice_decoded, r.decode_failures.len());
            println!(
                "  experts vs f32 hats (bit for bit): {} records, {} weights, {} differing",
                r.expert_f32_records, r.expert_f32_weights, r.expert_f32_differing
            );
            println!(
                "  experts vs f16 hats (decode narrowed to f16, then bit for bit): {} records, {} weights, {} differing",
                r.expert_f16_records, r.expert_f16_weights, r.expert_f16_differing
            );
            println!(
                "  inkwh records (bytes = file, decode = reconstruction): {} records, {} weights, {} differing, {} byte mismatches",
                r.wh_records, r.wh_weights, r.wh_differing, r.wh_bytes_differ.len()
            );
            for (l, mode) in &r.layer_modes {
                println!("  layer {l}: {mode}");
            }
            println!("  raw tensors: {} in the file, {} compared with their files, {} differ", r.raws, r.raws_compared, r.raws_differ.len());
            println!("  blobs: {}, no trailing byte", r.blobs);
            if inp.sidecar.is_some() {
                println!("  sidecar: {} tensors, {} bytes equal to the native file", r.sidecar_tensors, r.sidecar_bytes);
            }
            if inp.native.is_some() {
                println!(
                    "  native accounting: {} tensors, each in exactly one place: {:?}; {} errors; text-model tensors omitted: {}",
                    r.native_tensors, r.by_class, r.accounting_errors.len(), r.text_omitted
                );
                for e in r.accounting_errors.iter().take(5) {
                    println!("    accounting error: {e}");
                }
            }
            for f in r.decode_failures.iter().take(5) {
                println!("  decode failure: {f}");
            }
            println!("verify: {:.1} s", t0.elapsed().as_secs_f64());
            let fails = r.failures();
            ensure!(fails.is_empty(), "FAIL: {}", fails.join("; "));
            println!("PASS");
            Ok(())
        }
        "flip" => {
            let [src, dst] = pos.as_slice() else { bail!("flip takes IN OUT") };
            let (raw, k) = match (record, raw_index) {
                (Some(k), None) => (false, k),
                (None, Some(k)) => (true, k),
                _ => bail!("flip needs exactly one of --record K and --raw-index K"),
            };
            let off = inkseal::flip_bit(&PathBuf::from(src), &PathBuf::from(dst), raw, k, bit)?;
            let what = if raw { "raw tensor" } else { "record" };
            println!("flipped bit {bit} of {what} {k}'s payload, byte {off} of the file, into {dst}");
            Ok(())
        }
        other => bail!("unknown command {other}"),
    }
}
