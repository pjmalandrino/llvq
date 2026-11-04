// DRAFT, never compiled or run. Written read-only on 2026-11-02 for lead
// "bigmodel" (wave5-deep/bigmodel.md §5 step V2, prereg-bigmodel-2.md),
// revised the same day after review. Not part of the repository.
//
// What `llvq-cuda/src/bin/rotbench.rs` gains for the rot-identity job.
//
// ## The fixture is COMPILED IN, never read at run time
//
// The runtime image holds binaries, `fetch-qtip.sh` and `configs/` only
// (ops/Dockerfile.cuda:131-153, 158, 175): there is no repository tree on the
// card, so `std::fs::read_to_string("llvq-cuda/tests/fixtures/...")` would
// die at the first width, after billing starts. The build stage copies the
// whole uploaded tree (`COPY . .`, :31) and `llvq-*/**` is in UPLOAD_ALLOW
// (ops/run.py:751), so the file exists at compile time. A missing fixture then
// fails the FREE image build, by name, instead of a billed job. Under that
// condition, and only under it, the recipe does not change: rotbench is
// already in both lists (Dockerfile :87 and :147). A runtime COPY of the file
// is rejected: COPY, UPLOAD_ALLOW and the build list would all have to move
// together, the two-lists trap that has bitten four times.
//
//     const FIXTURE: &str = include_str!("../../tests/fixtures/rot-hashes.txt");
//
// (path relative to llvq-cuda/src/bin/rotbench.rs: ../../ is llvq-cuda/.)
//
// Fixture line format, written by the Mac host test (host_rotate_wide.cpp
// through rotation_matches_rust.rs), one per (n, seed):
//
//     <n> <seed hex> <sha256 of the INPUT bytes> <sha256 of rot_apply's OUTPUT>
//
// The input hash covers, in this order and little-endian: the n f16 bits of
// the activation, the sign words, the KMAX*KMAX padded `small` floats, and
// `inv`. Comparing it first separates "the card drew another input" (a drift
// between the host test's draw and this binary's, gate 5 of the prereg) from
// "the kernel computed other bits" (rule I). Both sides hash with the same
// portable `sha256_hex` (moved out of the Linux-only gpu.rs:189).
//
// ## Two handles for every shared-staged kernel
//
// The opt-in is a property of the loaded function and may move its carveout
// (rotbench.rs:114-126). Today's load path poses ONE opt-in per model, for its
// widest rotation (fused_cuda.rs:768-783): the served 4B poses none
// (9,728 * 4 = 38,912 B fits the default; func_dynamic_shared returns `func`
// untouched, gpu.rs:423-447), the 14B poses 69,632 B on every width. So K0 and
// R1 are each obtained from TWO module loads of the same NVRTC text (two
// `Cuda::new(&src)` on the primary context, or one extra `load_module`):
//
//   handle A: no attribute set. Widths <= 12,288 (5,120 included). The served
//             4B's object, the object of the 2026-08-05 run, and what a 32B's
//             narrow widths get once 25,600 leaves the shared path.
//   handle B: CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 69,632. Widths
//             5,120 and 17,408. The served 14B's object and the object of the
//             2026-08-17 run.
//
// R0 requests 0 dynamic shared bytes and R1w gb*m*4 <= 49,152 (default bound):
// one handle each, never an attribute.
//
// ## What runs (prereg-bigmodel-2 §3)
//
//  1. WIDTHS below; per width the arms are those `rot_kernel()` allows.
//  2. Identity: input hash, then output hash, against the fixture; then
//     today's f64 tolerance and norm checks as a second net; rows twins at
//     1, 3, 4 rows, each row byte-equal to the single-row output.
//  3. Timing: ROUNDS rounds; per round, every applicable (kernel, handle) runs
//     WARMUP discarded and LAUNCHES timed chained launches, in an order rotated
//     by one each round. µs = elapsed / LAUNCHES after a sync.
//  4. Report: median and [min-max] per (kernel, handle, width); ratios R1/K0
//     (same handle), R0/K0 (handle A up to 12,288, handle B at 17,408) and
//     R1w/R0, formed ROUND BY ROUND, then their median and range (METHODE §2,
//     rule 7). Never a quotient of two medians.
//  5. Registers and spill of every entry point; refuse local_bytes != 0.

const WIDTHS: [(usize, u64, &str); 12] = [
    (2_560, 0x5, "Qwen3-4B hidden, k=5"),
    (4_096, 0x6, "k=1, m=4,096"),
    (5_120, 0x9, "Qwen3-14B/32B hidden, k=5, m=1,024 (handles A and B)"),
    (6_144, 0xc, "Qwen3.8-27B out_proj/o_proj input, k=3, m=2,048"),
    (8_192, 0xd, "Qwen3-32B o_proj input, k=1"),
    (9_728, 0x7, "Qwen3-4B down, k=19"),
    (12_288, 0x8, "Qwen3-8B down, k=3, m=4,096, on the default bound"),
    (17_408, 0xa, "Qwen3-14B down, k=17, handle B only"),
    (25_600, 0xb, "Qwen3-32B down, k=25: K0 and R1 refused on sm_89"),
    (31_744, 0xe, "synthetic widest odd k = 31"),
    (3_072, 0x4, "Qwen3-0.6B intermediate"),
    (1_024, 0x3, "narrow, k=1"),
];

const ROUNDS: usize = 5;
const LAUNCHES: usize = 200;
const WARMUP: usize = 20;
const OPTIN_14B: u32 = 69_632; // 17,408 * 4: the 14B's object

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Handle {
    A,        // no attribute
    B,        // opt-in at OPTIN_14B
    C(u32),   // opt-in at n * 4, its own module load: only where the card's
              // opt-in allows a width past 17,408 (A100, H200; step V5)
    Only,     // R0, R1w: never an attribute
}

/// The (kernel, handle) arms a width runs on a card with these bounds.
/// K0 and R1 run on A when n*4 fits the default, on B at 5,120 and 17,408,
/// on C(n*4) at any other width past the default that the opt-in still fits;
/// R0 and R1w run everywhere.
fn arms_for(n: usize, default_limit: usize, optin_limit: usize) -> Vec<(&'static str, Handle)> {
    let mut v = Vec::new();
    let bytes = n * 4;
    let on_b = n == 5_120 || n == 17_408;
    if bytes <= default_limit {
        v.push(("rot_apply", Handle::A));
        v.push(("rot_apply_r", Handle::A));
    }
    if on_b {
        v.push(("rot_apply", Handle::B));
        v.push(("rot_apply_r", Handle::B));
    } else if bytes > default_limit && bytes <= optin_limit {
        v.push(("rot_apply", Handle::C(bytes as u32)));
        v.push(("rot_apply_r", Handle::C(bytes as u32)));
    }
    v.push(("rot_apply_dev", Handle::Only));
    v.push(("rot_apply_rw", Handle::Only));
    v
}

/// One fixture line: (n, seed) -> (input sha256, output sha256).
fn parse_fixture(text: &str) -> Result<std::collections::BTreeMap<(usize, u64), (String, String)>, String> {
    let mut m = std::collections::BTreeMap::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() != 4 {
            return Err(format!("fixture line {}: {} fields, expected 4", i + 1, f.len()));
        }
        let n: usize = f[0].parse().map_err(|e| format!("fixture line {}: n: {e}", i + 1))?;
        let seed = u64::from_str_radix(f[1].trim_start_matches("0x"), 16)
            .map_err(|e| format!("fixture line {}: seed: {e}", i + 1))?;
        if m.insert((n, seed), (f[2].to_string(), f[3].to_string())).is_some() {
            return Err(format!("fixture line {}: ({n}, {seed:#x}) twice", i + 1));
        }
    }
    Ok(m)
}

// fn run() sketch (Linux only, as today):
//
//   const FIXTURE: &str = include_str!("../../tests/fixtures/rot-hashes.txt");
//   let fx = parse_fixture(FIXTURE)?;
//   for (n, seed, _) in WIDTHS { fx.get(&(n, seed)).ok_or("no fixture line")?; } // all present before any launch
//   // One BENCH unit: llvq_rot.cuh, rotate.cu (K0) and rotate_wide.cu (R0, R1, R1w),
//   // through load_sources_many, so LLVQ_KERNEL_DIR stays all-or-nothing over the three.
//   // embedded_source() gains a "rotate_wide.cu" arm, which its completeness test pins.
//   // Not the served unit: the served unit's text and printed sha256 never see rotate_wide.cu.
//   let set = llvq_cuda::load_sources_many(&["llvq_rot.cuh", "rotate.cu", "rotate_wide.cu"])?;
//   let src = KernelSource::new(&set.parts.iter().map(String::as_str).collect::<Vec<_>>());
//   let (cuda_a, cuda_b) = (Cuda::new(&src)?, Cuda::new(&src)?);                 // two module loads
//   let k0_b = cuda_b.func_dynamic_shared("rot_apply", OPTIN_14B)?;              // handle B
//   let r1_b = cuda_b.func_dynamic_shared("rot_apply_r", OPTIN_14B)?;
//   // handle A: cuda_a.func(..), no attribute, ever.
//   for (n, seed, what) in WIDTHS {
//       let input = draw(n, seed);                        // today's draw, unchanged
//       if sha256_hex(&input.bytes()) != fx[&(n, seed)].0 { return Err(format!("{what}: INPUT drift")); }
//       for (kernel, handle) in arms_for(n, dev.shared_per_block as usize, dev.shared_per_block_optin as usize) {
//           let out = run_once(kernel, handle, &input)?;
//           let h = sha256_hex(&le_bytes(&out));
//           if h != fx[&(n, seed)].1 { return Err(format!("{what}: {kernel}/{handle:?} sha256 {h} != host")); }
//           check_f64_tolerance_and_norm(&out, n, seed)?;  // today's checks, second net
//           check_rows_equal_single(kernel, handle, &input, &[1, 3, 4])?;
//       }
//       timings.push(interleaved_rounds(&arms_for(n, def, optin), &input)?);
//   }
//   for each entry point: report(name), refuse local_bytes != 0.
//
// The journal prints the handle beside every µs column, and the opt-in value
// of handle B, so no µs figure is read against a run on another object.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_4b_widths_run_on_handle_a_and_the_14b_down_on_handle_b() {
        let (def, l40s) = (49_152, 101_376);
        assert!(arms_for(2_560, def, l40s).contains(&("rot_apply", Handle::A)));
        assert!(arms_for(9_728, def, l40s).contains(&("rot_apply", Handle::A)));
        assert!(!arms_for(9_728, def, l40s).contains(&("rot_apply", Handle::B)));
        assert!(arms_for(12_288, def, l40s).contains(&("rot_apply", Handle::A))); // on the bound
        let a = arms_for(17_408, def, l40s);
        assert!(a.contains(&("rot_apply", Handle::B)) && !a.contains(&("rot_apply", Handle::A)));
        let b = arms_for(5_120, def, l40s);
        assert!(b.contains(&("rot_apply", Handle::A)) && b.contains(&("rot_apply", Handle::B)));
        let w = arms_for(25_600, def, l40s);
        assert!(!w.iter().any(|&(k, _)| k == "rot_apply" || k == "rot_apply_r"));
        assert!(w.contains(&("rot_apply_dev", Handle::Only)) && w.contains(&("rot_apply_rw", Handle::Only)));
    }

    #[test]
    fn on_the_a100_the_32b_width_runs_k0_on_its_own_opt_in() {
        let (def, a100) = (49_152, 166_912);
        let w = arms_for(25_600, def, a100);
        assert!(w.contains(&("rot_apply", Handle::C(102_400))));
        assert!(w.contains(&("rot_apply_dev", Handle::Only)));
        assert!(arms_for(31_744, def, a100).contains(&("rot_apply", Handle::C(126_976))));
    }

    #[test]
    fn a_fixture_with_a_missing_field_or_a_duplicate_is_refused() {
        assert!(parse_fixture("2560 0x5 aa\n").is_err());
        assert!(parse_fixture("2560 0x5 aa bb\n2560 0x5 cc dd\n").is_err());
        let m = parse_fixture("# comment\n2560 0x5 aa bb\n").unwrap();
        assert_eq!(m[&(2560, 5)], ("aa".to_string(), "bb".to_string()));
    }
    // Mutants this file must kill: `<` for `<=` in shared_ok_default (12,288
    // leaves handle A); 17,408 put on A (the opt-in object lost); the input
    // hash skipped (a draw drift then reads as a kernel mismatch).
}
