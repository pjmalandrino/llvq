//! One copy of the int4 bytes: the eager readers on 4E v9's tiled words, against
//! the served readers. One file because the served side is built mutated here.

use super::*;

// ---- One copy of the int4 bytes: the eager readers on 4E v9's tiled words ---

/// The served side built from a mutated copy of the sources, each edit
/// applied exactly once (`mutant_many`'s rule, for the served kernels).
fn mutant_served(tag: &str, edits: &[(&str, &str, &str)]) -> PathBuf {
    let root = std::env::temp_dir().join(format!("llvq_pers_mutant_served_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    for dir in ["llvq-cuda/kernels", "llvq-cuda/tests", "llvq-llm/kernels"] {
        let (src, dst) = (repo().join(dir), root.join(dir));
        std::fs::create_dir_all(&dst).unwrap();
        for e in std::fs::read_dir(&src).unwrap() {
            let p = e.unwrap().path();
            let ext = p.extension().and_then(|x| x.to_str()).unwrap_or("");
            if ["cu", "cuh", "h", "cpp", "inc"].contains(&ext) {
                std::fs::copy(&p, dst.join(p.file_name().unwrap())).unwrap();
            }
        }
    }
    for &(file, from, to) in edits {
        let path = root.join(file);
        let text = std::fs::read_to_string(&path).unwrap();
        let n = text.matches(from).count();
        assert_eq!(n, 1, "mutant {tag}: {n} sites for {from:?} in {file}, expected exactly one");
        std::fs::write(&path, text.replacen(from, to, 1)).unwrap();
    }
    build_side(&root, 0, "off", &format!("served_{tag}"))
}

/// Random served-order words for `d_out` rows of `d_in` columns.
fn q4_words(d_out: u32, d_in: u32, seed: u64) -> Vec<u32> {
    let mut r = llvq_core::SplitMix64::new(seed);
    (0..d_out as usize * d_in as usize / 8).map(|_| r.next() as u32).collect()
}

/// The eager int4 entry point `entry` (0 tv_q4_hh, 1 tv_q4_h, 2 tv_q4_hht,
/// 3 tv_q4_ht) on `words`: y then each row's f32, as bytes.
fn q4_cmd(d_out: u32, d_in: u32, entry: u32, seed: u64, words: &[u32]) -> Vec<u8> {
    Cmd::new(b'q').u32(d_out).u32(d_in).u32(entry).u64(seed).u32s(words).0
}

/// The widths of every int4 record of the three sealed files (v, o and down
/// d_in at 4B, 8B and 14B), and two that are not: one full tile, and a width
/// below one tile, all tail.
const Q4_WIDTHS: [u32; 8] = [2560, 4096, 5120, 9728, 12288, 17408, 1024, 512];

/// `pers::q4_tile_index` (the loader's permutation, on the host) is
/// `pers_q4_tile_index` (the device's, which 4E v9's relayout and reader use)
/// on every word of every width.
#[test]
fn the_host_tile_index_is_the_device_one() {
    let widths: Vec<u32> = Q4_WIDTHS.iter().map(|d| d / 8).collect();
    let got = u32s(&run(persistent(), &Cmd::new(b'i').u32(widths.len() as u32).u32s(&widths).0));
    let want: Vec<u32> = widths.iter().flat_map(|&nw| (0..nw).map(move |wi| pers::q4_tile_index(wi, nw))).collect();
    assert_eq!(got.len(), want.len());
    assert!(got == want, "the host permutation differs from pers_q4_tile_index");
}

/// The permutation keeps every word of a row, in that row, and moves no word
/// past the last full tile.
#[test]
fn the_tile_permutation_keeps_each_row_s_words() {
    for &d_in in &Q4_WIDTHS {
        let (d_out, nw) = (8usize, d_in as usize / 8);
        let words: Vec<u32> = (0..(d_out * nw) as u32).collect();
        let t = pers::q4_tile_words(&words, d_out, nw).unwrap();
        for r in 0..d_out {
            let mut row = t[r * nw..(r + 1) * nw].to_vec();
            let tail = nw / 128 * 128;
            assert!(row[tail..] == words[r * nw + tail..(r + 1) * nw], "d_in {d_in}: the tail moved");
            row.sort_unstable();
            assert!(row == words[r * nw..(r + 1) * nw], "d_in {d_in}: row {r} lost or gained a word");
        }
    }
    assert!(pers::q4_tile_words(&[0; 6], 1, 6).is_err(), "a row of 6 words is not whole 16-byte loads");
    assert!(pers::q4_tile_words(&[0; 8], 1, 4).is_err(), "8 words for one row of 4");
}

/// The single copy's claim: `tv_q4_hht` / `tv_q4_ht` on the words the loader
/// tiles return `tv_q4_hh` / `tv_q4_h`'s bytes on the file's words, every
/// row, f32 before the narrowing included, at every sealed width.
#[test]
fn the_tiled_eager_readers_are_the_served_readers() {
    for (i, &d_in) in Q4_WIDTHS.iter().enumerate() {
        let d_out = 24u32;
        let seed = 0x7a10 + i as u64;
        let words = q4_words(d_out, d_in, seed);
        let tiled = pers::q4_tile_words(&words, d_out as usize, d_in as usize / 8).unwrap();
        for (served_entry, tiled_entry) in [(0u32, 2u32), (1, 3)] {
            let want = run(served(), &q4_cmd(d_out, d_in, served_entry, seed, &words));
            let got = run(served(), &q4_cmd(d_out, d_in, tiled_entry, seed, &tiled));
            assert!(got == want, "d_in {d_in}: entry {tiled_entry} on tiled words against entry {served_entry}");
            if d_in >= 1024 {
                // the file's words through a tiled reader: wrong numbers, so the
                // comparison above is not vacuous
                let wrong = run(served(), &q4_cmd(d_out, d_in, tiled_entry, seed, &words));
                assert!(wrong != want, "d_in {d_in}: the tiled reader reads the file's order too");
            }
        }
    }
}

/// Each guard of `tv_q4_tile.cu` has a mutant, killed by a byte.
#[test]
fn the_tiled_eager_reader_mutants_are_killed() {
    let cases: Vec<(Vec<u8>, Vec<u8>)> = [2560u32, 9728, 4096]
        .iter()
        .flat_map(|&d_in| {
            let words = q4_words(16, d_in, 0x7a20 + d_in as u64);
            let tiled = pers::q4_tile_words(&words, 16, d_in as usize / 8).unwrap();
            [2u32, 3].map(|e| {
                let cmd = q4_cmd(16, d_in, e, 0x7a20 + d_in as u64, &tiled);
                let want = run(served(), &cmd);
                (cmd, want)
            })
        })
        .collect();
    let f = "llvq-llm/kernels/tv_q4_tile.cu";
    for (tag, from, to) in [
        ("tile_lane", "wrow + (t << 7) + 4u * lane);", "wrow + (t << 7) + 4u * (lane ^ 1u));"),
        ("tile_base", "wrow + (t << 7) + 4u * lane);", "wrow + 4u * lane);"),
        ("tile_word_order", "const u32 w[4] = {P.x, P.y, P.z, P.w};", "const u32 w[4] = {P.y, P.x, P.z, P.w};"),
        ("tile_column", "u32 c = ((t << 7) + 32u * s + lane) << 3;", "u32 c = ((t << 7) + 8u * s + lane) << 3;"),
        ("tile_group", "u32 g  = s0 + (c >> 7);\n            u32 sb", "u32 g  = s0 + (c >> 8);\n            u32 sb"),
        ("tile_count", "u32 nt = nwords >> 7;", "u32 nt = nwords >> 8;"),
        ("tile_tail_start", "for (u32 wi = (nt << 7) + lane; wi < nwords; wi += 32u) {", "for (u32 wi = (nt << 7) + lane + 32u; wi < nwords; wi += 32u) {"),
        ("tile_row_stride", "const u32* wrow = wq + (unsigned long long)row * nwords;", "const u32* wrow = wq + (unsigned long long)row * (nwords - 4u);"),
        ("tile_hh_widen", "for (u32 i = threadIdx.x; i < d_in; i += blockDim.x) xs[i] = h2f(x[i]);\n    __syncthreads();\n    q4t_staged_dot", "for (u32 i = threadIdx.x; i < d_in; i += blockDim.x) xs[i] = h2f(x[i]) * 0.5f;\n    __syncthreads();\n    q4t_staged_dot"),
    ] {
        let b = mutant_served(tag, &[(f, from, to)]);
        let killed = cases.iter().any(|(c, w)| run_mutant(&b, c).is_none_or(|o| &o != w));
        assert!(killed, "tiled reader mutant {tag} survived");
    }
}
