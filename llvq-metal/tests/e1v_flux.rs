//! **The `e1v-flux` arm decodes the same thing as `marche-bloc`, and that has
//! to be a fact about the text rather than an intention.**
//!
//! P1c runs the two arms in one run so that their gap is the price of E1v's
//! addressing — base word, fixed-stride header, warp-scan over 32 widths, a
//! field read at an arbitrary bit offset — and of nothing else. That reading
//! holds only while the two decode *identically*. Nothing in a Metal build
//! notices when they stop: both files compile, both produce plausible dot
//! products, and V0 would keep passing on both, each against its own etalon.
//! The gap would just silently stop measuring addressing.
//!
//! So the shared text is compared byte for byte. `binomial_block.metal` is the
//! arm P1b measured and its document is **sealed**: this test is written so
//! that file never has to be touched — the two regions are delimited by anchors
//! that already exist in it.
//!
//! | region | from | to |
//! |---|---|---|
//! | helpers | `struct Cursor …` | the first banner comment after it |
//! | decode | `uint w    = r.w;` | `out[gid] = dot * gscale[gain];` |
//!
//! The E1v-only helpers therefore live **above** `struct Cursor` in the new
//! file, and its addressing preamble between the two regions. That placement is
//! not cosmetic: it is what makes the two regions contiguous and comparable.

use llvq_core::Golay;
use llvq_metal::p1host::{block_records, e1v_payload_bits, E1V_GROUP, E1V_ORIGIN_ID};

/// The origin id as a table index. The format states it once, as a `u64`
/// (`llvq_artifact::e1v`), because that is what a header field holds; a table
/// is indexed by `usize`, and the conversion is named here rather than sprinkled.
const ORIGIN_IDX: usize = E1V_ORIGIN_ID as usize;
use llvq_search::fastdec::FastDecoder;

const BLOCK: &str = include_str!("../shaders/binomial_block.metal");
const FLUX: &str = include_str!("../shaders/e1v_flux.metal");

/// Text from `start` up to `end`. `end` is excluded when it is a banner and
/// included when it is the region's last line — the two regions genuinely end
/// differently, and pretending otherwise would need an anchor that does not
/// exist in the sealed file.
fn region(src: &str, name: &str, start: &str, end: &str, inclusive: bool) -> String {
    let s = src
        .find(start)
        .unwrap_or_else(|| panic!("{name}: the start anchor `{start}` is gone"));
    let rest = &src[s..];
    let e = rest
        .find(end)
        .unwrap_or_else(|| panic!("{name}: the end anchor is gone"));
    let e = if inclusive { e + end.len() } else { e };
    rest[..e].to_string()
}

fn helpers(src: &str, name: &str) -> String {
    region(src, name, "struct Cursor { ulong lo; uint hi; };", "\n// ---", false)
}

fn decode_body(src: &str, name: &str) -> String {
    region(
        src,
        name,
        "    uint w    = r.w;",
        "    out[gid] = dot * gscale[gain];",
        true,
    )
}

/// 🚨 **The load-bearing one.** `Cursor`, `take`, the `BlockRec` mirror,
/// `walk_kind` and `walk_all` are the same text in both files.
///
/// A drift here would not be a compile error and would not be a V0 failure. It
/// would be two decoders being priced against each other as if they were one.
#[test]
fn the_two_arms_share_their_helpers_byte_for_byte() {
    let a = helpers(BLOCK, "binomial_block.metal");
    let b = helpers(FLUX, "e1v_flux.metal");
    assert!(
        a.contains("struct BlockRec") && a.contains("walk_all") && a.len() > 1500,
        "the extraction does not return the expected region ({} bytes), the anchor moved",
        a.len()
    );
    assert_eq!(
        a, b,
        "the two arms no longer share their helpers: the gap between them stops being \
         the price of the addressing"
    );
}

/// The decode itself — the codeword lookup, the two walks, the three sign
/// rules, the parity repair and the deposit — is the same text in both files.
#[test]
fn the_two_arms_share_their_decode_byte_for_byte() {
    let a = decode_body(BLOCK, "binomial_block.metal");
    let b = decode_body(FLUX, "e1v_flux.metal");
    assert!(
        a.contains("walk_all(cur") && a.contains("par ^=") && a.len() > 1500,
        "the extraction does not return the decode body ({} bytes)",
        a.len()
    );
    assert_eq!(
        a, b,
        "the decode body has diverged between `marche-bloc` and `e1v-flux`"
    );
}

/// The E1v arm's addressing must be *outside* the shared regions, or the
/// comparison above would be vacuous — two identical files also share
/// everything.
#[test]
fn the_addressing_is_the_only_difference() {
    for needle in [
        "simd_prefix_exclusive_sum",
        "window96",
        "read_small",
        "bases[gid / E1V_GROUP]",
    ] {
        assert!(FLUX.contains(needle), "`{needle}` is gone from the E1v arm");
        assert!(
            !BLOCK.contains(needle),
            "`{needle}` appeared in `binomial_block.metal`, whose document is sealed"
        );
    }
    // And the shared regions must not have swallowed it.
    let h = helpers(FLUX, "e1v_flux.metal");
    let d = decode_body(FLUX, "e1v_flux.metal");
    for needle in ["simd_prefix_exclusive_sum", "window96", "read_small"] {
        assert!(!h.contains(needle) && !d.contains(needle), "`{needle}` is in a shared region");
    }
}

/// The three constants the shader restates from `llvq_artifact::e1v`, since MSL
/// cannot import them.
///
/// 🕳️ `p1host` used to restate two of them as well, and `bin/rankbench` pinned
/// its copies against the writer's at run time. Both are gone: the table moved
/// to `llvq_artifact::blockrec`, which reads the format's own constants, so
/// there is one definition and nothing to pin it to. What remains genuinely
/// duplicated is the **shader's** copy — MSL imports nothing — and that is what
/// this checks.
#[test]
fn the_shader_restates_the_streams_constants() {
    for (decl, want) in [
        ("constant uint E1V_GROUP       = ", E1V_GROUP),
        ("constant uint E1V_HEADER_BITS = ", 10),
        ("constant uint E1V_ORIGIN_ID   = ", ORIGIN_IDX),
    ] {
        let at = FLUX
            .find(decl)
            .unwrap_or_else(|| panic!("the declaration `{decl}` is gone from the shader"));
        let tail = &FLUX[at + decl.len()..];
        let lit: String = tail.chars().take_while(char::is_ascii_digit).collect();
        assert_eq!(
            lit.parse::<usize>().expect("a decimal literal"),
            want,
            "the shader and the host disagree on `{decl}`"
        );
    }
    // The header is 10 bits in the search crate too, which is where the writer
    // takes it from.
    assert_eq!(llvq_search::cns::HEADER_BITS, 10);
}

/// The payload-width table, built on the real codebook.
///
/// [`e1v_payload_bits`] asserts its own two identities — the scan's width
/// against the cursor's, and the 96-bit window's capacity — but only the caller
/// that has an archive ever runs it. This runs it on every one of the 383
/// classes without an archive, so the identities are checked on every commit
/// rather than on every bench.
#[test]
#[cfg_attr(debug_assertions, ignore = "builds the whole class table, run in release")]
fn the_payload_widths_are_what_the_cursor_consumes() {
    let fd = FastDecoder::new();
    let golay = Golay::new();
    let recs = block_records(&fd, &golay);
    let pay = e1v_payload_bits(&fd, &golay, &recs);

    assert_eq!(pay[ORIGIN_IDX], 0, "the origin spends only its header");
    for (ci, &bits) in pay.iter().enumerate().take(fd.n_classes()) {
        assert!(bits > 0, "class {ci}: an empty payload");
    }
    // The pre-registration's §1 divulges 56 bits on 8 fields, class 323. That
    // is a **record**, header included; what the warp-scan sums is a
    // **payload**, which is 10 bits less. Both are checked, and separately —
    // the whole reason this test exists is that the two are easy to conflate,
    // and a scan that summed records would put ten bits of drift into every
    // lane of every group.
    let (at, max) = pay
        .iter()
        .enumerate()
        .max_by_key(|(_, &b)| b)
        .expect("the table is not empty");
    assert_eq!(at, 323, "the widest class is no longer the one in §1");
    assert_eq!(
        u64::from(*max) + llvq_search::cns::HEADER_BITS,
        56,
        "the widest record in §1 is 56 bits, header included; the payload that the \
         prefix sum adds up is {max}"
    );
    // Ids no class takes, and which no block ever reads.
    for (id, &bits) in pay
        .iter()
        .enumerate()
        .take(ORIGIN_IDX)
        .skip(fd.n_classes())
    {
        assert_eq!(bits, 0, "id {id} is no class");
    }
}
