//! The kernel texts the bench compiles: the embedded copies and their `LLVQ_KERNEL_DIR` / `LLVQ_QTIP_DIR`
//! loaders. One file because they share one contract, embedded text unless a directory is named.

/// The two Planes14 sources are not embedded in `lib.rs` — new files
/// only, that file belongs to another lot — so the bin carries its own
/// copies and honours `LLVQ_KERNEL_DIR` with the same contract as
/// `load_sources_many`: without the variable, the embedded text; with it,
/// the files from the directory, disclosed loudly by the caller.
const PLANES_CUH_EMBED: &str = include_str!("../../../../kernels/llvq_planes.cuh");
const PLANES_CU_EMBED: &str = include_str!("../../../../kernels/planes.cu");
const PLANES12_CUH_EMBED: &str = include_str!("../../../../kernels/llvq_planes12.cuh");
const PLANES12_CU_EMBED: &str = include_str!("../../../../kernels/planes12.cu");
const GOLAY_CUH_EMBED: &str = include_str!("../../../../kernels/llvq_golay.cuh");
const GOLAY_CU_EMBED: &str = include_str!("../../../../kernels/golay70.cu");
/// The segmented Planes14 kernel — item A4. One file, no header of its
/// own: it reuses `llvq_planes.cuh`'s `planes_dot`. It lives apart from
/// `planes.cu` on purpose — `llvq-llm/src/fused_cuda.rs` concatenates
/// `llvq_planes.cuh` + `planes.cu` + `tv_planes_h.cu` as the *shipped*
/// inference kernel, and appending to `planes.cu` would change that
/// string's bytes and its sha256 for a bench's convenience.
const PLANES_SEG_CU_EMBED: &str = include_str!("../../../../kernels/planes_seg.cu");
/// The competitor arm. It depends on `h2f`/`f2h`, hence on `matvec.cu`,
/// so it is concatenated AFTER it, and last, so that adding a competitor
/// never reorders the fragments of the in-house arms.
const AWQ_CU_EMBED: &str = include_str!("../../../../kernels/awq_gemv.cu");
/// The witness arm: the FROZEN copy of the published Golay70 decoder
/// (v1), symbols renamed, buffers shared with v2. Concatenated after AWQ,
/// since the arm added last reorders no existing fragment.
const GOLAY_V1_CU_EMBED: &str = include_str!("../../../../kernels/golay70_v1.cu");

pub(super) fn load_planes_sources() -> Result<(String, String, Option<String>), String> {
    match std::env::var("LLVQ_KERNEL_DIR") {
        Err(_) => Ok((PLANES_CUH_EMBED.to_string(), PLANES_CU_EMBED.to_string(), None)),
        Ok(dir) => {
            let rd = |n: &str| {
                std::fs::read_to_string(std::path::Path::new(&dir).join(n))
                    .map_err(|e| format!("LLVQ_KERNEL_DIR={dir}: {n}: {e}"))
            };
            let cuh = rd("llvq_planes.cuh")?;
            let cu = rd("planes.cu")?;
            Ok((cuh, cu, Some(dir)))
        }
    }
}

/// Same contract for the two Planes12x sources — the M2 arm's kernel.
pub(super) fn load_planes12_sources() -> Result<(String, String, Option<String>), String> {
    match std::env::var("LLVQ_KERNEL_DIR") {
        Err(_) => Ok((
            PLANES12_CUH_EMBED.to_string(),
            PLANES12_CU_EMBED.to_string(),
            None,
        )),
        Ok(dir) => {
            let rd = |n: &str| {
                std::fs::read_to_string(std::path::Path::new(&dir).join(n))
                    .map_err(|e| format!("LLVQ_KERNEL_DIR={dir}: {n}: {e}"))
            };
            let cuh = rd("llvq_planes12.cuh")?;
            let cu = rd("planes12.cu")?;
            Ok((cuh, cu, Some(dir)))
        }
    }
}

/// Same contract for the two Golay70 sources — the E2 arm's kernel.
pub(super) fn load_golay_sources() -> Result<(String, String, Option<String>), String> {
    match std::env::var("LLVQ_KERNEL_DIR") {
        Err(_) => Ok((GOLAY_CUH_EMBED.to_string(), GOLAY_CU_EMBED.to_string(), None)),
        Ok(dir) => {
            let rd = |n: &str| {
                std::fs::read_to_string(std::path::Path::new(&dir).join(n))
                    .map_err(|e| format!("LLVQ_KERNEL_DIR={dir}: {n}: {e}"))
            };
            let cuh = rd("llvq_golay.cuh")?;
            let cu = rd("golay70.cu")?;
            Ok((cuh, cu, Some(dir)))
        }
    }
}

/// Same contract for the one segmented source — the A4 arm's kernel.
/// Same contract as the others: without the variable, the embedded text;
/// with it, the file from the directory, announced loudly. It is also the
/// path by which a kernel we cannot vendor, QTIP being GPL-3, would enter
/// without being committed.
pub(super) fn load_awq_source() -> Result<(String, Option<String>), String> {
    match std::env::var("LLVQ_KERNEL_DIR") {
        Err(_) => Ok((AWQ_CU_EMBED.to_string(), None)),
        Ok(dir) => {
            let cu =
                std::fs::read_to_string(std::path::Path::new(&dir).join("awq_gemv.cu"))
                    .map_err(|e| format!("LLVQ_KERNEL_DIR={dir}: awq_gemv.cu: {e}"))?;
            Ok((cu, Some(dir)))
        }
    }
}

/// Our QTIP shims — committed, so always available, and version-locked to
/// the binary that launches them.
const QTIP_GLUE_EMBED: &str = include_str!("../../../../kernels/qtip_glue.cu");
/// The typedefs the fetched kernel needs and NVRTC does not carry. Ours,
/// and concatenated **before** the fetched half.
const QTIP_PRELUDE_EMBED: &str = include_str!("../../../../kernels/qtip_prelude.cuh");

/// The QTIP device half, and the one loader here that reads a **different**
/// variable from every other.
///
/// 🚨 `LLVQ_QTIP_DIR`, not `LLVQ_KERNEL_DIR`, and the distinction is not
/// cosmetic. `LLVQ_KERNEL_DIR` means *override every kernel source from
/// this directory*: every other loader in this file and in
/// `load_sources_many` reads ITS OWN files from it and **fails hard** when
/// one is missing. Pointing it at the QTIP fetch output would therefore
/// break the whole bench on `matvec.cu: No such file or directory`. QTIP is
/// not an override of anything — it is an ADDITION — so it gets a variable
/// of its own, and the two compose instead of colliding.
///
/// Only the device half is read from disk: the kernel is GPL v3 and this
/// workspace is MIT OR Apache-2.0, so it is fetched at job time by
/// `ops/fetch-qtip.sh` and never committed. The shims above are ours.
/// `Ok(None)` means "no QTIP here", the normal state on any machine that
/// has not run the script; the arm is then simply not dispatchable, which
/// `arms::HAS_KERNEL` already says. What must never happen is a *silent*
/// fallback to some other text, which is why there is nothing to fall
/// back to.
pub(super) fn load_qtip_sources() -> Result<Option<(String, String)>, String> {
    let Ok(dir) = std::env::var("LLVQ_QTIP_DIR") else {
        return Ok(None);
    };
    let cuh = std::fs::read_to_string(std::path::Path::new(&dir).join("qtip_device.cuh"))
        .map_err(|e| {
            format!(
                "LLVQ_QTIP_DIR={dir}: qtip_device.cuh: {e}, run ops/fetch-qtip.sh \
                 and point the variable at its output"
            )
        })?;
    // Prelude first — it declares the types the fetched half names.
    Ok(Some((format!("{QTIP_PRELUDE_EMBED}\n{cuh}"), QTIP_GLUE_EMBED.to_string())))
}

pub(super) fn load_planes_seg_source() -> Result<(String, Option<String>), String> {
    match std::env::var("LLVQ_KERNEL_DIR") {
        Err(_) => Ok((PLANES_SEG_CU_EMBED.to_string(), None)),
        Ok(dir) => {
            let cu = std::fs::read_to_string(
                std::path::Path::new(&dir).join("planes_seg.cu"),
            )
            .map_err(|e| format!("LLVQ_KERNEL_DIR={dir}: planes_seg.cu: {e}"))?;
            Ok((cu, Some(dir)))
        }
    }
}

/// Same contract for the frozen v1 copy. Overriding the WITNESS is
/// allowed, but announced as loudly as the rest: a witness silently
/// replaced attests to nothing.
pub(super) fn load_golay_v1_source() -> Result<(String, Option<String>), String> {
    match std::env::var("LLVQ_KERNEL_DIR") {
        Err(_) => Ok((GOLAY_V1_CU_EMBED.to_string(), None)),
        Ok(dir) => {
            let cu = std::fs::read_to_string(
                std::path::Path::new(&dir).join("golay70_v1.cu"),
            )
            .map_err(|e| format!("LLVQ_KERNEL_DIR={dir}: golay70_v1.cu: {e}"))?;
            Ok((cu, Some(dir)))
        }
    }
}
