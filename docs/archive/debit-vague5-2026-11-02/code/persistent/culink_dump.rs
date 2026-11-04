// DRAFT, never compiled or run. Written read-only on 2026-11-02 for the
// persistent-kernel lead (wave5-deep/persistent.md, section 5, step L1), after
// the adversarial review pointed out that a devel-stage `cuobjdump` cannot
// show the card's SASS: the repository hands NVRTC's PTX to the driver
// (gpu.rs:270-280, `load_module`), and the driver's own ptxas (580.178.04 on
// the L40S, a3-gpu.txt) makes the last contraction decisions on bare
// `mul.f32` + `add.f32` pairs.
//
// What it does: on the card, during a bench job, JIT the same PTX through the
// driver's linker and write the resulting cubin to the bucket. The cubin is
// then disassembled off the card, at $0, with `nvdisasm` or `cuobjdump -sass`
// from a CUDA devel image at least as new as the driver (13.0 for a 580
// driver). `cuLinkCreate_v2`, `cuLinkAddData_v2`, `cuLinkComplete` and
// `cuLinkDestroy` are in cudarc 0.19.8's sys bindings (sys/mod.rs:12030-12094).
//
// The residual risk, stated: `cuModuleLoadData` and `cuLinkComplete` both run
// the driver JIT with default options, which is expected to give the same
// SASS. The dump prints each function's register count from the loaded module
// (cuFuncGetAttribute NUM_REGS) beside the one `nvdisasm` reports off the
// card; a mismatch voids the reading, not the job.
//
// Where it goes: llvq-cuda/src/gpu.rs (cfg linux), behind LLVQ_DUMP_CUBIN=<dir>,
// a measurement flag of planesbench only; `unsafe` at the hardware boundary,
// as CLAUDE.md allows.

use cudarc::driver::sys;

/// JIT `ptx` with the driver's linker and return the cubin bytes.
pub fn driver_cubin(ptx: &str, name: &str) -> Result<Vec<u8>, String> {
    let ptx_c = std::ffi::CString::new(ptx).map_err(|e| e.to_string())?;
    let name_c = std::ffi::CString::new(name).map_err(|e| e.to_string())?;
    let mut state: sys::CUlinkState = std::ptr::null_mut();
    // SAFETY: plain driver calls on a context the caller made current; every
    // pointer outlives the call; the state is destroyed on every path.
    unsafe {
        let r = sys::cuLinkCreate_v2(0, std::ptr::null_mut(), std::ptr::null_mut(), &mut state);
        if r != sys::CUresult::CUDA_SUCCESS {
            return Err(format!("cuLinkCreate: {r:?}"));
        }
        let r = sys::cuLinkAddData_v2(
            state,
            sys::CUjitInputType::CU_JIT_INPUT_PTX,
            ptx_c.as_ptr() as *mut std::ffi::c_void,
            ptx_c.as_bytes_with_nul().len(),
            name_c.as_ptr(),
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
        if r != sys::CUresult::CUDA_SUCCESS {
            sys::cuLinkDestroy(state);
            return Err(format!("cuLinkAddData: {r:?}"));
        }
        let mut cubin: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut size: usize = 0;
        let r = sys::cuLinkComplete(state, &mut cubin, &mut size);
        if r != sys::CUresult::CUDA_SUCCESS {
            sys::cuLinkDestroy(state);
            return Err(format!("cuLinkComplete: {r:?}"));
        }
        // The image belongs to the link state: copy it out before destroying it.
        let bytes = std::slice::from_raw_parts(cubin as *const u8, size).to_vec();
        sys::cuLinkDestroy(state);
        Ok(bytes)
    }
}

// In planesbench, after each module is loaded (bench unit, persistent module,
// micro module, and the served unit assembled from its 11 parts with its
// digest printed and compared with 6193094310bc18ac... or with the served
// door's line of the same job):
//
//   if let Ok(dir) = std::env::var("LLVQ_DUMP_CUBIN") {
//       let bytes = driver_cubin(&ptx_text, "served")?;
//       std::fs::write(format!("{dir}/served.cubin"), &bytes).map_err(|e| e.to_string())?;
//       println!("cubin served: {} bytes, sha256 {}", bytes.len(), sha256_hex(&bytes));
//   }
//
// Off the card, the reading the lead needs (persistent.md 3.7, 3.8):
//   * tv_tetra48_h: LDS widths at the 24 x reads; FFMA against FMUL + FADD at
//     `acc += dot`, the tail loop and `acc*rscale + tv`;
//   * tv_f1r_v3g and tv_tetra48_seg: the same three sites, side by side;
//   * the persistent arms (step L2): the same, so the explicit intrinsics of
//     the persistent kernel can be made to match the served SASS if they do not.
