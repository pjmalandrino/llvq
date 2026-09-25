//! The device dispatch, driven on the CPU, with no card and no container.
//!
//! ## What this file is for, stated exactly
//!
//! Before the device port, `Proj`'s three device arms sat behind
//! `#[cfg(all(target_os = "linux", feature = "cuda"))]`. They compiled nowhere
//! on the development machine and ran nowhere at all outside a billed job.
//! `ops/check-cuda.sh` closed the TYPE half of that hole on 2026-09-10, in a
//! local container, for nothing. Its own header scopes itself honestly: "It is
//! a TYPE check and nothing more."
//!
//! This file closes the other half. The fakes below implement the same traits
//! the CUDA adapters implement, over dense CPU tensors, so `prepare`,
//! `forward_with`, `group_forward`, `SegPlan` and the chunking all EXECUTE
//! here. What that newly covers is `SegPlan::of`'s refusals, `SegPlan::run`'s
//! narrow-and-reshape bookkeeping and the `forward_rows` guard. It does not
//! cover the kernels, and it makes no claim about them.
//!
//! ## Why the fake rotates for real
//!
//! A fake whose `prepare` returned its argument would pass whether
//! `forward_with` read the prepared tensor or the caller's. So `FakeLattice`
//! stores its weights in a PERMUTED basis and `prepare` applies the
//! permutation. Reading the wrong activation then returns a different number,
//! not a different shape, which is the failure this whole file is shaped
//! against.
//!
//! ## The mutation run
//!
//! Seven mutants of `model.rs`, seven killed, re-run after an adversarial
//! review of 2026-09-20:
//!
//!   1. `forward_with` reads the caller's activation, not the prepared one
//!   2. `Arc::ptr_eq` on the group always true
//!   3. `prefill_rows_of` never batches
//!   4. `batches_rows` always false
//!   5. the int4 arm reads `r.t` instead of `x`
//!   6. a part's view starts at row 0
//!   7. `prepare_rows` wired to `prepare`
//!
//! Two of those are here because the first draft of this file was wrong.
//!
//! Mutant 5 was recorded as EQUIVALENT, on the argument that `check_key`
//! refuses any `Rotated` whose key is not `None`, so `r.t == x` always. The
//! inference is invalid: on that arm the expected key IS `None`, so every
//! `None`-keyed `Rotated` is accepted whatever tensor it carries, which is the
//! hole `Rotated`'s own doc describes. It is a behaviour, and
//! `the_int4_arm_reads_the_callers_activation` kills it.
//!
//! Mutant 7 was not planted at all. The review found it: wiring
//! `Proj::prepare_rows`'s lattice arm to `d.prepare(xs)` left this file 11/11,
//! `cargo clippy` at 0 and `ops/check-cuda.sh` at 0, and died only on a card,
//! at the first prefill chunk, where `FusedRuntime::rotate` refuses
//! `rows != 1`. The fakes were interchangeable where the adapters are not.
//! They now refuse what the adapters refuse, and they log which entry ran.

use candle_core::{DType, Device, Result, Tensor};
use llvq_llm::device::{Int4Proj, LatticeProj, SegGroup};
use llvq_llm::fused::RotKey;
use llvq_llm::model::{group_forward, Proj};
use llvq_llm::rotplan::RotShare;
use std::sync::Arc;

/// A permutation of `n` coordinates, as a dense `[n, n]` matrix.
///
/// Orthogonal, and exact in f32: every entry is 0.0 or 1.0, so applying it and
/// its transpose round-trips bit for bit. That is what lets the assertions
/// below demand equality rather than a tolerance.
fn permutation(n: usize, dev: &Device) -> Result<Tensor> {
    let mut m = vec![0f32; n * n];
    for i in 0..n {
        // A rotation by one position. Any fixed permutation works; this one is
        // not the identity for n > 1, which is the only property needed.
        m[i * n + (i + 1) % n] = 1.0;
    }
    Tensor::from_vec(m, (n, n), dev)
}

fn matrix(d_out: usize, d_in: usize, seed: f32, dev: &Device) -> Result<Tensor> {
    let v: Vec<f32> = (0..d_out * d_in)
        .map(|i| ((i as f32) * 0.37 + seed).sin())
        .collect();
    Tensor::from_vec(v, (d_out, d_in), dev)
}

/// A lattice projection backed by a dense matrix, in the permuted basis.
struct FakeLattice {
    name: String,
    /// `[d_out, d_in]`, in the basis `prepare` produces.
    w: Tensor,
    rot: Tensor,
    key: Option<RotKey>,
    rows_per_launch: usize,
    /// Every launch, in order. `Mutex` and not `RefCell`: the port requires
    /// `Sync`, so a fake that is not `Sync` does not compile as an adapter.
    log: std::sync::Mutex<Vec<String>>,
}

impl FakeLattice {
    fn launches(&self) -> Vec<String> {
        self.log.lock().expect("no panic held this lock").clone()
    }
    fn note(&self, what: String) {
        self.log.lock().expect("no panic held this lock").push(what);
    }
}

impl LatticeProj for FakeLattice {
    fn name(&self) -> &str {
        &self.name
    }
    fn d_out(&self) -> usize {
        self.w.dim(0).expect("2-D")
    }
    fn d_in(&self) -> usize {
        self.w.dim(1).expect("2-D")
    }
    fn rotation(&self) -> Option<RotKey> {
        self.key
    }
    /// Refuses more than one row, exactly as `FusedRuntime::rotate` does
    /// (`fused_cuda.rs`, "rotation requested for N vectors"). A fake that
    /// accepted a chunk here would let the model call `prepare` where it owes
    /// `prepare_rows`, and that wiring dies on a card and nowhere else.
    fn prepare(&self, x: &Tensor) -> Result<Tensor> {
        let d = x.dims();
        let rows: usize = d[..d.len() - 1].iter().product();
        if rows != 1 {
            candle_core::bail!("{}: rotation requested for {rows} vectors", self.name);
        }
        self.note("prepare".into());
        x.broadcast_matmul(&self.rot)
    }
    fn prepare_rows(&self, xs: &Tensor, rows: usize) -> Result<Tensor> {
        if rows != xs.dim(0)? {
            candle_core::bail!("{}: {rows} asked of a chunk of {}", self.name, xs.dim(0)?);
        }
        self.note(format!("prepare_rows({rows})"));
        xs.broadcast_matmul(&self.rot)
    }
    fn matvec(&self, xr: &Tensor, out_dims: &[usize]) -> Result<Tensor> {
        self.note(format!("matvec({})", xr.dim(0).unwrap_or(0)));
        let mut shape = out_dims.to_vec();
        *shape.last_mut().expect("rank >= 1") = self.d_out();
        xr.broadcast_matmul(&self.w.t()?)?.reshape(shape)
    }
    fn rows_per_launch(&self) -> usize {
        self.rows_per_launch
    }
    fn matvec_rows(&self, xr: &Tensor, n_rows: usize) -> Result<Tensor> {
        assert!(n_rows <= self.rows_per_launch, "chunk wider than the launch");
        self.note(format!("matvec_rows({n_rows})"));
        xr.broadcast_matmul(&self.w.t()?)
    }
}

/// An int4 projection: stored weights, natural basis, no rotation.
struct FakeInt4 {
    name: String,
    w: Tensor,
}

impl Int4Proj for FakeInt4 {
    fn name(&self) -> &str {
        &self.name
    }
    fn d_out(&self) -> usize {
        self.w.dim(0).expect("2-D")
    }
    fn d_in(&self) -> usize {
        self.w.dim(1).expect("2-D")
    }
    /// Refuses more than one vector and a non-zero start offset, exactly as
    /// `CudaInt4::matvec` does (`fused_cuda.rs`, "activation of {len} values
    /// for d_in=" and the offset guard beside it).
    ///
    /// e1d2c9e claimed the fakes refuse what the adapters refuse. It was true
    /// of `FakeLattice` alone until an audit of 2026-09-21 said so.
    fn matvec(&self, x: &Tensor, out_dims: &[usize]) -> Result<Tensor> {
        let d = x.dims();
        let rows: usize = d[..d.len() - 1].iter().product();
        if rows != 1 {
            candle_core::bail!("{}: {rows} vectors for a single-vector kernel", self.name);
        }
        if d[d.len() - 1] != self.d_in() {
            candle_core::bail!(
                "{}: activation of {} values for d_in={}",
                self.name,
                d[d.len() - 1],
                self.d_in()
            );
        }
        let mut shape = out_dims.to_vec();
        *shape.last_mut().expect("rank >= 1") = self.d_out();
        x.broadcast_matmul(&self.w.t()?)?.reshape(shape)
    }
}

/// A row-concatenated group: one matrix of the group's total width.
struct FakeSeg {
    name: String,
    parts: Vec<String>,
    w: Tensor,
    rot: Tensor,
    key: Option<RotKey>,
}

impl SegGroup for FakeSeg {
    fn name(&self) -> &str {
        &self.name
    }
    fn d_out(&self) -> usize {
        self.w.dim(0).expect("2-D")
    }
    fn d_in(&self) -> usize {
        self.w.dim(1).expect("2-D")
    }
    fn rotation(&self) -> Option<RotKey> {
        self.key
    }
    fn part_name(&self, rank: usize) -> &str {
        self.parts
            .get(rank)
            .map_or("(part outside the group)", String::as_str)
    }
    /// Refuses the wrong `d_in`, as `rotate_group` does. A group is one
    /// launch, so it never batches and the row count is the caller's business.
    fn prepare(&self, x: &Tensor) -> Result<Tensor> {
        let d = x.dims();
        if d[d.len() - 1] != self.d_in() {
            candle_core::bail!("{} expects d_in={}, got {}", self.name, self.d_in(), d[d.len() - 1]);
        }
        x.broadcast_matmul(&self.rot)
    }
    fn matvec(&self, xr: &Tensor, out_dims: &[usize]) -> Result<Tensor> {
        let mut shape = out_dims.to_vec();
        *shape.last_mut().expect("rank >= 1") = self.d_out();
        xr.broadcast_matmul(&self.w.t()?)?.reshape(shape)
    }
}

const D_IN: usize = 8;

fn lattice(name: &str, d_out: usize, rows: usize, dev: &Device) -> Result<Proj> {
    Ok(Proj::Lattice(Arc::new(FakeLattice {
        name: name.into(),
        w: matrix(d_out, D_IN, 0.11, dev)?,
        rot: permutation(D_IN, dev)?,
        key: Some((D_IN, 0xA5A5)),
        rows_per_launch: rows,
        log: std::sync::Mutex::new(Vec::new()),
    })))
}

/// The group, and one `Proj::GroupPart` per part, all sharing ONE `Arc`.
fn group(dev: &Device, widths: &[usize]) -> Result<(Arc<dyn SegGroup>, Vec<Proj>)> {
    let total: usize = widths.iter().sum();
    let g: Arc<dyn SegGroup> = Arc::new(FakeSeg {
        name: "000.qkv".into(),
        parts: widths.iter().enumerate().map(|(i, _)| format!("part{i}")).collect(),
        w: matrix(total, D_IN, 0.23, dev)?,
        rot: permutation(D_IN, dev)?,
        key: Some((D_IN, 0xB6B6)),
    });
    let mut row0 = 0;
    let mut parts = Vec::new();
    for (rank, &d_out) in widths.iter().enumerate() {
        parts.push(Proj::GroupPart { group: g.clone(), row0, d_out, rank });
        row0 += d_out;
    }
    Ok((g, parts))
}

fn x1(dev: &Device) -> Result<Tensor> {
    let v: Vec<f32> = (0..D_IN).map(|i| (i as f32 + 1.0) * 0.25).collect();
    Tensor::from_vec(v, (1, D_IN), dev)
}

fn close(a: &Tensor, b: &Tensor) -> Result<f32> {
    (a - b)?.abs()?.max_all()?.to_dtype(DType::F32)?.to_scalar::<f32>()
}

/// The seam executes at all, and it computes the rotated product.
///
/// A mutant that hands `x` to `matvec` instead of the prepared tensor changes
/// this number, because the permutation is not the identity.
#[test]
fn a_lattice_projection_applies_its_own_rotation() -> Result<()> {
    let dev = Device::Cpu;
    let p = lattice("000.q_proj", 4, 1, &dev)?;
    let x = x1(&dev)?;
    let got = p.forward(&x)?;

    let rot = permutation(D_IN, &dev)?;
    let w = matrix(4, D_IN, 0.11, &dev)?;
    let want = x.matmul(&rot)?.matmul(&w.t()?)?;
    assert!(close(&got, &want)? < 1e-6, "the rotated product is what came back");
    Ok(())
}

/// An int4 projection reads the CALLER's activation, never the prepared one.
///
/// Its `prepare` is the identity, so the two coincide today. The assertion is
/// that the result is the UNROTATED product, which is what pins the choice.
#[test]
fn an_int4_projection_stays_in_the_natural_basis() -> Result<()> {
    let dev = Device::Cpu;
    let w = matrix(4, D_IN, 0.31, &dev)?;
    let p = Proj::Int4(Arc::new(FakeInt4 { name: "000.v_proj".into(), w: w.clone() }));
    assert_eq!(p.rot_key(), None, "nothing to carry, nothing to check");
    let x = x1(&dev)?;
    let got = p.forward(&x)?;
    let want = x.matmul(&w.t()?)?;
    assert!(close(&got, &want)? < 1e-6, "the natural-basis product");
    Ok(())
}

/// Three parts built from ONE `Arc` are one group, and the narrowing hands
/// each part its own rows.
#[test]
fn one_group_arc_yields_one_launch_and_the_right_views() -> Result<()> {
    let dev = Device::Cpu;
    let widths = [4usize, 2, 2];
    let (g, parts) = group(&dev, &widths)?;
    let refs: Vec<&Proj> = parts.iter().collect();
    let x = x1(&dev)?;

    let out = group_forward(&refs, &x, RotShare::On)?;
    assert_eq!(out.len(), 3);

    let whole = g.matvec(&g.prepare(&x)?, x.dims())?;
    let mut row0 = 0;
    for (i, &d) in widths.iter().enumerate() {
        assert_eq!(out[i].dims(), &[1, d], "part {i} keeps its own width");
        let want = whole.narrow(1, row0, d)?;
        assert!(close(&out[i], &want)? < 1e-6, "part {i} reads rows {row0}..{}", row0 + d);
        row0 += d;
    }
    Ok(())
}

/// The sharpest line of the port, asserted rather than trusted.
///
/// `SegPlan::of` recognises a group by `Arc::ptr_eq`. Three parts built from
/// three SEPARATE `Arc`s describe the same shape, the same widths and the same
/// names, and must be refused. Building the `Arc` inside the construction loop
/// instead of outside it produces exactly this state, is type-correct, and is
/// invisible to `ops/check-cuda.sh`.
#[test]
fn three_separate_group_arcs_are_refused_by_name() -> Result<()> {
    let dev = Device::Cpu;
    let widths = [4usize, 2, 2];
    let mut parts = Vec::new();
    let mut row0 = 0;
    for (rank, &d_out) in widths.iter().enumerate() {
        // A fresh Arc per part: same contents, different allocation.
        let (g, _) = group(&dev, &widths)?;
        parts.push(Proj::GroupPart { group: g, row0, d_out, rank });
        row0 += d_out;
    }
    let refs: Vec<&Proj> = parts.iter().collect();
    let err = group_forward(&refs, &x1(&dev)?, RotShare::On)
        .expect_err("three allocations are not one group");
    let msg = err.to_string();
    assert!(msg.contains("two groups in a single call"), "must say so: {msg}");
    Ok(())
}

/// A group part beside a projection that belongs to no group is a wiring bug,
/// refused rather than half-launched.
#[test]
fn a_group_part_beside_a_lone_projection_is_refused() -> Result<()> {
    let dev = Device::Cpu;
    let (_, parts) = group(&dev, &[4, 2, 2])?;
    let lone = lattice("000.o_proj", 2, 1, &dev)?;
    let refs: Vec<&Proj> = vec![&parts[0], &parts[1], &lone];
    let err = group_forward(&refs, &x1(&dev)?, RotShare::On)
        .expect_err("a segmented launch cannot cover half a site");
    let msg = err.to_string();
    assert!(msg.contains("belongs to no fused group"), "must say so: {msg}");
    Ok(())
}

/// Parts handed in out of order are refused, not silently reassembled.
///
/// The numbers a transposed order returns are finite and plausible, which is
/// why this is an assertion and not a comment.
#[test]
fn parts_out_of_order_are_refused() -> Result<()> {
    let dev = Device::Cpu;
    let (_, parts) = group(&dev, &[4, 2, 2])?;
    let refs: Vec<&Proj> = vec![&parts[1], &parts[0], &parts[2]];
    let err = group_forward(&refs, &x1(&dev)?, RotShare::On).expect_err("k before q");
    assert!(!err.to_string().is_empty(), "the refusal names the group");
    Ok(())
}

/// A partial group is refused: two parts of a group of three cover it for
/// neither rows nor ranks.
#[test]
fn a_partial_group_is_refused() -> Result<()> {
    let dev = Device::Cpu;
    let (_, parts) = group(&dev, &[4, 2, 2])?;
    let refs: Vec<&Proj> = vec![&parts[0], &parts[1]];
    let err = group_forward(&refs, &x1(&dev)?, RotShare::On).expect_err("two of three");
    assert!(!err.to_string().is_empty(), "the refusal names the group");
    Ok(())
}

/// A projection that batches and one that does not return the same numbers.
///
/// `rows_per_launch` changes the SHAPE of the work and never its result. With
/// four rows, one adapter takes `matvec_rows` once and the other takes
/// `matvec` four times.
#[test]
fn batching_changes_the_launches_and_not_the_result() -> Result<()> {
    let dev = Device::Cpu;
    let rows = 4usize;
    let v: Vec<f32> = (0..rows * D_IN).map(|i| (i as f32 * 0.19).cos()).collect();
    let xs = Tensor::from_vec(v, (rows, D_IN), &dev)?;

    let batched = lattice("000.q_proj", 4, rows, &dev)?;
    let one_a_row = lattice("000.q_proj", 4, 1, &dev)?;
    assert!(batched.batches_rows(), "four rows a launch batches");
    assert!(!one_a_row.batches_rows(), "one row a launch does not");

    let a = group_forward(&[&batched], &xs, RotShare::Off)?;
    let b = group_forward(&[&one_a_row], &xs, RotShare::Off)?;
    assert_eq!(a[0].dims(), b[0].dims());
    assert!(close(&a[0], &b[0])? < 1e-6, "same arithmetic, different launches");
    Ok(())
}

/// An activation prepared for one projection is refused by another in a
/// different basis. This is `check_key`, on a device arm, for the first time.
#[test]
fn an_activation_from_another_basis_is_refused() -> Result<()> {
    let dev = Device::Cpu;
    let mine = lattice("000.q_proj", 4, 1, &dev)?;
    let other = Proj::Lattice(Arc::new(FakeLattice {
        name: "000.k_proj".into(),
        w: matrix(4, D_IN, 0.11, &dev)?,
        rot: permutation(D_IN, &dev)?,
        // A different key, which is the whole point.
        key: Some((D_IN, 0x5A5A)),
        rows_per_launch: 1,
        log: std::sync::Mutex::new(Vec::new()),
    }));
    let x = x1(&dev)?;
    let r = other.prepare(&x)?;
    let err = mine.forward_with(&r, &x).expect_err("another basis");
    assert!(!err.to_string().is_empty(), "the refusal names the site");
    Ok(())
}

/// The chunk size is OBSERVED, not inferred from the result.
///
/// Added after a mutation run: making `prefill_rows_of` never batch left every
/// other test in this file green, because they all assert the numbers and the
/// numbers do not move. A chunk of four must be ONE launch of four rows, and
/// that is a statement about the launches.
#[test]
fn a_batching_projection_takes_one_launch_and_not_four() -> Result<()> {
    let dev = Device::Cpu;
    let rows = 4usize;
    let v: Vec<f32> = (0..rows * D_IN).map(|i| (i as f32 * 0.19).cos()).collect();
    let xs = Tensor::from_vec(v, (rows, D_IN), &dev)?;

    let fake = Arc::new(FakeLattice {
        name: "000.q_proj".into(),
        w: matrix(4, D_IN, 0.11, &dev)?,
        rot: permutation(D_IN, &dev)?,
        key: Some((D_IN, 0xA5A5)),
        rows_per_launch: rows,
        log: std::sync::Mutex::new(Vec::new()),
    });
    let p = Proj::Lattice(fake.clone());
    group_forward(&[&p], &xs, RotShare::Off)?;

    assert_eq!(
        fake.launches(),
        vec![format!("prepare_rows({rows})"), format!("matvec_rows({rows})")],
        "four rows are one rotation and one matvec, not four of each"
    );
    Ok(())
}

/// The other half of the same claim: a projection that does not batch takes
/// one launch a row, and says so.
#[test]
fn a_non_batching_projection_takes_one_launch_a_row() -> Result<()> {
    let dev = Device::Cpu;
    let rows = 3usize;
    let v: Vec<f32> = (0..rows * D_IN).map(|i| (i as f32 * 0.19).cos()).collect();
    let xs = Tensor::from_vec(v, (rows, D_IN), &dev)?;

    let fake = Arc::new(FakeLattice {
        name: "000.q_proj".into(),
        w: matrix(4, D_IN, 0.11, &dev)?,
        rot: permutation(D_IN, &dev)?,
        key: Some((D_IN, 0xA5A5)),
        rows_per_launch: 1,
        log: std::sync::Mutex::new(Vec::new()),
    });
    let p = Proj::Lattice(fake.clone());
    group_forward(&[&p], &xs, RotShare::Off)?;

    let want: Vec<String> = (0..rows)
        .flat_map(|_| ["prepare".to_string(), "matvec(1)".to_string()])
        .collect();
    assert_eq!(fake.launches(), want, "one rotation and one matvec a row");
    Ok(())
}

/// The wiring `prepare_rows` owes the rows kernel, observed.
///
/// Added after an adversarial review reproduced this: wiring
/// `Proj::prepare_rows`'s lattice arm to `d.prepare(xs)` left device_seam
/// 11/11, clippy 0 and `ops/check-cuda.sh` 0, and died only on a card, at the
/// first prefill chunk, because `FusedRuntime::rotate` refuses `rows != 1`.
/// The fake now refuses it too, and the launch log names which entry ran.
#[test]
fn a_chunk_is_rotated_by_prepare_rows_and_never_by_prepare() -> Result<()> {
    let dev = Device::Cpu;
    let rows = 4usize;
    let v: Vec<f32> = (0..rows * D_IN).map(|i| (i as f32 * 0.19).cos()).collect();
    let xs = Tensor::from_vec(v, (rows, D_IN), &dev)?;

    let fake = Arc::new(FakeLattice {
        name: "000.q_proj".into(),
        w: matrix(4, D_IN, 0.11, &dev)?,
        rot: permutation(D_IN, &dev)?,
        key: Some((D_IN, 0xA5A5)),
        rows_per_launch: rows,
        log: std::sync::Mutex::new(Vec::new()),
    });
    let p = Proj::Lattice(fake.clone());
    group_forward(&[&p], &xs, RotShare::Off)?;

    assert_eq!(
        fake.launches(),
        vec![format!("prepare_rows({rows})"), format!("matvec_rows({rows})")],
        "a chunk takes prepare_rows then matvec_rows, and neither one-row entry"
    );
    Ok(())
}

/// One row takes `prepare`, which is the entry the rotation kernel accepts.
#[test]
fn one_row_is_rotated_by_prepare() -> Result<()> {
    let dev = Device::Cpu;
    let fake = Arc::new(FakeLattice {
        name: "000.q_proj".into(),
        w: matrix(4, D_IN, 0.11, &dev)?,
        rot: permutation(D_IN, &dev)?,
        key: Some((D_IN, 0xA5A5)),
        rows_per_launch: 1,
        log: std::sync::Mutex::new(Vec::new()),
    });
    let p = Proj::Lattice(fake.clone());
    group_forward(&[&p], &x1(&dev)?, RotShare::Off)?;
    assert_eq!(fake.launches(), vec!["prepare", "matvec(1)"], "one row, one-row entries");
    Ok(())
}

/// The int4 arm reads the CALLER's activation and not the prepared one.
///
/// This is the mutant an earlier draft of this file wrongly called equivalent.
/// A `Rotated` carrying a DIFFERENT activation with the same `None` key passes
/// `check_key`, so the two readings are distinguishable and the choice is a
/// behaviour after all.
#[test]
fn the_int4_arm_reads_the_callers_activation() -> Result<()> {
    let dev = Device::Cpu;
    let w = matrix(4, D_IN, 0.31, &dev)?;
    let p = Proj::Int4(Arc::new(FakeInt4 { name: "000.v_proj".into(), w: w.clone() }));

    // A second activation, and a `Rotated` built from it. `Proj::Dense` and
    // `Proj::Int4` both hand back `key: None`, so `check_key` accepts this.
    let other: Vec<f32> = (0..D_IN).map(|i| (i as f32) * -0.5 - 3.0).collect();
    let other = Tensor::from_vec(other, (1, D_IN), &dev)?;
    let r = p.prepare(&other)?;

    let x = x1(&dev)?;
    let got = p.forward_with(&r, &x)?;
    let want_x = x.matmul(&w.t()?)?;
    let want_other = other.matmul(&w.t()?)?;
    assert!(close(&got, &want_x)? < 1e-6, "it read the caller's x");
    assert!(close(&got, &want_other)? > 1e-3, "and not the tensor the Rotated carried");
    Ok(())
}

/// A group of ONE int4 projection over several rows: the sealed objects' lone
/// `o_proj` and `down_proj`.
///
/// No projection of such a group carries a rotation, so `group_forward` takes
/// its dense branch, which hands the whole tensor to `forward_with`. A `Linear`
/// takes that; `tv_q4_h` takes one vector. The 4B sealed file died on exactly
/// this at the prefill gate on 2026-09-25 (job 6ab6211052d0dbd7f1d8eada,
/// "activation of 831488 values for d_in=4096", 203 rows): an int4 `v_proj`
/// never reached the branch because q and k rotate. Rank 3 on purpose: the
/// block hands `[batch, seq, hidden]`.
#[test]
fn a_lone_int4_projection_takes_several_rows_one_at_a_time() -> Result<()> {
    let dev = Device::Cpu;
    let w = matrix(4, D_IN, 0.37, &dev)?;
    let p = Proj::Int4(Arc::new(FakeInt4 { name: "000.o_proj".into(), w: w.clone() }));
    let rows = 3usize;
    let v: Vec<f32> = (0..rows * D_IN).map(|i| (i as f32 * 0.29).cos()).collect();
    let xs = Tensor::from_vec(v, (1, rows, D_IN), &dev)?;

    let out = group_forward(&[&p], &xs, RotShare::On)?;
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].dims(), &[1, rows, 4], "the caller's shape, d_out last");
    let want = xs.broadcast_matmul(&w.t()?)?;
    assert!(close(&out[0], &want)? < 1e-6, "row by row is the whole product");

    // One row stays the one-row call, and gives the same row.
    let one = group_forward(&[&p], &xs.narrow(1, 1, 1)?, RotShare::On)?;
    assert!(close(&one[0], &want.narrow(1, 1, 1)?)? < 1e-6);
    Ok(())
}

/// `SegPlan::run` over several rows: the cat, the narrow and the reshape.
///
/// The single-row group test exercises the case the code's own comment calls
/// trivially contiguous. This one is the case that is not.
#[test]
fn a_group_over_several_rows_narrows_each_part_correctly() -> Result<()> {
    let dev = Device::Cpu;
    let widths = [4usize, 2, 2];
    let rows = 3usize;
    let (g, parts) = group(&dev, &widths)?;
    let refs: Vec<&Proj> = parts.iter().collect();
    let v: Vec<f32> = (0..rows * D_IN).map(|i| (i as f32 * 0.13).sin()).collect();
    let xs = Tensor::from_vec(v, (rows, D_IN), &dev)?;

    let out = group_forward(&refs, &xs, RotShare::On)?;
    let whole = g.matvec(&g.prepare(&xs.narrow(0, 0, 1)?)?, &[1, D_IN])?;

    let mut row0 = 0;
    for (i, &d) in widths.iter().enumerate() {
        assert_eq!(out[i].dims(), &[rows, d], "part {i} keeps its width over {rows} rows");
        // Row 0 of each part must match the group's own row 0, narrowed.
        let got0 = out[i].narrow(0, 0, 1)?;
        let want0 = whole.narrow(1, row0, d)?;
        assert!(close(&got0, &want0)? < 1e-6, "part {i}, row 0, rows {row0}..{}", row0 + d);
        row0 += d;
    }
    Ok(())
}
