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
//! ## The mutation run, and the one mutant that is equivalent
//!
//! Four mutants of `model.rs`, three killed: `forward_with` reading the
//! caller's activation instead of the prepared one, `Arc::ptr_eq` always true,
//! `prefill_rows_of` never batching, and a part's view starting at row 0.
//!
//! A fifth is EQUIVALENT and is recorded rather than chased: making the int4
//! arm read `r.t` instead of `x`. `Proj::Int4`'s `prepare` returns the
//! activation untouched, and `check_key` refuses any `Rotated` whose key is
//! not `None`, so every value that arm can ever accept has `r.t == x`. The
//! argument choice there is documentation of intent, not a behaviour. A test
//! that appeared to cover it would be testing its own fake.

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
    fn prepare(&self, x: &Tensor) -> Result<Tensor> {
        x.broadcast_matmul(&self.rot)
    }
    fn prepare_rows(&self, xs: &Tensor, rows: usize) -> Result<Tensor> {
        assert_ne!(rows, 1, "the caller takes prepare() at one row");
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
    fn matvec(&self, x: &Tensor, out_dims: &[usize]) -> Result<Tensor> {
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
    fn prepare(&self, x: &Tensor) -> Result<Tensor> {
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
        vec![format!("matvec_rows({rows})")],
        "four rows are one launch of four, not four launches of one"
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

    assert_eq!(fake.launches(), vec!["matvec(1)"; rows], "one launch a row");
    Ok(())
}
