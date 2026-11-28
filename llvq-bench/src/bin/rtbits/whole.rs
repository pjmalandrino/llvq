//! The whole-model accounting over the sealed file and an optional sidecar,
//! with its comparison against the Inkling plan: every count read from the
//! files, printed by part.

use super::*;

/// One part of a model: parameters, and the bits they occupy.
#[derive(Default, Clone, Copy)]
pub(super) struct Part {
    params: u64,
    bits: u64,
}

impl Part {
    pub(super) fn add(&mut self, params: u64, bits: u64) {
        self.params += params;
        self.bits += bits;
    }
    fn bpp(&self) -> f64 {
        self.bits as f64 / self.params.max(1) as f64
    }
}

/// The whole-model accounting over the sealed file **and** a sidecar
/// safetensors (the Inkling seal, `llvq_llm::inkseal`), every count read
/// from the files.
///
/// Lattice records are counted twice, in two accountings that are never
/// blended: **archive** is the record as stored (48-bit words, f64 row
/// scales, f64 centroids, f32 tail), the accounting of `bin/extq`'s
/// `archive_bpw`; **kernel** is words plus an f32 row scale and an f32 tail,
/// the accounting of [`V5Rate::kernel_bpw`]`(true)`. The expert records
/// (`.mlp.experts.` in the name) are kept apart from the other lattice
/// records. Sidecar tensors are charged at their stored dtype width, and
/// `model.visual.*` is kept apart, so the text model and the model with
/// vision are both stated.
#[derive(Default)]
pub(super) struct Whole {
    expert_archive: Part,
    expert_kernel: Part,
    lattice_archive: Part,
    lattice_kernel: Part,
    pub(super) int4: Part,
    pub(super) raw: Part,
    sidecar_text: Part,
    sidecar_vision: Part,
    sidecar_tensors: u64,
    /// Native parameters a partial seal did not carry (`inkling-seal.json`).
    pub(super) omitted: u64,
    pub(super) seal_blob: bool,
}

impl Whole {
    pub(super) fn push_lattice(&mut self, m: &llvq_artifact::RawMatrix) {
        let (ib, gb) = llvq_artifact::code_widths(m.kind, &m.name, m.shell_cap, m.centroids.len())
            .expect("lattice widths");
        let words = m.indices.len() as u64 * (ib + gb) as u64;
        let weights = (m.d_out * m.d_in) as u64;
        let archive = words
            + m.row_scales.len() as u64 * 64
            + m.centroids.len() as u64 * 64
            + m.tail.len() as u64 * F32_BITS;
        let kernel = words + m.row_scales.len() as u64 * F32_BITS + m.tail.len() as u64 * F32_BITS;
        if m.name.contains(".mlp.experts.") {
            self.expert_archive.add(weights, archive);
            self.expert_kernel.add(weights, kernel);
        } else {
            self.lattice_archive.add(weights, archive);
            self.lattice_kernel.add(weights, kernel);
        }
    }

    pub(super) fn read_sidecar(&mut self, path: &str) {
        let mut f = File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
        let mut b = [0u8; 8];
        f.read_exact(&mut b).expect("sidecar header length");
        let n = u64::from_le_bytes(b) as usize;
        let mut h = vec![0u8; n];
        f.read_exact(&mut h).expect("sidecar header");
        let v = json::parse(&String::from_utf8_lossy(&h)).expect("sidecar header JSON");
        let json::Value::Object(obj) = v else { panic!("{path}: header is not an object") };
        for (name, t) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype = t.get("dtype").and_then(json::Value::as_str).expect("dtype");
            let width = match dtype {
                "F64" | "I64" => 64,
                "F32" | "I32" => 32,
                "F16" | "BF16" => 16,
                "U8" | "I8" | "F8_E4M3" | "F8_E5M2" => 8,
                other => panic!("{path}: {name}: dtype {other}"),
            };
            let params: u64 = match t.get("shape") {
                Some(json::Value::Array(d)) => d.iter().map(|x| x.as_u64().expect("shape")).product(),
                _ => panic!("{path}: {name}: shape"),
            };
            self.sidecar_tensors += 1;
            if name.starts_with("model.visual.") {
                self.sidecar_vision.add(params, params * width);
            } else {
                self.sidecar_text.add(params, params * width);
            }
        }
    }

    pub(super) fn print(&self, sidecar: Option<&str>) {
        if !self.seal_blob && sidecar.is_none() {
            return;
        }
        println!("\n  ── whole model over the sealed file{} ──", if sidecar.is_some() { " and its sidecar" } else { "" });
        let row = |name: &str, p: &Part| {
            println!(
                "  {name:<34}{:>16}{:>18}{:>12.6}{:>10.4}",
                p.params,
                p.bits,
                p.bpp(),
                p.bits as f64 / 8e9
            );
        };
        println!("  {:<34}{:>16}{:>18}{:>12}{:>10}", "part", "parameters", "bits", "b/param", "GB");
        row("experts, lattice, archive", &self.expert_archive);
        row("experts, lattice, kernel", &self.expert_kernel);
        row("other lattice, archive", &self.lattice_archive);
        row("other lattice, kernel", &self.lattice_kernel);
        row("int4 g128 records", &self.int4);
        row("raw tensors (file, as stored)", &self.raw);
        row("sidecar, text model", &self.sidecar_text);
        row("sidecar, model.visual.*", &self.sidecar_vision);
        if let Some(s) = sidecar {
            println!("  sidecar {s}: {} tensors", self.sidecar_tensors);
        }
        for (acc, ex, la) in [
            ("archive", self.expert_archive, self.lattice_archive),
            ("kernel", self.expert_kernel, self.lattice_kernel),
        ] {
            let mut text = Part::default();
            for p in [ex, la, self.int4, self.raw, self.sidecar_text] {
                text.add(p.params, p.bits);
            }
            let mut all = text;
            all.add(self.sidecar_vision.params, self.sidecar_vision.bits);
            println!(
                "  b/param, lattice at {acc:<8}: text model {:.6} over {} parameters; with vision {:.6} over {}",
                text.bpp(),
                text.params,
                all.bpp(),
                all.params
            );
        }
        self.print_plan();
        if self.omitted > 0 {
            println!(
                "  PARTIAL SEAL: {} native tensor elements are not carried (inkling-seal.json; an NVFP4\n           byte holds two weights); the b/param above covers the carried parameters only and\n           is not a whole-model figure.",
                self.omitted
            );
        }
    }
}

impl Whole {
    /// The rows of `docs/plan-inkling2-mini.md` section 1 (*computed*
    /// estimates from the checkpoint header), beside the same rows read from
    /// the files, lattice at the kernel accounting as the plan has it. The
    /// plan's rows map one to one: experts = expert records; attention Tetra
    /// = the other lattice records; attention int4 = the int4 records; tables
    /// = the raw tensors; the 16-bit rest = the sidecar's text tensors.
    fn print_plan(&self) {
        let plan: [(&str, u64, f64, Part); 5] = [
            ("experts (Tetra)", 24_461_180_928, 6.172, self.expert_kernel),
            ("attention, Tetra", 226_492_416, 0.057, self.lattice_kernel),
            ("attention, int4 g128", 417_595_392, 0.222, self.int4),
            ("embed + unembed, int4 g64", 926_318_592, 0.521, self.raw),
            ("rest as shipped (sidecar)", 38_797_824, 0.078, self.sidecar_text),
        ];
        println!("\n  against plan section 1 (plan figures computed; file figures measured, lattice at kernel)");
        println!(
            "  {:<28}{:>16}{:>16}{:>9}{:>9}{:>10}{:>10}",
            "row", "params plan", "params file", "GB plan", "GB file", "b/p plan", "b/p file"
        );
        let mut text = Part::default();
        for (name, pp, gb, p) in plan {
            text.add(p.params, p.bits);
            println!(
                "  {name:<28}{pp:>16}{:>16}{gb:>9.3}{:>9.3}{:>10.4}{:>10.4}",
                p.params,
                p.bits as f64 / 8e9,
                gb * 8e9 / pp as f64,
                p.bpp()
            );
        }
        let line = |name: &str, pp: u64, gb: f64, bpp: f64, p: &Part| {
            println!(
                "  {name:<28}{pp:>16}{:>16}{gb:>9.3}{:>9.3}{bpp:>10.4}{:>10.4}",
                p.params,
                p.bits as f64 / 8e9,
                p.bpp()
            );
        };
        line("text model", 26_070_385_152, 7.049, 2.163, &text);
        line("vision tower (sidecar)", 318_256_128, 0.637, 16.0, &self.sidecar_vision);
        let mut all = text;
        all.add(self.sidecar_vision.params, self.sidecar_vision.bits);
        line("with vision", 26_388_641_280, 7.686, 2.330, &all);
    }
}
