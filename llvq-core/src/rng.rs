//! Minimal deterministic RNG (SplitMix64) for tests and benches.
//!
//! Kept in the library (not `#[cfg(test)]`) so integration tests and future
//! benches can use it without pulling an external `rand` dependency.

/// SplitMix64 — tiny, fast, and statistically fine for test sampling.
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    /// Seeded construction; the same seed always yields the same stream.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Next 64 pseudo-random bits.
    ///
    /// Named like `Iterator::next` on purpose — this is the conventional
    /// name for a raw RNG step; the type does not implement `Iterator`.
    #[allow(clippy::should_implement_trait)]
    #[inline]
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)` (53-bit mantissa).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Standard normal `N(0, 1)` via Box–Muller (cosine branch).
    pub fn next_gaussian(&mut self) -> f64 {
        let u1 = ((self.next() >> 11) as f64 + 1.0) * (1.0 / 9_007_199_254_740_993.0); // (0,1]
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
    }
}
