//! Summary statistics the bench binaries share, so each one stops carrying
//! its own copy (audit batch 10, 2026-11-08). Only copies that were identical
//! moved here; `rotbench` keeps its own `median_range`, which averages the two
//! central values of an even count where these take the upper one.

/// `(min, median, max)`, the median being the upper one of an even count.
pub fn spread(mut v: Vec<f64>) -> (f64, f64, f64) {
    v.sort_by(f64::total_cmp);
    (v[0], v[v.len() / 2], v[v.len() - 1])
}

/// `(median, min, max)`, the median being the upper one of an even count.
pub fn median_range(v: &[f64]) -> (f64, f64, f64) {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    (s[s.len() / 2], s[0], s[s.len() - 1])
}

/// The largest `|got - want| / scale` over the rows, `scale` floored at 1e-12.
pub fn worst_error(got: &[f32], want: &[f64], scale: &[f64]) -> f64 {
    got.iter()
        .zip(want)
        .zip(scale)
        .map(|((&g, &w), &s)| (g as f64 - w).abs() / s.max(1e-12))
        .fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_median_of_an_even_count_is_the_upper_one() {
        assert_eq!(spread(vec![4.0, 1.0, 3.0, 2.0]), (1.0, 3.0, 4.0));
        assert_eq!(median_range(&[4.0, 1.0, 3.0, 2.0]), (3.0, 1.0, 4.0));
        assert_eq!(median_range(&[2.0, 9.0, 5.0]), (5.0, 2.0, 9.0));
        assert_eq!(worst_error(&[1.0, 2.5], &[1.0, 2.0], &[1.0, 0.25]), 2.0);
    }
}
