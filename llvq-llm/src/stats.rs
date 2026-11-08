//! Summary statistics shared by the bench binaries.

/// Median, min and max of a series. An even count takes the mean of the two
/// central values.
pub fn median_range(v: &[f64]) -> (f64, f64, f64) {
    assert!(!v.is_empty(), "median over zero rounds");
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    let med = if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 };
    (med, s[0], s[n - 1])
}

#[cfg(test)]
mod tests {
    use super::median_range;

    /// Odd count: the middle value; even count: the mean of the two central
    /// ones; the range is `[min, max]` whatever the input order.
    #[test]
    fn median_and_range_of_shuffled_series() {
        assert_eq!(median_range(&[3.0, 1.0, 2.0]), (2.0, 1.0, 3.0));
        assert_eq!(median_range(&[4.0, 1.0, 3.0, 2.0]), (2.5, 1.0, 4.0));
        assert_eq!(median_range(&[7.0]), (7.0, 7.0, 7.0));
    }
}
