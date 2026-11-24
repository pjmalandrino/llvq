//! The Tetra mark for a terminal: four nested chevrons, the outer one in the
//! accent colour, and the name. The drawing and its colour are those of
//! `docs/brand/LOGO.md`; a test holds it equal to `docs/brand/ascii/`.

/// Terracotta, `#B4502D`, as a 24-bit foreground.
const ACCENT: &str = "\x1b[38;2;180;80;45m";
const RESET: &str = "\x1b[0m";

/// The banner, four lines, each ended by a newline. `color` paints the outer
/// chevron; without it the text is plain ASCII.
pub fn banner(color: bool) -> String {
    let (a, r) = if color { (ACCENT, RESET) } else { ("", "") };
    let mut out = String::new();
    for k in 0..4 {
        out.push_str(&" ".repeat(3 - k));
        out.push_str(&format!("{a}/{r}{}{}{a}\\{r}", "/".repeat(k), "\\".repeat(k)));
        if k == 2 {
            out.push_str("    tetra");
        }
        out.push('\n');
    }
    out
}

/// Whether a stream that is a terminal should get colour: not under
/// `NO_COLOR` (any non-empty value, <https://no-color.org>) and not on a
/// `dumb` terminal.
pub fn wants_color(is_terminal: bool) -> bool {
    is_terminal
        && std::env::var_os("NO_COLOR").is_none_or(|v| v.is_empty())
        && std::env::var("TERM").map_or(true, |t| t != "dumb")
}

#[cfg(test)]
mod tests {
    use super::banner;

    #[test]
    fn the_banner_is_the_brand_file() {
        assert_eq!(banner(false), include_str!("../../docs/brand/ascii/banner.txt"));
        assert_eq!(banner(true), include_str!("../../docs/brand/ascii/banner.ansi"));
    }
}
