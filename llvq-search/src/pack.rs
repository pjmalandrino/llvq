//! Dense bit packing for artifact streams.
//!
//! A Leech index is 47 or 48 bits wide and a gain code is 0–2 bits. Storing
//! either in a byte-aligned slot would waste 12.5 % of the file, which is the
//! whole margin the shell cap was trying to win — so the artifact packs them
//! back to back, with no padding except at the very end.
//!
//! ## Byte order, and why it is spelled out
//!
//! Values are written **most-significant bit first**, filling each byte from
//! its high bit down. A value of width `w` occupies the next `w` bits of the
//! stream in that order, whatever byte boundary it straddles. This is a
//! format decision, not an implementation detail: a reader that disagrees
//! about it silently returns different weights rather than failing, so it is
//! stated here, pinned by tests, and must not change without a format bump.
//!
//! The final byte is zero-padded. A stream therefore does not carry its own
//! length — the header does.

/// Writes values of arbitrary bit width into a dense stream.
#[derive(Default)]
pub struct BitWriter {
    buf: Vec<u8>,
    /// Bits not yet flushed, left-aligned in the low `nbits`.
    acc: u64,
    nbits: u32,
}

impl BitWriter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Pre-allocate for a stream of `bits` total.
    pub fn with_capacity(bits: u64) -> Self {
        Self {
            buf: Vec::with_capacity(bits.div_ceil(8) as usize),
            acc: 0,
            nbits: 0,
        }
    }

    /// Append the low `width` bits of `value`, most significant first.
    ///
    /// Panics if `width > 64`, or if `value` does not fit in `width` bits —
    /// a silently truncated index would be indistinguishable from a valid one
    /// belonging to a different lattice point.
    pub fn push(&mut self, value: u64, width: u32) {
        assert!(width <= 64, "width {width} exceeds 64");
        if width == 0 {
            return;
        }
        if width < 64 {
            assert!(
                value < (1u64 << width),
                "value {value} does not fit in {width} bits"
            );
        }
        let mut left = width;
        while left > 0 {
            let take = left.min(64 - self.nbits);
            // The `take` most significant bits of what is left of `value`.
            let chunk = (value >> (left - take)) & mask(take);
            // `take` reaches 64 only with an empty accumulator (`nbits == 0`,
            // hence `acc == 0`), where a literal `acc << 64` overflows the
            // shift width — a debug panic, and correct in release only by the
            // accident that the masked shift keeps an accumulator that is 0.
            self.acc = if take == 64 {
                chunk
            } else {
                (self.acc << take) | chunk
            };
            self.nbits += take;
            left -= take;
            while self.nbits >= 8 {
                let shift = self.nbits - 8;
                self.buf.push(((self.acc >> shift) & 0xFF) as u8);
                self.nbits -= 8;
                self.acc &= mask(self.nbits);
            }
        }
    }

    /// Bits written so far, padding excluded.
    pub fn bit_len(&self) -> u64 {
        self.buf.len() as u64 * 8 + self.nbits as u64
    }

    /// Flush the trailing partial byte (zero-padded) and yield the stream.
    pub fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            let pad = 8 - self.nbits;
            self.buf.push(((self.acc << pad) & 0xFF) as u8);
            self.nbits = 0;
            self.acc = 0;
        }
        self.buf
    }
}

/// Reads back a [`BitWriter`] stream.
pub struct BitReader<'a> {
    buf: &'a [u8],
    pos: u64,
}

impl<'a> BitReader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    /// Read the next `width` bits, most significant first.
    pub fn read(&mut self, width: u32) -> u64 {
        assert!(width <= 64, "width {width} exceeds 64");
        if width == 0 {
            return 0;
        }
        assert!(
            self.pos + width as u64 <= self.buf.len() as u64 * 8,
            "read of {width} bits past the end of a {}-byte stream",
            self.buf.len()
        );
        let mut out = 0u64;
        let mut left = width;
        while left > 0 {
            let byte = (self.pos / 8) as usize;
            let bit_in_byte = (self.pos % 8) as u32;
            let avail = 8 - bit_in_byte;
            let take = left.min(avail);
            let b = self.buf[byte] as u64;
            let chunk = (b >> (avail - take)) & mask(take);
            out = (out << take) | chunk;
            self.pos += take as u64;
            left -= take;
        }
        out
    }

    /// Bits consumed so far.
    pub fn bit_pos(&self) -> u64 {
        self.pos
    }
}

#[inline]
fn mask(bits: u32) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}
