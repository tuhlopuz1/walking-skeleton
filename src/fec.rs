//! CRC-16, Hamming(7,4) and the block interleaver.
//!
//! These three together are the frame's error control. Hamming fixes one bit
//! per codeword, the interleaver spreads a burst across codewords so Hamming
//! can, and the CRC is the final word on whether the result is trustworthy.

/// Codewords per interleaver block.
///
/// One M-FSK symbol error corrupts up to `bits_per_symbol` *consecutive* bits,
/// which without interleaving all land in one codeword and are uncorrectable.
/// The depth is chosen so the 6-byte header is exactly one block — that is what
/// lets a receiver de-interleave the header before it knows the frame length.
pub const INTERLEAVE_CW: usize = 12;
/// Bytes per interleaver block (two nibbles per byte).
pub const INTERLEAVE_BYTES: usize = INTERLEAVE_CW / 2;

const BLOCK_BITS: usize = 7 * INTERLEAVE_CW;

// --------------------------------------------------------------------------- //
// CRC-16 / CCITT-FALSE
// --------------------------------------------------------------------------- //
const fn crc_table() -> [u16; 256] {
    let mut tbl = [0u16; 256];
    let mut b = 0usize;
    while b < 256 {
        let mut crc = (b as u16) << 8;
        let mut i = 0;
        while i < 8 {
            crc = if crc & 0x8000 != 0 {
                (crc << 1) ^ 0x1021
            } else {
                crc << 1
            };
            i += 1;
        }
        tbl[b] = crc;
        b += 1;
    }
    tbl
}

static CRC_TBL: [u16; 256] = crc_table();

pub fn crc16(data: &[u8]) -> u16 {
    crc16_init(data, 0xFFFF)
}

pub fn crc16_init(data: &[u8], init: u16) -> u16 {
    let mut crc = init;
    for &b in data {
        crc = (crc << 8) ^ CRC_TBL[(((crc >> 8) ^ b as u16) & 0xFF) as usize];
    }
    crc
}

// --------------------------------------------------------------------------- //
// Hamming(7,4)
// --------------------------------------------------------------------------- //
/// Bits produced for `n` bytes: two nibbles each, seven bits per nibble.
pub fn hamming_bit_count(n_bytes: usize) -> usize {
    n_bytes * 2 * 7
}

/// Encode a nibble (low 4 bits of `nib`) into 7 code bits.
///
/// Written out rather than kept as a matrix: the generator is fixed and the
/// explicit form is what the parity-check below has to agree with.
#[inline]
fn encode_nibble(nib: u8, out: &mut [u8]) {
    let d0 = (nib >> 3) & 1; // most significant data bit
    let d1 = (nib >> 2) & 1;
    let d2 = (nib >> 1) & 1;
    let d3 = nib & 1;
    out[0] = d0 ^ d1 ^ d3;
    out[1] = d0 ^ d2 ^ d3;
    out[2] = d0;
    out[3] = d1 ^ d2 ^ d3;
    out[4] = d1;
    out[5] = d2;
    out[6] = d3;
}

pub fn hamming_encode(data: &[u8]) -> Vec<u8> {
    let mut bits = vec![0u8; hamming_bit_count(data.len())];
    for (i, &byte) in data.iter().enumerate() {
        encode_nibble(byte >> 4, &mut bits[i * 14..i * 14 + 7]);
        encode_nibble(byte & 0x0F, &mut bits[i * 14 + 7..i * 14 + 14]);
    }
    bits
}

/// Decode bits back to bytes, correcting a single bit error per codeword.
///
/// Trailing bits that do not complete a byte are dropped, matching the encoder:
/// a half-decoded byte carries no meaning the CRC could check.
pub fn hamming_decode(bits: &[u8]) -> Vec<u8> {
    let n = bits.len() / 7;
    let mut nibbles = Vec::with_capacity(n);
    for c in 0..n {
        let mut cw = [0u8; 7];
        cw.copy_from_slice(&bits[c * 7..c * 7 + 7]);
        let s0 = cw[0] ^ cw[2] ^ cw[4] ^ cw[6];
        let s1 = cw[1] ^ cw[2] ^ cw[5] ^ cw[6];
        let s2 = cw[3] ^ cw[4] ^ cw[5] ^ cw[6];
        let pos = (s0 + 2 * s1 + 4 * s2) as usize;
        if pos != 0 {
            cw[pos - 1] ^= 1;
        }
        nibbles.push((cw[2] << 3) | (cw[4] << 2) | (cw[5] << 1) | cw[6]);
    }
    nibbles
        .chunks_exact(2)
        .map(|p| (p[0] << 4) | p[1])
        .collect()
}

// --------------------------------------------------------------------------- //
// Block interleaver
// --------------------------------------------------------------------------- //
/// Write codewords as rows, read out columns. Truncates to whole blocks.
pub fn interleave(bits: &[u8]) -> Vec<u8> {
    let blocks = bits.len() / BLOCK_BITS;
    let mut out = vec![0u8; blocks * BLOCK_BITS];
    for b in 0..blocks {
        let base = b * BLOCK_BITS;
        for cw in 0..INTERLEAVE_CW {
            for bit in 0..7 {
                out[base + bit * INTERLEAVE_CW + cw] = bits[base + cw * 7 + bit];
            }
        }
    }
    out
}

/// Inverse of [`interleave`]. Truncates to whole blocks.
pub fn deinterleave(bits: &[u8]) -> Vec<u8> {
    let blocks = bits.len() / BLOCK_BITS;
    let mut out = vec![0u8; blocks * BLOCK_BITS];
    for b in 0..blocks {
        let base = b * BLOCK_BITS;
        for cw in 0..INTERLEAVE_CW {
            for bit in 0..7 {
                out[base + cw * 7 + bit] = bits[base + bit * INTERLEAVE_CW + cw];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc_known_vector() {
        // The canonical CCITT-FALSE check value.
        assert_eq!(crc16(b"123456789"), 0x29B1);
    }

    #[test]
    fn hamming_round_trips() {
        let data: Vec<u8> = (0u8..=255).collect();
        assert_eq!(hamming_decode(&hamming_encode(&data)), data);
    }

    #[test]
    fn hamming_corrects_one_bit_per_codeword() {
        let data = b"correct me".to_vec();
        let clean = hamming_encode(&data);
        for flip in 0..clean.len() {
            let mut bad = clean.clone();
            bad[flip] ^= 1;
            assert_eq!(hamming_decode(&bad), data, "flipping bit {flip}");
        }
    }

    #[test]
    fn interleaver_round_trips() {
        let bits: Vec<u8> = (0..BLOCK_BITS * 3).map(|i| (i % 2) as u8).collect();
        assert_eq!(deinterleave(&interleave(&bits)), bits);
    }

    #[test]
    fn interleaving_survives_a_burst() {
        // A burst as long as the interleaver is deep lands one bit in each
        // codeword, which is exactly what Hamming can repair.
        let data = b"burst test!!".to_vec(); // 12 bytes = one block
        let mut wire = interleave(&hamming_encode(&data));
        for bit in wire.iter_mut().take(INTERLEAVE_CW) {
            *bit ^= 1;
        }
        assert_eq!(hamming_decode(&deinterleave(&wire)), data);
    }
}
