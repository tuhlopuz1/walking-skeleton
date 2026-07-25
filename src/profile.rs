//! Modulation profiles — HOW to modulate: tone spacing and symbol timing, with
//! no absolute frequencies in them.
//!
//! A profile is band-agnostic, so FAST/NORMAL/ROBUST applies whether the link
//! runs inaudibly at 18 kHz or audibly at 3 kHz. Where the tones actually land
//! comes from the [`Band`](crate::band::Band). Profile ids travel in the frame
//! header, so a receiver adapts per frame.

use crate::SAMPLE_RATE;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Profile {
    pub name: &'static str,
    /// Hz between adjacent tones.
    pub spacing: f64,
    /// M-FSK order as a power of two: 2 -> 4 tones, 3 -> 8 tones.
    pub bits_per_symbol: u32,
    /// Tone duration.
    pub symbol_ms: f64,
    /// Silence between symbols, for echo decay.
    pub guard_ms: f64,
}

impl Profile {
    pub fn n_tones(&self) -> usize {
        1 << self.bits_per_symbol
    }

    pub fn symbol_samples(&self) -> usize {
        (SAMPLE_RATE as f64 * self.symbol_ms / 1000.0) as usize
    }

    pub fn guard_samples(&self) -> usize {
        (SAMPLE_RATE as f64 * self.guard_ms / 1000.0) as usize
    }

    pub fn step_samples(&self) -> usize {
        self.symbol_samples() + self.guard_samples()
    }

    /// Raw channel bits per second, before the 4/7 FEC overhead.
    pub fn bitrate(&self) -> f64 {
        self.bits_per_symbol as f64 * SAMPLE_RATE as f64 / self.step_samples() as f64
    }

    /// Distance from the lowest tone to the highest.
    pub fn tone_spread(&self) -> f64 {
        self.spacing * (self.n_tones() - 1) as f64
    }
}

/// FAST is 4-FSK rather than 8-FSK on purpose. Eight tones only fit the usable
/// 18.0–19.5 kHz window at ~170 Hz spacing, and measured over the air that
/// spacing loses to reverb badly enough to fail whole frames. Four tones 400 Hz
/// apart measured at or near 0% symbol error while still running 64% faster
/// than NORMAL, because the symbols are shorter.
pub static PROFILES: [Profile; 3] = [
    Profile { name: "FAST",   spacing: 400.0, bits_per_symbol: 2, symbol_ms: 15.0, guard_ms: 5.0 },
    Profile { name: "NORMAL", spacing: 200.0, bits_per_symbol: 2, symbol_ms: 25.0, guard_ms: 8.0 },
    Profile { name: "ROBUST", spacing: 300.0, bits_per_symbol: 2, symbol_ms: 45.0, guard_ms: 15.0 },
];

pub const DEFAULT_PROFILE: usize = 1;

pub fn profile(id: usize) -> Option<&'static Profile> {
    PROFILES.get(id)
}

/// Ids in the order a receiver should try them, our own setting first.
pub fn profiles_preferring(first: usize) -> Vec<usize> {
    let mut ids: Vec<usize> = (0..PROFILES.len()).collect();
    if let Some(pos) = ids.iter().position(|&i| i == first) {
        ids.remove(pos);
        ids.insert(0, first);
    }
    ids
}

/// Widest base->top tone distance any profile needs — used when checking that a
/// retuned band still fits under Nyquist.
pub fn max_tone_spread() -> f64 {
    PROFILES
        .iter()
        .map(|p| p.tone_spread())
        .fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timing_matches_the_reference_implementation() {
        // These exact sample counts are what the Python produces; the two
        // implementations have to agree or they cannot decode each other.
        assert_eq!(PROFILES[0].symbol_samples(), 720);
        assert_eq!(PROFILES[0].guard_samples(), 240);
        assert_eq!(PROFILES[1].step_samples(), 1200 + 384);
        assert_eq!(PROFILES[2].symbol_samples(), 2160);
        assert_eq!(PROFILES[0].n_tones(), 4);
    }

    #[test]
    fn bitrates_are_the_documented_ones() {
        let r: Vec<i64> = PROFILES.iter().map(|p| p.bitrate().round() as i64).collect();
        assert_eq!(r, vec![100, 61, 33]);
    }

    #[test]
    fn preference_order_puts_our_own_profile_first() {
        assert_eq!(profiles_preferring(2), vec![2, 0, 1]);
        assert_eq!(profiles_preferring(0), vec![0, 1, 2]);
    }
}
