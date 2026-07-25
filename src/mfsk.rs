//! M-FSK modulator / demodulator.
//!
//! The demodulator correlates each symbol window against the *exact* tone
//! frequencies rather than picking the nearest FFT bin: short symbols have
//! coarse bins and the tones do not land on them.

use rustfft::num_complex::Complex;

use crate::band::{tone_freqs, Band};
use crate::chirp::hann;
use crate::profile::Profile;
use crate::SAMPLE_RATE;

pub struct MfskModulator {
    pub profile: Profile,
    pub band: Band,
    freqs: Vec<f64>,
    /// Row-major `(symbol_samples, n_tones)` correlation kernel, built once and
    /// reused across every fine-sync offset the receiver tries.
    kernel: Vec<Complex<f64>>,
}

impl MfskModulator {
    pub fn new(profile: Profile, band: Band) -> Self {
        let freqs = tone_freqs(&profile, &band);
        let n = profile.symbol_samples();
        let w = hann(n);
        let mut kernel = Vec::with_capacity(n * freqs.len());
        for k in 0..n {
            let t = k as f64 / SAMPLE_RATE as f64;
            for &f in &freqs {
                let ph = -2.0 * std::f64::consts::PI * t * f;
                kernel.push(Complex::new(ph.cos() * w[k], ph.sin() * w[k]));
            }
        }
        Self { profile, band, freqs, kernel }
    }

    pub fn freqs(&self) -> &[f64] {
        &self.freqs
    }

    // ---- TX ---------------------------------------------------------------- //
    fn tone(&self, freq: f64, n: usize) -> Vec<f32> {
        // Fade in and out to limit spectral splatter into the neighbouring tones.
        let fade = 96.min(n / 8);
        (0..n)
            .map(|i| {
                let t = i as f64 / SAMPLE_RATE as f64;
                let env = if fade > 1 && i < fade {
                    i as f64 / (fade - 1) as f64
                } else if fade > 1 && i >= n - fade {
                    (n - 1 - i) as f64 / (fade - 1) as f64
                } else {
                    1.0
                };
                (0.55 * (2.0 * std::f64::consts::PI * freq * t).sin() * env) as f32
            })
            .collect()
    }

    /// Pack bits into symbols (MSB first) and render them as audio.
    pub fn bits_to_audio(&self, bits: &[u8]) -> Vec<f32> {
        let bps = self.profile.bits_per_symbol as usize;
        let pad = (bps - bits.len() % bps) % bps;
        let total = bits.len() + pad;
        if total == 0 {
            return Vec::new();
        }
        let n = self.profile.symbol_samples();
        let guard = self.profile.guard_samples();

        // Each distinct tone is rendered once; a frame reuses them heavily.
        let rendered: Vec<Vec<f32>> = self.freqs.iter().map(|&f| self.tone(f, n)).collect();

        let mut out = Vec::with_capacity((total / bps) * (n + guard));
        for chunk in 0..total / bps {
            let mut sym = 0usize;
            for b in 0..bps {
                let idx = chunk * bps + b;
                let bit = if idx < bits.len() { bits[idx] } else { 0 };
                sym = (sym << 1) | (bit & 1) as usize;
            }
            out.extend_from_slice(&rendered[sym]);
            out.extend(std::iter::repeat(0.0).take(guard));
        }
        out
    }

    // ---- RX ---------------------------------------------------------------- //
    /// Returns `(symbols, confidence)`, where confidence is the mean ratio of
    /// the winning tone's energy to the runner-up's. It is the demodulator's own
    /// opinion of how clear the signal was, and it is what the framing layer
    /// reports back so a caller can see a marginal frame coming.
    pub fn demod(&self, audio: &[f32], start: isize, n_symbols: usize) -> (Vec<usize>, f64) {
        if n_symbols == 0 {
            return (Vec::new(), 0.0);
        }
        let sym_n = self.profile.symbol_samples();
        let step = self.profile.step_samples();
        let need = (n_symbols - 1) * step + sym_n;
        if start < 0 || start as usize + need > audio.len() {
            return (Vec::new(), 0.0);
        }
        let start = start as usize;
        let n_tones = self.freqs.len();

        let mut symbols = Vec::with_capacity(n_symbols);
        let mut conf_sum = 0.0;
        let mut acc = vec![Complex::new(0.0f64, 0.0); n_tones];
        for s in 0..n_symbols {
            let base = start + s * step;
            acc.iter_mut().for_each(|c| *c = Complex::new(0.0, 0.0));
            for k in 0..sym_n {
                let v = audio[base + k] as f64;
                let row = &self.kernel[k * n_tones..(k + 1) * n_tones];
                for (a, kv) in acc.iter_mut().zip(row) {
                    *a += kv * v;
                }
            }
            let mut energy: Vec<f64> = acc.iter().map(|c| c.norm()).collect();
            let winner = (0..n_tones)
                .max_by(|&a, &b| energy[a].total_cmp(&energy[b]))
                .unwrap();
            symbols.push(winner);
            energy.sort_by(f64::total_cmp);
            conf_sum += energy[n_tones - 1] / (energy[n_tones - 2] + 1e-12);
        }
        (symbols, conf_sum / n_symbols as f64)
    }

    /// Demodulate and unpack to bits, MSB first — the inverse of the packing in
    /// [`MfskModulator::bits_to_audio`].
    pub fn audio_to_bits(&self, audio: &[f32], n_symbols: usize, start: isize) -> (Vec<u8>, f64) {
        let (symbols, conf) = self.demod(audio, start, n_symbols);
        let bps = self.profile.bits_per_symbol as usize;
        let mut bits = Vec::with_capacity(symbols.len() * bps);
        for s in symbols {
            for shift in (0..bps).rev() {
                bits.push(((s >> shift) & 1) as u8);
            }
        }
        (bits, conf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::band::ULTRA;
    use crate::profile::PROFILES;

    #[test]
    fn bits_survive_a_clean_round_trip_in_every_profile() {
        for p in PROFILES.iter() {
            let m = MfskModulator::new(*p, ULTRA);
            let bits: Vec<u8> = (0..96).map(|i| ((i * 7 + i / 3) % 2) as u8).collect();
            let audio = m.bits_to_audio(&bits);
            let n_symbols = bits.len() / p.bits_per_symbol as usize;
            let (back, conf) = m.audio_to_bits(&audio, n_symbols, 0);
            assert_eq!(back, bits, "{} round trip", p.name);
            assert!(conf > 2.0, "{} confidence {conf} should be decisive", p.name);
        }
    }

    #[test]
    fn a_read_past_the_end_is_refused_rather_than_panicking() {
        let m = MfskModulator::new(PROFILES[0], ULTRA);
        let audio = m.bits_to_audio(&[1, 0, 1, 0]);
        let (bits, conf) = m.audio_to_bits(&audio, 999, 0);
        assert!(bits.is_empty());
        assert_eq!(conf, 0.0);
        let (bits, _) = m.audio_to_bits(&audio, 1, -5);
        assert!(bits.is_empty());
    }

    #[test]
    fn tones_land_where_the_band_says() {
        let m = MfskModulator::new(PROFILES[0], ULTRA);
        assert_eq!(m.freqs(), &[18_200.0, 18_600.0, 19_000.0, 19_400.0]);
    }
}
