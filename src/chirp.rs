//! Chirp preamble generation and the streaming normalized matched filter.
//!
//! A 60 ms linear sweep is matched-filtered against the incoming stream with a
//! *complex* (analytic) template, so detection is insensitive to the phase
//! rotation the room and hardware apply. A real-valued template loses most of
//! its peak when the recorded copy comes back phase-shifted.

use std::cell::RefCell;
use std::collections::HashMap;

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use crate::band::Band;
use crate::{ms_samples, DETECT_THRESHOLD, LEAD_GUARD_MS, SAMPLE_RATE};

/// Symmetric Hann window, matching `numpy.hanning` exactly.
///
/// The denominator is `n - 1`, not `n`. Getting this wrong shifts the chirp's
/// envelope and costs correlation peak against a peer that got it right.
pub fn hann(n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![1.0];
    }
    (0..n)
        .map(|k| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * k as f64 / (n - 1) as f64).cos())
        .collect()
}

fn chirp_phase(band: &Band) -> Vec<f64> {
    let n = band.chirp_samples();
    let period = n as f64 / SAMPLE_RATE as f64;
    let k = (band.chirp_f1() - band.chirp_f0()) / period;
    (0..n)
        .map(|i| {
            let t = i as f64 / SAMPLE_RATE as f64;
            2.0 * std::f64::consts::PI * (band.chirp_f0() * t + 0.5 * k * t * t)
        })
        .collect()
}

/// The real linear-sweep chirp that goes on the wire as a packet preamble.
pub fn make_chirp(band: &Band, amplitude: f64) -> Vec<f32> {
    let w = hann(band.chirp_samples());
    chirp_phase(band)
        .iter()
        .zip(&w)
        .map(|(p, wi)| (amplitude * p.sin() * wi) as f32)
        .collect()
}

/// The complex template the matched filter correlates against.
pub fn make_chirp_analytic(band: &Band) -> Vec<Complex<f64>> {
    let w = hann(band.chirp_samples());
    chirp_phase(band)
        .iter()
        .zip(&w)
        .map(|(p, wi)| Complex::new(p.cos() * wi, p.sin() * wi))
        .collect()
}

/// Streaming normalized matched filter for one band's chirp.
///
/// `scores(x)[i]` is the normalized correlation of `x[i..i+M]` with the chirp,
/// in `[0, 1]`. A clean recorded chirp peaks around 0.7; broadband noise sits
/// near the reciprocal square root of the time-bandwidth product.
pub struct PreambleDetector {
    pub band: Band,
    pub threshold: f32,
    tpl: Vec<Complex<f64>>,
    m: usize,
    tpl_norm: f64,
    floor_ss: f64,
    planner: RefCell<FftPlanner<f64>>,
    tpl_fft: RefCell<HashMap<usize, Vec<Complex<f64>>>>,
}

impl PreambleDetector {
    /// Windows quieter than this RMS are treated as silence rather than
    /// normalized -- otherwise 0/0 turns a digitally silent gap into a fake peak.
    pub const SILENCE_RMS: f64 = 1e-4;

    pub fn new(band: Band, threshold: f32) -> Self {
        let tpl = make_chirp_analytic(&band);
        let m = tpl.len();
        let tpl_norm = tpl.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        Self {
            band,
            threshold,
            tpl,
            m,
            tpl_norm,
            floor_ss: Self::SILENCE_RMS * Self::SILENCE_RMS * m as f64,
            planner: RefCell::new(FftPlanner::new()),
            tpl_fft: RefCell::new(HashMap::new()),
        }
    }

    pub fn with_default_threshold(band: Band) -> Self {
        Self::new(band, DETECT_THRESHOLD)
    }

    /// Template length in samples: the shortest window that can hold a chirp.
    pub fn m(&self) -> usize {
        self.m
    }

    /// Conjugated spectrum of the template, zero-padded to `nfft`. Cached
    /// because a live receiver calls `scores` on every incoming block.
    fn template_spectrum(&self, nfft: usize) -> Vec<Complex<f64>> {
        if let Some(v) = self.tpl_fft.borrow().get(&nfft) {
            return v.clone();
        }
        let fft = self.planner.borrow_mut().plan_fft_forward(nfft);
        let mut buf = vec![Complex::new(0.0, 0.0); nfft];
        buf[..self.m].copy_from_slice(&self.tpl);
        fft.process(&mut buf);
        for c in buf.iter_mut() {
            *c = c.conj();
        }
        self.tpl_fft.borrow_mut().insert(nfft, buf.clone());
        buf
    }

    pub fn scores(&self, x: &[f32]) -> Vec<f32> {
        let n = x.len();
        if n < self.m {
            return Vec::new();
        }
        let l = n - self.m + 1;
        let nfft = (n + self.m).next_power_of_two();

        // Cross-correlation by FFT: ifft(fft(x) * conj(fft(tpl)))[i] is exactly
        // sum_k x[i+k] * conj(tpl[k]) as long as nfft >= n + m, which keeps the
        // circular wrap out of the first `l` outputs.
        let mut buf = vec![Complex::new(0.0, 0.0); nfft];
        for (b, &v) in buf.iter_mut().zip(x) {
            b.re = v as f64;
        }
        let fwd = self.planner.borrow_mut().plan_fft_forward(nfft);
        fwd.process(&mut buf);
        let spec = self.template_spectrum(nfft);
        for (b, s) in buf.iter_mut().zip(&spec) {
            *b *= *s;
        }
        let inv = self.planner.borrow_mut().plan_fft_inverse(nfft);
        inv.process(&mut buf);
        let scale = 1.0 / nfft as f64;

        // Local energy of every candidate window, via a prefix sum.
        let mut prefix = Vec::with_capacity(n + 1);
        prefix.push(0.0f64);
        let mut acc = 0.0f64;
        for &v in x {
            acc += (v as f64) * (v as f64);
            prefix.push(acc);
        }

        (0..l)
            .map(|i| {
                let energy = (prefix[i + self.m] - prefix[i]).max(self.floor_ss);
                ((buf[i].norm() * scale) / (energy.sqrt() * self.tpl_norm)) as f32
            })
            .collect()
    }

    /// Indices of distinct local maxima above the threshold, strongest first.
    ///
    /// Peaks within one template length of each other are one event -- a chirp
    /// correlates over its whole width, so the raw score has a broad hump.
    pub fn peaks(&self, sc: &[f32]) -> Vec<usize> {
        let above: Vec<usize> = sc
            .iter()
            .enumerate()
            .filter(|(_, &v)| v >= self.threshold)
            .map(|(i, _)| i)
            .collect();
        let mut out = Vec::new();
        let mut k = 0;
        while k < above.len() {
            let lo = above[k];
            let mut hi = lo;
            while k < above.len() && above[k] <= hi + self.m {
                hi = above[k];
                k += 1;
            }
            let peak = (lo..=hi)
                .max_by(|&a, &b| sc[a].total_cmp(&sc[b]))
                .unwrap_or(lo);
            out.push(peak);
        }
        out.sort_by(|&a, &b| sc[b].total_cmp(&sc[a]));
        out
    }

    /// RMS of `seg` restricted to this band, comparable across window sizes.
    pub fn band_rms(&self, seg: &[f32]) -> f64 {
        let n = seg.len();
        if n < 64 {
            return 0.0;
        }
        let w = hann(n);
        let mut buf: Vec<Complex<f64>> = seg
            .iter()
            .zip(&w)
            .map(|(&v, wi)| Complex::new(v as f64 * wi, 0.0))
            .collect();
        let fft = self.planner.borrow_mut().plan_fft_forward(n);
        fft.process(&mut buf);
        let bin_hz = SAMPLE_RATE as f64 / n as f64;
        let mut sum = 0.0;
        // Only the non-negative frequencies, mirroring numpy's rfft.
        for (i, c) in buf.iter().take(n / 2 + 1).enumerate() {
            let f = i as f64 * bin_hz;
            if f >= self.band.chirp_f0() && f <= self.band.chirp_f1() {
                sum += c.norm_sqr();
            }
        }
        (2.0 * sum).sqrt() / n as f64
    }

    /// True if the audio just before index `i` is quiet enough to be a lead-in.
    ///
    /// Every packet starts with silence, so a genuine chirp always has a quiet
    /// run-up. A false peak found inside a packet's own payload is preceded by
    /// full-volume data tones and fails here. This is what lets the threshold
    /// stay low without turning into a storm of self-detections.
    ///
    /// The comparison is in-band on purpose. Broadband RMS does not work: a
    /// near-ultrasonic chirp returns from the mic far weaker than ordinary room
    /// rumble, so a broadband test throws away perfectly good preambles.
    pub fn lead_in_ok(&self, x: &[f32], i: usize, ratio: f64) -> bool {
        let g = ms_samples(LEAD_GUARD_MS);
        if i < g || i + self.m > x.len() {
            return true; // not enough history to judge
        }
        self.band_rms(&x[i - g..i]) <= ratio * self.band_rms(&x[i..i + self.m])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::band::{AUDIO, ULTRA};

    fn locate(band: Band) -> (usize, f32, f32) {
        let det = PreambleDetector::with_default_threshold(band);
        let pad = 3000usize;
        let mut stream = vec![0.0f32; pad];
        stream.extend(make_chirp(&band, 0.7));
        stream.extend(std::iter::repeat(0.0).take(pad));
        // A little noise, so the normalization is exercised rather than
        // dividing a clean signal by a clean floor.
        let mut seed = 12345u64;
        for v in stream.iter_mut() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v += ((seed >> 33) as f64 / (1u64 << 31) as f64 - 1.0) as f32 * 0.01;
        }
        let sc = det.scores(&stream);
        let peak = (0..sc.len()).max_by(|&a, &b| sc[a].total_cmp(&sc[b])).unwrap();
        let mut sorted = sc.clone();
        sorted.sort_by(f32::total_cmp);
        (peak, sc[peak], sorted[sorted.len() / 2])
    }

    #[test]
    fn locks_onto_the_chirp_sample_accurately() {
        for band in [ULTRA, AUDIO] {
            let (peak, score, noise) = locate(band);
            assert_eq!(peak, 3000, "{}: chirp starts at sample 3000", band.name);
            assert!(score > 0.6, "{}: peak {score} should be ~0.7", band.name);
            assert!(noise < 0.1, "{}: noise floor {noise} too high", band.name);
        }
    }

    #[test]
    fn digital_silence_produces_no_detection() {
        let det = PreambleDetector::with_default_threshold(ULTRA);
        let sc = det.scores(&vec![0.0f32; 20_000]);
        assert!(det.peaks(&sc).is_empty());
        assert!(sc.iter().all(|&v| v < det.threshold));
    }

    #[test]
    fn hann_matches_numpy() {
        assert_eq!(hann(1), vec![1.0]);
        let w = hann(5);
        assert!((w[0] - 0.0).abs() < 1e-12);
        assert!((w[2] - 1.0).abs() < 1e-12);
        assert!((w[4] - 0.0).abs() < 1e-12);
        // symmetric
        assert!((w[1] - w[3]).abs() < 1e-12);
    }

    #[test]
    fn payload_tones_are_rejected_by_the_lead_in_check() {
        // A steady in-band tone should not be accepted as a preamble even if it
        // scrapes past the threshold: it has no quiet run-up.
        let det = PreambleDetector::with_default_threshold(ULTRA);
        let n = 20_000;
        let tone: Vec<f32> = (0..n)
            .map(|i| {
                (0.5 * (2.0 * std::f64::consts::PI * 18_600.0 * i as f64
                    / SAMPLE_RATE as f64)
                    .sin()) as f32
            })
            .collect();
        let mid = n / 2;
        assert!(!det.lead_in_ok(&tone, mid, 0.5));
    }
}
