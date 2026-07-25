//! Bands -- WHERE in the spectrum a link lives.
//!
//! A band fixes the lowest data tone and the chirp preamble that finds and
//! time-aligns packets. Profiles say how to modulate and are band-agnostic, so
//! "switch to the audible band" and "change speed" stay independent knobs, and
//! either band can be retuned to any centre frequency.
//!
//! Unlike the Python original there is no global registry: a [`BandSet`] is
//! owned by whoever needs it. That keeps two links in one process independent,
//! which is what a UI driving several sessions needs.

use crate::profile::max_tone_spread;
use crate::SAMPLE_RATE;

/// Below this, most listeners can hear the tones.
pub const AUDIBLE_LIMIT: f64 = 17_000.0;
/// Keep the chirp clear of DC and room rumble.
pub const MIN_BAND_FREQ: f64 = 300.0;
/// Leave headroom under Nyquist.
pub const MAX_BAND_FREQ: f64 = 0.45 * SAMPLE_RATE as f64;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Band {
    pub name: &'static str,
    /// Lowest data tone.
    pub base_freq: f64,
    /// Hz below `base_freq` where the chirp starts.
    pub lead: f64,
    /// Hz above `base_freq` where the chirp ends.
    pub span: f64,
    pub chirp_ms: f64,
}

impl Band {
    pub fn chirp_f0(&self) -> f64 {
        self.base_freq - self.lead
    }

    pub fn chirp_f1(&self) -> f64 {
        self.base_freq + self.span
    }

    pub fn chirp_samples(&self) -> usize {
        (SAMPLE_RATE as f64 * self.chirp_ms / 1000.0) as usize
    }

    pub fn audible(&self) -> bool {
        self.base_freq < AUDIBLE_LIMIT
    }

    /// A signature that distinguishes chirps. Two bands retuned onto the same
    /// spot share one matched filter instead of double-firing on every packet.
    pub fn chirp_signature(&self) -> (i64, i64, i64) {
        (
            (self.chirp_f0() * 1000.0).round() as i64,
            (self.chirp_f1() * 1000.0).round() as i64,
            (self.chirp_ms * 1000.0).round() as i64,
        )
    }
}

/// Near-ultrasonic: inaudible to most people, but consumer speakers and mics
/// fall off a cliff above ~19.5 kHz. The chirp deliberately stops below that --
/// a sweep whose top half the hardware cannot reproduce only correlates on the
/// part that survives, which throws away detection margin.
pub const ULTRA: Band = Band { name: "ULTRA", base_freq: 18_200.0, lead: 400.0, span: 1400.0, chirp_ms: 60.0 };
/// Audible fallback: survives any speaker/mic pair, including Bluetooth.
pub const AUDIO: Band = Band { name: "AUDIO", base_freq: 3_000.0, lead: 200.0, span: 2400.0, chirp_ms: 60.0 };

pub const DEFAULT_BAND: &str = "ULTRA";

/// The bands one link knows about, plus a revision that bumps on every retune.
///
/// A running receiver watches the revision so it rebuilds its matched filters
/// instead of listening on the old frequency.
#[derive(Clone, Debug)]
pub struct BandSet {
    bands: Vec<Band>,
    revision: u64,
}

impl Default for BandSet {
    fn default() -> Self {
        Self { bands: vec![ULTRA, AUDIO], revision: 0 }
    }
}

impl BandSet {
    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn all(&self) -> &[Band] {
        &self.bands
    }

    pub fn get(&self, name: &str) -> Option<Band> {
        self.bands.iter().find(|b| b.name == name).copied()
    }

    /// Panics only on a name that is not in the set -- callers hold names that
    /// came from `all()` or from the `ULTRA`/`AUDIO` constants.
    pub fn expect(&self, name: &str) -> Band {
        self.get(name)
            .unwrap_or_else(|| panic!("unknown band {name:?}"))
    }

    /// Retune a band's lowest data tone. The chirp and every profile's tones
    /// move with it, so both ends must agree -- it is a channel, not a preference.
    pub fn set_base_freq(&mut self, name: &str, hz: f64) -> Result<Band, String> {
        check_band_freq(hz)?;
        let slot = self
            .bands
            .iter_mut()
            .find(|b| b.name == name)
            .ok_or_else(|| format!("unknown band {name:?}"))?;
        slot.base_freq = hz;
        let updated = *slot;
        self.revision += 1;
        Ok(updated)
    }

    /// One entry per *distinct* chirp, so co-tuned bands do not double-detect.
    pub fn distinct_chirps(&self) -> Vec<Band> {
        let mut seen = Vec::new();
        let mut out = Vec::new();
        for b in &self.bands {
            let sig = b.chirp_signature();
            if !seen.contains(&sig) {
                seen.push(sig);
                out.push(*b);
            }
        }
        out
    }
}

/// `Ok(())` if `hz` is a usable base frequency, else why it is not.
pub fn check_band_freq(hz: f64) -> Result<(), String> {
    if hz.is_nan() || hz <= 0.0 {
        return Err("frequency must be a positive number".into());
    }
    // Judge against the widest lead/span any band uses, so a frequency accepted
    // here stays legal whichever band is retuned to it.
    let lead = ULTRA.lead.max(AUDIO.lead);
    let span = ULTRA.span.max(AUDIO.span);
    let f0 = hz - lead;
    if f0 < MIN_BAND_FREQ {
        return Err(format!(
            "too low: the chirp would start at {f0:.0} Hz, below the {MIN_BAND_FREQ:.0} Hz floor"
        ));
    }
    let top = (hz + span).max(hz + max_tone_spread());
    if top > MAX_BAND_FREQ {
        return Err(format!(
            "too high: tones/chirp would reach {top:.0} Hz, above the {MAX_BAND_FREQ:.0} Hz \
             ceiling (Nyquist is {} Hz)",
            SAMPLE_RATE / 2
        ));
    }
    Ok(())
}

/// Human-readable description of where a band currently sits.
pub fn band_plan(band: &Band, only: Option<usize>) -> Vec<String> {
    use crate::profile::PROFILES;
    let mut out = vec![format!(
        "band {}: base {:.3} kHz, chirp {:.3}-{:.3} kHz ({})",
        band.name,
        band.base_freq / 1000.0,
        band.chirp_f0() / 1000.0,
        band.chirp_f1() / 1000.0,
        if band.audible() { "AUDIBLE" } else { "inaudible to most people" }
    )];
    for (pid, p) in PROFILES.iter().enumerate() {
        if only.is_some_and(|o| o != pid) {
            continue;
        }
        let tones: Vec<String> = tone_freqs(p, band)
            .iter()
            .map(|f| format!("{:.3}", f / 1000.0))
            .collect();
        out.push(format!(
            "  profile {pid} {:7} {}-FSK  tones {} kHz",
            p.name,
            p.n_tones(),
            tones.join(", ")
        ));
    }
    out
}

/// Absolute M-FSK tone frequencies for a profile in a given band.
pub fn tone_freqs(p: &crate::profile::Profile, band: &Band) -> Vec<f64> {
    (0..p.n_tones())
        .map(|i| band.base_freq + p.spacing * i as f64)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chirp_geometry_matches_the_reference() {
        assert_eq!(ULTRA.chirp_f0(), 17_800.0);
        assert_eq!(ULTRA.chirp_f1(), 19_600.0);
        assert_eq!(ULTRA.chirp_samples(), 2880);
        assert!(!ULTRA.audible());
        assert!(AUDIO.audible());
    }

    #[test]
    fn retuning_moves_the_chirp_and_bumps_the_revision() {
        let mut set = BandSet::default();
        assert_eq!(set.revision(), 0);
        let b = set.set_base_freq("ULTRA", 19_000.0).unwrap();
        assert_eq!(b.base_freq, 19_000.0);
        assert_eq!(b.chirp_f0(), 18_600.0);
        assert_eq!(b.chirp_f1(), 20_400.0);
        assert_eq!(set.revision(), 1);
        assert_eq!(set.expect("ULTRA").base_freq, 19_000.0);
        // The other band is untouched.
        assert_eq!(set.expect("AUDIO").base_freq, 3_000.0);
    }

    #[test]
    fn out_of_range_frequencies_are_refused() {
        assert!(check_band_freq(100.0).is_err(), "near DC");
        assert!(check_band_freq(23_000.0).is_err(), "above Nyquist headroom");
        assert!(check_band_freq(-5.0).is_err());
        assert!(check_band_freq(f64::NAN).is_err());
        assert!(check_band_freq(12_000.0).is_ok(), "a sane midband frequency");
    }

    #[test]
    fn co_tuned_bands_collapse_to_one_detector() {
        let mut set = BandSet::default();
        assert_eq!(set.distinct_chirps().len(), 2);
        set.set_base_freq("AUDIO", ULTRA.base_freq).unwrap();
        // Same base, but different lead/span, so still distinct chirps.
        assert_eq!(set.distinct_chirps().len(), 2);
    }

    #[test]
    fn a_band_tuned_low_reports_as_audible() {
        let mut set = BandSet::default();
        let b = set.set_base_freq("ULTRA", 4_000.0).unwrap();
        assert!(b.audible());
    }
}
