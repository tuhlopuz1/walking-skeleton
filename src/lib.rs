//! Acoustic near-ultrasonic modem: M-FSK + Hamming(7,4) FEC + block interleaving
//! + CRC-16, with a chirp preamble located by a matched filter, and half-duplex
//! transport with stop-and-wait ARQ.
//!
//! Layering, lowest first:
//!
//! - [`fec`] -- CRC-16, Hamming(7,4), block interleaver.
//! - [`profile`] / [`band`] -- how to modulate, and where in the spectrum.
//! - [`chirp`] -- preamble generation and the normalized matched filter.
//! - [`mfsk`] -- the modulator/demodulator.
//! - [`packet`] -- framing, encoding, and decoding whole packets.
//!
//! Everything up to `packet` is pure DSP over `&[f32]` with no audio device in
//! sight, so it is testable without a sound card -- the property that made the
//! Python version debuggable, and worth keeping.

pub mod audio;
pub mod band;
pub mod chirp;
pub mod diag;
pub mod fec;
pub mod framing;
pub mod mfsk;
pub mod packet;
pub mod profile;
pub mod trx;

/// Fixed at 48 kHz: Nyquist at 24 kHz leaves room for the 18–20 kHz band.
pub const SAMPLE_RATE: u32 = 48_000;

// ---- packet layout timing (milliseconds) ---------------------------------- //
/// Silence before the chirp. Two jobs: sound cards clip the first few ms of
/// playback, and the receiver uses this quiet run-up to reject false preamble
/// hits (see [`chirp::PreambleDetector::lead_in_ok`]).
pub const LEADIN_MS: f64 = 120.0;
/// Chirp -> data gap, letting the chirp's own echo decay.
pub const GAP_MS: f64 = 20.0;
/// Silence after the frame.
pub const TAIL_MS: f64 = 60.0;
/// How much of the run-up the receiver inspects.
pub const LEAD_GUARD_MS: f64 = 50.0;

/// Detection threshold for the normalized matched filter.
///
/// A clean chirp scores ~0.7 and room noise ~0.02–0.05. The tempting move is to
/// put the threshold just above the noise, but the payload's own M-FSK tones sit
/// inside the chirp's sweep range and partially correlate with it: measured over
/// the air, the data region of a packet reaches 0.21–0.26. Every false lock on
/// the payload blinds the receiver for a header's worth of audio.
///
/// Rather than raise the threshold above 0.26 -- which would drop real chirps
/// that measure as low as 0.30 -- payload hits are rejected structurally by the
/// in-band lead-in silence check. That frees this to sit low enough not to miss
/// weak preambles while keeping a 4–10x margin over room noise.
pub const DETECT_THRESHOLD: f32 = 0.20;

/// Samples for a duration in milliseconds, truncated the way the reference
/// implementation truncates. Both ends must round identically or their sample
/// counts drift apart and nothing decodes.
pub fn ms_samples(ms: f64) -> usize {
    (SAMPLE_RATE as f64 * ms / 1000.0) as usize
}
