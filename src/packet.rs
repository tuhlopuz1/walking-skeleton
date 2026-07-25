//! Framing and whole-packet encode/decode.
//!
//! On the wire: `[CHIRP][gap][MARKER 2B][HEADER 4B][PAYLOAD][CRC16]`, with
//! everything after the gap Hamming(7,4)-coded and interleaved.
//!
//! The header carries the payload length *and* the profile id, so a receiver
//! adapts per frame: it decodes whatever profile the sender used, regardless of
//! its own setting.

use crate::band::Band;
use crate::fec::{
    crc16, deinterleave, hamming_bit_count, hamming_decode, hamming_encode, interleave,
    INTERLEAVE_BYTES,
};
use crate::mfsk::MfskModulator;
use crate::profile::{profile, Profile};
use crate::{ms_samples, GAP_MS, LEADIN_MS, TAIL_MS};

pub const PREAMBLE_MARKER: [u8; 2] = [0xAA, 0x55];
/// marker + (profile, len_hi, len_lo, header_crc)
pub const HEADER_BODY_BYTES: usize = PREAMBLE_MARKER.len() + 4;
/// Header body plus the frame CRC-16.
pub const FRAME_OVERHEAD_BYTES: usize = HEADER_BODY_BYTES + 2;

fn ceil_div(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

/// Frame body rounded up to a whole number of interleaver blocks.
pub fn padded_body_bytes(payload_len: usize) -> usize {
    let total = payload_len + FRAME_OVERHEAD_BYTES;
    total + (INTERLEAVE_BYTES - total % INTERLEAVE_BYTES) % INTERLEAVE_BYTES
}

pub fn header_symbols(p: &Profile) -> usize {
    ceil_div(hamming_bit_count(HEADER_BODY_BYTES), p.bits_per_symbol as usize)
}

pub fn frame_symbols(payload_len: usize, p: &Profile) -> usize {
    ceil_div(
        hamming_bit_count(padded_body_bytes(payload_len)),
        p.bits_per_symbol as usize,
    )
}

pub fn frame_samples(payload_len: usize, p: &Profile) -> usize {
    let n = frame_symbols(payload_len, p);
    if n == 0 {
        0
    } else {
        (n - 1) * p.step_samples() + p.symbol_samples()
    }
}

/// Wall-clock seconds one packet occupies, including lead-in and tail silence.
pub fn packet_duration_s(payload_len: usize, p: &Profile, band: &Band) -> f64 {
    let total = ms_samples(LEADIN_MS + GAP_MS + TAIL_MS)
        + band.chirp_samples()
        + frame_samples(payload_len, p);
    total as f64 / crate::SAMPLE_RATE as f64
}

/// Samples from the start of the chirp to the first data symbol.
pub fn data_offset_in_packet(band: &Band) -> usize {
    band.chirp_samples() + ms_samples(GAP_MS)
}

pub fn build_frame_bits(payload: &[u8], profile_id: usize) -> Vec<u8> {
    let length = payload.len();
    let mut header = vec![profile_id as u8, (length >> 8) as u8, (length & 0xFF) as u8];
    header.push((crc16(&header) & 0xFF) as u8);

    let mut body = Vec::with_capacity(HEADER_BODY_BYTES + length + 2);
    body.extend_from_slice(&PREAMBLE_MARKER);
    body.extend_from_slice(&header);
    body.extend_from_slice(payload);
    body.extend_from_slice(&crc16(&body).to_be_bytes());
    // Fill the last interleaver block, so the receiver can de-interleave it.
    while body.len() % INTERLEAVE_BYTES != 0 {
        body.push(0);
    }
    interleave(&hamming_encode(&body))
}

/// Full TX audio for one packet: silence, chirp preamble, modulated frame.
pub fn encode_packet(payload: &[u8], profile_id: usize, band: &Band) -> Vec<f32> {
    let p = *profile(profile_id).expect("unknown profile id");
    let m = MfskModulator::new(p, *band);
    let mut audio = vec![0.0f32; ms_samples(LEADIN_MS)];
    audio.extend(crate::chirp::make_chirp(band, 0.7));
    audio.extend(std::iter::repeat(0.0).take(ms_samples(GAP_MS)));
    audio.extend(m.bits_to_audio(&build_frame_bits(payload, profile_id)));
    audio.extend(std::iter::repeat(0.0).take(ms_samples(TAIL_MS)));

    let peak = audio.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let gain = if peak > 0.0 { 0.9 / peak } else { 1.0 };
    audio.iter_mut().for_each(|v| *v *= gain);
    audio
}

// --------------------------------------------------------------------------- //
// Decoding
// --------------------------------------------------------------------------- //
#[derive(Clone, Debug, PartialEq)]
pub struct DecodeResult {
    pub ok: bool,
    pub payload: Vec<u8>,
    pub profile_id: Option<usize>,
    pub reason: String,
    pub offset: isize,
    pub length: usize,
    pub confidence: f64,
}

impl DecodeResult {
    fn fail(reason: &str, offset: isize, confidence: f64) -> Self {
        Self {
            ok: false,
            payload: Vec::new(),
            profile_id: None,
            reason: reason.into(),
            offset,
            length: 0,
            confidence,
        }
    }
}

/// Try to read the 6-byte header body at an exact sample offset.
pub fn decode_header(
    audio: &[f32],
    start: isize,
    m: &MfskModulator,
    max_payload: usize,
) -> DecodeResult {
    let need_bits = hamming_bit_count(HEADER_BODY_BYTES);
    let (bits, conf) = m.audio_to_bits(audio, header_symbols(&m.profile), start);
    if bits.len() < need_bits {
        return DecodeResult::fail("short read", start, conf);
    }
    let body = hamming_decode(&deinterleave(&bits[..need_bits]));
    if body.len() < HEADER_BODY_BYTES || body[..2] != PREAMBLE_MARKER {
        return DecodeResult::fail("marker mismatch", start, conf);
    }
    let header = &body[2..6];
    if (crc16(&header[..3]) & 0xFF) as u8 != header[3] {
        return DecodeResult::fail("header CRC fail", start, conf);
    }
    let length = ((header[1] as usize) << 8) | header[2] as usize;
    if length > max_payload {
        return DecodeResult::fail("payload too large", start, conf);
    }
    let Some(pid) = profile(header[0] as usize).map(|_| header[0] as usize) else {
        return DecodeResult::fail("unknown profile id", start, conf);
    };
    DecodeResult {
        ok: true,
        payload: Vec::new(),
        profile_id: Some(pid),
        reason: String::new(),
        offset: start,
        length,
        confidence: conf,
    }
}

/// Decode the full frame once the header has said how long it is.
pub fn decode_frame(
    audio: &[f32],
    start: isize,
    m: &MfskModulator,
    length: usize,
) -> DecodeResult {
    let total_body = length + FRAME_OVERHEAD_BYTES;
    let (bits, conf) = m.audio_to_bits(audio, frame_symbols(length, &m.profile), start);
    let need = hamming_bit_count(padded_body_bytes(length));
    if bits.len() < need {
        return DecodeResult::fail("truncated frame", start, conf);
    }
    let mut body = hamming_decode(&deinterleave(&bits[..need]));
    if body.len() < total_body {
        return DecodeResult::fail("truncated frame", start, conf);
    }
    body.truncate(total_body);
    let want = u16::from_be_bytes([body[total_body - 2], body[total_body - 1]]);
    if crc16(&body[..total_body - 2]) != want {
        return DecodeResult::fail("frame CRC fail", start, conf);
    }
    DecodeResult {
        ok: true,
        payload: body[HEADER_BODY_BYTES..HEADER_BODY_BYTES + length].to_vec(),
        profile_id: None,
        reason: String::new(),
        offset: start,
        length,
        confidence: conf,
    }
}

/// Candidate fine-sync offsets, searched nearest-first around the chirp estimate.
///
/// The matched filter is sample-accurate on paper, but sound-card buffering and
/// resampling shift the data start by a millisecond or two. Symbols are tens of
/// ms long, so a handful of sub-symbol probes covers it.
pub fn sync_offsets(span_ms: f64, step_ms: f64) -> Vec<isize> {
    let step = ms_samples(step_ms).max(1) as isize;
    let k = (ms_samples(span_ms) as isize / step).max(1);
    let mut order = vec![0isize];
    for i in 1..=k {
        order.push(i * step);
        order.push(-i * step);
    }
    order
}

pub fn default_sync_offsets() -> Vec<isize> {
    sync_offsets(4.0, 0.75)
}

/// Locate and decode a frame whose data begins near `data_start`.
///
/// Returns a successful result, or the most informative failure — "marker
/// mismatch" tells you the timing is off, "frame CRC fail" tells you the timing
/// was right and the signal was not.
pub fn try_decode(
    audio: &[f32],
    profile_ids: &[usize],
    data_start: isize,
    band: &Band,
    max_payload: usize,
) -> DecodeResult {
    let mut best = DecodeResult::fail("no candidate", data_start, 0.0);
    for &pid in profile_ids {
        let Some(p) = profile(pid) else { continue };
        // One modulator per profile, reused across every offset: building the
        // correlation kernel is the expensive part and it does not vary with
        // the offset.
        let m = MfskModulator::new(*p, *band);
        for off in default_sync_offsets() {
            let start = data_start + off;
            if start < 0 {
                continue;
            }
            let hdr = decode_header(audio, start, &m, max_payload);
            if !hdr.ok {
                if hdr.reason != "short read" && best.reason == "no candidate" {
                    best = hdr;
                }
                continue;
            }
            let mut res = decode_frame(audio, start, &m, hdr.length);
            res.profile_id = hdr.profile_id;
            if res.ok {
                return res;
            }
            best = res;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::band::{AUDIO, ULTRA};
    use crate::chirp::PreambleDetector;
    use crate::profile::PROFILES;

    fn round_trip(payload: &[u8], pid: usize, band: Band) -> DecodeResult {
        let pkt = encode_packet(payload, pid, &band);
        let mut stream = vec![0.0f32; 3000];
        stream.extend_from_slice(&pkt);
        stream.extend(std::iter::repeat(0.0).take(3000));
        let det = PreambleDetector::with_default_threshold(band);
        let sc = det.scores(&stream);
        let peak = (0..sc.len()).max_by(|&a, &b| sc[a].total_cmp(&sc[b])).unwrap();
        try_decode(
            &stream,
            &[pid],
            (peak + data_offset_in_packet(&band)) as isize,
            &band,
            4096,
        )
    }

    #[test]
    fn every_profile_round_trips_in_every_band() {
        let payloads: Vec<Vec<u8>> = vec![
            b"".to_vec(),
            b"x".to_vec(),
            b"hello acoustic world".to_vec(),
            (0u8..=200).collect(),
        ];
        for band in [ULTRA, AUDIO] {
            for pid in 0..PROFILES.len() {
                for pl in &payloads {
                    let r = round_trip(pl, pid, band);
                    assert!(r.ok, "{}/{} failed: {}", band.name, PROFILES[pid].name, r.reason);
                    assert_eq!(&r.payload, pl, "{}/{}", band.name, PROFILES[pid].name);
                }
            }
        }
    }

    #[test]
    fn the_receiver_adapts_to_the_senders_profile() {
        // The decoder is told to try every profile; the header says which one
        // was really used, and that is what comes back.
        for pid in 0..PROFILES.len() {
            let band = ULTRA;
            let pkt = encode_packet(b"adapt me", pid, &band);
            let mut stream = vec![0.0f32; 3000];
            stream.extend_from_slice(&pkt);
            stream.extend(std::iter::repeat(0.0).take(3000));
            let det = PreambleDetector::with_default_threshold(band);
            let sc = det.scores(&stream);
            let peak = (0..sc.len()).max_by(|&a, &b| sc[a].total_cmp(&sc[b])).unwrap();
            let all: Vec<usize> = (0..PROFILES.len()).collect();
            let r = try_decode(
                &stream,
                &all,
                (peak + data_offset_in_packet(&band)) as isize,
                &band,
                4096,
            );
            assert!(r.ok, "{}: {}", PROFILES[pid].name, r.reason);
            assert_eq!(r.profile_id, Some(pid));
            assert_eq!(r.payload, b"adapt me");
        }
    }

    #[test]
    fn a_receiver_on_the_wrong_frequency_cannot_read_it() {
        // Retuning really is a channel, not a preference.
        let mut tuned = ULTRA;
        tuned.base_freq = 19_000.0;
        let pkt = encode_packet(b"secret", 1, &tuned);
        let mut stream = vec![0.0f32; 3000];
        stream.extend_from_slice(&pkt);
        stream.extend(std::iter::repeat(0.0).take(3000));
        // Decode at the correct sample position but with the old tuning.
        let start = (3000 + data_offset_in_packet(&ULTRA)) as isize;
        let r = try_decode(&stream, &[1], start, &ULTRA, 4096);
        assert!(!r.ok);
    }

    #[test]
    fn sizes_match_the_reference_implementation() {
        assert_eq!(HEADER_BODY_BYTES, 6);
        assert_eq!(FRAME_OVERHEAD_BYTES, 8);
        assert_eq!(padded_body_bytes(0), 12);
        assert_eq!(padded_body_bytes(48), 60);
        assert_eq!(header_symbols(&PROFILES[0]), 42);
        assert_eq!(default_sync_offsets().len(), 11);
        assert_eq!(default_sync_offsets()[..3], [0, 36, -36]);
    }

    #[test]
    fn packet_durations_match_the_documented_numbers() {
        let d = packet_duration_s(59, &PROFILES[0], &ULTRA);
        assert!((d - 10.34).abs() < 0.02, "FAST 59B was {d}");
        let d = packet_duration_s(59, &PROFILES[2], &ULTRA);
        assert!((d - 30.48).abs() < 0.02, "ROBUST 59B was {d}");
    }
}
