//! Diagnostics — these answer "is the link dead, or is my hardware dead?"
//!
//! All three run over the [`Backend`] trait, so they work against a real sound
//! card or a virtual room alike.

use std::sync::mpsc::channel;
use std::sync::Arc;
use std::time::Duration;

use crate::audio::Backend;
use crate::band::{Band, AUDIO};
use crate::chirp::PreambleDetector;
use crate::framing::{FragHeader, BROADCAST, T_TEXT};
use crate::packet::{data_offset_in_packet, encode_packet, try_decode};
use crate::profile::{profile, PROFILES};
use crate::SAMPLE_RATE;

/// Capture for the whole duration of `body`, plus a tail so the last of the
/// audio has time to come back through the microphone.
fn record_while<F: FnOnce()>(backend: &Arc<dyn Backend>, tail: Duration, body: F) -> Result<Vec<f32>, String> {
    let (tx, rx) = channel::<Vec<f32>>();
    let capture = backend.start_capture(tx)?;
    std::thread::sleep(Duration::from_millis(250)); // let the stream settle
    body();
    std::thread::sleep(tail);
    drop(capture);
    let mut out = Vec::new();
    while let Ok(block) = rx.try_recv() {
        out.extend_from_slice(&block);
    }
    Ok(out)
}

/// Encode in memory and decode it straight back. No sound card involved, so it
/// separates "is my configuration decodable" from "does my hardware carry it".
pub fn selftest(band: &Band) -> Vec<String> {
    let payload: Vec<u8> = b"selftest ".iter().copied().chain(0u8..32).collect();
    let mut out = crate::band::band_plan(band, None);
    for (pid, p) in PROFILES.iter().enumerate() {
        let mut stream = vec![0.0f32; 3000];
        stream.extend(encode_packet(&payload, pid, band));
        stream.extend(std::iter::repeat(0.0).take(3000));
        let det = PreambleDetector::with_default_threshold(*band);
        let sc = det.scores(&stream);
        let peak = (0..sc.len()).max_by(|&a, &b| sc[a].total_cmp(&sc[b])).unwrap_or(0);
        let r = try_decode(
            &stream,
            &[pid],
            (peak + data_offset_in_packet(band)) as isize,
            band,
            4096,
        );
        let ok = r.ok && r.payload == payload;
        out.push(format!(
            "  {:8} score={:.2} {}",
            p.name,
            sc.get(peak).copied().unwrap_or(0.0),
            if ok { "PASS".to_string() } else { format!("FAIL {}", r.reason) }
        ));
    }
    out
}

fn level_at(seg: &[f32], hz: f64) -> f64 {
    // A single-bin Goertzel over a Hann-windowed segment: cheaper than an FFT
    // and it lands exactly on the frequency, which an FFT bin would not.
    if seg.len() < 64 {
        return -120.0;
    }
    let n = seg.len();
    let w = crate::chirp::hann(n);
    let k = 2.0 * std::f64::consts::PI * hz / SAMPLE_RATE as f64;
    let (mut re, mut im) = (0.0f64, 0.0f64);
    for (i, (&v, wi)) in seg.iter().zip(&w).enumerate() {
        let ph = -k * i as f64;
        re += v as f64 * wi * ph.cos();
        im += v as f64 * wi * ph.sin();
    }
    20.0 * (re.hypot(im) + 1e-12).log10()
}

/// Play tones through the speaker and measure what the microphone hears.
///
/// The fastest way to find out whether a hardware pair can carry a band at all.
/// Many laptop mics roll off above ~18 kHz, and OS-level "enhancements" (noise
/// suppression, echo cancellation) delete a steady tone outright.
pub fn probe(backend: &Arc<dyn Backend>, band: &Band) -> Result<Vec<String>, String> {
    let mut freqs: Vec<f64> = vec![
        1000.0, 3000.0, 4000.0, 6000.0, 10000.0, 14000.0, 16000.0, 17000.0, 18200.0, 18600.0,
        19000.0, 19600.0, 20000.0,
    ];
    // Always include what this band is currently tuned to, so the answer stays
    // meaningful after a retune.
    let mut tuned: Vec<f64> = PROFILES
        .iter()
        .flat_map(|p| crate::band::tone_freqs(p, band))
        .collect();
    tuned.push(band.chirp_f0());
    tuned.push(band.chirp_f1());
    freqs.extend(tuned.iter().copied());
    freqs.sort_by(f64::total_cmp);
    freqs.dedup_by(|a, b| (*a - *b).abs() < 1.0);

    let tone_n = (SAMPLE_RATE as f64 * 0.30) as usize;
    let pause_n = (SAMPLE_RATE as f64 * 0.08) as usize;
    let step = tone_n + pause_n;

    // A known chirp goes first so the recording can be aligned: play and capture
    // are separate streams here, and their relative delay is unknown.
    let marker = crate::chirp::make_chirp(&AUDIO, 0.9);
    let mut sig = marker.clone();
    sig.extend(std::iter::repeat(0.0).take(pause_n));
    let align_len = sig.len();
    for &f in &freqs {
        for i in 0..tone_n {
            let t = i as f64 / SAMPLE_RATE as f64;
            let env = if i < 480 {
                i as f64 / 480.0
            } else if i >= tone_n - 480 {
                (tone_n - i) as f64 / 480.0
            } else {
                1.0
            };
            sig.push((0.5 * (2.0 * std::f64::consts::PI * f * t).sin() * env) as f32);
        }
        sig.extend(std::iter::repeat(0.0).take(pause_n));
    }

    let b = backend.clone();
    let sig2 = sig.clone();
    let rec = record_while(backend, Duration::from_millis(600), move || {
        let _ = b.play(&sig2);
    })?;
    if rec.len() < align_len {
        return Ok(vec!["probe failed: nothing recorded".into()]);
    }

    let det = PreambleDetector::new(AUDIO, 0.0);
    let sc = det.scores(&rec);
    let anchor = (0..sc.len())
        .max_by(|&a, &b| sc[a].total_cmp(&sc[b]))
        .unwrap_or(0);
    let base = anchor + align_len;

    let mut levels = Vec::new();
    for (i, &f) in freqs.iter().enumerate() {
        let s = base + i * step;
        let e = (s + tone_n).min(rec.len());
        if s >= rec.len() || e <= s {
            break;
        }
        levels.push((f, level_at(&rec[s..e], f)));
    }
    if levels.is_empty() {
        return Ok(vec!["probe failed: recording too short to align".into()]);
    }

    // Bars are relative to the loudest tone: absolute dBFS depends on the volume
    // knob, while the SHAPE of the response is what decides the band.
    let top = levels.iter().map(|l| l.1).fold(f64::MIN, f64::max);
    let mut out = vec![format!(
        "(alignment score {:.2}; bars are relative to the strongest tone, {top:.1} dB; \
         '<' marks the tuned band)",
        sc.get(anchor).copied().unwrap_or(0.0)
    )];
    for (f, db) in &levels {
        let mark = if tuned.iter().any(|t| (t - f).abs() < 1.0) { " <" } else { "" };
        let bars = (44.0 + (db - top)).clamp(0.0, 44.0) as usize;
        out.push(format!("{f:8.0} Hz  {:6.1}  {}{mark}", db - top, "#".repeat(bars)));
    }

    let reference = levels
        .iter()
        .filter(|(f, _)| (999.0..4001.0).contains(f))
        .map(|l| l.1)
        .fold(f64::MIN, f64::max);
    let here = levels
        .iter()
        .filter(|(f, _)| tuned.iter().any(|t| (t - f).abs() < 1.0))
        .map(|l| l.1)
        .fold(f64::MAX, f64::min);
    out.push(String::new());
    if here != f64::MAX && reference != f64::MIN {
        let drop = reference - here;
        out.push(format!(
            "VERDICT for {} @ {:.2} kHz: weakest tone is {drop:.0} dB below the audible reference.",
            band.name,
            band.base_freq / 1000.0
        ));
        if drop > 30.0 {
            out.push("         Too weak — this hardware will not carry that tuning.".into());
        }
    }
    out.push(
        "NOTE: Windows quietens playback while a mic is open ('communications activity')."
            .into(),
    );
    out.push("      Sound Control Panel -> Communications -> 'Do nothing' disables it.".into());
    Ok(out)
}

/// Full round trip on one machine: speaker -> air -> mic -> decoder.
///
/// Worth knowing: on the audible band this can under-perform on a laptop that
/// plays and records at once, because the OS echo canceller actively subtracts
/// the speaker signal from the mic. That is a same-machine artifact and does not
/// happen between two separate devices.
pub fn loopback(
    backend: &Arc<dyn Backend>,
    profile_id: usize,
    text: &str,
    band: &Band,
) -> Result<Vec<String>, String> {
    let p = *profile(profile_id).ok_or("unknown profile")?;
    let hdr = FragHeader {
        kind: T_TEXT,
        ack_req: false,
        src: BROADCAST,
        dst: BROADCAST,
        msg_id: 1,
        frag_idx: 0,
        frag_total: 1,
    };
    let pkt = encode_packet(&hdr.encode(text.as_bytes()), profile_id, band);

    let b = backend.clone();
    let rec = record_while(backend, Duration::from_millis(800), move || {
        let _ = b.play(&pkt);
    })?;
    if rec.len() < band.chirp_samples() {
        return Ok(vec!["loopback: recording too short".into()]);
    }

    let rms = (rec.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / rec.len() as f64).sqrt();
    let det = PreambleDetector::with_default_threshold(*band);
    let sc = det.scores(&rec);
    let peak = (0..sc.len()).max_by(|&a, &b| sc[a].total_cmp(&sc[b])).unwrap_or(0);
    let score = sc.get(peak).copied().unwrap_or(0.0);
    let mut sorted = sc.clone();
    sorted.sort_by(f32::total_cmp);
    let noise = sorted.get(sorted.len() / 2).copied().unwrap_or(0.0);

    let mut out = vec![
        format!("recorded rms={:.1} dBFS ({} profile)", 20.0 * (rms + 1e-9).log10(), p.name),
        format!(
            "best preamble score={score:.3} (need >= {:.2}), noise median={noise:.3}",
            det.threshold
        ),
    ];
    if score < det.threshold {
        out.push("FAIL: preamble not detected — the mic never heard the chirp.".into());
        out.push("      Raise the volume, or run `probe` to check the band.".into());
        return Ok(out);
    }
    let r = try_decode(
        &rec,
        &[profile_id],
        (peak + data_offset_in_packet(band)) as isize,
        band,
        4096,
    );
    if r.ok {
        // Read it back through the real framing layer rather than slicing by a
        // hardcoded offset: if the two ever disagree, this should say so.
        let got = match crate::framing::parse(&r.payload) {
            Some(crate::framing::Frame::Data { payload, .. }) => {
                String::from_utf8_lossy(&payload).into_owned()
            }
            _ => "<unparseable frame>".into(),
        };
        out.push(format!("PASS: decoded {got:?} (conf={:.0})", r.confidence));
    } else {
        out.push(format!("FAIL at decode: {}", r.reason));
        out.push(
            "      Preamble was heard, so timing/SNR is marginal — try a slower profile \
             or move the devices closer."
                .into(),
        );
    }
    Ok(out)
}
