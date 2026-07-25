//! The streaming receiver, driven from a fake sound card.
//!
//! A tape of pre-rendered audio is fed to one `Transceiver` at a chosen speed.
//! This is where the receiver's harder properties get pinned down: that it
//! adapts to the sender's profile, survives a weak signal in noise, follows a
//! live retune, and invents nothing out of pure noise.

use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use acoustic_modem::audio::{spawn_capture, Backend, CaptureHandle};
use acoustic_modem::band::{Band, AUDIO, ULTRA};
use acoustic_modem::framing::{FragHeader, BROADCAST, T_TEXT};
use acoustic_modem::packet::encode_packet;
use acoustic_modem::profile::PROFILES;
use acoustic_modem::trx::{Event, Transceiver};
use acoustic_modem::SAMPLE_RATE;

/// Plays a fixed stream into the receiver and then silence.
struct Tape {
    samples: Mutex<Vec<f32>>,
    speed: f64,
}

impl Backend for Tape {
    fn play(&self, _samples: &[f32]) -> Result<(), String> {
        // The receiver under test never transmits; if it tries, that is a bug
        // the test should surface rather than silently absorb.
        Err("this test backend is receive-only".into())
    }

    fn start_capture(&self, sink: Sender<Vec<f32>>) -> Result<CaptureHandle, String> {
        let data = self.samples.lock().unwrap().clone();
        let speed = self.speed;
        spawn_capture(move |stop| {
            const BS: usize = 1024;
            let mut i = 0usize;
            while !stop() {
                let end = (i + BS).min(data.len());
                let block: Vec<f32> = if i < data.len() {
                    let mut b = data[i..end].to_vec();
                    b.resize(BS, 0.0);
                    b
                } else {
                    vec![0.0; BS]
                };
                i = end.max(i + BS);
                if sink.send(block).is_err() {
                    break;
                }
                std::thread::sleep(Duration::from_secs_f64(BS as f64 / SAMPLE_RATE as f64 / speed));
            }
        })
    }

    fn latency_hint(&self) -> f64 {
        0.05
    }
}

/// Build a broadcast text frame — deliberately not ack-requesting, so the
/// receive path is exercised without dragging the transmitter in.
fn frame(text: &str, pid: usize, idx: u16, total: u16, mid: u16, gain: f32, band: &Band) -> Vec<f32> {
    let hdr = FragHeader {
        kind: T_TEXT,
        ack_req: false,
        src: BROADCAST,
        dst: BROADCAST,
        msg_id: mid,
        frag_idx: idx,
        frag_total: total,
    };
    encode_packet(&hdr.encode(text.as_bytes()), pid, band)
        .into_iter()
        .map(|v| v * gain)
        .collect()
}

struct Outcome {
    texts: Vec<String>,
    ok: u64,
    bad: u64,
    detections: u64,
}

fn run(frames: Vec<Vec<f32>>, hint: usize, noise: f32, speed: f64, tune: Option<f64>) -> Outcome {
    let mut stream = vec![0.0f32; (SAMPLE_RATE as f64 * 0.3) as usize];
    for f in &frames {
        stream.extend_from_slice(f);
        stream.extend(std::iter::repeat(0.0).take((SAMPLE_RATE as f64 * 0.15) as usize));
    }
    stream.extend(std::iter::repeat(0.0).take((SAMPLE_RATE as f64 * 0.3) as usize));

    let mut seed = 0x9E3779B97F4A7C15u64;
    for v in stream.iter_mut() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        *v += ((seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * noise;
    }
    let audio_secs = stream.len() as f64 / SAMPLE_RATE as f64;

    let backend = Arc::new(Tape { samples: Mutex::new(stream), speed });
    let (trx, events): (Transceiver, Receiver<Event>) = Transceiver::new(backend, 0x1111);
    trx.update_config(|c| c.active_profile = hint);
    if let Some(hz) = tune {
        trx.set_frequency("ULTRA", hz).unwrap();
    }
    trx.start_rx().unwrap();

    let want = frames.len().max(1);
    let deadline = Instant::now() + Duration::from_secs_f64(audio_secs / speed + 6.0);
    let mut texts = Vec::new();
    while Instant::now() < deadline {
        while let Ok(ev) = events.try_recv() {
            if let Event::Text(t) = ev {
                texts.push(t);
            }
        }
        if texts.len() >= want {
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    std::thread::sleep(Duration::from_millis(150));
    while let Ok(ev) = events.try_recv() {
        if let Event::Text(t) = ev {
            texts.push(t);
        }
    }
    let s = trx.stats();
    trx.stop_rx();
    Outcome { texts, ok: s.rx_ok, bad: s.rx_bad, detections: s.detections }
}

#[test]
fn a_single_frame_arrives_in_every_profile() {
    for (pid, p) in PROFILES.iter().enumerate() {
        let o = run(vec![frame("hi there", pid, 0, 1, 1, 0.5, &ULTRA)], pid, 0.01, 30.0, None);
        assert_eq!(o.texts, vec!["hi there".to_string()], "{}", p.name);
        assert_eq!((o.ok, o.bad), (1, 0), "{} det={}", p.name, o.detections);
    }
}

#[test]
fn the_receiver_adapts_when_the_sender_used_another_profile() {
    // Configured for NORMAL, sent as FAST. The header says which, and that wins.
    let o = run(vec![frame("adapt me", 0, 0, 1, 1, 0.5, &ULTRA)], 1, 0.01, 30.0, None);
    assert_eq!(o.texts, vec!["adapt me".to_string()]);
}

#[test]
fn the_audible_band_is_a_peer_not_a_fallback() {
    // A receiver whose own setting says ULTRA still picks up an AUDIO packet:
    // it runs one matched filter per band.
    let o = run(vec![frame("audible band works", 1, 0, 1, 1, 0.5, &AUDIO)], 1, 0.01, 30.0, None);
    assert_eq!(o.texts, vec!["audible band works".to_string()]);
}

#[test]
fn a_weak_signal_in_louder_noise_still_decodes() {
    let o = run(vec![frame("quiet", 2, 0, 1, 1, 0.12, &ULTRA)], 2, 0.02, 30.0, None);
    assert_eq!(o.texts, vec!["quiet".to_string()], "det={} bad={}", o.detections, o.bad);
}

#[test]
fn a_multi_fragment_message_is_reassembled() {
    let body = "A".repeat(40) + &"B".repeat(30);
    let parts: Vec<String> = body
        .as_bytes()
        .chunks(acoustic_modem::framing::FRAG_PAYLOAD)
        .map(|c| String::from_utf8(c.to_vec()).unwrap())
        .collect();
    let total = parts.len() as u16;
    let frames: Vec<Vec<f32>> = parts
        .iter()
        .enumerate()
        .map(|(i, s)| frame(s, 0, i as u16, total, 5, 0.5, &ULTRA))
        .collect();
    let o = run(frames, 0, 0.01, 30.0, None);
    assert_eq!(o.texts, vec![body]);
    assert_eq!(o.detections, total as u64);
}

#[test]
fn a_live_receiver_follows_a_retune() {
    // Retuning must rebuild the matched filters, not leave them on the old spot.
    let mut tuned = ULTRA;
    tuned.base_freq = 16_000.0;
    let o = run(
        vec![frame("retuned link", 1, 0, 1, 1, 0.5, &tuned)],
        1,
        0.01,
        30.0,
        Some(16_000.0),
    );
    assert_eq!(o.texts, vec!["retuned link".to_string()]);
}

#[test]
fn pure_noise_produces_no_phantom_messages() {
    let o = run(Vec::new(), 1, 0.01, 30.0, None);
    assert!(o.texts.is_empty(), "invented {:?}", o.texts);
    assert_eq!(o.ok, 0);
}

#[test]
fn real_time_pacing_works_too() {
    // Everything above runs 30x faster than real time; check the loop is not
    // secretly relying on that.
    let o = run(vec![frame("realtime", 1, 0, 1, 1, 0.5, &ULTRA)], 1, 0.01, 1.0, None);
    assert_eq!(o.texts, vec!["realtime".to_string()]);
}
