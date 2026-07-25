//! CLI for the acoustic modem.
//!
//! The `encode`/`decode` subcommands read and write raw little-endian f32 mono
//! at 48 kHz — the same thing the DSP passes around internally. That is what
//! lets the Python reference and this implementation check each other on the
//! wire without either of them needing a sound card.

use std::io::{Read, Write};

use acoustic_modem::band::{band_plan, Band, BandSet, DEFAULT_BAND};
use acoustic_modem::chirp::PreambleDetector;
use acoustic_modem::packet::{data_offset_in_packet, encode_packet, packet_duration_s, try_decode};
use acoustic_modem::profile::{profile, DEFAULT_PROFILE, PROFILES};

fn usage() -> ! {
    eprintln!(
        "usage:
  modem plan     [--band NAME] [--freq kHz]
  modem selftest [--band NAME] [--freq kHz]
  modem encode <text> <out.f32> [--profile N] [--band NAME] [--freq kHz]
  modem decode <in.f32>         [--band NAME] [--freq kHz]

f32 files are raw little-endian mono at 48 kHz."
    );
    std::process::exit(2)
}

struct Opts {
    profile: usize,
    band: Band,
}

fn parse_opts(args: &[String]) -> Opts {
    let mut set = BandSet::default();
    let mut band_name = DEFAULT_BAND.to_string();
    let mut profile_id = DEFAULT_PROFILE;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--profile" => {
                profile_id = args
                    .get(i + 1)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| usage());
                i += 2;
            }
            "--band" => {
                band_name = args.get(i + 1).cloned().unwrap_or_else(|| usage());
                i += 2;
            }
            "--freq" => {
                let khz: f64 = args
                    .get(i + 1)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| usage());
                if let Err(e) = set.set_base_freq(&band_name, khz * 1000.0) {
                    eprintln!("bad --freq: {e}");
                    std::process::exit(2);
                }
                i += 2;
            }
            other => {
                eprintln!("unknown option {other}");
                usage()
            }
        }
    }
    if profile(profile_id).is_none() {
        eprintln!("unknown profile {profile_id}");
        std::process::exit(2);
    }
    let Some(band) = set.get(&band_name) else {
        eprintln!("unknown band {band_name}");
        std::process::exit(2);
    };
    Opts { profile: profile_id, band }
}

fn read_f32(path: &str) -> std::io::Result<Vec<f32>> {
    let mut raw = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut raw)?;
    Ok(raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn write_f32(path: &str, data: &[f32]) -> std::io::Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    for v in data {
        f.write_all(&v.to_le_bytes())?;
    }
    f.flush()
}

/// Locate the strongest preamble and decode whatever follows it.
fn locate_and_decode(audio: &[f32], band: &Band) -> (f32, usize, acoustic_modem::packet::DecodeResult) {
    let det = PreambleDetector::with_default_threshold(*band);
    let sc = det.scores(audio);
    let peak = (0..sc.len())
        .max_by(|&a, &b| sc[a].total_cmp(&sc[b]))
        .unwrap_or(0);
    let all: Vec<usize> = (0..PROFILES.len()).collect();
    let r = try_decode(
        audio,
        &all,
        (peak + data_offset_in_packet(band)) as isize,
        band,
        4096,
    );
    (sc.get(peak).copied().unwrap_or(0.0), peak, r)
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = argv.first().cloned() else { usage() };

    match cmd.as_str() {
        "plan" => {
            let o = parse_opts(&argv[1..]);
            for line in band_plan(&o.band, None) {
                println!("{line}");
            }
        }

        "selftest" => {
            let o = parse_opts(&argv[1..]);
            for line in band_plan(&o.band, None) {
                println!("{line}");
            }
            let payload: Vec<u8> = b"selftest ".iter().copied().chain(0u8..32).collect();
            let mut failures = 0;
            for (pid, p) in PROFILES.iter().enumerate() {
                let pkt = encode_packet(&payload, pid, &o.band);
                let mut stream = vec![0.0f32; 3000];
                stream.extend_from_slice(&pkt);
                stream.extend(std::iter::repeat(0.0).take(3000));
                let (score, _, r) = locate_and_decode(&stream, &o.band);
                let ok = r.ok && r.payload == payload;
                failures += !ok as i32;
                println!(
                    "  {:8} score={score:.2} {}",
                    p.name,
                    if ok { "PASS".to_string() } else { format!("FAIL {}", r.reason) }
                );
            }
            if failures > 0 {
                std::process::exit(1);
            }
        }

        "encode" => {
            if argv.len() < 3 {
                usage()
            }
            let (text, out) = (argv[1].clone(), argv[2].clone());
            let o = parse_opts(&argv[3..]);
            let audio = encode_packet(text.as_bytes(), o.profile, &o.band);
            let p = profile(o.profile).unwrap();
            if let Err(e) = write_f32(&out, &audio) {
                eprintln!("write failed: {e}");
                std::process::exit(1);
            }
            println!(
                "wrote {} samples ({:.2}s) profile {} {} band {} @ {:.3} kHz",
                audio.len(),
                packet_duration_s(text.len(), p, &o.band),
                o.profile,
                p.name,
                o.band.name,
                o.band.base_freq / 1000.0
            );
        }

        "decode" => {
            if argv.len() < 2 {
                usage()
            }
            let path = argv[1].clone();
            let o = parse_opts(&argv[2..]);
            let audio = match read_f32(&path) {
                Ok(a) => a,
                Err(e) => {
                    eprintln!("read failed: {e}");
                    std::process::exit(1);
                }
            };
            if audio.len() < o.band.chirp_samples() {
                eprintln!("audio too short");
                std::process::exit(1);
            }
            let (score, peak, r) = locate_and_decode(&audio, &o.band);
            println!("preamble score {score:.3} at sample {peak}");
            if r.ok {
                println!(
                    "OK profile={} conf={:.0} {} bytes",
                    r.profile_id.map(|p| PROFILES[p].name).unwrap_or("?"),
                    r.confidence,
                    r.payload.len()
                );
                println!("payload: {}", String::from_utf8_lossy(&r.payload));
            } else {
                println!("FAIL: {}", r.reason);
                std::process::exit(1);
            }
        }

        _ => usage(),
    }
}
