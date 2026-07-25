//! CLI for the acoustic modem.
//!
//! The `encode`/`decode` subcommands read and write raw little-endian f32 mono
//! at 48 kHz -- the same thing the DSP passes around internally. That is what
//! lets the Python reference and this implementation check each other on the
//! wire without either of them needing a sound card.

use std::io::{Read, Write};
use std::sync::Arc;

use acoustic_modem::audio::{list_devices, Backend, CpalBackend};
use acoustic_modem::band::{band_plan, Band, BandSet, DEFAULT_BAND};
use acoustic_modem::chirp::PreambleDetector;
use acoustic_modem::diag;
use acoustic_modem::packet::{data_offset_in_packet, encode_packet, packet_duration_s, try_decode};
use acoustic_modem::profile::{profile, DEFAULT_PROFILE, PROFILES};
use acoustic_modem::trx::{Event, Level, Transceiver};

fn usage() -> ! {
    eprintln!(
        "usage:
  modem devices
  modem plan     [opts]
  modem selftest [opts]                      encode->decode in memory
  modem probe    [opts]                      what this speaker+mic pair carries
  modem loopback [opts]                      speaker -> air -> mic round trip
  modem listen   [opts] [--id HHHH]          receive until Ctrl-C
  modem send <text> [opts] [--id HHHH] [--no-arq]
  modem encode <text> <out.f32> [opts]
  modem decode <in.f32>         [opts]

opts: [--profile N] [--band NAME] [--freq kHz] [--in NAME] [--out NAME]
f32 files are raw little-endian mono at 48 kHz."
    );
    std::process::exit(2)
}

struct Opts {
    profile: usize,
    band: Band,
    band_name: String,
    freq_hz: Option<f64>,
    input: Option<String>,
    output: Option<String>,
    device_id: u16,
    arq: bool,
}

fn parse_opts(args: &[String]) -> Opts {
    let mut set = BandSet::default();
    let mut band_name = DEFAULT_BAND.to_string();
    let mut profile_id = DEFAULT_PROFILE;
    let mut freq_hz = None;
    let mut input = None;
    let mut output = None;
    // A random-but-stable-per-run id keeps two processes on one machine distinct
    // without a config file; `--id` pins it when that matters.
    let mut device_id: u16 = (std::process::id() as u16) | 1;
    let mut arq = true;

    let mut i = 0;
    let want = |args: &[String], i: usize| -> String {
        args.get(i + 1).cloned().unwrap_or_else(|| usage())
    };
    while i < args.len() {
        match args[i].as_str() {
            "--profile" => {
                profile_id = want(args, i).parse().unwrap_or_else(|_| usage());
                i += 2;
            }
            "--band" => {
                band_name = want(args, i);
                i += 2;
            }
            "--freq" => {
                let khz: f64 = want(args, i).parse().unwrap_or_else(|_| usage());
                freq_hz = Some(khz * 1000.0);
                i += 2;
            }
            "--in" => {
                input = Some(want(args, i));
                i += 2;
            }
            "--out" => {
                output = Some(want(args, i));
                i += 2;
            }
            "--id" => {
                device_id = u16::from_str_radix(&want(args, i), 16).unwrap_or_else(|_| usage());
                i += 2;
            }
            "--no-arq" => {
                arq = false;
                i += 1;
            }
            other => {
                eprintln!("unknown option {other}");
                usage()
            }
        }
    }
    if let Some(hz) = freq_hz {
        if let Err(e) = set.set_base_freq(&band_name, hz) {
            eprintln!("bad --freq: {e}");
            std::process::exit(2);
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
    if device_id == 0 {
        device_id = 1;
    }
    Opts { profile: profile_id, band, band_name, freq_hz, input, output, device_id, arq }
}

/// A live transceiver configured from the command line, with its receiver up.
fn live(o: &Opts) -> (Transceiver, std::sync::mpsc::Receiver<Event>) {
    let backend: Arc<dyn Backend> =
        Arc::new(CpalBackend::new(o.input.clone(), o.output.clone()));
    let (trx, events) = Transceiver::new(backend, o.device_id);
    trx.update_config(|c| {
        c.active_profile = o.profile;
        c.active_band = o.band_name.clone();
        c.arq = o.arq;
    });
    if let Some(hz) = o.freq_hz {
        if let Err(e) = trx.set_frequency(&o.band_name, hz) {
            eprintln!("retune failed: {e}");
            std::process::exit(1);
        }
    }
    (trx, events)
}

/// Prints events, collapsing the progress stream.
///
/// A receiver emits progress on every audio block, which for a 30-second ROBUST
/// frame is hundreds of lines saying almost the same thing. Only report when it
/// has actually moved.
#[derive(Default)]
struct Printer {
    last_pct: Option<u8>,
    last_note: String,
}

impl Printer {
    fn print(&mut self, ev: &Event) {
        match ev {
            Event::Progress { pct, note } => {
                let moved = self.last_pct.is_none_or(|p| pct.abs_diff(p) >= 10 || *pct == 100);
                if moved || *note != self.last_note {
                    println!("       {note} [{pct}%]");
                    self.last_pct = Some(*pct);
                    self.last_note = note.clone();
                }
            }
            other => {
                self.last_pct = None;
                print_event(other);
            }
        }
    }
}

fn print_event(ev: &Event) {
    match ev {
        Event::Log(level, text) => {
            let tag = match level {
                Level::Info => "info",
                Level::Warn => "WARN",
                Level::Error => "ERR ",
            };
            println!("[{tag}] {text}");
        }
        Event::Progress { pct, note } => println!("       {note} [{pct}%]"),
        Event::Text(t) => println!("<< TEXT: {t}"),
        Event::File { meta, data } => {
            let dir = std::path::Path::new("received");
            let _ = std::fs::create_dir_all(dir);
            let name = std::path::Path::new(&meta.name)
                .file_name()
                .map(|s| s.to_owned())
                .unwrap_or_else(|| "received.bin".into());
            let path = dir.join(name);
            match std::fs::write(&path, data) {
                Ok(()) => println!("<< FILE: {} ({} B) -> {}", meta.name, data.len(), path.display()),
                Err(e) => println!("<< FILE: {} could not be saved: {e}", meta.name),
            }
        }
        Event::PeerChanged(id) => println!("       peer is now {id:04X}"),
    }
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
        "devices" => {
            for line in list_devices() {
                println!("{line}");
            }
        }

        "plan" => {
            let o = parse_opts(&argv[1..]);
            for line in band_plan(&o.band, None) {
                println!("{line}");
            }
        }

        "selftest" => {
            let o = parse_opts(&argv[1..]);
            let lines = diag::selftest(&o.band);
            let failed = lines.iter().any(|l| l.contains("FAIL"));
            for line in lines {
                println!("{line}");
            }
            if failed {
                std::process::exit(1);
            }
        }

        "probe" => {
            let o = parse_opts(&argv[1..]);
            let backend: Arc<dyn Backend> =
                Arc::new(CpalBackend::new(o.input.clone(), o.output.clone()));
            match diag::probe(&backend, &o.band) {
                Ok(lines) => lines.iter().for_each(|l| println!("{l}")),
                Err(e) => {
                    eprintln!("probe failed: {e}");
                    std::process::exit(1);
                }
            }
        }

        "loopback" => {
            let o = parse_opts(&argv[1..]);
            let backend: Arc<dyn Backend> =
                Arc::new(CpalBackend::new(o.input.clone(), o.output.clone()));
            match diag::loopback(&backend, o.profile, "loopback test", &o.band) {
                Ok(lines) => {
                    let failed = lines.iter().any(|l| l.contains("FAIL"));
                    lines.iter().for_each(|l| println!("{l}"));
                    if failed {
                        std::process::exit(1);
                    }
                }
                Err(e) => {
                    eprintln!("loopback failed: {e}");
                    std::process::exit(1);
                }
            }
        }

        "listen" => {
            let o = parse_opts(&argv[1..]);
            let (trx, events) = live(&o);
            if let Err(e) = trx.start_rx() {
                eprintln!("cannot start RX: {e}");
                eprintln!("try `modem devices` then --in <name>; the mic must accept 48 kHz");
                std::process::exit(1);
            }
            println!(
                "listening as {:04X} on {} @ {:.2} kHz, profile {} -- Ctrl-C to stop",
                trx.device_id(),
                o.band.name,
                o.band.base_freq / 1000.0,
                o.profile
            );
            // The receiver runs on its own thread; this one just relays events.
            let mut printer = Printer::default();
            for ev in events.iter() {
                printer.print(&ev);
            }
        }

        "send" => {
            if argv.len() < 2 {
                usage()
            }
            let text = argv[1].clone();
            let o = parse_opts(&argv[2..]);
            let (trx, events) = live(&o);
            // ARQ needs our own receiver up to hear the acknowledgement.
            if o.arq {
                if let Err(e) = trx.start_rx() {
                    eprintln!("cannot start RX (needed for ARQ): {e}");
                    std::process::exit(1);
                }
            }
            let printer = std::thread::spawn(move || {
                let mut p = Printer::default();
                for ev in events.iter() {
                    p.print(&ev);
                }
            });
            let result = trx.send_text(&text);
            std::thread::sleep(std::time::Duration::from_millis(200));
            trx.stop_rx();
            drop(trx); // closes the event channel, ending the printer
            let _ = printer.join();
            if let Err(e) = result {
                eprintln!("send failed: {e}");
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
