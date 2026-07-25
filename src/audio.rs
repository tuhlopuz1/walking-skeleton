//! Audio I/O, kept behind a trait.
//!
//! The DSP and the protocol never touch cpal directly, so the whole stack stays
//! testable on a machine with no sound card — the property that made the Python
//! version debuggable. Tests substitute a virtual room; production uses
//! [`CpalBackend`].

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::SAMPLE_RATE;

/// Stops the capture stream when dropped.
pub struct CaptureHandle {
    stop: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
}

impl Drop for CaptureHandle {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

pub trait Backend: Send + Sync {
    /// Play `samples` and return once they have been handed to the device.
    fn play(&self, samples: &[f32]) -> Result<(), String>;

    /// Begin capturing mono f32 at [`SAMPLE_RATE`], pushing blocks into `sink`.
    fn start_capture(&self, sink: Sender<Vec<f32>>) -> Result<CaptureHandle, String>;

    /// Round-trip device buffering, in seconds.
    ///
    /// Used to size the ACK turnaround budget. Deliberately allowed to be a
    /// pessimistic guess: waiting too long costs nothing when the ack does
    /// arrive, while timing out early costs a full data packet resend.
    fn latency_hint(&self) -> f64 {
        1.0
    }
}

// --------------------------------------------------------------------------- //
// cpal
// --------------------------------------------------------------------------- //
#[derive(Default, Clone)]
pub struct CpalBackend {
    pub input: Option<String>,
    pub output: Option<String>,
}

impl CpalBackend {
    pub fn new(input: Option<String>, output: Option<String>) -> Self {
        Self { input, output }
    }

    fn device(&self, want_input: bool) -> Result<cpal::Device, String> {
        let host = cpal::default_host();
        let wanted = if want_input { &self.input } else { &self.output };
        if let Some(name) = wanted {
            let mut devices = if want_input {
                host.input_devices().map_err(|e| e.to_string())?.collect::<Vec<_>>()
            } else {
                host.output_devices().map_err(|e| e.to_string())?.collect::<Vec<_>>()
            };
            devices.retain(|d| d.name().map(|n| n == *name).unwrap_or(false));
            return devices
                .into_iter()
                .next()
                .ok_or_else(|| format!("no such {} device: {name}", if want_input { "input" } else { "output" }));
        }
        if want_input {
            host.default_input_device().ok_or_else(|| "no default input device".into())
        } else {
            host.default_output_device().ok_or_else(|| "no default output device".into())
        }
    }

    /// Pick a config at exactly 48 kHz, preferring mono.
    ///
    /// The sample rate is not negotiable: every timing constant in the modem is
    /// expressed in samples at 48 kHz, so a device that will not run there
    /// cannot carry this link at all.
    fn config(dev: &cpal::Device, want_input: bool) -> Result<cpal::StreamConfig, String> {
        let ranges: Vec<cpal::SupportedStreamConfigRange> = if want_input {
            dev.supported_input_configs().map_err(|e| e.to_string())?.collect()
        } else {
            dev.supported_output_configs().map_err(|e| e.to_string())?.collect()
        };
        let usable: Vec<&cpal::SupportedStreamConfigRange> = ranges
            .iter()
            .filter(|r| {
                r.sample_format() == cpal::SampleFormat::F32
                    && r.min_sample_rate().0 <= SAMPLE_RATE
                    && r.max_sample_rate().0 >= SAMPLE_RATE
            })
            .collect();
        let chosen = usable
            .iter()
            .min_by_key(|r| r.channels())
            .ok_or_else(|| {
                format!(
                    "device does not support f32 at {SAMPLE_RATE} Hz \
                     (set it to 48000 Hz in the OS sound settings)"
                )
            })?;
        Ok(cpal::StreamConfig {
            channels: chosen.channels(),
            sample_rate: cpal::SampleRate(SAMPLE_RATE),
            buffer_size: cpal::BufferSize::Default,
        })
    }
}

impl Backend for CpalBackend {
    fn play(&self, samples: &[f32]) -> Result<(), String> {
        let dev = self.device(false)?;
        let cfg = Self::config(&dev, false)?;
        let channels = cfg.channels as usize;

        let done = Arc::new(AtomicBool::new(false));
        let cursor = Arc::new(std::sync::Mutex::new(0usize));
        let data: Arc<Vec<f32>> = Arc::new(samples.to_vec());

        let (d2, c2, s2) = (done.clone(), cursor.clone(), data.clone());
        let stream = dev
            .build_output_stream(
                &cfg,
                move |out: &mut [f32], _| {
                    let mut pos = c2.lock().unwrap();
                    for frame in out.chunks_mut(channels) {
                        // Mono source fanned out to however many channels the
                        // device insists on.
                        let v = if *pos < s2.len() { s2[*pos] } else { 0.0 };
                        if *pos < s2.len() {
                            *pos += 1;
                        }
                        frame.fill(v);
                    }
                    if *pos >= s2.len() {
                        d2.store(true, Ordering::SeqCst);
                    }
                },
                move |err| eprintln!("output stream error: {err}"),
                None,
            )
            .map_err(|e| e.to_string())?;
        stream.play().map_err(|e| e.to_string())?;

        // Wait for the callback to consume everything, with a ceiling so a dead
        // device cannot wedge the transmitter forever.
        let expected = Duration::from_secs_f64(samples.len() as f64 / SAMPLE_RATE as f64);
        let deadline = std::time::Instant::now() + expected + Duration::from_secs(5);
        while !done.load(Ordering::SeqCst) {
            if std::time::Instant::now() > deadline {
                return Err("playback timed out".into());
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        // Let the device drain what is still buffered before the stream dies.
        std::thread::sleep(Duration::from_millis(120));
        Ok(())
    }

    fn start_capture(&self, sink: Sender<Vec<f32>>) -> Result<CaptureHandle, String> {
        let dev = self.device(true)?;
        let cfg = Self::config(&dev, true)?;
        let channels = cfg.channels as usize;
        let stop = Arc::new(AtomicBool::new(false));
        let stop2 = stop.clone();

        // cpal streams are not Send on every platform, so the stream is created
        // and dropped on one dedicated thread that simply waits.
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), String>>();
        let join = std::thread::spawn(move || {
            let built = dev.build_input_stream(
                &cfg,
                move |input: &[f32], _| {
                    // Take channel 0; a mic array's extra channels are the same
                    // sound from a slightly different place and only add noise.
                    let mono: Vec<f32> = input.chunks(channels).map(|f| f[0]).collect();
                    let _ = sink.send(mono);
                },
                move |err| eprintln!("input stream error: {err}"),
                None,
            );
            let stream = match built {
                Ok(s) => s,
                Err(e) => {
                    let _ = ready_tx.send(Err(e.to_string()));
                    return;
                }
            };
            if let Err(e) = stream.play() {
                let _ = ready_tx.send(Err(e.to_string()));
                return;
            }
            let _ = ready_tx.send(Ok(()));
            while !stop2.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(20));
            }
        });

        match ready_rx.recv_timeout(Duration::from_secs(5)) {
            Ok(Ok(())) => Ok(CaptureHandle { stop, join: Some(join) }),
            Ok(Err(e)) => Err(e),
            Err(_) => Err("input stream did not start".into()),
        }
    }

    fn latency_hint(&self) -> f64 {
        // Measured, not reported. On hardware whose PortAudio/cpal descriptor
        // claims 0.18 s, opening a fresh output stream per packet was measured
        // at ~0.75 s before sound emerged. The descriptor cannot be trusted to
        // size a turnaround budget, so this is deliberately pessimistic.
        1.0
    }
}

/// Human-readable device inventory, for a `/devices`-style listing.
pub fn list_devices() -> Vec<String> {
    let host = cpal::default_host();
    let mut out = Vec::new();
    let def_in = host.default_input_device().and_then(|d| d.name().ok());
    let def_out = host.default_output_device().and_then(|d| d.name().ok());
    if let Ok(devs) = host.input_devices() {
        for d in devs {
            let name = d.name().unwrap_or_else(|_| "?".into());
            let star = if Some(&name) == def_in.as_ref() { " *default" } else { "" };
            out.push(format!("  in   {name}{star}"));
        }
    }
    if let Ok(devs) = host.output_devices() {
        for d in devs {
            let name = d.name().unwrap_or_else(|_| "?".into());
            let star = if Some(&name) == def_out.as_ref() { " *default" } else { "" };
            out.push(format!("  out  {name}{star}"));
        }
    }
    if out.is_empty() {
        out.push("  (no audio devices found)".into());
    }
    out
}
