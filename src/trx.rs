//! Half-duplex transceiver: streaming receiver, fragmentation, and stop-and-wait
//! ARQ with peer identification.
//!
//! # Receiver design
//!
//! A state machine over an absolutely-indexed sample buffer:
//!
//! ```text
//! SEARCH -> matched-filter the chirp preamble over every new sample window
//! HEADER -> once enough samples past the chirp have ARRIVED, read the 6-byte
//!           header (fine-sync search + per-profile trial) to learn the length
//! BODY   -> wait for the rest of the frame to arrive, then decode and CRC it
//! ```
//!
//! Waiting for *arrival* is the whole point: a frame is seconds of wall-clock
//! audio, so a receiver that decodes the instant it sees a preamble is reading a
//! buffer that does not contain the data yet.
//!
//! # Events
//!
//! Everything the caller needs comes out of one [`Event`] channel rather than
//! callbacks -- logs, progress, and delivered messages alike. A UI can select on
//! it directly, and nothing is swallowed.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU16, Ordering};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::audio::Backend;
use crate::band::{Band, BandSet, DEFAULT_BAND};
use crate::chirp::PreambleDetector;
use crate::framing::{
    decode_file_blob, encode_ack, encode_file_meta, encode_hello, parse, CtrlFrame, DeviceId,
    FileMeta, FragHeader, Frame, ACK_HDR_LEN, BROADCAST, FRAG_HDR_LEN, FRAG_PAYLOAD,
    HELLO_HDR_LEN, T_ACK, T_FILE, T_HELLO, T_HELLO_ACK, T_TEXT,
};
use crate::mfsk::MfskModulator;
use crate::packet::{
    decode_frame, decode_header, default_sync_offsets, encode_packet, frame_samples,
    header_symbols, packet_duration_s,
};
use crate::profile::{profile, profiles_preferring, DEFAULT_PROFILE, PROFILES};
use crate::{ms_samples, DETECT_THRESHOLD, GAP_MS, LEAD_GUARD_MS, SAMPLE_RATE};

/// Seconds before an incomplete message is dropped.
pub const REASM_TIMEOUT: Duration = Duration::from_secs(300);
/// Attempts per fragment before the transfer is abandoned.
pub const ARQ_RETRIES: usize = 4;
/// Attempts to find a peer during explicit discovery.
pub const HANDSHAKE_RETRIES: usize = 3;

/// Slack for the peer to re-arm and begin replying.
///
/// Deliberately generous. Opening a fresh output stream per packet was measured
/// at ~0.75 s on hardware whose descriptor claims 0.18 s, so reported latency
/// cannot size this. The asymmetry settles it: waiting too long costs nothing
/// when the ack does arrive -- the wait ends on the ack, not on the clock -- while
/// timing out early costs a full data packet resend, which is tens of seconds.
pub const TURNAROUND_S: f64 = 2.5;

/// Wait before answering, so the sender has finished re-arming.
///
/// A sender keeps its receiver muted for a moment after it stops playing, to let
/// the room's echo of its own packet decay. A peer that replies *inside* that
/// window loses the first thing it says -- the chirp preamble -- and the exchange
/// stalls with neither side at fault. This guard must exceed that tail.
pub const REPLY_GUARD_S: f64 = 0.4;

/// How much silence to leave after our own transmission before listening again.
const ECHO_TAIL: Duration = Duration::from_millis(200);
/// Only run the matched filter once this much audio is new.
const MIN_SEARCH_NEW: usize = 4096;

// --------------------------------------------------------------------------- //
// Events and statistics
// --------------------------------------------------------------------------- //
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Info,
    Warn,
    Error,
}

#[derive(Clone, Debug)]
pub enum Event {
    Log(Level, String),
    Progress { pct: u8, note: String },
    Text(String),
    File { meta: FileMeta, data: Vec<u8> },
    /// The peer we are addressing changed (discovered, or forgotten after a
    /// transfer failed).
    PeerChanged(DeviceId),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RxState {
    #[default]
    Off,
    Search,
    Header,
    Body,
    Transmitting,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub state: RxState,
    pub rms_db: f32,
    pub band_db: f32,
    pub peak_score: f32,
    pub peak_hold: f32,
    pub noise_score: f32,
    pub rx_ok: u64,
    pub rx_bad: u64,
    pub detections: u64,
    pub rejected: u64,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub active_band: String,
    pub active_profile: usize,
    pub detect_threshold: f32,
    pub arq: bool,
    pub arq_retries: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            active_band: DEFAULT_BAND.to_string(),
            active_profile: DEFAULT_PROFILE,
            detect_threshold: DETECT_THRESHOLD,
            arq: true,
            arq_retries: ARQ_RETRIES,
        }
    }
}

struct Reasm {
    parts: HashMap<u16, Vec<u8>>,
    total: u16,
    kind: u8,
    touched: Instant,
}

struct Shared {
    backend: Arc<dyn Backend>,
    events: Sender<Event>,
    bands: Mutex<BandSet>,
    config: Mutex<Config>,
    device_id: AtomicU16,
    peer_id: AtomicU16,

    running: AtomicBool,
    /// Set while we are the one making noise; the receiver discards audio then.
    tx_active: AtomicBool,
    /// Set when a transmission just finished, so the receiver throws away the
    /// audio it queued while deaf instead of decoding our own packet.
    rearm: AtomicBool,
    /// One speaker: every playback serialises through this.
    tx_lock: Mutex<()>,

    ctrl: Mutex<Option<CtrlFrame>>,
    ctrl_cv: Condvar,

    msg_id: AtomicU16,
    reasm: Mutex<HashMap<(DeviceId, u16), Reasm>>,
    delivered: Mutex<HashMap<(DeviceId, u16), Instant>>,
    stats: Mutex<Stats>,
}

impl Shared {
    fn log(&self, level: Level, text: impl Into<String>) {
        let _ = self.events.send(Event::Log(level, text.into()));
    }

    fn progress(&self, pct: u8, note: impl Into<String>) {
        let _ = self.events.send(Event::Progress { pct, note: note.into() });
    }

    fn band(&self, name: &str) -> Band {
        self.bands.lock().unwrap().expect(name)
    }

    fn active_band(&self) -> Band {
        let name = self.config.lock().unwrap().active_band.clone();
        self.band(&name)
    }

    // ---- playback ---------------------------------------------------------- //
    /// Transmit one packet with our own receiver muted for the duration.
    ///
    /// The mute is per packet, not per message: ARQ needs the receiver live in
    /// the gaps between fragments, which is exactly where the ack arrives.
    fn play(&self, audio: &[f32]) -> Result<(), String> {
        let _guard = self.tx_lock.lock().unwrap();
        self.tx_active.store(true, Ordering::SeqCst);
        let result = self.backend.play(audio);
        std::thread::sleep(ECHO_TAIL);
        self.rearm.store(true, Ordering::SeqCst);
        self.tx_active.store(false, Ordering::SeqCst);
        result
    }

    fn send_ctrl(&self, kind: u8, dst: DeviceId, pid: usize, band: &Band, msg_id: u16, idx: u16) -> Result<(), String> {
        let body = if kind == T_ACK {
            encode_ack(self.device_id.load(Ordering::SeqCst), dst, msg_id, idx)
        } else {
            encode_hello(kind, self.device_id.load(Ordering::SeqCst), dst)
        };
        // Let the sender finish re-arming before we start talking; see REPLY_GUARD_S.
        std::thread::sleep(Duration::from_secs_f64(REPLY_GUARD_S));
        self.play(&encode_packet(&body, pid, band))
    }

    // ---- control-frame rendezvous ------------------------------------------ //
    fn post_ctrl(&self, c: CtrlFrame) {
        *self.ctrl.lock().unwrap() = Some(c);
        self.ctrl_cv.notify_all();
    }

    fn clear_ctrl(&self) {
        *self.ctrl.lock().unwrap() = None;
    }

    /// Block until a control frame addressed to us matches, or time runs out.
    ///
    /// Frames that do not match -- a stray ack, or a reply from a device that is
    /// not our peer -- are discarded and the wait continues. That is the identity
    /// check the whole scheme rests on.
    fn await_ctrl<F>(&self, kinds: &[u8], deadline: Instant, matches: F) -> Option<CtrlFrame>
    where
        F: Fn(&CtrlFrame) -> bool,
    {
        let mut slot = self.ctrl.lock().unwrap();
        loop {
            if let Some(c) = slot.take() {
                if kinds.contains(&c.kind) && matches(&c) {
                    return Some(c);
                }
                self.log(
                    Level::Warn,
                    format!("ignored control frame {:#04x} from {:04X}", c.kind, c.src),
                );
                continue;
            }
            let now = Instant::now();
            if now >= deadline {
                return None;
            }
            let (g, _) = self.ctrl_cv.wait_timeout(slot, deadline - now).unwrap();
            slot = g;
        }
    }
}

// --------------------------------------------------------------------------- //
// Transceiver
// --------------------------------------------------------------------------- //
pub struct Transceiver {
    shared: Arc<Shared>,
    rx_thread: Mutex<Option<JoinHandle<()>>>,
}

impl Transceiver {
    pub fn new(backend: Arc<dyn Backend>, device_id: DeviceId) -> (Self, Receiver<Event>) {
        let (tx, rx) = channel();
        let shared = Arc::new(Shared {
            backend,
            events: tx,
            bands: Mutex::new(BandSet::default()),
            config: Mutex::new(Config::default()),
            device_id: AtomicU16::new(device_id),
            peer_id: AtomicU16::new(BROADCAST),
            running: AtomicBool::new(false),
            tx_active: AtomicBool::new(false),
            rearm: AtomicBool::new(false),
            tx_lock: Mutex::new(()),
            ctrl: Mutex::new(None),
            ctrl_cv: Condvar::new(),
            msg_id: AtomicU16::new(0),
            reasm: Mutex::new(HashMap::new()),
            delivered: Mutex::new(HashMap::new()),
            stats: Mutex::new(Stats::default()),
        });
        (Self { shared, rx_thread: Mutex::new(None) }, rx)
    }

    // ---- accessors --------------------------------------------------------- //
    pub fn device_id(&self) -> DeviceId {
        self.shared.device_id.load(Ordering::SeqCst)
    }

    pub fn set_device_id(&self, id: DeviceId) -> Result<(), String> {
        if id == BROADCAST {
            return Err("0000 is the broadcast address, not a device id".into());
        }
        self.shared.device_id.store(id, Ordering::SeqCst);
        Ok(())
    }

    pub fn peer_id(&self) -> DeviceId {
        self.shared.peer_id.load(Ordering::SeqCst)
    }

    pub fn stats(&self) -> Stats {
        *self.shared.stats.lock().unwrap()
    }

    pub fn config(&self) -> Config {
        self.shared.config.lock().unwrap().clone()
    }

    pub fn update_config(&self, f: impl FnOnce(&mut Config)) {
        f(&mut self.shared.config.lock().unwrap());
    }

    pub fn bands(&self) -> BandSet {
        self.shared.bands.lock().unwrap().clone()
    }

    /// Retune a band. Both ends must agree -- it is a channel, not a preference.
    /// A running receiver notices the revision bump and rebuilds its filters.
    pub fn set_frequency(&self, band: &str, hz: f64) -> Result<Band, String> {
        let b = self.shared.bands.lock().unwrap().set_base_freq(band, hz)?;
        self.shared.log(
            Level::Info,
            format!(
                "{} retuned to {:.3} kHz (chirp {:.2}-{:.2} kHz){}",
                b.name,
                b.base_freq / 1000.0,
                b.chirp_f0() / 1000.0,
                b.chirp_f1() / 1000.0,
                if b.audible() { "  [AUDIBLE]" } else { "" }
            ),
        );
        Ok(b)
    }

    pub fn is_running(&self) -> bool {
        self.shared.running.load(Ordering::SeqCst)
    }

    // ---- receiver lifecycle ------------------------------------------------ //
    pub fn start_rx(&self) -> Result<(), String> {
        if self.shared.running.swap(true, Ordering::SeqCst) {
            return Ok(());
        }
        let (audio_tx, audio_rx) = channel::<Vec<f32>>();
        let capture = match self.shared.backend.start_capture(audio_tx) {
            Ok(c) => c,
            Err(e) => {
                self.shared.running.store(false, Ordering::SeqCst);
                return Err(e);
            }
        };
        let shared = self.shared.clone();
        let handle = std::thread::spawn(move || {
            // Held for the lifetime of the loop; dropping it stops the stream.
            let _capture = capture;
            if let Err(e) = rx_loop(&shared, audio_rx) {
                shared.running.store(false, Ordering::SeqCst);
                shared.stats.lock().unwrap().state = RxState::Off;
                shared.log(Level::Error, format!("RX stopped: {e}"));
            }
        });
        *self.rx_thread.lock().unwrap() = Some(handle);
        Ok(())
    }

    pub fn stop_rx(&self) {
        self.shared.running.store(false, Ordering::SeqCst);
        if let Some(h) = self.rx_thread.lock().unwrap().take() {
            let _ = h.join();
        }
        self.shared.stats.lock().unwrap().state = RxState::Off;
    }

    // ---- transmit ---------------------------------------------------------- //
    pub fn send_text(&self, text: &str) -> Result<(), String> {
        self.send_payload(T_TEXT, text.as_bytes(), &[])
    }

    pub fn send_file(&self, path: &str) -> Result<(), String> {
        let data = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        let name = std::path::Path::new(path)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "file.bin".into());
        let meta = encode_file_meta(&name, data.len() as u32);
        self.send_payload(T_FILE, &data, &meta)
    }

    /// Explicit hello -> hello-ack, with no payload attached.
    ///
    /// Sending does not need this -- the first fragment introduces us by itself.
    /// It answers "is anyone out there?" without committing to a transfer, which
    /// is what you want when a link is not working.
    pub fn discover(&self) -> Result<DeviceId, String> {
        if !self.is_running() {
            return Err("start RX first -- discovery must hear the reply".into());
        }
        let cfg = self.config();
        let band = self.shared.active_band();
        let peer = self.handshake(cfg.active_profile, &band)?;
        if peer != BROADCAST {
            self.shared.peer_id.store(peer, Ordering::SeqCst);
            let _ = self.shared.events.send(Event::PeerChanged(peer));
        }
        Ok(peer)
    }

    fn handshake(&self, pid: usize, band: &Band) -> Result<DeviceId, String> {
        let p = profile(pid).ok_or("unknown profile")?;
        let wait = packet_duration_s(HELLO_HDR_LEN, p, band)
            + TURNAROUND_S
            + self.shared.backend.latency_hint() * 2.0;
        let me = self.device_id();
        for attempt in 1..=HANDSHAKE_RETRIES {
            self.shared.log(
                Level::Info,
                format!("handshake {attempt}/{HANDSHAKE_RETRIES}: hello from {me:04X}"),
            );
            self.shared.clear_ctrl();
            let body = encode_hello(T_HELLO, me, BROADCAST);
            self.shared.play(&encode_packet(&body, pid, band))?;
            let deadline = Instant::now() + Duration::from_secs_f64(wait);
            if let Some(c) = self
                .shared
                .await_ctrl(&[T_HELLO_ACK], deadline, |c| c.dst == me)
            {
                self.shared.log(Level::Info, format!("peer {:04X} answered", c.src));
                return Ok(c.src);
            }
        }
        Ok(BROADCAST)
    }

    fn send_payload(&self, kind: u8, data: &[u8], meta: &[u8]) -> Result<(), String> {
        let cfg = self.config();
        let band = self.shared.active_band();
        let pid = cfg.active_profile;
        let p = *profile(pid).ok_or("unknown profile")?;

        let mid = self.shared.msg_id.fetch_add(1, Ordering::SeqCst).wrapping_add(1);

        let mut blob = meta.to_vec();
        blob.extend_from_slice(data);
        let frags: Vec<&[u8]> = if blob.is_empty() {
            vec![&[]]
        } else {
            blob.chunks(FRAG_PAYLOAD).collect()
        };
        let total = frags.len();
        let est: f64 = frags
            .iter()
            .map(|f| packet_duration_s(f.len() + FRAG_HDR_LEN, &p, &band))
            .sum();
        self.shared.log(
            Level::Info,
            format!(
                "TX {total} frag(s) {} @ {:.2} kHz ({}), ~{est:.1}s of audio",
                p.name,
                band.base_freq / 1000.0,
                band.name
            ),
        );

        // ACKs need our own receiver running to be heard.
        let arq = cfg.arq && self.is_running();
        if cfg.arq && !self.is_running() {
            self.shared
                .log(Level::Warn, "ARQ needs RX on to hear ACKs -- sending blind");
        }

        // No separate hello: the first fragment goes out addressed to whoever we
        // last spoke to (or to everyone, if nobody yet), asking to be acked. The
        // ack that comes back names the peer, so introductions cost no air time
        // of their own.
        let mut dst = if arq { self.peer_id() } else { BROADCAST };
        let me = self.device_id();
        let ack_wait = packet_duration_s(ACK_HDR_LEN, &p, &band)
            + TURNAROUND_S
            + REPLY_GUARD_S
            + self.shared.backend.latency_hint() * 2.0;
        let retries = if arq { cfg.arq_retries.max(1) } else { 1 };

        for (idx, frag) in frags.iter().enumerate() {
            let mut acked = false;
            for attempt in 1..=retries {
                let hdr = FragHeader {
                    kind,
                    ack_req: arq,
                    src: me,
                    dst,
                    msg_id: mid,
                    frag_idx: idx as u16,
                    frag_total: total as u16,
                };
                let packet = encode_packet(&hdr.encode(frag), pid, &band);
                self.shared.clear_ctrl();
                self.shared.play(&packet)?;
                if !arq {
                    acked = true;
                    break;
                }
                let expect = dst;
                let deadline = Instant::now() + Duration::from_secs_f64(ack_wait);
                let got = self.shared.await_ctrl(&[T_ACK], deadline, |c| {
                    // The identity check the whole scheme rests on: once we know
                    // the peer, only that peer can advance us to the next fragment.
                    (expect == BROADCAST || c.src == expect)
                        && c.dst == me
                        && c.msg_id == mid
                        && c.frag_idx == idx as u16
                });
                if let Some(c) = got {
                    if dst == BROADCAST {
                        dst = c.src;
                        self.shared.peer_id.store(dst, Ordering::SeqCst);
                        let _ = self.shared.events.send(Event::PeerChanged(dst));
                        self.shared.log(
                            Level::Info,
                            format!("peer {dst:04X} identified itself in its ack"),
                        );
                    }
                    self.shared.log(
                        Level::Info,
                        format!("frag {}/{total} acked by {dst:04X}", idx + 1),
                    );
                    acked = true;
                    break;
                }
                self.shared.log(
                    Level::Warn,
                    format!(
                        "frag {}/{total} unacked (attempt {attempt}/{retries}) -- resending",
                        idx + 1
                    ),
                );
            }
            if !acked {
                // The peer we were addressing has gone quiet. Forget it, so the
                // next message rediscovers instead of talking to a ghost.
                self.shared.peer_id.store(BROADCAST, Ordering::SeqCst);
                let _ = self.shared.events.send(Event::PeerChanged(BROADCAST));
                self.shared.progress(0, "TX failed");
                let msg = format!(
                    "frag {}/{total} gave up after {retries} attempts; message {mid:04X} is incomplete",
                    idx + 1
                );
                self.shared.log(Level::Error, msg.clone());
                return Err(msg);
            }
            self.shared
                .progress(((idx + 1) * 100 / total) as u8, format!("TX {}/{total}", idx + 1));
            // Inter-frame gap; lets the peer re-arm.
            std::thread::sleep(Duration::from_millis(150));
        }
        Ok(())
    }
}

impl Drop for Transceiver {
    fn drop(&mut self) {
        self.stop_rx();
    }
}

// --------------------------------------------------------------------------- //
// Receiver
// --------------------------------------------------------------------------- //
fn max_frame_samples() -> usize {
    let biggest = FRAG_PAYLOAD + FRAG_HDR_LEN;
    PROFILES.iter().map(|p| frame_samples(biggest, p)).max().unwrap_or(0)
}

fn median(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(f32::total_cmp);
    s[s.len() / 2]
}

fn rx_loop(sh: &Arc<Shared>, audio: Receiver<Vec<f32>>) -> Result<(), String> {
    let mut band_rev = sh.bands.lock().unwrap().revision();
    let mut detectors = build_detectors(sh);
    let gap = ms_samples(GAP_MS) as i64;
    let lead_guard = ms_samples(LEAD_GUARD_MS) as i64;
    let max_hold = (max_frame_samples() + 3 * SAMPLE_RATE as usize) as i64;
    let offsets = default_sync_offsets();
    let max_off = *offsets.iter().max().unwrap() as i64;
    let min_off = *offsets.iter().min().unwrap() as i64;

    let mut buf: Vec<f32> = Vec::new();
    let mut origin: i64 = 0; // absolute index of buf[0]
    let mut search: i64 = 0; // absolute index of the next window to filter
    let mut state = RxState::Search;
    let mut det_band = sh.active_band();
    let mut data_start: i64 = 0;
    let mut lock_pid = 0usize;
    let mut lock_off = 0i64;
    let mut lock_len = 0usize;
    let mut need: i64 = 0;
    let mut cands: Vec<usize> = Vec::new();

    sh.log(
        Level::Info,
        format!(
            "RX listening @ {SAMPLE_RATE} Hz (threshold {:.2})",
            sh.config.lock().unwrap().detect_threshold
        ),
    );
    sh.stats.lock().unwrap().state = RxState::Search;

    while sh.running.load(Ordering::SeqCst) {
        let mut chunk = match audio.recv_timeout(Duration::from_millis(300)) {
            Ok(c) => c,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => return Err("audio input closed".into()),
        };
        while let Ok(more) = audio.try_recv() {
            chunk.extend_from_slice(&more);
        }

        update_level(sh, &chunk, &det_band);

        let rev = sh.bands.lock().unwrap().revision();
        if rev != band_rev {
            band_rev = rev;
            detectors = build_detectors(sh);
            det_band = sh.active_band();
            origin += buf.len() as i64 + chunk.len() as i64;
            search = origin;
            buf.clear();
            state = RxState::Search;
            sh.log(Level::Info, "retuned -- matched filters rebuilt");
            continue;
        }

        let transmitting = sh.tx_active.load(Ordering::SeqCst);
        if transmitting || sh.rearm.swap(false, Ordering::SeqCst) {
            // Either we are transmitting right now, or we just finished and the
            // queue still holds our own packet. Both cases: drop everything and
            // re-arm past it. Absolute indices are our own bookkeeping, so
            // skipping samples outright stays consistent.
            origin += buf.len() as i64 + chunk.len() as i64;
            search = origin;
            buf.clear();
            state = RxState::Search;
            sh.stats.lock().unwrap().state =
                if transmitting { RxState::Transmitting } else { RxState::Search };
            continue;
        }

        buf.extend_from_slice(&chunk);
        let total = origin + buf.len() as i64;

        // Trim, but never below what the current state still needs. In SEARCH we
        // hold an extra lead-guard of history behind the cursor, so lead_in_ok
        // can inspect the run-up to a candidate at the very start of the window.
        let want = if state == RxState::Search {
            search - lead_guard
        } else {
            data_start + min_off
        };
        let keep = want.min(total).max(total - max_hold).max(0);
        if keep > origin {
            buf.drain(..(keep - origin) as usize);
            origin = keep;
        }

        if state == RxState::Search {
            sh.stats.lock().unwrap().state = RxState::Search;
            let off = (search - origin).max(0) as usize;
            if off > buf.len() {
                continue;
            }
            let seg = &buf[off..];
            let mut hit: Option<(Band, f32, i64)> = None;
            let mut consumed = 0usize;

            for det in &detectors {
                if seg.len() < det.m() + MIN_SEARCH_NEW {
                    continue;
                }
                let sc = det.scores(seg);
                if sc.is_empty() {
                    continue;
                }
                consumed = consumed.max(sc.len());
                {
                    let mut st = sh.stats.lock().unwrap();
                    st.noise_score = median(&sc);
                    let i = (0..sc.len()).max_by(|&a, &b| sc[a].total_cmp(&sc[b])).unwrap();
                    st.peak_score = sc[i];
                    st.peak_hold = st.peak_hold.max(sc[i]);
                }
                // Strongest first, but only peaks with a quiet run-up: that is
                // what separates a real chirp from the packet's own payload
                // tones, which also correlate with it.
                for j in det.peaks(&sc) {
                    if !det.lead_in_ok(&buf, off + j, 0.5) {
                        sh.stats.lock().unwrap().rejected += 1;
                        continue;
                    }
                    let score = sc[j];
                    if hit.as_ref().is_none_or(|h| score > h.1) {
                        hit = Some((det.band, score, origin + off as i64 + j as i64));
                    }
                    break;
                }
            }
            if consumed > 0 {
                search = origin + off as i64 + consumed as i64;
            }
            if let Some((band, score, peak)) = hit {
                det_band = band;
                data_start = peak + band.chirp_samples() as i64 + gap;
                search = data_start; // never re-detect this chirp
                sh.stats.lock().unwrap().detections += 1;
                sh.log(
                    Level::Info,
                    format!(
                        "preamble {} score={score:.2} at t={:.2}s",
                        band.name,
                        peak as f64 / SAMPLE_RATE as f64
                    ),
                );
                // Profiles are band-agnostic, so any of them may be in use; try
                // our own setting first since it usually matches.
                cands = profiles_preferring(sh.config.lock().unwrap().active_profile);
                lock_pid = usize::MAX;
                need = data_start
                    + max_off
                    + cands
                        .iter()
                        .filter_map(|&c| profile(c))
                        .map(|p| ((header_symbols(p) - 1) * p.step_samples() + p.symbol_samples()) as i64)
                        .max()
                        .unwrap_or(0);
                state = RxState::Header;
            }
        }

        if state == RxState::Header && total >= need {
            sh.stats.lock().unwrap().state = RxState::Header;
            let mut found: Option<(usize, i64, usize, f64)> = None;
            'outer: for &pid in &cands {
                let Some(p) = profile(pid) else { continue };
                let m = MfskModulator::new(*p, det_band);
                for &o in &offsets {
                    let s = data_start + o as i64 - origin;
                    if s < 0 {
                        continue;
                    }
                    let r = decode_header(&buf, s as isize, &m, FRAG_PAYLOAD + FRAG_HDR_LEN);
                    if r.ok && r.profile_id == Some(pid) {
                        found = Some((pid, o as i64, r.length, r.confidence));
                        break 'outer;
                    }
                }
            }
            match found {
                None => {
                    sh.stats.lock().unwrap().rx_bad += 1;
                    sh.log(
                        Level::Warn,
                        "preamble found but header would not decode (weak signal / wrong profile)",
                    );
                    state = RxState::Search;
                }
                Some((pid, off, len, conf)) => {
                    lock_pid = pid;
                    lock_off = off;
                    lock_len = len;
                    let p = profile(pid).unwrap();
                    // Budget for the largest offset, so the body can be retried
                    // at any of them without waiting for more audio.
                    need = data_start + max_off + frame_samples(len, p) as i64;
                    sh.log(
                        Level::Info,
                        format!(
                            "header ok: {} on {} {len}B conf={conf:.0} (~{:.1}s)",
                            p.name,
                            det_band.name,
                            frame_samples(len, p) as f64 / SAMPLE_RATE as f64
                        ),
                    );
                    state = RxState::Body;
                }
            }
        }

        if state == RxState::Body {
            sh.stats.lock().unwrap().state = RxState::Body;
            if total < need {
                let pct = (100 * (total - data_start) / (need - data_start).max(1)).clamp(0, 99);
                sh.progress(pct as u8, "RX frame");
                continue;
            }
            let p = *profile(lock_pid).unwrap();
            let m = MfskModulator::new(p, det_band);
            // The offset that decoded the header is not always the best one for
            // the whole body -- reverb shifts the optimum by a millisecond or two.
            // The audio is already buffered, so on a CRC failure retry the
            // neighbouring offsets before giving up.
            let mut res = decode_frame(&buf, (data_start + lock_off - origin) as isize, &m, lock_len);
            let mut tries = 1;
            if !res.ok {
                for &o in &offsets {
                    if o as i64 == lock_off {
                        continue;
                    }
                    let s2 = data_start + o as i64 - origin;
                    if s2 < 0 {
                        continue;
                    }
                    tries += 1;
                    let alt = decode_frame(&buf, s2 as isize, &m, lock_len);
                    if alt.ok {
                        res = alt;
                        break;
                    }
                }
            }
            if res.ok {
                sh.stats.lock().unwrap().rx_ok += 1;
                sh.log(
                    Level::Info,
                    if tries > 1 {
                        format!("frame ok ({lock_len}B) after {tries} sync tries")
                    } else {
                        format!("frame ok ({lock_len}B)")
                    },
                );
                // Replies go out on the profile and band the frame arrived on,
                // not on ours -- the sender is listening there.
                handle_frame(sh, &res.payload, lock_pid, &det_band);
            } else {
                sh.stats.lock().unwrap().rx_bad += 1;
                sh.log(
                    Level::Warn,
                    format!("frame dropped: {} ({tries} sync offsets tried)", res.reason),
                );
            }
            search = search.max(need);
            state = RxState::Search;
        }
    }

    sh.stats.lock().unwrap().state = RxState::Off;
    Ok(())
}

fn build_detectors(sh: &Arc<Shared>) -> Vec<PreambleDetector> {
    let threshold = sh.config.lock().unwrap().detect_threshold;
    sh.bands
        .lock()
        .unwrap()
        .distinct_chirps()
        .into_iter()
        .map(|b| PreambleDetector::new(b, threshold))
        .collect()
}

fn update_level(sh: &Arc<Shared>, chunk: &[f32], band: &Band) {
    if chunk.is_empty() {
        return;
    }
    let rms = (chunk.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / chunk.len() as f64).sqrt();
    let mut st = sh.stats.lock().unwrap();
    st.rms_db = (20.0 * (rms + 1e-9).log10()) as f32;
    drop(st);

    // A fixed power-of-two window keeps the FFT planner's cache small; the
    // number is a level meter, not a measurement.
    const W: usize = 4096;
    if chunk.len() >= W {
        let det = PreambleDetector::new(*band, 1.0);
        let v = det.band_rms(&chunk[chunk.len() - W..]);
        sh.stats.lock().unwrap().band_db = (20.0 * (v + 1e-9).log10()) as f32;
    }
}

/// Dispatch one decoded frame: control traffic, or a data fragment.
fn handle_frame(sh: &Arc<Shared>, payload: &[u8], pid: usize, band: &Band) {
    let me = sh.device_id.load(Ordering::SeqCst);
    let Some(frame) = parse(payload) else {
        sh.log(Level::Warn, "frame decoded but is not a known message type");
        return;
    };

    match frame {
        Frame::Ctrl(c) if c.kind == T_HELLO => {
            // Someone is about to transmit. Answer with our own id so they know
            // who they are talking to -- that id gates every later ack.
            sh.log(
                Level::Info,
                format!("hello from {:04X} -- answering as {me:04X}", c.src),
            );
            sh.peer_id.store(c.src, Ordering::SeqCst);
            let _ = sh.events.send(Event::PeerChanged(c.src));
            if let Err(e) = sh.send_ctrl(T_HELLO_ACK, c.src, pid, band, 0, 0) {
                sh.log(Level::Error, format!("hello-ack failed: {e}"));
            }
        }
        Frame::Ctrl(c) if c.kind == T_HELLO_ACK => {
            if c.dst == me || c.dst == BROADCAST {
                sh.post_ctrl(c);
            }
        }
        Frame::Ctrl(c) => {
            if c.dst == me {
                sh.post_ctrl(c);
            }
        }
        Frame::Data { hdr, payload } => {
            if hdr.dst != me && hdr.dst != BROADCAST {
                sh.log(Level::Info, format!("frag for {:04X}, not us -- ignored", hdr.dst));
                return;
            }
            if hdr.ack_req {
                // The sender is blocked until it hears from us. Acknowledge
                // before reassembling -- its clock is already running. A
                // broadcast frame is acked too: that is how a sender that has
                // never met us learns our id.
                sh.peer_id.store(hdr.src, Ordering::SeqCst);
                let _ = sh.events.send(Event::PeerChanged(hdr.src));
                match sh.send_ctrl(T_ACK, hdr.src, pid, band, hdr.msg_id, hdr.frag_idx) {
                    Ok(()) => sh.log(
                        Level::Info,
                        format!(
                            "acked frag {}/{} to {:04X}",
                            hdr.frag_idx + 1,
                            hdr.frag_total,
                            hdr.src
                        ),
                    ),
                    Err(e) => sh.log(Level::Error, format!("ack failed: {e}")),
                }
            }
            reassemble(sh, hdr, payload);
        }
    }
}

fn reassemble(sh: &Arc<Shared>, hdr: FragHeader, frag: Vec<u8>) {
    let now = Instant::now();
    let key = (hdr.src, hdr.msg_id);

    {
        let mut done = sh.delivered.lock().unwrap();
        done.retain(|_, t| now.duration_since(*t) <= REASM_TIMEOUT);
        // A lost ack makes the sender resend a fragment we already have. It must
        // still be acknowledged (done above, unconditionally) or the sender never
        // advances -- but it must not be delivered twice.
        if done.contains_key(&key) {
            sh.log(
                Level::Info,
                format!(
                    "duplicate frag {}/{} of msg {:04X} -- re-acked, not re-delivered",
                    hdr.frag_idx + 1,
                    hdr.frag_total,
                    hdr.msg_id
                ),
            );
            return;
        }
    }

    let complete = {
        let mut reasm = sh.reasm.lock().unwrap();
        reasm.retain(|_, r| now.duration_since(r.touched) <= REASM_TIMEOUT);
        let slot = reasm.entry(key).or_insert_with(|| Reasm {
            parts: HashMap::new(),
            total: hdr.frag_total,
            kind: hdr.kind,
            touched: now,
        });
        // The first fragment of a message establishes its shape. A later frame
        // claiming a different total is a corrupted header that happened to pass
        // CRC, or a msg_id collision; either way, splicing it in would silently
        // produce a wrong message.
        if slot.total != hdr.frag_total {
            sh.log(
                Level::Warn,
                format!(
                    "frag {} of msg {:04X} claims {} fragments, not {} -- ignored",
                    hdr.frag_idx + 1,
                    hdr.msg_id,
                    hdr.frag_total,
                    slot.total
                ),
            );
            return;
        }
        slot.parts.insert(hdr.frag_idx, frag);
        slot.touched = now;
        let got = slot.parts.len();
        let total = slot.total as usize;
        sh.progress((got * 100 / total) as u8, format!("RX {got}/{total}"));
        if got == total {
            let mut blob = Vec::new();
            for i in 0..slot.total {
                blob.extend_from_slice(&slot.parts[&i]);
            }
            let kind = slot.kind;
            reasm.remove(&key);
            Some((kind, blob))
        } else {
            None
        }
    };

    if let Some((kind, blob)) = complete {
        sh.delivered.lock().unwrap().insert(key, now);
        deliver(sh, kind, blob);
    }
}

fn deliver(sh: &Arc<Shared>, kind: u8, blob: Vec<u8>) {
    match kind {
        T_TEXT => {
            let _ = sh
                .events
                .send(Event::Text(String::from_utf8_lossy(&blob).into_owned()));
        }
        T_FILE => match decode_file_blob(&blob) {
            Some((meta, data)) => {
                let _ = sh.events.send(Event::File { meta, data });
            }
            None => sh.log(Level::Warn, "file message was malformed"),
        },
        other => sh.log(Level::Warn, format!("unknown message type {other:#04x}")),
    }
}
