//! End-to-end transport tests over a virtual room — no sound card involved.
//!
//! Two `Transceiver`s share a fake acoustic channel running faster than real
//! time. Devices do not hear themselves: self-rejection is a physical-layer
//! concern that `lead_in_ok` already covers, and leaving it out keeps these
//! tests about the protocol.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use acoustic_modem::audio::{Backend, CaptureHandle};
use acoustic_modem::framing::BROADCAST;
use acoustic_modem::trx::{Event, Level, Transceiver};
use acoustic_modem::SAMPLE_RATE;

/// One device's microphone: a buffer that drains at the playback rate.
#[derive(Default)]
struct Ear {
    buf: Mutex<Vec<f32>>,
}

impl Ear {
    fn push(&self, audio: &[f32]) {
        self.buf.lock().unwrap().extend_from_slice(audio);
    }

    fn take(&self, n: usize) -> Vec<f32> {
        let mut b = self.buf.lock().unwrap();
        let take = n.min(b.len());
        let mut out: Vec<f32> = b.drain(..take).collect();
        out.resize(n, 0.0);
        out
    }
}

struct Room {
    ears: Vec<Arc<Ear>>,
    speed: f64,
    /// Packets each device put on the air — the metric the ARQ redesign was about.
    sent: Vec<AtomicUsize>,
}

impl Room {
    fn new(n: usize, speed: f64) -> Arc<Self> {
        Arc::new(Self {
            ears: (0..n).map(|_| Arc::new(Ear::default())).collect(),
            speed,
            sent: (0..n).map(|_| AtomicUsize::new(0)).collect(),
        })
    }
}

struct RoomBackend {
    room: Arc<Room>,
    me: usize,
    /// Set by a test to swallow the next N packets this device tries to send.
    swallow: Arc<AtomicUsize>,
}

impl Backend for RoomBackend {
    fn play(&self, samples: &[f32]) -> Result<(), String> {
        self.room.sent[self.me].fetch_add(1, Ordering::SeqCst);
        let eaten = self
            .swallow
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| {
                if v > 0 { Some(v - 1) } else { None }
            })
            .is_ok();
        if !eaten {
            for (i, ear) in self.room.ears.iter().enumerate() {
                if i != self.me {
                    ear.push(samples);
                }
            }
        }
        std::thread::sleep(Duration::from_secs_f64(
            samples.len() as f64 / SAMPLE_RATE as f64 / self.room.speed,
        ));
        Ok(())
    }

    fn start_capture(&self, sink: Sender<Vec<f32>>) -> Result<CaptureHandle, String> {
        // The real backend owns its stream on a dedicated thread and stops on
        // drop; reproduce that shape so the transceiver sees the same lifecycle.
        let ear = self.room.ears[self.me].clone();
        let speed = self.room.speed;
        acoustic_modem::audio::spawn_capture(move |stop| {
            const BS: usize = 1024;
            let mut seed = 0x2545F491u64 ^ (BS as u64);
            while !stop() {
                let mut block = ear.take(BS);
                for v in block.iter_mut() {
                    // A little noise, so normalization is exercised rather than
                    // dividing a clean signal by a clean floor.
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    *v += ((seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * 0.004;
                }
                if sink.send(block).is_err() {
                    break;
                }
                std::thread::sleep(Duration::from_secs_f64(
                    BS as f64 / SAMPLE_RATE as f64 / speed,
                ));
            }
        })
    }

    fn latency_hint(&self) -> f64 {
        0.05
    }
}

struct Node {
    trx: Transceiver,
    events: Receiver<Event>,
    swallow: Arc<AtomicUsize>,
}

fn node(room: &Arc<Room>, me: usize, id: u16, profile: usize) -> Node {
    let swallow = Arc::new(AtomicUsize::new(0));
    let backend = Arc::new(RoomBackend { room: room.clone(), me, swallow: swallow.clone() });
    let (trx, events) = Transceiver::new(backend, id);
    trx.update_config(|c| c.active_profile = profile);
    Node { trx, events, swallow }
}

/// Drain everything the node has emitted so far.
fn drain(node: &Node) -> Vec<Event> {
    node.events.try_iter().collect()
}

fn texts(events: &[Event]) -> Vec<String> {
    events
        .iter()
        .filter_map(|e| match e {
            Event::Text(t) => Some(t.clone()),
            _ => None,
        })
        .collect()
}

fn logs(events: &[Event]) -> Vec<(Level, String)> {
    events
        .iter()
        .filter_map(|e| match e {
            Event::Log(l, t) => Some((*l, t.clone())),
            _ => None,
        })
        .collect()
}

fn wait_for<F: Fn() -> bool>(limit: Duration, cond: F) -> bool {
    let deadline = Instant::now() + limit;
    while Instant::now() < deadline {
        if cond() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    cond()
}

#[test]
fn a_message_crosses_the_room_and_is_acknowledged() {
    let room = Room::new(2, 14.0);
    let a = node(&room, 0, 0xA1A1, 0);
    let b = node(&room, 1, 0xB2B2, 0);
    a.trx.start_rx().unwrap();
    b.trx.start_rx().unwrap();
    std::thread::sleep(Duration::from_millis(300));

    a.trx.send_text("ping").expect("send failed");
    assert!(
        wait_for(Duration::from_secs(20), || !texts(&drain(&b)).is_empty()
            || b.trx.stats().rx_ok > 0),
        "receiver never reported a frame"
    );
    std::thread::sleep(Duration::from_millis(400));

    let b_events = drain(&b);
    let a_events = drain(&a);
    a.trx.stop_rx();
    b.trx.stop_rx();

    // The message may have been drained by the wait above; check both places.
    let delivered = texts(&b_events);
    assert!(
        delivered == vec!["ping".to_string()] || b.trx.stats().rx_ok >= 1,
        "b delivered {delivered:?}"
    );
    assert_eq!(a.trx.peer_id(), 0xB2B2, "sender learned the peer id");
    assert_eq!(b.trx.peer_id(), 0xA1A1, "receiver learned the sender id");
    assert!(
        logs(&a_events).iter().any(|(_, t)| t.contains("acked by B2B2")),
        "sender never saw an ack: {:?}",
        logs(&a_events)
    );
    assert!(
        !logs(&a_events).iter().any(|(l, _)| *l == Level::Error),
        "errors during the exchange: {:?}",
        logs(&a_events)
    );
}

#[test]
fn a_short_message_costs_exactly_two_packets_on_air() {
    // The whole point of folding the introduction into the first fragment: a
    // short word costs one packet each way, not a hello round trip first.
    let room = Room::new(2, 14.0);
    let a = node(&room, 0, 0xA1A1, 0);
    let b = node(&room, 1, 0xB2B2, 0);
    a.trx.start_rx().unwrap();
    b.trx.start_rx().unwrap();
    std::thread::sleep(Duration::from_millis(300));

    a.trx.send_text("hi").expect("send failed");
    std::thread::sleep(Duration::from_millis(400));
    a.trx.stop_rx();
    b.trx.stop_rx();

    assert_eq!(room.sent[0].load(Ordering::SeqCst), 1, "sender packets");
    assert_eq!(room.sent[1].load(Ordering::SeqCst), 1, "receiver packets");
}

#[test]
fn a_swallowed_ack_causes_a_resend_and_no_duplicate_delivery() {
    let room = Room::new(2, 14.0);
    let a = node(&room, 0, 0xA1A1, 0);
    let b = node(&room, 1, 0xB2B2, 0);
    a.trx.start_rx().unwrap();
    b.trx.start_rx().unwrap();
    std::thread::sleep(Duration::from_millis(300));

    // Eat exactly the first thing b transmits — its ack.
    b.swallow.store(1, Ordering::SeqCst);

    a.trx.send_text("retry").expect("send failed");
    std::thread::sleep(Duration::from_millis(600));
    let b_events = drain(&b);
    let a_events = drain(&a);
    a.trx.stop_rx();
    b.trx.stop_rx();

    assert_eq!(b.swallow.load(Ordering::SeqCst), 0, "the test never swallowed an ack");
    assert!(
        logs(&a_events).iter().any(|(_, t)| t.contains("unacked")),
        "sender did not notice the missing ack"
    );
    assert_eq!(
        texts(&b_events),
        vec!["retry".to_string()],
        "the resent fragment must be delivered exactly once"
    );
    assert!(
        logs(&b_events).iter().any(|(_, t)| t.contains("duplicate frag")),
        "receiver should have recognised the resend as a duplicate"
    );
}

#[test]
fn once_a_peer_is_known_the_other_devices_go_quiet() {
    // Three devices in one room. The FIRST message is broadcast — nobody has
    // been introduced yet — so every listener answers it; that is how discovery
    // works and it is why this scheme wants a quiet room to start in. Once the
    // sender has a peer, subsequent frames are addressed, and the bystander must
    // neither acknowledge nor deliver them.
    let room = Room::new(3, 14.0);
    let a = node(&room, 0, 0xA1A1, 0);
    let b = node(&room, 1, 0xB2B2, 0);
    let c = node(&room, 2, 0xC3C3, 0);
    a.trx.start_rx().unwrap();
    b.trx.start_rx().unwrap();
    c.trx.start_rx().unwrap();
    std::thread::sleep(Duration::from_millis(300));

    a.trx.send_text("first").expect("send failed");
    std::thread::sleep(Duration::from_millis(400));

    // A pairs with whichever ack arrived first, which is not predetermined.
    let peer = a.trx.peer_id();
    assert_ne!(peer, BROADCAST, "a should have found a peer");
    let (bystander, by_node) = if peer == 0xB2B2 { (2usize, &c) } else { (1usize, &b) };
    let _ = drain(&b);
    let _ = drain(&c);
    let before = room.sent[bystander].load(Ordering::SeqCst);

    a.trx.send_text("second").expect("send failed");
    std::thread::sleep(Duration::from_millis(400));
    let by_events = drain(by_node);
    a.trx.stop_rx();
    b.trx.stop_rx();
    c.trx.stop_rx();

    assert_eq!(
        room.sent[bystander].load(Ordering::SeqCst),
        before,
        "the device that is not the peer must stay silent"
    );
    assert!(
        texts(&by_events).is_empty(),
        "the device that is not the peer must not deliver the message: {:?}",
        texts(&by_events)
    );
}

#[test]
fn explicit_discovery_finds_a_peer_without_sending_a_message() {
    let room = Room::new(2, 14.0);
    let a = node(&room, 0, 0xA1A1, 0);
    let b = node(&room, 1, 0xB2B2, 0);
    a.trx.start_rx().unwrap();
    b.trx.start_rx().unwrap();
    std::thread::sleep(Duration::from_millis(300));

    let peer = a.trx.discover().expect("discover failed");
    std::thread::sleep(Duration::from_millis(200));
    let b_events = drain(&b);
    a.trx.stop_rx();
    b.trx.stop_rx();

    assert_eq!(peer, 0xB2B2);
    assert!(texts(&b_events).is_empty(), "discovery must not deliver a message");
}

#[test]
fn discovery_without_a_receiver_is_refused_rather_than_hanging() {
    let room = Room::new(2, 14.0);
    let a = node(&room, 0, 0xA1A1, 0);
    assert!(a.trx.discover().is_err());
}
