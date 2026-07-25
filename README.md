# Acoustic Near-Ultrasonic Modem (half-duplex transceiver)

Data-over-sound modem: M-FSK + Hamming(7,4) FEC + block interleaving + CRC-16,
chirp preamble with a matched-filter sync, TUI for half-duplex text & file
transfer. Sample rate fixed at 48 kHz. Pure-Python, extensible.

## Files

- `modem_core.py` — DSP + framing + FEC + CRC + preamble detector. **Zero audio
  deps** (numpy only). Unit-testable without a sound card.
- `trx.py` — half-duplex transceiver (audio I/O, streaming receiver, fragmentation,
  text/file) plus the `selftest` / `probe` / `loopback` diagnostics.
- `tui.py` — terminal UI (Windows Terminal friendly).
- `test_modem.py` — offline test suite, **no sound card needed**:
  `python test_modem.py`. Covers FEC/interleaving/CRC, sync, the codec across
  all profiles, and the streaming receiver driven by a fake sound card.

## Run

```
python -m pip install -r requirements.txt
python tui.py
```

### First time on a new pair of machines — do this in order

1. `/probe` on **each** machine. It plays a tone sweep and measures what that
   machine's own mic hears, then tells you whether the band you are tuned to
   survives.
2. `/loopback` on each machine. Full speaker→air→mic round trip; it should
   print `PASS: decoded ...`.
3. `/rx on` on **both** machines, with the **same** `/mode` and `/freq`.
4. Type a message on one. Watch the other machine's meter line.

The meter line is the thing to read when something does not work:

```
mic  -34.1dB [########--------]  band  -48.0dB  preamble now=0.31 best=0.55 noise=0.02 (need>=0.20)  search  ok=3 bad=0 det=3 rej=0
```

- `mic` near `-90 dB` → the microphone is not capturing at all (wrong input
  device, muted, or no permission). Use `/devices` and `/in <n>`.
- `mic` fine but `preamble best` stays below the threshold → the sender's tones
  are not reaching this mic. Raise the volume, move closer, or `/mode audible`.
- `det` counts up but `bad` counts too → sync works, SNR does not. Try
  `/profile 2` (ROBUST), or `/mode audible`.

## Commands

`/mode audible|inaudible` · `/freq <kHz>` · `/profile 0..2` · `/rx on|off` ·
`/file <path>` · `/devices` · `/in <n>` · `/out <n>` · `/id [hhhh]` ·
`/arq on|off` · `/ping` · `/probe` · `/selftest` · `/loopback` · `/thresh <x>` ·
`/clear` · `/quit`. Anything else typed is sent as a text message.

## Handshake and ARQ

Every device has a 16-bit id, random on first run and kept in `.device_id` next
to the module (gitignored — an id that two machines share is not an id). `/id`
shows it, `/id A1B2` sets it.

A transfer is a conversation, not a broadcast:

```
A -> *   frag 1/3  from A1A1, ack requested   <- doubles as "who is out there?"
B -> A   ACK 1     from B2B2                  <- A now knows the peer
A -> B   frag 2/3  addressed to B2B2          <- only after that ACK matched
B -> A   ACK 2     from B2B2
```

There is deliberately **no hello round trip before a message**. The first
fragment already carries our id, and the ack already carries the peer's, so a
separate introduction would double the air time of a short message to learn
nothing new. A three-letter word costs one packet each way — on FAST that is
~7.9 s instead of ~14.9 s. `/ping` still does an explicit hello when you want to
ask "is anyone there?" without committing to a transfer; that is a debugging
tool, not part of sending.

The high bit of the type byte is the "acknowledge this" flag, so a receiver
knows whether the sender is blocked waiting. Once a peer is known it is
remembered and addressed directly; if it stops answering for a whole fragment's
worth of retries it is forgotten, and the next message rediscovers.

Every frame names its sender **and** its recipient, so the ACK check is exact:
the sender advances only on an ACK that carries the peer id it shook hands with
*and* the fragment index it just sent. Anything else — a stray ACK, a third
device answering — is ignored and the fragment is resent, up to `arq_retries`
(4) times. A frame addressed to someone else is dropped without a reply, so two
pairs can share a room without acking each other's traffic.

Two consequences worth knowing:

- **ARQ needs `/rx on` at both ends.** The sender has to hear the ACK. Without a
  running receiver it warns and falls back to sending blind.
- **A lost ACK causes a resend, and the resend is deduplicated.** The receiver
  re-acknowledges a fragment it already has (otherwise the sender never
  advances) but does not deliver it twice.

Cost: one extra packet per fragment — the ack itself — and the turnaround guards
around it. `/arq off` returns to fire-and-forget, which is faster and tells you
nothing about whether anything arrived.

A message under 48 bytes is a single fragment; above that it splits at 48-byte
boundaries, so the round trips scale with length and short messages pay for
exactly one.

The turnaround guard matters more than it looks. A sender keeps its own receiver
muted for a moment after it stops playing, so the room's echo of its own packet
is not decoded as an incoming one. A peer that replies *inside* that window
loses the first thing it says — the chirp preamble — and the exchange stalls
with neither side at fault. `REPLY_GUARD_S` is that delay before answering, and
it must exceed the sender's echo tail.

## Frequency: `/mode` and `/freq`

Where the link sits and how it modulates are **independent** settings.

- **`/mode audible` / `/mode inaudible`** switches band. Inaudible is the
  near-ultrasonic default; audible drops to ~3 kHz and works on any speaker/mic
  pair on earth, including Bluetooth. (Aliases: `ultrasonic`, `audio`, `ultra`…)
- **`/freq 18.6`** retunes the *current* band's lowest tone — accepts kHz
  (`18.6`) or Hz (`18600`), and `/freq reset` restores the default. `/freq` with
  no argument prints the current tone plan.
- **`/profile 0|1|2`** picks FAST / NORMAL / ROBUST. Profiles are band-agnostic:
  the same three work whether you are at 18 kHz or 3 kHz.

The chirp preamble and all data tones move together with `/freq`, so a retune is
a genuine **channel change** — a receiver left on the old frequency will not
read you. Both devices must agree. The offsets are rejected up front if the
tones would run past Nyquist or down into DC.

```
/mode inaudible
/freq 18.6            # tones at 18.60 / 18.80 / 19.00 / 19.20 kHz on NORMAL
/probe                # confirm this machine actually passes 18.6-19.2 kHz
```

Because retuning is live, a running receiver notices and rebuilds its matched
filters — no need to stop and restart RX.

Two devices can also just be put on **different** bands to run two independent
links in the same room; the receiver matched-filters every band at once and
tells you which one a packet arrived on.

## Modulation profiles (the adaptivity knob)

Tone spacing is relative to the band's base frequency, set by `/freq`.

| id | name   | tones            | scheme | symbol/guard | raw bit/s | net B/s | use                    |
| -- | ------ | ---------------- | ------ | ------------ | --------- | ------- | ---------------------- |
| 0  | FAST   | base … base+1200 | 4-FSK  | 15/5 ms      | 100       | ~7.1    | low noise, close range |
| 1  | NORMAL | base … base+600  | 4-FSK  | 25/8 ms      | 61        | ~4.3    | default                |
| 2  | ROBUST | base … base+900  | 4-FSK  | 45/15 ms     | 33        | ~2.4    | noisy / >2 m           |

FAST is 4-FSK rather than 8-FSK deliberately. Eight tones only fit the usable
18.0–19.5 kHz window at ~170 Hz spacing, and measured over the air that spacing
loses to reverb badly enough to fail whole frames. Four tones 400 Hz apart
measured at or near 0% symbol error while still running 64% faster than NORMAL,
because the symbols are shorter.

`net B/s` is after the 4/7 FEC overhead. These are genuinely slow links — a
40-byte message takes ~5 s on FAST and ~23 s on ROBUST. That is inherent to
M-FSK with guard intervals, not a bug.

The profile id travels in the frame header, so the receiver adapts per-frame: it
will decode any profile in the band it detected, regardless of its own setting.
The `/profile` setting only picks what **you transmit** with (and which profile
is tried first on receive). `/mode` and `/freq` are different — those really do
have to match on both ends.

## Why the inaudible band defaults to 18.2 kHz and not 18.5–21 kHz

Measured on real consumer hardware, the speaker→mic response falls off a cliff
above ~19.5 kHz (about 15 dB down at 20 kHz relative to 18.6 kHz). Tones and
chirps placed up there simply do not come back. The default keeps everything —
data tones and the preamble chirp — below that knee, while staying inaudible to
essentially everyone.

`/freq` lets you move it anyway: run `/probe` first and pick a spot where your
own hardware is strong. The permitted range is roughly 0.7–20.2 kHz (the guard
rails keep the chirp above 300 Hz and the top tone under 21.6 kHz).

## Troubleshooting: receiver sees nothing

Work down this list; the first two are by far the most common on Windows.

1. **Windows ducks playback while a mic is open.** Sound Control Panel →
   **Communications** → set to **"Do nothing"**. Windows treats a held-open
   capture stream as "communications activity" and quietens everything else —
   measured at **~7 dB** on the test machine. Both devices run `/rx on`, so this
   attenuates the *sender* exactly when it transmits. `/probe` reports it.
2. **Microphone "enhancements".** Sound Control Panel → Recording → your mic →
   Properties → Advanced/Enhancements: turn **off** noise suppression, echo
   cancellation, and "audio enhancements". These DSP blocks are tuned for speech
   and will delete a steady near-ultrasonic tone outright.
3. **Wrong device.** `/devices` then `/in <n>` / `/out <n>`. Bluetooth headsets
   are the classic trap: in hands-free mode they run at 8–16 kHz and cannot
   carry the inaudible band at all — use `/mode audible` or wired/built-in audio.
4. **Mic sample rate.** The mic must accept 48 kHz. `/rx on` reports the error if
   it cannot. Set the device to "48000 Hz" in the Windows device properties.
5. **Volume, and check the speaker is not muted.** The sender needs to be
   reasonably loud; ultrasonic content is attenuated by the speaker itself.
   `/probe` quantifies this — if every bar is deep in the negative, the output is
   muted or turned down, not broken.
6. **Mismatched tuning.** `/mode` and `/freq` must be identical on both ends.
   Run `/freq` on each and compare, or `/freq reset` on both.
7. Still nothing? `/thresh 0.12` lowers the detection bar (at the cost of more
   false triggers), and `/mode audible` moves the whole link into the audible
   band, which any hardware can carry.

Note: `/loopback` in the **audible** band can under-perform on a laptop that
plays and records at once — Windows' acoustic echo canceller actively subtracts
the speaker signal from the mic. That is a same-machine artifact; it does not
happen between two separate devices.

## Use as a module in your own project

```python
from trx import Transceiver

def got(kind, data, meta):
    print(kind, meta, data[:40])

trx = Transceiver(on_message=got, on_event=lambda lvl, txt: print(lvl, txt))

trx.set_band("AUDIO")          # or "ULTRA" (default)
trx.set_frequency(3_500)       # Hz; moves tones AND the chirp, live
print("\n".join(trx.plan()))   # where the tones actually are right now

trx.start_rx()
trx.send_text("hello", profile_id=1)
```

`on_event` is the diagnostic channel — preamble detections, header/CRC failures,
and RX-thread errors all surface there rather than being swallowed.

Below that, `modem_core` is pure DSP with no audio dependency:
`BANDS` / `PROFILES`, `set_band_base_freq(band, hz)`, `check_band_freq(hz)`,
`band_plan(band)`, `tone_freqs(profile, band)`, `encode_packet(payload,
profile_id, band)` and `try_decode(audio, profile_ids, data_start, band)`.

## How the receiver works

1. **Detect.** A 60 ms linear chirp is matched-filtered against the incoming
   stream with a *complex* (analytic) template, so detection is insensitive to
   the phase rotation the room and hardware apply. The normalized score is ~0.7
   for a clean chirp and ~0.02–0.05 for room noise.
2. **Reject self-detections.** The payload's own tones sit inside the chirp's
   sweep range and correlate with it — measured up to 0.26, overlapping the
   real-chirp range. So a candidate is only accepted if the audio just before it
   is quiet *in-band*, which every real packet's lead-in silence guarantees and
   no mid-payload false peak can fake. (The comparison must be in-band: a
   near-ultrasonic chirp returns weaker than ordinary room rumble, so a
   broadband test throws away good preambles.)
3. **Wait.** The matched filter locates the packet start to the sample, but the
   frame itself is seconds of audio that has not arrived yet. The receiver holds
   state and waits for it — first enough for the header, then, once the header
   says how long the payload is, for the rest.
4. **Fine-sync + decode.** A sub-symbol offset search around the chirp estimate
   is validated by the header CRC, then locked in for the body. If the body
   still fails CRC, the neighbouring offsets are retried against the buffered
   audio before the frame is dropped — reverb moves the best sampling instant by
   a millisecond or two, and this alone recovered a large share of frames in
   testing.

## Build a standalone Windows .exe

```
pip install pyinstaller
pyinstaller --onefile --name acoustic_modem tui.py
# -> dist\acoustic_modem.exe
```

(A real Windows/macOS binary must be built on that OS — a Linux build box cannot
cross-compile PortAudio-backed exes.)

## Roadmap (architecture already supports these)

- **Resume**: stop-and-wait ARQ is in (see above), so a fragment now survives a
  loss. What is still missing is *persistence* — the received `frag_idx` set per
  `msg_id` is in memory only, so a restart mid-transfer starts the message over.
- **Selective repeat**: stop-and-wait pays a full round trip per fragment. A
  window plus a bitmap of missing indices would cut that sharply on long files.
- **Full duplex**: run TX and RX in non-overlapping sub-bands simultaneously.
  `Band`/`Profile` already parametrize the spectrum; define two bands and run
  two streams.
- **CSS / Chirp Spread Spectrum** for the data symbols (the preamble already is
  one): the recommended upgrade for multipath beyond ~2 m.
- **Reed–Solomon** instead of Hamming for burst-error resilience.
- **Adaptive rate**: the demodulator already reports a confidence figure per
  frame; feed it back over the reverse link and step FAST↔NORMAL↔ROBUST.

## Known limits

- Half-duplex only (TX mutes RX so the sender does not decode itself).
- Stop-and-wait ARQ: one fragment in flight at a time, so throughput costs a
  full round trip per fragment. With `/arq off`, or with no peer answering the
  handshake, there is no retransmission at all and every fragment must land.
- Reassembly state is in memory: a restart mid-transfer loses the partial message.
- Hamming(7,4) with depth-12 interleaving fixes scattered single-bit errors and
  short bursts; sustained interference needs RS.
