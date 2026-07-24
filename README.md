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
   machine's own mic hears, then tells you whether the 18–19 kHz band survives.
2. `/loopback` on each machine. Full speaker→air→mic round trip; it should
   print `PASS: decoded ...`.
3. `/rx on` on **both** machines, and set the **same** `/profile` on both.
4. Type a message on one. Watch the other machine's meter line.

The meter line is the thing to read when something does not work:

```
mic  -34.1dB [########--------]  band  -48.0dB  preamble now=0.31 best=0.55 noise=0.02 (need>=0.15)  search  ok=3 bad=0 det=3
```

- `mic` near `-90 dB` → the microphone is not capturing at all (wrong input
  device, muted, or no permission). Use `/devices` and `/in <n>`.
- `mic` fine but `preamble best` stays below the threshold → the sender's tones
  are not reaching this mic. Raise the volume, move closer, or drop to
  `/profile 3`.
- `det` counts up but `bad` counts too → sync works, SNR does not. Try
  `/profile 2` (ROBUST), or `/profile 3` if you are in the ultrasonic band.

## Commands

`/rx on|off` · `/profile 0..3` · `/file <path>` · `/devices` · `/in <n>` ·
`/out <n>` · `/probe` · `/selftest` · `/loopback` · `/thresh <x>` · `/clear` ·
`/quit`. Anything else typed is sent as a text message.

## Modulation profiles (the adaptivity knob)

| id | name    | band           | scheme | symbol/guard | raw bit/s | net B/s | use                         |
| -- | ------- | -------------- | ------ | ------------ | --------- | ------- | --------------------------- |
| 0  | FAST    | 18.20–19.40 k | 4-FSK  | 15/5 ms      | 100       | ~7.1    | low noise, close range      |
| 1  | NORMAL  | 18.20–18.80 k | 4-FSK  | 25/8 ms      | 61        | ~4.3    | default                     |
| 2  | ROBUST  | 18.20–19.10 k | 4-FSK  | 45/15 ms     | 33        | ~2.4    | noisy / >2 m                |
| 3  | AUDIBLE | 3.00–4.75 k   | 8-FSK  | 20/6 ms      | 115       | ~8.2    | hardware that can't do 18 k |

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
is tried first on receive).

## Why the ultrasonic band is 18.2–19.6 kHz and not 18.5–21 kHz

Measured on real consumer hardware, the speaker→mic response falls off a cliff
above ~19.5 kHz (about 15 dB down at 20 kHz relative to 18.6 kHz). Tones and
chirps placed up there simply do not come back. Everything — data tones and the
preamble chirp — is kept below that knee. It is still inaudible to essentially
everyone.

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
   carry the ultrasonic band at all — use `/profile 3` or wired/built-in audio.
4. **Mic sample rate.** The mic must accept 48 kHz. `/rx on` reports the error if
   it cannot. Set the device to "48000 Hz" in the Windows device properties.
5. **Volume.** The sender needs to be reasonably loud; ultrasonic content is
   often attenuated by the speaker itself. `/probe` quantifies this.
6. Still nothing? `/thresh 0.12` lowers the detection bar (at the cost of more
   false triggers), and `/profile 3` moves the whole link into the audible band,
   which any hardware can carry.

Note: `/loopback` on the **audible** profile can under-perform on a laptop that
plays and records at once — Windows' acoustic echo canceller actively subtracts
the speaker signal from the mic. That is a same-machine artifact; it does not
happen between two separate devices.

## Use as a module in your own project

```python
from trx import Transceiver

def got(kind, data, meta):
    print(kind, meta, data[:40])

trx = Transceiver(on_message=got, on_event=lambda lvl, txt: print(lvl, txt))
trx.start_rx()
trx.send_text("hello", profile_id=1)
```

`on_event` is the diagnostic channel — preamble detections, header/CRC failures,
and RX-thread errors all surface there rather than being swallowed.

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

- **ARQ / resume**: fragments carry `msg_id/frag_idx/frag_total`; persist the
  received `frag_idx` set per `msg_id` and request the missing indices. This is
  the single biggest reliability win left — today one corrupted fragment loses
  the whole message.
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
- No retransmission: every fragment of a message must arrive intact.
- Hamming(7,4) with depth-12 interleaving fixes scattered single-bit errors and
  short bursts; sustained interference needs RS.
