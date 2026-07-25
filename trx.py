"""
trx.py
======
Half-duplex acoustic transceiver built on modem_core.

Public API (import this as a module in your own projects):

    trx = Transceiver(on_message=cb, on_event=log)
    trx.start_rx()                       # begin listening
    trx.send_text("hello", profile_id=1)
    trx.send_file("path/to/file.bin", profile_id=1)
    trx.stop()

Message framing above the physical frame supports multi-fragment payloads so
files larger than one frame are split, numbered, and reassembled. This is where
"resume paused transfer" will hook in later (track received fragment ids).

Receiver design
---------------
The RX side is a streaming state machine over an absolutely-indexed sample
buffer:

    SEARCH -> matched-filter the chirp preamble over every new sample window
    HEADER -> once enough samples past the chirp have ARRIVED, read the 6-byte
              header (fine sync search + per-profile trial) to learn the length
    BODY   -> wait for the rest of the frame to arrive, then decode + CRC

Waiting for arrival is the whole point: a frame takes seconds of wall-clock
audio, so a receiver that decodes the instant it sees a preamble is reading a
buffer that does not contain the data yet.

Audio backend: sounddevice (PortAudio). Isolated here so modem_core stays
dependency-light and unit-testable without a sound card.
"""

from __future__ import annotations
import os
import queue
import struct
import threading
import time
import numpy as np

from modem_core import (
    SAMPLE_RATE, PROFILES, ALL_PROFILES, BANDS, DEFAULT_PROFILE, DEFAULT_BAND,
    GAP_MS, LEAD_GUARD_MS, DETECT_THRESHOLD,
    PreambleDetector, encode_packet, decode_header, decode_frame,
    frame_samples, header_symbols, sync_offsets, packet_duration_s,
    band_revision, set_band_base_freq, band_plan, tone_freqs,
)

try:
    import sounddevice as sd
    HAVE_AUDIO = True
    AUDIO_ERROR = ""
except Exception as exc:                      # pragma: no cover - env dependent
    HAVE_AUDIO = False
    AUDIO_ERROR = str(exc)


# ---- application-layer framing --------------------------------------------- #
# Every frame names its sender and its intended recipient, so a receiver can tell
# "this is for me, and I know who it is from" with no out-of-band state. That is
# what makes stop-and-wait ARQ workable on a shared acoustic channel: an ACK is
# trusted only when it carries the peer id we shook hands with.
#
#   data : type(1) src(2) dst(2) msg_id(2) frag_idx(2) frag_total(2) [payload]
#   hello: type(1) src(2) dst(2)
#   ack  : type(1) src(2) dst(2) msg_id(2) frag_idx(2)
FRAG_HDR = struct.Struct(">BHHHHH")     # 11 B
HELLO_HDR = struct.Struct(">BHH")       # 5 B
ACK_HDR = struct.Struct(">BHHHH")       # 9 B

T_TEXT = 0x01
T_FILE = 0x02
T_HELLO = 0x10             # "I am about to transmit -- who is listening?"
T_HELLO_ACK = 0x11         # "I am, and this is my id"
T_ACK = 0x12               # "I got fragment N of message M"

BROADCAST = 0x0000         # dst everyone accepts and nobody acknowledges
FRAG_PAYLOAD = 48          # bytes per fragment; small keeps frames short & retryable
REASM_TIMEOUT = 300.0      # seconds before an incomplete message is dropped

ARQ_RETRIES = 4            # attempts per fragment before the transfer is abandoned
HANDSHAKE_RETRIES = 3      # attempts to find a peer before the transfer starts
TURNAROUND_S = 1.5         # slack for the peer to re-arm and begin replying
REPLY_GUARD_S = 0.4        # wait before answering, so the sender has re-armed
#   The sender keeps its receiver muted for a moment after it stops playing, to
#   let the room's echo of its own packet decay. Replying inside that window
#   means the first syllable of the reply -- the chirp preamble -- is thrown away
#   with the echo, and the exchange stalls. This guard must exceed that tail.

RX_BLOCK = 1024            # ~21 ms input blocks
MIN_SEARCH_NEW = 4096      # only run the matched filter once this much is new

DEVICE_ID_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              ".device_id")


def load_device_id() -> int:
    """A stable 16-bit name for this machine, persisted next to the module.

    Stability matters: the peer remembers who it shook hands with, so an id that
    changed on every run would strand an interrupted transfer.
    """
    try:
        with open(DEVICE_ID_FILE) as f:
            v = int(f.read().strip(), 16) & 0xFFFF
        if v != BROADCAST:
            return v
    except Exception:
        pass
    v = int.from_bytes(os.urandom(2), "big") % 0xFFFF + 1        # never 0
    try:
        with open(DEVICE_ID_FILE, "w") as f:
            f.write(f"{v:04X}")
    except Exception:
        pass
    return v


def list_devices() -> list[str]:
    if not HAVE_AUDIO:
        return [f"sounddevice unavailable: {AUDIO_ERROR}"]
    out = []
    apis = sd.query_hostapis()
    for i, d in enumerate(sd.query_devices()):
        api = apis[d["hostapi"]]["name"]
        out.append(f"{i:3d}  {d['name']}  [{api}]  in={d['max_input_channels']} "
                   f"out={d['max_output_channels']}")
    return out


class Transceiver:
    def __init__(self, on_message=None, on_progress=None, on_event=None,
                 device_in=None, device_out=None):
        self.on_message = on_message or (lambda kind, data, meta: None)
        self.on_progress = on_progress or (lambda pct, note: None)
        self.on_event = on_event or (lambda level, text: None)
        self.device_in = device_in
        self.device_out = device_out
        self.active_profile = DEFAULT_PROFILE
        self.active_band = DEFAULT_BAND      # which band we TRANSMIT in
        self.detect_threshold = DETECT_THRESHOLD

        self.device_id = load_device_id()
        self.peer_id = BROADCAST         # learned by the handshake, 0 = unknown
        self.arq = True                  # stop-and-wait with per-fragment ACKs
        self.arq_retries = ARQ_RETRIES

        self._rx_thread: threading.Thread | None = None
        self._running = False
        self._tx_active = threading.Event()
        self._msg_id = 0
        self._reasm: dict[tuple, dict] = {}
        self._done: dict[tuple, float] = {}   # (src, msg_id) already delivered
        self._lock = threading.Lock()
        self._tx_lock = threading.RLock()   # one speaker: serialise every _play
        self._rearm = threading.Event()     # RX must drop audio it made itself
        self._ctrl_lock = threading.Lock()
        self._ctrl_event = threading.Event()
        self._ctrl_last: tuple | None = None   # (type, src, dst, msg_id, frag_idx)
        self.stats = {"rms_db": -99.0, "band_db": -99.0, "peak_score": 0.0,
                      "peak_hold": 0.0, "noise_score": 0.0, "state": "off",
                      "rx_ok": 0, "rx_bad": 0, "detections": 0, "rejected": 0}

    # ------------------------------------------------------------------ TX -- #
    def _play(self, audio: np.ndarray):
        """Transmit one packet, with our own receiver muted for the duration.

        The mute is per-packet rather than per-message: ARQ needs the receiver
        live in the gaps between fragments, which is exactly where the ACK
        arrives. `_rearm` tells the RX loop to throw away the audio that was
        queued while we were the one making noise.
        """
        if not HAVE_AUDIO:
            raise RuntimeError(f"sounddevice not available: {AUDIO_ERROR}")
        with self._tx_lock:
            self._tx_active.set()
            try:
                sd.play(audio, SAMPLE_RATE, device=self.device_out, blocking=True)
            finally:
                time.sleep(0.2)           # let the room's echo of our own TX decay
                self._rearm.set()
                self._tx_active.clear()

    # -- control frames -- #
    def _send_hello(self, profile_id: int, band_name: str):
        self._play(encode_packet(HELLO_HDR.pack(T_HELLO, self.device_id, BROADCAST),
                                 profile_id, band_name))

    def _send_ctrl(self, kind: int, dst: int, profile_id: int, band_name: str,
                   mid: int = 0, idx: int = 0):
        if kind == T_HELLO_ACK:
            body = HELLO_HDR.pack(kind, self.device_id, dst)
        else:
            body = ACK_HDR.pack(kind, self.device_id, dst, mid, idx)
        time.sleep(REPLY_GUARD_S)     # let the sender finish re-arming; see above
        self._play(encode_packet(body, profile_id, band_name))

    def _await_ctrl(self, kinds: tuple, deadline: float, match=None):
        """Block until a control frame addressed to us matches, or time runs out.

        Returns the matching tuple, or None. Frames that do not match (a stray
        ACK, or a reply from a device that is not our peer) are discarded and
        the wait continues -- that is the id check the protocol turns on.
        """
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            self._ctrl_event.wait(remaining)
            with self._ctrl_lock:
                got = self._ctrl_last
                self._ctrl_last = None
                self._ctrl_event.clear()
            if got is None:
                continue
            if got[0] in kinds and (match is None or match(got)):
                return got
            self.on_event("warn", f"ignored control frame {got[0]:#04x} "
                                  f"from {got[1]:04X} (not what we waited for)")

    def _post_ctrl(self, info: tuple):
        """Hand a control frame to whoever is waiting in `_await_ctrl`."""
        with self._ctrl_lock:
            self._ctrl_last = info
            self._ctrl_event.set()

    def set_device_id(self, value: int) -> int:
        value &= 0xFFFF
        if value == BROADCAST:
            raise ValueError("0000 is the broadcast address, not a device id")
        self.device_id = value
        try:
            with open(DEVICE_ID_FILE, "w") as f:
                f.write(f"{value:04X}")
        except Exception as exc:
            self.on_event("warn", f"device id not persisted: {exc}")
        return value

    def _handshake(self, profile_id: int, band_name: str) -> int:
        """Announce ourselves and learn who is listening. Returns a peer id or 0."""
        wait = (packet_duration_s(HELLO_HDR.size, profile_id, band_name)
                + TURNAROUND_S * 2)
        for attempt in range(1, HANDSHAKE_RETRIES + 1):
            self.on_event("info", f"handshake {attempt}/{HANDSHAKE_RETRIES}: "
                                  f"hello from {self.device_id:04X}")
            with self._ctrl_lock:
                self._ctrl_last = None
                self._ctrl_event.clear()
            self._send_hello(profile_id, band_name)
            got = self._await_ctrl((T_HELLO_ACK,), time.monotonic() + wait,
                                   match=lambda g: g[2] == self.device_id)
            if got:
                self.on_event("info", f"peer {got[1]:04X} answered")
                return got[1]
        return BROADCAST

    def _send_payload(self, kind: int, data: bytes, profile_id: int,
                      band_name: str, meta: bytes = b""):
        if profile_id not in PROFILES:
            raise ValueError(f"unknown profile id {profile_id}")
        if band_name not in BANDS:
            raise ValueError(f"unknown band {band_name!r}")
        with self._lock:
            self._msg_id = (self._msg_id + 1) & 0xFFFF
            mid = self._msg_id
        blob = meta + data
        frags = [blob[i:i + FRAG_PAYLOAD] for i in range(0, len(blob), FRAG_PAYLOAD)] or [b""]
        total = len(frags)
        est = sum(packet_duration_s(len(f) + FRAG_HDR.size, profile_id, band_name)
                  for f in frags)
        band = BANDS[band_name]
        self.on_event("info", f"TX {total} frag(s) {PROFILES[profile_id].name} @ "
                              f"{band.base_freq/1000:.2f} kHz ({band_name}), "
                              f"~{est:.1f}s of audio")

        arq = self.arq and self._running     # ACKs need our own receiver running
        if self.arq and not self._running:
            self.on_event("warn", "ARQ needs RX on to hear ACKs -- sending blind")

        dst = BROADCAST
        if arq:
            dst = self._handshake(profile_id, band_name)
            if dst == BROADCAST:
                self.on_event("warn", "no peer answered the handshake -- "
                                      "sending blind, nothing will be acknowledged")
                arq = False
            else:
                self.peer_id = dst

        ack_wait = (packet_duration_s(ACK_HDR.size, profile_id, band_name)
                    + TURNAROUND_S * 2)
        for idx, frag in enumerate(frags):
            head = FRAG_HDR.pack(kind, self.device_id, dst, mid, idx, total)
            packet = encode_packet(head + frag, profile_id, band_name)
            for attempt in range(1, (self.arq_retries if arq else 1) + 1):
                with self._ctrl_lock:
                    self._ctrl_last = None
                    self._ctrl_event.clear()
                self._play(packet)
                if not arq:
                    break
                got = self._await_ctrl(
                    (T_ACK,), time.monotonic() + ack_wait,
                    # the id check the whole scheme rests on: only the peer we
                    # shook hands with can advance us to the next fragment
                    match=lambda g: (g[1] == dst and g[2] == self.device_id
                                     and g[3] == mid and g[4] == idx))
                if got:
                    self.on_event("info", f"frag {idx+1}/{total} acked by {dst:04X}")
                    break
                self.on_event("warn", f"frag {idx+1}/{total} unacked "
                                      f"(attempt {attempt}/{self.arq_retries}) "
                                      f"-- resending")
            else:
                self.on_event("error", f"frag {idx+1}/{total} gave up after "
                                       f"{self.arq_retries} attempts; message "
                                       f"{mid:04X} is incomplete")
                self.on_progress(0, "TX failed")
                return
            self.on_progress(int((idx + 1) / total * 100), f"TX {idx+1}/{total}")
            time.sleep(0.15)              # inter-frame gap; lets the peer re-arm

    def send_text(self, text: str, profile_id: int | None = None,
                  band_name: str | None = None):
        self._send_payload(T_TEXT, text.encode("utf-8"),
                           self.active_profile if profile_id is None else profile_id,
                           self.active_band if band_name is None else band_name)

    def send_file(self, path: str, profile_id: int | None = None,
                  band_name: str | None = None):
        pid = self.active_profile if profile_id is None else profile_id
        bnd = self.active_band if band_name is None else band_name
        name = os.path.basename(path).encode("utf-8")[:255]
        with open(path, "rb") as f:
            data = f.read()
        meta = struct.pack(">B", len(name)) + name + struct.pack(">I", len(data))
        self._send_payload(T_FILE, data, pid, bnd, meta=meta)

    # -------------------------------------------------------------- tuning -- #
    def set_frequency(self, hz: float, band_name: str | None = None):
        """Retune a band's lowest tone. Both ends must agree -- it is a channel.

        Takes effect immediately: a running receiver notices the revision bump
        and rebuilds its matched filters rather than listening on the old spot.
        """
        band = set_band_base_freq(band_name or self.active_band, hz)
        self.on_event("info", f"{band.name} retuned to {band.base_freq/1000:.3f} kHz "
                              f"(chirp {band.chirp_f0/1000:.2f}-"
                              f"{band.chirp_f1/1000:.2f} kHz)"
                              + ("  [AUDIBLE]" if band.audible else ""))
        return band

    def set_band(self, band_name: str):
        if band_name not in BANDS:
            raise ValueError(f"unknown band {band_name!r}; known: {', '.join(BANDS)}")
        self.active_band = band_name
        return BANDS[band_name]

    def plan(self) -> list[str]:
        return band_plan(self.active_band, self.active_profile)

    # ------------------------------------------------------------------ RX -- #
    def start_rx(self):
        if not HAVE_AUDIO:
            raise RuntimeError(f"sounddevice not available: {AUDIO_ERROR}")
        if self._running:
            return
        sd.check_input_settings(device=self.device_in, channels=1,
                                samplerate=SAMPLE_RATE, dtype="float32")
        self._running = True
        self._rx_thread = threading.Thread(target=self._rx_guard, daemon=True)
        self._rx_thread.start()

    def stop_rx(self):
        self._running = False
        t, self._rx_thread = self._rx_thread, None
        if t:
            t.join(timeout=2.0)
        self.stats["state"] = "off"

    stop = stop_rx

    @property
    def rx_running(self) -> bool:
        return self._running

    def _rx_guard(self):
        """Never let an exception die silently inside the RX thread."""
        try:
            self._rx_loop()
        except Exception as exc:
            self._running = False
            self.stats["state"] = "error"
            self.on_event("error", f"RX stopped: {type(exc).__name__}: {exc}")

    # -- buffer/state helpers -- #
    def _max_frame_samples(self) -> int:
        biggest = FRAG_PAYLOAD + FRAG_HDR.size
        return max(frame_samples(biggest, p) for p in PROFILES.values())

    def _build_detectors(self) -> dict:
        """One matched filter per DISTINCT chirp.

        Bands are retunable, so two of them can end up on the same frequency.
        Keying by the chirp's signature collapses those into one detector
        instead of double-firing on every packet.
        """
        out, seen = {}, {}
        for name, band in BANDS.items():
            sig = (band.chirp_f0, band.chirp_f1, band.chirp_ms)
            if sig in seen:
                continue
            seen[sig] = name
            out[name] = PreambleDetector(band, self.detect_threshold)
        return out

    def _rx_loop(self):
        detectors = self._build_detectors()
        band_rev = band_revision()
        gap = int(SAMPLE_RATE * GAP_MS / 1000)
        lead_guard = int(SAMPLE_RATE * LEAD_GUARD_MS / 1000)
        max_hold = self._max_frame_samples() + 3 * SAMPLE_RATE
        max_off = int(max(sync_offsets(p).max() for p in PROFILES.values()))
        min_off = int(min(sync_offsets(p).min() for p in PROFILES.values()))

        q: queue.Queue = queue.Queue()

        def cb(indata, frames, tinfo, status):
            q.put(indata[:, 0].copy())

        buf = np.zeros(0, dtype=np.float32)
        origin = 0            # absolute sample index of buf[0]
        search = 0            # absolute index of the next window to matched-filter
        state = "SEARCH"
        det_band = ""
        data_start = 0
        lock_pid = lock_off = lock_len = 0
        need = 0
        cands: list[int] = []

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=RX_BLOCK, device=self.device_in,
                            latency="high", callback=cb):
            self.on_event("info", f"RX listening @ {SAMPLE_RATE} Hz "
                                  f"(threshold {self.detect_threshold:.2f})")
            self.stats["state"] = "search"
            while self._running:
                try:
                    chunk = q.get(timeout=0.3)
                except queue.Empty:
                    continue
                while True:                       # drain whatever else queued up
                    try:
                        chunk = np.concatenate([chunk, q.get_nowait()])
                    except queue.Empty:
                        break

                self._update_level(chunk)

                if band_revision() != band_rev:
                    band_rev = band_revision()
                    detectors = self._build_detectors()
                    origin = search = origin + len(buf) + len(chunk)
                    buf = np.zeros(0, dtype=np.float32)
                    state = "SEARCH"
                    self.on_event("info", "retuned -- matched filters rebuilt")
                    continue

                if self._tx_active.is_set() or self._rearm.is_set():
                    # Either we are transmitting right now, or we just finished
                    # and the queue still holds our own packet. Both cases: drop
                    # everything and re-arm past it. Absolute indices are our own
                    # bookkeeping, so skipping samples outright is consistent.
                    self._rearm.clear()
                    origin = search = origin + len(buf) + len(chunk)
                    buf = np.zeros(0, dtype=np.float32)
                    state = "SEARCH"
                    self.stats["state"] = "tx" if self._tx_active.is_set() else "search"
                    continue

                buf = np.concatenate([buf, chunk])
                total = origin + len(buf)         # absolute index one past the end

                # ---- trim, but never below what the current state still needs ---
                # In SEARCH we hold an extra lead-guard of history behind the
                # search cursor, so lead_in_ok() can inspect the run-up to a
                # candidate that sits at the very start of the new window.
                keep = (search - lead_guard) if state == "SEARCH" else data_start + min_off
                keep = max(min(keep, total), total - max_hold, 0)
                if keep > origin:
                    buf = buf[keep - origin:]
                    origin = keep

                if state == "SEARCH":
                    self.stats["state"] = "search"
                    off = max(search - origin, 0)
                    seg = buf[off:]
                    hit, consumed = None, 0
                    for name, det in detectors.items():
                        if len(seg) < det.M + MIN_SEARCH_NEW:
                            continue
                        sc = det.scores(seg)
                        if not len(sc):
                            continue
                        consumed = max(consumed, len(sc))
                        self.stats["noise_score"] = round(float(np.median(sc)), 3)
                        i = int(np.argmax(sc))
                        self.stats["peak_score"] = round(float(sc[i]), 3)
                        self.stats["peak_hold"] = max(self.stats.get("peak_hold", 0.0),
                                                      round(float(sc[i]), 3))
                        # strongest first, but only peaks with a quiet run-up:
                        # that is what separates a real chirp from the packet's
                        # own payload tones, which also correlate with it.
                        for j in det.peaks(sc):
                            if not det.lead_in_ok(buf, off + j):
                                self.stats["rejected"] = self.stats.get("rejected", 0) + 1
                                continue
                            if hit is None or sc[j] > hit[1]:
                                hit = (name, float(sc[j]), origin + off + j)
                            break
                    if consumed:
                        search = origin + off + consumed
                    if hit:
                        det_band, score, peak = hit
                        data_start = peak + BANDS[det_band].chirp_samples + gap
                        search = data_start        # never re-detect this chirp
                        self.stats["detections"] += 1
                        self.on_event("info",
                                      f"preamble {det_band} score={score:.2f} "
                                      f"at t={peak / SAMPLE_RATE:.2f}s")
                        # Profiles are band-agnostic, so any of them may be in
                        # use; try our own setting first since it usually matches.
                        cands = list(ALL_PROFILES)
                        if self.active_profile in cands:
                            cands.remove(self.active_profile)
                            cands.insert(0, self.active_profile)
                        lock_pid = -1
                        need = data_start + max_off + max(
                            (header_symbols(PROFILES[c]) - 1) * PROFILES[c].step_samples
                            + PROFILES[c].symbol_samples for c in cands)
                        state = "HEADER"

                if state == "HEADER" and total >= need:
                    self.stats["state"] = "header"
                    found = None
                    for pid in cands:
                        for o in sync_offsets(PROFILES[pid]):
                            s = data_start + int(o) - origin
                            if s < 0:
                                continue
                            r = decode_header(buf, s, pid, det_band,
                                              max_payload=FRAG_PAYLOAD + FRAG_HDR.size)
                            if r.ok and r.profile_id == pid:
                                found = (pid, int(o), r.length, r.confidence)
                                break
                        if found:
                            break
                    if not found:
                        self.stats["rx_bad"] += 1
                        self.on_event("warn", "preamble found but header would not "
                                              "decode (weak signal / wrong profile)")
                        state = "SEARCH"
                    else:
                        lock_pid, lock_off, lock_len, conf = found
                        p = PROFILES[lock_pid]
                        # Budget for the largest offset so the body can be retried
                        # at any of them without waiting for more audio.
                        need = data_start + max_off + frame_samples(lock_len, p)
                        self.on_event("info",
                                      f"header ok: {p.name} on {det_band} "
                                      f"{lock_len}B conf={conf:.0f} "
                                      f"(~{frame_samples(lock_len, p)/SAMPLE_RATE:.1f}s)")
                        state = "BODY"

                if state == "BODY":
                    self.stats["state"] = "body"
                    if total < need:
                        pct = int(100 * (total - data_start) /
                                  max(1, need - data_start))
                        self.on_progress(min(pct, 99), "RX frame")
                        continue
                    # The offset that decoded the header is not always the best
                    # one for the whole body -- reverb shifts the optimum by a
                    # millisecond or two. The audio is already buffered, so on a
                    # CRC failure retry the neighbouring offsets before giving up.
                    res = decode_frame(buf, data_start + lock_off - origin,
                                       lock_pid, lock_len, det_band)
                    tries = 1
                    if not res.ok:
                        for o in sync_offsets(PROFILES[lock_pid]):
                            if int(o) == lock_off:
                                continue
                            s2 = data_start + int(o) - origin
                            if s2 < 0:
                                continue
                            tries += 1
                            alt = decode_frame(buf, s2, lock_pid, lock_len, det_band)
                            if alt.ok:
                                res = alt
                                break
                    if res.ok:
                        self.stats["rx_ok"] += 1
                        self.on_event("info", f"frame ok ({lock_len}B)"
                                              + (f" after {tries} sync tries"
                                                 if tries > 1 else ""))
                        # Replies go out on the profile/band the frame arrived
                        # on, not on ours -- the sender is listening there.
                        self._on_frame(res.payload, lock_pid, det_band)
                    else:
                        self.stats["rx_bad"] += 1
                        self.on_event("warn", f"frame dropped: {res.reason} "
                                              f"({tries} sync offsets tried)")
                    search = max(search, need)
                    state = "SEARCH"

        self.stats["state"] = "off"

    def _update_level(self, chunk: np.ndarray):
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2))) if len(chunk) else 0.0
        self.stats["rms_db"] = round(20 * np.log10(rms + 1e-9), 1)
        band = BANDS[self.active_band]
        n = min(len(chunk), 8192)
        if n >= 512:
            seg = chunk[-n:].astype(np.float64) * np.hanning(n)
            spec = np.abs(np.fft.rfft(seg)) / n
            fb = np.fft.rfftfreq(n, 1 / SAMPLE_RATE)
            m = (fb >= band.chirp_f0) & (fb <= band.chirp_f1)
            p = float(np.sqrt(np.sum(spec[m] ** 2))) if m.any() else 0.0
            self.stats["band_db"] = round(20 * np.log10(p + 1e-9), 1)

    def reset_peak(self):
        self.stats["peak_hold"] = 0.0

    # ---------------------------------------------------------- reassembly -- #
    def _on_frame(self, payload: bytes, profile_id: int, band_name: str):
        """Dispatch one decoded frame: control traffic, or a data fragment.

        Control frames are handed to whichever thread is blocked in
        `_await_ctrl`; data frames are reassembled and acknowledged.
        """
        if not payload:
            return
        kind = payload[0]

        if kind in (T_HELLO, T_HELLO_ACK):
            if len(payload) < HELLO_HDR.size:
                return
            _, src, dst = HELLO_HDR.unpack(payload[:HELLO_HDR.size])
            if kind == T_HELLO:
                # Someone is about to transmit. Answer with our own id so they
                # know who they are talking to -- that id gates every later ACK.
                self.on_event("info", f"hello from {src:04X} -- answering as "
                                      f"{self.device_id:04X}")
                self.peer_id = src
                try:
                    self._send_ctrl(T_HELLO_ACK, src, profile_id, band_name)
                except Exception as exc:
                    self.on_event("error", f"hello-ack failed: {exc}")
                return
            if dst not in (self.device_id, BROADCAST):
                return
            self._post_ctrl((kind, src, dst, 0, 0))
            return

        if kind == T_ACK:
            if len(payload) < ACK_HDR.size:
                return
            _, src, dst, mid, idx = ACK_HDR.unpack(payload[:ACK_HDR.size])
            if dst != self.device_id:
                return
            self._post_ctrl((kind, src, dst, mid, idx))
            return

        if len(payload) < FRAG_HDR.size:
            return
        kind, src, dst, mid, idx, total = FRAG_HDR.unpack(payload[:FRAG_HDR.size])
        if total == 0 or idx >= total:
            return
        if dst not in (self.device_id, BROADCAST):
            self.on_event("info", f"frag for {dst:04X}, not us -- ignored")
            return
        if dst == self.device_id:
            # Addressed to us, so the sender is waiting on an ACK before it will
            # move on. Acknowledge before reassembling: the peer's clock is
            # already running.
            self.peer_id = src
            try:
                self._send_ctrl(T_ACK, src, profile_id, band_name, mid, idx)
                self.on_event("info", f"acked frag {idx+1}/{total} to {src:04X}")
            except Exception as exc:
                self.on_event("error", f"ack failed: {exc}")
        frag = payload[FRAG_HDR.size:]
        now = time.time()
        for k, v in list(self._reasm.items()):        # expire stale partials
            if now - v["t"] > REASM_TIMEOUT:
                del self._reasm[k]
        for k, ts in list(self._done.items()):
            if now - ts > REASM_TIMEOUT:
                del self._done[k]

        # A lost ACK makes the sender resend a fragment we already have. It must
        # still be acknowledged (done above, unconditionally) or the sender never
        # advances -- but it must not be delivered twice.
        key = (src, mid)
        if key in self._done:
            self.on_event("info", f"duplicate frag {idx+1}/{total} of msg "
                                  f"{mid:04X} -- re-acked, not re-delivered")
            return
        slot = self._reasm.setdefault(key, {"parts": {}, "total": total,
                                            "kind": kind, "t": now})
        slot["parts"][idx] = frag
        slot["t"] = now
        got = len(slot["parts"])
        self.on_progress(int(got / total * 100), f"RX {got}/{total}")
        if got == total:
            blob = b"".join(slot["parts"][i] for i in range(total))
            del self._reasm[key]
            self._done[key] = now
            self._deliver(kind, blob)

    def _deliver(self, kind: int, blob: bytes):
        try:
            if kind == T_TEXT:
                self.on_message("text", blob.decode("utf-8", "replace"), None)
            elif kind == T_FILE:
                nlen = blob[0]
                name = blob[1:1 + nlen].decode("utf-8", "replace")
                size = struct.unpack(">I", blob[1 + nlen:5 + nlen])[0]
                data = blob[5 + nlen:5 + nlen + size]
                self.on_message("file", data, {"name": name, "size": size})
            else:
                self.on_event("warn", f"unknown message type 0x{kind:02X}")
        except Exception as exc:
            self.on_event("error", f"deliver failed: {exc}")


# --------------------------------------------------------------------------- #
# Diagnostics -- these answer "is the link dead, or is my hardware dead?"
# --------------------------------------------------------------------------- #
def selftest(profile_id: int = DEFAULT_PROFILE,
             band_name: str | None = None) -> list[str]:
    """Encode -> decode entirely in memory, at the CURRENT tuning.

    No sound card involved, so this isolates "is my configuration decodable"
    from "does my hardware carry it".
    """
    from modem_core import try_decode, data_offset_in_packet
    out = []
    payload = b"selftest " + bytes(range(32))
    bands = [band_name] if band_name in BANDS else list(BANDS)
    for bnd in bands:
        out.extend(band_plan(bnd))
        for pid in ([profile_id] if profile_id in PROFILES else PROFILES):
            pkt = encode_packet(payload, pid, bnd)
            stream = np.concatenate([np.zeros(3000, np.float32), pkt,
                                     np.zeros(3000, np.float32)])
            det = PreambleDetector(BANDS[bnd])
            sc = det.scores(stream)
            peak = int(np.argmax(sc))
            r = try_decode(stream, [pid], peak + data_offset_in_packet(bnd), bnd)
            ok = r.ok and r.payload == payload
            out.append(f"  {PROFILES[pid].name:8s} score={sc[peak]:.2f} "
                       f"{'PASS' if ok else 'FAIL ' + r.reason}")
    return out


def probe(device_in=None, device_out=None, seconds: float = 0.3,
          band_name: str | None = None) -> list[str]:
    """Play tones through the speaker and measure what the mic hears.

    This is the fastest way to find out whether this hardware pair can carry a
    given band. Many laptop mics roll off above ~18 kHz, and Windows mic
    "enhancements" (noise suppression / AEC) delete it outright. The sweep
    always includes the frequencies the active band is CURRENTLY tuned to, so
    it stays meaningful after /freq.
    """
    if not HAVE_AUDIO:
        return [f"sounddevice not available: {AUDIO_ERROR}"]
    freqs = [1000, 3000, 4000, 6000, 10000, 14000, 16000, 17000,
             18200, 18600, 19000, 19600, 20000]
    bnd = BANDS[band_name] if band_name in BANDS else None
    tuned: list[int] = []
    if bnd is not None:
        tuned = sorted({int(round(f)) for p in PROFILES.values()
                        for f in tone_freqs(p, bnd)}
                       | {int(bnd.chirp_f0), int(bnd.chirp_f1)})
        freqs = sorted(set(freqs) | set(tuned))
    n = int(SAMPLE_RATE * seconds)
    t = np.arange(n) / SAMPLE_RATE
    env = np.ones(n)
    env[:480] = np.linspace(0, 1, 480)
    env[-480:] = np.linspace(1, 0, 480)
    pause = np.zeros(int(SAMPLE_RATE * 0.08), np.float32)
    sig = np.concatenate([np.concatenate([(0.5 * np.sin(2 * np.pi * f * t) * env)
                                          .astype(np.float32), pause]) for f in freqs])
    rec = sd.playrec(sig, SAMPLE_RATE, channels=1, blocking=True,
                     device=(device_in, device_out))[:, 0]
    step = n + len(pause)
    levels = {}
    for i, f in enumerate(freqs):
        seg = rec[i * step:i * step + n]
        if len(seg) < n:
            break
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        fb = np.fft.rfftfreq(len(seg), 1 / SAMPLE_RATE)
        b = int(np.argmin(np.abs(fb - f)))
        levels[f] = 20 * np.log10(spec[max(0, b - 2):b + 3].max() + 1e-12)
    if not levels:
        return ["probe failed: nothing recorded"]

    # Bars are relative to the loudest tone measured -- absolute dBFS depends on
    # the volume knob, while the SHAPE of the response is what decides the band.
    top = max(levels.values())
    out = [f"(bars are relative to the strongest tone, {top:.1f} dB; "
           f"'<' marks the tuned band)"]
    for f, db in levels.items():
        mark = " <" if f in tuned else ""
        out.append(f"{f:6d} Hz  {db - top:6.1f}  "
                   f"{'#' * max(0, min(44, int(44 + (db - top))))}{mark}")

    ref = max(levels.get(f, -99) for f in (1000, 3000, 4000))
    out.append("")
    if bnd is not None and tuned:
        here = min(levels.get(f, -99) for f in tuned)
        drop = ref - here
        out.append(f"VERDICT for {bnd.name} @ {bnd.base_freq/1000:.2f} kHz: weakest "
                   f"tone is {drop:.0f} dB below the audible reference.")
        if drop > 30:
            out.append("         Too weak -- this hardware will not carry that "
                       "tuning.")
            out.append("         Try '/mode audible', or '/freq <kHz>' somewhere "
                       "stronger in the list above.")
        else:
            out.append("         That should work.")
    else:
        ultra = max(levels.get(f, -99) for f in (18200, 18600, 19000))
        if ultra < ref - 30:
            out.append(f"VERDICT: near-ultrasonic is {ref - ultra:.0f} dB below the "
                       f"audible band -- this hardware will not carry it.")
            out.append("         Use '/mode audible' on BOTH devices.")
        else:
            out.append(f"VERDICT: 18-19 kHz is {ref - ultra:.0f} dB down -- "
                       f"the inaudible band should work.")
    out.append("NOTE: Windows quietens playback while a mic is open ('communications"
               " activity'),")
    out.append("      measured at ~7 dB here. Since both devices keep RX on, that hits"
               " the sender.")
    out.append("      Sound Control Panel -> Communications -> 'Do nothing' to disable"
               " it.")
    return out


def loopback(profile_id: int = DEFAULT_PROFILE, text: str = "loopback test",
             device_in=None, device_out=None,
             band_name: str = DEFAULT_BAND) -> list[str]:
    """Full round trip on one machine: speaker -> air -> mic -> decoder."""
    if not HAVE_AUDIO:
        return [f"sounddevice not available: {AUDIO_ERROR}"]
    from modem_core import try_decode, data_offset_in_packet
    # Broadcast src/dst: loopback is a link test, not a conversation.
    payload = FRAG_HDR.pack(T_TEXT, BROADCAST, BROADCAST, 1, 0, 1) + text.encode()
    pkt = encode_packet(payload, profile_id, band_name)
    pad = np.zeros(int(SAMPLE_RATE * 0.4), np.float32)
    sig = np.concatenate([pad, pkt, pad])
    rec = sd.playrec(sig, SAMPLE_RATE, channels=1, blocking=True,
                     device=(device_in, device_out))[:, 0]
    band = BANDS[band_name]
    det = PreambleDetector(band)
    sc = det.scores(rec)
    if not len(sc):
        return ["loopback: recording too short"]
    peak = int(np.argmax(sc))
    out = [f"{PROFILES[profile_id].name} on {band_name} @ "
           f"{band.base_freq/1000:.2f} kHz",
           f"recorded rms={20*np.log10(np.sqrt(np.mean(rec**2))+1e-9):.1f} dBFS",
           f"best preamble score={sc[peak]:.3f} (need >= {det.threshold}), "
           f"noise median={np.median(sc):.3f}"]
    if sc[peak] < det.threshold:
        out.append("FAIL: preamble not detected -- the mic never heard the chirp.")
        out.append("      Raise the volume, or run /probe to check this tuning.")
        return out
    r = try_decode(rec, [profile_id], peak + data_offset_in_packet(band_name),
                   band_name)
    if r.ok:
        got = r.payload[FRAG_HDR.size:].decode("utf-8", "replace")
        out.append(f"PASS: decoded {got!r} (conf={r.confidence:.0f})")
    else:
        out.append(f"FAIL at decode: {r.reason}")
        out.append("      Preamble was heard, so timing/SNR is marginal -- "
                   "try '/profile 2' (ROBUST), '/mode audible', or move closer.")
    return out
