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
    SAMPLE_RATE, PROFILES, BANDS, DEFAULT_PROFILE, GAP_MS, LEAD_GUARD_MS,
    DETECT_THRESHOLD,
    PreambleDetector, encode_packet, decode_header, decode_frame,
    frame_samples, header_symbols, sync_offsets, profiles_in_band,
    packet_duration_s,
)

try:
    import sounddevice as sd
    HAVE_AUDIO = True
    AUDIO_ERROR = ""
except Exception as exc:                      # pragma: no cover - env dependent
    HAVE_AUDIO = False
    AUDIO_ERROR = str(exc)


# ---- application-layer fragment header ------------------------------------ #
# type(1) msg_id(2) frag_idx(2) frag_total(2) [payload]
FRAG_HDR = struct.Struct(">BHHH")
T_TEXT = 0x01
T_FILE = 0x02
FRAG_PAYLOAD = 48          # bytes per fragment; small keeps frames short & retryable
REASM_TIMEOUT = 300.0      # seconds before an incomplete message is dropped

RX_BLOCK = 1024            # ~21 ms input blocks
MIN_SEARCH_NEW = 4096      # only run the matched filter once this much is new


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
        self.detect_threshold = DETECT_THRESHOLD

        self._rx_thread: threading.Thread | None = None
        self._running = False
        self._tx_active = threading.Event()
        self._msg_id = 0
        self._reasm: dict[int, dict] = {}
        self._lock = threading.Lock()
        self.stats = {"rms_db": -99.0, "band_db": -99.0, "peak_score": 0.0,
                      "peak_hold": 0.0, "noise_score": 0.0, "state": "off",
                      "rx_ok": 0, "rx_bad": 0, "detections": 0, "rejected": 0}

    # ------------------------------------------------------------------ TX -- #
    def _play(self, audio: np.ndarray):
        if not HAVE_AUDIO:
            raise RuntimeError(f"sounddevice not available: {AUDIO_ERROR}")
        sd.play(audio, SAMPLE_RATE, device=self.device_out, blocking=True)

    def _send_payload(self, kind: int, data: bytes, profile_id: int, meta: bytes = b""):
        if profile_id not in PROFILES:
            raise ValueError(f"unknown profile id {profile_id}")
        with self._lock:
            self._msg_id = (self._msg_id + 1) & 0xFFFF
            mid = self._msg_id
        blob = meta + data
        frags = [blob[i:i + FRAG_PAYLOAD] for i in range(0, len(blob), FRAG_PAYLOAD)] or [b""]
        total = len(frags)
        est = sum(packet_duration_s(len(f) + FRAG_HDR.size, profile_id) for f in frags)
        self.on_event("info", f"TX {total} frag(s) on {PROFILES[profile_id].name}, "
                              f"~{est:.1f}s of audio")
        # Half-duplex: silence our own receiver so it does not decode the speaker.
        self._tx_active.set()
        try:
            for idx, frag in enumerate(frags):
                head = FRAG_HDR.pack(kind, mid, idx, total)
                self._play(encode_packet(head + frag, profile_id))
                self.on_progress(int((idx + 1) / total * 100), f"TX {idx+1}/{total}")
                time.sleep(0.15)          # inter-frame gap; lets RX re-arm
        finally:
            time.sleep(0.2)               # let the room's echo of our own TX decay
            self._tx_active.clear()

    def send_text(self, text: str, profile_id: int | None = None):
        self._send_payload(T_TEXT, text.encode("utf-8"),
                           self.active_profile if profile_id is None else profile_id)

    def send_file(self, path: str, profile_id: int | None = None):
        pid = self.active_profile if profile_id is None else profile_id
        name = os.path.basename(path).encode("utf-8")[:255]
        with open(path, "rb") as f:
            data = f.read()
        meta = struct.pack(">B", len(name)) + name + struct.pack(">I", len(data))
        self._send_payload(T_FILE, data, pid, meta=meta)

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

    def _rx_loop(self):
        detectors = {name: PreambleDetector(band, self.detect_threshold)
                     for name, band in BANDS.items()}
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

                if self._tx_active.is_set():
                    # Transmitting: drop everything and re-arm past our own audio.
                    origin = search = origin + len(buf) + len(chunk)
                    buf = np.zeros(0, dtype=np.float32)
                    state, self.stats["state"] = "SEARCH", "tx"
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
                        cands = profiles_in_band(det_band)
                        if self.active_profile in cands:   # try the likely one first
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
                            r = decode_header(buf, s, pid, max_payload=FRAG_PAYLOAD +
                                              FRAG_HDR.size)
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
                                      f"header ok: {p.name} {lock_len}B "
                                      f"conf={conf:.0f} "
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
                                       lock_pid, lock_len)
                    tries = 1
                    if not res.ok:
                        for o in sync_offsets(PROFILES[lock_pid]):
                            if int(o) == lock_off:
                                continue
                            s2 = data_start + int(o) - origin
                            if s2 < 0:
                                continue
                            tries += 1
                            alt = decode_frame(buf, s2, lock_pid, lock_len)
                            if alt.ok:
                                res = alt
                                break
                    if res.ok:
                        self.stats["rx_ok"] += 1
                        self.on_event("info", f"frame ok ({lock_len}B)"
                                              + (f" after {tries} sync tries"
                                                 if tries > 1 else ""))
                        self._on_frame(res.payload)
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
        band = BANDS[PROFILES[self.active_profile].band]
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
    def _on_frame(self, payload: bytes):
        if len(payload) < FRAG_HDR.size:
            return
        kind, mid, idx, total = FRAG_HDR.unpack(payload[:FRAG_HDR.size])
        if total == 0 or idx >= total:
            return
        frag = payload[FRAG_HDR.size:]
        now = time.time()
        for k, v in list(self._reasm.items()):        # expire stale partials
            if now - v["t"] > REASM_TIMEOUT:
                del self._reasm[k]
        slot = self._reasm.setdefault(mid, {"parts": {}, "total": total,
                                            "kind": kind, "t": now})
        slot["parts"][idx] = frag
        slot["t"] = now
        got = len(slot["parts"])
        self.on_progress(int(got / total * 100), f"RX {got}/{total}")
        if got == total:
            blob = b"".join(slot["parts"][i] for i in range(total))
            del self._reasm[mid]
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
def selftest(profile_id: int = DEFAULT_PROFILE) -> list[str]:
    """Encode -> decode entirely in memory. No sound card involved."""
    from modem_core import try_decode, data_offset_in_packet
    out = []
    payload = b"selftest " + bytes(range(32))
    for pid in ([profile_id] if profile_id in PROFILES else PROFILES):
        pkt = encode_packet(payload, pid)
        stream = np.concatenate([np.zeros(3000, np.float32), pkt, np.zeros(3000, np.float32)])
        det = PreambleDetector(BANDS[PROFILES[pid].band])
        sc = det.scores(stream)
        peak = int(np.argmax(sc))
        r = try_decode(stream, [pid], peak + data_offset_in_packet(pid))
        ok = r.ok and r.payload == payload
        out.append(f"{PROFILES[pid].name:8s} score={sc[peak]:.2f} "
                   f"{'PASS' if ok else 'FAIL ' + r.reason}")
    return out


def probe(device_in=None, device_out=None, seconds: float = 0.3) -> list[str]:
    """Play tones through the speaker and measure what the mic hears.

    This is the fastest way to find out whether this hardware pair can carry the
    near-ultrasonic band at all. Many laptop mics roll off above ~18 kHz, and
    Windows mic "enhancements" (noise suppression / AEC) delete it outright.
    """
    if not HAVE_AUDIO:
        return [f"sounddevice not available: {AUDIO_ERROR}"]
    freqs = [1000, 3000, 4000, 6000, 10000, 14000, 16000, 17000,
             18200, 18600, 19000, 19600, 20000]
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
    out = [f"(bars are relative to the strongest tone, {top:.1f} dB)"]
    for f, db in levels.items():
        out.append(f"{f:6d} Hz  {db - top:6.1f}  {'#' * max(0, min(46, int(46 + (db - top))))}")

    ref = max(levels.get(f, -99) for f in (1000, 3000, 4000))
    ultra = max(levels.get(f, -99) for f in (18200, 18600, 19000))
    out.append("")
    if ultra < ref - 30:
        out.append(f"VERDICT: near-ultrasonic is {ref - ultra:.0f} dB below the audible "
                   f"band -- this hardware will not carry profiles 0-2.")
        out.append("         Use profile 3 (AUDIBLE) on BOTH devices: /profile 3")
    else:
        out.append(f"VERDICT: 18-19 kHz is {ref - ultra:.0f} dB down -- "
                   f"profiles 0-2 should work.")
    out.append("NOTE: Windows quietens playback while a mic is open ('communications"
               " activity'),")
    out.append("      measured at ~7 dB here. Since both devices keep RX on, that hits"
               " the sender.")
    out.append("      Sound Control Panel -> Communications -> 'Do nothing' to disable"
               " it.")
    return out


def loopback(profile_id: int = DEFAULT_PROFILE, text: str = "loopback test",
             device_in=None, device_out=None) -> list[str]:
    """Full round trip on one machine: speaker -> air -> mic -> decoder."""
    if not HAVE_AUDIO:
        return [f"sounddevice not available: {AUDIO_ERROR}"]
    from modem_core import try_decode, data_offset_in_packet
    payload = FRAG_HDR.pack(T_TEXT, 1, 0, 1) + text.encode()
    pkt = encode_packet(payload, profile_id)
    pad = np.zeros(int(SAMPLE_RATE * 0.4), np.float32)
    sig = np.concatenate([pad, pkt, pad])
    rec = sd.playrec(sig, SAMPLE_RATE, channels=1, blocking=True,
                     device=(device_in, device_out))[:, 0]
    band = BANDS[PROFILES[profile_id].band]
    det = PreambleDetector(band)
    sc = det.scores(rec)
    if not len(sc):
        return ["loopback: recording too short"]
    peak = int(np.argmax(sc))
    out = [f"recorded rms={20*np.log10(np.sqrt(np.mean(rec**2))+1e-9):.1f} dBFS",
           f"best preamble score={sc[peak]:.3f} (need >= {det.threshold}), "
           f"noise median={np.median(sc):.3f}"]
    if sc[peak] < det.threshold:
        out.append("FAIL: preamble not detected -- the mic never heard the chirp.")
        out.append("      Raise the volume, or run /probe to check the band.")
        return out
    r = try_decode(rec, [profile_id], peak + data_offset_in_packet(profile_id))
    if r.ok:
        got = r.payload[FRAG_HDR.size:].decode("utf-8", "replace")
        out.append(f"PASS: decoded {got!r} (conf={r.confidence:.0f})")
    else:
        out.append(f"FAIL at decode: {r.reason}")
        out.append("      Preamble was heard, so timing/SNR is marginal -- "
                   "try /profile 2 (ROBUST) or move the devices closer.")
    return out
