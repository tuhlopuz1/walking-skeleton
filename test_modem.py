"""
test_modem.py
=============
Offline test suite -- no sound card required. Run:  python test_modem.py

Covers the codec (FEC, interleaving, CRC, sync) and, with a fake sound card,
the streaming receiver state machine in trx.py. For hardware checks use the
TUI's /probe and /loopback instead.
"""
from __future__ import annotations
import sys
import threading
import time
import types

import numpy as np

from modem_core import (
    SAMPLE_RATE, PROFILES, ALL_PROFILES, BANDS, PreambleDetector,
    crc16, hamming_encode, hamming_decode, interleave, deinterleave,
    encode_packet, try_decode, data_offset_in_packet, packet_duration_s,
    build_frame_bits,
)

FAILURES: list[str] = []


def check(cond, label):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    if not cond:
        FAILURES.append(label)


def section(name):
    print(f"\n=== {name} ===")


# --------------------------------------------------------------------------- #
def test_fec():
    section("FEC, interleaving, CRC")
    rng = np.random.default_rng(1)
    ok = True
    for L in (1, 2, 6, 55, 200):
        d = bytes(rng.integers(0, 256, L, dtype=np.uint8))
        ok &= hamming_decode(hamming_encode(d)) == d
    check(ok, "Hamming(7,4) round-trips")

    d = bytes(rng.integers(0, 256, 24, dtype=np.uint8))
    bits = hamming_encode(d)
    for i in range(0, len(bits), 7):          # one error per codeword
        bits[i + (i // 7) % 7] ^= 1
    check(hamming_decode(bits) == d, "Hamming corrects 1 bit in every codeword")

    b = hamming_encode(bytes(range(12)))
    check(np.array_equal(deinterleave(interleave(b)), b), "interleaver round-trips")

    # A burst of INTERLEAVE_CW consecutive bit errors must survive: after
    # de-interleaving it becomes one error in each of 12 separate codewords.
    d = bytes(range(6))
    bits = interleave(hamming_encode(d))
    bits[20:32] ^= 1
    check(hamming_decode(deinterleave(bits)) == d,
          "12-bit burst corrected thanks to interleaving")

    check(crc16(b"123456789") == 0x29B1, "CRC-16/CCITT-FALSE known vector (0x29B1)")


def test_sync():
    section("chirp preamble matched filter")
    rng = np.random.default_rng(2)
    for name, band in BANDS.items():
        det = PreambleDetector(band)
        noise = (rng.standard_normal(int(SAMPLE_RATE * 1.5)) * 0.02).astype(np.float32)
        pos = 30_000
        from modem_core import make_chirp
        ch = make_chirp(band)
        sig = noise.copy()
        sig[pos:pos + len(ch)] += ch * 0.3
        sc = det.scores(sig)
        peak = int(np.argmax(sc))
        check(abs(peak - pos) <= 2 and sc[peak] > 4 * np.median(sc),
              f"{name}: sample-accurate lock (err={peak - pos}, "
              f"score={sc[peak]:.2f} vs noise {np.median(sc):.3f})")

    det = PreambleDetector(BANDS["ULTRA"])
    sc = det.scores(np.zeros(SAMPLE_RATE, np.float32))
    check(len(sc) and float(np.max(sc)) < det.threshold,
          "digital silence produces no false detection")


def test_codec():
    section("packet encode -> decode, every profile in every band")
    rng = np.random.default_rng(3)
    for bnd in BANDS:
        for pid, p in PROFILES.items():
            good = 0
            for _ in range(4):
                payload = bytes(rng.integers(0, 256, 40, dtype=np.uint8))
                pkt = encode_packet(payload, pid, bnd)
                pre = int(rng.integers(1000, 9000))
                stream = np.concatenate([
                    (rng.standard_normal(pre) * 0.01).astype(np.float32),
                    (pkt * 0.5 + rng.standard_normal(len(pkt)) * 0.01).astype(np.float32),
                    (rng.standard_normal(20_000) * 0.01).astype(np.float32)])
                det = PreambleDetector(BANDS[bnd])
                peak = int(np.argmax(det.scores(stream)))
                r = try_decode(stream, [pid], peak + data_offset_in_packet(bnd), bnd)
                good += bool(r.ok and r.payload == payload)
            check(good == 4, f"{bnd}/{p.name}: 4/4 payloads recovered "
                             f"({packet_duration_s(40, pid, bnd):.1f}s of audio, "
                             f"{p.bitrate:.0f} raw bit/s)")

    section("per-frame profile adaptation")
    for bnd in BANDS:
        for pid, p in PROFILES.items():
            pkt = encode_packet(b"auto-detect me", pid, bnd)
            stream = np.concatenate([np.zeros(5000, np.float32), pkt,
                                     np.zeros(5000, np.float32)])
            det = PreambleDetector(BANDS[bnd])
            peak = int(np.argmax(det.scores(stream)))
            r = try_decode(stream, ALL_PROFILES,
                           peak + data_offset_in_packet(bnd), bnd)
            check(r.ok and r.profile_id == pid and r.payload == b"auto-detect me",
                  f"{bnd}/{p.name} decoded without the receiver being told "
                  f"the profile")


def test_tuning():
    section("retuning the band (/freq)")
    from modem_core import (set_band_base_freq, check_band_freq, band_revision,
                            tone_freqs, MAX_BAND_FREQ)
    original = BANDS["ULTRA"]
    try:
        rev0 = band_revision()
        b = set_band_base_freq("ULTRA", 19_000)
        check(b.base_freq == 19_000 and band_revision() > rev0,
              "set_band_base_freq moves the base and bumps the revision")
        check(abs(b.chirp_f0 - 18_600) < 1 and abs(b.chirp_f1 - 20_400) < 1,
              f"chirp follows the base ({b.chirp_f0:.0f}-{b.chirp_f1:.0f} Hz)")
        check(abs(tone_freqs(PROFILES[0], b)[0] - 19_000) < 1,
              "tones follow the base too")

        # a packet built at the new tuning must decode at the new tuning
        pkt = encode_packet(b"tuned", 1, "ULTRA")
        stream = np.concatenate([np.zeros(4000, np.float32), pkt,
                                 np.zeros(4000, np.float32)])
        det = PreambleDetector(BANDS["ULTRA"])
        peak = int(np.argmax(det.scores(stream)))
        r = try_decode(stream, ALL_PROFILES, peak + data_offset_in_packet("ULTRA"),
                       "ULTRA")
        check(r.ok and r.payload == b"tuned", "packet round-trips at 19.0 kHz")

        # ...and must NOT be readable by a receiver left on the old frequency.
        # A small retune still overlaps the old chirp sweep, so the preamble may
        # well still be seen -- what has to fail is the DATA, whose tones moved.
        old = PreambleDetector(original)
        peak_old = int(np.argmax(old.scores(stream)))
        BANDS["ULTRA"] = original          # decode as the stale receiver would
        r_old = try_decode(stream, ALL_PROFILES,
                           peak_old + data_offset_in_packet("ULTRA"), "ULTRA")
        check(not r_old.ok,
              f"a receiver left on the old frequency cannot read it "
              f"({r_old.reason}) -- the tuning really is a channel")
        set_band_base_freq("ULTRA", 19_000)

        # A far retune is not even detected.
        set_band_base_freq("ULTRA", 8_000)
        far = encode_packet(b"far away", 1, "ULTRA")
        far_stream = np.concatenate([np.zeros(4000, np.float32), far,
                                     np.zeros(4000, np.float32)])
        check(float(np.max(old.scores(far_stream))) < old.threshold,
              "a far retune (8 kHz vs 18.2 kHz) is invisible to the old tuning")

        check("too high" in check_band_freq(MAX_BAND_FREQ + 1000),
              "frequencies above the Nyquist headroom are rejected")
        check("too low" in check_band_freq(50), "frequencies near DC are rejected")
        check(check_band_freq(8_000) == "", "a sane midband frequency is accepted")

        set_band_base_freq("ULTRA", 4_000)
        check(BANDS["ULTRA"].audible, "a band tuned to 4 kHz reports as audible")
    finally:
        BANDS["ULTRA"] = original


def test_symbol_errors():
    section("tolerance to random symbol errors (interleaving pays off here)")
    rng = np.random.default_rng(4)
    pid = 1
    p = PROFILES[pid]
    payload = b"x" * 30
    for rate in (0.0, 0.01, 0.02):
        good = 0
        for _ in range(20):
            bits = build_frame_bits(payload, pid)
            bps = p.bits_per_symbol
            b2 = np.concatenate([bits, np.zeros((-len(bits)) % bps, np.uint8)])
            syms = (b2.reshape(-1, bps) @ (1 << np.arange(bps)[::-1]))
            hit = rng.random(len(syms)) < rate
            syms[hit] = rng.integers(0, p.n_tones, hit.sum())
            back = ((syms[:, None] >> np.arange(bps - 1, -1, -1)) & 1) \
                .astype(np.uint8).reshape(-1)
            from modem_core import (deinterleave as di, hamming_decode as hd,
                                    padded_body_bytes, hamming_bit_count,
                                    FRAME_OVERHEAD_BYTES, HEADER_BODY_BYTES)
            need = hamming_bit_count(padded_body_bytes(len(payload)))
            body = hd(di(back[:need]))[:len(payload) + FRAME_OVERHEAD_BYTES]
            good += (crc16(body[:-2]) == int.from_bytes(body[-2:], "big")
                     and body[HEADER_BODY_BYTES:HEADER_BODY_BYTES + len(payload)] == payload)
        print(f"        symbol error rate {rate*100:4.1f}%  ->  {good}/20 frames intact")
        if rate == 0.0:
            check(good == 20, "0% symbol errors -> every frame intact")
        if rate == 0.01:
            check(good >= 12, "1% symbol errors -> most frames still intact")


# --------------------------------------------------------------------------- #
def test_rx_state_machine():
    """Drive the real Transceiver._rx_loop from a fake sound card."""
    section("streaming receiver (fake sound card)")
    import trx as T
    from trx import FRAG_HDR, T_TEXT, FRAG_PAYLOAD, BROADCAST

    state = {"stream": None, "speed": 30.0}

    class FakeStream:
        def __init__(self, **kw):
            self.cb, self.bs, self._stop = kw["callback"], kw["blocksize"], False

        def __enter__(self):
            self.t = threading.Thread(target=self._run, daemon=True)
            self.t.start()
            return self

        def __exit__(self, *a):
            self._stop = True

        def _run(self):
            s, i = state["stream"], 0
            while not self._stop and i + self.bs <= len(s):
                self.cb(s[i:i + self.bs].reshape(-1, 1).astype(np.float32),
                        self.bs, None, None)
                i += self.bs
                time.sleep(self.bs / SAMPLE_RATE / state["speed"])

    real_sd, real_have = T.sd, T.HAVE_AUDIO
    T.sd = types.SimpleNamespace(InputStream=lambda **kw: FakeStream(**kw),
                                 check_input_settings=lambda **kw: None)
    T.HAVE_AUDIO = True
    try:
        def run(label, frames, hint, expect, noise=0.01, speed=30.0):
            rng = np.random.default_rng(5)
            parts = [np.zeros(int(SAMPLE_RATE * 0.3), np.float32)]
            for f in frames:
                parts += [f, np.zeros(int(SAMPLE_RATE * 0.15), np.float32)]
            parts.append(np.zeros(int(SAMPLE_RATE * 0.3), np.float32))
            s = np.concatenate(parts)
            state["stream"] = s + rng.standard_normal(len(s)).astype(np.float32) * noise
            state["speed"] = speed
            got = []
            t = T.Transceiver(on_message=lambda k, d, m: got.append(d))
            t.active_profile = hint
            t.start_rx()
            deadline = time.time() + len(s) / SAMPLE_RATE / speed + 5
            while time.time() < deadline and not got:
                time.sleep(0.05)
            time.sleep(0.2)
            t.stop_rx()
            check(bool(got) and got[0] == expect,
                  f"{label} (det={t.stats['detections']} ok={t.stats['rx_ok']} "
                  f"bad={t.stats['rx_bad']})")

        def frame(text, pid, idx=0, total=1, mid=1, gain=0.5, band="ULTRA"):
            # Broadcast dst: this suite exercises the receiver, not the ARQ
            # conversation, and a broadcast frame is deliberately not acked.
            return encode_packet(FRAG_HDR.pack(T_TEXT, BROADCAST, BROADCAST,
                                               mid, idx, total)
                                 + text.encode(), pid, band) * gain

        for pid, p in PROFILES.items():
            run(f"{p.name}: single frame end-to-end",
                [frame("hi there", pid)], pid, "hi there")

        run("receiver adapts when TX used a different profile in the band",
            [frame("adapt me", 0)], 1, "adapt me")
        run("weak signal (0.12 gain) + louder noise",
            [frame("quiet", 2, gain=0.12)], 2, "quiet", noise=0.02)
        run("real-time pacing (1x)", [frame("realtime", 1)], 1, "realtime", speed=1.0)

        body = "A" * 40 + "B" * 30
        parts = [body[i:i + FRAG_PAYLOAD] for i in range(0, len(body), FRAG_PAYLOAD)]
        run("multi-fragment message reassembled",
            [frame(f, 0, i, len(parts), mid=5) for i, f in enumerate(parts)],
            0, body)

        # The audible band is a full peer, not a fallback: same profiles, and a
        # receiver whose own setting says ULTRA still picks an AUDIO packet up.
        run("audible band, receiver configured for the inaudible one",
            [frame("audible band works", 1, band="AUDIO")], 1,
            "audible band works")

        # Retuning a LIVE receiver must rebuild its matched filters.
        from modem_core import set_band_base_freq, BANDS as _B
        keep = _B["ULTRA"]
        try:
            set_band_base_freq("ULTRA", 16_000)
            run("retuned to 16 kHz: a live receiver follows the new frequency",
                [frame("retuned link", 1)], 1, "retuned link")
        finally:
            _B["ULTRA"] = keep
            set_band_base_freq("ULTRA", keep.base_freq)

        # silence must not produce phantom messages
        state["stream"] = (np.random.default_rng(6)
                           .standard_normal(int(SAMPLE_RATE * 2))
                           .astype(np.float32) * 0.01)
        state["speed"] = 30.0
        got = []
        t = T.Transceiver(on_message=lambda k, d, m: got.append(d))
        t.start_rx()
        time.sleep(2.0 / 30 + 1.5)
        t.stop_rx()
        check(not got, "pure noise produces no phantom messages")
    finally:
        T.sd, T.HAVE_AUDIO = real_sd, real_have


class _Ear:
    """One device's microphone: a buffer that drains at the playback rate."""

    def __init__(self, noise):
        self.buf = np.zeros(0, np.float32)
        self.lock = threading.Lock()
        self.noise = noise

    def push(self, audio):
        with self.lock:
            self.buf = np.concatenate([self.buf, audio])

    def take(self, n, rng):
        with self.lock:
            if len(self.buf) >= n:
                out, self.buf = self.buf[:n], self.buf[n:]
            elif len(self.buf):
                out = np.concatenate([self.buf,
                                      np.zeros(n - len(self.buf), np.float32)])
                self.buf = np.zeros(0, np.float32)
            else:
                out = np.zeros(n, np.float32)
        return out + rng.standard_normal(n).astype(np.float32) * self.noise


def _virtual_room(speed, noise, names=("A", "B")):
    """A fake `sd` where every device hears every OTHER device, in scaled time.

    Devices do not hear themselves: self-rejection is a physical-layer concern
    that `lead_in_ok` already covers, and leaving it out keeps this test about
    the protocol.
    """
    ears = {n: _Ear(noise) for n in names}
    sent = {n: 0 for n in names}          # packets each device put on the air

    def play(audio, sr, device=None, blocking=True):
        audio = np.asarray(audio, np.float32)
        sent[device] = sent.get(device, 0) + 1
        for name, ear in ears.items():
            if name != device:
                ear.push(audio)
        time.sleep(len(audio) / SAMPLE_RATE / speed)

    class InStream:
        def __init__(self, **kw):
            self.cb, self.bs = kw["callback"], kw["blocksize"]
            self.ear = ears[kw["device"]]
            self._stop = False

        def __enter__(self):
            self.t = threading.Thread(target=self._run, daemon=True)
            self.t.start()
            return self

        def __exit__(self, *a):
            self._stop = True

        def _run(self):
            rng = np.random.default_rng(7)
            while not self._stop:
                self.cb(self.ear.take(self.bs, rng).reshape(-1, 1), self.bs,
                        None, None)
                time.sleep(self.bs / SAMPLE_RATE / speed)

    return types.SimpleNamespace(play=play, InputStream=lambda **kw: InStream(**kw),
                                 check_input_settings=lambda **kw: None,
                                 sent=sent)


def test_handshake_arq():
    section("handshake + stop-and-wait ARQ (two virtual devices)")
    import trx as T

    real_sd, real_have = T.sd, T.HAVE_AUDIO
    T.HAVE_AUDIO = True
    try:
        def pair(speed=14.0, noise=0.004):
            T.sd = _virtual_room(speed, noise)
            got, events = [], []
            a = T.Transceiver(device_in="A", device_out="A",
                              on_event=lambda l, t: events.append((l, t)))
            b = T.Transceiver(device_in="B", device_out="B",
                              on_message=lambda k, d, m: got.append(d),
                              on_event=lambda l, t: events.append((l, t)))
            a.device_id, b.device_id = 0xA1A1, 0xB2B2   # distinct, not persisted
            a.active_profile = b.active_profile = 0     # FAST: shortest packets
            return a, b, got, events

        # -- 1. full exchange: hello -> hello-ack -> data -> ack ---------------
        a, b, got, events = pair()
        a.start_rx()
        b.start_rx()
        time.sleep(0.3)
        tx = threading.Thread(target=a.send_text, args=("ping",), daemon=True)
        tx.start()
        tx.join(timeout=90)
        time.sleep(0.5)
        a.stop_rx()
        b.stop_rx()

        check(got == ["ping"], f"message delivered end-to-end (got {got!r})")
        check(a.peer_id == 0xB2B2, f"sender learned the peer id "
                                   f"(peer_id={a.peer_id:04X})")
        check(b.peer_id == 0xA1A1, f"receiver learned the sender id "
                                   f"(peer_id={b.peer_id:04X})")
        check(any("acked by B2B2" in t for _, t in events),
              "sender saw the fragment acknowledged")
        check(not any(lvl == "error" for lvl, _ in events),
              "no errors raised during the exchange")
        # The whole point of folding the introduction into the first fragment:
        # a short word costs one packet each way, not a hello round trip first.
        check(T.sd.sent["A"] == 1 and T.sd.sent["B"] == 1,
              f"a short message costs exactly 2 packets on air "
              f"(sender {T.sd.sent['A']}, receiver {T.sd.sent['B']})")

        # -- 2. a swallowed ACK must cause a resend, not a lost fragment -------
        a, b, got, events = pair()
        real_ctrl = b._send_ctrl
        dropped = []

        def flaky(kind, dst, pid, band, mid=0, idx=0):
            if kind == T.T_ACK and not dropped:
                dropped.append((mid, idx))     # eat exactly the first ACK
                return
            return real_ctrl(kind, dst, pid, band, mid, idx)

        b._send_ctrl = flaky
        a.start_rx()
        b.start_rx()
        time.sleep(0.3)
        tx = threading.Thread(target=a.send_text, args=("retry",), daemon=True)
        tx.start()
        tx.join(timeout=120)
        time.sleep(0.5)
        a.stop_rx()
        b.stop_rx()

        check(bool(dropped), "the test actually swallowed an ACK")
        check(any("unacked" in t for _, t in events),
              "sender noticed the missing ACK")
        check(got == ["retry"],
              f"fragment was resent and delivered exactly once (got {got!r})")
    finally:
        T.sd, T.HAVE_AUDIO = real_sd, real_have


def test_rx_thread_reports_errors():
    section("RX thread failures are reported, not swallowed")
    import trx as T
    real_sd, real_have = T.sd, T.HAVE_AUDIO

    def boom(**kw):
        raise RuntimeError("device exploded")

    T.sd = types.SimpleNamespace(InputStream=boom,
                                 check_input_settings=lambda **kw: None)
    T.HAVE_AUDIO = True
    try:
        seen = []
        t = T.Transceiver(on_event=lambda lvl, txt: seen.append((lvl, txt)))
        t.start_rx()
        time.sleep(0.5)
        t.stop_rx()
        check(any(lvl == "error" and "device exploded" in txt for lvl, txt in seen),
              "an exception inside the RX thread surfaces via on_event")
    finally:
        T.sd, T.HAVE_AUDIO = real_sd, real_have


if __name__ == "__main__":
    test_fec()
    test_sync()
    test_codec()
    test_tuning()
    test_symbol_errors()
    test_rx_state_machine()
    test_handshake_arq()
    test_rx_thread_reports_errors()
    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED:")
        for f in FAILURES:
            print("  -", f)
        sys.exit(1)
    print("all tests passed")
