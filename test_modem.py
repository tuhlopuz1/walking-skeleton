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
    SAMPLE_RATE, PROFILES, BANDS, PreambleDetector, Profile,
    crc16, hamming_encode, hamming_decode, interleave, deinterleave,
    encode_packet, try_decode, data_offset_in_packet, packet_duration_s,
    build_frame_bits, MFSKModulator, frame_symbols,
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
    section("packet encode -> decode over all profiles")
    rng = np.random.default_rng(3)
    for pid, p in PROFILES.items():
        good = 0
        for _ in range(4):
            payload = bytes(rng.integers(0, 256, 40, dtype=np.uint8))
            pkt = encode_packet(payload, pid)
            pre = int(rng.integers(1000, 9000))
            stream = np.concatenate([
                (rng.standard_normal(pre) * 0.01).astype(np.float32),
                (pkt * 0.5 + rng.standard_normal(len(pkt)) * 0.01).astype(np.float32),
                (rng.standard_normal(20_000) * 0.01).astype(np.float32)])
            det = PreambleDetector(BANDS[p.band])
            peak = int(np.argmax(det.scores(stream)))
            r = try_decode(stream, [pid], peak + data_offset_in_packet(pid))
            good += bool(r.ok and r.payload == payload)
        check(good == 4, f"{p.name}: 4/4 payloads recovered "
                         f"({packet_duration_s(40, pid):.1f}s of audio, "
                         f"{p.bitrate:.0f} raw bit/s)")

    section("per-frame profile adaptation")
    for pid, p in PROFILES.items():
        from modem_core import profiles_in_band
        pkt = encode_packet(b"auto-detect me", pid)
        stream = np.concatenate([np.zeros(5000, np.float32), pkt,
                                 np.zeros(5000, np.float32)])
        det = PreambleDetector(BANDS[p.band])
        peak = int(np.argmax(det.scores(stream)))
        r = try_decode(stream, profiles_in_band(p.band),
                       peak + data_offset_in_packet(pid))
        check(r.ok and r.profile_id == pid and r.payload == b"auto-detect me",
              f"{p.name} decoded without the receiver being told the profile")


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
    from trx import FRAG_HDR, T_TEXT, FRAG_PAYLOAD

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

        def frame(text, pid, idx=0, total=1, mid=1, gain=0.5):
            return encode_packet(FRAG_HDR.pack(T_TEXT, mid, idx, total)
                                 + text.encode(), pid) * gain

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
    test_symbol_errors()
    test_rx_state_machine()
    test_rx_thread_reports_errors()
    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED:")
        for f in FAILURES:
            print("  -", f)
        sys.exit(1)
    print("all tests passed")
