"""
modem_core.py
=============
Acoustic near-ultrasonic modem core (physical + link layer).

Design goals baked in for future extension:
 - Modulation is pluggable (MFSKModulator). M-FSK implemented now; add CSS/Chirp
   later without touching framing/FEC/CRC.
 - Frame format carries a HEADER with payload length + modulation profile id, so a
   receiver can adapt (adaptive rate / adaptive frequency) per-frame.
 - Chirp preamble + matched filter -> sample-accurate sync (an energy threshold on
   a plain tone cannot align symbols well enough to demodulate).
 - Sample rate fixed at 48 kHz (Nyquist -> 24 kHz, room for the 18-20 kHz band).

Dependencies: numpy only (audio I/O lives in trx.py so this file is import-safe
even on machines without a sound device -- good for unit tests / reuse as a module).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, replace
from functools import lru_cache

SAMPLE_RATE = 48_000            # Hz, per Nyquist requirement in spec

# Packet layout timing (milliseconds)
LEADIN_MS = 120.0               # silence before the chirp. Two jobs: sound cards clip
                                # the first few ms of playback, and the receiver uses
                                # this quiet run-up to reject false preamble hits
                                # (see PreambleDetector.lead_in_ok).
GAP_MS = 20.0                   # chirp -> data gap, lets the chirp's echo decay
TAIL_MS = 60.0                  # silence after the frame
LEAD_GUARD_MS = 50.0            # how much of the run-up the receiver checks


# --------------------------------------------------------------------------- #
# CRC-16 (CCITT-FALSE) -- 2 bytes appended to each frame
# --------------------------------------------------------------------------- #
def _crc16_table() -> list[int]:
    tbl = []
    for b in range(256):
        crc = b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
        tbl.append(crc)
    return tbl


_CRC_TBL = _crc16_table()


def crc16(data: bytes, init: int = 0xFFFF) -> int:
    crc = init
    for b in data:
        crc = ((crc << 8) & 0xFFFF) ^ _CRC_TBL[((crc >> 8) ^ b) & 0xFF]
    return crc & 0xFFFF


# --------------------------------------------------------------------------- #
# Hamming(7,4) FEC -- corrects 1 bit error per nibble. Simple, fast, good enough
# as a first line of defense. Reed-Solomon can replace this later (see README).
# --------------------------------------------------------------------------- #
_H_G = np.array([  # generator: 4 data bits -> 7 code bits
    [1, 1, 0, 1],
    [1, 0, 1, 1],
    [1, 0, 0, 0],
    [0, 1, 1, 1],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
], dtype=np.uint8)
_H_H = np.array([  # parity-check matrix
    [1, 0, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
], dtype=np.uint8)


def hamming_encode(data: bytes) -> np.ndarray:
    """Return array of bits (0/1) after Hamming(7,4) encoding. 1 byte -> 14 bits."""
    if not data:
        return np.zeros(0, dtype=np.uint8)
    arr = np.frombuffer(data, dtype=np.uint8)
    nibs = np.empty(arr.size * 2, dtype=np.uint8)
    nibs[0::2] = arr >> 4
    nibs[1::2] = arr & 0x0F
    d = ((nibs[:, None] >> np.array([3, 2, 1, 0], dtype=np.uint8)) & 1).astype(np.uint8)
    code = (d @ _H_G.T) % 2                      # (n_nibbles, 7)
    return code.reshape(-1).astype(np.uint8)


def hamming_decode(bits: np.ndarray) -> bytes:
    """Decode a bit array back to bytes, correcting single-bit errors per nibble."""
    n = len(bits) // 7
    if n == 0:
        return b""
    code = np.asarray(bits[:n * 7], dtype=np.uint8).reshape(n, 7).copy()
    syn = (code @ _H_H.T) % 2                    # (n, 3)
    pos = (syn[:, 0] + 2 * syn[:, 1] + 4 * syn[:, 2]).astype(np.int64)
    bad = np.nonzero(pos)[0]
    if bad.size:
        code[bad, pos[bad] - 1] ^= 1             # flip the erroneous bit
    nib = (code[:, 2] << 3) | (code[:, 4] << 2) | (code[:, 5] << 1) | code[:, 6]
    m = (nib.size // 2) * 2
    return ((nib[0:m:2] << 4) | nib[1:m:2]).astype(np.uint8).tobytes()


def hamming_bit_count(n_bytes: int) -> int:
    return n_bytes * 2 * 7


# --------------------------------------------------------------------------- #
# Block interleaver.
#
# Hamming(7,4) fixes one bit per codeword, but a single M-FSK symbol error
# corrupts up to `bits_per_symbol` CONSECUTIVE bits -- which without
# interleaving all land in the same codeword and are uncorrectable. Writing
# codewords as rows and reading out columns spreads any burst of up to
# INTERLEAVE_CW consecutive bit errors across that many separate codewords,
# where Hamming can fix each one.
#
# The depth is chosen so the 6-byte header is exactly one interleaver block,
# which is what lets the receiver de-interleave the header before it knows how
# long the rest of the frame is.
# --------------------------------------------------------------------------- #
INTERLEAVE_CW = 12                              # codewords per block
INTERLEAVE_BYTES = INTERLEAVE_CW // 2           # = 6 bytes per block


def interleave(bits: np.ndarray) -> np.ndarray:
    n = len(bits) // (7 * INTERLEAVE_CW)
    return bits[:n * 7 * INTERLEAVE_CW].reshape(n, INTERLEAVE_CW, 7) \
                                       .transpose(0, 2, 1).reshape(-1)


def deinterleave(bits: np.ndarray) -> np.ndarray:
    n = len(bits) // (7 * INTERLEAVE_CW)
    return bits[:n * 7 * INTERLEAVE_CW].reshape(n, 7, INTERLEAVE_CW) \
                                       .transpose(0, 2, 1).reshape(-1)


# --------------------------------------------------------------------------- #
# Bands -- WHERE in the spectrum a link lives. A band is the tunable part of the
# design: it fixes the lowest data tone and the chirp preamble that finds and
# time-aligns packets. Profiles (below) say HOW to modulate and are band-
# agnostic, so "switch to the audible band" and "change speed" are independent
# knobs, and either band can be retuned to any centre frequency.
#
# The receiver runs one matched filter per band, so an audible and an inaudible
# link coexist and auto-detect.
# --------------------------------------------------------------------------- #
AUDIBLE_LIMIT = 17_000.0        # below this, most listeners can hear the tones
MIN_BAND_FREQ = 300.0           # keep the chirp clear of DC / room rumble
MAX_BAND_FREQ = 0.45 * SAMPLE_RATE   # 21.6 kHz -- leave headroom under Nyquist


@dataclass(frozen=True)
class Band:
    name: str
    base_freq: float          # lowest data tone
    lead: float               # Hz below base_freq where the chirp starts
    span: float               # Hz above base_freq where the chirp ends
    chirp_ms: float

    @property
    def chirp_f0(self) -> float:
        return self.base_freq - self.lead

    @property
    def chirp_f1(self) -> float:
        return self.base_freq + self.span

    @property
    def chirp_samples(self) -> int:
        return int(SAMPLE_RATE * self.chirp_ms / 1000)

    @property
    def audible(self) -> bool:
        return self.base_freq < AUDIBLE_LIMIT


BANDS = {
    # Near-ultrasonic: inaudible to most people, but consumer speakers/mics fall
    # off a cliff above ~19.5 kHz. The chirp deliberately stops below that: a
    # sweep whose top half the hardware cannot reproduce only correlates on the
    # part that survives, which throws away detection margin.
    "ULTRA": Band("ULTRA", 18_200, 400, 1400, 60.0),
    # Audible fallback: survives literally any speaker/mic pair, incl. Bluetooth.
    # Use this when ULTRA does not get through -- see /probe in the TUI.
    "AUDIO": Band("AUDIO", 3_000, 200, 2400, 60.0),
}
DEFAULT_BAND = "ULTRA"

# Bumped whenever a band is retuned, so a running receiver can notice and
# rebuild its matched filters instead of listening on the old frequency.
_BAND_REV = [0]


def band_revision() -> int:
    return _BAND_REV[0]


def max_tone_spread() -> float:
    """Widest base->top tone distance any profile needs."""
    return max(p.spacing * (p.n_tones - 1) for p in PROFILES.values())


def check_band_freq(hz: float) -> str:
    """Return '' if `hz` is a usable base frequency, else why it is not."""
    if hz != hz or hz <= 0:
        return "frequency must be a positive number"
    probe = Band("probe", float(hz), max(b.lead for b in BANDS.values()),
                 max(b.span for b in BANDS.values()), 60.0)
    if probe.chirp_f0 < MIN_BAND_FREQ:
        return (f"too low: the chirp would start at {probe.chirp_f0:.0f} Hz, "
                f"below the {MIN_BAND_FREQ:.0f} Hz floor")
    top = max(probe.chirp_f1, hz + max_tone_spread())
    if top > MAX_BAND_FREQ:
        return (f"too high: tones/chirp would reach {top:.0f} Hz, above the "
                f"{MAX_BAND_FREQ:.0f} Hz ceiling (Nyquist is "
                f"{SAMPLE_RATE // 2} Hz)")
    return ""


def set_band_base_freq(name: str, hz: float) -> Band:
    """Retune a band's lowest data tone. Chirp and tones move with it."""
    if name not in BANDS:
        raise ValueError(f"unknown band {name!r}; known: {', '.join(BANDS)}")
    why = check_band_freq(hz)
    if why:
        raise ValueError(why)
    BANDS[name] = replace(BANDS[name], base_freq=float(hz))
    _BAND_REV[0] += 1
    return BANDS[name]


def band_plan(name: str, profile_id: int | None = None) -> list[str]:
    """Human-readable description of where a band currently sits."""
    b = BANDS[name]
    out = [f"band {b.name}: base {b.base_freq/1000:.3f} kHz, "
           f"chirp {b.chirp_f0/1000:.3f}-{b.chirp_f1/1000:.3f} kHz "
           f"({'AUDIBLE' if b.audible else 'inaudible to most people'})"]
    for pid in (PROFILES if profile_id is None else [profile_id]):
        p = PROFILES[pid]
        f = tone_freqs(p, b)
        out.append(f"  profile {pid} {p.name:7s} {p.n_tones}-FSK  tones "
                   + ", ".join(f"{x/1000:.3f}" for x in f) + " kHz")
    return out

# Detection threshold for the normalized matched filter.
#
# A clean chirp scores ~0.7 and room noise ~0.02-0.05. The tempting move is to
# put the threshold just above the noise, but the payload's own M-FSK tones live
# inside the chirp's sweep range and partially correlate with it: measured over
# the air, the DATA region of a packet reaches 0.21-0.26. Every false lock on
# the payload blinds the receiver for a header's worth of audio.
#
# Rather than raise the threshold above 0.26 -- which would drop the real chirps
# that measure as low as 0.30 -- the payload hits are rejected structurally by
# the in-band lead-in silence check (see PreambleDetector.lead_in_ok). That
# frees this to sit low enough not to miss weak preambles, while still keeping a
# 4-10x margin over room noise. /thresh tunes it live.
DETECT_THRESHOLD = 0.20


def make_chirp(band: Band, amplitude: float = 0.7) -> np.ndarray:
    """Real linear-sweep chirp used as the packet preamble."""
    n = band.chirp_samples
    t = np.arange(n) / SAMPLE_RATE
    T = n / SAMPLE_RATE
    k = (band.chirp_f1 - band.chirp_f0) / T
    phase = 2 * np.pi * (band.chirp_f0 * t + 0.5 * k * t * t)
    return (amplitude * np.sin(phase) * np.hanning(n)).astype(np.float32)


def make_chirp_analytic(band: Band) -> np.ndarray:
    """Complex (analytic) chirp -- the matched-filter template.

    Correlating against the complex template and taking the magnitude makes
    detection insensitive to the carrier phase, which the speaker/mic/room
    rotates arbitrarily. A real-valued template loses most of its peak when the
    recorded copy comes back phase-shifted.
    """
    n = band.chirp_samples
    t = np.arange(n) / SAMPLE_RATE
    T = n / SAMPLE_RATE
    k = (band.chirp_f1 - band.chirp_f0) / T
    phase = 2 * np.pi * (band.chirp_f0 * t + 0.5 * k * t * t)
    return (np.exp(1j * phase) * np.hanning(n)).astype(np.complex64)


class PreambleDetector:
    """Streaming normalized matched filter for one band's chirp.

    `scores(x)[i]` is the normalized correlation of `x[i:i+M]` with the chirp,
    in [0, 1]. A clean recorded chirp peaks around 0.7; broadband noise sits
    near 1/sqrt(time-bandwidth product), about 0.08 for these chirps.
    """

    # Windows quieter than this RMS are treated as silence rather than being
    # normalized -- otherwise 0/0 turns a digitally silent gap into a fake peak.
    SILENCE_RMS = 1e-4

    def __init__(self, band: Band, threshold: float = DETECT_THRESHOLD):
        self.band = band
        self.threshold = threshold
        self.tpl = make_chirp_analytic(band)
        self.M = len(self.tpl)
        self.tpl_norm = float(np.sqrt(np.sum(np.abs(self.tpl) ** 2)))
        self._floor_ss = (self.SILENCE_RMS ** 2) * self.M

    def scores(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        if n < self.M:
            return np.zeros(0, dtype=np.float32)
        L = n - self.M + 1
        nfft = 1 << int(np.ceil(np.log2(n + self.M)))
        xf = np.asarray(x, dtype=np.float64)
        c = np.fft.ifft(np.fft.fft(xf, nfft) * np.conj(np.fft.fft(self.tpl, nfft)))[:L]
        # local energy of every candidate window, via prefix sums
        cs = np.concatenate([[0.0], np.cumsum(xf * xf)])
        loc = np.sqrt(np.maximum(cs[self.M:self.M + L] - cs[:L], self._floor_ss))
        return (np.abs(c) / (loc * self.tpl_norm)).astype(np.float32)

    def peaks(self, sc: np.ndarray) -> list[int]:
        """Indices of distinct local maxima above the threshold, strongest first."""
        above = np.nonzero(sc >= self.threshold)[0]
        out, k = [], 0
        while k < len(above):                     # collapse each cluster to its peak
            lo = hi = int(above[k])
            while k < len(above) and above[k] <= hi + self.M:
                hi = int(above[k])
                k += 1
            out.append(lo + int(np.argmax(sc[lo:hi + 1])))
        return sorted(out, key=lambda i: -sc[i])

    def band_rms(self, seg: np.ndarray) -> float:
        """RMS of `seg` restricted to this band, comparable across window sizes."""
        n = len(seg)
        if n < 64:
            return 0.0
        spec = np.abs(np.fft.rfft(np.asarray(seg, dtype=np.float64) * np.hanning(n)))
        fb = np.fft.rfftfreq(n, 1 / SAMPLE_RATE)
        m = (fb >= self.band.chirp_f0) & (fb <= self.band.chirp_f1)
        return float(np.sqrt(2.0 * np.sum(spec[m] ** 2))) / n

    def lead_in_ok(self, x: np.ndarray, i: int, ratio: float = 0.5) -> bool:
        """True if the audio just before index `i` is quiet enough to be a lead-in.

        Every packet starts with LEADIN_MS of silence, so a genuine chirp always
        has a quiet run-up. A false peak found inside a packet's own payload is
        preceded by full-volume data tones, and fails here. This is what keeps a
        modest threshold from turning into a storm of self-detections.

        The comparison is IN-BAND on purpose. Broadband RMS does not work: a
        near-ultrasonic chirp comes back from the mic far weaker than ordinary
        room rumble, so a broadband test rejects perfectly good preambles.
        """
        g = int(SAMPLE_RATE * LEAD_GUARD_MS / 1000)
        lo = i - g
        if lo < 0 or i + self.M > len(x):
            return True                           # not enough history to judge
        return self.band_rms(x[lo:i]) <= ratio * self.band_rms(x[i:i + self.M])


# --------------------------------------------------------------------------- #
# Modulation profiles -- HOW to modulate: tone spacing and symbol timing, no
# absolute frequencies. A profile is band-agnostic, so the same FAST/NORMAL/
# ROBUST choice applies whether the link is running inaudibly at 18 kHz or
# audibly at 3 kHz. Where the tones actually land comes from the Band.
# This is also the extension point for adaptivity; ids travel in the header.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Profile:
    name: str
    spacing: float            # Hz between tones
    bits_per_symbol: int      # M-FSK: 2 -> 4 tones, 3 -> 8 tones
    symbol_ms: float          # tone duration
    guard_ms: float           # silence between symbols (echo decay)

    @property
    def n_tones(self) -> int:
        return 1 << self.bits_per_symbol

    @property
    def symbol_samples(self) -> int:
        return int(SAMPLE_RATE * self.symbol_ms / 1000)

    @property
    def guard_samples(self) -> int:
        return int(SAMPLE_RATE * self.guard_ms / 1000)

    @property
    def step_samples(self) -> int:
        return self.symbol_samples + self.guard_samples

    @property
    def bitrate(self) -> float:
        """Raw channel bits per second (before the 4/7 FEC overhead)."""
        return self.bits_per_symbol * SAMPLE_RATE / self.step_samples


PROFILES = {
    # FAST is 4-FSK rather than 8-FSK on purpose. Eight tones only fit in the
    # usable 18.0-19.5 kHz window at ~170 Hz spacing, and measured over the air
    # that spacing loses to reverb (~0.5-2% symbol errors, enough to fail whole
    # frames). Four tones 400 Hz apart measured at or near 0% while still
    # running 64% faster than NORMAL, because the symbols are shorter.
    0: Profile("FAST",   400, 2, 15, 5),    # base .. base+1200 Hz
    1: Profile("NORMAL", 200, 2, 25, 8),    # base .. base+ 600 Hz
    2: Profile("ROBUST", 300, 2, 45, 15),   # base .. base+ 900 Hz
}
DEFAULT_PROFILE = 1
ALL_PROFILES = list(PROFILES)


def tone_freqs(profile: Profile, band: Band) -> np.ndarray:
    """The absolute M-FSK tone frequencies for a profile in a given band."""
    return band.base_freq + profile.spacing * np.arange(profile.n_tones)


# --------------------------------------------------------------------------- #
# M-FSK modulator / demodulator
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=64)
def _dft_kernel(n: int, base: float, spacing: float, n_tones: int) -> np.ndarray:
    """(n, n_tones) complex DFT kernel at the exact tone frequencies.

    Correlating against the exact frequencies beats picking the nearest FFT bin:
    short symbols have coarse bins and the tones do not land on them. Cached
    because the fine-sync search rebuilds a demodulator on every trial offset.
    """
    t = np.arange(n) / SAMPLE_RATE
    freqs = base + spacing * np.arange(n_tones)
    return (np.exp(-2j * np.pi * np.outer(t, freqs))
            * np.hanning(n)[:, None]).astype(np.complex64)


class MFSKModulator:
    def __init__(self, profile: Profile, band: Band):
        self.p = profile
        self.band = band
        self.freqs = tone_freqs(profile, band)

    # ---- TX ---------------------------------------------------------------- #
    def _tone(self, freq: float, n: int) -> np.ndarray:
        t = np.arange(n) / SAMPLE_RATE
        env = np.ones(n)
        fade = min(96, n // 8)          # fade in/out to limit spectral splatter
        if fade:
            env[:fade] = np.linspace(0, 1, fade)
            env[-fade:] = np.linspace(1, 0, fade)
        return (0.55 * np.sin(2 * np.pi * freq * t) * env).astype(np.float32)

    def bits_to_audio(self, bits: np.ndarray) -> np.ndarray:
        p = self.p
        bps = p.bits_per_symbol
        pad = (-len(bits)) % bps
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
        symbols = (bits.reshape(-1, bps) @ (1 << np.arange(bps)[::-1])).astype(int)
        guard = np.zeros(p.guard_samples, dtype=np.float32)
        tones = {s: self._tone(self.freqs[s], p.symbol_samples)
                 for s in set(symbols.tolist())}
        out = []
        for s in symbols:
            out.append(tones[s])
            out.append(guard)
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    # ---- RX ---------------------------------------------------------------- #
    def _matrix(self) -> np.ndarray:
        return _dft_kernel(self.p.symbol_samples, self.band.base_freq,
                           self.p.spacing, self.p.n_tones)

    def demod(self, audio: np.ndarray, start: int, n_symbols: int):
        """Return (symbols, confidence). confidence = mean winner/runner-up ratio."""
        p = self.p
        if n_symbols <= 0:
            return np.zeros(0, dtype=int), 0.0
        need = (n_symbols - 1) * p.step_samples + p.symbol_samples
        if start < 0 or start + need > len(audio):
            return np.zeros(0, dtype=int), 0.0
        idx = (start + np.arange(n_symbols)[:, None] * p.step_samples
               + np.arange(p.symbol_samples)[None, :])
        segs = audio[idx].astype(np.float32)
        energy = np.abs(segs @ self._matrix())            # (n_symbols, n_tones)
        symbols = energy.argmax(axis=1)
        srt = np.sort(energy, axis=1)
        conf = float(np.mean(srt[:, -1] / (srt[:, -2] + 1e-12)))
        return symbols, conf

    def audio_to_bits(self, audio: np.ndarray, n_symbols: int, start: int = 0):
        symbols, conf = self.demod(audio, start, n_symbols)
        bps = self.p.bits_per_symbol
        shifts = np.arange(bps - 1, -1, -1)
        bits = ((symbols[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)
        return bits, conf


# --------------------------------------------------------------------------- #
# Framing: [CHIRP][gap][MARKER 2B][HEADER 4B][PAYLOAD][CRC16], all Hamming(7,4).
# --------------------------------------------------------------------------- #
PREAMBLE_MARKER = b"\xAA\x55"
HEADER_BODY_BYTES = len(PREAMBLE_MARKER) + 4      # marker + (profile, len_hi, len_lo, hdr_crc)
FRAME_OVERHEAD_BYTES = HEADER_BODY_BYTES + 2      # + frame CRC-16


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def padded_body_bytes(payload_len: int) -> int:
    """Frame body rounded up to a whole number of interleaver blocks."""
    total = payload_len + FRAME_OVERHEAD_BYTES
    return total + (-total) % INTERLEAVE_BYTES


def header_symbols(p: Profile) -> int:
    return _ceil_div(hamming_bit_count(HEADER_BODY_BYTES), p.bits_per_symbol)


def frame_symbols(payload_len: int, p: Profile) -> int:
    return _ceil_div(hamming_bit_count(padded_body_bytes(payload_len)), p.bits_per_symbol)


def frame_samples(payload_len: int, p: Profile) -> int:
    n = frame_symbols(payload_len, p)
    return (n - 1) * p.step_samples + p.symbol_samples if n else 0


def packet_duration_s(payload_len: int, profile_id: int,
                      band_name: str = DEFAULT_BAND) -> float:
    p = PROFILES[profile_id]
    total = (int(SAMPLE_RATE * (LEADIN_MS + GAP_MS + TAIL_MS) / 1000)
             + BANDS[band_name].chirp_samples + frame_samples(payload_len, p))
    return total / SAMPLE_RATE


def build_frame_bits(payload: bytes, profile_id: int) -> np.ndarray:
    length = len(payload)
    header = bytes([profile_id, (length >> 8) & 0xFF, length & 0xFF])
    header += bytes([crc16(header) & 0xFF])
    body = PREAMBLE_MARKER + header + payload
    body += crc16(body).to_bytes(2, "big")
    body += b"\x00" * ((-len(body)) % INTERLEAVE_BYTES)   # fill the last block
    return interleave(hamming_encode(body))


def encode_packet(payload: bytes, profile_id: int,
                  band_name: str = DEFAULT_BAND) -> np.ndarray:
    """Full TX audio for one packet: silence + chirp preamble + modulated frame."""
    band = BANDS[band_name]
    mod = MFSKModulator(PROFILES[profile_id], band)
    audio = np.concatenate([
        np.zeros(int(SAMPLE_RATE * LEADIN_MS / 1000), dtype=np.float32),
        make_chirp(band),
        np.zeros(int(SAMPLE_RATE * GAP_MS / 1000), dtype=np.float32),
        mod.bits_to_audio(build_frame_bits(payload, profile_id)),
        np.zeros(int(SAMPLE_RATE * TAIL_MS / 1000), dtype=np.float32),
    ])
    peak = float(np.max(np.abs(audio))) or 1.0
    return (audio * (0.9 / peak)).astype(np.float32)


def data_offset_in_packet(band_name: str = DEFAULT_BAND) -> int:
    """Samples from the start of the chirp to the first data symbol."""
    return BANDS[band_name].chirp_samples + int(SAMPLE_RATE * GAP_MS / 1000)


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #
@dataclass
class DecodeResult:
    ok: bool
    payload: bytes = b""
    profile_id: int = -1
    reason: str = ""
    offset: int = 0
    length: int = 0
    confidence: float = 0.0


def decode_header(audio: np.ndarray, start: int, profile_id: int,
                  band_name: str = DEFAULT_BAND,
                  max_payload: int = 4096) -> DecodeResult:
    """Try to read the 6-byte header body at an exact sample offset."""
    p = PROFILES[profile_id]
    mod = MFSKModulator(p, BANDS[band_name])
    bits, conf = mod.audio_to_bits(audio, header_symbols(p), start)
    if bits.size < hamming_bit_count(HEADER_BODY_BYTES):
        return DecodeResult(False, reason="short read", offset=start)
    body = hamming_decode(deinterleave(bits[:hamming_bit_count(HEADER_BODY_BYTES)]))
    if body[:2] != PREAMBLE_MARKER:
        return DecodeResult(False, reason="marker mismatch", offset=start, confidence=conf)
    header = body[2:6]
    if crc16(header[:3]) & 0xFF != header[3]:
        return DecodeResult(False, reason="header CRC fail", offset=start, confidence=conf)
    length = (header[1] << 8) | header[2]
    if length > max_payload:
        return DecodeResult(False, reason="payload too large", offset=start, confidence=conf)
    if header[0] not in PROFILES:
        return DecodeResult(False, reason="unknown profile id", offset=start, confidence=conf)
    return DecodeResult(True, profile_id=header[0], offset=start,
                        length=length, confidence=conf)


def decode_frame(audio: np.ndarray, start: int, profile_id: int, length: int,
                 band_name: str = DEFAULT_BAND) -> DecodeResult:
    """Decode the full frame once the header told us how long it is."""
    p = PROFILES[profile_id]
    total_body = length + FRAME_OVERHEAD_BYTES
    mod = MFSKModulator(p, BANDS[band_name])
    bits, conf = mod.audio_to_bits(audio, frame_symbols(length, p), start)
    need = hamming_bit_count(padded_body_bytes(length))
    if bits.size < need:
        return DecodeResult(False, reason="truncated frame", offset=start)
    body = hamming_decode(deinterleave(bits[:need]))[:total_body]
    if crc16(body[:-2]) != int.from_bytes(body[-2:], "big"):
        return DecodeResult(False, reason="frame CRC fail", offset=start, confidence=conf)
    return DecodeResult(True, payload=body[HEADER_BODY_BYTES:HEADER_BODY_BYTES + length],
                        profile_id=profile_id, offset=start, length=length, confidence=conf)


def sync_offsets(p: Profile, span_ms: float = 4.0, step_ms: float = 0.75) -> np.ndarray:
    """Candidate fine-sync offsets, searched nearest-first around the chirp estimate.

    The matched filter is sample-accurate on paper, but sound-card buffering and
    resampling shift the data start by a millisecond or two. Symbols are tens of
    ms long, so a handful of sub-symbol probes covers it.
    """
    step = max(1, int(SAMPLE_RATE * step_ms / 1000))
    k = max(1, int(SAMPLE_RATE * span_ms / 1000) // step)
    order = [0]
    for i in range(1, k + 1):
        order += [i * step, -i * step]
    return np.array(order, dtype=int)


def try_decode(audio: np.ndarray, profile_ids, data_start: int = 0,
               band_name: str = DEFAULT_BAND, max_payload: int = 4096):
    """Locate and decode a frame whose data begins near `data_start`.

    Returns a successful DecodeResult, or the most informative failure.
    Also usable as a two-stage helper: on success `.length`/`.offset` tell the
    caller exactly how many samples the rest of the frame needs.
    """
    best = DecodeResult(False, reason="no candidate")
    for pid in profile_ids:
        p = PROFILES[pid]
        for off in sync_offsets(p):
            start = data_start + int(off)
            if start < 0:
                continue
            hdr = decode_header(audio, start, pid, band_name, max_payload)
            if not hdr.ok:
                if hdr.reason != "short read":
                    best = hdr if best.reason == "no candidate" else best
                continue
            res = decode_frame(audio, start, pid, hdr.length, band_name)
            res.profile_id = hdr.profile_id
            if res.ok:
                return res
            best = res
    return best
