"""Wire-compatibility check between the Python reference and the Rust port.

Both sides speak raw little-endian f32 mono at 48 kHz, so neither needs a sound
card. Run from the repo root after `cargo build --release`:

    python tools/cross_check.py

Exit code is non-zero if any direction fails, which makes it usable in CI.
"""
from __future__ import annotations
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from modem_core import (
    BANDS, PROFILES, encode_packet, try_decode, data_offset_in_packet,
    PreambleDetector, set_band_base_freq,
)

EXE = os.path.join("target", "release", "modem.exe")
if not os.path.exists(EXE):
    EXE = os.path.join("target", "release", "modem")
if not os.path.exists(EXE):
    sys.exit("build first:  cargo build --release")

FAILS: list[str] = []


def check(ok: bool, label: str):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        FAILS.append(label)


def run(*args) -> subprocess.CompletedProcess:
    return subprocess.run([EXE, *args], capture_output=True, text=True)


def py_decode(audio: np.ndarray, band_name: str) -> tuple[bool, bytes, float]:
    band = BANDS[band_name]
    det = PreambleDetector(band)
    sc = det.scores(audio)
    if not len(sc):
        return False, b"", 0.0
    peak = int(np.argmax(sc))
    r = try_decode(audio, list(PROFILES), peak + data_offset_in_packet(band_name),
                   band_name)
    return r.ok, r.payload, float(sc[peak])


def main():
    tmp = tempfile.mkdtemp(prefix="xcheck_")

    print("=== Rust encodes -> Python decodes ===")
    for band_name in ("ULTRA", "AUDIO"):
        for pid in PROFILES:
            text = f"rust->py {PROFILES[pid].name} {band_name}"
            path = os.path.join(tmp, f"r_{band_name}_{pid}.f32")
            p = run("encode", text, path, "--profile", str(pid), "--band", band_name)
            if p.returncode != 0:
                check(False, f"{band_name}/{PROFILES[pid].name}: rust encode failed: "
                             f"{p.stderr.strip()}")
                continue
            audio = np.fromfile(path, dtype="<f4")
            ok, payload, score = py_decode(audio, band_name)
            check(ok and payload.decode() == text,
                  f"{band_name}/{PROFILES[pid].name:6s} score={score:.2f} "
                  f"payload={payload[:24]!r}")

    print("\n=== Python encodes -> Rust decodes ===")
    for band_name in ("ULTRA", "AUDIO"):
        for pid in PROFILES:
            text = f"py->rust {PROFILES[pid].name} {band_name}"
            audio = encode_packet(text.encode(), pid, band_name)
            # Pad, so the detector has the quiet run-up a real capture would have.
            audio = np.concatenate([np.zeros(3000, np.float32), audio,
                                    np.zeros(3000, np.float32)])
            path = os.path.join(tmp, f"p_{band_name}_{pid}.f32")
            audio.astype("<f4").tofile(path)
            p = run("decode", path, "--band", band_name)
            got = [l for l in p.stdout.splitlines() if l.startswith("payload: ")]
            check(p.returncode == 0 and got and got[0][len("payload: "):] == text,
                  f"{band_name}/{PROFILES[pid].name:6s} -> {p.stdout.splitlines()[-1][:60]}")

    print("\n=== retuned band (the tuning is a channel, both must agree) ===")
    freq_khz = 19.0
    set_band_base_freq("ULTRA", freq_khz * 1000)
    text = "retuned link"
    audio = encode_packet(text.encode(), 1, "ULTRA")
    audio = np.concatenate([np.zeros(3000, np.float32), audio,
                            np.zeros(3000, np.float32)])
    path = os.path.join(tmp, "retuned.f32")
    audio.astype("<f4").tofile(path)
    p = run("decode", path, "--band", "ULTRA", "--freq", str(freq_khz))
    got = [l for l in p.stdout.splitlines() if l.startswith("payload: ")]
    check(p.returncode == 0 and got and got[0][len("payload: "):] == text,
          f"python @ {freq_khz} kHz -> rust @ {freq_khz} kHz")
    # ... and the same audio must NOT decode at the default tuning.
    p = run("decode", path, "--band", "ULTRA")
    check(p.returncode != 0, "rust on the old frequency cannot read it")

    print("\n" + "=" * 60)
    if FAILS:
        print(f"{len(FAILS)} FAILED:")
        for f in FAILS:
            print("  -", f)
        sys.exit(1)
    print("wire formats agree")


if __name__ == "__main__":
    main()
