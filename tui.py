"""
tui.py
======
Terminal UI for the acoustic half-duplex transceiver (Windows-friendly).

Run:  python tui.py
Build: pyinstaller --onefile --name acoustic_modem tui.py

Commands (type in the input box):
    /mode audible|inaudible   switch frequency band (must match on both devices)
    /freq <kHz>       set the frequency the link runs on, e.g. /freq 18.6
    /freq             show the current tone plan
    /profile 0..2     modulation profile (FAST/NORMAL/ROBUST) - any band
    /rx on | off      start / stop listening
    /file <path>      send a file
    /devices          list audio devices
    /in <n>  /out <n> pick input / output device by number
    /probe            measure what this speaker+mic pair can actually carry
    /selftest         encode->decode in memory (no sound card)
    /loopback         full speaker->mic round trip on this machine
    /thresh <x>       preamble detection threshold (default 0.20)
    /id [hhhh]        show or set this device's 16-bit id
    /arq on|off       per-fragment ACKs with resend (default on)
    /clear  /quit
    <anything else>   sent as a text message
"""
from __future__ import annotations
import datetime
import os
import threading

try:
    from prompt_toolkit import Application
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.layout import Layout, HSplit, Window
    from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.styles import Style
except ImportError:
    raise SystemExit(
        "prompt_toolkit is required for the TUI.\n"
        "Install with:  pip install prompt_toolkit sounddevice numpy\n"
        "(modem_core.py and trx.py work as importable modules without it.)"
    )

import trx as trxmod
from trx import Transceiver, HAVE_AUDIO, AUDIO_ERROR, PROFILES, list_devices
from modem_core import BANDS, check_band_freq

# Captured at import, before any /freq, so "/freq reset" has somewhere to go.
DEFAULT_BASE_FREQ = {name: b.base_freq for name, b in BANDS.items()}

LOG: list[str] = []
STATUS = {"note": "", "busy": False}
_loglock = threading.Lock()

# "audible" / "inaudible" is how a user thinks about this; ULTRA / AUDIO is how
# the protocol names it. Accept both, plus the obvious shorthands.
MODE_ALIASES = {
    "inaudible": "ULTRA", "ultra": "ULTRA", "ultrasonic": "ULTRA",
    "ultrasound": "ULTRA", "us": "ULTRA", "hidden": "ULTRA", "silent": "ULTRA",
    "audible": "AUDIO", "audio": "AUDIO", "hearable": "AUDIO",
    "sound": "AUDIO", "loud": "AUDIO",
}


def log(line: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    with _loglock:
        for sub in str(line).splitlines() or [""]:
            LOG.append(f"{ts}  {sub}")
        del LOG[:-500]


def on_message(kind, data, meta):
    if kind == "text":
        log(f"<< TEXT: {data}")
    elif kind == "file":
        outdir = os.path.join(os.getcwd(), "received")
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, os.path.basename(meta["name"]) or "received.bin")
        with open(path, "wb") as f:
            f.write(data)
        log(f"<< FILE: {meta['name']} ({meta['size']}B) saved -> {path}")


def on_progress(pct, note):
    STATUS["note"] = f"{note} [{pct}%]"


def on_event(level, text):
    log(f"   [{level}] {text}")


trx = Transceiver(on_message=on_message, on_progress=on_progress, on_event=on_event)
input_buffer = Buffer()


# ------------------------------------------------------------------ display -- #
def _bar(db: float, lo: float = -70.0, hi: float = -6.0, width: int = 16) -> str:
    frac = (db - lo) / (hi - lo)
    n = max(0, min(width, int(frac * width)))
    return "#" * n + "-" * (width - n)


def header_text():
    pid = trx.active_profile
    p = PROFILES[pid]
    b = BANDS[trx.active_band]
    audio = "OK" if HAVE_AUDIO else f"NO AUDIO ({AUDIO_ERROR[:30]})"
    rx = "ON " if trx.rx_running else "OFF"
    mode = "audible" if b.audible else "inaudible"
    peer = f"{trx.peer_id:04X}" if trx.peer_id else "-"
    return [("class:title",
             f" ACOUSTIC MODEM {trx.device_id:04X}>{peer} "
             f"| RX:{rx} | ARQ:{'on ' if trx.arq else 'off'} "
             f"| {mode} @ {b.base_freq/1000:.2f}kHz "
             f"| profile {pid}:{p.name} ({p.n_tones}-FSK, {p.bitrate:.0f}b/s) "
             f"| audio:{audio} | {STATUS['note']} ")]


def meter_text():
    s = trx.stats
    if not trx.rx_running:
        return [("class:meter", "  RX is OFF -- type '/rx on' to listen. "
                                "'/probe' checks your hardware, '/loopback' tests the link. ")]
    return [("class:meter",
             f"  mic {s['rms_db']:6.1f}dB [{_bar(s['rms_db'])}]  "
             f"band {s['band_db']:6.1f}dB  "
             f"preamble now={s['peak_score']:.2f} best={s['peak_hold']:.2f} "
             f"noise={s['noise_score']:.2f} (need>={trx.detect_threshold:.2f})  "
             f"{s['state']}  ok={s['rx_ok']} bad={s['rx_bad']} det={s['detections']} "
             f"rej={s['rejected']} ")]


def body_text():
    with _loglock:
        return "\n".join(LOG[-200:]) or "Type a message and press Enter. /help for commands."


# ----------------------------------------------------------------- commands -- #
def _bg(fn, *a):
    """Run a blocking audio op off the UI thread so the TUI stays responsive."""
    if STATUS["busy"]:
        log("busy -- wait for the current operation to finish")
        return

    def run():
        STATUS["busy"] = True
        try:
            fn(*a)
        except Exception as exc:
            log(f"!! {type(exc).__name__}: {exc}")
        finally:
            STATUS["busy"] = False
            STATUS["note"] = ""
    threading.Thread(target=run, daemon=True).start()


def _run_lines(fn, *a):
    for line in fn(*a):
        log("   " + line)


def handle_command(text: str):
    cmd = text.split()
    head = cmd[0].lower() if cmd else ""

    if head in ("/help", "/?"):
        log("/mode audible|inaudible   switch band (set the SAME on both devices)")
        log("/freq <kHz>               tune the link, e.g. /freq 18.6   (/freq = show)")
        log("/profile 0|1|2            0 FAST  1 NORMAL  2 ROBUST -- works in any band")
        log("/rx on|off  /file <path>  /devices  /in <n>  /out <n>")
        log("/id [hhhh]                this device's id (/id = show, /id A1B2 = set)")
        log("/arq on|off               per-fragment ACKs + resend (needs /rx on)")
        log("/ping                     ask who is out there, without sending anything")
        log("/probe  /selftest  /loopback  /thresh <x>  /clear  /quit")
    elif head == "/quit":
        trx.stop_rx()
        if app is not None:
            app.exit()
    elif head == "/clear":
        with _loglock:
            LOG.clear()
    elif head == "/devices":
        for line in list_devices():
            log("   " + line)
        log(f"   current: in={trx.device_in} out={trx.device_out} (None = system default)")
    elif head in ("/in", "/out"):
        try:
            n = int(cmd[1])
        except (IndexError, ValueError):
            log(f"usage: {head} <device number>  (see /devices)")
            return
        was = trx.rx_running
        trx.stop_rx()
        setattr(trx, "device_in" if head == "/in" else "device_out", n)
        log(f"{head[1:]} device -> {n}")
        if was:
            try:
                trx.start_rx()
            except Exception as exc:
                log(f"!! could not reopen input: {exc}")
    elif head == "/rx":
        if len(cmd) > 1 and cmd[1].lower() == "off":
            trx.stop_rx()
            log("RX stopped")
        else:
            try:
                trx.start_rx()
                log("RX started")
            except Exception as exc:
                log(f"!! RX error: {type(exc).__name__}: {exc}")
                log("   try '/devices' then '/in <n>' to pick a mic that supports 48 kHz")
    elif head in ("/mode", "/band"):
        arg = cmd[1].lower() if len(cmd) > 1 else ""
        target = MODE_ALIASES.get(arg, arg.upper() if arg.upper() in BANDS else "")
        if not target:
            b = BANDS[trx.active_band]
            log(f"mode: {'audible' if b.audible else 'inaudible'} "
                f"({trx.active_band} @ {b.base_freq/1000:.3f} kHz)")
            log("usage: /mode audible   |   /mode inaudible")
            return
        trx.set_band(target)
        b = BANDS[target]
        log(f"mode -> {'AUDIBLE' if b.audible else 'inaudible'} "
            f"({target} @ {b.base_freq/1000:.3f} kHz)")
        log("   set the SAME mode on the other device -- it is a channel, not a "
            "preference")
        for line in trx.plan():
            log("   " + line)
    elif head == "/freq":
        if len(cmd) < 2:
            for line in trx.plan():
                log("   " + line)
            log("usage: /freq 18.6 (kHz) | /freq 18600 (Hz) | /freq reset")
            return
        if cmd[1].lower() in ("reset", "default"):
            hz = DEFAULT_BASE_FREQ[trx.active_band]
        else:
            try:
                val = float(cmd[1].replace(",", "."))
            except ValueError:
                log("usage: /freq 18.6 (kHz) | /freq 18600 (Hz) | /freq reset")
                return
            hz = val * 1000.0 if val < 100 else val  # accept kHz or Hz
        why = check_band_freq(hz)
        if why:
            log(f"cannot tune to {hz:.0f} Hz: {why}")
            return
        was_rx = trx.rx_running
        trx.set_frequency(hz)
        b = BANDS[trx.active_band]
        log(f"{trx.active_band} -> base {b.base_freq/1000:.3f} kHz")
        if b.audible:
            log("   NOTE: this is in the audible range -- you will hear the data.")
        log("   set the SAME frequency on the other device.")
        for line in trx.plan():
            log("   " + line)
        if not was_rx:
            log("   (takes effect for RX when you '/rx on')")
    elif head == "/profile":
        try:
            pid = int(cmd[1])
            if pid not in PROFILES:
                raise ValueError
        except (IndexError, ValueError):
            log("usage: /profile 0|1|2   (0 FAST, 1 NORMAL, 2 ROBUST)")
            return
        trx.active_profile = pid
        p = PROFILES[pid]
        log(f"profile -> {p.name} ({p.n_tones}-FSK, {p.bitrate:.0f} raw bit/s). "
            f"Receiver auto-adapts, but matching it is faster.")
        for line in trx.plan():
            log("   " + line)
    elif head == "/id":
        if len(cmd) < 2:
            log(f"this device is {trx.device_id:04X}; peer is "
                + (f"{trx.peer_id:04X}" if trx.peer_id else "unknown "
                   "(no handshake yet)"))
            log(f"   stored in {trxmod.DEVICE_ID_FILE}")
            return
        try:
            trx.set_device_id(int(cmd[1], 16))
        except ValueError as exc:
            log(f"usage: /id A1B2  (4 hex digits, not 0000) -- {exc}")
            return
        log(f"device id -> {trx.device_id:04X}")
    elif head == "/ping":
        if not trx.rx_running:
            log("/ping needs '/rx on' here -- it has to hear the reply")
            return
        log(">> hello ... (the peer must have '/rx on' too)")

        def _ping():
            peer = trx.discover()
            log(f"   peer {peer:04X} answered" if peer
                else "   nobody answered -- check /rx on, /mode and /freq "
                     "on the other device")
        _bg(_ping)
    elif head == "/arq":
        if len(cmd) > 1 and cmd[1].lower() == "off":
            trx.arq = False
            log("ARQ off -- fragments are sent blind, nothing is acknowledged")
        else:
            trx.arq = True
            log(f"ARQ on -- handshake, then {trx.arq_retries} attempts per "
                f"fragment. Needs '/rx on' here AND on the peer.")
    elif head == "/thresh":
        try:
            trx.detect_threshold = float(cmd[1])
        except (IndexError, ValueError):
            log("usage: /thresh 0.15")
            return
        log(f"detection threshold -> {trx.detect_threshold:.2f} "
            f"({'restart RX to apply' if trx.rx_running else 'applies on /rx on'})")
    elif head == "/selftest":
        log(">> selftest at the current tuning (no sound card)")
        _bg(_run_lines, trxmod.selftest, -1, trx.active_band)
    elif head == "/probe":
        log(">> probing speaker -> mic frequency response ...")
        _bg(_run_lines, trxmod.probe, trx.device_in, trx.device_out, 0.3,
            trx.active_band)
    elif head == "/loopback":
        log(f">> loopback: profile {trx.active_profile} on {trx.active_band} ...")
        _bg(_run_lines, trxmod.loopback, trx.active_profile, "loopback test",
            trx.device_in, trx.device_out, trx.active_band)
    elif head == "/file":
        path = text[5:].strip().strip('"')
        if not os.path.isfile(path):
            log(f"no such file: {path}")
            return
        log(f">> sending file {path} ...")
        _bg(trx.send_file, path)
    elif head.startswith("/"):
        log(f"unknown command {head} -- /help for the list")
    else:
        log(f">> TEXT: {text}")
        _bg(trx.send_text, text)


# --------------------------------------------------------------------- app -- #
kb = KeyBindings()


@kb.add("c-c")
@kb.add("c-q")
def _(event):
    trx.stop_rx()
    event.app.exit()


@kb.add("enter")
def _(event):
    text = input_buffer.text.strip()
    input_buffer.reset()
    if text:
        handle_command(text)


app = None      # built in main(); constructing it needs a real console


def build_app():
    root = HSplit([
        Window(height=1, content=FormattedTextControl(header_text), style="class:title"),
        Window(height=1, content=FormattedTextControl(meter_text), style="class:meter"),
        Window(content=FormattedTextControl(body_text), wrap_lines=True),
        Window(height=1, char="-"),
        Window(height=1, content=BufferControl(buffer=input_buffer)),
    ])
    return Application(layout=Layout(root, focused_element=root.children[-1]),
                       key_bindings=kb, full_screen=True, refresh_interval=0.2,
                       style=Style.from_dict({"title": "reverse", "meter": "bold"}))


def main():
    global app
    log(f"Acoustic modem TUI -- this device is {trx.device_id:04X}.  "
        f"/help for commands.")
    log("First run on a new machine:  /probe   (checks the speaker+mic can carry the band)")
    log("Then on BOTH devices:  /rx on   and use the SAME /profile.")
    log("ARQ is on: the first message does a handshake, then every fragment is "
        "acked. Both ends need '/rx on' for that.")
    if not HAVE_AUDIO:
        log(f"!! sounddevice unavailable: {AUDIO_ERROR}")
    app = build_app()
    try:
        app.run()
    finally:
        trx.stop_rx()


if __name__ == "__main__":
    main()
