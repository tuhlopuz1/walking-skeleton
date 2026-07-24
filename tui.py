"""
tui.py
======
Terminal UI for the acoustic half-duplex transceiver (Windows-friendly).

Run:  python tui.py
Build: pyinstaller --onefile --name acoustic_modem tui.py

Commands (type in the input box):
    /rx on | off      start / stop listening
    /profile 0..3     modulation profile (FAST/NORMAL/ROBUST/AUDIBLE)
    /file <path>      send a file
    /devices          list audio devices
    /in <n>  /out <n> pick input / output device by number
    /probe            measure what this speaker+mic pair can actually carry
    /selftest         encode->decode in memory (no sound card)
    /loopback         full speaker->mic round trip on this machine
    /thresh <x>       preamble detection threshold (default 0.15)
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

LOG: list[str] = []
STATUS = {"profile": 1, "note": "", "busy": False}
_loglock = threading.Lock()


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
    p = PROFILES[STATUS["profile"]]
    audio = "OK" if HAVE_AUDIO else f"NO AUDIO ({AUDIO_ERROR[:30]})"
    rx = "ON " if trx.rx_running else "OFF"
    return [("class:title",
             f" ACOUSTIC MODEM | RX:{rx} | profile {STATUS['profile']}:{p.name} "
             f"({p.base_freq/1000:.2f}kHz {p.n_tones}-FSK {p.band}) | audio:{audio} "
             f"| {STATUS['note']} ")]


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
        log("/rx on|off  /profile 0..3  /file <path>  /devices  /in <n>  /out <n>")
        log("/probe  /selftest  /loopback  /thresh <x>  /clear  /quit")
        log("profiles: 0 FAST  1 NORMAL  2 ROBUST (18-19kHz)  3 AUDIBLE (3-5kHz, "
            "use if 18kHz does not get through)")
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
    elif head == "/profile":
        try:
            pid = int(cmd[1])
            if pid not in PROFILES:
                raise ValueError
        except (IndexError, ValueError):
            log("usage: /profile 0|1|2|3")
            return
        STATUS["profile"] = pid
        trx.active_profile = pid
        p = PROFILES[pid]
        log(f"profile -> {p.name} ({p.band} band, {p.n_tones}-FSK, "
            f"{p.bitrate:.0f} raw bit/s). Set the SAME profile on the other device.")
    elif head == "/thresh":
        try:
            trx.detect_threshold = float(cmd[1])
        except (IndexError, ValueError):
            log("usage: /thresh 0.15")
            return
        log(f"detection threshold -> {trx.detect_threshold:.2f} "
            f"({'restart RX to apply' if trx.rx_running else 'applies on /rx on'})")
    elif head == "/selftest":
        log(">> selftest (no sound card)")
        _bg(_run_lines, trxmod.selftest, -1)
    elif head == "/probe":
        log(">> probing speaker -> mic frequency response ...")
        _bg(_run_lines, trxmod.probe, trx.device_in, trx.device_out)
    elif head == "/loopback":
        log(f">> loopback on profile {STATUS['profile']} ...")
        _bg(_run_lines, trxmod.loopback, STATUS["profile"], "loopback test",
            trx.device_in, trx.device_out)
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
    log("Acoustic modem TUI.  /help for commands.")
    log("First run on a new machine:  /probe   (checks the speaker+mic can carry the band)")
    log("Then on BOTH devices:  /rx on   and use the SAME /profile.")
    if not HAVE_AUDIO:
        log(f"!! sounddevice unavailable: {AUDIO_ERROR}")
    app = build_app()
    try:
        app.run()
    finally:
        trx.stop_rx()


if __name__ == "__main__":
    main()
