"""
muon_detector.py
================
Two-tab Tkinter GUI for muon detector data acquisition using the
Digilent WaveForms ADS hardware (waveforms_ads.py).

Tab 1 – Signal Viewer
Tab 2 – Live Histogram

Processing logic
----------------
* The hardware trigger fires on the chosen trigger channel at the software
  first-trigger level for that channel.  The non-trigger channel is captured
  simultaneously (free-run, same buffer).
* Expected pulses = 1  → waveform must pass the FIRST trigger only.
* Expected pulses = 2  → waveform must pass the FIRST trigger AND have a
  second crossing of the SECOND trigger level within [start_us, stop_us].
* The full raw buffer is stored / displayed; the start/stop window is used
  only for second-trigger counting and dt/pulse-2 measurements.
* Both trigger levels are drawn as dashed lines on every signal plot.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import scipy.signal as sig
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import queue
import datetime
import os
import time

# ---------------------------------------------------------------------------
# Import ADS driver (graceful fallback when hardware is absent)
# ---------------------------------------------------------------------------
try:
    from waveforms_ads import WaveFormsADS, DWFError
    ADS_AVAILABLE = True
except Exception as _ads_import_err:
    ADS_AVAILABLE = False
    print(f"[muon_detector] waveforms_ads import failed: {_ads_import_err}")

# ===========================================================================
# Colour palette  –  dark navy background, gold / amber accents
# Chosen to be distinguishable for deuteranopia / protanopia.
# ===========================================================================

C = {
    "bg":          "#0d1b2a",   # deep navy
    "sidebar":     "#112240",   # slightly lighter panel
    "panel":       "#1a2e4a",   # card / section bg
    "border":      "#1e3a5f",   # separator tint
    "fg":          "#e8dcc8",   # warm off-white text
    "fg_dim":      "#8a9bb0",   # muted text
    "gold":        "#f0a500",   # primary accent (gold)
    "gold_dim":    "#a06800",   # darker gold for disabled
    "amber":       "#ffcf47",   # bright highlight
    "teal":        "#4ecdc4",   # secondary accent (colorblind-safe)
    "red_warn":    "#e05c5c",   # warning / error
    "entry_bg":    "#0a1628",   # entry field background
    "entry_sel":   "#1e3a5f",   # entry selection
    "btn":         "#1e3a5f",   # button face
    "btn_active":  "#2a4f7a",   # button hover
    # matplotlib colours
    "trace_ch0":   "#4ecdc4",   # teal  – ch0 traces
    "trace_ch1":   "#f0a500",   # gold  – ch1 traces
    "trig1":       "#ffcf47",   # amber – first trigger line
    "trig2":       "#e05c5c",   # red   – second trigger line
    "hist_dt":     "#4ecdc4",
    "hist_p1":     "#6dd5a8",   # muted green-teal
    "hist_p2":     "#f0a500",
    "plot_bg":     "#0d1b2a",
    "plot_axes":   "#1a2e4a",
    "grid":        "#1e3a5f",
    "tick_fg":     "#8a9bb0",
}

def _apply_global_theme(root: tk.Tk):
    """Set ttk + tk default colours on the root window."""
    root.configure(bg=C["bg"])

    style = ttk.Style(root)
    # Use a base theme then override
    try:
        style.theme_use("clam")
    except Exception:
        pass

    style.configure(".",
        background=C["bg"],
        foreground=C["fg"],
        fieldbackground=C["entry_bg"],
        bordercolor=C["border"],
        darkcolor=C["bg"],
        lightcolor=C["panel"],
        troughcolor=C["bg"],
        selectbackground=C["entry_sel"],
        selectforeground=C["amber"],
        font=("Helvetica", 9),
    )
    style.configure("TNotebook",
        background=C["bg"],
        bordercolor=C["border"],
        tabmargins=[2, 4, 2, 0],
    )
    style.configure("TNotebook.Tab",
        background=C["panel"],
        foreground=C["fg_dim"],
        padding=[12, 4],
    )
    style.map("TNotebook.Tab",
        background=[("selected", C["sidebar"]), ("active", C["btn_active"])],
        foreground=[("selected", C["amber"]), ("active", C["fg"])],
    )
    style.configure("TFrame", background=C["bg"])
    style.configure("Vertical.TScrollbar",
        background=C["panel"],
        troughcolor=C["bg"],
        arrowcolor=C["fg_dim"],
    )


def _themed_frame(parent, **kw) -> tk.Frame:
    return tk.Frame(parent, bg=C["bg"], **kw)


def _section_label(parent, text):
    tk.Label(parent,
             text=text,
             bg=C["panel"],
             fg=C["gold"],
             font=("Helvetica", 9, "bold"),
             padx=4, pady=2,
             anchor="w",
             ).pack(fill="x", pady=(8, 2))


def _field_label(parent, text):
    tk.Label(parent, text=text, bg=C["bg"], fg=C["fg_dim"],
             font=("Helvetica", 8), anchor="w").pack(anchor="w")


def _entry(parent, var) -> tk.Entry:
    e = tk.Entry(parent,
                 textvariable=var,
                 bg=C["entry_bg"],
                 fg=C["amber"],
                 insertbackground=C["amber"],
                 selectbackground=C["entry_sel"],
                 selectforeground=C["amber"],
                 relief="flat",
                 highlightthickness=1,
                 highlightbackground=C["border"],
                 highlightcolor=C["gold"],
                 font=("Helvetica", 9))
    e.pack(fill="x", pady=2)
    return e


def _button(parent, text, command, **kw) -> tk.Button:
    b = tk.Button(parent,
                  text=text,
                  command=command,
                  bg=C["btn"],
                  fg=C["fg"],
                  activebackground=C["btn_active"],
                  activeforeground=C["amber"],
                  relief="flat",
                  highlightthickness=0,
                  cursor="hand2",
                  font=("Helvetica", 9),
                  **kw)
    return b


def _checkbutton(parent, text, variable, command=None) -> tk.Checkbutton:
    kw = dict(command=command) if command else {}
    return tk.Checkbutton(parent,
                          text=text,
                          variable=variable,
                          bg=C["bg"],
                          fg=C["fg"],
                          activebackground=C["bg"],
                          activeforeground=C["amber"],
                          selectcolor=C["entry_bg"],
                          font=("Helvetica", 9),
                          **kw)


def _status_label(parent, var, **kw) -> tk.Label:
    return tk.Label(parent,
                    textvariable=var,
                    bg=C["bg"],
                    fg=C["teal"],
                    wraplength=240,
                    justify="left",
                    font=("Helvetica", 8),
                    **kw)


def _counter_label(parent, text):
    tk.Label(parent, text=text, bg=C["bg"], fg=C["fg_dim"],
             font=("Helvetica", 8)).pack(anchor="w", pady=(6, 0))


def _counter_value(parent, var):
    tk.Label(parent, textvariable=var, bg=C["bg"], fg=C["amber"],
             font=("Helvetica", 9, "bold")).pack(anchor="w")


def _style_figure(fig):
    """Apply dark theme to a matplotlib Figure."""
    fig.patch.set_facecolor(C["plot_bg"])


def _style_ax(ax, title="", xlabel="", ylabel="", title_size=9,
              label_size=8, tick_size=7):
    ax.set_facecolor(C["plot_axes"])
    ax.tick_params(colors=C["tick_fg"], labelsize=tick_size)
    for sp in ax.spines.values():
        sp.set_edgecolor(C["grid"])
    ax.xaxis.label.set_color(C["tick_fg"])
    ax.yaxis.label.set_color(C["tick_fg"])
    ax.title.set_color(C["gold"])
    ax.grid(True, color=C["grid"], linewidth=0.5, linestyle=":")
    if title:
        ax.set_title(title, fontsize=title_size, color=C["gold"])
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=label_size)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_size)


# ===========================================================================
# Shared acquisition layer
# ===========================================================================

class AcquisitionManager:
    """
    Owns the WaveFormsADS device and the background capture thread.
    Captured traces arrive in self.queue as {"ch0": ndarray, "ch1": ndarray|None}.
    """

    def __init__(self):
        self.device = None
        self.queue: queue.Queue = queue.Queue(maxsize=2000)
        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Hardware parameters written by AdsPanel before Start
        self.sample_rate_hz        = 100e6
        self.trigger_channel       = 0
        self.auto_timeout_s        = 0.0
        self.acquisition_timeout_s = 5.0
        self.ch0_range_v           = 5.0
        self.ch1_range_v           = 5.0
        self.ch0_attenuation       = 1.0
        self.ch1_attenuation       = 1.0
        self.use_ch1               = False
        # trigger_level_v is derived from the UI first-trigger of the trigger channel
        self.trigger_level_v       = 0.2

        self.status_var = None  # tk.StringVar set by AdsPanel

    # ------------------------------------------------------------------ #

    def connect(self) -> str:
        if not ADS_AVAILABLE:
            return "ERROR: waveforms_ads not importable"
        with self._lock:
            if self.device is not None:
                return "Already connected"
            try:
                self.device = WaveFormsADS()
                return f"Connected – DWF {self.device.get_version()}"
            except Exception as e:
                self.device = None
                return f"ERROR: {e}"

    def disconnect(self):
        self.stop_acquisition()
        with self._lock:
            if self.device is not None:
                try:
                    self.device.close()
                except Exception:
                    pass
                self.device = None

    def start_acquisition(self) -> str:
        with self._lock:
            if self.device is None:
                return "Not connected"
            if self._thread is not None and self._thread.is_alive():
                return "Already running"
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return "Acquisition started"

    def stop_acquisition(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------ #

    def _capture_loop(self):
        # Compute buffer size from sample_rate and a 200 µs window (generous)
        buf = max(4096, int(self.sample_rate_hz * 400e-6))
        buf = min(buf, 32768)

        while not self._stop_event.is_set():
            with self._lock:
                dev = self.device
            if dev is None:
                break
            try:
                ch0 = dev.analog_in_capture(
                    channel=0,
                    sample_rate_hz=self.sample_rate_hz,
                    buffer_size=buf,
                    trigger_level_v=self.trigger_level_v if self.trigger_channel == 0 else None,
                    trigger_channel=self.trigger_channel,
                    auto_timeout_s=self.auto_timeout_s,
                    timeout_s=self.acquisition_timeout_s,
                )
                ch1 = None
                if self.use_ch1:
                    ch1 = dev.analog_in_capture(
                        channel=1,
                        sample_rate_hz=self.sample_rate_hz,
                        buffer_size=buf,
                        trigger_level_v=self.trigger_level_v if self.trigger_channel == 1 else None,
                        trigger_channel=self.trigger_channel,
                        auto_timeout_s=self.auto_timeout_s,
                        timeout_s=self.acquisition_timeout_s,
                    )
                if not self.queue.full():
                    self.queue.put_nowait({"ch0": ch0, "ch1": ch1})
            except TimeoutError:
                pass
            except Exception as e:
                if self._stop_event.is_set():
                    break
                self._set_status(f"Capture error: {e}")
                time.sleep(0.5)

    def _set_status(self, msg):
        if self.status_var is not None:
            try:
                self.status_var.set(msg)
            except Exception:
                pass


# ===========================================================================
# Shared signal-processing helpers
# ===========================================================================

def find_first_trigger_index(row, level):
    above  = row >= level
    rising = np.where(~above[:-1] & above[1:])[0] + 1
    return int(rising[0]) if len(rising) > 0 else None


def count_pulses_in_window(window, trigger_level, holdoff_samples):
    above  = window >= trigger_level
    rising = np.where(~above[:-1] & above[1:])[0] + 1
    if len(rising) == 0:
        return 0
    count = 1
    last  = rising[0]
    for e in rising[1:]:
        if e - last >= holdoff_samples:
            count += 1
            last   = e
    return count


def measure_pulse(window, fs, half_width_only=False):
    if len(window) == 0:
        return np.nan, np.nan
    peak_idx = int(np.argmax(window))
    height   = float(window[peak_idx])
    half_max = height / 2.0

    right_idx = np.nan
    for i in range(peak_idx, len(window) - 1):
        if window[i] >= half_max >= window[i + 1]:
            frac      = (window[i] - half_max) / (window[i] - window[i + 1])
            right_idx = i + frac
            break

    if half_width_only:
        fwhm_us = (np.nan if np.isnan(right_idx)
                   else 2.0 * (right_idx - peak_idx) / fs * 1e6)
    else:
        left_idx = np.nan
        for i in range(peak_idx, 0, -1):
            if window[i - 1] <= half_max <= window[i]:
                frac     = (half_max - window[i - 1]) / (window[i] - window[i - 1])
                left_idx = (i - 1) + frac
                break
        fwhm_us = (np.nan if (np.isnan(left_idx) or np.isnan(right_idx))
                   else (right_idx - left_idx) / fs * 1e6)
    return height, fwhm_us


def passes_trigger_logic(
    raw, fs, first_trig, second_trig, expected_pulses,
    start_idx, stop_idx, holdoff_samples, do_filter, b=None, a_coef=None
):
    """
    Returns True if `raw` satisfies the expected-pulse criterion.

    expected_pulses == 1  → must cross first_trig at least once (anywhere)
    expected_pulses == 2  → must cross first_trig AND cross second_trig
                            at least once inside [start_idx, stop_idx)
    """
    # First trigger: anywhere in the full trace
    cross = find_first_trigger_index(raw, first_trig)
    if cross is None:
        return False, None

    if expected_pulses == 1:
        return True, cross

    # Second trigger: within the analysis window, after filtering
    chopped = raw[cross:]
    if stop_idx > len(chopped):
        return False, cross

    win = chopped[start_idx:stop_idx]
    filt = (sig.filtfilt(b, a_coef, win)
            if (do_filter and len(win) > 9)
            else win.copy())
    if count_pulses_in_window(filt, second_trig, holdoff_samples) >= 1:
        return True, cross
    return False, cross


# ===========================================================================
# Shared scrollable sidebar builder
# ===========================================================================

def make_scrollable_sidebar(parent, width=275):
    container  = tk.Frame(parent, bg=C["sidebar"])
    canvas     = tk.Canvas(container, width=width, highlightthickness=0,
                           bg=C["sidebar"])
    scrollbar  = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill="y")
    canvas.pack(side=tk.LEFT, fill="y", expand=True)

    frame        = tk.Frame(canvas, bg=C["bg"])
    frame_window = canvas.create_window((0, 0), window=frame, anchor="nw")

    def _on_frame(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _on_canvas(event):
        canvas.itemconfig(frame_window, width=event.width)

    frame.bind("<Configure>", _on_frame)
    canvas.bind("<Configure>", _on_canvas)
    canvas.bind_all("<MouseWheel>",
                    lambda e: canvas.yview_scroll(int(-1 * e.delta / 120), "units"))
    canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
    canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll( 1, "units"))

    return container, frame, canvas


def add_row(frame, label, var):
    _field_label(frame, label)
    _entry(frame, var)


# ===========================================================================
# ADS + shared-settings panel
# ===========================================================================

class AdsPanel:
    """
    Single merged settings panel: Ch0, Ch1, shared acquisition, ADS hardware,
    device control.  The panel holds references to the per-channel first-trigger
    vars so it can push the correct level to AcquisitionManager on Start.
    """

    def __init__(self, frame, acq: AcquisitionManager,
                 use_ch1_var: tk.BooleanVar,
                 first_trigger_ch0_var: tk.DoubleVar,
                 first_trigger_ch1_var: tk.DoubleVar,
                 ch0_range_var: tk.DoubleVar,
                 ch1_range_var: tk.DoubleVar,
                 ch0_attenuation_var: tk.DoubleVar,
                 ch1_attenuation_var: tk.DoubleVar):

        self.acq                   = acq
        self.use_ch1_var           = use_ch1_var
        self.first_trigger_ch0_var = first_trigger_ch0_var
        self.first_trigger_ch1_var = first_trigger_ch1_var
        self.ch0_range_var         = ch0_range_var
        self.ch1_range_var         = ch1_range_var
        self.ch0_attenuation_var   = ch0_attenuation_var
        self.ch1_attenuation_var   = ch1_attenuation_var

        self.sample_rate_var   = tk.DoubleVar(value=acq.sample_rate_hz)
        self.trigger_ch_var    = tk.IntVar(value=acq.trigger_channel)
        self.auto_timeout_var  = tk.DoubleVar(value=acq.auto_timeout_s)
        self.acq_timeout_var   = tk.DoubleVar(value=acq.acquisition_timeout_s)

        self.status_var = tk.StringVar(value="Not connected")
        acq.status_var  = self.status_var

        self._build(frame)

    def _build(self, frame):
        _section_label(frame, "── ADS Hardware ──")
        add_row(frame, "Sample rate (Hz)",           self.sample_rate_var)
        add_row(frame, "HW trigger channel (0 / 1)", self.trigger_ch_var)
        add_row(frame, "Auto-timeout (s, 0 = strict)",self.auto_timeout_var)
        add_row(frame, "Acquisition timeout (s)",    self.acq_timeout_var)

        bf = tk.Frame(frame, bg=C["bg"])
        bf.pack(fill="x", pady=(8, 2))
        _button(bf, "Connect",    self._connect,    width=10).pack(side=tk.LEFT, padx=2)
        _button(bf, "Disconnect", self._disconnect, width=10).pack(side=tk.LEFT, padx=2)

        bf2 = tk.Frame(frame, bg=C["bg"])
        bf2.pack(fill="x", pady=2)
        _button(bf2, "▶ Start", self._start, width=10).pack(side=tk.LEFT, padx=2)
        _button(bf2, "■ Stop",  self._stop,  width=10).pack(side=tk.LEFT, padx=2)

        _status_label(frame, self.status_var).pack(anchor="w", pady=(4, 0))

    def _push_to_acq(self):
        a = self.acq
        a.sample_rate_hz        = self.sample_rate_var.get()
        a.trigger_channel       = self.trigger_ch_var.get()
        a.auto_timeout_s        = self.auto_timeout_var.get()
        a.acquisition_timeout_s = self.acq_timeout_var.get()
        a.ch0_range_v           = self.ch0_range_var.get()
        a.ch1_range_v           = self.ch1_range_var.get()
        a.ch0_attenuation       = self.ch0_attenuation_var.get()
        a.ch1_attenuation       = self.ch1_attenuation_var.get()
        a.use_ch1               = self.use_ch1_var.get()
        # HW trigger level = first-trigger of the chosen trigger channel
        tch = self.trigger_ch_var.get()
        if tch == 1:
            a.trigger_level_v = self.first_trigger_ch1_var.get()
        else:
            a.trigger_level_v = self.first_trigger_ch0_var.get()

    def _connect(self):
        self.status_var.set(self.acq.connect())

    def _disconnect(self):
        self.acq.disconnect()
        self.status_var.set("Disconnected")

    def _start(self):
        self._push_to_acq()
        self.status_var.set(self.acq.start_acquisition())

    def _stop(self):
        self.acq.stop_acquisition()
        self.status_var.set("Acquisition stopped")


# ===========================================================================
# Tab 1 – Signal Viewer
# ===========================================================================

class SignalViewerTab:

    def __init__(self, parent: tk.Frame, acq: AcquisitionManager):
        self.acq     = acq
        self.running = True

        self.displayed_ch0: list = []   # full raw traces that passed
        self.displayed_ch1: list = []
        self.total_counts  = 0

        # ---- Tk vars ----
        self.use_ch1_var           = tk.BooleanVar(value=False)

        self.first_trigger_ch0_var = tk.DoubleVar(value=0.2)
        self.trigger_ch0_var       = tk.DoubleVar(value=0.01)
        self.pulses_ch0_var        = tk.IntVar(value=1)
        self.ch0_range_var         = tk.DoubleVar(value=5.0)
        self.ch0_attenuation_var   = tk.DoubleVar(value=1.0)

        self.first_trigger_ch1_var = tk.DoubleVar(value=0.2)
        self.trigger_ch1_var       = tk.DoubleVar(value=0.01)
        self.pulses_ch1_var        = tk.IntVar(value=1)
        self.ch1_range_var         = tk.DoubleVar(value=5.0)
        self.ch1_attenuation_var   = tk.DoubleVar(value=1.0)

        self.fs_var          = tk.DoubleVar(value=100e6)
        self.start_us_var    = tk.DoubleVar(value=0.5)
        self.stop_us_var     = tk.DoubleVar(value=40.0)
        self.filter_var      = tk.DoubleVar(value=100e6)
        self.holdoff_us_var  = tk.DoubleVar(value=0.5)
        self.max_display_var = tk.IntVar(value=50)

        self.save_traces_var = tk.BooleanVar(value=False)
        self.counts_var      = tk.IntVar(value=0)
        self.passing_var     = tk.IntVar(value=0)

        self.trace_csv_path = None

        self._build(parent)
        parent.after(500, self._update_loop)

    # ------------------------------------------------------------------ #

    def _build(self, parent):
        sidebar_container, frame, _ = make_scrollable_sidebar(parent)
        sidebar_container.pack(side=tk.LEFT, fill="y")

        # Ch0
        _section_label(frame, "── Channel 0 ──")
        add_row(frame, "First trigger level (V)",  self.first_trigger_ch0_var)
        add_row(frame, "Second trigger level (V)", self.trigger_ch0_var)
        add_row(frame, "Expected pulses (1 or 2)", self.pulses_ch0_var)
        add_row(frame, "Voltage range (V p-p)",    self.ch0_range_var)
        add_row(frame, "Attenuation",              self.ch0_attenuation_var)

        # Ch1
        _section_label(frame, "── Channel 1 ──")
        _checkbutton(frame, "Use Channel 1", self.use_ch1_var,
                     command=self._toggle_ch1).pack(anchor="w", pady=(2, 4))
        self.ch1_widgets = []

        def add_ch1_row(label, var):
            lbl = tk.Label(frame, text=label, bg=C["bg"], fg=C["fg_dim"],
                           font=("Helvetica", 8), anchor="w")
            lbl.pack(anchor="w")
            ent = _entry(frame, var)
            self.ch1_widgets += [lbl, ent]

        add_ch1_row("First trigger level (V)",  self.first_trigger_ch1_var)
        add_ch1_row("Second trigger level (V)", self.trigger_ch1_var)
        add_ch1_row("Expected pulses (1 or 2)", self.pulses_ch1_var)
        add_ch1_row("Voltage range (V p-p)",    self.ch1_range_var)
        add_ch1_row("Attenuation",              self.ch1_attenuation_var)

        # Acquisition / shared
        _section_label(frame, "── Acquisition ──")
        add_row(frame, "Sample rate (Hz)",            self.fs_var)
        add_row(frame, "Low-pass cutoff (Hz)",        self.filter_var)
        add_row(frame, "Window start after trig (µs)",self.start_us_var)
        add_row(frame, "Window stop after trig (µs)", self.stop_us_var)
        add_row(frame, "Holdoff (µs)",                self.holdoff_us_var)
        add_row(frame, "Max traces to display",       self.max_display_var)

        _button(frame, "Reset", self._reset).pack(fill="x", pady=(8, 4))

        # Data saving
        _section_label(frame, "── Data Saving ──")
        _checkbutton(frame, "Save raw traces to CSV",
                     self.save_traces_var,
                     command=self._toggle_save_traces).pack(anchor="w")
        self.trace_path_label = tk.Label(frame, text="(traces not saved)",
                                         bg=C["bg"], fg=C["fg_dim"],
                                         wraplength=230, justify="left",
                                         font=("Helvetica", 8))
        self.trace_path_label.pack(anchor="w")

        # ADS hardware panel
        self.ads_panel = AdsPanel(
            frame, self.acq, self.use_ch1_var,
            self.first_trigger_ch0_var, self.first_trigger_ch1_var,
            self.ch0_range_var, self.ch1_range_var,
            self.ch0_attenuation_var, self.ch1_attenuation_var,
        )

        # Counters
        _section_label(frame, "── Status ──")
        _counter_label(frame, "Events read:")
        _counter_value(frame, self.counts_var)
        _counter_label(frame, "Passing events:")
        _counter_value(frame, self.passing_var)

        self._toggle_ch1()
        self._build_plot(parent)

    def _build_plot(self, parent):
        plot_frame = tk.Frame(parent, bg=C["plot_bg"])
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.fig    = plt.Figure(figsize=(10, 6))
        _style_figure(self.fig)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._rebuild_axes()

    def _rebuild_axes(self):
        self.fig.clf()
        use_ch1 = self.use_ch1_var.get()
        if use_ch1:
            self.ax0, self.ax1 = self.fig.subplots(2, 1, sharex=True)
            _style_ax(self.ax1, xlabel="Time (µs)", ylabel="Amplitude (V)")
        else:
            self.ax0 = self.fig.subplots(1, 1)
            self.ax1 = None
            _style_ax(self.ax0, xlabel="Time (µs)", ylabel="Amplitude (V)")
        _style_ax(self.ax0, ylabel="Amplitude (V)")
        self.fig.tight_layout(pad=1.5)
        self.canvas.draw()

    def _toggle_ch1(self):
        state = "normal" if self.use_ch1_var.get() else "disabled"
        for w in self.ch1_widgets:
            w.config(state=state)
        if hasattr(self, "fig"):
            self._rebuild_axes()

    def _toggle_save_traces(self):
        if self.save_traces_var.get():
            if not messagebox.askokcancel(
                "Memory warning",
                "Saving raw traces can consume many GB of disk space during "
                "long runs.\n\nProceed and choose a file?",
                icon="warning"
            ):
                self.save_traces_var.set(False)
                return
            path = filedialog.asksaveasfilename(
                title="Save traces CSV",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")]
            )
            if path:
                self.trace_csv_path = path
                self.trace_path_label.config(text=os.path.basename(path),
                                             fg=C["teal"])
            else:
                self.save_traces_var.set(False)
                self.trace_path_label.config(text="(traces not saved)",
                                             fg=C["fg_dim"])
        else:
            self.trace_csv_path = None
            self.trace_path_label.config(text="(traces not saved)", fg=C["fg_dim"])

    def _reset(self):
        self.displayed_ch0.clear()
        self.displayed_ch1.clear()
        self.total_counts = 0
        self.counts_var.set(0)
        self.passing_var.set(0)
        self._rebuild_axes()

    # ------------------------------------------------------------------ #

    def _update_loop(self):
        if not self.running:
            return
        self._drain_queue()
        try:
            self.fig.canvas.get_tk_widget().after(500, self._update_loop)
        except Exception:
            pass

    def _drain_queue(self):
        batch = []
        for _ in range(50):
            try:
                batch.append(self.acq.queue.get_nowait())
            except queue.Empty:
                break
        if batch:
            self._process_batch(batch)

    def _process_batch(self, batch):
        fs              = self.fs_var.get()
        fc              = self.filter_var.get()
        start_idx       = int(self.start_us_var.get() * 1e-6 * fs)
        stop_idx        = int(self.stop_us_var.get()  * 1e-6 * fs)
        holdoff_samples = int(self.holdoff_us_var.get() * 1e-6 * fs)
        first_trig_ch0  = self.first_trigger_ch0_var.get()
        trig_ch0        = self.trigger_ch0_var.get()
        pulses_ch0      = self.pulses_ch0_var.get()
        use_ch1         = self.use_ch1_var.get()
        if use_ch1:
            first_trig_ch1 = self.first_trigger_ch1_var.get()
            trig_ch1       = self.trigger_ch1_var.get()
            pulses_ch1     = self.pulses_ch1_var.get()

        nyq       = fs / 2.0
        do_filter = 0 < fc < nyq
        b = a_coef = None
        if do_filter:
            b, a_coef = sig.butter(4, fc / nyq, btype="low")

        new_ch0 = []
        new_ch1 = []
        trace_rows = []

        for item in batch:
            raw_ch0 = item["ch0"]
            self.total_counts += 1

            ok_ch0, _ = passes_trigger_logic(
                raw_ch0, fs, first_trig_ch0, trig_ch0, pulses_ch0,
                start_idx, stop_idx, holdoff_samples, do_filter, b, a_coef
            )
            if not ok_ch0:
                continue

            if use_ch1 and item["ch1"] is not None:
                raw_ch1 = item["ch1"]
                ok_ch1, _ = passes_trigger_logic(
                    raw_ch1, fs, first_trig_ch1, trig_ch1, pulses_ch1,
                    start_idx, stop_idx, holdoff_samples, do_filter, b, a_coef
                )
                if not ok_ch1:
                    continue
                new_ch1.append(raw_ch1)
                if self.save_traces_var.get():
                    trace_rows.append(raw_ch1.tolist())

            new_ch0.append(raw_ch0)
            if self.save_traces_var.get():
                trace_rows.append(raw_ch0.tolist())

        self.counts_var.set(self.total_counts)
        if not new_ch0:
            return

        self.displayed_ch0.extend(new_ch0)
        if use_ch1:
            self.displayed_ch1.extend(new_ch1)
        self.passing_var.set(len(self.displayed_ch0))

        if self.save_traces_var.get() and trace_rows and self.trace_csv_path:
            self._append_trace_csv(trace_rows)

        self._update_plot(fs)

    def _append_trace_csv(self, rows):
        try:
            write_header = not os.path.exists(self.trace_csv_path)
            pd.DataFrame(rows).to_csv(
                self.trace_csv_path, mode="a", header=write_header, index=False
            )
        except Exception as e:
            print(f"[SignalViewer] trace CSV write error: {e}")

    def _update_plot(self, fs):
        use_ch1     = self.use_ch1_var.get()
        max_display = self.max_display_var.get()
        n_pts       = len(self.displayed_ch0[-1])
        time_us     = np.arange(n_pts) / fs * 1e6

        first_trig_ch0 = self.first_trigger_ch0_var.get()
        trig_ch0       = self.trigger_ch0_var.get()

        self.ax0.clear()
        _style_ax(self.ax0,
                  title=f"Channel 0 — last {min(len(self.displayed_ch0), max_display)} traces",
                  ylabel="Amplitude (V)")
        for trace in self.displayed_ch0[-max_display:]:
            self.ax0.plot(time_us[:len(trace)], trace,
                          color=C["trace_ch0"], alpha=0.3, linewidth=0.7)
        self.ax0.axhline(first_trig_ch0, color=C["trig1"], linestyle="--",
                         linewidth=1.0, label=f"First trig ({first_trig_ch0:.3g} V)")
        self.ax0.axhline(trig_ch0, color=C["trig2"], linestyle="--",
                         linewidth=1.0, label=f"Second trig ({trig_ch0:.3g} V)")
        self.ax0.legend(fontsize=7, facecolor=C["panel"],
                        edgecolor=C["border"], labelcolor=C["fg"])

        if use_ch1 and self.ax1 is not None and self.displayed_ch1:
            first_trig_ch1 = self.first_trigger_ch1_var.get()
            trig_ch1       = self.trigger_ch1_var.get()
            self.ax1.clear()
            _style_ax(self.ax1,
                      title=f"Channel 1 — last {min(len(self.displayed_ch1), max_display)} traces",
                      xlabel="Time (µs)", ylabel="Amplitude (V)")
            for trace in self.displayed_ch1[-max_display:]:
                self.ax1.plot(time_us[:len(trace)], trace,
                              color=C["trace_ch1"], alpha=0.3, linewidth=0.7)
            self.ax1.axhline(first_trig_ch1, color=C["trig1"], linestyle="--",
                             linewidth=1.0, label=f"First trig ({first_trig_ch1:.3g} V)")
            self.ax1.axhline(trig_ch1, color=C["trig2"], linestyle="--",
                             linewidth=1.0, label=f"Second trig ({trig_ch1:.3g} V)")
            self.ax1.legend(fontsize=7, facecolor=C["panel"],
                            edgecolor=C["border"], labelcolor=C["fg"])
        elif not use_ch1:
            self.ax0.set_xlabel("Time (µs)")

        self.fig.tight_layout(pad=1.5)
        self.canvas.draw()


# ===========================================================================
# Tab 2 – Live Histogram
# ===========================================================================

class HistogramTab:

    def __init__(self, parent: tk.Frame, acq: AcquisitionManager):
        self.acq     = acq
        self.running = True

        self.records: list = []
        self.total_counts  = 0

        # ---- Tk vars ----
        self.use_ch1_var           = tk.BooleanVar(value=False)

        self.first_trigger_ch0_var = tk.DoubleVar(value=0.2)
        self.trigger_ch0_var       = tk.DoubleVar(value=0.01)
        self.pulses_ch0_var        = tk.IntVar(value=1)
        self.ch0_range_var         = tk.DoubleVar(value=5.0)
        self.ch0_attenuation_var   = tk.DoubleVar(value=1.0)

        self.first_trigger_ch1_var = tk.DoubleVar(value=0.2)
        self.trigger_ch1_var       = tk.DoubleVar(value=0.01)
        self.pulses_ch1_var        = tk.IntVar(value=1)
        self.ch1_range_var         = tk.DoubleVar(value=5.0)
        self.ch1_attenuation_var   = tk.DoubleVar(value=1.0)

        self.fs_var          = tk.DoubleVar(value=100e6)
        self.start_us_var    = tk.DoubleVar(value=0.5)
        self.stop_us_var     = tk.DoubleVar(value=40.0)
        self.bins_var        = tk.IntVar(value=100)
        self.filter_var      = tk.DoubleVar(value=100e6)
        self.holdoff_us_var  = tk.DoubleVar(value=0.5)

        self.save_traces_var = tk.BooleanVar(value=False)
        self.counts_var      = tk.IntVar(value=0)
        self.triggers_var    = tk.IntVar(value=0)

        self.time_log_path  = None
        self.trace_csv_path = None

        self._build(parent)
        parent.after(500, self._update_loop)

    # ------------------------------------------------------------------ #

    def _build(self, parent):
        sidebar_container, frame, _ = make_scrollable_sidebar(parent)
        sidebar_container.pack(side=tk.LEFT, fill="y")

        # Ch0
        _section_label(frame, "── Channel 0 ──")
        add_row(frame, "First trigger level (V)",  self.first_trigger_ch0_var)
        add_row(frame, "Second trigger level (V)", self.trigger_ch0_var)
        add_row(frame, "Expected pulses (1 or 2)", self.pulses_ch0_var)
        add_row(frame, "Voltage range (V p-p)",    self.ch0_range_var)
        add_row(frame, "Attenuation",              self.ch0_attenuation_var)

        # Ch1
        _section_label(frame, "── Channel 1 ──")
        _checkbutton(frame, "Use Channel 1", self.use_ch1_var,
                     command=self._toggle_ch1).pack(anchor="w", pady=(2, 4))
        self.ch1_widgets = []

        def add_ch1_row(label, var):
            lbl = tk.Label(frame, text=label, bg=C["bg"], fg=C["fg_dim"],
                           font=("Helvetica", 8), anchor="w")
            lbl.pack(anchor="w")
            ent = _entry(frame, var)
            self.ch1_widgets += [lbl, ent]

        add_ch1_row("First trigger level (V)",  self.first_trigger_ch1_var)
        add_ch1_row("Second trigger level (V)", self.trigger_ch1_var)
        add_ch1_row("Expected pulses (1 or 2)", self.pulses_ch1_var)
        add_ch1_row("Voltage range (V p-p)",    self.ch1_range_var)
        add_ch1_row("Attenuation",              self.ch1_attenuation_var)

        # Acquisition / shared
        _section_label(frame, "── Acquisition ──")
        add_row(frame, "Sample rate (Hz)",            self.fs_var)
        add_row(frame, "Low-pass cutoff (Hz)",        self.filter_var)
        add_row(frame, "Window start after trig (µs)",self.start_us_var)
        add_row(frame, "Window stop after trig (µs)", self.stop_us_var)
        add_row(frame, "Histogram bins",              self.bins_var)
        add_row(frame, "Holdoff (µs)",                self.holdoff_us_var)

        bf = tk.Frame(frame, bg=C["bg"])
        bf.pack(fill="x", pady=(8, 4))
        _button(bf, "Reset",      self._reset,      width=9).pack(side=tk.LEFT, padx=2)
        _button(bf, "Export CSV", self._export_csv, width=9).pack(side=tk.LEFT, padx=2)

        # Data saving
        _section_label(frame, "── Data Saving ──")
        _button(frame, "Set peak-data log path",
                self._set_log_path).pack(fill="x", pady=2)
        self.log_label = tk.Label(frame, text="(no log file set)",
                                  bg=C["bg"], fg=C["fg_dim"],
                                  wraplength=230, justify="left",
                                  font=("Helvetica", 8))
        self.log_label.pack(anchor="w")

        _checkbutton(frame, "Save raw traces to CSV",
                     self.save_traces_var,
                     command=self._toggle_save_traces).pack(anchor="w", pady=(6, 0))
        self.trace_path_label = tk.Label(frame, text="(traces not saved)",
                                         bg=C["bg"], fg=C["fg_dim"],
                                         wraplength=230, justify="left",
                                         font=("Helvetica", 8))
        self.trace_path_label.pack(anchor="w")

        # ADS hardware
        self.ads_panel = AdsPanel(
            frame, self.acq, self.use_ch1_var,
            self.first_trigger_ch0_var, self.first_trigger_ch1_var,
            self.ch0_range_var, self.ch1_range_var,
            self.ch0_attenuation_var, self.ch1_attenuation_var,
        )

        # Status counters
        _section_label(frame, "── Status ──")
        _counter_label(frame, "Events read:")
        _counter_value(frame, self.counts_var)
        _counter_label(frame, "Passing triggers:")
        _counter_value(frame, self.triggers_var)

        self._toggle_ch1()
        self._build_plot(parent)

    def _build_plot(self, parent):
        plot_frame = tk.Frame(parent, bg=C["plot_bg"])
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.fig = plt.Figure(figsize=(10, 7))
        _style_figure(self.fig)
        gs = gridspec.GridSpec(2, 4, figure=self.fig,
                               height_ratios=[1.7, 1],
                               hspace=0.6, wspace=0.5)
        self.ax_dt    = self.fig.add_subplot(gs[0, :])
        self.ax_a1    = self.fig.add_subplot(gs[1, 0])
        self.ax_fwhm1 = self.fig.add_subplot(gs[1, 1])
        self.ax_a2    = self.fig.add_subplot(gs[1, 2])
        self.ax_fwhm2 = self.fig.add_subplot(gs[1, 3])
        self._label_axes()
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _label_axes(self):
        _style_ax(self.ax_dt,
                  title="dt — time to second pulse",
                  xlabel="Time after first trigger (µs)",
                  ylabel="Count")
        for ax, title, xlabel in [
            (self.ax_a1,    "Pulse 1 height", "Amplitude (V)"),
            (self.ax_fwhm1, "Pulse 1 FWHM",   "FWHM (µs)"),
            (self.ax_a2,    "Pulse 2 height", "Amplitude (V)"),
            (self.ax_fwhm2, "Pulse 2 FWHM",   "FWHM (µs)"),
        ]:
            _style_ax(ax, title=title, xlabel=xlabel, ylabel="Count",
                      title_size=8, label_size=7, tick_size=6)

    def _toggle_ch1(self):
        state = "normal" if self.use_ch1_var.get() else "disabled"
        for w in self.ch1_widgets:
            w.config(state=state)

    def _set_log_path(self):
        path = filedialog.asksaveasfilename(
            title="Peak-data log CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if path:
            self.time_log_path = path
            self.log_label.config(text=os.path.basename(path), fg=C["teal"])

    def _toggle_save_traces(self):
        if self.save_traces_var.get():
            if not messagebox.askokcancel(
                "Memory warning",
                "Saving raw traces can consume many GB of disk space during "
                "long runs.\n\nProceed and choose a file?",
                icon="warning"
            ):
                self.save_traces_var.set(False)
                return
            path = filedialog.asksaveasfilename(
                title="Save traces CSV",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")]
            )
            if path:
                self.trace_csv_path = path
                self.trace_path_label.config(text=os.path.basename(path),
                                             fg=C["teal"])
            else:
                self.save_traces_var.set(False)
                self.trace_path_label.config(text="(traces not saved)",
                                             fg=C["fg_dim"])
        else:
            self.trace_csv_path = None
            self.trace_path_label.config(text="(traces not saved)", fg=C["fg_dim"])

    def _reset(self):
        self.records.clear()
        self.total_counts = 0
        self.counts_var.set(0)
        self.triggers_var.set(0)
        for ax in (self.ax_dt, self.ax_a1, self.ax_fwhm1, self.ax_a2, self.ax_fwhm2):
            ax.clear()
        self._label_axes()
        self.canvas.draw()

    def _export_csv(self):
        if not self.records:
            messagebox.showwarning("No Data", "No records to export.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV files", "*.csv")]
        )
        if path:
            pd.DataFrame(self.records,
                         columns=["timestamp","dt_us","a1","fwhm1_us","a2","fwhm2_us"]
                         ).to_csv(path, index=False)

    # ------------------------------------------------------------------ #

    def _update_loop(self):
        if not self.running:
            return
        self._drain_queue()
        try:
            self.fig.canvas.get_tk_widget().after(500, self._update_loop)
        except Exception:
            pass

    def _drain_queue(self):
        batch = []
        for _ in range(50):
            try:
                batch.append(self.acq.queue.get_nowait())
            except queue.Empty:
                break
        if batch:
            self._process_batch(batch)

    def _process_batch(self, batch):
        fs              = self.fs_var.get()
        fc              = self.filter_var.get()
        start_idx       = int(self.start_us_var.get() * 1e-6 * fs)
        stop_idx        = int(self.stop_us_var.get()  * 1e-6 * fs)
        holdoff_samples = int(self.holdoff_us_var.get() * 1e-6 * fs)
        first_trig_ch0  = self.first_trigger_ch0_var.get()
        trig_ch0        = self.trigger_ch0_var.get()
        pulses_ch0      = self.pulses_ch0_var.get()
        use_ch1         = self.use_ch1_var.get()
        if use_ch1:
            first_trig_ch1 = self.first_trigger_ch1_var.get()
            trig_ch1       = self.trigger_ch1_var.get()
            pulses_ch1     = self.pulses_ch1_var.get()

        nyq       = fs / 2.0
        do_filter = 0 < fc < nyq
        b = a_coef = None
        if do_filter:
            b, a_coef = sig.butter(4, fc / nyq, btype="low")

        new_records = []
        trace_rows  = []

        for item in batch:
            raw_ch0 = item["ch0"]
            self.total_counts += 1

            ok_ch0, cross_ch0 = passes_trigger_logic(
                raw_ch0, fs, first_trig_ch0, trig_ch0, pulses_ch0,
                start_idx, stop_idx, holdoff_samples, do_filter, b, a_coef
            )
            if not ok_ch0:
                continue

            if use_ch1 and item["ch1"] is not None:
                ok_ch1, _ = passes_trigger_logic(
                    item["ch1"], fs, first_trig_ch1, trig_ch1, pulses_ch1,
                    start_idx, stop_idx, holdoff_samples, do_filter, b, a_coef
                )
                if not ok_ch1:
                    continue

            # ---- Pulse measurements (on chopped signal from first trigger) ----
            chopped_ch0 = raw_ch0[cross_ch0:]

            # Pulse 1: region from first trigger up to start_idx
            win1      = chopped_ch0[:start_idx]
            filt1     = (sig.filtfilt(b, a_coef, win1)
                         if (do_filter and len(win1) > 9) else win1.copy())
            a1, fwhm1_us = measure_pulse(filt1, fs, half_width_only=True)

            # Pulse 2: analysis window [start_idx, stop_idx)
            if stop_idx <= len(chopped_ch0):
                win2  = chopped_ch0[start_idx:stop_idx]
                filt2 = (sig.filtfilt(b, a_coef, win2)
                         if do_filter else win2.copy())
            else:
                filt2 = np.array([])

            # dt: time of second pulse rising edge from first trigger
            dt_us = np.nan
            a2    = np.nan
            fwhm2_us = np.nan
            if len(filt2):
                above2  = filt2 >= trig_ch0
                rising2 = np.where(~above2[:-1] & above2[1:])[0] + 1
                if len(rising2):
                    dt_us = (rising2[0] + start_idx) / fs * 1e6
                a2, fwhm2_us = measure_pulse(filt2, fs)

            ts = datetime.datetime.now().isoformat(timespec="seconds")
            new_records.append({
                "timestamp": ts,
                "dt_us":     dt_us,
                "a1":        a1,
                "fwhm1_us":  fwhm1_us,
                "a2":        a2,
                "fwhm2_us":  fwhm2_us,
            })

            if self.save_traces_var.get():
                trace_rows.append(raw_ch0.tolist())
                if use_ch1 and item["ch1"] is not None:
                    trace_rows.append(item["ch1"].tolist())

        self.counts_var.set(self.total_counts)
        if not new_records:
            return

        self.records.extend(new_records)
        self.triggers_var.set(len(self.records))
        self._save_records(new_records)

        if self.save_traces_var.get() and trace_rows and self.trace_csv_path:
            self._append_trace_csv(trace_rows)

        self._update_histograms()

    def _save_records(self, new_records):
        if not self.time_log_path or not new_records:
            return
        df = pd.DataFrame(new_records,
                          columns=["timestamp","dt_us","a1","fwhm1_us","a2","fwhm2_us"])
        write_header = not os.path.exists(self.time_log_path)
        try:
            df.to_csv(self.time_log_path, mode="a", header=write_header, index=False)
        except Exception as e:
            print(f"[Histogram] log write error: {e}")

    def _append_trace_csv(self, rows):
        try:
            write_header = not os.path.exists(self.trace_csv_path)
            pd.DataFrame(rows).to_csv(
                self.trace_csv_path, mode="a", header=write_header, index=False
            )
        except Exception as e:
            print(f"[Histogram] trace CSV write error: {e}")

    def _update_histograms(self):
        if not self.records:
            return
        bins = self.bins_var.get()
        df   = pd.DataFrame(self.records)

        def _hist(ax, data, title, xlabel, color):
            ax.clear()
            _style_ax(ax, title=title, xlabel=xlabel, ylabel="Count",
                      title_size=8, label_size=7, tick_size=6)
            clean = data.dropna()
            if len(clean):
                ax.hist(clean, bins=bins, color=color,
                        edgecolor=C["plot_bg"], linewidth=0.5)

        _hist(self.ax_dt,    df["dt_us"],    "dt — time to second pulse",
              "Time after first trigger (µs)", C["hist_dt"])
        _hist(self.ax_a1,    df["a1"],       "Pulse 1 height",
              "Amplitude (V)", C["hist_p1"])
        _hist(self.ax_fwhm1, df["fwhm1_us"], "Pulse 1 FWHM",
              "FWHM (µs)",    C["hist_p1"])
        _hist(self.ax_a2,    df["a2"],       "Pulse 2 height",
              "Amplitude (V)", C["hist_p2"])
        _hist(self.ax_fwhm2, df["fwhm2_us"], "Pulse 2 FWHM",
              "FWHM (µs)",    C["hist_p2"])

        self.fig.subplots_adjust(left=0.07, right=0.97, top=0.93,
                                 bottom=0.1, hspace=0.65, wspace=0.5)
        self.canvas.draw()


# ===========================================================================
# Top-level application
# ===========================================================================

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Muon Detector DAQ")
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        _apply_global_theme(root)

        self.acq = AcquisitionManager()

        nb = ttk.Notebook(root)
        nb.pack(fill=tk.BOTH, expand=True)

        tab1 = tk.Frame(nb, bg=C["bg"])
        tab2 = tk.Frame(nb, bg=C["bg"])
        nb.add(tab1, text="  Signal Viewer  ")
        nb.add(tab2, text="  Live Histogram  ")

        self.viewer    = SignalViewerTab(tab1, self.acq)
        self.histogram = HistogramTab(tab2, self.acq)

    def _on_close(self):
        self.viewer.running    = False
        self.histogram.running = False
        self.acq.disconnect()
        self.root.quit()
        self.root.destroy()


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app  = App(root)
    root.mainloop()