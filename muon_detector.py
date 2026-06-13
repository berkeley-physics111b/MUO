"""
muon_detector.py
================
Two-tab Tkinter GUI for muon detector data acquisition.

Processing logic
----------------
* analog_in_capture_multiple is used when Ch1 is enabled (shared buffer zero).
* analog_in_capture is used for Ch0-only mode.
* Expected pulses = 1  → must cross first_trig anywhere in full trace.
* Expected pulses = 2  → must also cross second_trig inside [start_us, stop_us].
* "Events read"    = events passing the trigger channel first trigger.
* "Passing triggers" = events passing ALL configured trigger criteria.

Timing (from buffer start, i.e. absolute):
  t1_ch0  = first crossing of first_trig_ch0 / fs
  t2_ch0  = first crossing of second_trig_ch0 in window / fs  (NaN if pulses=1)
  dt_ch0  = t2_ch0 - t1_ch0
  t1_ch1  = first crossing of first_trig_ch1 / fs             (NaN if ch1 off)
  t2_ch1  = first crossing of second_trig_ch1 in window / fs  (NaN if ch1 off or pulses_ch1=1)
  dt_ch1  = t2_ch1 - t1_ch1
  dt_inter = t1_ch1 - t1_ch0  (inter-channel delay, NaN if ch1 off)

Signal viewer plots:
  t=0 at first trigger crossing; ~1 µs pretrigger shown; stop_us cutoff applied.
  Vertical dashed line at window start (start_us).
  Three traces shown per channel: raw, offset-removed, filtered.

Histogram layout (ch1 enabled, trigger ch=ch0, both ch have 2 pulses):
  Row 0 (full width): dt_trig_ch = t2 - t1 for the trigger channel
  Row 1: Ch0 P1 height | Ch0 P1 FWHM | Ch0 P2 height | Ch0 P2 FWHM
  Row 2: Ch1 P1 height | Ch1 P1 FWHM | Ch1 P2 height | Ch1 P2 FWHM
  Row 3 (full width): dt_inter = t1_ch1 - t1_ch0

Performance suggestions (see bottom of file).
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import scipy.signal as sig
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import queue
import datetime
import os
import time

try:
    from waveforms_ads import WaveFormsADS, DWFError
    ADS_AVAILABLE = True
except Exception as _e:
    ADS_AVAILABLE = False
    print(f"[muon_detector] waveforms_ads unavailable: {_e}")

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C = {
    "bg":         "#0d1b2a",
    "sidebar":    "#112240",
    "panel":      "#1a2e4a",
    "border":     "#1e3a5f",
    "fg":         "#e8dcc8",
    "fg_dim":     "#8a9bb0",
    "gold":       "#f0a500",
    "amber":      "#ffcf47",
    "teal":       "#4ecdc4",
    "red_warn":   "#e05c5c",
    "entry_bg":   "#0a1628",
    "entry_sel":  "#1e3a5f",
    "btn":        "#1e3a5f",
    "btn_active": "#2a4f7a",
    # matplotlib
    "trace_raw":  "#3a7ca5",
    "trace_offset":"#8ecae6",
    "trace_ch0":  "#4ecdc4",
    "trace_ch1":  "#f0a500",
    "trace_ch1_raw":    "#b5651d",
    "trace_ch1_offset": "#ffa94d",
    "trig1":      "#ffcf47",
    "trig2":      "#e05c5c",
    "win_start":  "#8a9bb0",
    "hist_dt":    "#4ecdc4",
    "hist_ch0p1": "#4ecdc4",
    "hist_ch0p2": "#2a9d8f",
    "hist_ch1p1": "#f0a500",
    "hist_ch1p2": "#e76f51",
    "hist_inter": "#c77dff",
    "hist_rate":  "#4ecdc4",
    "plot_bg":    "#0d1b2a",
    "plot_axes":  "#1a2e4a",
    "grid":       "#1e3a5f",
    "tick_fg":    "#8a9bb0",
}

PRETRIG_US = 1.0   # µs of pretrigger to display


# ---------------------------------------------------------------------------
# Theme helpers
# ---------------------------------------------------------------------------

def _apply_global_theme(root):
    root.configure(bg=C["bg"])
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure(".",
        background=C["bg"], foreground=C["fg"],
        fieldbackground=C["entry_bg"], bordercolor=C["border"],
        darkcolor=C["bg"], lightcolor=C["panel"],
        troughcolor=C["bg"], selectbackground=C["entry_sel"],
        selectforeground=C["amber"], font=("Helvetica", 9),
    )
    style.configure("TNotebook", background=C["bg"], bordercolor=C["border"],
                    tabmargins=[2, 4, 2, 0])
    style.configure("TNotebook.Tab", background=C["panel"], foreground=C["fg_dim"],
                    padding=[12, 4])
    style.map("TNotebook.Tab",
        background=[("selected", C["sidebar"]), ("active", C["btn_active"])],
        foreground=[("selected", C["amber"]),   ("active", C["fg"])],
    )
    style.configure("TFrame", background=C["bg"])
    style.configure("Vertical.TScrollbar", background=C["panel"],
                    troughcolor=C["bg"], arrowcolor=C["fg_dim"])


def _section_label(parent, text):
    tk.Label(parent, text=text, bg=C["panel"], fg=C["gold"],
             font=("Helvetica", 9, "bold"), padx=4, pady=2,
             anchor="w").pack(fill="x", pady=(8, 2))


def _field_label(parent, text):
    tk.Label(parent, text=text, bg=C["bg"], fg=C["fg_dim"],
             font=("Helvetica", 8), anchor="w").pack(anchor="w")


def _entry(parent, var):
    e = tk.Entry(parent, textvariable=var, bg=C["entry_bg"], fg=C["amber"],
                 insertbackground=C["amber"], selectbackground=C["entry_sel"],
                 selectforeground=C["amber"], relief="flat",
                 highlightthickness=1, highlightbackground=C["border"],
                 highlightcolor=C["gold"], font=("Helvetica", 9))
    e.pack(fill="x", pady=2)
    return e


def _button(parent, text, command, **kw):
    return tk.Button(parent, text=text, command=command, bg=C["btn"], fg=C["fg"],
                     activebackground=C["btn_active"], activeforeground=C["amber"],
                     relief="flat", highlightthickness=0, cursor="hand2",
                     font=("Helvetica", 9), **kw)


def _checkbutton(parent, text, variable, command=None):
    kw = dict(command=command) if command else {}
    return tk.Checkbutton(parent, text=text, variable=variable,
                          bg=C["bg"], fg=C["fg"], activebackground=C["bg"],
                          activeforeground=C["amber"], selectcolor=C["entry_bg"],
                          font=("Helvetica", 9), **kw)


def _status_label(parent, var):
    return tk.Label(parent, textvariable=var, bg=C["bg"], fg=C["teal"],
                    wraplength=240, justify="left", font=("Helvetica", 8))


def _counter_label(parent, text):
    tk.Label(parent, text=text, bg=C["bg"], fg=C["fg_dim"],
             font=("Helvetica", 8)).pack(anchor="w", pady=(6, 0))


def _counter_value(parent, var):
    tk.Label(parent, textvariable=var, bg=C["bg"], fg=C["amber"],
             font=("Helvetica", 9, "bold")).pack(anchor="w")


def add_row(frame, label, var):
    _field_label(frame, label)
    _entry(frame, var)


def _style_figure(fig):
    fig.patch.set_facecolor(C["plot_bg"])


def _style_ax(ax, title="", xlabel="", ylabel="",
              title_size=9, label_size=8, tick_size=7):
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


# ---------------------------------------------------------------------------
# Scrollable sidebar
# ---------------------------------------------------------------------------

def make_scrollable_sidebar(parent, width=280):
    container = tk.Frame(parent, bg=C["sidebar"])
    canvas    = tk.Canvas(container, width=width, highlightthickness=0,
                          bg=C["sidebar"])
    scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill="y")
    canvas.pack(side=tk.LEFT, fill="y", expand=True)

    frame        = tk.Frame(canvas, bg=C["bg"])
    frame_window = canvas.create_window((0, 0), window=frame, anchor="nw")

    def _on_frame(e):
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _on_canvas(e):
        canvas.itemconfig(frame_window, width=e.width)

    frame.bind("<Configure>", _on_frame)
    canvas.bind("<Configure>", _on_canvas)
    canvas.bind_all("<MouseWheel>",
                    lambda e: canvas.yview_scroll(int(-1 * e.delta / 120), "units"))
    canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
    canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

    return container, frame, canvas


def make_scrollable_plot_frame(parent):
    """Create a scrollable right-side panel for histogram graphs."""
    container = tk.Frame(parent, bg=C["plot_bg"])
    canvas    = tk.Canvas(container, highlightthickness=0, bg=C["plot_bg"])
    scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill="y")
    canvas.pack(side=tk.LEFT, fill="both", expand=True)

    inner = tk.Frame(canvas, bg=C["plot_bg"])
    win   = canvas.create_window((0, 0), window=inner, anchor="nw")

    def _on_inner(e):
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _on_canvas(e):
        canvas.itemconfig(win, width=e.width)

    inner.bind("<Configure>", _on_inner)
    canvas.bind("<Configure>", _on_canvas)

    # Bind scroll only when mouse is over this canvas
    def _on_enter(e):
        canvas.bind_all("<MouseWheel>",
                        lambda ev: canvas.yview_scroll(int(-1 * ev.delta / 120), "units"))
        canvas.bind_all("<Button-4>", lambda ev: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda ev: canvas.yview_scroll(1, "units"))

    canvas.bind("<Enter>", _on_enter)

    return container, inner, canvas


# ===========================================================================
# Acquisition manager
# ===========================================================================

class AcquisitionManager:
    """
    Owns the WaveFormsADS device and background capture thread.
    Puts {"ch0": ndarray, "ch1": ndarray|None, "timestamp": datetime} into self.queue.
    Uses analog_in_capture_multiple when use_ch1=True.
    """

    def __init__(self):
        self.device = None
        self.queue  = queue.Queue(maxsize=2000)
        self._thread     = None
        self._stop_event = threading.Event()
        self._lock       = threading.Lock()

        self.sample_rate_hz        = 100e6
        self.trigger_channel       = 0
        self.trigger_level_v       = 0.2
        self.auto_timeout_s        = 0.0
        self.acquisition_timeout_s = 5.0
        self.ch0_range_v           = 5.0
        self.ch1_range_v           = 5.0
        self.ch0_attenuation       = 1.0
        self.ch1_attenuation       = 1.0
        self.ch0_offset_v          = 0.0
        self.ch1_offset_v          = 0.0
        self.use_ch1               = False
        self.status_var            = None

    def connect(self):
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

    def start_acquisition(self):
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
    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def _capture_loop(self):
        buf = max(4096, int(self.sample_rate_hz * 200e-6))
        buf = min(buf, 32768)

        while not self._stop_event.is_set():
            with self._lock:
                dev = self.device
            if dev is None:
                break
            try:
                ts = datetime.datetime.now()   # timestamp at acquisition time
                if self.use_ch1:
                    channel_settings = {
                        0: {"attenuation": self.ch0_attenuation,
                            "y_offset":    self.ch0_offset_v,
                            "y_range":     self.ch0_range_v},
                        1: {"attenuation": self.ch1_attenuation,
                            "y_offset":    self.ch1_offset_v,
                            "y_range":     self.ch1_range_v},
                    }
                    result = dev.analog_in_capture_multiple(
                        channel_settings=channel_settings,
                        sample_rate_hz=self.sample_rate_hz,
                        buffer_size=buf,
                        trigger_channel=self.trigger_channel,
                        trigger_level_v=self.trigger_level_v,
                        auto_timeout_s=self.auto_timeout_s,
                        timeout_s=self.acquisition_timeout_s,
                    )
                    ch0 = result[0]
                    ch1 = result[1]
                else:
                    ch0 = dev.analog_in_capture(
                        channel=0,
                        sample_rate_hz=self.sample_rate_hz,
                        buffer_size=buf,
                        y_range=self.ch0_range_v,
                        attenuation=self.ch0_attenuation,
                        trigger_level_v=self.trigger_level_v,
                        trigger_channel=self.trigger_channel,
                        auto_timeout_s=self.auto_timeout_s,
                        timeout_s=self.acquisition_timeout_s,
                    )
                    ch1 = None

                if not self.queue.full():
                    self.queue.put_nowait({"ch0": ch0, "ch1": ch1, "timestamp": ts})

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
# Signal processing helpers
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
    count, last = 1, rising[0]
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


def subtract_pretrigger_offset(raw, cross_idx, fs):
    """
    Average the signal before (cross_idx - 1 µs) and subtract from entire trace.
    Returns offset-corrected array (same length as raw).
    """
    pretrig_end   = max(0, cross_idx - int(1e-6 * fs))
    if pretrig_end == 0:
        return raw.copy()
    baseline = float(np.mean(raw[:pretrig_end]))
    return raw - baseline


def analyse_channel(raw, fs, first_trig, second_trig, expected_pulses,
                    start_idx, stop_idx, holdoff_samples,
                    do_filter, sos=None):
    """
    Returns (passes_first, passes_second, cross_idx, t1_us, t2_us, dt_us,
             a1, fwhm1_us, a2, fwhm2_us, offset_corrected, filtered).

    Processing order:
      1. Find trigger crossing on raw trace.
      2. Subtract pretrigger baseline (offset correction).
      3. Apply filter (sosfiltfilt for numerical stability).
      4. Extract amplitude/width from filtered trace.

    cross_idx  : buffer-absolute index of first trigger crossing (or None)
    t1_us      : cross_idx / fs * 1e6
    t2_us      : buffer-absolute time of second pulse crossing (or NaN)
    dt_us      : t2_us - t1_us  (or NaN)
    a1/fwhm1   : pulse-1 metrics (measured on region before start_idx)
    a2/fwhm2   : pulse-2 metrics (measured in window [start_idx, stop_idx))
    offset_corrected : full-length baseline-subtracted trace
    filtered         : full-length filtered trace (or offset_corrected if no filter)
    """
    nan_result = (False, False, None,
                  np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan,
                  raw.copy(), raw.copy())

    # Step 1: find trigger crossing on raw
    cross = find_first_trigger_index(raw, first_trig)
    if cross is None:
        return nan_result

    t1_us = cross / fs * 1e6

    # Step 2: offset correction
    offset_corrected = subtract_pretrigger_offset(raw, cross, fs)

    # Step 3: apply filter to full offset-corrected trace (sosfiltfilt is
    # numerically more stable and faster than direct-form filtfilt)
    if do_filter and sos is not None and len(offset_corrected) > 9:
        filtered = sig.sosfiltfilt(sos, offset_corrected)
    else:
        filtered = offset_corrected.copy()

    # Step 4: extract pulse metrics from filtered trace
    chopped_filt = filtered[cross:]

    win1   = chopped_filt[:start_idx]
    a1, fwhm1_us = measure_pulse(win1, fs, half_width_only=True)

    if expected_pulses == 1:
        return (True, True, cross,
                t1_us, np.nan, np.nan,
                a1, fwhm1_us, np.nan, np.nan,
                offset_corrected, filtered)

    if stop_idx > len(chopped_filt):
        return (True, False, cross,
                t1_us, np.nan, np.nan,
                a1, fwhm1_us, np.nan, np.nan,
                offset_corrected, filtered)

    win2 = chopped_filt[start_idx:stop_idx]

    above2  = win2 >= second_trig
    rising2 = np.where(~above2[:-1] & above2[1:])[0] + 1
    if len(rising2) == 0:
        return (True, False, cross,
                t1_us, np.nan, np.nan,
                a1, fwhm1_us, np.nan, np.nan,
                offset_corrected, filtered)

    t2_us    = (cross + start_idx + rising2[0]) / fs * 1e6
    dt_us    = t2_us - t1_us
    a2, fwhm2_us = measure_pulse(win2, fs)

    return (True, True, cross,
            t1_us, t2_us, dt_us,
            a1, fwhm1_us, a2, fwhm2_us,
            offset_corrected, filtered)



# ===========================================================================
# Processing thread  (change 1 & 6: background processing, unbounded drain)
# ===========================================================================

class ProcessingThread:
    """
    Reads raw waveforms from acq.queue, performs all numpy/scipy work in a
    background thread, and pushes finished result dicts into results_queue.
    The GUI only reads results_queue for plotting / histogramming.

    Change 2: filter coefficients (sos) are pre-computed once and only
    recomputed when fs or fc change (dirty flag).
    Change 6: the queue is drained completely on every processing cycle —
    no 50-item cap.
    """

    def __init__(self, acq: "AcquisitionManager", process_fn):
        """
        acq         : AcquisitionManager whose .queue supplies raw items
        process_fn  : callable(batch, sos, do_filter) -> list of result dicts
        """
        self.acq         = acq
        self.process_fn  = process_fn
        self.results_queue = queue.Queue(maxsize=10000)

        self._thread     = None
        self._stop_event = threading.Event()

        # Filter cache (change 2)
        self._cached_fs  = None
        self._cached_fc  = None
        self._sos        = None
        self._do_filter  = False

        # Callbacks so the processing thread can read current settings
        # without touching Tkinter vars (set by owner before starting)
        self.get_fs = lambda: 100e6
        self.get_fc = lambda: 100e6

    def _refresh_filter(self):
        """Recompute sos only when fs or fc has changed (change 2)."""
        fs = self.get_fs()
        fc = self.get_fc()
        if fs == self._cached_fs and fc == self._cached_fc:
            return
        self._cached_fs = fs
        self._cached_fc = fc
        nyq = fs / 2.0
        self._do_filter = 0 < fc < nyq
        if self._do_filter:
            self._sos = sig.butter(4, fc / nyq, btype="low", output="sos")
        else:
            self._sos = None

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def _loop(self):
        while not self._stop_event.is_set():
            # Change 6: drain the entire queue in one shot — no item cap
            batch = []
            while True:
                try:
                    batch.append(self.acq.queue.get_nowait())
                except queue.Empty:
                    break

            if batch:
                self._refresh_filter()
                results = self.process_fn(batch, self._sos, self._do_filter,
                                          self._cached_fs)
                for r in results:
                    try:
                        self.results_queue.put_nowait(r)
                    except queue.Full:
                        pass  # drop oldest-ish; GUI can't keep up
            else:
                time.sleep(0.01)  # idle wait — avoid busy spin



class AdsPanel:
    def __init__(self, frame, acq: AcquisitionManager,
                 use_ch1_var,
                 first_trigger_ch0_var, first_trigger_ch1_var,
                 ch0_range_var, ch1_range_var,
                 ch0_attenuation_var, ch1_attenuation_var,
                 clear_graphs_cmd=None):

        self.acq                   = acq
        self.use_ch1_var           = use_ch1_var
        self.first_trigger_ch0_var = first_trigger_ch0_var
        self.first_trigger_ch1_var = first_trigger_ch1_var
        self.ch0_range_var         = ch0_range_var
        self.ch1_range_var         = ch1_range_var
        self.ch0_attenuation_var   = ch0_attenuation_var
        self.ch1_attenuation_var   = ch1_attenuation_var
        self.clear_graphs_cmd      = clear_graphs_cmd

        self.sample_rate_var  = tk.DoubleVar(value=acq.sample_rate_hz)
        self.trigger_ch_var   = tk.IntVar(value=acq.trigger_channel)

        self.status_var = tk.StringVar(value="Not connected")
        acq.status_var  = self.status_var

        self._build(frame)

    def _build(self, frame):
        _section_label(frame, "── Acquisition ──")
        add_row(frame, "Sample rate (Hz)",             self.sample_rate_var)
        add_row(frame, "HW trigger channel (0 / 1)",   self.trigger_ch_var)

        bf1 = tk.Frame(frame, bg=C["bg"])
        bf1.pack(fill="x", pady=(10, 2))
        _button(bf1, "Connect",    self._connect,    width=10).pack(side=tk.LEFT, padx=2)
        _button(bf1, "Disconnect", self._disconnect, width=10).pack(side=tk.LEFT, padx=2)

        bf2 = tk.Frame(frame, bg=C["bg"])
        bf2.pack(fill="x", pady=2)
        _button(bf2, "▶ Start",      self._start,           width=8).pack(side=tk.LEFT, padx=2)
        _button(bf2, "■ Stop",       self._stop,            width=8).pack(side=tk.LEFT, padx=2)
        _button(bf2, "✕ Clear",
                self.clear_graphs_cmd if self.clear_graphs_cmd else lambda: None,
                width=8).pack(side=tk.LEFT, padx=2)

        _status_label(frame, self.status_var).pack(anchor="w", pady=(4, 0))

    def _push_to_acq(self):
        a = self.acq
        a.sample_rate_hz        = self.sample_rate_var.get()
        a.trigger_channel       = self.trigger_ch_var.get()
        a.ch0_range_v           = self.ch0_range_var.get()
        a.ch1_range_v           = self.ch1_range_var.get()
        a.ch0_attenuation       = self.ch0_attenuation_var.get()
        a.ch1_attenuation       = self.ch1_attenuation_var.get()
        a.use_ch1               = self.use_ch1_var.get()
        tch = self.trigger_ch_var.get()
        a.trigger_level_v = (self.first_trigger_ch1_var.get() if tch == 1
                             else self.first_trigger_ch0_var.get())

    def _connect(self):    self.status_var.set(self.acq.connect())
    def _disconnect(self): self.acq.disconnect(); self.status_var.set("Disconnected")
    def _start(self):      self._push_to_acq(); self.status_var.set(self.acq.start_acquisition())
    def _stop(self):       self.acq.stop_acquisition(); self.status_var.set("Acquisition stopped")


# ===========================================================================
# Tab 1 – Signal Viewer
# ===========================================================================

class SignalViewerTab:

    def __init__(self, parent, acq: AcquisitionManager):
        self.acq     = acq
        self.running = True

        # Change 8: only keep the last N traces (N = max_display_var).
        # We use a collections.deque with maxlen; it is updated whenever
        # max_display_var changes.
        import collections
        self._collections = collections

        # stored as (raw_trim, offset_trim, filt_trim, cross_in_trim) tuples
        # Change 8: deques with bounded length — no unbounded list growth
        self.stored_ch0 = collections.deque()
        self.stored_ch1 = collections.deque()
        self.first_trigger_count = 0
        self.passing_count       = 0

        # ---- Tk vars ----
        self.use_ch1_var           = tk.BooleanVar(value=False)

        self.first_trigger_ch0_var = tk.DoubleVar(value=0.2)
        self.trigger_ch0_var       = tk.DoubleVar(value=0.01)
        self.pulses_ch0_var        = tk.IntVar(value=2)
        self.ch0_range_var         = tk.DoubleVar(value=1.0)
        self.ch0_attenuation_var   = tk.DoubleVar(value=-1.0)

        self.first_trigger_ch1_var = tk.DoubleVar(value=0.2)
        self.trigger_ch1_var       = tk.DoubleVar(value=0.01)
        self.pulses_ch1_var        = tk.IntVar(value=1)
        self.ch1_range_var         = tk.DoubleVar(value=1.0)
        self.ch1_attenuation_var   = tk.DoubleVar(value=-1.0)

        self.fs_var          = tk.DoubleVar(value=100e6)
        self.start_us_var    = tk.DoubleVar(value=0.5)
        self.stop_us_var     = tk.DoubleVar(value=40.0)
        self.filter_var      = tk.DoubleVar(value=100e6)
        self.holdoff_us_var  = tk.DoubleVar(value=0.5)
        self.max_display_var = tk.IntVar(value=50)

        # Change 8: trace of raw waveforms to CSV removed; no save_traces_var
        self.first_trigger_count_var = tk.IntVar(value=0)
        self.passing_var             = tk.IntVar(value=0)

        # Change 5: decoupled redraw timer (1 s)
        self._plot_dirty   = False
        self._last_start_idx = 0

        self._build(parent)

        # Change 1: start ProcessingThread
        self._proc_thread = ProcessingThread(self.acq, self._processing_fn)
        self._proc_thread.get_fs = lambda: self.ads_panel.sample_rate_var.get()
        self._proc_thread.get_fc = lambda: self.filter_var.get()
        self._proc_thread.start()

        # Change 5: fast results-drain (100 ms), slow redraw (1000 ms)
        parent.after(100,  self._drain_results)
        parent.after(1000, self._redraw_timer)

    def _build(self, parent):
        sidebar_container, frame, _ = make_scrollable_sidebar(parent)
        sidebar_container.pack(side=tk.LEFT, fill="y")

        _section_label(frame, "── Channel 0 ──")
        add_row(frame, "First trigger level (V)",  self.first_trigger_ch0_var)
        add_row(frame, "Second trigger level (V)", self.trigger_ch0_var)
        add_row(frame, "Expected pulses (1 or 2)", self.pulses_ch0_var)
        add_row(frame, "Voltage range (V p-p)",    self.ch0_range_var)
        add_row(frame, "Attenuation",              self.ch0_attenuation_var)

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

        _section_label(frame, "── Signal Processing ──")
        add_row(frame, "Low-pass cutoff (Hz)",         self.filter_var)
        add_row(frame, "Window start after trig (µs)", self.start_us_var)
        add_row(frame, "Window stop after trig (µs)",  self.stop_us_var)
        add_row(frame, "Holdoff (µs)",                 self.holdoff_us_var)
        add_row(frame, "Max traces to display",        self.max_display_var)
        # Change 8: resize deques when the user changes max_display
        self.max_display_var.trace_add("write", self._on_max_display_change)
        # Change 8: raw-trace CSV saving removed from Signal Viewer

        self.ads_panel = AdsPanel(
            frame, self.acq, self.use_ch1_var,
            self.first_trigger_ch0_var, self.first_trigger_ch1_var,
            self.ch0_range_var, self.ch1_range_var,
            self.ch0_attenuation_var, self.ch1_attenuation_var,
            clear_graphs_cmd=self._clear_graphs,
        )
        self.fs_var = self.ads_panel.sample_rate_var

        _section_label(frame, "── Status ──")
        _counter_label(frame, "Events read (first trig):")
        _counter_value(frame, self.first_trigger_count_var)
        _counter_label(frame, "Passing triggers:")
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
            self.ax0, self.ax1 = self.fig.subplots(2, 1, sharex=False)
            _style_ax(self.ax1, ylabel="Amplitude (V)",
                      xlabel="Time relative to first trigger (µs)")
        else:
            self.ax0 = self.fig.subplots(1, 1)
            self.ax1 = None
            _style_ax(self.ax0, ylabel="Amplitude (V)",
                      xlabel="Time relative to first trigger (µs)")
        _style_ax(self.ax0, ylabel="Amplitude (V)")
        self.fig.tight_layout(pad=1.5)
        self.canvas.draw()

    def _toggle_ch1(self):
        state = "normal" if self.use_ch1_var.get() else "disabled"
        for w in self.ch1_widgets:
            w.config(state=state)
        if hasattr(self, "fig"):
            self._rebuild_axes()

    def _on_max_display_change(self, *_):
        """Change 8: resize deques so they only hold max_display traces."""
        n = max(1, self.max_display_var.get())
        self.stored_ch0 = self._collections.deque(self.stored_ch0, maxlen=n)
        self.stored_ch1 = self._collections.deque(self.stored_ch1, maxlen=n)

    def _clear_graphs(self):
        n = max(1, self.max_display_var.get())
        self.stored_ch0 = self._collections.deque(maxlen=n)
        self.stored_ch1 = self._collections.deque(maxlen=n)
        self.first_trigger_count = 0
        self.passing_count       = 0
        self.first_trigger_count_var.set(0)
        self.passing_var.set(0)
        self._rebuild_axes()

    # ------------------------------------------------------------------ #
    # Change 1: processing_fn runs inside ProcessingThread (background)   #
    # ------------------------------------------------------------------ #

    def _processing_fn(self, batch, sos, do_filter, fs):
        """
        Called from ProcessingThread — no Tkinter calls allowed here.
        Returns a list of result dicts to be consumed by the GUI thread.
        Change 4: uses sosfiltfilt via sos argument.
        """
        start_idx       = int(self.start_us_var.get()  * 1e-6 * fs)
        stop_idx        = int(self.stop_us_var.get()   * 1e-6 * fs)
        holdoff_samples = int(self.holdoff_us_var.get() * 1e-6 * fs)
        pretrig_samples = int(PRETRIG_US * 1e-6 * fs)

        first_trig_ch0 = self.first_trigger_ch0_var.get()
        trig_ch0       = self.trigger_ch0_var.get()
        pulses_ch0     = self.pulses_ch0_var.get()
        use_ch1        = self.use_ch1_var.get()
        first_trig_ch1 = self.first_trigger_ch1_var.get()
        trig_ch1       = self.trigger_ch1_var.get()
        pulses_ch1     = self.pulses_ch1_var.get()

        results = []

        for item in batch:
            raw_ch0 = item["ch0"]

            (pass1_ch0, pass2_ch0, cross_ch0,
             t1_ch0, _, _,
             _, _, _, _,
             offset_ch0, filtered_ch0) = analyse_channel(
                raw_ch0, fs, first_trig_ch0, trig_ch0, pulses_ch0,
                start_idx, stop_idx, holdoff_samples, do_filter, sos)

            if not pass1_ch0:
                continue
            first_trig_hits = 1   # counted per-item; aggregated in GUI

            if not pass2_ch0:
                results.append({"type": "first_trig_only",
                                 "count": first_trig_hits})
                continue

            ch1_tuple = None
            if use_ch1 and item["ch1"] is not None:
                (pass1_ch1, pass2_ch1, cross_ch1,
                 _, _, _,
                 _, _, _, _,
                 offset_ch1, filtered_ch1) = analyse_channel(
                    item["ch1"], fs, first_trig_ch1, trig_ch1, pulses_ch1,
                    start_idx, stop_idx, holdoff_samples, do_filter, sos)
                if not (pass1_ch1 and pass2_ch1):
                    results.append({"type": "first_trig_only",
                                     "count": first_trig_hits})
                    continue

                trim_start_ch1 = max(0, cross_ch1 - pretrig_samples)
                trim_end_ch1   = cross_ch1 + stop_idx
                ch1_tuple = (
                    item["ch1"][trim_start_ch1:trim_end_ch1],
                    offset_ch1[trim_start_ch1:trim_end_ch1],
                    filtered_ch1[trim_start_ch1:trim_end_ch1],
                    cross_ch1 - trim_start_ch1,
                )

            trim_start = max(0, cross_ch0 - pretrig_samples)
            trim_end   = cross_ch0 + stop_idx
            ch0_tuple = (
                raw_ch0[trim_start:trim_end],
                offset_ch0[trim_start:trim_end],
                filtered_ch0[trim_start:trim_end],
                cross_ch0 - trim_start,
            )

            results.append({
                "type":       "passing",
                "ch0":        ch0_tuple,
                "ch1":        ch1_tuple,
                "start_idx":  start_idx,
                "first_trig_hits": first_trig_hits,
            })

        return results

    # Change 5: GUI drains results fast but only redraws on a fixed timer
    def _drain_results(self):
        """GUI thread: drain results_queue, accumulate, mark dirty."""
        if not self.running:
            return
        try:
            changed = False
            use_ch1 = self.use_ch1_var.get()
            while True:
                try:
                    r = self._proc_thread.results_queue.get_nowait()
                except queue.Empty:
                    break
                if r["type"] == "first_trig_only":
                    self.first_trigger_count += r["count"]
                    changed = True
                elif r["type"] == "passing":
                    self.first_trigger_count += r["first_trig_hits"]
                    self.stored_ch0.append(r["ch0"])
                    if use_ch1 and r["ch1"] is not None:
                        self.stored_ch1.append(r["ch1"])
                    self.passing_count += 1
                    self._last_start_idx = r["start_idx"]
                    changed = True

            if changed:
                self.first_trigger_count_var.set(self.first_trigger_count)
                self.passing_var.set(self.passing_count)
                self._plot_dirty = True
        except Exception:
            pass
        try:
            self.fig.canvas.get_tk_widget().after(100, self._drain_results)
        except Exception:
            pass

    def _redraw_timer(self):
        """Change 5: redraw at most once per second."""
        if not self.running:
            return
        if self._plot_dirty and self.stored_ch0:
            self._update_plot(self.ads_panel.sample_rate_var.get(),
                              self._last_start_idx)
            self._plot_dirty = False
        try:
            self.fig.canvas.get_tk_widget().after(1000, self._redraw_timer)
        except Exception:
            pass

    def _update_plot(self, fs, start_idx):
        use_ch1     = self.use_ch1_var.get()
        max_display = self.max_display_var.get()
        start_us    = self.start_us_var.get()

        first_trig_ch0 = self.first_trigger_ch0_var.get()
        trig_ch0       = self.trigger_ch0_var.get()

        def _plot_channel(ax, stored, first_trig, second_trig,
                          color_raw, color_off, color_filt, title):
            ax.clear()
            _style_ax(ax, title=title, ylabel="Amplitude (V)")
            # Change 8: stored is already a deque(maxlen=max_display)
            for idx, (raw_tr, off_tr, filt_tr, cross_in) in enumerate(stored):
                t_us  = (np.arange(len(raw_tr)) - cross_in) / fs * 1e6
                alpha = 0.3
                if idx == 0:
                    ax.plot(t_us, raw_tr,  color=color_raw,  alpha=alpha,
                            linewidth=0.7, label="Raw")
                    ax.plot(t_us, off_tr,  color=color_off,  alpha=alpha,
                            linewidth=0.7, label="Offset removed")
                    ax.plot(t_us, filt_tr, color=color_filt, alpha=alpha,
                            linewidth=0.9, label="Filtered")
                else:
                    ax.plot(t_us, raw_tr,  color=color_raw,  alpha=alpha, linewidth=0.7)
                    ax.plot(t_us, off_tr,  color=color_off,  alpha=alpha, linewidth=0.7)
                    ax.plot(t_us, filt_tr, color=color_filt, alpha=alpha, linewidth=0.9)
            ax.axhline(first_trig, color=C["trig1"], linestyle="--",
                       linewidth=1.0, label=f"First trig ({first_trig:.3g} V)")
            ax.axhline(second_trig, color=C["trig2"], linestyle="--",
                       linewidth=1.0, label=f"Second trig ({second_trig:.3g} V)")
            ax.axvline(start_us, color=C["win_start"], linestyle="--",
                       linewidth=0.8, label=f"Window start ({start_us:.3g} µs)")
            ax.legend(fontsize=7, facecolor=C["panel"],
                      edgecolor=C["border"], labelcolor=C["fg"])

        _plot_channel(self.ax0, self.stored_ch0,
                      first_trig_ch0, trig_ch0,
                      C["trace_raw"], C["trace_offset"], C["trace_ch0"],
                      f"Ch0 — last {len(self.stored_ch0)} traces")

        if use_ch1 and self.ax1 is not None and self.stored_ch1:
            first_trig_ch1 = self.first_trigger_ch1_var.get()
            trig_ch1       = self.trigger_ch1_var.get()
            _plot_channel(self.ax1, self.stored_ch1,
                          first_trig_ch1, trig_ch1,
                          C["trace_ch1_raw"], C["trace_ch1_offset"], C["trace_ch1"],
                          f"Ch1 — last {len(self.stored_ch1)} traces")
            self.ax1.set_xlabel("Time relative to first trigger (µs)")
        else:
            self.ax0.set_xlabel("Time relative to first trigger (µs)")

        self.fig.tight_layout(pad=1.5)
        self.canvas.draw()


# ===========================================================================
# Tab 2 – Live Histogram
# ===========================================================================

class HistogramTab:

    def __init__(self, parent, acq: AcquisitionManager):
        self.acq     = acq
        self.running = True

        self.records             = []      # fully-passing records
        self.all_charged_records = []      # records for all charged particles (if enabled)
        self.first_trigger_count = 0
        self.passing_count       = 0
        self._run_start_time     = None    # datetime when acquisition started
        self._count_times        = []      # list of datetime for each passing event

        # Change 7: accumulate rows in memory; flush to disk every N records
        # or every T seconds rather than on every batch.
        self._pending_records     = []     # unflushed fully-passing rows
        self._pending_all_charged = []     # unflushed all-charged rows
        self._pending_traces      = []     # unflushed trace rows
        self._last_flush_time     = time.monotonic()
        self._FLUSH_INTERVAL_S    = 5.0    # flush at most every 5 s
        self._FLUSH_N_RECORDS     = 200    # … or every 200 records
        self._header_written      = False

        # Change 5: decouple processing from plotting
        self._plot_dirty = False

        # ---- Tk vars ----
        self.use_ch1_var           = tk.BooleanVar(value=False)

        self.first_trigger_ch0_var = tk.DoubleVar(value=0.2)
        self.trigger_ch0_var       = tk.DoubleVar(value=0.01)
        self.pulses_ch0_var        = tk.IntVar(value=2)
        self.ch0_range_var         = tk.DoubleVar(value=5.0)
        self.ch0_attenuation_var   = tk.DoubleVar(value=1.0)

        self.first_trigger_ch1_var = tk.DoubleVar(value=0.2)
        self.trigger_ch1_var       = tk.DoubleVar(value=0.01)
        self.pulses_ch1_var        = tk.IntVar(value=1)
        self.ch1_range_var         = tk.DoubleVar(value=5.0)
        self.ch1_attenuation_var   = tk.DoubleVar(value=1.0)

        self.start_us_var   = tk.DoubleVar(value=0.5)
        self.stop_us_var    = tk.DoubleVar(value=40.0)
        self.bins_var       = tk.IntVar(value=100)
        self.filter_var     = tk.DoubleVar(value=100e6)
        self.holdoff_us_var = tk.DoubleVar(value=0.5)

        self.save_traces_var          = tk.BooleanVar(value=False)
        self.save_all_charged_var     = tk.BooleanVar(value=False)
        self.first_trigger_count_var  = tk.IntVar(value=0)
        self.passing_var              = tk.IntVar(value=0)
        self.count_rate_var           = tk.StringVar(value="Rate: — Hz")

        self.time_log_path  = None
        self.trace_csv_path = None

        self._hist_axes = {}

        self._build(parent)

        # Change 1: start ProcessingThread
        self._proc_thread = ProcessingThread(self.acq, self._processing_fn)
        self._proc_thread.get_fs = lambda: self.ads_panel.sample_rate_var.get()
        self._proc_thread.get_fc = lambda: self.filter_var.get()
        self._proc_thread.start()

        # Change 5: fast results-drain (100 ms), slow redraw (1500 ms)
        parent.after(100,  self._drain_results)
        parent.after(1500, self._redraw_timer)

    def _build(self, parent):
        sidebar_container, frame, _ = make_scrollable_sidebar(parent)
        sidebar_container.pack(side=tk.LEFT, fill="y")

        _section_label(frame, "── Channel 0 ──")
        add_row(frame, "First trigger level (V)",  self.first_trigger_ch0_var)
        add_row(frame, "Second trigger level (V)", self.trigger_ch0_var)
        add_row(frame, "Expected pulses (1 or 2)", self.pulses_ch0_var)
        add_row(frame, "Voltage range (V p-p)",    self.ch0_range_var)
        add_row(frame, "Attenuation",              self.ch0_attenuation_var)

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

        _section_label(frame, "── Signal Processing ──")
        add_row(frame, "Low-pass cutoff (Hz)",         self.filter_var)
        add_row(frame, "Window start after trig (µs)", self.start_us_var)
        add_row(frame, "Window stop after trig (µs)",  self.stop_us_var)
        add_row(frame, "Histogram bins",               self.bins_var)
        add_row(frame, "Holdoff (µs)",                 self.holdoff_us_var)

        bf = tk.Frame(frame, bg=C["bg"])
        bf.pack(fill="x", pady=(8, 4))
        _button(bf, "Reset",      self._reset,      width=9).pack(side=tk.LEFT, padx=2)
        _button(bf, "Export CSV", self._export_csv, width=9).pack(side=tk.LEFT, padx=2)

        _section_label(frame, "── Data Saving ──")
        _button(frame, "Set peak-data log path", self._set_log_path).pack(fill="x", pady=2)
        self.log_label = tk.Label(frame, text="(no log file set)", bg=C["bg"], fg=C["fg_dim"],
                                  wraplength=240, justify="left", font=("Helvetica", 8))
        self.log_label.pack(anchor="w")

        # Save all charged particles option
        _checkbutton(frame, "Save all charged particles",
                     self.save_all_charged_var).pack(anchor="w", pady=(4, 0))
        tk.Label(frame, text="(adds 'passes_all' column; records first-trig\npulses that fail full selection)",
                 bg=C["bg"], fg=C["fg_dim"], font=("Helvetica", 7),
                 justify="left", wraplength=240).pack(anchor="w")

        _checkbutton(frame, "Save raw traces to CSV",
                     self.save_traces_var,
                     command=self._toggle_save_traces).pack(anchor="w", pady=(6, 0))
        self.trace_path_label = tk.Label(frame, text="(traces not saved)",
                                         bg=C["bg"], fg=C["fg_dim"],
                                         wraplength=240, justify="left",
                                         font=("Helvetica", 8))
        self.trace_path_label.pack(anchor="w")

        self.ads_panel = AdsPanel(
            frame, self.acq, self.use_ch1_var,
            self.first_trigger_ch0_var, self.first_trigger_ch1_var,
            self.ch0_range_var, self.ch1_range_var,
            self.ch0_attenuation_var, self.ch1_attenuation_var,
            clear_graphs_cmd=self._clear_graphs,
        )

        _section_label(frame, "── Status ──")
        _counter_label(frame, "Events read (first trig):")
        _counter_value(frame, self.first_trigger_count_var)
        _counter_label(frame, "Passing triggers:")
        _counter_value(frame, self.passing_var)

        # Count rate display
        tk.Label(frame, textvariable=self.count_rate_var,
                 bg=C["bg"], fg=C["teal"],
                 font=("Helvetica", 9, "bold")).pack(anchor="w", pady=(4, 0))

        self._toggle_ch1()
        self._build_plot(parent)

    # ------------------------------------------------------------------
    # Dynamic histogram layout with scrollable right panel
    # ------------------------------------------------------------------

    def _build_plot(self, parent):
        # Scrollable container for the histogram figures
        scroll_container, self.plot_inner, self.plot_canvas = \
            make_scrollable_plot_frame(parent)
        scroll_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = plt.Figure(figsize=(11, 8))
        _style_figure(self.fig)
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, master=self.plot_inner)
        self.mpl_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._rebuild_hist_axes()

    def _rebuild_hist_axes(self):
        self.fig.clf()
        self._hist_axes = {}

        use_ch1    = self.use_ch1_var.get()
        pulses_ch0 = self.pulses_ch0_var.get()
        pulses_ch1 = self.pulses_ch1_var.get() if use_ch1 else 0
        trig_ch    = self.ads_panel.trigger_ch_var.get() if hasattr(self, "ads_panel") else 0

        n_ch0_cols = pulses_ch0 * 2
        n_ch1_cols = pulses_ch1 * 2 if use_ch1 else 0
        max_cols   = max(n_ch0_cols, n_ch1_cols, 2)

        # The trigger channel's dt histogram is shown in the top row
        has_dt_trig  = (pulses_ch0 == 2 and trig_ch == 0) or \
                       (use_ch1 and pulses_ch1 == 2 and trig_ch == 1)
        has_dt_inter = use_ch1
        has_rate_row = True   # always show count rate

        row_specs = []
        if has_dt_trig:
            row_specs.append(("dt_trig", 1.4))
        row_specs.append(("ch0", 1.0))
        if use_ch1:
            row_specs.append(("ch1", 1.0))
        if has_dt_inter:
            row_specs.append(("dt_inter", 1.4))
        row_specs.append(("rate", 1.2))

        # Resize figure height based on number of rows
        fig_h = max(4, 2.2 * len(row_specs))
        self.fig.set_size_inches(11, fig_h)
        self.mpl_canvas.get_tk_widget().config(height=int(fig_h * 96))

        n_rows = len(row_specs)
        ratios = [r for _, r in row_specs]
        gs = gridspec.GridSpec(n_rows, max_cols, figure=self.fig,
                               height_ratios=ratios,
                               hspace=0.65, wspace=0.45)

        for row_i, (row_type, _) in enumerate(row_specs):
            if row_type == "dt_trig":
                ax = self.fig.add_subplot(gs[row_i, :])
                lbl = ("Ch1" if trig_ch == 1 else "Ch0")
                _style_ax(ax, title=f"{lbl} dt = t2 − t1 (inter-pulse, trigger ch)",
                          xlabel="dt (µs)", ylabel="Count")
                self._hist_axes["dt_trig"] = ax

            elif row_type == "dt_inter":
                ax = self.fig.add_subplot(gs[row_i, :])
                _style_ax(ax, title="Inter-channel dt = t1_ch1 − t1_ch0",
                          xlabel="dt (µs)", ylabel="Count")
                self._hist_axes["dt_inter"] = ax

            elif row_type == "rate":
                ax = self.fig.add_subplot(gs[row_i, :])
                _style_ax(ax, title="Count rate over time",
                          xlabel="Time (s from run start)", ylabel="Rate (Hz)")
                self._hist_axes["rate"] = ax

            elif row_type == "ch0":
                col = 0
                for p in range(1, pulses_ch0 + 1):
                    ax_h = self.fig.add_subplot(gs[row_i, col])
                    _style_ax(ax_h, title=f"Ch0 P{p} height",
                              xlabel="Amplitude (V)", ylabel="Count",
                              title_size=8, label_size=7, tick_size=6)
                    self._hist_axes[f"ch0_p{p}_height"] = ax_h
                    col += 1
                    ax_f = self.fig.add_subplot(gs[row_i, col])
                    _style_ax(ax_f, title=f"Ch0 P{p} FWHM",
                              xlabel="FWHM (µs)", ylabel="Count",
                              title_size=8, label_size=7, tick_size=6)
                    self._hist_axes[f"ch0_p{p}_fwhm"] = ax_f
                    col += 1

            elif row_type == "ch1":
                col = 0
                for p in range(1, pulses_ch1 + 1):
                    ax_h = self.fig.add_subplot(gs[row_i, col])
                    _style_ax(ax_h, title=f"Ch1 P{p} height",
                              xlabel="Amplitude (V)", ylabel="Count",
                              title_size=8, label_size=7, tick_size=6)
                    self._hist_axes[f"ch1_p{p}_height"] = ax_h
                    col += 1
                    ax_f = self.fig.add_subplot(gs[row_i, col])
                    _style_ax(ax_f, title=f"Ch1 P{p} FWHM",
                              xlabel="FWHM (µs)", ylabel="Count",
                              title_size=8, label_size=7, tick_size=6)
                    self._hist_axes[f"ch1_p{p}_fwhm"] = ax_f
                    col += 1

        self.fig.subplots_adjust(left=0.07, right=0.97, top=0.95, bottom=0.08)
        self.mpl_canvas.draw()
        # Update scroll region
        self.plot_inner.update_idletasks()
        self.plot_canvas.configure(scrollregion=self.plot_canvas.bbox("all"))

    def _toggle_ch1(self):
        state = "normal" if self.use_ch1_var.get() else "disabled"
        for w in self.ch1_widgets:
            w.config(state=state)
        if hasattr(self, "fig"):
            self._rebuild_hist_axes()

    def _set_log_path(self):
        path = filedialog.asksaveasfilename(
            title="Peak-data log CSV", defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")])
        if path:
            self.time_log_path = path
            self._header_written = False   # reset header flag for new file
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
                title="Save traces CSV", defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")])
            if path:
                self.trace_csv_path = path
                self.trace_path_label.config(text=os.path.basename(path), fg=C["teal"])
            else:
                self.save_traces_var.set(False)
                self.trace_path_label.config(text="(traces not saved)", fg=C["fg_dim"])
        else:
            self.trace_csv_path = None
            self.trace_path_label.config(text="(traces not saved)", fg=C["fg_dim"])

    def _clear_graphs(self):
        self.records.clear()
        self.all_charged_records.clear()
        self._count_times.clear()
        self._run_start_time  = None
        self.first_trigger_count = 0
        self.passing_count       = 0
        self.first_trigger_count_var.set(0)
        self.passing_var.set(0)
        self.count_rate_var.set("Rate: — Hz")
        # Change 7: discard any pending (unwritten) data
        self._pending_records.clear()
        self._pending_all_charged.clear()
        self._pending_traces.clear()
        self._header_written  = False
        self._plot_dirty      = False
        self._rebuild_hist_axes()

    def _reset(self):
        self._clear_graphs()

    def _export_csv(self):
        src = (self.all_charged_records if (self.save_all_charged_var.get()
               and self.all_charged_records) else self.records)
        if not src:
            messagebox.showwarning("No Data", "No records to export.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if path:
            pd.DataFrame(src).to_csv(path, index=False)

    # ------------------------------------------------------------------
    # Change 1: processing_fn runs inside ProcessingThread (background)
    # ------------------------------------------------------------------

    def _processing_fn(self, batch, sos, do_filter, fs):
        """
        Called from ProcessingThread — no Tkinter calls allowed here.
        Change 4: uses sosfiltfilt via sos argument.
        Returns a list of result dicts consumed by the GUI thread.
        """
        start_idx       = int(self.start_us_var.get()  * 1e-6 * fs)
        stop_idx        = int(self.stop_us_var.get()   * 1e-6 * fs)
        holdoff_samples = int(self.holdoff_us_var.get() * 1e-6 * fs)

        first_trig_ch0 = self.first_trigger_ch0_var.get()
        trig_ch0       = self.trigger_ch0_var.get()
        pulses_ch0     = self.pulses_ch0_var.get()
        use_ch1        = self.use_ch1_var.get()
        first_trig_ch1 = self.first_trigger_ch1_var.get()
        trig_ch1       = self.trigger_ch1_var.get()
        pulses_ch1     = self.pulses_ch1_var.get()
        save_all       = self.save_all_charged_var.get()
        trig_ch        = self.ads_panel.trigger_ch_var.get()

        results = []

        for item in batch:
            raw_ch0  = item["ch0"]
            item_ts  = item.get("timestamp", datetime.datetime.now())

            (pass1_ch0, pass2_ch0, cross_ch0,
             t1_ch0, t2_ch0, dt_ch0,
             a1_ch0, fwhm1_ch0, a2_ch0, fwhm2_ch0,
             offset_ch0, filtered_ch0) = analyse_channel(
                raw_ch0, fs, first_trig_ch0, trig_ch0, pulses_ch0,
                start_idx, stop_idx, holdoff_samples, do_filter, sos)

            if not pass1_ch0:
                continue

            ts_str = item_ts.isoformat(timespec="milliseconds")

            if not pass2_ch0:
                if save_all:
                    # Full ch0 data even for first-trig-only events
                    results.append({
                        "type":       "first_trig_only",
                        "timestamp":  item_ts,
                        "ts_str":     ts_str,
                        "t1_ch0":     t1_ch0,
                        "t2_ch0":     np.nan,
                        "dt_ch0":     np.nan,
                        "a1_ch0":     a1_ch0,
                        "fwhm1_ch0":  fwhm1_ch0,
                        "a2_ch0":     np.nan,
                        "fwhm2_ch0":  np.nan,
                        "t1_ch1":     np.nan,
                        "t2_ch1":     np.nan,
                        "dt_ch1":     np.nan,
                        "a1_ch1":     np.nan,
                        "fwhm1_ch1":  np.nan,
                        "a2_ch1":     np.nan,
                        "fwhm2_ch1":  np.nan,
                        "dt_inter":   np.nan,
                        "raw_ch0":    raw_ch0 if self.save_traces_var.get() else None,
                    })
                else:
                    results.append({"type": "first_trig_count"})
                continue

            # Ch1
            t1_ch1 = t2_ch1 = dt_ch1 = np.nan
            a1_ch1 = fwhm1_ch1 = a2_ch1 = fwhm2_ch1 = np.nan
            dt_inter = np.nan
            raw_ch1_data = None

            if use_ch1 and item["ch1"] is not None:
                (pass1_ch1, pass2_ch1, cross_ch1,
                 t1_ch1, t2_ch1, dt_ch1,
                 a1_ch1, fwhm1_ch1, a2_ch1, fwhm2_ch1,
                 offset_ch1, filtered_ch1) = analyse_channel(
                    item["ch1"], fs, first_trig_ch1, trig_ch1, pulses_ch1,
                    start_idx, stop_idx, holdoff_samples, do_filter, sos)
                raw_ch1_data = item["ch1"]

                if not (pass1_ch1 and pass2_ch1):
                    if save_all:
                        # ch0 passed both triggers; ch1 failed — record with full ch0 data
                        results.append({
                            "type":       "first_trig_only",
                            "timestamp":  item_ts,
                            "ts_str":     ts_str,
                            "t1_ch0":     t1_ch0,
                            "t2_ch0":     t2_ch0,
                            "dt_ch0":     dt_ch0,
                            "a1_ch0":     a1_ch0,
                            "fwhm1_ch0":  fwhm1_ch0,
                            "a2_ch0":     a2_ch0,
                            "fwhm2_ch0":  fwhm2_ch0,
                            "t1_ch1":     t1_ch1,
                            "t2_ch1":     np.nan,
                            "dt_ch1":     np.nan,
                            "a1_ch1":     a1_ch1,
                            "fwhm1_ch1":  fwhm1_ch1,
                            "a2_ch1":     np.nan,
                            "fwhm2_ch1":  np.nan,
                            "dt_inter":   np.nan,
                            "raw_ch0":    raw_ch0 if self.save_traces_var.get() else None,
                            "raw_ch1":    raw_ch1_data if self.save_traces_var.get() else None,
                        })
                    else:
                        results.append({"type": "first_trig_count"})
                    continue
                dt_inter = t1_ch1 - t1_ch0

            dt_trig = dt_ch1 if trig_ch == 1 else dt_ch0
            results.append({
                "type":       "passing",
                "timestamp":  item_ts,
                "ts_str":     ts_str,
                "t1_ch0":     t1_ch0,
                "t2_ch0":     t2_ch0,
                "dt_ch0":     dt_ch0,
                "a1_ch0":     a1_ch0,
                "fwhm1_ch0":  fwhm1_ch0,
                "a2_ch0":     a2_ch0,
                "fwhm2_ch0":  fwhm2_ch0,
                "t1_ch1":     t1_ch1,
                "t2_ch1":     t2_ch1,
                "dt_ch1":     dt_ch1,
                "a1_ch1":     a1_ch1,
                "fwhm1_ch1":  fwhm1_ch1,
                "a2_ch1":     a2_ch1,
                "fwhm2_ch1":  fwhm2_ch1,
                "dt_inter":   dt_inter,
                "raw_ch0":    raw_ch0 if self.save_traces_var.get() else None,
                "raw_ch1":    raw_ch1_data if self.save_traces_var.get() else None,
            })

        return results

    # Change 5: GUI drains results fast; redraws on a fixed timer
    def _drain_results(self):
        """GUI thread: consume results_queue, accumulate pending I/O."""
        if not self.running:
            return
        try:
            new_records     = []
            new_all_charged = []

            while True:
                try:
                    r = self._proc_thread.results_queue.get_nowait()
                except queue.Empty:
                    break

                if r["type"] == "first_trig_count":
                    self.first_trigger_count += 1

                elif r["type"] == "first_trig_only":
                    self.first_trigger_count += 1
                    if self._run_start_time is None:
                        self._run_start_time = r["timestamp"]
                    ac_rec = {
                        "timestamp":   r["ts_str"],
                        "passes_all":  0,
                        "t1_ch0":      r["t1_ch0"],
                        "t2_ch0":      r["t2_ch0"],
                        "dt_ch0":      r["dt_ch0"],
                        "a1_ch0":      r["a1_ch0"],
                        "fwhm1_ch0":   r["fwhm1_ch0"],
                        "a2_ch0":      r["a2_ch0"],
                        "fwhm2_ch0":   r["fwhm2_ch0"],
                        "t1_ch1":      r["t1_ch1"],
                        "t2_ch1":      r["t2_ch1"],
                        "dt_ch1":      r["dt_ch1"],
                        "a1_ch1":      r["a1_ch1"],
                        "fwhm1_ch1":   r["fwhm1_ch1"],
                        "a2_ch1":      r["a2_ch1"],
                        "fwhm2_ch1":   r["fwhm2_ch1"],
                        "dt_inter":    r["dt_inter"],
                    }
                    self.all_charged_records.append(ac_rec)
                    new_all_charged.append(ac_rec)
                    if r.get("raw_ch0") is not None:
                        self._pending_traces.append(r["raw_ch0"].tolist())
                    if r.get("raw_ch1") is not None:
                        self._pending_traces.append(r["raw_ch1"].tolist())

                elif r["type"] == "passing":
                    self.first_trigger_count += 1
                    if self._run_start_time is None:
                        self._run_start_time = r["timestamp"]
                    self._count_times.append(r["timestamp"])
                    self.passing_count += 1

                    rec = {
                        "timestamp":   r["ts_str"],
                        "t1_ch0":      r["t1_ch0"],
                        "t2_ch0":      r["t2_ch0"],
                        "dt_ch0":      r["dt_ch0"],
                        "a1_ch0":      r["a1_ch0"],
                        "fwhm1_ch0":   r["fwhm1_ch0"],
                        "a2_ch0":      r["a2_ch0"],
                        "fwhm2_ch0":   r["fwhm2_ch0"],
                        "t1_ch1":      r["t1_ch1"],
                        "t2_ch1":      r["t2_ch1"],
                        "dt_ch1":      r["dt_ch1"],
                        "a1_ch1":      r["a1_ch1"],
                        "fwhm1_ch1":   r["fwhm1_ch1"],
                        "a2_ch1":      r["a2_ch1"],
                        "fwhm2_ch1":   r["fwhm2_ch1"],
                        "dt_inter":    r["dt_inter"],
                    }
                    if self.save_all_charged_var.get():
                        rec["passes_all"] = 1
                    self.records.append(rec)
                    new_records.append(rec)

                    if self.save_all_charged_var.get():
                        ac_rec = dict(rec)
                        ac_rec["passes_all"] = 1
                        self.all_charged_records.append(ac_rec)
                        new_all_charged.append(ac_rec)

                    if r.get("raw_ch0") is not None:
                        self._pending_traces.append(r["raw_ch0"].tolist())
                    if r.get("raw_ch1") is not None:
                        self._pending_traces.append(r["raw_ch1"].tolist())

            if new_records or new_all_charged:
                self.first_trigger_count_var.set(self.first_trigger_count)
                self.passing_var.set(self.passing_count)
                self._update_count_rate()
                self._plot_dirty = True

                # Change 7: accumulate pending I/O; flush on interval/size
                self._pending_records.extend(new_records)
                self._pending_all_charged.extend(new_all_charged)
                now = time.monotonic()
                elapsed = now - self._last_flush_time
                total_pending = len(self._pending_records) + len(self._pending_all_charged)
                if (elapsed >= self._FLUSH_INTERVAL_S or
                        total_pending >= self._FLUSH_N_RECORDS):
                    self._flush_pending_io()
                    self._last_flush_time = now
            elif self.first_trigger_count > 0:
                # First-trig-only counts still need updating
                self.first_trigger_count_var.set(self.first_trigger_count)

        except Exception:
            pass
        try:
            self.mpl_canvas.get_tk_widget().after(100, self._drain_results)
        except Exception:
            pass

    def _redraw_timer(self):
        """Change 5: redraw histograms at most once per 1.5 seconds."""
        if not self.running:
            return
        if self._plot_dirty and self.records:
            self._update_histograms()
            self._plot_dirty = False
        try:
            self.mpl_canvas.get_tk_widget().after(1500, self._redraw_timer)
        except Exception:
            pass

    def _update_count_rate(self):
        """Compute rolling count rate over all events and update display."""
        if not self._count_times:
            self.count_rate_var.set("Rate: — Hz")
            return
        # Use last 60 s window for rolling rate
        now   = self._count_times[-1]
        cutoff = now - datetime.timedelta(seconds=60)
        recent = [t for t in self._count_times if t >= cutoff]
        if len(recent) >= 2:
            span = (recent[-1] - recent[0]).total_seconds()
            rate = (len(recent) - 1) / span if span > 0 else 0.0
            self.count_rate_var.set(f"Rate (60 s): {rate:.3f} Hz")
        else:
            total_span = (now - self._run_start_time).total_seconds()
            if total_span > 0:
                rate = self.passing_count / total_span
                self.count_rate_var.set(f"Rate (total): {rate:.3f} Hz")

    def _build_csv_header(self):
        """Return header lines describing run settings."""
        fs       = self.ads_panel.sample_rate_var.get()
        trig_ch  = self.ads_panel.trigger_ch_var.get()
        t1_ch0   = self.first_trigger_ch0_var.get()
        t2_ch0   = self.trigger_ch0_var.get()
        p_ch0    = self.pulses_ch0_var.get()
        t1_ch1   = self.first_trigger_ch1_var.get()
        t2_ch1   = self.trigger_ch1_var.get()
        p_ch1    = self.pulses_ch1_var.get()
        use_ch1  = self.use_ch1_var.get()
        now      = datetime.datetime.now().isoformat(timespec="seconds")
        save_all = self.save_all_charged_var.get()

        lines = [
            f"# Muon Detector Run — started {now}",
            f"# Sample rate (Hz): {fs:.6g}",
            f"# Trigger channel: {trig_ch}",
            f"# Ch0 first trigger level (V): {t1_ch0}",
        ]
        if p_ch0 == 2:
            lines.append(f"# Ch0 second trigger level (V): {t2_ch0}")
        lines.append(f"# Ch0 expected pulses: {p_ch0}")

        if use_ch1:
            lines.append(f"# Ch1 first trigger level (V): {t1_ch1}")
            if p_ch1 == 2:
                lines.append(f"# Ch1 second trigger level (V): {t2_ch1}")
            lines.append(f"# Ch1 expected pulses: {p_ch1}")
        else:
            lines.append("# Ch1: disabled")

        if save_all:
            lines.append(
                "# passes_all=1 : passed all triggers; "
                "passes_all=0 : passed first trigger only"
            )
        lines.append("#")
        return "\n".join(lines) + "\n"

    def _flush_pending_io(self):
        """
        Change 7: batch-write all accumulated records and traces to disk.
        Called on a time/size threshold rather than every batch.
        """
        if not self.time_log_path:
            self._pending_records.clear()
            self._pending_all_charged.clear()
        else:
            save_all     = self.save_all_charged_var.get()
            rows_to_save = (self._pending_all_charged
                            if save_all and self._pending_all_charged
                            else self._pending_records)
            if rows_to_save:
                file_exists = os.path.exists(self.time_log_path)
                try:
                    df = pd.DataFrame(rows_to_save)
                    if not file_exists or not self._header_written:
                        with open(self.time_log_path, "w") as f:
                            f.write(self._build_csv_header())
                        df.to_csv(self.time_log_path, mode="a",
                                  header=True, index=False)
                        self._header_written = True
                    else:
                        df.to_csv(self.time_log_path, mode="a",
                                  header=False, index=False)
                except Exception as e:
                    print(f"[Histogram] log write error: {e}")
            self._pending_records.clear()
            self._pending_all_charged.clear()

        # Flush pending traces
        if self._pending_traces and self.save_traces_var.get() and self.trace_csv_path:
            try:
                write_header = not os.path.exists(self.trace_csv_path)
                pd.DataFrame(self._pending_traces).to_csv(
                    self.trace_csv_path, mode="a",
                    header=write_header, index=False)
            except Exception as e:
                print(f"[Histogram] trace CSV write error: {e}")
        self._pending_traces.clear()

    def _update_histograms(self):
        if not self.records:
            return
        df   = pd.DataFrame(self.records)
        bins = self.bins_var.get()

        use_ch1    = self.use_ch1_var.get()
        pulses_ch0 = self.pulses_ch0_var.get()
        pulses_ch1 = self.pulses_ch1_var.get() if use_ch1 else 0
        trig_ch    = self.ads_panel.trigger_ch_var.get()

        def _hist(key, data_col, color):
            ax = self._hist_axes.get(key)
            if ax is None:
                return
            ax.clear()
            title  = ax.get_title() or key
            xlabel = ax.get_xlabel() or ""
            _style_ax(ax, title=title, xlabel=xlabel, ylabel="Count",
                      title_size=8, label_size=7, tick_size=6)
            clean = df[data_col].dropna() if data_col in df.columns else pd.Series(dtype=float)
            if len(clean):
                ax.hist(clean, bins=bins, color=color, histtype="stepfilled")

        # Trigger-channel dt histogram
        dt_col = "dt_ch1" if trig_ch == 1 else "dt_ch0"
        _hist("dt_trig", dt_col, C["hist_dt"])
        if "dt_trig" in self._hist_axes:
            lbl = "Ch1" if trig_ch == 1 else "Ch0"
            ax = self._hist_axes["dt_trig"]
            ax.set_title(f"{lbl} dt = t2 − t1 (inter-pulse, trigger ch)",
                         fontsize=9, color=C["gold"])

        _hist("dt_inter", "dt_inter", C["hist_inter"])

        for p in range(1, pulses_ch0 + 1):
            col_h = "a1_ch0"    if p == 1 else "a2_ch0"
            col_f = "fwhm1_ch0" if p == 1 else "fwhm2_ch0"
            clr   = C["hist_ch0p1"] if p == 1 else C["hist_ch0p2"]
            _hist(f"ch0_p{p}_height", col_h, clr)
            _hist(f"ch0_p{p}_fwhm",   col_f, clr)

        for p in range(1, pulses_ch1 + 1):
            col_h = "a1_ch1"    if p == 1 else "a2_ch1"
            col_f = "fwhm1_ch1" if p == 1 else "fwhm2_ch1"
            clr   = C["hist_ch1p1"] if p == 1 else C["hist_ch1p2"]
            _hist(f"ch1_p{p}_height", col_h, clr)
            _hist(f"ch1_p{p}_fwhm",   col_f, clr)

        # Count rate over time plot
        ax_rate = self._hist_axes.get("rate")
        if ax_rate is not None and self._count_times and self._run_start_time:
            ax_rate.clear()
            _style_ax(ax_rate, title="Count rate over time",
                      xlabel="Time (s from run start)", ylabel="Rate (Hz)")
            times_s = np.array([(t - self._run_start_time).total_seconds()
                                 for t in self._count_times])
            if len(times_s) > 1:
                # Compute rate in 10 s bins
                bin_w = 10.0
                t_max = times_s[-1]
                edges = np.arange(0, t_max + bin_w, bin_w)
                counts, _ = np.histogram(times_s, bins=edges)
                centres    = 0.5 * (edges[:-1] + edges[1:])
                rates      = counts / bin_w
                ax_rate.step(centres, rates, color=C["hist_rate"], where="mid", linewidth=1.2)
                ax_rate.fill_between(centres, rates, step="mid",
                                     color=C["hist_rate"], alpha=0.25)

        # Include all_charged events in rate if save_all enabled
        if self.save_all_charged_var.get() and self.all_charged_records and ax_rate is not None:
            # The rate plot already covers passing events; this serves as a reminder
            # that all_charged_records includes both passes_all=0 and passes_all=1
            pass

        self.fig.subplots_adjust(left=0.07, right=0.97, top=0.95, bottom=0.08)
        self.mpl_canvas.draw()
        self.plot_inner.update_idletasks()
        self.plot_canvas.configure(scrollregion=self.plot_canvas.bbox("all"))


# ===========================================================================
# Top-level app
# ===========================================================================

class App:
    def __init__(self, root):
        self.root = root
        root.title("Muon Detector Program")
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        _apply_global_theme(root)

        nb = ttk.Notebook(root)
        nb.pack(fill=tk.BOTH, expand=True)

        tab1 = tk.Frame(nb, bg=C["bg"])
        tab2 = tk.Frame(nb, bg=C["bg"])
        nb.add(tab1, text="  Signal Viewer  ")
        nb.add(tab2, text="  Live Histogram  ")

        self.viewer    = SignalViewerTab(tab1, AcquisitionManager())
        self.histogram = HistogramTab(tab2, AcquisitionManager())

    def _on_close(self):
        self.viewer.running    = False
        self.histogram.running = False
        # Stop background processing threads before disconnecting
        self.viewer._proc_thread.stop()
        self.histogram._proc_thread.stop()
        # Flush any data still pending in the histogram's I/O buffer
        self.histogram._flush_pending_io()
        self.viewer.acq.disconnect()
        self.histogram.acq.disconnect()
        self.root.quit()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()