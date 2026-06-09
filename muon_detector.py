"""
muo_gui.py
==========
Two-tab Tkinter GUI for muon detector data acquisition using the
Digilent WaveForms ADS hardware (waveforms_ads.py).

Tab 1 – Signal Viewer   (based on muo_view_signals.py)
Tab 2 – Live Histogram  (based on muo_histogram.py)

Key differences from the original CSV-based scripts
----------------------------------------------------
* Data comes directly from the ADS device via a background acquisition
  thread; there is no CSV file to select or poll.
* "Save traces" checkbox (default OFF) – when OFF only the peak-data
  records (save_records / time_log) are written.  When ON the raw
  waveform rows are also written to a CSV alongside the records log.
* Auto-delete logic is removed; it is replaced by the save-traces toggle.
* Both tabs share a single WaveFormsADS device handle managed by the
  top-level App object.
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

# ---------------------------------------------------------------------------
# Import ADS driver (graceful fallback when hardware is absent)
# ---------------------------------------------------------------------------
try:
    from waveforms_ads import WaveFormsADS, DWFError
    ADS_AVAILABLE = True
except Exception as _ads_import_err:
    ADS_AVAILABLE = False
    print(f"[muo_gui] waveforms_ads import failed: {_ads_import_err}")
    print("[muo_gui] Running in 'no-hardware' mode – acquisition disabled.")


# ===========================================================================
# Shared acquisition layer
# ===========================================================================

class AcquisitionManager:
    """
    Owns the WaveFormsADS device handle and the background capture thread.

    Captured traces are placed in `self.queue` as:
        {"ch0": np.ndarray, "ch1": np.ndarray | None}
    """

    def __init__(self):
        self.device: "WaveFormsADS | None" = None
        self.queue: queue.Queue = queue.Queue(maxsize=2000)

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # -- ADS hardware parameters (set from UI before starting) --
        self.sample_rate_hz   = 100e6
        self.buffer_size      = 8192
        self.trigger_level_v  = 0.2       # hardware trigger (rising edge)
        self.trigger_channel  = 0
        self.auto_timeout_s   = 1.0
        self.acquisition_timeout_s = 5.0
        self.ch0_range_v      = 5.0
        self.ch1_range_v      = 5.0
        self.use_ch1          = False

        self.status_var: tk.StringVar | None = None  # set by UI

    # ------------------------------------------------------------------ #

    def connect(self) -> str:
        """Open the first available ADS device. Returns status string."""
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

    # ------------------------------------------------------------------ #

    def start_acquisition(self) -> str:
        """Start the background capture thread. Returns status string."""
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
        """Runs in background thread – continuously captures single waveforms."""
        while not self._stop_event.is_set():
            with self._lock:
                dev = self.device
            if dev is None:
                break
            try:
                ch0 = dev.analog_in_capture(
                    channel=0,
                    sample_rate_hz=self.sample_rate_hz,
                    buffer_size=self.buffer_size,
                    trigger_level_v=self.trigger_level_v,
                    trigger_channel=self.trigger_channel,
                    auto_timeout_s=self.auto_timeout_s,
                    timeout_s=self.acquisition_timeout_s,
                )
                ch1 = None
                if self.use_ch1:
                    ch1 = dev.analog_in_capture(
                        channel=1,
                        sample_rate_hz=self.sample_rate_hz,
                        buffer_size=self.buffer_size,
                        trigger_level_v=None,          # ch1 free-runs with ch0 trigger
                        auto_timeout_s=self.auto_timeout_s,
                        timeout_s=self.acquisition_timeout_s,
                    )
                if not self.queue.full():
                    self.queue.put_nowait({"ch0": ch0, "ch1": ch1})
            except TimeoutError:
                pass   # no trigger – loop and try again
            except Exception as e:
                if self._stop_event.is_set():
                    break
                self._set_status(f"Capture error: {e}")
                time.sleep(0.5)

    def _set_status(self, msg: str):
        if self.status_var is not None:
            try:
                self.status_var.set(msg)
            except Exception:
                pass


# ===========================================================================
# Shared signal-processing helpers (used by both tabs)
# ===========================================================================

def find_first_trigger_index(row, level):
    above  = row >= level
    rising = np.where(~above[:-1] & above[1:])[0] + 1
    return int(rising[0]) if len(rising) > 0 else None


def count_pulses(window, trigger_level, holdoff_samples):
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


# ===========================================================================
# Shared scrollable sidebar builder
# ===========================================================================

def make_scrollable_sidebar(parent, width=270):
    """Return (outer_container, inner_frame, scrollable_canvas)."""
    container = tk.Frame(parent)
    canvas    = tk.Canvas(container, width=width, highlightthickness=0)
    scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill="y")
    canvas.pack(side=tk.LEFT, fill="y", expand=True)

    frame        = tk.Frame(canvas)
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


# ===========================================================================
# ADS settings sub-panel (shared by both tabs via reference to acq manager)
# ===========================================================================

class AdsPanel:
    """
    Builds the 'ADS Hardware' section inside a given parent frame.
    Reads/writes values on the shared AcquisitionManager.
    """

    def __init__(self, frame: tk.Frame, acq: AcquisitionManager,
                 use_ch1_var: tk.BooleanVar):
        self.acq         = acq
        self.use_ch1_var = use_ch1_var

        # ---- Tk vars mirroring AcquisitionManager fields ----
        self.sample_rate_var       = tk.DoubleVar(value=acq.sample_rate_hz)
        self.buffer_size_var       = tk.IntVar(value=acq.buffer_size)
        self.hw_trigger_var        = tk.DoubleVar(value=acq.trigger_level_v)
        self.hw_trigger_ch_var     = tk.IntVar(value=acq.trigger_channel)
        self.auto_timeout_var      = tk.DoubleVar(value=acq.auto_timeout_s)
        self.acq_timeout_var       = tk.DoubleVar(value=acq.acquisition_timeout_s)
        self.ch0_range_var         = tk.DoubleVar(value=acq.ch0_range_v)
        self.ch1_range_var         = tk.DoubleVar(value=acq.ch1_range_v)

        self.status_var = tk.StringVar(value="Not connected")
        acq.status_var  = self.status_var

        self._build(frame)

    def _add(self, frame, label, var):
        tk.Label(frame, text=label).pack(anchor="w")
        tk.Entry(frame, textvariable=var).pack(fill="x", pady=2)

    def _build(self, frame):
        tk.Label(frame, text="── ADS Hardware ──",
                 font=("", 9, "bold")).pack(anchor="w", pady=(10, 2))

        self._add(frame, "Sample rate (Hz)",          self.sample_rate_var)
        self._add(frame, "Buffer size (samples)",     self.buffer_size_var)
        self._add(frame, "HW trigger level (V)",      self.hw_trigger_var)
        self._add(frame, "HW trigger channel (0/1)",  self.hw_trigger_ch_var)
        self._add(frame, "Auto-timeout (s, 0=strict)",self.auto_timeout_var)
        self._add(frame, "Acq timeout (s)",           self.acq_timeout_var)
        self._add(frame, "Ch0 voltage range (V p-p)", self.ch0_range_var)
        self._add(frame, "Ch1 voltage range (V p-p)", self.ch1_range_var)

        btn_frame = tk.Frame(frame)
        btn_frame.pack(fill="x", pady=(6, 2))
        tk.Button(btn_frame, text="Connect",    command=self._connect,
                  width=10).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Disconnect", command=self._disconnect,
                  width=10).pack(side=tk.LEFT, padx=2)

        btn_frame2 = tk.Frame(frame)
        btn_frame2.pack(fill="x", pady=2)
        tk.Button(btn_frame2, text="▶ Start",   command=self._start,
                  width=10).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame2, text="■ Stop",    command=self._stop,
                  width=10).pack(side=tk.LEFT, padx=2)

        tk.Label(frame, textvariable=self.status_var,
                 fg="darkblue", wraplength=240,
                 justify="left", font=("", 8)).pack(anchor="w", pady=(4, 0))

    def _push_to_acq(self):
        """Write current UI values into the AcquisitionManager."""
        a = self.acq
        a.sample_rate_hz          = self.sample_rate_var.get()
        a.buffer_size             = self.buffer_size_var.get()
        a.trigger_level_v         = self.hw_trigger_var.get()
        a.trigger_channel         = self.hw_trigger_ch_var.get()
        a.auto_timeout_s          = self.auto_timeout_var.get()
        a.acquisition_timeout_s   = self.acq_timeout_var.get()
        a.ch0_range_v             = self.ch0_range_var.get()
        a.ch1_range_v             = self.ch1_range_var.get()
        a.use_ch1                 = self.use_ch1_var.get()

    def _connect(self):
        msg = self.acq.connect()
        self.status_var.set(msg)

    def _disconnect(self):
        self.acq.disconnect()
        self.status_var.set("Disconnected")

    def _start(self):
        self._push_to_acq()
        msg = self.acq.start_acquisition()
        self.status_var.set(msg)

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

        self.displayed_ch0: list = []
        self.displayed_ch1: list = []
        self.total_counts  = 0

        # ---- Tk vars ----
        self.use_ch1_var           = tk.BooleanVar(value=False)
        self.first_trigger_ch0_var = tk.DoubleVar(value=0.2)
        self.trigger_ch0_var       = tk.DoubleVar(value=0.01)
        self.pulses_ch0_var        = tk.IntVar(value=1)
        self.first_trigger_ch1_var = tk.DoubleVar(value=0.2)
        self.trigger_ch1_var       = tk.DoubleVar(value=0.01)
        self.pulses_ch1_var        = tk.IntVar(value=1)
        self.fs_var                = tk.DoubleVar(value=100e6)
        self.start_us_var          = tk.DoubleVar(value=0.5)
        self.stop_us_var           = tk.DoubleVar(value=40.0)
        self.filter_var            = tk.DoubleVar(value=100e6)
        self.holdoff_us_var        = tk.DoubleVar(value=0.5)
        self.max_display_var       = tk.IntVar(value=50)
        self.save_traces_var       = tk.BooleanVar(value=False)
        self.counts_var            = tk.IntVar(value=0)
        self.passing_var           = tk.IntVar(value=0)

        self.trace_csv_path: str | None = None

        self._build(parent)
        parent.after(500, self._update_loop)

    # ------------------------------------------------------------------ #

    def _build(self, parent):
        sidebar_container, frame, _ = make_scrollable_sidebar(parent)
        sidebar_container.pack(side=tk.LEFT, fill="y")

        def add_row(label, var):
            tk.Label(frame, text=label).pack(anchor="w")
            tk.Entry(frame, textvariable=var).pack(fill="x", pady=2)

        # Ch0
        tk.Label(frame, text="── Channel 0 ──",
                 font=("", 9, "bold")).pack(anchor="w", pady=(4, 2))
        add_row("Ch0 first trigger level (V)",  self.first_trigger_ch0_var)
        add_row("Ch0 second trigger level (V)", self.trigger_ch0_var)
        add_row("Ch0 expected pulse count",     self.pulses_ch0_var)

        # Ch1
        tk.Label(frame, text="── Channel 1 ──",
                 font=("", 9, "bold")).pack(anchor="w", pady=(8, 2))
        tk.Checkbutton(frame, text="Use Channel 1", variable=self.use_ch1_var,
                       command=self._toggle_ch1).pack(anchor="w")
        self.ch1_widgets = []

        def add_ch1_row(label, var):
            lbl = tk.Label(frame, text=label)
            lbl.pack(anchor="w")
            ent = tk.Entry(frame, textvariable=var)
            ent.pack(fill="x", pady=2)
            self.ch1_widgets += [lbl, ent]

        add_ch1_row("Ch1 first trigger level (V)",  self.first_trigger_ch1_var)
        add_ch1_row("Ch1 second trigger level (V)", self.trigger_ch1_var)
        add_ch1_row("Ch1 expected pulse count",     self.pulses_ch1_var)

        # Shared
        tk.Label(frame, text="── Shared ──",
                 font=("", 9, "bold")).pack(anchor="w", pady=(8, 2))
        add_row("Sampling frequency (Hz)",       self.fs_var)
        add_row("Low-pass cutoff (Hz)",          self.filter_var)
        add_row("Start time after trigger (µs)", self.start_us_var)
        add_row("Stop time after trigger (µs)",  self.stop_us_var)
        add_row("Holdoff (µs)",                  self.holdoff_us_var)
        add_row("Max traces to display",         self.max_display_var)

        tk.Button(frame, text="Reset", command=self._reset).pack(fill="x", pady=4)

        # Save traces
        tk.Label(frame, text="── Data Saving ──",
                 font=("", 9, "bold")).pack(anchor="w", pady=(10, 2))
        tk.Checkbutton(frame, text="Save raw traces to CSV",
                       variable=self.save_traces_var,
                       command=self._toggle_save_traces).pack(anchor="w")
        self.trace_path_label = tk.Label(frame, text="(traces not saved)",
                                         wraplength=230, justify="left",
                                         fg="gray", font=("", 8))
        self.trace_path_label.pack(anchor="w")

        # ADS panel
        self.ads_panel = AdsPanel(frame, self.acq, self.use_ch1_var)

        # Counters
        tk.Label(frame, text="Counts (events read):").pack(anchor="w", pady=(10, 0))
        tk.Label(frame, textvariable=self.counts_var).pack(anchor="w")
        tk.Label(frame, text="Passing events:").pack(anchor="w", pady=(6, 0))
        tk.Label(frame, textvariable=self.passing_var).pack(anchor="w")

        self._toggle_ch1()
        self._build_plot(parent)

    def _build_plot(self, parent):
        self.fig    = plt.Figure(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self._rebuild_axes()

    def _rebuild_axes(self):
        self.fig.clf()
        use_ch1 = self.use_ch1_var.get()
        if use_ch1:
            self.ax0, self.ax1 = self.fig.subplots(2, 1, sharex=True)
        else:
            self.ax0 = self.fig.subplots(1, 1)
            self.ax1 = None
        self.fig.tight_layout()
        self.canvas.draw()

    def _toggle_ch1(self):
        state = "normal" if self.use_ch1_var.get() else "disabled"
        for w in self.ch1_widgets:
            w.config(state=state)
        if hasattr(self, "fig"):
            self._rebuild_axes()

    def _toggle_save_traces(self):
        if self.save_traces_var.get():
            path = filedialog.asksaveasfilename(
                title="Save traces CSV",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")]
            )
            if path:
                self.trace_csv_path = path
                self.trace_path_label.config(text=os.path.basename(path), fg="darkgreen")
            else:
                self.save_traces_var.set(False)
                self.trace_path_label.config(text="(traces not saved)", fg="gray")
        else:
            self.trace_csv_path = None
            self.trace_path_label.config(text="(traces not saved)", fg="gray")

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
        """Process up to 50 buffered waveforms per tick."""
        batch = []
        for _ in range(50):
            try:
                item = self.acq.queue.get_nowait()
                batch.append(item)
            except queue.Empty:
                break
        if not batch:
            return
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
        if do_filter:
            b, a_coef = sig.butter(4, fc / nyq, btype="low")

        new_ch0 = []
        new_ch1 = []
        trace_rows = []

        for item in batch:
            raw_ch0 = item["ch0"]
            self.total_counts += 1

            cross_ch0 = find_first_trigger_index(raw_ch0, first_trig_ch0)
            if cross_ch0 is None:
                continue
            chopped_ch0 = raw_ch0[cross_ch0:]
            if stop_idx > len(chopped_ch0):
                continue
            win_ch0  = chopped_ch0[start_idx:stop_idx]
            filt_ch0 = (sig.filtfilt(b, a_coef, win_ch0) if do_filter
                        else win_ch0.copy())
            if count_pulses(filt_ch0, trig_ch0, holdoff_samples) != pulses_ch0:
                continue

            if use_ch1 and item["ch1"] is not None:
                raw_ch1   = item["ch1"]
                cross_ch1 = find_first_trigger_index(raw_ch1, first_trig_ch1)
                if cross_ch1 is None:
                    continue
                chopped_ch1 = raw_ch1[cross_ch1:]
                if stop_idx > len(chopped_ch1):
                    continue
                win_ch1  = chopped_ch1[start_idx:stop_idx]
                filt_ch1 = (sig.filtfilt(b, a_coef, win_ch1) if do_filter
                            else win_ch1.copy())
                if count_pulses(filt_ch1, trig_ch1, holdoff_samples) != pulses_ch1:
                    continue
                new_ch1.append(win_ch1)
                if self.save_traces_var.get():
                    trace_rows.append(win_ch0.tolist())
                    trace_rows.append(win_ch1.tolist())
            else:
                if self.save_traces_var.get():
                    trace_rows.append(win_ch0.tolist())

            new_ch0.append(win_ch0)

        self.counts_var.set(self.total_counts)

        if not new_ch0:
            return

        self.displayed_ch0.extend(new_ch0)
        if use_ch1:
            self.displayed_ch1.extend(new_ch1)
        self.passing_var.set(len(self.displayed_ch0))

        # optionally save raw traces
        if self.save_traces_var.get() and trace_rows and self.trace_csv_path:
            self._append_trace_csv(trace_rows)

        self._update_plot(fs, start_idx)

    def _append_trace_csv(self, rows):
        try:
            write_header = not os.path.exists(self.trace_csv_path)
            pd.DataFrame(rows).to_csv(
                self.trace_csv_path, mode="a", header=write_header, index=False
            )
        except Exception as e:
            print(f"[SignalViewer] trace CSV write error: {e}")

    def _update_plot(self, fs, start_idx):
        use_ch1     = self.use_ch1_var.get()
        max_display = self.max_display_var.get()
        n_pts       = self.displayed_ch0[-1].shape[0]
        time_us     = (np.arange(n_pts) + start_idx) / fs * 1e6

        self.ax0.clear()
        for trace in self.displayed_ch0[-max_display:]:
            self.ax0.plot(time_us, trace, color="steelblue", alpha=0.3, linewidth=0.8)
        self.ax0.axhline(self.trigger_ch0_var.get(), color="red", linestyle="--",
                         linewidth=1, label="Second trigger")
        self.ax0.set_ylabel("Amplitude (V)")
        self.ax0.set_title(
            f"Channel 0 — last {min(len(self.displayed_ch0), max_display)} traces")
        self.ax0.legend(fontsize=7)

        if use_ch1 and self.ax1 is not None and self.displayed_ch1:
            self.ax1.clear()
            for trace in self.displayed_ch1[-max_display:]:
                self.ax1.plot(time_us, trace, color="darkorange", alpha=0.3, linewidth=0.8)
            self.ax1.axhline(self.trigger_ch1_var.get(), color="red", linestyle="--",
                             linewidth=1, label="Second trigger")
            self.ax1.set_xlabel("Time after first trigger (µs)")
            self.ax1.set_ylabel("Amplitude (V)")
            self.ax1.set_title(
                f"Channel 1 — last {min(len(self.displayed_ch1), max_display)} traces")
            self.ax1.legend(fontsize=7)
        else:
            self.ax0.set_xlabel("Time after first trigger (µs)")

        self.fig.tight_layout()
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
        self.first_trigger_ch1_var = tk.DoubleVar(value=0.2)
        self.trigger_ch1_var       = tk.DoubleVar(value=0.01)
        self.pulses_ch1_var        = tk.IntVar(value=1)
        self.fs_var                = tk.DoubleVar(value=100e6)
        self.start_us_var          = tk.DoubleVar(value=0.5)
        self.stop_us_var           = tk.DoubleVar(value=40.0)
        self.bins_var              = tk.IntVar(value=100)
        self.filter_var            = tk.DoubleVar(value=100e6)
        self.holdoff_us_var        = tk.DoubleVar(value=0.5)
        self.save_traces_var       = tk.BooleanVar(value=False)
        self.counts_var            = tk.IntVar(value=0)
        self.triggers_var          = tk.IntVar(value=0)

        self.time_log_path: str | None  = None
        self.trace_csv_path: str | None = None

        self._build(parent)
        parent.after(500, self._update_loop)

    # ------------------------------------------------------------------ #

    def _build(self, parent):
        sidebar_container, frame, _ = make_scrollable_sidebar(parent)
        sidebar_container.pack(side=tk.LEFT, fill="y")

        def add_row(label, var):
            tk.Label(frame, text=label).pack(anchor="w")
            tk.Entry(frame, textvariable=var).pack(fill="x", pady=2)

        tk.Label(frame, text="── Channel 0 ──",
                 font=("", 9, "bold")).pack(anchor="w", pady=(4, 2))
        add_row("Ch0 first trigger level (V)",  self.first_trigger_ch0_var)
        add_row("Ch0 second trigger level (V)", self.trigger_ch0_var)
        add_row("Ch0 expected pulse count",     self.pulses_ch0_var)

        tk.Label(frame, text="── Channel 1 ──",
                 font=("", 9, "bold")).pack(anchor="w", pady=(8, 2))
        tk.Checkbutton(frame, text="Use Channel 1", variable=self.use_ch1_var,
                       command=self._toggle_ch1).pack(anchor="w")
        self.ch1_widgets = []

        def add_ch1_row(label, var):
            lbl = tk.Label(frame, text=label)
            lbl.pack(anchor="w")
            ent = tk.Entry(frame, textvariable=var)
            ent.pack(fill="x", pady=2)
            self.ch1_widgets += [lbl, ent]

        add_ch1_row("Ch1 first trigger level (V)",  self.first_trigger_ch1_var)
        add_ch1_row("Ch1 second trigger level (V)", self.trigger_ch1_var)
        add_ch1_row("Ch1 expected pulse count",     self.pulses_ch1_var)

        tk.Label(frame, text="── Shared ──",
                 font=("", 9, "bold")).pack(anchor="w", pady=(8, 2))
        add_row("Sampling frequency (Hz)",       self.fs_var)
        add_row("Low-pass cutoff (Hz)",          self.filter_var)
        add_row("Start time after trigger (µs)", self.start_us_var)
        add_row("Stop time after trigger (µs)",  self.stop_us_var)
        add_row("Histogram bins",                self.bins_var)
        add_row("Holdoff (µs)",                  self.holdoff_us_var)

        tk.Button(frame, text="Reset",      command=self._reset).pack(fill="x", pady=4)
        tk.Button(frame, text="Export CSV", command=self._export_csv).pack(fill="x", pady=2)

        # Peak-data log
        tk.Label(frame, text="── Data Saving ──",
                 font=("", 9, "bold")).pack(anchor="w", pady=(10, 2))

        tk.Button(frame, text="Set peak-data log path",
                  command=self._set_log_path).pack(fill="x", pady=2)
        self.log_label = tk.Label(frame, text="(no log file set)",
                                  wraplength=230, fg="gray",
                                  justify="left", font=("", 8))
        self.log_label.pack(anchor="w")

        tk.Checkbutton(frame, text="Save raw traces to CSV",
                       variable=self.save_traces_var,
                       command=self._toggle_save_traces).pack(anchor="w", pady=(6, 0))
        self.trace_path_label = tk.Label(frame, text="(traces not saved)",
                                         wraplength=230, fg="gray",
                                         justify="left", font=("", 8))
        self.trace_path_label.pack(anchor="w")

        # ADS panel — NOTE: histogram tab shares the same acq object but gets
        # its own AdsPanel UI; only the last "Start" pressed takes effect.
        self.ads_panel = AdsPanel(frame, self.acq, self.use_ch1_var)

        tk.Label(frame, text="Counts (events read):").pack(anchor="w", pady=(10, 0))
        tk.Label(frame, textvariable=self.counts_var).pack(anchor="w")
        tk.Label(frame, text="Triggers (passing events):").pack(anchor="w", pady=(6, 0))
        tk.Label(frame, textvariable=self.triggers_var).pack(anchor="w")

        self._toggle_ch1()
        self._build_plot(parent)

    def _build_plot(self, parent):
        self.fig = plt.Figure(figsize=(10, 7))
        gs = gridspec.GridSpec(2, 4, figure=self.fig,
                               height_ratios=[1.7, 1],
                               hspace=0.6, wspace=0.5)
        self.ax_dt    = self.fig.add_subplot(gs[0, :])
        self.ax_a1    = self.fig.add_subplot(gs[1, 0])
        self.ax_fwhm1 = self.fig.add_subplot(gs[1, 1])
        self.ax_a2    = self.fig.add_subplot(gs[1, 2])
        self.ax_fwhm2 = self.fig.add_subplot(gs[1, 3])
        self._label_axes()
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def _label_axes(self):
        self.ax_dt.set_xlabel("Time after first trigger (µs)")
        self.ax_dt.set_ylabel("Count")
        self.ax_dt.set_title("dt — time to second pulse")
        for ax, title, xlabel in [
            (self.ax_a1,    "Pulse 1 height", "Amplitude (V)"),
            (self.ax_fwhm1, "Pulse 1 FWHM",   "FWHM (µs)"),
            (self.ax_a2,    "Pulse 2 height", "Amplitude (V)"),
            (self.ax_fwhm2, "Pulse 2 FWHM",   "FWHM (µs)"),
        ]:
            ax.set_title(title, fontsize=8)
            ax.set_xlabel(xlabel, fontsize=7)
            ax.set_ylabel("Count", fontsize=7)
            ax.tick_params(labelsize=6)

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
            self.log_label.config(text=os.path.basename(path), fg="darkgreen")

    def _toggle_save_traces(self):
        if self.save_traces_var.get():
            path = filedialog.asksaveasfilename(
                title="Save traces CSV",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")]
            )
            if path:
                self.trace_csv_path = path
                self.trace_path_label.config(text=os.path.basename(path), fg="darkgreen")
            else:
                self.save_traces_var.set(False)
                self.trace_path_label.config(text="(traces not saved)", fg="gray")
        else:
            self.trace_csv_path = None
            self.trace_path_label.config(text="(traces not saved)", fg="gray")

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
                item = self.acq.queue.get_nowait()
                batch.append(item)
            except queue.Empty:
                break
        if not batch:
            return
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
        if do_filter:
            b, a_coef = sig.butter(4, fc / nyq, btype="low")

        new_records = []
        trace_rows  = []

        for item in batch:
            raw_ch0 = item["ch0"]
            self.total_counts += 1

            cross_ch0 = find_first_trigger_index(raw_ch0, first_trig_ch0)
            if cross_ch0 is None:
                continue
            chopped_ch0 = raw_ch0[cross_ch0:]
            if stop_idx > len(chopped_ch0):
                continue

            win1_ch0  = chopped_ch0[:start_idx]
            filt1_ch0 = (sig.filtfilt(b, a_coef, win1_ch0)
                         if (do_filter and len(win1_ch0) > 9)
                         else win1_ch0.copy())
            a1, fwhm1_us = measure_pulse(filt1_ch0, fs, half_width_only=True)

            win2_ch0  = chopped_ch0[start_idx:stop_idx]
            filt2_ch0 = (sig.filtfilt(b, a_coef, win2_ch0)
                         if do_filter else win2_ch0.copy())

            if count_pulses(filt2_ch0, trig_ch0, holdoff_samples) != pulses_ch0:
                continue

            if use_ch1 and item["ch1"] is not None:
                raw_ch1   = item["ch1"]
                cross_ch1 = find_first_trigger_index(raw_ch1, first_trig_ch1)
                if cross_ch1 is None:
                    continue
                chopped_ch1 = raw_ch1[cross_ch1:]
                if stop_idx > len(chopped_ch1):
                    continue
                win_ch1  = chopped_ch1[start_idx:stop_idx]
                filt_ch1 = (sig.filtfilt(b, a_coef, win_ch1)
                            if do_filter else win_ch1.copy())
                if count_pulses(filt_ch1, trig_ch1, holdoff_samples) != pulses_ch1:
                    continue

            above2  = filt2_ch0 >= trig_ch0
            rising2 = np.where(~above2[:-1] & above2[1:])[0] + 1
            if len(rising2) == 0:
                continue
            dt_us = (rising2[0] + start_idx) / fs * 1e6

            a2, fwhm2_us = measure_pulse(filt2_ch0, fs)

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
                trace_rows.append(chopped_ch0[:stop_idx].tolist())
                if use_ch1 and item["ch1"] is not None:
                    trace_rows.append(item["ch1"][cross_ch1:][:stop_idx].tolist())

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
            clean = data.dropna()
            if len(clean):
                ax.hist(clean, bins=bins, color=color, edgecolor="black", linewidth=0.5)
            ax.set_title(title, fontsize=8)
            ax.set_xlabel(xlabel, fontsize=7)
            ax.set_ylabel("Count", fontsize=7)
            ax.tick_params(labelsize=6)

        _hist(self.ax_dt,    df["dt_us"],     "dt — time to second pulse",
              "Time after first trigger (µs)", "steelblue")
        _hist(self.ax_a1,    df["a1"],        "Pulse 1 height", "Amplitude (V)", "mediumseagreen")
        _hist(self.ax_fwhm1, df["fwhm1_us"],  "Pulse 1 FWHM",   "FWHM (µs)",    "mediumseagreen")
        _hist(self.ax_a2,    df["a2"],        "Pulse 2 height", "Amplitude (V)", "tomato")
        _hist(self.ax_fwhm2, df["fwhm2_us"],  "Pulse 2 FWHM",   "FWHM (µs)",    "tomato")

        self.fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.1,
                                 hspace=0.65, wspace=0.5)
        self.canvas.draw()


# ===========================================================================
# Top-level application
# ===========================================================================

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Muon Detector")
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Shared acquisition manager
        self.acq = AcquisitionManager()

        # Notebook
        nb = ttk.Notebook(root)
        nb.pack(fill=tk.BOTH, expand=True)

        tab1 = tk.Frame(nb)
        tab2 = tk.Frame(nb)
        nb.add(tab1, text="Signal Viewer")
        nb.add(tab2, text="Live Histogram")

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