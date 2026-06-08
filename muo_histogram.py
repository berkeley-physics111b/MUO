import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import scipy.signal as sig
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import time
import datetime


class LiveHistogramGUI:
    def __init__(self, root):
        self.root = root
        root.title("Live Peak Histogram")

        self.filepath = None
        self.current_filepath = None

        self.last_processed_rows = 0
        self.total_counts = 0

        self.time_log_path = None

        # ---- Input variables ----
        self.use_ch1_var = tk.BooleanVar(value=False)

        # Ch0
        self.first_trigger_ch0_var = tk.DoubleVar(value=0.2)
        self.trigger_ch0_var       = tk.DoubleVar(value=0.01)
        self.pulses_ch0_var        = tk.IntVar(value=1)

        # Ch1
        self.first_trigger_ch1_var = tk.DoubleVar(value=0.2)
        self.trigger_ch1_var       = tk.DoubleVar(value=0.01)
        self.pulses_ch1_var        = tk.IntVar(value=1)

        # Shared
        self.fs_var         = tk.DoubleVar(value=100e6)
        self.start_us_var   = tk.DoubleVar(value=0.5)
        self.stop_us_var    = tk.DoubleVar(value=40.0)
        self.bins_var       = tk.IntVar(value=100)
        self.filter_var     = tk.DoubleVar(value=100e6)
        self.holdoff_us_var = tk.DoubleVar(value=0.5)

        # ---- File cycling ----
        self.auto_delete_var = tk.BooleanVar(value=True)

        # ---- Counters ----
        self.counts_var   = tk.IntVar(value=0)
        self.triggers_var = tk.IntVar(value=0)

        # Each record: dict with keys timestamp, dt_us, a1, fwhm1_us, a2, fwhm2_us
        self.records = []

        self.running = True
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._build_controls()
        self._build_plot()

        self.update_interval_ms = 500
        self.root.after(self.update_interval_ms, self.update_loop)

    # ------------------------------------------------------------------
    # File cycling: delete then wait for re-creation
    # ------------------------------------------------------------------

    def _delete_and_wait_for_file(self):
        try:
            os.remove(self.current_filepath)
        except Exception as e:
            print(f"Warning: could not delete file: {e}")

        poll_s    = 1.0
        timeout_s = 300
        deadline  = time.time() + timeout_s

        while time.time() < deadline:
            try:
                if os.path.exists(self.current_filepath):
                    self.last_processed_rows = 0
                    return
            except Exception:
                pass
            try:
                self.root.update()
            except Exception:
                return
            time.sleep(poll_s)

    # ------------------------------------------------------------------
    # Pulse measurement
    # ------------------------------------------------------------------

    def _measure_pulse(self, window, fs, half_width_only=False):
        """
        Return (height, fwhm_us) for the dominant peak in `window`.

        height          : max amplitude in the window
        fwhm_us         : full-width at half-maximum in microseconds.
        half_width_only : if True (first pulse case), the left half-max crossing
                          is unavailable because the window starts mid-rise.
                          FWHM is estimated as 2x the right-hand half-width instead.
        """
        if len(window) == 0:
            return np.nan, np.nan

        peak_idx = int(np.argmax(window))
        height   = float(window[peak_idx])
        half_max = height / 2.0

        # Falling half-max crossing (walk right from peak)
        right_idx = np.nan
        for i in range(peak_idx, len(window) - 1):
            if window[i] >= half_max >= window[i + 1]:
                frac      = (window[i] - half_max) / (window[i] - window[i + 1])
                right_idx = i + frac
                break

        if half_width_only:
            # Estimate FWHM as 2x the right-hand half-width
            if np.isnan(right_idx):
                fwhm_us = np.nan
            else:
                fwhm_us = 2.0 * (right_idx - peak_idx) / fs * 1e6
        else:
            # Rising half-max crossing (walk left from peak)
            left_idx = np.nan
            for i in range(peak_idx, 0, -1):
                if window[i - 1] <= half_max <= window[i]:
                    frac     = (half_max - window[i - 1]) / (window[i] - window[i - 1])
                    left_idx = (i - 1) + frac
                    break

            if np.isnan(left_idx) or np.isnan(right_idx):
                fwhm_us = np.nan
            else:
                fwhm_us = (right_idx - left_idx) / fs * 1e6

        return height, fwhm_us

    # ------------------------------------------------------------------
    # Data logging
    # ------------------------------------------------------------------

    def _save_records(self, new_records):
        """Append new event records to the log CSV."""
        if self.time_log_path is None or not new_records:
            return
        now  = datetime.datetime.now().isoformat(timespec="seconds")
        rows = [{**r, "timestamp": now} for r in new_records]
        df   = pd.DataFrame(rows, columns=["timestamp", "dt_us", "a1", "fwhm1_us", "a2", "fwhm2_us"])
        write_header = not os.path.exists(self.time_log_path)
        try:
            df.to_csv(self.time_log_path, mode="a", header=write_header, index=False)
        except Exception as e:
            print(f"Warning: could not write log: {e}")

    # ------------------------------------------------------------------
    # GUI construction
    # ------------------------------------------------------------------

    def on_close(self):
        self.running = False
        self.root.quit()
        self.root.destroy()

    def _build_controls(self):
        container = tk.Frame(self.root)
        container.pack(side=tk.LEFT, fill="y")

        canvas = tk.Canvas(container, width=260)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill="y")
        canvas.pack(side=tk.LEFT, fill="y", expand=True)

        frame = tk.Frame(canvas)
        frame_window = canvas.create_window((0, 0), window=frame, anchor="nw")

        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def on_canvas_configure(event):
            canvas.itemconfig(frame_window, width=event.width)

        frame.bind("<Configure>", on_frame_configure)
        canvas.bind("<Configure>", on_canvas_configure)

        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", on_mousewheel)
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll( 1, "units"))

        def add_row(label, var):
            tk.Label(frame, text=label).pack(anchor="w")
            tk.Entry(frame, textvariable=var).pack(fill="x", pady=2)

        # ---- Ch0 ----
        tk.Label(frame, text="── Channel 0 ──", font=("", 9, "bold")).pack(anchor="w", pady=(4, 2))
        add_row("Ch0 first trigger level (V)",  self.first_trigger_ch0_var)
        add_row("Ch0 second trigger level (V)", self.trigger_ch0_var)
        add_row("Ch0 expected pulse count",     self.pulses_ch0_var)

        # ---- Ch1 ----
        tk.Label(frame, text="── Channel 1 ──", font=("", 9, "bold")).pack(anchor="w", pady=(8, 2))
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

        # ---- Shared ----
        tk.Label(frame, text="── Shared ──", font=("", 9, "bold")).pack(anchor="w", pady=(8, 2))
        add_row("Sampling frequency (Hz)",       self.fs_var)
        add_row("Lower cutoff frequency (Hz)",   self.filter_var)
        add_row("Start time after trigger (µs)", self.start_us_var)
        add_row("Stop time after trigger (µs)",  self.stop_us_var)
        add_row("Histogram bins",                self.bins_var)
        add_row("Holdoff (µs)",                  self.holdoff_us_var)

        tk.Button(frame, text="Select Data File", command=self.select_file).pack(fill="x", pady=8)
        tk.Button(frame, text="Reset",            command=self.reset_histogram).pack(fill="x", pady=4)
        tk.Button(frame, text="Export CSV",       command=self.export_csv).pack(fill="x", pady=4)

        # ---- Auto-delete toggle ----
        tk.Label(frame, text="── Data File ──", font=("", 9, "bold")).pack(anchor="w", pady=(10, 2))
        tk.Checkbutton(frame, text="Auto-delete after 500 rows",
                       variable=self.auto_delete_var,
                       command=self._toggle_auto_delete).pack(anchor="w")
        self.auto_delete_warning = tk.Label(
            frame,
            text="⚠ Only use for short runs.\nDisabling auto-delete can result\nin many MB/GB of stored data.",
            fg="red",
            wraplength=220,
            justify="left",
            font=("", 8),
        )
        # warning hidden by default (checkbox starts checked)

        tk.Label(frame, text="Counts (events read):").pack(anchor="w", pady=(10, 0))
        tk.Label(frame, textvariable=self.counts_var).pack(anchor="w")
        tk.Label(frame, text="Triggers (passing events):").pack(anchor="w", pady=(6, 0))
        tk.Label(frame, textvariable=self.triggers_var).pack(anchor="w")

        self.file_label = tk.Label(frame, text="No file selected", wraplength=220)
        self.file_label.pack(pady=10)

        self._toggle_ch1()

    def _toggle_ch1(self):
        state = "normal" if self.use_ch1_var.get() else "disabled"
        for w in self.ch1_widgets:
            w.config(state=state)

    def _toggle_auto_delete(self):
        if self.auto_delete_var.get():
            self.auto_delete_warning.pack_forget()
        else:
            self.auto_delete_warning.pack(anchor="w", pady=(2, 0))

    def _build_plot(self):
        self.fig = plt.figure(figsize=(10, 7))

        # dt spans the full top row; four small plots share the bottom row
        gs = gridspec.GridSpec(
            2, 4,
            figure=self.fig,
            height_ratios=[1.7, 1],
            hspace=0.6,
            wspace=0.5,
        )

        self.ax_dt    = self.fig.add_subplot(gs[0, :])   # full-width top
        self.ax_a1    = self.fig.add_subplot(gs[1, 0])
        self.ax_fwhm1 = self.fig.add_subplot(gs[1, 1])
        self.ax_a2    = self.fig.add_subplot(gs[1, 2])
        self.ax_fwhm2 = self.fig.add_subplot(gs[1, 3])

        self._label_axes()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
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

    # ------------------------------------------------------------------
    # File / histogram management
    # ------------------------------------------------------------------

    def select_file(self):
        self.reset_histogram()
        path = filedialog.askopenfilename()
        if not path:
            return

        self.filepath         = path
        self.current_filepath = path

        base_dir  = os.path.dirname(path)
        base_name = os.path.splitext(os.path.basename(path))[0]
        self.time_log_path = os.path.join(base_dir, f"peak_data_{base_name}.csv")

        self.file_label.config(text=os.path.basename(path))

    def reset_histogram(self):
        self.records.clear()
        self.last_processed_rows = 0
        self.total_counts        = 0
        self.counts_var.set(0)
        self.triggers_var.set(0)
        self.filepath         = None
        self.current_filepath = None
        self.time_log_path    = None

        if hasattr(self, 'file_label'):
            self.file_label.config(text="No file selected")

        if hasattr(self, 'ax_dt'):
            for ax in (self.ax_dt, self.ax_a1, self.ax_fwhm1, self.ax_a2, self.ax_fwhm2):
                ax.clear()
            self._label_axes()
            self.canvas.draw()

    def export_csv(self):
        if not self.records:
            messagebox.showwarning("No Data", "No records to export.")
            return
        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV files", "*.csv")]
        )
        if save_path:
            df = pd.DataFrame(self.records,
                              columns=["timestamp", "dt_us", "a1", "fwhm1_us", "a2", "fwhm2_us"])
            df.to_csv(save_path, index=False)

    # ------------------------------------------------------------------
    # Main update loop
    # ------------------------------------------------------------------

    FILE_STRIDE = 500

    def update_loop(self):
        if not self.running:
            return

        if self.filepath and self.current_filepath:
            if self.auto_delete_var.get() and self.last_processed_rows >= self.FILE_STRIDE:
                self._delete_and_wait_for_file()

            if os.path.exists(self.current_filepath):
                self.process_file()

        self.root.after(self.update_interval_ms, self.update_loop)

    # ------------------------------------------------------------------
    # Signal utilities
    # ------------------------------------------------------------------

    def _find_first_trigger_index(self, row, level):
        above  = row >= level
        rising = np.where(~above[:-1] & above[1:])[0] + 1
        return int(rising[0]) if len(rising) > 0 else None

    def _count_pulses(self, window, trigger_level, holdoff_samples):
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

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def process_file(self):
        use_ch1 = self.use_ch1_var.get()

        try:
            if use_ch1:
                if self.last_processed_rows == 0:
                    ch0_data = pd.read_csv(self.current_filepath, header=None,
                                           skiprows=lambda i: i % 2 != 0)
                    ch1_data = pd.read_csv(self.current_filepath, header=None,
                                           skiprows=lambda i: i % 2 != 1)
                else:
                    already  = self.last_processed_rows * 2
                    ch0_data = pd.read_csv(self.current_filepath, header=None,
                                           skiprows=lambda i: i % 2 != 0 or i < already)
                    ch1_data = pd.read_csv(self.current_filepath, header=None,
                                           skiprows=lambda i: i % 2 != 1 or i < already)
            else:
                if self.last_processed_rows == 0:
                    ch0_data = pd.read_csv(self.current_filepath, header=None)
                else:
                    already  = self.last_processed_rows
                    ch0_data = pd.read_csv(self.current_filepath, header=None,
                                           skiprows=lambda i: i < already)
                ch1_data = None
        except Exception:
            return

        if ch0_data.empty:
            return

        ch0_traces = ch0_data.to_numpy(dtype=float)
        if ch0_traces.ndim == 1:
            ch0_traces = ch0_traces[np.newaxis, :]
        ch0_traces = ch0_traces[~np.isnan(ch0_traces).any(axis=1)]

        if use_ch1:
            if ch1_data.empty:
                return
            ch1_traces = ch1_data.to_numpy(dtype=float)
            if ch1_traces.ndim == 1:
                ch1_traces = ch1_traces[np.newaxis, :]
            ch1_traces = ch1_traces[~np.isnan(ch1_traces).any(axis=1)]
            n_events   = min(len(ch0_traces), len(ch1_traces))
            ch0_traces = ch0_traces[:n_events]
            ch1_traces = ch1_traces[:n_events]
        else:
            n_events = len(ch0_traces)

        if n_events == 0:
            return

        self.last_processed_rows += n_events
        self.total_counts        += n_events
        self.counts_var.set(self.total_counts)

        # ---- Parameters ----
        fs              = self.fs_var.get()
        fc              = self.filter_var.get()
        start_idx       = int(self.start_us_var.get() * 1e-6 * fs)
        stop_idx        = int(self.stop_us_var.get()  * 1e-6 * fs)
        holdoff_samples = int(self.holdoff_us_var.get() * 1e-6 * fs)

        first_trig_ch0 = self.first_trigger_ch0_var.get()
        trig_ch0       = self.trigger_ch0_var.get()
        pulses_ch0     = self.pulses_ch0_var.get()

        if use_ch1:
            first_trig_ch1 = self.first_trigger_ch1_var.get()
            trig_ch1       = self.trigger_ch1_var.get()
            pulses_ch1     = self.pulses_ch1_var.get()

        nyq       = fs / 2.0
        do_filter = 0 < fc < nyq
        if do_filter:
            b, a = sig.butter(4, fc / nyq, btype='low')

        new_records = []

        for idx in range(n_events):
            raw_ch0   = ch0_traces[idx]
            cross_ch0 = self._find_first_trigger_index(raw_ch0, first_trig_ch0)
            if cross_ch0 is None:
                continue

            chopped_ch0 = raw_ch0[cross_ch0:]
            if stop_idx > len(chopped_ch0):
                continue

            # ---- First pulse: region from first trigger up to start_idx ----
            win1_ch0  = chopped_ch0[:start_idx]
            filt1_ch0 = (sig.filtfilt(b, a, win1_ch0)
                         if (do_filter and len(win1_ch0) > 9)
                         else win1_ch0.copy())
            a1, fwhm1_us = self._measure_pulse(filt1_ch0, fs, half_width_only=True)

            # ---- Second pulse: normal analysis window ----
            win2_ch0  = chopped_ch0[start_idx:stop_idx]
            filt2_ch0 = sig.filtfilt(b, a, win2_ch0) if do_filter else win2_ch0.copy()

            if self._count_pulses(filt2_ch0, trig_ch0, holdoff_samples) != pulses_ch0:
                continue

            # ---- Ch1 gate (if enabled) ----
            if use_ch1:
                raw_ch1   = ch1_traces[idx]
                cross_ch1 = self._find_first_trigger_index(raw_ch1, first_trig_ch1)
                if cross_ch1 is None:
                    continue

                chopped_ch1 = raw_ch1[cross_ch1:]
                if stop_idx > len(chopped_ch1):
                    continue

                win_ch1  = chopped_ch1[start_idx:stop_idx]
                filt_ch1 = sig.filtfilt(b, a, win_ch1) if do_filter else win_ch1.copy()

                if self._count_pulses(filt_ch1, trig_ch1, holdoff_samples) != pulses_ch1:
                    continue

            # ---- dt: time of second pulse rising edge from first trigger ----
            above2  = filt2_ch0 >= trig_ch0
            rising2 = np.where(~above2[:-1] & above2[1:])[0] + 1
            if len(rising2) == 0:
                continue
            dt_us = (rising2[0] + start_idx) / fs * 1e6

            # ---- Second pulse height and FWHM ----
            a2, fwhm2_us = self._measure_pulse(filt2_ch0, fs)

            new_records.append({
                "timestamp": None,    # filled in _save_records
                "dt_us":     dt_us,
                "a1":        a1,
                "fwhm1_us":  fwhm1_us,
                "a2":        a2,
                "fwhm2_us":  fwhm2_us,
            })

        if not new_records:
            return

        self.records.extend(new_records)
        self.triggers_var.set(len(self.records))

        self._save_records(new_records)
        self.update_histograms()

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def update_histograms(self):
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

        # Large top: dt
        _hist(self.ax_dt, df["dt_us"],
              "dt — time to second pulse", "Time after first trigger (µs)", "steelblue")

        # Small 2×2: pulse 1 (green), pulse 2 (red/orange)
        _hist(self.ax_a1,    df["a1"],       "Pulse 1 height", "Amplitude (V)", "mediumseagreen")
        _hist(self.ax_fwhm1, df["fwhm1_us"], "Pulse 1 FWHM",   "FWHM (µs)",     "mediumseagreen")
        _hist(self.ax_a2,    df["a2"],       "Pulse 2 height", "Amplitude (V)", "tomato")
        _hist(self.ax_fwhm2, df["fwhm2_us"], "Pulse 2 FWHM",   "FWHM (µs)",     "tomato")

        self.fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.1,
                                  hspace=0.65, wspace=0.5)
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app  = LiveHistogramGUI(root)
    root.mainloop()