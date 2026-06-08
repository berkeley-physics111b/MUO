import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import scipy.signal as sig
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import time

class ViewSignalsGUI:
    def __init__(self, root):
        self.root = root
        root.title("View Signals")

        self.filepath = None
        self.current_filepath = None
        self.last_processed_rows = 0
        self.total_counts = 0
        self.displayed_ch0 = []
        self.displayed_ch1 = []

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
        self.fs_var          = tk.DoubleVar(value=100e6)
        self.start_us_var    = tk.DoubleVar(value=0.5)
        self.stop_us_var     = tk.DoubleVar(value=40.0)
        self.filter_var      = tk.DoubleVar(value=100e6)
        self.holdoff_us_var  = tk.DoubleVar(value=0.5)
        self.max_display_var = tk.IntVar(value=50)

        # ---- File cycling ----
        self.auto_delete_var = tk.BooleanVar(value=True)

        # ---- Counters ----
        self.counts_var  = tk.IntVar(value=0)
        self.passing_var = tk.IntVar(value=0)

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
        add_row("Holdoff (µs)",                  self.holdoff_us_var)
        add_row("Max traces to display",         self.max_display_var)

        tk.Button(frame, text="Select Data File", command=self.select_file).pack(fill="x", pady=8)
        tk.Button(frame, text="Reset",            command=self.reset).pack(fill="x", pady=4)

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
        tk.Label(frame, text="Passing events:").pack(anchor="w", pady=(6, 0))
        tk.Label(frame, textvariable=self.passing_var).pack(anchor="w")

        self.file_label = tk.Label(frame, text="No file selected", wraplength=220)
        self.file_label.pack(pady=10)

        self._toggle_ch1()

    def _toggle_ch1(self):
        state = "normal" if self.use_ch1_var.get() else "disabled"
        for w in self.ch1_widgets:
            w.config(state=state)
        if hasattr(self, 'fig'):
            self._rebuild_plot()

    def _toggle_auto_delete(self):
        if self.auto_delete_var.get():
            self.auto_delete_warning.pack_forget()
        else:
            self.auto_delete_warning.pack(anchor="w", pady=(2, 0))

    def _rebuild_plot(self):
        self.fig.clf()
        use_ch1 = self.use_ch1_var.get()
        if use_ch1:
            self.ax0, self.ax1 = self.fig.subplots(2, 1, sharex=True)
            self.ax1.set_xlabel("Time after first trigger (µs)")
            self.ax1.set_ylabel("Amplitude")
            self.ax1.set_title("Channel 1")
        else:
            self.ax0 = self.fig.subplots(1, 1)
            self.ax1 = None
            self.ax0.set_xlabel("Time after first trigger (µs)")

        self.ax0.set_ylabel("Amplitude")
        self.ax0.set_title("Channel 0")
        self.fig.tight_layout()
        self.canvas.draw()

    def _build_plot(self):
        self.fig = plt.figure(figsize=(7, 6))
        self.ax0 = self.fig.subplots(1, 1)
        self.ax1 = None
        self.ax0.set_xlabel("Time after first trigger (µs)")
        self.ax0.set_ylabel("Amplitude")
        self.ax0.set_title("Channel 0")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def select_file(self):
        self.reset()
        path = filedialog.askopenfilename()
        if not path:
            return
        self.filepath         = path
        self.current_filepath = path
        self.file_label.config(text=os.path.basename(path))

    def reset(self):
        self.displayed_ch0.clear()
        self.displayed_ch1.clear()
        self.last_processed_rows = 0
        self.total_counts        = 0
        self.counts_var.set(0)
        self.passing_var.set(0)
        self.filepath         = None
        self.current_filepath = None
        if hasattr(self, 'file_label'):
            self.file_label.config(text="No file selected")
        self._rebuild_plot()

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

    def _find_first_trigger_index(self, row, level):
        above = row >= level
        rising = np.where(~above[:-1] & above[1:])[0] + 1
        return int(rising[0]) if len(rising) > 0 else None

    def _count_pulses(self, window, trigger_level, holdoff_samples):
        above = window >= trigger_level
        rising = np.where(~above[:-1] & above[1:])[0] + 1
        if len(rising) == 0:
            return 0
        count = 1
        last = rising[0]
        for e in rising[1:]:
            if e - last >= holdoff_samples:
                count += 1
                last = e
        return count

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
                    already = self.last_processed_rows * 2
                    ch0_data = pd.read_csv(self.current_filepath, header=None,
                                           skiprows=lambda i: i % 2 != 0 or i < already)
                    ch1_data = pd.read_csv(self.current_filepath, header=None,
                                           skiprows=lambda i: i % 2 != 1 or i < already)
            else:
                if self.last_processed_rows == 0:
                    ch0_data = pd.read_csv(self.current_filepath, header=None)
                else:
                    already = self.last_processed_rows
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
            n_events = min(len(ch0_traces), len(ch1_traces))
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

        nyq = fs / 2.0
        do_filter = 0 < fc < nyq
        if do_filter:
            b, a = sig.butter(4, fc / nyq, btype='low')

        new_ch0 = []
        new_ch1 = []

        for idx in range(n_events):
            # ---- Ch0 ----
            raw_ch0   = ch0_traces[idx]
            cross_ch0 = self._find_first_trigger_index(raw_ch0, first_trig_ch0)
            if cross_ch0 is None:
                continue

            chopped_ch0 = raw_ch0[cross_ch0:]
            if stop_idx > len(chopped_ch0):
                continue

            win_ch0  = chopped_ch0[start_idx:stop_idx]
            filt_ch0 = sig.filtfilt(b, a, win_ch0) if do_filter else win_ch0.copy()

            if self._count_pulses(filt_ch0, trig_ch0, holdoff_samples) != pulses_ch0:
                continue

            # ---- Ch1 (if enabled) ----
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

                new_ch1.append(win_ch1)

            new_ch0.append(win_ch0)

        if not new_ch0:
            return

        self.displayed_ch0.extend(new_ch0)
        if use_ch1:
            self.displayed_ch1.extend(new_ch1)

        self.passing_var.set(len(self.displayed_ch0))
        self.update_plot(fs, start_idx)

    def update_plot(self, fs, start_idx):
        use_ch1 = self.use_ch1_var.get()
        max_display = self.max_display_var.get()

        n_pts   = self.displayed_ch0[-1].shape[0]
        time_us = (np.arange(n_pts) + start_idx) / fs * 1e6

        # ---- Ch0 ----
        self.ax0.clear()
        for trace in self.displayed_ch0[-max_display:]:
            self.ax0.plot(time_us, trace, color='steelblue', alpha=0.3, linewidth=0.8)
        self.ax0.axhline(self.trigger_ch0_var.get(), color='red', linestyle='--',
                         linewidth=1, label='Second trigger')
        self.ax0.set_ylabel("Amplitude")
        self.ax0.set_title(f"Channel 0 — last {min(len(self.displayed_ch0), max_display)} traces")
        self.ax0.legend(fontsize=7)

        # ---- Ch1 ----
        if use_ch1 and self.ax1 is not None and self.displayed_ch1:
            self.ax1.clear()
            for trace in self.displayed_ch1[-max_display:]:
                self.ax1.plot(time_us, trace, color='darkorange', alpha=0.3, linewidth=0.8)
            self.ax1.axhline(self.trigger_ch1_var.get(), color='red', linestyle='--',
                             linewidth=1, label='Second trigger')
            self.ax1.set_xlabel("Time after first trigger (µs)")
            self.ax1.set_ylabel("Amplitude")
            self.ax1.set_title(f"Channel 1 — last {min(len(self.displayed_ch1), max_display)} traces")
            self.ax1.legend(fontsize=7)
        elif not use_ch1:
            self.ax0.set_xlabel("Time after first trigger (µs)")

        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = ViewSignalsGUI(root)
    root.mainloop()