from tkinter import Frame, Label, Button, Toplevel
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class ResultsMixin:
    def open_results_window(self):
        if self.results_win is not None and self.results_win.winfo_exists():
            self.results_win.lift()
            return

        self.results_win = Toplevel(self.window)
        self.results_win.title(self.tr("stats_title"))
        self.results_win.geometry("620x450")
        self.results_win.protocol("WM_DELETE_WINDOW", self.close_results_window)

        top = Frame(self.results_win)
        top.pack(fill="x", padx=8, pady=6)

        Button(top, text=self.tr("plot_hist"), command=self.open_hist_window).pack(side="left")
        Button(top, text=self.tr("refresh"), command=self.update_results_window).pack(side="left", padx=6)

        self.results_info = Label(self.results_win, text="", anchor="w", justify="left")
        self.results_info.pack(fill="x", padx=8)

        table_frame = Frame(self.results_win)
        table_frame.pack(fill="both", expand=True, padx=8, pady=8)

        cols = ("type", "cls", "count", "avg")
        self.results_tree = ttk.Treeview(table_frame, columns=cols, show="headings")

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=vsb.set)

        self.results_tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        self.update_results_window()
        self.schedule_results_update()

    def close_results_window(self, silent=False):
        if self.results_update_job is not None and self.results_win is not None:
            try:
                self.results_win.after_cancel(self.results_update_job)
            except Exception:
                pass
            self.results_update_job = None

        if self.results_win is not None and self.results_win.winfo_exists():
            self.results_win.destroy()

        self.results_win = None
        self.results_tree = None
        self.results_info = None

    def schedule_results_update(self):
        if self.results_win is None or not self.results_win.winfo_exists():
            return
        self.results_update_job = self.results_win.after(250, self._tick_results_update)

    def _tick_results_update(self):
        if self.results_win is None or not self.results_win.winfo_exists():
            return
        try:
            self.update_results_window()
        finally:
            self.schedule_results_update()

    def _normalize_runtime(self):
        runtime = self.copy_runtime_stats()
        if isinstance(runtime, dict):
            return runtime
        fc, current_fps, avg_fps = runtime
        return {
            "frame_count": fc,
            "current_fps": current_fps,
            "avg_fps": avg_fps,
            "inference_fps": current_fps,
            "detection_fps": avg_fps,
            "display_fps": 0.0,
            "avg_infer_ms": 0.0,
        }

    def update_results_window(self):
        if self.results_win is None or not self.results_win.winfo_exists() or self.results_tree is None:
            return

        self.results_tree.heading("type", text=self.tr("model_type"))
        self.results_tree.heading("cls", text=self.tr("class"))
        self.results_tree.heading("count", text=self.tr("count"))
        self.results_tree.heading("avg", text=self.tr("avg_conf"))

        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        stats, _ = self.copy_stats()
        runtime = self._normalize_runtime()
        fc = int(runtime.get("frame_count", 0))

        total_det = sum(v["count"] for v in stats.values())
        if self.results_info:
            if self.lang == "pl":
                info_text = (
                    f"Klatka: {fc} | Wykrycia: {total_det}\n"
                    f"{self.tr('fps_inference')}: {runtime.get('inference_fps', 0.0):.1f} | "
                    f"{self.tr('fps_detection')}: {runtime.get('detection_fps', 0.0):.1f} | "
                    f"{self.tr('fps_display')}: {runtime.get('display_fps', 0.0):.1f} | "
                    f"{self.tr('infer_time_avg')}: {runtime.get('avg_infer_ms', 0.0):.1f} ms"
                )
            else:
                info_text = (
                    f"Frame: {fc} | Detections: {total_det}\n"
                    f"{self.tr('fps_inference')}: {runtime.get('inference_fps', 0.0):.1f} | "
                    f"{self.tr('fps_detection')}: {runtime.get('detection_fps', 0.0):.1f} | "
                    f"{self.tr('fps_display')}: {runtime.get('display_fps', 0.0):.1f} | "
                    f"{self.tr('infer_time_avg')}: {runtime.get('avg_infer_ms', 0.0):.1f} ms"
                )
            self.results_info.config(text=info_text)

        if not stats:
            return

        rows = []
        for key, st in stats.items():
            model_type, class_name = key.split(":", 1)
            cnt = st["count"]
            avg = (st["conf_sum"] / cnt) if cnt else 0.0
            rows.append((model_type, class_name, cnt, avg))
        rows.sort(key=lambda x: x[2], reverse=True)

        for model_type, class_name, cnt, avg in rows:
            model_disp = self.tr_model_type(model_type)
            class_disp = self.translate_class_name(class_name)
            self.results_tree.insert("", "end", values=(model_disp, class_disp, cnt, f"{avg:.2f}"))

    def open_hist_window(self):
        if self.hist_win is not None and self.hist_win.winfo_exists():
            self.hist_win.lift()
            return

        self.hist_win = Toplevel(self.window)
        self.hist_win.title(self.tr("hist_title"))
        self.hist_win.geometry("900x520")
        self.hist_win.protocol("WM_DELETE_WINDOW", self.close_hist_window)

        self.hist_fig = Figure(figsize=(9, 4.8), dpi=100)
        self.hist_ax = self.hist_fig.add_subplot(111)

        self.hist_canvas = FigureCanvasTkAgg(self.hist_fig, master=self.hist_win)
        self.hist_canvas.get_tk_widget().pack(fill="both", expand=True)

        self.update_histogram()
        self.schedule_hist_update()

    def close_hist_window(self, silent=False):
        if self.hist_update_job is not None and self.hist_win is not None:
            try:
                self.hist_win.after_cancel(self.hist_update_job)
            except Exception:
                pass
            self.hist_update_job = None

        if self.hist_win is not None and self.hist_win.winfo_exists():
            self.hist_win.destroy()

        self.hist_win = None
        self.hist_canvas = None
        self.hist_fig = None
        self.hist_ax = None

    def schedule_hist_update(self):
        if self.hist_win is None or not self.hist_win.winfo_exists():
            return
        self.hist_update_job = self.hist_win.after(500, self._tick_hist_update)

    def _tick_hist_update(self):
        if self.hist_win is None or not self.hist_win.winfo_exists():
            return
        self.update_histogram()
        self.schedule_hist_update()

    def update_histogram(self):
        if self.hist_win is None or not self.hist_win.winfo_exists() or self.hist_ax is None:
            return

        self.hist_ax.clear()
        stats, _ = self.copy_stats()

        if not stats:
            self.hist_ax.set_title(self.tr("hist_title"))
            self.hist_ax.text(0.5, 0.5, self.tr("no_stats"), ha="center", va="center")
            self.hist_canvas.draw()
            return

        items = []
        for key, st in stats.items():
            model_type, class_name = key.split(":", 1)
            label = f"{self.tr_model_type(model_type)} / {self.translate_class_name(class_name)}"
            items.append((label, st["count"]))
        items.sort(key=lambda x: x[1], reverse=True)

        labels = [x[0] for x in items]
        counts = [x[1] for x in items]
        x = range(len(labels))

        self.hist_ax.bar(x, counts)
        self.hist_ax.set_title(self.tr("hist_title"))
        self.hist_ax.set_ylabel(self.tr("count"))
        self.hist_ax.set_xticks(list(x))
        self.hist_ax.set_xticklabels(labels, rotation=35, ha="right")

        self.hist_fig.tight_layout()
        self.hist_canvas.draw()
