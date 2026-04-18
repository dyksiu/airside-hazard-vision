import queue
import threading
import time
import cv2
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox


class VideoMixin:

    def reset_fps_stats(self):
        with self.stats_lock:
            self.frame_count = 0
            self.current_fps = 0.0
            self.avg_fps = 0.0
            self.input_fps = 0.0
            self.inference_fps = 0.0
            self.detection_fps = 0.0
            self.display_fps = 0.0
            self.avg_infer_ms = 0.0

    def copy_runtime_stats(self):
        with self.stats_lock:
            return {
                "frame_count": self.frame_count,
                "current_fps": self.current_fps,
                "avg_fps": self.avg_fps,
                "input_fps": self.input_fps,
                "inference_fps": self.inference_fps,
                "detection_fps": self.detection_fps,
                "display_fps": self.display_fps,
                "avg_infer_ms": self.avg_infer_ms,
            }

    def draw_fps_overlay(self, frame, runtime_stats):
        lines = [
            f"{self.tr('fps_inference')}: {runtime_stats['inference_fps']:.1f}",
            f"{self.tr('fps_detection')}: {runtime_stats['detection_fps']:.1f}",
            f"{self.tr('fps_display')}: {runtime_stats['display_fps']:.1f}",
            f"{self.tr('infer_time_avg')}: {runtime_stats['avg_infer_ms']:.1f} ms",
        ]

        x = 12
        y = 28
        line_h = 26
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        max_w = 0
        for line in lines:
            (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
            max_w = max(max_w, tw)

        box_w = max_w + 20
        box_h = line_h * len(lines) + 12
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 8, y - 22), (x - 8 + box_w, y - 22 + box_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

        for idx, line in enumerate(lines):
            yy = y + idx * line_h
            cv2.putText(frame, line, (x, yy), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(frame, line, (x, yy), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return frame

    def select_video(self):
        if (not self.surface_ready()) and (not self.line_ready()):
            messagebox.showwarning(self.tr("no_model_title"), self.tr("no_model_msg"))
            return

        video_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
        if not video_path:
            return

        self.clear_video(silent=True)
        self.reset_fps_stats()

        self.conf_value = self.conf_slider.get() / 100
        self.process_stride = 2 if (self.surface_ready() and self.line_ready()) else 1

        self.worker_done = False
        self.worker_error = None
        self.stop_event.clear()
        self.running = True

        self.paused = False
        self.pause_event.clear()
        self.pause_button.config(state="normal", text=self.tr("pause"))

        self.video_panel.config(text="")

        self.worker_thread = threading.Thread(
            target=self.worker_loop,
            args=(video_path,),
            daemon=True
        )
        self.worker_thread.start()
        self.schedule_display_loop()

    def clear_video(self, silent=False):
        self.stop_processing(silent=True)

        with self.stats_lock:
            self.class_confidences = {}
            self.detections = []
            self.class_stats = {}

        self.reset_fps_stats()

        self.reset_tracking()
        self.reset_event_log_state()
        self.last_surface_roi = None
        self.display_surface_roi = None
        self.worker_done = False
        self.worker_error = None

        try:
            while True:
                self.frame_queue.get_nowait()
        except queue.Empty:
            pass

        self._tkimg = None
        self.video_panel.config(image="", text=self.tr("video_placeholder"))

        if self.results_win is not None and self.results_win.winfo_exists():
            self.update_results_window()

        if self.hist_win is not None and self.hist_win.winfo_exists():
            self.update_histogram()

    def schedule_display_loop(self):
        if self.display_job is not None:
            try:
                self.window.after_cancel(self.display_job)
            except Exception:
                pass
            self.display_job = None
        self.display_loop()

    def display_loop(self):
        if not self.running:
            return

        try:
            frame = self.frame_queue.get_nowait()
            self.show_frame_in_tk(frame)
        except queue.Empty:
            pass

        self.drain_event_log_queue()

        if self.worker_done:
            self.drain_event_log_queue()
            self.running = False
            self.pause_button.config(state="disabled", text=self.tr("pause"))
            if self.worker_error:
                messagebox.showerror("Blad", str(self.worker_error))
            else:
                messagebox.showinfo(self.tr("done"), self.tr("done_msg"))
            return

        self.display_job = self.window.after(self.display_delay_ms, self.display_loop)

    def worker_loop(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.worker_error = "Nie udalo sie otworzyc wideo."
                self.worker_done = True
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 30.0
            frame_time = 1.0 / fps

            started_at = time.perf_counter()
            last_annotated = None
            frame_idx = 0
            display_count = 0
            infer_count = 0
            infer_time_sum = 0.0

            with self.stats_lock:
                self.input_fps = float(fps)

            while not self.stop_event.is_set():
                while self.pause_event.is_set() and not self.stop_event.is_set():
                    time.sleep(0.05)

                loop_started_at = time.perf_counter()

                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (self.TARGET_WIDTH, self.TARGET_HEIGHT))
                do_infer = (frame_idx % self.process_stride == 0)

                if do_infer:
                    infer_started_this = time.perf_counter()
                    frame_vis = self.run_detection_on_frame(frame, frame_idx)
                    infer_finished_this = time.perf_counter()

                    last_annotated = frame_vis
                    infer_count += 1
                    infer_time_sum += max(0.0, infer_finished_this - infer_started_this)
                else:
                    frame_vis = last_annotated.copy() if last_annotated is not None else frame

                work_dt = time.perf_counter() - loop_started_at
                sleep = frame_time - work_dt
                if sleep > 0:
                    time.sleep(sleep)

                display_count += 1
                elapsed_wall = max(time.perf_counter() - started_at, 1e-6)
                runtime_stats = {
                    "frame_count": frame_idx,
                    "current_fps": infer_count / max(infer_time_sum, 1e-6),
                    "avg_fps": infer_count / elapsed_wall,
                    "input_fps": float(fps),
                    "inference_fps": infer_count / max(infer_time_sum, 1e-6),
                    "detection_fps": infer_count / elapsed_wall,
                    "display_fps": display_count / elapsed_wall,
                    "avg_infer_ms": (infer_time_sum / infer_count * 1000.0) if infer_count else 0.0,
                }

                with self.stats_lock:
                    self.frame_count = runtime_stats["frame_count"]
                    self.current_fps = runtime_stats["current_fps"]
                    self.avg_fps = runtime_stats["avg_fps"]
                    self.input_fps = runtime_stats["input_fps"]
                    self.inference_fps = runtime_stats["inference_fps"]
                    self.detection_fps = runtime_stats["detection_fps"]
                    self.display_fps = runtime_stats["display_fps"]
                    self.avg_infer_ms = runtime_stats["avg_infer_ms"]

                frame_vis = self.draw_fps_overlay(frame_vis, runtime_stats)

                try:
                    while True:
                        _ = self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                self.frame_queue.put(frame_vis)

                frame_idx += 1

            cap.release()
            self.worker_done = True

        except Exception as e:
            self.worker_error = e
            self.worker_done = True

    def stop_processing(self, silent=False):
        self.running = False
        self.stop_event.set()

        if self.display_job is not None:
            try:
                self.window.after_cancel(self.display_job)
            except Exception:
                pass
            self.display_job = None

        if self.worker_thread is not None and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        self.worker_thread = None

        if not silent:
            self._tkimg = None
            self.video_panel.config(image="", text=self.tr("video_stopped"))

        self.pause_event.clear()
        self.paused = False
        self.reset_tracking()

        if hasattr(self, "pause_button"):
            self.pause_button.config(state="disabled", text=self.tr("pause"))

    def toggle_pause(self):
        if not self.running:
            return
        self.paused = not self.paused
        if self.paused:
            self.pause_event.set()
            self.pause_button.config(text=self.tr("resume"))
        else:
            self.pause_event.clear()
            self.pause_button.config(text=self.tr("pause"))

    def on_close(self):
        self.stop_processing(silent=True)
        self.close_results_window(silent=True)
        self.close_hist_window(silent=True)
        self.window.destroy()

    def show_frame_in_tk(self, bgr_frame):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        panel_w = self.video_panel.winfo_width()
        panel_h = self.video_panel.winfo_height()
        if panel_w < 10 or panel_h < 10:
            panel_w, panel_h = self.TARGET_WIDTH, self.TARGET_HEIGHT

        iw, ih = img.size
        scale = min(panel_w / iw, panel_h / ih)
        new_w, new_h = max(1, int(iw * scale)), max(1, int(ih * scale))

        try:
            resample = Image.Resampling.LANCZOS
        except Exception:
            resample = Image.LANCZOS

        img = img.resize((new_w, new_h), resample)
        self._tkimg = ImageTk.PhotoImage(img)
        self.video_panel.config(image=self._tkimg, text="")

    pass
