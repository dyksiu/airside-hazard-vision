import os
import queue
import sys
import threading
import time
import traceback
from datetime import datetime

import cv2
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox


class VideoMixin:

    def reset_fps_stats(self):
        with self.stats_lock:
            self.frame_count = 0
            self.current_fps = 0.0
            self.avg_fps = 0.0

    def copy_runtime_stats(self):
        with self.stats_lock:
            return self.frame_count, self.current_fps, self.avg_fps

    def draw_fps_overlay(self, frame, current_fps, avg_fps):
        lines = [
            f"{self.tr('fps_current')}: {current_fps:.1f}",
            f"{self.tr('fps_average')}: {avg_fps:.1f}",
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

    def _worker_is_alive(self):
        return self.worker_thread is not None and self.worker_thread.is_alive()

    def _app_base_dir(self):
        if getattr(sys, "frozen", False):
            return os.path.dirname(sys.executable)
        return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

    def _write_error_log(self, exc):
        try:
            logs_dir = os.path.join(self._app_base_dir(), "logs")
            os.makedirs(logs_dir, exist_ok=True)
            log_path = os.path.join(logs_dir, f"error_{datetime.now():%Y%m%d_%H%M%S}.log")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"Czas: {datetime.now().isoformat(timespec='seconds')}\n")
                f.write(f"Typ błędu: {type(exc).__name__}\n")
                f.write(f"Błąd: {exc}\n\n")
                f.write("Traceback:\n")
                f.write(traceback.format_exc())
            return log_path
        except Exception:
            return None

    def _reset_video_state_after_stop(self):
        with self.stats_lock:
            self.class_confidences = {}
            self.detections = []
            self.class_stats = {}

        self.reset_fps_stats()
        self.reset_tracking()
        self.reset_event_log_state()
        self.last_surface_roi = None
        self.worker_done = False
        self.worker_error = None
        self.worker_error_log_path = None
        self._pending_clear_video = False
        self._stop_requested_by_user = False
        self._stopping = False

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

    def _finish_worker_in_ui(self):
        self.display_job = None
        self.running = False
        self.paused = False
        self.pause_event.clear()
        self._stopping = False

        if self.worker_thread is not None and not self.worker_thread.is_alive():
            self.worker_thread = None

        if hasattr(self, "pause_button"):
            self.pause_button.config(state="disabled", text=self.tr("pause"))
        if hasattr(self, "set_processing_controls_state"):
            self.set_processing_controls_state(False)

        pending_clear = bool(getattr(self, "_pending_clear_video", False))
        stop_requested = bool(getattr(self, "_stop_requested_by_user", False))
        error = getattr(self, "worker_error", None)
        error_log_path = getattr(self, "worker_error_log_path", None)

        self._pending_clear_video = False
        self._stop_requested_by_user = False

        if pending_clear:
            self._reset_video_state_after_stop()
            return

        if error:
            msg = str(error)
            if error_log_path:
                msg = f"{msg}\n\n{self.tr('error_log_saved')}\n{error_log_path}"
            messagebox.showerror("Błąd", msg)
            return

        if stop_requested:
            self._tkimg = None
            self.video_panel.config(image="", text=self.tr("video_stopped"))
            return

        messagebox.showinfo(self.tr("done"), self.tr("done_msg"))

    def select_video(self):
        if self.running or self._worker_is_alive():
            messagebox.showwarning(self.tr("processing_locked_title"), self.tr("processing_locked_msg"))
            return

        if (not self.surface_ready()) and (not self.line_ready()):
            messagebox.showwarning(self.tr("no_model_title"), self.tr("no_model_msg"))
            return

        video_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
        if not video_path:
            return

        self.clear_video(silent=True)
        if self._worker_is_alive():
            return

        self.reset_fps_stats()

        self.conf_value = getattr(self, "conf_value", self.conf_slider.get() / 100)
        self.process_stride = 2 if (self.surface_ready() and self.line_ready()) else 1

        self.worker_done = False
        self.worker_error = None
        self.worker_error_log_path = None
        self._pending_clear_video = False
        self._stop_requested_by_user = False
        self._stopping = False
        self.stop_event.clear()
        self.running = True
        if hasattr(self, "set_processing_controls_state"):
            self.set_processing_controls_state(True)

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
        if self.running or self._worker_is_alive():
            self._pending_clear_video = True
            self.stop_processing(silent=True)
            return

        self._reset_video_state_after_stop()

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
            self._finish_worker_in_ui()
            return

        self.display_job = self.window.after(self.display_delay_ms, self.display_loop)

    def worker_loop(self, video_path):
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.worker_error = "Nie udało się otworzyć wideo."
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 30.0
            frame_time = 1.0 / fps

            last_annotated = None
            frame_idx = 0
            fps_started_at = time.perf_counter()

            while not self.stop_event.is_set():
                while self.pause_event.is_set() and not self.stop_event.is_set():
                    time.sleep(0.05)

                if self.stop_event.is_set():
                    break

                loop_started_at = time.perf_counter()

                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (self.TARGET_WIDTH, self.TARGET_HEIGHT))
                do_infer = (frame_idx % self.process_stride == 0)

                if do_infer:
                    frame_vis = self.run_detection_on_frame(frame, frame_idx)
                    last_annotated = frame_vis
                else:
                    frame_vis = last_annotated.copy() if last_annotated is not None else frame

                work_dt = time.perf_counter() - loop_started_at
                sleep = frame_time - work_dt
                if sleep > 0 and not self.stop_event.is_set():
                    time.sleep(sleep)

                loop_dt = max(time.perf_counter() - loop_started_at, 1e-6)
                instant_fps = 1.0 / loop_dt
                elapsed_total = max(time.perf_counter() - fps_started_at, 1e-6)
                avg_fps = (frame_idx + 1) / elapsed_total

                with self.stats_lock:
                    prev_fps = float(getattr(self, 'current_fps', 0.0) or 0.0)
                    self.frame_count = frame_idx
                    self.current_fps = instant_fps if prev_fps <= 0 else (prev_fps * 0.8 + instant_fps * 0.2)
                    self.avg_fps = avg_fps

                # Nakładka FPS została wyłączona w wersji produkcyjnej.
                # Statystyki FPS nadal są liczone wewnętrznie dla wyników/CSV,
                # ale nie są rysowane na podglądzie wideo.

                try:
                    while True:
                        _ = self.frame_queue.get_nowait()
                except queue.Empty:
                    pass

                if not self.stop_event.is_set():
                    self.frame_queue.put(frame_vis)

                frame_idx += 1

        except Exception as e:
            self.worker_error = e
            self.worker_error_log_path = self._write_error_log(e)
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            self.worker_done = True

    def stop_processing(self, silent=False):
        worker_alive = self._worker_is_alive()

        self.stop_event.set()
        self.pause_event.clear()
        self.paused = False
        self._stop_requested_by_user = True
        self._stopping = worker_alive

        if hasattr(self, "pause_button"):
            self.pause_button.config(state="disabled", text=self.tr("pause"))

        if worker_alive:
            if not silent:
                self._tkimg = None
                self.video_panel.config(image="", text=self.tr("video_stopping"))

            # Nie robimy join() w wątku GUI. Wątek roboczy sam zakończy się po
            # bieżącej klatce/inferencji, a display_loop odblokuje interfejs.
            if self.running and self.display_job is None:
                self.display_job = self.window.after(self.display_delay_ms, self.display_loop)
            return

        self.running = False
        self.worker_thread = None
        self.reset_tracking()

        if self.display_job is not None:
            try:
                self.window.after_cancel(self.display_job)
            except Exception:
                pass
            self.display_job = None

        if not silent:
            self._tkimg = None
            self.video_panel.config(image="", text=self.tr("video_stopped"))

        if hasattr(self, "set_processing_controls_state"):
            self.set_processing_controls_state(False)

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
