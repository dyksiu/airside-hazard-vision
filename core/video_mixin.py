import queue
import threading
import time
import cv2
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox


class VideoMixin:
    def _begin_profile_frame(self):
        self._profile_frame_sums = {}

    def _add_profile_frame_ms(self, key, value_ms):
        if not hasattr(self, "_profile_frame_sums") or self._profile_frame_sums is None:
            self._profile_frame_sums = {}
        self._profile_frame_sums[key] = self._profile_frame_sums.get(key, 0.0) + float(value_ms)

    def _commit_profile_frame(self):
        frame_profile = getattr(self, "_profile_frame_sums", None) or {}
        if not frame_profile:
            return
        with self.stats_lock:
            sums = getattr(self, "stage_profile_sums", {})
            counts = getattr(self, "stage_profile_counts", {})
            for key, value in frame_profile.items():
                sums[key] = sums.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1
            self.stage_profile_sums = sums
            self.stage_profile_counts = counts
            self.last_stage_profile = dict(frame_profile)

    def _record_stage_metric_ms(self, key, value_ms):
        with self.stats_lock:
            sums = getattr(self, "stage_profile_sums", {})
            counts = getattr(self, "stage_profile_counts", {})
            sums[key] = sums.get(key, 0.0) + float(value_ms)
            counts[key] = counts.get(key, 0) + 1
            self.stage_profile_sums = sums
            self.stage_profile_counts = counts

    def copy_stage_profile_stats(self):
        with self.stats_lock:
            sums = dict(getattr(self, "stage_profile_sums", {}))
            counts = dict(getattr(self, "stage_profile_counts", {}))
            last = dict(getattr(self, "last_stage_profile", {}))
        avg = {}
        for key, value in sums.items():
            count = counts.get(key, 0)
            if count > 0:
                avg[key] = value / count
        return avg, last

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
            self.gui_render_count = 0
            self.stage_profile_sums = {}
            self.stage_profile_counts = {}
            self.last_stage_profile = {}
        self._profile_frame_sums = {}

    def copy_runtime_stats(self):
        with self.stats_lock:
            return {
                "frame_count": self.frame_count,
                "current_fps": self.current_fps,
                "avg_fps": self.avg_fps,
                "input_fps": getattr(self, "input_fps", 0.0),
                "inference_fps": getattr(self, "inference_fps", 0.0),
                "detection_fps": getattr(self, "detection_fps", 0.0),
                "display_fps": getattr(self, "display_fps", 0.0),
                "avg_infer_ms": getattr(self, "avg_infer_ms", 0.0),
                "gui_render_count": getattr(self, "gui_render_count", 0),
            }

    def get_runtime_stats_dict(self):
        return self.copy_runtime_stats()

    def draw_fps_overlay(self, frame, runtime_stats):
        if not getattr(self, "show_runtime_overlay", True):
            return frame

        if not isinstance(runtime_stats, dict):
            runtime_stats = {
                "inference_fps": float(runtime_stats),
                "detection_fps": float(runtime_stats),
                "display_fps": 0.0,
                "avg_infer_ms": 0.0,
            }

        lines = [
            f"{self.tr('fps_inference')}: {runtime_stats.get('inference_fps', 0.0):.1f}",
            f"{self.tr('fps_detection')}: {runtime_stats.get('detection_fps', 0.0):.1f}",
            f"{self.tr('fps_display')}: {runtime_stats.get('display_fps', 0.0):.1f}",
            f"{self.tr('infer_time_avg')}: {runtime_stats.get('avg_infer_ms', 0.0):.1f} ms",
        ]

        if getattr(self, "benchmark_headless", False):
            lines.append(self.tr("benchmark_headless_badge"))
        elif getattr(self, "benchmark_mode", False):
            lines.append(f"{self.tr('benchmark_badge')}: x{int(getattr(self, 'benchmark_display_stride', 1))}")

        profile_avg, _ = self.copy_stage_profile_stats()
        profile_keys = [
            ("hailo_preprocess_ms", "Hailo pre"),
            ("hailo_infer_ms", "Hailo infer"),
            ("hailo_postprocess_ms", "Hailo post"),
            ("hailo_post_groups_ms", "Post groups"),
            ("hailo_post_decode_ms", "Post decode"),
            ("hailo_post_filter_ms", "Post filter"),
            ("hailo_post_nms_ms", "Post NMS"),
            ("hailo_mask_decode_ms", "Mask decode"),
            ("hailo_post_scale_boxes_ms", "Scale boxes"),
            ("hailo_post_project_masks_ms", "Project masks"),
            ("hailo_contours_ms", "Contours"),
            ("hailo_contour_draw_ms", "Contour draw"),
            ("hailo_label_draw_ms", "Label draw"),
            ("gui_render_ms", "GUI render"),
        ]
        for key, label in profile_keys:
            value = profile_avg.get(key)
            if value is not None:
                lines.append(f"{label}: {value:.1f} ms")

        frame_h, frame_w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.68 if frame_w >= 900 else 0.60
        thickness = 2
        line_h = max(22, int(24 * (font_scale / 0.68)))
        pad_x = 18
        pad_top = 16
        pad_bottom = 14
        col_gap = 28

        n_cols = 2 if len(lines) > 6 else 1
        rows = (len(lines) + n_cols - 1) // n_cols
        columns = [lines[i * rows:(i + 1) * rows] for i in range(n_cols)]

        box_x = 14
        box_y = 12
        box_w = max(220, frame_w - 28)
        box_h = pad_top + rows * line_h + pad_bottom

        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.38, frame, 0.62, 0)
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (235, 235, 235), 1)

        usable_w = box_w - 2 * pad_x
        if n_cols == 1:
            x_positions = [box_x + pad_x]
        else:
            col_w = max(100, (usable_w - col_gap) // 2)
            x_positions = [box_x + pad_x, box_x + pad_x + col_w + col_gap]

        for col_idx, col in enumerate(columns):
            base_x = x_positions[col_idx]
            for row_idx, line in enumerate(col):
                yy = box_y + pad_top + (row_idx + 1) * line_h - 6
                cv2.putText(frame, line, (base_x, yy), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                cv2.putText(frame, line, (base_x, yy), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return frame

    def select_video(self):
        if (not self.surface_ready()) and (not self.line_ready()):
            messagebox.showwarning(self.tr("no_model_title"), self.tr("no_model_msg"))
            return

        video_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
        if not video_path:
            return

        self.last_video_path = video_path
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
            daemon=True,
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

        if not getattr(self, "benchmark_headless", False):
            try:
                frame = self.frame_queue.get_nowait()
                render_started_at = time.perf_counter()
                self.show_frame_in_tk(frame)
                render_ms = (time.perf_counter() - render_started_at) * 1000.0
                self._record_stage_metric_ms("gui_render_ms", render_ms)
                with self.stats_lock:
                    self.gui_render_count = getattr(self, "gui_render_count", 0) + 1
            except queue.Empty:
                pass
        else:
            try:
                while True:
                    _ = self.frame_queue.get_nowait()
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

            benchmark_mode = bool(getattr(self, "benchmark_mode", False))
            benchmark_headless = bool(getattr(self, "benchmark_headless", False))
            benchmark_display_stride = max(1, int(getattr(self, "benchmark_display_stride", 1)))
            gui_stride = benchmark_display_stride if (benchmark_mode and not benchmark_headless) else 1

            started_at = time.perf_counter()
            last_annotated = None
            frame_idx = 0
            infer_count = 0
            infer_time_sum = 0.0

            with self.stats_lock:
                self.input_fps = float(fps)
                self.gui_render_count = 0

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
                    self._begin_profile_frame()
                    infer_started_this = time.perf_counter()
                    frame_vis = self.run_detection_on_frame(frame, frame_idx)
                    infer_finished_this = time.perf_counter()
                    self._commit_profile_frame()

                    last_annotated = frame_vis
                    infer_count += 1
                    infer_time_sum += max(0.0, infer_finished_this - infer_started_this)
                else:
                    frame_vis = last_annotated.copy() if last_annotated is not None else frame

                work_dt = time.perf_counter() - loop_started_at
                sleep = frame_time - work_dt
                if sleep > 0:
                    time.sleep(sleep)

                elapsed_wall = max(time.perf_counter() - started_at, 1e-6)
                with self.stats_lock:
                    gui_render_count = getattr(self, "gui_render_count", 0)

                inference_fps = infer_count / max(infer_time_sum, 1e-6)
                detection_fps = infer_count / elapsed_wall
                display_fps = gui_render_count / elapsed_wall
                runtime_stats = {
                    "frame_count": frame_idx,
                    "current_fps": inference_fps,
                    "avg_fps": detection_fps,
                    "input_fps": float(fps),
                    "inference_fps": inference_fps,
                    "detection_fps": detection_fps,
                    "display_fps": display_fps,
                    "avg_infer_ms": (infer_time_sum / infer_count * 1000.0) if infer_count else 0.0,
                    "gui_render_count": gui_render_count,
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

                if not benchmark_headless:
                    frame_vis = self.draw_fps_overlay(frame_vis, runtime_stats)

                should_push_preview = (not benchmark_headless) and ((frame_idx == 0) or (frame_idx % gui_stride == 0))
                if should_push_preview:
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
