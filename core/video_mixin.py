import queue
import threading
import time
import cv2
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox


class VideoMixin:

    def select_video(self):
        if (not self.surface_ready()) and (not self.line_ready()):
            messagebox.showwarning(self.tr("no_model_title"), self.tr("no_model_msg"))
            return

        video_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
        if not video_path:
            return

        self.stop_processing(silent=True)

        with self.stats_lock:
            self.class_confidences = {}
            self.detections = []
            self.class_stats = {}
            self.frame_count = 0

        self.reset_tracking()
        self.last_surface_roi = None

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

        if self.worker_done:
            self.running = False
            self.pause_button.config(state="disabled", text=self.tr("pause"))
            if self.worker_error:
                messagebox.showerror("Błąd", str(self.worker_error))
            else:
                messagebox.showinfo(self.tr("done"), self.tr("done_msg"))
            return

        self.display_job = self.window.after(self.display_delay_ms, self.display_loop)

    def worker_loop(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.worker_error = "Nie udało się otworzyć wideo."
                self.worker_done = True
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 30.0
            frame_time = 1.0 / fps

            last_annotated = None
            frame_idx = 0
            t0 = time.time()

            while not self.stop_event.is_set():
                while self.pause_event.is_set() and not self.stop_event.is_set():
                    time.sleep(0.05)

                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (self.TARGET_WIDTH, self.TARGET_HEIGHT))
                do_infer = (frame_idx % self.process_stride == 0)

                if do_infer:
                    frame_vis = self.run_detection_on_frame(frame, frame_idx)
                    last_annotated = frame_vis
                else:
                    frame_vis = last_annotated if last_annotated is not None else frame

                with self.stats_lock:
                    self.frame_count = frame_idx

                try:
                    while True:
                        _ = self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                self.frame_queue.put(frame_vis)

                dt = time.time() - t0
                sleep = frame_time - dt
                if sleep > 0:
                    time.sleep(sleep)
                t0 = time.time()
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
            self.video_panel.config(image="", text="(Zatrzymano)")

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