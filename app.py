import queue
import threading
from datetime import datetime
import numpy as np
import torch

from tkinter import (
    Label, Button, Scale, HORIZONTAL, Frame, Text
)
from tkinter import ttk
from tkinter import StringVar

from ui.translations import TRANSLATIONS, CLASS_NAME_TRANSLATIONS
from ui.results_mixin import ResultsMixin
from core.tracking_mixin import TrackingMixin
from core.roi_mixin import RoiMixin
from core.video_mixin import VideoMixin
from core.detection_mixin import DetectionMixin
from core.export_csv import save_results_csv


class YoloVideoApp(
    DetectionMixin,
    VideoMixin,
    RoiMixin,
    TrackingMixin,
    ResultsMixin,
):
    def __init__(self, window):
        self.window = window
        self.window.title("System ostrzegania")

        # ustawienie kolorow dla GUI
        self.bg_main = "#5a5a5a"
        self.bg_panel = "#6a6a6a"
        self.bg_button = "#707070"
        self.bg_separator = "#2b2b2b"
        self.fg_main = "white"
        self.fg_dim = "#d8d8d8"
        self.fg_disabled = "#b0b0b0"
        self.active_bg = "#808080"

        self.TARGET_WIDTH = 960
        self.TARGET_HEIGHT = 540

        left_w = 460
        total_w = left_w + self.TARGET_WIDTH + 30
        total_h = max(720, self.TARGET_HEIGHT + 120)
        self.window.geometry(f"{total_w}x{total_h}")
        self.window.resizable(True, True)
        self.window.configure(bg=self.bg_main)

        self.main = Frame(window, bg=self.bg_main)
        self.main.pack(fill="both", expand=True)

        self.controls = Frame(self.main, width=left_w, bg=self.bg_main)
        self.controls.pack(side="left", fill="y")
        self.controls.pack_propagate(False)

        self.preview = Frame(self.main, bg="black")
        self.preview.pack(side="right", fill="both", expand=True)

        self.video_panel = Label(
            self.preview,
            text="",
            bg="black",
            fg="white",
            anchor="center",
            justify="center",
        )
        self.video_panel.pack(fill="both", expand=True, padx=10, pady=10)
        self._tkimg = None

        self.surface_model = None
        self.line_model = None
        self.surface_hailo_model = None
        self.line_hailo_model = None
        self.line_unet_model = None
        self.unet_model = None
        self.surface_deeplab_model = None
        self.line_deeplab_model = None
        self.surface_segformer_model = None
        self.line_segformer_model = None
        self.surface_segformer_processor = None
        self.line_segformer_processor = None

        self.line_backend_var = StringVar(value="YOLO")
        self.surface_backend_var = StringVar(value="YOLO")

        self.line_yolo_name = None
        self.line_hailo_name = None
        self.line_unet_name = None
        self.line_deeplab_name = None
        self.surface_yolo_name = None
        self.surface_hailo_name = None
        self.surface_unet_name = None
        self.surface_deeplab_name = None
        self.line_segformer_name = None
        self.surface_segformer_name = None
        self.line_deeplab_arch = None
        self.surface_deeplab_arch = None

        self.unet_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.line_unet_device = self.unet_device
        self.deeplab_device = self.unet_device
        self.line_deeplab_device = self.unet_device
        self.segformer_device = self.unet_device
        self.line_segformer_device = self.unet_device
        self.yolo_device = 0 if torch.cuda.is_available() else "cpu"

        self.line_unet_num_classes = 4
        self.line_unet_base = 32
        self.line_unet_img_size = 384
        self.line_unet_min_area = 250
        self.line_unet_names = {
            3: "yellow_line",
            2: "white_line",
            1: "red_line",
        }

        self.unet_num_classes = 5
        self.unet_base = 32
        self.unet_img_size = 384
        self.unet_min_area = 800
        self.unet_names = {
            0: "asfalt",
            1: "beton",
            2: "kostka",
            3: "trawa",
        }

        self.line_deeplab_num_classes = self.line_unet_num_classes
        self.line_deeplab_img_size = self.line_unet_img_size
        self.line_deeplab_min_area = self.line_unet_min_area
        self.line_deeplab_names = dict(self.line_unet_names)

        self.deeplab_num_classes = self.unet_num_classes
        self.deeplab_img_size = self.unet_img_size
        self.deeplab_min_area = self.unet_min_area
        self.deeplab_names = dict(self.unet_names)

        self.line_segformer_num_classes = self.line_unet_num_classes
        self.line_segformer_min_area = self.line_unet_min_area
        self.line_segformer_names = {0: "background", **dict(self.line_unet_names)}

        self.segformer_num_classes = self.unet_num_classes
        self.segformer_min_area = self.unet_min_area
        self.segformer_names = {0: "background", **dict(self.unet_names)}

        self.unet_colors_bgr = np.array([
            [0, 0, 0],
            [0, 0, 255],
            [0, 255, 0],
            [255, 0, 0],
            [0, 255, 255],
        ], dtype=np.uint8)

        self.last_surface_roi = None
        self.surface_roi_dilate = 15
        
        self.surface_update_stride = 3


        self.cached_surface_name = None
        self.cached_surface_area = 0.0

        self.min_confirm_frames = 3
        self.max_track_gap = 2
        self.match_distance_px = 80
        self.active_tracks = {}
        self.next_track_id = 1

        self.class_confidences = {}
        self.detections = []
        self.class_stats = {}

        self.max_event_log_entries = 200
        self.event_log_entries = []
        self.event_log_queue = queue.Queue()
        self.current_surface_name = None
        self.pending_surface_name = None
        self.pending_surface_hits = 0
        self.surface_change_confirm_frames = 2
        self.red_line_active = False
        self.red_line_miss_frames = 0
        self.red_line_clear_after = 5
        self._frame_best_surface = None
        self._frame_best_surface_area = 0
        self._frame_red_line_seen = False

        self.results_win = None
        self.results_tree = None
        self.results_info = None
        self.results_update_job = None

        self.hist_win = None
        self.hist_canvas = None
        self.hist_fig = None
        self.hist_ax = None
        self.hist_update_job = None

        self.stats_lock = threading.Lock()
        self.frame_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.worker_thread = None
        self.worker_done = False
        self.worker_error = None
        self.display_job = None
        self.display_delay_ms = 33

        self.pause_event = threading.Event()
        self.paused = False

        self.benchmark_mode = False
        self.benchmark_headless = False
        self.benchmark_display_stride = 4
        self.show_runtime_overlay = True

        self.lang = "pl"
        self.translations = TRANSLATIONS
        self.class_name_translations = CLASS_NAME_TRANSLATIONS
        self.video_panel.config(text=self.tr("video_placeholder"))

        # ustawienie stylu
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        self.style.configure(
            "Dark.TCombobox",
            fieldbackground=self.bg_panel,
            background=self.bg_panel,
            foreground=self.fg_main,
            arrowcolor=self.fg_main,
            bordercolor=self.bg_separator,
            lightcolor=self.bg_panel,
            darkcolor=self.bg_panel,
        )

        self._build_ui()

        self.running = False
        self.frame_count = 0
        self.current_fps = 0.0
        self.avg_fps = 0.0
        self.process_stride = 1
        self.conf_value = 0.25

        self.window.bind("<KeyPress-q>", lambda e: self.stop_processing())
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

        self.on_surface_backend_changed()
        self.on_line_backend_changed()
        self.update_roi_label(self.roi_slider.get())

    def _btn_style(self):
        return {
            "bg": self.bg_button,
            "fg": self.fg_main,
            "activebackground": self.active_bg,
            "activeforeground": self.fg_main,
            "disabledforeground": "#dddddd",
            "highlightbackground": self.bg_main,
        }

    def _build_ui(self):
        self.model_label = Label(
            self.controls,
            text=self.tr("select_model"),
            bg=self.bg_main,
            fg=self.fg_main
        )
        self.model_label.pack(pady=10)

        models_row = Frame(self.controls, bg=self.bg_main)
        models_row.pack(pady=5)

        surface_col = Frame(models_row, bg=self.bg_main)
        surface_col.pack(side="left", padx=15)

        Label(surface_col, text="Model", bg=self.bg_main, fg=self.fg_main).pack()
        self.surface_backend_combo = ttk.Combobox(
            surface_col,
            textvariable=self.surface_backend_var,
            values=["YOLO", "HAILO", "UNET", "DEEPLABV3", "SEGFORMER"],
            state="readonly",
            width=12,
            style="Dark.TCombobox"
        )
        self.surface_backend_combo.pack(pady=2)
        self.surface_backend_combo.bind("<<ComboboxSelected>>", lambda e: self.on_surface_backend_changed())

        self.surface_model_button = Button(
            surface_col,
            text=self.tr("surface_model"),
            command=self.select_surface_model,
            **self._btn_style()
        )
        self.surface_model_button.pack(pady=5)

        self.surface_clear_button = Button(
            surface_col,
            text=self.tr("clear_model"),
            command=self.clear_surface_model,
            **self._btn_style()
        )
        self.surface_clear_button.pack(pady=2)

        self.surface_model_name_label = Label(
            surface_col,
            text=f"{self.tr('current_surface_model')}{self.tr('no_model')}",
            fg=self.fg_disabled,
            bg=self.bg_main,
            wraplength=200,
            justify="left"
        )
        self.surface_model_name_label.pack()

        line_col = Frame(models_row, bg=self.bg_main)
        line_col.pack(side="left", padx=15)

        Label(line_col, text="Model", bg=self.bg_main, fg=self.fg_main).pack()
        self.line_backend_combo = ttk.Combobox(
            line_col,
            textvariable=self.line_backend_var,
            values=["YOLO", "HAILO", "UNET", "DEEPLABV3", "SEGFORMER"],
            state="readonly",
            width=12,
            style="Dark.TCombobox"
        )
        self.line_backend_combo.pack(pady=2)
        self.line_backend_combo.bind("<<ComboboxSelected>>", lambda e: self.on_line_backend_changed())

        self.line_model_button = Button(
            line_col,
            text=self.tr("line_model"),
            command=self.select_line_model,
            **self._btn_style()
        )
        self.line_model_button.pack(pady=5)

        self.line_clear_button = Button(
            line_col,
            text=self.tr("clear_model"),
            command=self.clear_line_model,
            **self._btn_style()
        )
        self.line_clear_button.pack(pady=2)

        self.line_model_name_label = Label(
            line_col,
            text=f"{self.tr('current_line_model')}{self.tr('no_model')}",
            fg=self.fg_disabled,
            bg=self.bg_main,
            wraplength=200,
            justify="left"
        )
        self.line_model_name_label.pack()

        sep1 = Frame(self.controls, bg=self.bg_separator, height=2, width=420)
        sep1.pack(pady=12)
        sep1.pack_propagate(False)

        self.video_label = Label(
            self.controls,
            text=self.tr("select_video_conf"),
            bg=self.bg_main,
            fg=self.fg_main
        )
        self.video_label.pack(pady=10)

        self.conf_label = Label(
            self.controls,
            text=f"{self.tr('confidence')}: 0.25",
            bg=self.bg_main,
            fg=self.fg_main
        )
        self.conf_label.pack()

        self.conf_slider = Scale(
            self.controls,
            from_=5,
            to=95,
            orient=HORIZONTAL,
            command=self.update_conf_label,
            bg=self.bg_main,
            fg=self.fg_main,
            highlightbackground=self.bg_main,
            troughcolor=self.bg_panel
        )
        self.conf_slider.set(25)
        self.conf_slider.pack()

        self.roi_label = Label(
            self.controls,
            text=f"{self.tr('roi_limit')}: 0% ({self.tr('roi_off')})",
            bg=self.bg_main,
            fg=self.fg_main
        )
        self.roi_label.pack()

        self.roi_slider = Scale(
            self.controls,
            from_=0,
            to=50,
            orient=HORIZONTAL,
            command=self.update_roi_label,
            bg=self.bg_main,
            fg=self.fg_main,
            highlightbackground=self.bg_main,
            troughcolor=self.bg_panel
        )
        self.roi_slider.set(0)
        self.roi_slider.pack()

        video_actions = Frame(self.controls, bg=self.bg_main)
        video_actions.pack(pady=10)

        self.video_button = Button(
            video_actions,
            text=self.tr("choose_video"),
            command=self.select_video,
            **self._btn_style()
        )
        self.video_button.pack(side="left", padx=(0, 6))

        self.video_clear_button = Button(
            video_actions,
            text=self.tr("clear_video"),
            command=self.clear_video,
            **self._btn_style()
        )
        self.video_clear_button.pack(side="left")

        self.pause_button = Button(
            self.controls,
            text=self.tr("pause"),
            command=self.toggle_pause,
            state="disabled",
            **self._btn_style()
        )
        self.pause_button.pack(pady=5)

        self.overlay_button = Button(
            self.controls,
            text=self._get_overlay_button_text(),
            command=self.toggle_runtime_overlay,
            **self._btn_style()
        )
        self.overlay_button.pack(pady=(0, 5))

        sep2 = Frame(self.controls, bg=self.bg_separator, height=2, width=420)
        sep2.pack(pady=12)
        sep2.pack_propagate(False)

        actions_row = Frame(self.controls, bg=self.bg_main)
        actions_row.pack(fill="x", padx=18, pady=10)

        left_actions = Frame(actions_row, bg=self.bg_main)
        left_actions.pack(side="left", anchor="n")

        right_actions = Frame(actions_row, bg=self.bg_main)
        right_actions.pack(side="right", anchor="n")

        self.save_csv_button = Button(
            left_actions,
            text=self.tr("save_csv"),
            command=self.save_csv,
            **self._btn_style()
        )
        self.save_csv_button.pack(pady=(0, 8), anchor="w")

        self.lang_button = Button(
            left_actions,
            text=self.tr("lang_toggle"),
            command=self.toggle_language,
            **self._btn_style()
        )
        self.lang_button.pack(anchor="w")

        self.results_button = Button(
            right_actions,
            text=self.tr("analysis_results"),
            command=self.open_results_window,
            **self._btn_style()
        )
        self.results_button.pack(anchor="e")

        sep3 = Frame(self.controls, bg=self.bg_separator, height=2, width=420)
        sep3.pack(pady=(6, 12))
        sep3.pack_propagate(False)

        self.event_log_label = Label(
            self.controls,
            text=self.tr("event_log"),
            bg=self.bg_main,
            fg=self.fg_main
        )
        self.event_log_label.pack(anchor="w", padx=18)

        log_frame = Frame(self.controls, bg=self.bg_main)
        log_frame.pack(fill="both", expand=True, padx=18, pady=(8, 14))

        self.event_log_text = Text(
            log_frame,
            height=10,
            wrap="none",
            state="disabled",
            bg=self.bg_panel,
            fg=self.fg_main,
            insertbackground=self.fg_main,
            relief="flat",
            padx=8,
            pady=8
        )
        self.event_log_text.pack(side="left", fill="both", expand=True)

        self.event_log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.event_log_text.yview)
        self.event_log_scroll.pack(side="right", fill="y")
        self.event_log_text.configure(yscrollcommand=self.event_log_scroll.set)

    def _get_benchmark_button_text(self):
        if self.benchmark_headless:
            mode = self.tr("benchmark_headless")
        elif self.benchmark_mode:
            mode = self.tr("benchmark_gui")
        else:
            mode = self.tr("benchmark_off")
        return f"{self.tr('benchmark_button')}: {mode}"

    def toggle_benchmark_mode(self):
        if self.running:
            return
        if not self.benchmark_mode and not self.benchmark_headless:
            self.benchmark_mode = True
            self.benchmark_headless = False
        elif self.benchmark_mode and not self.benchmark_headless:
            self.benchmark_mode = False
            self.benchmark_headless = True
        else:
            self.benchmark_mode = False
            self.benchmark_headless = False

    def _get_overlay_button_text(self):
        state = self.tr("overlay_on") if getattr(self, "show_runtime_overlay", True) else self.tr("overlay_off")
        return f"{self.tr('overlay_button')}: {state}"

    def toggle_runtime_overlay(self):
        self.show_runtime_overlay = not getattr(self, "show_runtime_overlay", True)
        if hasattr(self, "overlay_button"):
            self.overlay_button.config(text=self._get_overlay_button_text())

    def tr(self, key):
        return self.translations[self.lang].get(key, key)

    def toggle_language(self):
        self.lang = "en" if self.lang == "pl" else "pl"
        self.update_labels()

    def translate_class_name(self, name):
        return self.class_name_translations.get(self.lang, {}).get(name, name)

    def update_labels(self):
        self.model_label.config(text=self.tr("select_model"))
        self.video_label.config(text=self.tr("select_video_conf"))
        self.conf_label.config(text=f"{self.tr('confidence')}: {self.conf_slider.get() / 100:.2f}")
        self.video_button.config(text=self.tr("choose_video"))
        self.video_clear_button.config(text=self.tr("clear_video"))
        self.save_csv_button.config(text=self.tr("save_csv"))
        self.lang_button.config(text=self.tr("lang_toggle"))
        self.results_button.config(text=self.tr("analysis_results"))
        self.pause_button.config(text=self.tr("resume") if self.paused else self.tr("pause"))
        if hasattr(self, "overlay_button"):
            self.overlay_button.config(text=self._get_overlay_button_text())
        self.surface_clear_button.config(text=self.tr("clear_model"))
        self.line_clear_button.config(text=self.tr("clear_model"))
        self.event_log_label.config(text=self.tr("event_log"))
        self.update_roi_label(self.roi_slider.get())

        if self.results_win is not None and self.results_win.winfo_exists():
            self.results_win.title(self.tr("stats_title"))
            self.update_results_window()

        if self.hist_win is not None and self.hist_win.winfo_exists():
            self.hist_win.title(self.tr("hist_title"))
            self.update_histogram()

        self.on_surface_backend_changed()
        self.on_line_backend_changed()

    def update_conf_label(self, val):
        self.conf_label.config(text=f"{self.tr('confidence')}: {int(float(val)) / 100:.2f}")

    def update_roi_label(self, val):
        pct = int(float(val))
        if pct <= 0:
            txt = f"{self.tr('roi_limit')}: 0% ({self.tr('roi_off')})"
        else:
            txt = f"{self.tr('roi_limit')}: {pct}%"
        self.roi_label.config(text=txt)

    def tr_model_type(self, t):
        if self.lang == "pl":
            return {"surface": "nawierzchnia", "line": "linia"}.get(t, t)
        return {"surface": "surface", "line": "line"}.get(t, t)

    def line_ready(self):
        backend = self.line_backend_var.get()
        if backend == "YOLO":
            return self.line_model is not None
        if backend == "HAILO":
            return self.line_hailo_model is not None
        if backend == "UNET":
            return self.line_unet_model is not None
        if backend == "DEEPLABV3":
            return self.line_deeplab_model is not None
        return self.line_segformer_model is not None

    def surface_ready(self):
        backend = self.surface_backend_var.get()
        if backend == "YOLO":
            return self.surface_model is not None
        if backend == "HAILO":
            return self.surface_hailo_model is not None
        if backend == "UNET":
            return self.unet_model is not None
        if backend == "DEEPLABV3":
            return self.surface_deeplab_model is not None
        return self.surface_segformer_model is not None

    def on_line_backend_changed(self):
        backend = self.line_backend_var.get()
        self.line_model_button.config(text=self.tr("line_model"))

        if backend == "YOLO":
            ready = self.line_model is not None
            name = self.line_yolo_name if ready else self.tr("no_model")
        elif backend == "HAILO":
            ready = self.line_hailo_model is not None
            name = self.line_hailo_name if ready else self.tr("no_model")
        elif backend == "UNET":
            ready = self.line_unet_model is not None
            name = self.line_unet_name if ready else self.tr("no_model")
        elif backend == "DEEPLABV3":
            ready = self.line_deeplab_model is not None
            if ready and self.line_deeplab_arch:
                name = f"{self.line_deeplab_name} ({self.line_deeplab_arch})"
            else:
                name = self.line_deeplab_name if ready else self.tr("no_model")
        else:
            ready = self.line_segformer_model is not None
            name = self.line_segformer_name if ready else self.tr("no_model")

        self.line_model_name_label.config(
            text=f"{self.tr('current_line_model')}{name}",
            fg=self.fg_main if ready else self.fg_disabled,
            bg=self.bg_main
        )

    def on_surface_backend_changed(self):
        backend = self.surface_backend_var.get()
        self.surface_model_button.config(text=self.tr("surface_model"))

        if backend == "YOLO":
            ready = self.surface_model is not None
            name = self.surface_yolo_name if ready else self.tr("no_model")
        elif backend == "HAILO":
            ready = self.surface_hailo_model is not None
            name = self.surface_hailo_name if ready else self.tr("no_model")
        elif backend == "UNET":
            ready = self.unet_model is not None
            name = self.surface_unet_name if ready else self.tr("no_model")
        elif backend == "DEEPLABV3":
            ready = self.surface_deeplab_model is not None
            if ready and self.surface_deeplab_arch:
                name = f"{self.surface_deeplab_name} ({self.surface_deeplab_arch})"
            else:
                name = self.surface_deeplab_name if ready else self.tr("no_model")
        else:
            ready = self.surface_segformer_model is not None
            name = self.surface_segformer_name if ready else self.tr("no_model")

        self.surface_model_name_label.config(
            text=f"{self.tr('current_surface_model')}{name}",
            fg=self.fg_main if ready else self.fg_disabled,
            bg=self.bg_main
        )

    def begin_frame_event_scan(self):
        self._frame_best_surface = None
        self._frame_best_surface_area = 0
        self._frame_red_line_seen = False

    def observe_surface_candidate(self, class_name, area):
        try:
            area_val = float(area)
        except Exception:
            area_val = 0.0
        if area_val <= self._frame_best_surface_area:
            return
        self._frame_best_surface_area = area_val
        self._frame_best_surface = class_name

    def observe_line_candidate(self, class_name):
        if class_name == "red_line":
            self._frame_red_line_seen = True

    def finalize_frame_event_scan(self):
        current_candidate = self._frame_best_surface
        if current_candidate is None:
            self.pending_surface_name = None
            self.pending_surface_hits = 0
        elif current_candidate == self.current_surface_name:
            self.pending_surface_name = None
            self.pending_surface_hits = 0
        elif current_candidate == self.pending_surface_name:
            self.pending_surface_hits += 1
        else:
            self.pending_surface_name = current_candidate
            self.pending_surface_hits = 1

        if (
            current_candidate is not None
            and self.pending_surface_name == current_candidate
            and self.pending_surface_hits >= self.surface_change_confirm_frames
        ):
            previous_surface = self.current_surface_name
            self.current_surface_name = current_candidate
            self.pending_surface_name = None
            self.pending_surface_hits = 0

            if previous_surface is not None and previous_surface != current_candidate:
                prev_label = self.translate_class_name(previous_surface)
                curr_label = self.translate_class_name(current_candidate)
                self.enqueue_event_log(
                    self.tr("surface_change"),
                    f"{prev_label} -> {curr_label}"
                )

        if self._frame_red_line_seen:
            self.red_line_miss_frames = 0
            if not self.red_line_active:
                self.red_line_active = True
                self.enqueue_event_log(self.tr("warning"), self.tr("red_line_detected"))
        else:
            if self.red_line_active:
                self.red_line_miss_frames += 1
                if self.red_line_miss_frames >= self.red_line_clear_after:
                    self.red_line_active = False
                    self.red_line_miss_frames = 0

    def enqueue_event_log(self, event_type, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.event_log_queue.put((timestamp, event_type, message))

    def drain_event_log_queue(self):
        updated = False
        while True:
            try:
                entry = self.event_log_queue.get_nowait()
            except queue.Empty:
                break
            self.event_log_entries.append(entry)
            updated = True

        if not updated:
            return

        if len(self.event_log_entries) > self.max_event_log_entries:
            self.event_log_entries = self.event_log_entries[-self.max_event_log_entries:]

        self.render_event_log()

    def render_event_log(self):
        if not hasattr(self, "event_log_text") or self.event_log_text is None:
            return

        self.event_log_text.config(state="normal")
        self.event_log_text.delete("1.0", "end")
        for timestamp, event_type, message in self.event_log_entries:
            self.event_log_text.insert("end", f"{timestamp} | {event_type} | {message}\n")
        self.event_log_text.see("end")
        self.event_log_text.config(state="disabled")

    def reset_event_log_state(self):
        self.current_surface_name = None
        self.pending_surface_name = None
        self.pending_surface_hits = 0
        self.red_line_active = False
        self.red_line_miss_frames = 0
        self.cached_surface_name = None
        self.cached_surface_area = 0.0
        self.begin_frame_event_scan()

        while True:
            try:
                self.event_log_queue.get_nowait()
            except queue.Empty:
                break

        self.event_log_entries = []
        self.render_event_log()

    def save_csv(self):
        from tkinter import filedialog, messagebox

        with self.stats_lock:
            has_data = bool(self.class_confidences) and bool(self.detections)
            confidences_copy = {k: list(v) for k, v in self.class_confidences.items()}
            detections_copy = list(self.detections)
            current_fps_copy = float(self.current_fps)
            avg_fps_copy = float(self.avg_fps)

        event_log_copy = list(self.event_log_entries)

        if not has_data:
            messagebox.showwarning(self.tr("no_data"), self.tr("no_data_msg"))
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if not save_path:
            return

        try:
            surface_backend = self.surface_backend_var.get()
            line_backend = self.line_backend_var.get()

            if surface_backend == "DEEPLABV3":
                surface_num_classes = self.deeplab_num_classes
                surface_names = self.deeplab_names
            elif surface_backend == "SEGFORMER":
                surface_num_classes = self.segformer_num_classes
                surface_names = self.segformer_names
            else:
                surface_num_classes = self.unet_num_classes
                surface_names = self.unet_names

            if line_backend == "DEEPLABV3":
                line_num_classes = self.line_deeplab_num_classes
                line_names = self.line_deeplab_names
            elif line_backend == "SEGFORMER":
                line_num_classes = self.line_segformer_num_classes
                line_names = self.line_segformer_names
            else:
                line_num_classes = self.line_unet_num_classes
                line_names = self.line_unet_names

            surface_model_for_export = self.surface_hailo_model if surface_backend == "HAILO" else self.surface_model
            line_model_for_export = self.line_hailo_model if line_backend == "HAILO" else self.line_model

            save_results_csv(
                save_path=save_path,
                surface_backend=surface_backend,
                line_backend=line_backend,
                surface_model=surface_model_for_export,
                line_model=line_model_for_export,
                unet_num_classes=surface_num_classes,
                unet_names=surface_names,
                line_unet_num_classes=line_num_classes,
                line_unet_names=line_names,
                confidences_copy=confidences_copy,
                detections_copy=detections_copy,
                current_fps=current_fps_copy,
                avg_fps=avg_fps_copy,
                event_log_copy=event_log_copy,
            )
            messagebox.showinfo(self.tr("saved"), self.tr("saved_msg").format(save_path))
        except Exception as e:
            messagebox.showerror(self.tr("save_error"), self.tr("save_error_msg").format(e))
