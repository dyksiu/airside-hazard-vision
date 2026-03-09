import os
import cv2
import csv
import random
import time
import queue
import threading
import numpy as np
from ultralytics import YOLO
from tkinter import Tk, Label, Button, filedialog, Scale, HORIZONTAL, messagebox, Frame, Toplevel
from tkinter import ttk
from PIL import Image, ImageTk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import torch
import torch.nn as nn
import torch.nn.functional as F
from tkinter import StringVar


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=5, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.head(d1)


class YoloVideoApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Detektor nawierzchni i linii")

        # stała rozdzielczość podglądu
        self.TARGET_WIDTH = 960
        self.TARGET_HEIGHT = 540

        # układ okna: lewa kolumna (GUI) + prawa (podgląd)
        left_w = 460
        total_w = left_w + self.TARGET_WIDTH + 30
        total_h = max(600, self.TARGET_HEIGHT + 40)
        self.window.geometry(f"{total_w}x{total_h}")
        self.window.resizable(True, True)

        self.main = Frame(window)
        self.main.pack(fill="both", expand=True)

        self.controls = Frame(self.main, width=left_w)
        self.controls.pack(side="left", fill="y")
        self.controls.pack_propagate(False)

        self.preview = Frame(self.main, bg="black")
        self.preview.pack(side="right", fill="both", expand=True)

        # miejsce na obraz z filmu (na start puste/tekst)
        self.video_panel = Label(
            self.preview,
            text="(Podgląd pojawi się po kliknięciu „Wybierz wideo”)",
            bg="black",
            fg="white",
            anchor="center",
            justify="center",
        )
        self.video_panel.pack(fill="both", expand=True, padx=10, pady=10)
        self._tkimg = None

        # osobne modele
        self.surface_model = None
        self.line_model = None

        self.line_unet_model = None

        # --- wybór backendu linii: YOLO / UNET ---
        self.line_backend_var = StringVar(value="YOLO")  # "YOLO" albo "UNET"

        self.line_yolo_name = None
        self.line_unet_name = None

        self.unet_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.line_unet_device = self.unet_device  # albo osobno, jeśli chcesz

        # konfiguracja U-Net dla linii (dopasuj do swoich wag!)
        self.line_unet_num_classes = 4  # 0=tło + 3 klasy linii (np. yellow/white/red)
        self.line_unet_base = 32
        self.line_unet_img_size = 384
        self.line_unet_min_area = 250  # linie są cienkie -> mniejszy próg

        # nazwy klas U-Net dla linii (0=tło)
        # UWAGA: dopasuj kolejność do treningu linii!
        self.line_unet_names = {
            3: "yellow_line",
            2: "white_line",
            1: "red_line",
        }

        # --- wybór backendu nawierzchni: YOLO / UNET ---
        self.surface_backend_var = StringVar(value="YOLO")  # "YOLO" albo "UNET"

        self.surface_yolo_name = None
        self.surface_unet_name = None

        self.unet_model = None
        self.unet_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.yolo_device = 0 if torch.cuda.is_available() else "cpu"

        # konfiguracja U-Net (musi pasować do treningu)
        self.unet_num_classes = 5
        self.unet_base = 32
        self.unet_img_size = 384  # dla RTX 3050 zwykle lepiej 256/384 niż 512
        self.unet_min_area = 800  # filtr szumu dla etykiet obszarów

        # nazwy klas U-Net (0=tło, 1..4 materiały) — dopasowane do Twojego data.yaml
        self.unet_names = {
            2: "asfalt",
            1: "beton",
            3: "kostka",
            4: "trawa",
        }

        # stałe kolory (BGR) dla U-Net: 0=tło
        self.unet_colors_bgr = np.array([
            [0, 0, 0],        # tło
            [0, 0, 255],      # asfalt (czerwony)
            [0, 255, 0],      # beton  (zielony)
            [255, 0, 0],      # kostka (niebieski)
            [0, 255, 255],    # trawa  (żółty)
        ], dtype=np.uint8)

        # --- ROI nawierzchni (do ograniczania linii, ważne dla UNET) ---
        self.last_surface_roi = None          # uint8 (H,W): 0/1
        self.surface_roi_dilate = 15          # kernel (px) do lekkiego poszerzenia ROI

        # dane z analizy
        self.class_confidences = {}
        self.detections = []
        self.class_stats = {}  # key -> {"count": int, "conf_sum": float}

        # okna / wykresy
        self.results_win = None
        self.results_tree = None
        self.results_info = None
        self.results_update_job = None

        self.hist_win = None
        self.hist_canvas = None
        self.hist_fig = None
        self.hist_ax = None
        self.hist_update_job = None

        # threading / płynność
        self.stats_lock = threading.Lock()
        self.frame_queue = queue.Queue(maxsize=1)   # trzymamy tylko NAJNOWSZĄ klatkę
        self.stop_event = threading.Event()
        self.worker_thread = None
        self.worker_done = False
        self.worker_error = None
        self.display_job = None
        self.display_delay_ms = 33  # ~30 FPS wyświetlania

        self.pause_event = threading.Event()  # set() = pauza
        self.paused = False

        # domyślny język
        self.lang = "pl"

        self.translations = {
            "pl": {
                "select_model": "Wybierz modele YOLO",
                "surface_model": "Model nawierzchni",
                "line_model": "Model linii",
                "current_surface_model": "Aktualny model nawierzchni: ",
                "current_line_model": "Aktualny model linii: ",
                "no_model": "BRAK",

                "select_video_conf": "Wybierz plik MP4 i ustaw próg confidence",
                "confidence": "Confidence",
                "choose_video": "Wybierz wideo",
                "save_csv": "Zapisz wyniki do CSV",

                "model_loaded": "Model załadowany",
                "model_loaded_msg": "Pomyślnie załadowano model:\n{}",
                "model_error": "Błąd",
                "model_error_msg": "Nie udało się załadować modelu:\n{}",


                "no_model_title": "Brak modelu",
                "no_model_msg": "Nie wybrano żadnego modelu YOLO!\nWybierz przynajmniej jeden model przed rozpoczęciem analizy.",

                "done": "Gotowe",
                "done_msg": "Analiza zakończona. Możesz teraz zapisać wyniki do CSV.",

                "no_data": "Brak danych",
                "no_data_msg": "Nie ma danych do zapisania. Najpierw przeanalizuj film.",

                "saved": "Zapisano",
                "saved_msg": "Wyniki zapisane do:\n{}",
                "save_error": "Błąd",
                "save_error_msg": "Nie udało się zapisać pliku:\n{}",


                "analysis_results": "Wyniki z analizy",
                "stats_title": "Wyniki z analizy (na żywo)",
                "model_type": "Typ",
                "class": "Klasa",
                "count": "Liczba",
                "avg_conf": "Śr. conf",
                "refresh": "Odśwież",
                "plot_hist": "Histogram",
                "hist_title": "Histogram wystąpień",
                "no_stats": "Brak danych (uruchom wideo).",
                "pause": "Pauza",
                "resume": "Wznów",

                "lang_toggle": "Zmień język"
            },
            "en": {
                "select_model": "Select YOLO models",
                "surface_model": "Surface model",
                "line_model": "Line model",
                "current_surface_model": "Current surface model: ",
                "current_line_model": "Current line model: ",
                "no_model": "NONE",

                "select_video_conf": "Select MP4 file and set confidence threshold",
                "confidence": "Confidence",
                "choose_video": "Choose video",
                "save_csv": "Save results to CSV",

                "model_loaded": "Model loaded",
                "model_loaded_msg": "Successfully loaded model:\n{}",
                "model_error": "Error",
                "model_error_msg": "Failed to load model:\n{}",


                "no_model_title": "No model",
                "no_model_msg": "No YOLO model selected!\nPlease select at least one model before analysis.",

                "done": "Done",
                "done_msg": "Analysis completed. You can now save the results to CSV.",

                "no_data": "No data",
                "no_data_msg": "No data to save. Please analyze a video first.",

                "saved": "Saved",
                "saved_msg": "Results saved to:\n{}",
                "save_error": "Error",
                "save_error_msg": "Failed to save file:\n{}",


                "analysis_results": "Analysis results",
                "stats_title": "Live analysis results",
                "model_type": "Type",
                "class": "Class",
                "count": "Count",
                "avg_conf": "Avg conf",
                "refresh": "Refresh",
                "plot_hist": "Histogram",
                "hist_title": "Occurrences histogram",
                "no_stats": "No data (run video).",
                "pause": "Pause",
                "resume": "Resume",

                "lang_toggle": "Switch language"
            }
        }

        self.class_name_translations = {
            "pl": {
                "red_line": "czerwona linia",
                "white_line": "biala linia",
                "yellow_line": "zolta linia",
                "asphalt": "asfalt",
                "grass": "trawa",
                "concrete": "beton",
                "paving stones": "kostka brukowa",
            },
            "en": {
                "asfalt": "asphalt",
                "trawa": "grass",
                "beton": "concrete",
                "kostka": "paving stones",
                "red_line": "red line",
                "white_line": "white line",
                "yellow_line": "yellow line",
            }
        }

        # --- GUI (w self.controls) ---
        self.model_label = Label(self.controls, text=self.tr("select_model"))
        self.model_label.pack(pady=10)

        models_row = Frame(self.controls)
        models_row.pack(pady=5)

        surface_col = Frame(models_row)
        surface_col.pack(side="left", padx=15)

        Label(surface_col, text="Backend").pack()
        self.surface_backend_combo = ttk.Combobox(
            surface_col,
            textvariable=self.surface_backend_var,
            values=["YOLO", "UNET"],
            state="readonly",
            width=8
        )
        self.surface_backend_combo.pack(pady=2)
        self.surface_backend_combo.bind("<<ComboboxSelected>>", lambda e: self.on_surface_backend_changed())

        self.surface_model_button = Button(surface_col, text=self.tr("surface_model") + " (YOLO)",
                                           command=self.select_surface_model)
        self.surface_model_button.pack(pady=5)

        self.surface_model_name_label = Label(
            surface_col,
            text=f"{self.tr('current_surface_model')}{self.tr('no_model')}",
            fg="gray",
            wraplength=200,
            justify="left"
        )
        self.surface_model_name_label.pack()

        line_col = Frame(models_row)
        line_col.pack(side="left", padx=15)

        Label(line_col, text="Backend").pack()
        self.line_backend_combo = ttk.Combobox(
            line_col,
            textvariable=self.line_backend_var,
            values=["YOLO", "UNET"],
            state="readonly",
            width=8
        )
        self.line_backend_combo.pack(pady=2)
        self.line_backend_combo.bind("<<ComboboxSelected>>", lambda e: self.on_line_backend_changed())

        self.line_model_button = Button(line_col, text=self.tr("line_model") + " (YOLO)",
                                        command=self.select_line_model)
        self.line_model_button.pack(pady=5)

        self.line_model_name_label = Label(
            line_col,
            text=f"{self.tr('current_line_model')}{self.tr('no_model')}",
            fg="gray",
            wraplength=200,
            justify="left"
        )
        self.line_model_name_label.pack()

        sep1 = Frame(self.controls, bg="black", height=2, width=420)
        sep1.pack(pady=12)
        sep1.pack_propagate(False)

        self.video_label = Label(self.controls, text=self.tr("select_video_conf"))
        self.video_label.pack(pady=10)

        self.conf_label = Label(self.controls, text=f"{self.tr('confidence')}: 0.25")
        self.conf_label.pack()

        self.conf_slider = Scale(self.controls, from_=5, to=95, orient=HORIZONTAL,
                                 command=self.update_conf_label)
        self.conf_slider.set(25)
        self.conf_slider.pack()

        self.video_button = Button(self.controls, text=self.tr("choose_video"),
                                   command=self.select_video)
        self.video_button.pack(pady=10)
        self.pause_button = Button(self.controls, text=self.tr("pause"),
                                   command=self.toggle_pause, state="disabled")
        self.pause_button.pack(pady=5)

        sep2 = Frame(self.controls, bg="black", height=2, width=420)
        sep2.pack(pady=12)
        sep2.pack_propagate(False)

        self.save_csv_button = Button(self.controls, text=self.tr("save_csv"),
                                      command=self.save_csv)
        self.save_csv_button.pack(pady=10)

        self.lang_button = Button(self.controls, text=self.tr("lang_toggle"),
                                  command=self.toggle_language)
        self.lang_button.pack(pady=5)

        self.results_button = Button(self.controls, text=self.tr("analysis_results"),
                                     command=self.open_results_window)
        self.results_button.pack(pady=10)

        # stan
        self.running = False
        self.frame_count = 0
        self.process_stride = 1
        self.conf_value = 0.25

        self.window.bind("<KeyPress-q>", lambda e: self.stop_processing())
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------- UI / tłumaczenia ----------
    def tr(self, key):
        return self.translations[self.lang].get(key, key)

    def toggle_language(self):
        self.lang = "en" if self.lang == "pl" else "pl"
        self.update_labels()

    def translate_class_name(self, name):
        return self.class_name_translations.get(self.lang, {}).get(name, name)

    def update_labels(self):
        self.model_label.config(text=self.tr("select_model"))
        self.surface_model_button.config(text=self.tr("surface_model"))
        self.line_model_button.config(text=self.tr("line_model"))

        surface_name = self.surface_model_name_label.cget("text").split(":", 1)[-1].strip()
        line_name = self.line_model_name_label.cget("text").split(":", 1)[-1].strip()

        self.surface_model_name_label.config(
            text=f"{self.tr('current_surface_model')}{surface_name if self.surface_model else self.tr('no_model')}"
        )
        self.line_model_name_label.config(
            text=f"{self.tr('current_line_model')}{line_name if self.line_model else self.tr('no_model')}"
        )

        self.video_label.config(text=self.tr("select_video_conf"))
        self.conf_label.config(text=f"{self.tr('confidence')}: {self.conf_slider.get() / 100:.2f}")
        self.video_button.config(text=self.tr("choose_video"))
        self.save_csv_button.config(text=self.tr("save_csv"))
        self.lang_button.config(text=self.tr("lang_toggle"))
        self.results_button.config(text=self.tr("analysis_results"))

        if self.results_win is not None and self.results_win.winfo_exists():
            self.results_win.title(self.tr("stats_title"))
            self.update_results_window()

        if self.hist_win is not None and self.hist_win.winfo_exists():
            self.hist_win.title(self.tr("hist_title"))
            self.update_histogram()

        self.pause_button.config(text=self.tr("resume") if self.paused else self.tr("pause"))

        self.on_surface_backend_changed()
        self.on_line_backend_changed()

    # ---------- kolory ----------
    def get_class_color(self, cls_id):
        random.seed(cls_id)
        return tuple(random.randint(0, 255) for _ in range(3))

    def get_line_color_by_name(self, class_name):
        if class_name == "yellow_line":
            return (0, 255, 255)
        if class_name == "white_line":
            return (200, 200, 200)
        if class_name == "red_line":
            return (0, 0, 255)
        return (0, 255, 0)

    def update_conf_label(self, val):
        self.conf_label.config(text=f"{self.tr('confidence')}: {int(val) / 100:.2f}")

    def tr_model_type(self, t):
        if self.lang == "pl":
            return {"surface": "nawierzchnia", "line": "linia"}.get(t, t)
        return {"surface": "surface", "line": "line"}.get(t, t)

    def line_ready(self):
        if self.line_backend_var.get() == "YOLO":
            return self.line_model is not None
        return self.line_unet_model is not None

    def on_line_backend_changed(self):
        backend = self.line_backend_var.get()

        if backend == "YOLO":
            self.line_model_button.config(text=self.tr("line_model") + " (YOLO)")
            name = self.line_yolo_name if self.line_model is not None else self.tr("no_model")
            self.line_model_name_label.config(
                text=f"{self.tr('current_line_model')}{name}",
                fg="black" if self.line_model is not None else "gray"
            )
        else:
            self.line_model_button.config(text=self.tr("line_model") + " (U-Net)")
            name = self.line_unet_name if self.line_unet_model is not None else self.tr("no_model")
            self.line_model_name_label.config(
                text=f"{self.tr('current_line_model')}{name}",
                fg="black" if self.line_unet_model is not None else "gray"
            )

    def load_line_unet_weights(self, weights_path: str):
        model = UNet(in_channels=3, num_classes=self.line_unet_num_classes, base=self.line_unet_base).to(
            self.line_unet_device)

        state = torch.load(weights_path, map_location=self.line_unet_device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

        model.load_state_dict(state)
        model.eval()

        self.line_unet_model = model
        self.line_unet_name = os.path.basename(weights_path)

    def surface_ready(self):
        """Czy wybrany backend nawierzchni ma załadowany model."""
        if self.surface_backend_var.get() == "YOLO":
            return self.surface_model is not None
        return self.unet_model is not None

    def on_surface_backend_changed(self):
        """Aktualizacja UI po zmianie YOLO/UNET."""
        backend = self.surface_backend_var.get()

        if backend == "YOLO":
            self.surface_model_button.config(text=self.tr("surface_model") + " (YOLO)")
            name = self.surface_yolo_name if self.surface_model is not None else self.tr("no_model")
            self.surface_model_name_label.config(
                text=f"{self.tr('current_surface_model')}{name}",
                fg="black" if self.surface_model is not None else "gray"
            )
        else:
            self.surface_model_button.config(text=self.tr("surface_model") + " (U-Net)")
            name = self.surface_unet_name if self.unet_model is not None else self.tr("no_model")
            self.surface_model_name_label.config(
                text=f"{self.tr('current_surface_model')}{name}",
                fg="black" if self.unet_model is not None else "gray"
            )

    def load_unet_weights(self, weights_path: str):
        """Ładuje U-Net (.pth) na CPU/GPU."""
        model = UNet(in_channels=3, num_classes=self.unet_num_classes, base=self.unet_base).to(self.unet_device)

        state = torch.load(weights_path, map_location=self.unet_device)
        # obsługa różnych formatów checkpointów
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

        model.load_state_dict(state)
        model.eval()

        self.unet_model = model
        self.surface_unet_name = os.path.basename(weights_path)

    # ---------- thread-safe kopie danych ----------
    def copy_stats(self):
        with self.stats_lock:
            stats = {k: {"count": v["count"], "conf_sum": v["conf_sum"]} for k, v in self.class_stats.items()}
            fc = self.frame_count
        return stats, fc

    def bump_stat(self, key, conf):
        s = self.class_stats.setdefault(key, {"count": 0, "conf_sum": 0.0})
        s["count"] += 1
        s["conf_sum"] += float(conf)

    # ---------- wybór modeli ----------
    def select_surface_model(self):
        backend = self.surface_backend_var.get()

        if backend == "YOLO":
            model_path = filedialog.askopenfilename(filetypes=[("YOLO model files", "*.pt")])
            if model_path:
                try:
                    self.surface_model = YOLO(model_path)
                    self.surface_yolo_name = os.path.basename(model_path)
                    self.on_surface_backend_changed()
                    messagebox.showinfo(self.tr("model_loaded"),
                                        self.tr("model_loaded_msg").format(self.surface_yolo_name))
                except Exception as e:
                    messagebox.showerror(self.tr("model_error"), self.tr("model_error_msg").format(e))
            return

        # UNET
        weights_path = filedialog.askopenfilename(filetypes=[("U-Net weights", "*.pth")])
        if weights_path:
            try:
                self.load_unet_weights(weights_path)
                self.on_surface_backend_changed()
                messagebox.showinfo(self.tr("model_loaded"), self.tr("model_loaded_msg").format(self.surface_unet_name))
            except Exception as e:
                messagebox.showerror(self.tr("model_error"), self.tr("model_error_msg").format(e))

    def select_line_model(self):
        backend = self.line_backend_var.get()

        if backend == "YOLO":
            model_path = filedialog.askopenfilename(filetypes=[("YOLO model files", "*.pt")])
            if model_path:
                try:
                    self.line_model = YOLO(model_path)
                    self.line_yolo_name = os.path.basename(model_path)
                    self.on_line_backend_changed()
                    messagebox.showinfo(self.tr("model_loaded"),
                                        self.tr("model_loaded_msg").format(self.line_yolo_name))
                except Exception as e:
                    messagebox.showerror(self.tr("model_error"), self.tr("model_error_msg").format(e))
            return

        # UNET
        weights_path = filedialog.askopenfilename(filetypes=[("U-Net weights", "*.pth")])
        if weights_path:
            try:
                self.load_line_unet_weights(weights_path)
                self.on_line_backend_changed()
                messagebox.showinfo(self.tr("model_loaded"), self.tr("model_loaded_msg").format(self.line_unet_name))
            except Exception as e:
                messagebox.showerror(self.tr("model_error"), self.tr("model_error_msg").format(e))

    # ---------- START wideo (wątek + pętla wyświetlania) ----------
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

        # start worker
        self.worker_thread = threading.Thread(target=self.worker_loop, args=(video_path,), daemon=True)
        self.worker_thread.start()

        # start display loop (Tk)
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

        # pokaż najnowszą klatkę jeśli jest
        try:
            frame = self.frame_queue.get_nowait()
            self.show_frame_in_tk(frame)
        except queue.Empty:
            pass

        # zakończenie
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
                    # nie pokazuj "gołej" klatki -> brak migania masek
                    frame_vis = last_annotated if last_annotated is not None else frame

                with self.stats_lock:
                    self.frame_count = frame_idx

                # wrzuć tylko najnowszą klatkę (drop starych)
                try:
                    while True:
                        _ = self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                self.frame_queue.put(frame_vis)

                # pacing do fps jeśli zdążymy (jak YOLO wolny, to i tak nie nadąży)
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

    # ---------- wyświetlanie ----------
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

    # ---------- DETEKCJA ----------

    def run_unet_line_on_frame(self, frame_infer_bgr, frame_draw_bgr, frame_idx, roi_mask=None):
        """
        U-Net linie:
          - inferencja na surowym frame_infer_bgr (bez overlayu nawierzchni!)
          - rysowanie na frame_draw_bgr
          - opcjonalne ograniczenie do ROI nawierzchni (roi_mask 0/1)
        """
        if self.line_unet_model is None:
            return frame_draw_bgr

        h, w = frame_draw_bgr.shape[:2]
        overlay_line = frame_draw_bgr.copy()

        rgb = cv2.cvtColor(frame_infer_bgr, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(rgb, (self.line_unet_img_size, self.line_unet_img_size), interpolation=cv2.INTER_LINEAR)

        x = torch.from_numpy(inp).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        x = x.to(self.line_unet_device, non_blocking=True)

        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(self.line_unet_device.type == "cuda")):
            logits = self.line_unet_model(x)
            probs = F.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)

        pred = pred.squeeze(0).to("cpu").numpy().astype(np.uint8)
        conf = conf.squeeze(0).to("cpu").numpy().astype(np.float32)

        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
        conf = cv2.resize(conf, (w, h), interpolation=cv2.INTER_LINEAR)

        valid = (pred > 0) & (conf >= self.conf_value)

        roi_dilated = None
        if roi_mask is not None:
            roi = roi_mask.astype(np.uint8)
            k = max(1, int(self.surface_roi_dilate))
            roi_dilated = cv2.dilate(roi, np.ones((k, k), np.uint8), iterations=1)
            valid = valid & (roi_dilated.astype(bool))

        # overlay: tylko tam gdzie conf >= próg (i w ROI jeśli jest)
        for cls_id in range(1, self.line_unet_num_classes):
            class_name = self.line_unet_names.get(cls_id, str(cls_id))
            color = self.get_line_color_by_name(class_name)
            cls_mask = valid & (pred == cls_id)
            if not cls_mask.any():
                continue
            for c in range(3):
                overlay_line[:, :, c][cls_mask] = color[c]

        frame_draw_bgr = cv2.addWeighted(overlay_line, 0.4, frame_draw_bgr, 0.6, 0)

        # etykiety % na spójnych obszarach (jak w YOLO)
        for cls_id in range(1, self.line_unet_num_classes):
            class_name = self.line_unet_names.get(cls_id, str(cls_id))
            color = self.get_line_color_by_name(class_name)

            cls_mask = ((pred == cls_id) & (conf >= self.conf_value))
            if roi_dilated is not None:
                cls_mask = cls_mask & (roi_dilated.astype(bool))
            cls_mask = cls_mask.astype(np.uint8)

            if cls_mask.max() == 0:
                continue

            num, labels, stats, centroids = cv2.connectedComponentsWithStats(cls_mask, connectivity=8)

            best_cid = None
            best_area = 0
            for cid in range(1, num):
                area = int(stats[cid, cv2.CC_STAT_AREA])
                if area > best_area:
                    best_area = area
                    best_cid = cid

            if best_cid is None or best_area < self.line_unet_min_area:
                continue

            region = (labels == best_cid)
            region_conf = float(conf[region].mean())
            if region_conf < self.conf_value:
                continue

            reg_mask = (region.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(reg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                biggest = max(contours, key=cv2.contourArea)
                cv2.drawContours(frame_draw_bgr, [biggest], -1, color, 2)

            cx, cy = centroids[best_cid]
            translated_name = self.translate_class_name(class_name)
            label = f"{translated_name}: {region_conf:.2%}"
            cv2.putText(frame_draw_bgr, label, (int(cx), int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            key = f"line:{class_name}"
            with self.stats_lock:
                self.class_confidences.setdefault(key, []).append(region_conf)
                self.detections.append((frame_idx, "line", class_name, round(region_conf, 4)))
                self.bump_stat(key, region_conf)

        return frame_draw_bgr

    def run_unet_surface_on_frame(self, frame_vis, frame_idx):
        """Segmentacja nawierzchni U-Net + overlay + etykiety % jak w YOLO."""
        if self.unet_model is None:
            self.last_surface_roi = None
            return frame_vis

        h, w = frame_vis.shape[:2]
        overlay_surface = frame_vis.copy()

        rgb = cv2.cvtColor(frame_vis, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(rgb, (self.unet_img_size, self.unet_img_size), interpolation=cv2.INTER_LINEAR)

        x = torch.from_numpy(inp).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        x = x.to(self.unet_device, non_blocking=True)

        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(self.unet_device.type == "cuda")):
            logits = self.unet_model(x)  # [1,C,H,W]
            probs = F.softmax(logits, dim=1)  # [1,C,H,W]
            conf, pred = torch.max(probs, dim=1)  # [1,H,W], [1,H,W]

        pred = pred.squeeze(0).to("cpu").numpy().astype(np.uint8)
        conf = conf.squeeze(0).to("cpu").numpy().astype(np.float32)

        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
        conf = cv2.resize(conf, (w, h), interpolation=cv2.INTER_LINEAR)

        # próg confidence działa też dla U-Net (jak w YOLO)
        valid = (pred > 0) & (conf >= self.conf_value)

        # --- ROI nawierzchni dla linii (union wszystkich klas nawierzchni) ---
        roi = valid.astype(np.uint8)
        self.last_surface_roi = roi if roi.any() else None

        # overlay kolorów
        color_mask = self.unet_colors_bgr[pred]  # (H,W,3) BGR
        overlay_surface[valid] = color_mask[valid]
        frame_vis = cv2.addWeighted(overlay_surface, 0.4, frame_vis, 0.6, 0)

        # etykiety % na spójnych obszarach (connected components) — “jak YOLO”
        for cls_id in range(1, self.unet_num_classes):
            cls_mask = ((pred == cls_id) & (conf >= self.conf_value)).astype(np.uint8)
            if cls_mask.max() == 0:
                continue

            num, labels, stats, centroids = cv2.connectedComponentsWithStats(cls_mask, connectivity=8)

            for cid in range(1, num):
                area = int(stats[cid, cv2.CC_STAT_AREA])
                if area < self.unet_min_area:
                    continue

                region = (labels == cid)
                region_conf = float(conf[region].mean())
                if region_conf < self.conf_value:
                    continue

                reg_mask = (region.astype(np.uint8) * 255)
                contours, _ = cv2.findContours(reg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                color = tuple(int(c) for c in self.unet_colors_bgr[cls_id])
                if contours:
                    cv2.drawContours(frame_vis, contours, -1, color, 2)

                cx, cy = centroids[cid]
                class_name = self.unet_names.get(cls_id, str(cls_id))
                translated_name = self.translate_class_name(class_name)
                label = f"{translated_name}: {region_conf:.2%}"
                cv2.putText(frame_vis, label, (int(cx), int(cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                key = f"surface:{class_name}"
                with self.stats_lock:
                    self.class_confidences.setdefault(key, []).append(region_conf)
                    self.detections.append((frame_idx, "surface", class_name, round(region_conf, 4)))
                    self.bump_stat(key, region_conf)

        return frame_vis

    def run_detection_on_frame(self, frame, frame_idx):
        frame_vis = frame.copy()
        height, width = frame.shape[:2]

        # reset ROI per klatka
        self.last_surface_roi = None

        # 1) nawierzchnia (YOLO albo UNET)
        if self.surface_backend_var.get() == "YOLO":
            if self.surface_model is not None:
                overlay_surface = frame_vis.copy()
                surface_roi = np.zeros((height, width), dtype=np.uint8)

                results_surface = self.surface_model(frame, conf=self.conf_value, device=self.yolo_device,
                                                     verbose=False)

                for r in results_surface:
                    if hasattr(r, "masks") and r.masks is not None and r.boxes is not None:
                        masks = r.masks.data.cpu().numpy()
                        n = min(len(masks), len(r.boxes))

                        for idx in range(n):
                            mask = masks[idx]
                            cls_id = int(r.boxes.cls[idx].item())
                            conf = float(r.boxes.conf[idx].item())
                            class_name = self.surface_model.names.get(cls_id, str(cls_id))
                            key = f"surface:{class_name}"

                            color = self.get_class_color(cls_id)
                            mask_resized = cv2.resize((mask * 255).astype("uint8"), (width, height))

                            # ROI union
                            surface_roi[mask_resized > 0] = 1

                            for c in range(3):
                                overlay_surface[:, :, c][mask_resized > 0] = color[c]

                            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(frame_vis, contours, -1, color, 2)

                            if contours:
                                biggest = max(contours, key=cv2.contourArea)
                                M = cv2.moments(biggest)
                                if M["m00"] != 0:
                                    cx = int(M["m10"] / M["m00"])
                                    cy = int(M["m01"] / M["m00"])
                                    translated_name = self.translate_class_name(class_name)
                                    label = f"{translated_name}: {conf:.2%}"
                                    cv2.putText(frame_vis, label, (cx, cy),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                            with self.stats_lock:
                                self.class_confidences.setdefault(key, []).append(conf)
                                self.detections.append((frame_idx, "surface", class_name, round(conf, 4)))
                                self.bump_stat(key, conf)

                    else:
                        if r.boxes is None:
                            continue
                        for box in r.boxes:
                            cls_id = int(box.cls[0].item())
                            conf = float(box.conf[0].item())
                            class_name = self.surface_model.names.get(cls_id, str(cls_id))
                            key = f"surface:{class_name}"

                            color = self.get_class_color(cls_id)
                            x1, y1, x2, y2 = map(int, box.xyxy[0])

                            # ROI union (dla boxów)
                            x1c, y1c = max(0, x1), max(0, y1)
                            x2c, y2c = min(width, x2), min(height, y2)
                            surface_roi[y1c:y2c, x1c:x2c] = 1

                            cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color, 2)

                            translated_name = self.translate_class_name(class_name)
                            label = f"{translated_name}: {conf:.2%}"
                            cv2.putText(frame_vis, label, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                            with self.stats_lock:
                                self.class_confidences.setdefault(key, []).append(conf)
                                self.detections.append((frame_idx, "surface", class_name, round(conf, 4)))
                                self.bump_stat(key, conf)

                frame_vis = cv2.addWeighted(overlay_surface, 0.4, frame_vis, 0.6, 0)

                # zapisz ROI nawierzchni dla linii
                self.last_surface_roi = surface_roi if surface_roi.any() else None

        else:
            # UNET
            frame_vis = self.run_unet_surface_on_frame(frame_vis, frame_idx)

        # 2) linie (YOLO albo UNET)
        if self.line_backend_var.get() == "YOLO":
            if self.line_model is not None:
                overlay_line = frame_vis.copy()
                results_line = self.line_model(frame, conf=self.conf_value, device=self.yolo_device, verbose=False)

                roi = self.last_surface_roi
                roi255 = (roi * 255).astype(np.uint8) if roi is not None else None

                for r in results_line:
                    if hasattr(r, "masks") and r.masks is not None and r.boxes is not None:
                        masks = r.masks.data.cpu().numpy()
                        n = min(len(masks), len(r.boxes))

                        class_masks = {}
                        class_best_conf = {}

                        for idx in range(n):
                            mask = masks[idx]
                            cls_id = int(r.boxes.cls[idx].item())
                            conf = float(r.boxes.conf[idx].item())
                            class_name = self.line_model.names.get(cls_id, str(cls_id))
                            key = f"line:{class_name}"

                            mask_resized = cv2.resize((mask * 255).astype("uint8"), (width, height))

                            # ogranicz do ROI nawierzchni jeśli jest
                            if roi255 is not None:
                                mask_resized = cv2.bitwise_and(mask_resized, roi255)
                                if mask_resized.max() == 0:
                                    continue

                            if cls_id not in class_masks:
                                class_masks[cls_id] = mask_resized
                                class_best_conf[cls_id] = conf
                            else:
                                class_masks[cls_id] = np.maximum(class_masks[cls_id], mask_resized)
                                class_best_conf[cls_id] = max(class_best_conf[cls_id], conf)

                            with self.stats_lock:
                                self.class_confidences.setdefault(key, []).append(conf)
                                self.detections.append((frame_idx, "line", class_name, round(conf, 4)))
                                self.bump_stat(key, conf)

                        for cls_id, combined_mask in class_masks.items():
                            class_name = self.line_model.names.get(cls_id, str(cls_id))
                            color = self.get_line_color_by_name(class_name)

                            # (na wszelki wypadek) ogranicz combined_mask do ROI
                            if roi255 is not None:
                                combined_mask = cv2.bitwise_and(combined_mask, roi255)
                                if combined_mask.max() == 0:
                                    continue

                            for c in range(3):
                                overlay_line[:, :, c][combined_mask > 0] = color[c]

                            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if not contours:
                                continue

                            biggest = max(contours, key=cv2.contourArea)
                            cv2.drawContours(frame_vis, [biggest], -1, color, 2)

                            M = cv2.moments(biggest)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                translated_name = self.translate_class_name(class_name)
                                conf_best = class_best_conf.get(cls_id, 0.0)
                                label = f"{translated_name}: {conf_best:.2%}"
                                cv2.putText(frame_vis, label, (cx, cy),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    else:
                        if r.boxes is None:
                            continue
                        for box in r.boxes:
                            cls_id = int(box.cls[0].item())
                            conf = float(box.conf[0].item())
                            class_name = self.line_model.names.get(cls_id, str(cls_id))
                            key = f"line:{class_name}"

                            color = self.get_line_color_by_name(class_name)
                            x1, y1, x2, y2 = map(int, box.xyxy[0])

                            # ogranicz box do ROI (jeśli cały box poza ROI -> pomiń)
                            if roi is not None:
                                x1c, y1c = max(0, x1), max(0, y1)
                                x2c, y2c = min(width, x2), min(height, y2)
                                if x2c <= x1c or y2c <= y1c or not roi[y1c:y2c, x1c:x2c].any():
                                    continue

                            cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color, 2)

                            translated_name = self.translate_class_name(class_name)
                            label = f"{translated_name}: {conf:.2%}"
                            cv2.putText(frame_vis, label, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                            with self.stats_lock:
                                self.class_confidences.setdefault(key, []).append(conf)
                                self.detections.append((frame_idx, "line", class_name, round(conf, 4)))
                                self.bump_stat(key, conf)

                frame_vis = cv2.addWeighted(overlay_line, 0.4, frame_vis, 0.6, 0)

        else:
            # UNET (inferencja na surowej klatce "frame", rysowanie na "frame_vis")
            roi = self.last_surface_roi if (self.surface_ready() and self.line_ready()) else None
            frame_vis = self.run_unet_line_on_frame(frame, frame_vis, frame_idx, roi_mask=roi)

        return frame_vis

    # ---------- OKNO WYNIKÓW ----------
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
        self.update_results_window()
        self.schedule_results_update()

    def update_results_window(self):
        if self.results_win is None or not self.results_win.winfo_exists() or self.results_tree is None:
            return

        self.results_tree.heading("type", text=self.tr("model_type"))
        self.results_tree.heading("cls", text=self.tr("class"))
        self.results_tree.heading("count", text=self.tr("count"))
        self.results_tree.heading("avg", text=self.tr("avg_conf"))

        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        stats, fc = self.copy_stats()

        if not stats:
            if self.results_info:
                self.results_info.config(text=self.tr("no_stats"))
            return

        total_det = sum(v["count"] for v in stats.values())
        if self.results_info:
            self.results_info.config(text=f"Klatka: {fc} | Wykrycia: {total_det}" if self.lang == "pl"
                                         else f"Frame: {fc} | Detections: {total_det}")

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

    # ---------- HISTOGRAM ----------
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

    # ---------- CSV ----------
    def save_csv(self):
        with self.stats_lock:
            has_data = bool(self.class_confidences) and bool(self.detections)
            confidences_copy = {k: list(v) for k, v in self.class_confidences.items()}
            detections_copy = list(self.detections)

        if not has_data:
            messagebox.showwarning(self.tr("no_data"), self.tr("no_data_msg"))
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not save_path:
            return

        try:
            with open(save_path, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file, delimiter=";")

                writer.writerow(["# LEGENDA KLAS (format: typ - ID = Nazwa)"])
                # surface legenda zależna od backendu
                if self.surface_backend_var.get() == "YOLO":
                    if self.surface_model is not None:
                        for cls_id, name in self.surface_model.names.items():
                            writer.writerow([f"# surface: {cls_id} = {name}"])
                else:
                    # U-Net: 0=tło, 1..4 klasy
                    writer.writerow([f"# surface: 0 = tlo"])
                    for cls_id in range(1, self.unet_num_classes):
                        writer.writerow([f"# surface: {cls_id} = {self.unet_names.get(cls_id, str(cls_id))}"])

                # line legenda zależna od backendu
                if self.line_backend_var.get() == "YOLO":
                    if self.line_model is not None:
                        for cls_id, name in self.line_model.names.items():
                            writer.writerow([f"# line: {cls_id} = {name}"])
                else:
                    writer.writerow([f"# line: 0 = tlo"])
                    for cls_id in range(1, self.line_unet_num_classes):
                        writer.writerow([f"# line: {cls_id} = {self.line_unet_names.get(cls_id, str(cls_id))}"])

                writer.writerow([])

                writer.writerow(["PODSUMOWANIE"])
                writer.writerow(["Typ modelu", "Klasa", "Średni confidence", "Liczba wykryć"])
                for key, confs in confidences_copy.items():
                    model_type, class_name = key.split(":", 1)
                    avg_conf = sum(confs) / len(confs)
                    writer.writerow([model_type, class_name, round(avg_conf, 4), len(confs)])
                writer.writerow([])

                writer.writerow(["WSZYSTKIE WYKRYCIA"])
                writer.writerow(["Numer klatki", "Typ modelu", "Klasa", "Confidence"])
                for detection in detections_copy:
                    writer.writerow(detection)

            messagebox.showinfo(self.tr("saved"), self.tr("saved_msg").format(save_path))
        except Exception as e:
            messagebox.showerror(self.tr("save_error"), self.tr("save_error_msg").format(e))


if __name__ == "__main__":
    root = Tk()
    app = YoloVideoApp(root)
    root.mainloop()