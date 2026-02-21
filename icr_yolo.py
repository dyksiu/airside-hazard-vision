import os
import cv2
import csv
import random
import numpy as np
from ultralytics import YOLO
from tkinter import Tk, Label, Button, filedialog, Scale, HORIZONTAL, messagebox, Frame


class YoloVideoApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Detektor nawierzchni i linii")
        self.window.geometry("460x500")
        self.window.resizable(False, False)

        # osobne modele
        self.surface_model = None   # model nawierzchni
        self.line_model = None      # model linii

        # dane z analizy
        self.class_confidences = {}
        self.detections = []

        # domyslny jezyk
        self.lang = "pl"

        # stala rozdzielczosc
        self.TARGET_WIDTH = 960
        self.TARGET_HEIGHT = 540

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
                "reset": "Reset",
            },
            "en": {
                "asfalt": "asphalt",
                "trawa": "grass",
                "beton": "concrete",
                "kostka": "paving stones",
                "red_line": "red line",
                "white_line": "white line",
                "yellow_line": "yellow line",
                "reset": "Reset",
            }
        }

        # GUI
        self.model_label = Label(window, text=self.tr("select_model"))
        self.model_label.pack(pady=10)


        models_row = Frame(window)
        models_row.pack(pady=5)


        surface_col = Frame(models_row)
        surface_col.pack(side="left", padx=15)

        self.surface_model_button = Button(surface_col, text=self.tr("surface_model"),
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

        self.line_model_button = Button(line_col, text=self.tr("line_model"), command=self.select_line_model)
        self.line_model_button.pack(pady=5)

        self.line_model_name_label = Label(
            line_col,
            text=f"{self.tr('current_line_model')}{self.tr('no_model')}",
            fg="gray",
            wraplength=200,
            justify="left"
        )
        self.line_model_name_label.pack()


        sep1 = Frame(window, bg="black", height=2, width=420)
        sep1.pack(pady=12)
        sep1.pack_propagate(False)


        self.video_label = Label(window, text=self.tr("select_video_conf"))
        self.video_label.pack(pady=10)

        self.conf_label = Label(window, text=f"{self.tr('confidence')}: 0.25")
        self.conf_label.pack()

        self.conf_slider = Scale(window, from_=5, to=95, orient=HORIZONTAL, command=self.update_conf_label)
        self.conf_slider.set(25)
        self.conf_slider.pack()

        self.video_button = Button(window, text=self.tr("choose_video"), command=self.select_video)
        self.video_button.pack(pady=10)


        sep2 = Frame(window, bg="black", height=2, width=420)
        sep2.pack(pady=12)
        sep2.pack_propagate(False)


        self.save_csv_button = Button(window, text=self.tr("save_csv"), command=self.save_csv)
        self.save_csv_button.pack(pady=10)

        self.lang_button = Button(window, text=self.tr("lang_toggle"), command=self.toggle_language)
        self.lang_button.pack(pady=5)


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


    def select_surface_model(self):
        model_path = filedialog.askopenfilename(filetypes=[("YOLO model files", "*.pt")])
        if model_path:
            try:
                self.surface_model = YOLO(model_path)
                model_name = os.path.basename(model_path)
                self.surface_model_name_label.config(text=f"{self.tr('current_surface_model')}{model_name}")
                messagebox.showinfo(self.tr("model_loaded"), self.tr("model_loaded_msg").format(model_name))
            except Exception as e:
                messagebox.showerror(self.tr("model_error"), self.tr("model_error_msg").format(e))

    def select_line_model(self):
        model_path = filedialog.askopenfilename(filetypes=[("YOLO model files", "*.pt")])
        if model_path:
            try:
                self.line_model = YOLO(model_path)
                model_name = os.path.basename(model_path)
                self.line_model_name_label.config(text=f"{self.tr('current_line_model')}{model_name}")
                messagebox.showinfo(self.tr("model_loaded"), self.tr("model_loaded_msg").format(model_name))
            except Exception as e:
                messagebox.showerror(self.tr("model_error"), self.tr("model_error_msg").format(e))

    def reset_surface_model(self):
        self.surface_model = None
        self.surface_model_name_label.config(
            text=f"{self.tr('current_surface_model')}{self.tr('no_model')}",
            fg="gray"
        )

    def reset_line_model(self):
        self.line_model = None
        self.line_model_name_label.config(
            text=f"{self.tr('current_line_model')}{self.tr('no_model')}",
            fg="gray"
        )



    def select_video(self):
        if self.surface_model is None and self.line_model is None:
            messagebox.showwarning(self.tr("no_model_title"), self.tr("no_model_msg"))
            return

        video_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
        if not video_path:
            return


        self.class_confidences = {}
        self.detections = []

        conf_value = self.conf_slider.get() / 100
        cap = cv2.VideoCapture(video_path)


        process_stride = 2 if (self.surface_model is not None and self.line_model is not None) else 1

        last_frame_vis = None
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (self.TARGET_WIDTH, self.TARGET_HEIGHT))
            height, width = frame.shape[:2]


            if frame_count % process_stride != 0 and last_frame_vis is not None:
                cv2.imshow("YOLOv8 Detection", last_frame_vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                frame_count += 1
                continue

            frame_vis = frame.copy()

            # 1) model nawierzchni
            if self.surface_model is not None:
                overlay_surface = frame_vis.copy()
                results_surface = self.surface_model(frame, conf=conf_value, device=0)

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

                            self.class_confidences.setdefault(key, []).append(conf)
                            self.detections.append((frame_count, "surface", class_name, round(conf, 4)))

                            color = self.get_class_color(cls_id)
                            mask_resized = cv2.resize((mask * 255).astype("uint8"), (width, height))

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
                    else:

                        if r.boxes is None:
                            continue
                        for box in r.boxes:
                            cls_id = int(box.cls[0].item())
                            conf = float(box.conf[0].item())
                            class_name = self.surface_model.names.get(cls_id, str(cls_id))
                            key = f"surface:{class_name}"

                            self.class_confidences.setdefault(key, []).append(conf)
                            self.detections.append((frame_count, "surface", class_name, round(conf, 4)))

                            color = self.get_class_color(cls_id)
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color, 2)

                            translated_name = self.translate_class_name(class_name)
                            label = f"{translated_name}: {conf:.2%}"
                            cv2.putText(frame_vis, label, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                frame_vis = cv2.addWeighted(overlay_surface, 0.4, frame_vis, 0.6, 0)

            # 2) model linii
            if self.line_model is not None:
                overlay_line = frame_vis.copy()
                results_line = self.line_model(frame, conf=conf_value, device=0)

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

                            self.class_confidences.setdefault(key, []).append(conf)
                            self.detections.append((frame_count, "line", class_name, round(conf, 4)))

                            mask_resized = cv2.resize((mask * 255).astype("uint8"), (width, height))

                            if cls_id not in class_masks:
                                class_masks[cls_id] = mask_resized
                                class_best_conf[cls_id] = conf
                            else:
                                class_masks[cls_id] = np.maximum(class_masks[cls_id], mask_resized)
                                class_best_conf[cls_id] = max(class_best_conf[cls_id], conf)

                        for cls_id, combined_mask in class_masks.items():
                            class_name = self.line_model.names.get(cls_id, str(cls_id))
                            color = self.get_line_color_by_name(class_name)

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
                                conf = class_best_conf.get(cls_id, 0.0)
                                label = f"{translated_name}: {conf:.2%}"
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

                            self.class_confidences.setdefault(key, []).append(conf)
                            self.detections.append((frame_count, "line", class_name, round(conf, 4)))

                            color = self.get_line_color_by_name(class_name)
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color, 2)

                            translated_name = self.translate_class_name(class_name)
                            label = f"{translated_name}: {conf:.2%}"
                            cv2.putText(frame_vis, label, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                frame_vis = cv2.addWeighted(overlay_line, 0.4, frame_vis, 0.6, 0)

            # zapamietaj ostatni wynik
            last_frame_vis = frame_vis.copy()

            cv2.imshow("YOLOv8 Detection", frame_vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo(self.tr("done"), self.tr("done_msg"))


    def save_csv(self):
        if not self.class_confidences or not self.detections:
            messagebox.showwarning(self.tr("no_data"), self.tr("no_data_msg"))
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not save_path:
            return

        try:
            with open(save_path, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file, delimiter=";")

                writer.writerow(["# LEGENDA KLAS (format: typ - ID = Nazwa)"])
                if self.surface_model is not None:
                    for cls_id, name in self.surface_model.names.items():
                        writer.writerow([f"# surface: {cls_id} = {name}"])
                if self.line_model is not None:
                    for cls_id, name in self.line_model.names.items():
                        writer.writerow([f"# line: {cls_id} = {name}"])
                writer.writerow([])

                writer.writerow(["PODSUMOWANIE"])
                writer.writerow(["Typ modelu", "Klasa", "Średni confidence", "Liczba wykryć"])
                for key, confs in self.class_confidences.items():
                    model_type, class_name = key.split(":", 1)
                    avg_conf = sum(confs) / len(confs)
                    writer.writerow([model_type, class_name, round(avg_conf, 4), len(confs)])
                writer.writerow([])

                writer.writerow(["WSZYSTKIE WYKRYCIA"])
                writer.writerow(["Numer klatki", "Typ modelu", "Klasa", "Confidence"])
                for detection in self.detections:
                    writer.writerow(detection)

            messagebox.showinfo(self.tr("saved"), self.tr("saved_msg").format(save_path))
        except Exception as e:
            messagebox.showerror(self.tr("save_error"), self.tr("save_error_msg").format(e))


if __name__ == "__main__":
    root = Tk()
    app = YoloVideoApp(root)
    root.mainloop()