import os
from pathlib import Path
import cv2
import random
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from tkinter import filedialog, messagebox

from models.unet import UNet
from core.hailo_runtime import HailoYOLOSegModel


class DetectionMixin:

    def _get_segformer_components(self):
        try:
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        except ImportError as exc:
            raise ImportError(
                "Brakuje biblioteki transformers wymaganej do obsługi SegFormer. "
                "Zainstaluj: pip install transformers"
            ) from exc
        return SegformerForSemanticSegmentation, SegformerImageProcessor

    def _unwrap_loaded_state(self, state):
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        return state

    def _infer_deeplab_model_name(self, state_dict):
        keys = list(state_dict.keys())

        # MobileNetV3 backbone in torchvision DeepLabV3 is wrapped by IntermediateLayerGetter,
        # so keys are typically like: backbone.0.0.weight, backbone.1.block.0.0.weight, ...
        if any(k.startswith("backbone.0.") for k in keys) or any(k.startswith("backbone.1.") for k in keys):
            return "deeplabv3_mobilenet_v3_large"

        # ResNet backbones keep names like backbone.conv1 / backbone.layer1 / backbone.layer2 / ...
        if any(k.startswith("backbone.conv1") for k in keys) or any(k.startswith("backbone.layer") for k in keys):
            if any(k.startswith("backbone.layer3.22") for k in keys):
                return "deeplabv3_resnet101"
            return "deeplabv3_resnet50"

        # Fallback by ASPP input channels if key names were changed by save/export pipeline.
        aspp_w = state_dict.get("classifier.0.convs.0.0.weight")
        if isinstance(aspp_w, torch.Tensor) and aspp_w.ndim == 4:
            in_channels = int(aspp_w.shape[1])
            if in_channels == 960:
                return "deeplabv3_mobilenet_v3_large"
            if in_channels == 2048:
                if any(k.startswith("backbone.layer3.22") for k in keys):
                    return "deeplabv3_resnet101"
                return "deeplabv3_resnet50"

        raise ValueError("Nie udało się rozpoznać architektury DeepLabV3 z pliku wag.")

    def _build_deeplabv3(self, num_classes: int, model_name: str):
        if model_name == "deeplabv3_mobilenet_v3_large":
            from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
            try:
                return deeplabv3_mobilenet_v3_large(
                    weights=None,
                    weights_backbone=None,
                    num_classes=num_classes,
                    aux_loss=False,
                )
            except TypeError:
                try:
                    return deeplabv3_mobilenet_v3_large(
                        pretrained=False,
                        progress=True,
                        num_classes=num_classes,
                        aux_loss=False,
                        pretrained_backbone=False,
                    )
                except TypeError:
                    return deeplabv3_mobilenet_v3_large(
                        pretrained=False,
                        progress=True,
                        num_classes=num_classes,
                        aux_loss=False,
                    )

        if model_name == "deeplabv3_resnet101":
            from torchvision.models.segmentation import deeplabv3_resnet101
            try:
                return deeplabv3_resnet101(
                    weights=None,
                    weights_backbone=None,
                    num_classes=num_classes,
                    aux_loss=False,
                )
            except TypeError:
                try:
                    return deeplabv3_resnet101(
                        pretrained=False,
                        progress=True,
                        num_classes=num_classes,
                        aux_loss=False,
                        pretrained_backbone=False,
                    )
                except TypeError:
                    return deeplabv3_resnet101(
                        pretrained=False,
                        progress=True,
                        num_classes=num_classes,
                        aux_loss=False,
                    )

        from torchvision.models.segmentation import deeplabv3_resnet50
        try:
            return deeplabv3_resnet50(
                weights=None,
                weights_backbone=None,
                num_classes=num_classes,
                aux_loss=False,
            )
        except TypeError:
            try:
                return deeplabv3_resnet50(
                    pretrained=False,
                    progress=True,
                    num_classes=num_classes,
                    aux_loss=False,
                    pretrained_backbone=False,
                )
            except TypeError:
                return deeplabv3_resnet50(
                    pretrained=False,
                    progress=True,
                    num_classes=num_classes,
                    aux_loss=False,
                )

    def _forward_segmentation_logits(self, model, x):
        out = model(x)
        if hasattr(out, "logits"):
            return out.logits
        if isinstance(out, dict):
            return out.get("out", out.get("logits", out))
        return out

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

    def get_surface_color_by_name(self, class_name, cls_id=None):
        normalized = str(class_name).strip().lower()
        if normalized in {"beton", "concrete"}:
            return (0, 0, 255)
        if normalized in {"asfalt", "asphalt"}:
            return (0, 255, 0)
        if normalized in {"kostka", "paving stones", "paving_stones", "kostka brukowa"}:
            return (255, 0, 0)
        if normalized in {"trawa", "grass"}:
            return (0, 255, 255)
        fallback_id = int(cls_id) if cls_id is not None else abs(hash(normalized)) % 256
        return self.get_class_color(fallback_id)

    def _get_surface_display_name(self):
        return getattr(self, "current_surface_name", None) or getattr(self, "cached_surface_name", None)

    def should_update_surface_model(self, frame_idx):
        if not self.surface_ready():
            return False

        # Gdy działa już globalne pomijanie klatek (process_stride > 1),
        # nie dokładamy drugiego poziomu pomijania tylko dla nawierzchni.
        # W przeciwnym razie detekcje nawierzchni wypadają zbyt rzadko,
        # przez co śledzenie, statystyki i wizualizacja stają się niestabilne.
        if self.line_ready() and int(getattr(self, "process_stride", 1) or 1) > 1:
            return True

        stride = max(1, int(getattr(self, "surface_update_stride", 1) or 1))
        if stride <= 1:
            return True

        if not self.line_ready():
            return True

        if getattr(self, "last_surface_roi", None) is None:
            return True

        return (int(frame_idx) % stride) == 0

    def _get_surface_display_roi(self):
        surface_roi = getattr(self, "display_surface_roi", None)
        if surface_roi is not None:
            return surface_roi
        return None

    def draw_cached_surface_overlay(self, frame_draw_bgr):
        surface_roi = self._get_surface_display_roi()
        if surface_roi is None:
            return frame_draw_bgr

        mask = surface_roi.astype(bool)
        if not mask.any():
            return frame_draw_bgr

        surface_name = self._get_surface_display_name()
        if not surface_name:
            return frame_draw_bgr

        color = self.get_surface_color_by_name(surface_name)
        overlay = frame_draw_bgr.copy()
        for c in range(3):
            overlay[:, :, c][mask] = color[c]
        return cv2.addWeighted(overlay, 0.18, frame_draw_bgr, 0.82, 0)

    def draw_current_surface_status(self, frame_draw_bgr):
        surface_name = self._get_surface_display_name()
        if not surface_name:
            return frame_draw_bgr

        translated_name = self.translate_class_name(surface_name)
        prefix = "Nawierzchnia" if getattr(self, "lang", "pl") == "pl" else "Surface"
        text = f"{prefix}: {translated_name}"
        color = self.get_surface_color_by_name(surface_name)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

        x = 12
        y = 64
        box_w = tw + 20
        box_h = th + 18

        overlay = frame_draw_bgr.copy()
        cv2.rectangle(overlay, (x - 8, y - th - 10), (x - 8 + box_w, y - th - 10 + box_h), (0, 0, 0), -1)
        cv2.rectangle(overlay, (x - 8, y - th - 10), (x - 8 + box_w, y - th - 10 + box_h), color, 2)
        frame_draw_bgr = cv2.addWeighted(overlay, 0.45, frame_draw_bgr, 0.55, 0)

        cv2.putText(frame_draw_bgr, text, (x, y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame_draw_bgr, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return frame_draw_bgr

    def load_line_unet_weights(self, weights_path: str):
        model = UNet(
            in_channels=3,
            num_classes=self.line_unet_num_classes,
            base=self.line_unet_base
        ).to(self.line_unet_device)

        state = torch.load(weights_path, map_location=self.line_unet_device)
        state = self._unwrap_loaded_state(state)

        model.load_state_dict(state)
        model.eval()

        self.line_unet_model = model
        self.line_unet_name = os.path.basename(weights_path)

    def load_unet_weights(self, weights_path: str):
        model = UNet(
            in_channels=3,
            num_classes=self.unet_num_classes,
            base=self.unet_base
        ).to(self.unet_device)

        state = torch.load(weights_path, map_location=self.unet_device)
        state = self._unwrap_loaded_state(state)

        model.load_state_dict(state)
        model.eval()

        self.unet_model = model
        self.surface_unet_name = os.path.basename(weights_path)

    def load_line_deeplab_weights(self, weights_path: str):
        state = torch.load(weights_path, map_location=self.line_deeplab_device)
        state = self._unwrap_loaded_state(state)
        model_name = self._infer_deeplab_model_name(state)

        model = self._build_deeplabv3(self.line_deeplab_num_classes, model_name).to(self.line_deeplab_device)
        model.load_state_dict(state)
        model.eval()

        self.line_deeplab_model = model
        self.line_deeplab_name = os.path.basename(weights_path)
        self.line_deeplab_arch = model_name

    def load_surface_deeplab_weights(self, weights_path: str):
        state = torch.load(weights_path, map_location=self.deeplab_device)
        state = self._unwrap_loaded_state(state)
        model_name = self._infer_deeplab_model_name(state)

        model = self._build_deeplabv3(self.deeplab_num_classes, model_name).to(self.deeplab_device)
        model.load_state_dict(state)
        model.eval()

        self.surface_deeplab_model = model
        self.surface_deeplab_name = os.path.basename(weights_path)
        self.surface_deeplab_arch = model_name


    def _extract_segformer_label_info(self, model, fallback_num_classes, fallback_names):
        num_labels = int(getattr(model.config, "num_labels", fallback_num_classes))
        id2label = getattr(model.config, "id2label", None) or {}
        names = {}
        for idx in range(num_labels):
            label = id2label.get(idx, id2label.get(str(idx), fallback_names.get(idx, str(idx))))
            names[idx] = str(label)
        return num_labels, names

    def load_surface_segformer_model(self, model_dir: str):
        SegformerForSemanticSegmentation, SegformerImageProcessor = self._get_segformer_components()
        processor = SegformerImageProcessor.from_pretrained(model_dir)
        model = SegformerForSemanticSegmentation.from_pretrained(model_dir).to(self.segformer_device)
        model.eval()

        self.surface_segformer_model = model
        self.surface_segformer_processor = processor
        self.surface_segformer_name = os.path.basename(os.path.normpath(model_dir))
        self.segformer_num_classes, self.segformer_names = self._extract_segformer_label_info(
            model,
            self.segformer_num_classes,
            self.segformer_names,
        )

    def load_line_segformer_model(self, model_dir: str):
        SegformerForSemanticSegmentation, SegformerImageProcessor = self._get_segformer_components()
        processor = SegformerImageProcessor.from_pretrained(model_dir)
        model = SegformerForSemanticSegmentation.from_pretrained(model_dir).to(self.line_segformer_device)
        model.eval()

        self.line_segformer_model = model
        self.line_segformer_processor = processor
        self.line_segformer_name = os.path.basename(os.path.normpath(model_dir))
        self.line_segformer_num_classes, self.line_segformer_names = self._extract_segformer_label_info(
            model,
            self.line_segformer_num_classes,
            self.line_segformer_names,
        )



    def _default_hailo_labels(self, role: str):
        if role == "line":
            return [self.line_unet_names[k] for k in sorted(self.line_unet_names)]
        return [self.unet_names[k] for k in sorted(self.unet_names)]

    def _find_hailo_labels_path(self, hef_path: str):
        path = Path(hef_path)
        candidates = [
            path.with_suffix(".labels.txt"),
            path.with_suffix(".labels"),
            path.with_suffix(".txt"),
            path.parent / "labels.txt",
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return str(candidate)
        return None

    def load_surface_hailo_model(self, hef_path: str):
        labels_path = self._find_hailo_labels_path(hef_path)
        labels = labels_path or self._default_hailo_labels("surface")
        model = HailoYOLOSegModel(
            hef_path,
            labels=labels,
            default_labels=self._default_hailo_labels("surface"),
        )
        self.surface_hailo_model = model
        self.surface_hailo_name = os.path.basename(hef_path)

    def load_line_hailo_model(self, hef_path: str):
        labels_path = self._find_hailo_labels_path(hef_path)
        labels = labels_path or self._default_hailo_labels("line")
        model = HailoYOLOSegModel(
            hef_path,
            labels=labels,
            default_labels=self._default_hailo_labels("line"),
        )
        self.line_hailo_model = model
        self.line_hailo_name = os.path.basename(hef_path)

    def select_surface_model(self):
        backend = self.surface_backend_var.get()
        if backend == "YOLO":
            model_path = filedialog.askopenfilename(filetypes=[("YOLO model files", "*.pt")])
            if model_path:
                try:
                    self.surface_model = YOLO(model_path)
                    self.surface_yolo_name = os.path.basename(model_path)
                    self.on_surface_backend_changed()
                    messagebox.showinfo(
                        self.tr("model_loaded"),
                        self.tr("model_loaded_msg").format(self.surface_yolo_name)
                    )
                except Exception as e:
                    messagebox.showerror(self.tr("model_error"), self.tr("model_error_msg").format(e))
            return

        if backend == "HAILO":
            model_path = filedialog.askopenfilename(filetypes=[("Hailo model files", "*.hef")])
            if model_path:
                try:
                    self.load_surface_hailo_model(model_path)
                    self.on_surface_backend_changed()
                    messagebox.showinfo(
                        self.tr("model_loaded"),
                        self.tr("model_loaded_msg").format(self.surface_hailo_name)
                    )
                except Exception as e:
                    messagebox.showerror(self.tr("model_error"), self.tr("model_error_msg").format(e))
            return

        if backend == "UNET":
            weights_path = filedialog.askopenfilename(filetypes=[("U-Net weights", "*.pth *.pt"), ("PyTorch weights", "*.pth *.pt")])
            if weights_path:
                try:
                    self.load_unet_weights(weights_path)
                    self.on_surface_backend_changed()
                    messagebox.showinfo(
                        self.tr("model_loaded"),
                        self.tr("model_loaded_msg").format(self.surface_unet_name)
                    )
                except Exception as e:
                    messagebox.showerror(self.tr("model_error"), self.tr("model_error_msg").format(e))
            return

        if backend == "SEGFORMER":
            model_dir = filedialog.askdirectory(title="Wybierz katalog modelu SegFormer")
            if model_dir:
                try:
                    self.load_surface_segformer_model(model_dir)
                    self.on_surface_backend_changed()
                    messagebox.showinfo(
                        self.tr("model_loaded"),
                        self.tr("model_loaded_msg").format(self.surface_segformer_name)
                    )
                except Exception as e:
                    messagebox.showerror(self.tr("model_error"), self.tr("model_error_msg").format(e))
            return

        weights_path = filedialog.askopenfilename(filetypes=[("DeepLabV3 weights", "*.pt *.pth"), ("PyTorch weights", "*.pt *.pth")])
        if weights_path:
            try:
                self.load_surface_deeplab_weights(weights_path)
                self.on_surface_backend_changed()
                loaded_name = self.surface_deeplab_name
                if self.surface_deeplab_arch:
                    loaded_name = f"{loaded_name} ({self.surface_deeplab_arch})"
                messagebox.showinfo(
                    self.tr("model_loaded"),
                    self.tr("model_loaded_msg").format(loaded_name)
                )
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
                    messagebox.showinfo(
                        self.tr("model_loaded"),
                        self.tr("model_loaded_msg").format(self.line_yolo_name)
                    )
                except Exception as e:
                    messagebox.showerror(self.tr("model_error"), self.tr("model_error_msg").format(e))
            return

        if backend == "HAILO":
            model_path = filedialog.askopenfilename(filetypes=[("Hailo model files", "*.hef")])
            if model_path:
                try:
                    self.load_line_hailo_model(model_path)
                    self.on_line_backend_changed()
                    messagebox.showinfo(
                        self.tr("model_loaded"),
                        self.tr("model_loaded_msg").format(self.line_hailo_name)
                    )
                except Exception as e:
                    messagebox.showerror(self.tr("model_error"), self.tr("model_error_msg").format(e))
            return

        if backend == "UNET":
            weights_path = filedialog.askopenfilename(filetypes=[("U-Net weights", "*.pth *.pt"), ("PyTorch weights", "*.pth *.pt")])
            if weights_path:
                try:
                    self.load_line_unet_weights(weights_path)
                    self.on_line_backend_changed()
                    messagebox.showinfo(
                        self.tr("model_loaded"),
                        self.tr("model_loaded_msg").format(self.line_unet_name)
                    )
                except Exception as e:
                    messagebox.showerror(self.tr("model_error"), self.tr("model_error_msg").format(e))
            return

        if backend == "SEGFORMER":
            model_dir = filedialog.askdirectory(title="Wybierz katalog modelu SegFormer")
            if model_dir:
                try:
                    self.load_line_segformer_model(model_dir)
                    self.on_line_backend_changed()
                    messagebox.showinfo(
                        self.tr("model_loaded"),
                        self.tr("model_loaded_msg").format(self.line_segformer_name)
                    )
                except Exception as e:
                    messagebox.showerror(self.tr("model_error"), self.tr("model_error_msg").format(e))
            return

        weights_path = filedialog.askopenfilename(filetypes=[("DeepLabV3 weights", "*.pt *.pth"), ("PyTorch weights", "*.pt *.pth")])
        if weights_path:
            try:
                self.load_line_deeplab_weights(weights_path)
                self.on_line_backend_changed()
                loaded_name = self.line_deeplab_name
                if self.line_deeplab_arch:
                    loaded_name = f"{loaded_name} ({self.line_deeplab_arch})"
                messagebox.showinfo(
                    self.tr("model_loaded"),
                    self.tr("model_loaded_msg").format(loaded_name)
                )
            except Exception as e:
                messagebox.showerror(self.tr("model_error"), self.tr("model_error_msg").format(e))

    def clear_surface_model(self):
        if self.running:
            self.stop_processing(silent=True)

        backend = self.surface_backend_var.get()
        cleared_name = None

        if backend == "YOLO":
            if self.surface_model is not None:
                cleared_name = self.surface_yolo_name or self.tr("no_model")
                self.surface_model = None
                self.surface_yolo_name = None
        elif backend == "HAILO":
            if self.surface_hailo_model is not None:
                cleared_name = self.surface_hailo_name or self.tr("no_model")
                try:
                    self.surface_hailo_model.close()
                except Exception:
                    pass
                self.surface_hailo_model = None
                self.surface_hailo_name = None
        elif backend == "UNET":
            if self.unet_model is not None:
                cleared_name = self.surface_unet_name or self.tr("no_model")
                self.unet_model = None
                self.surface_unet_name = None
        elif backend == "DEEPLABV3":
            if self.surface_deeplab_model is not None:
                cleared_name = self.surface_deeplab_name or self.tr("no_model")
                self.surface_deeplab_model = None
                self.surface_deeplab_name = None
                self.surface_deeplab_arch = None
        else:
            if self.surface_segformer_model is not None:
                cleared_name = self.surface_segformer_name or self.tr("no_model")
                self.surface_segformer_model = None
                self.surface_segformer_processor = None
                self.surface_segformer_name = None

        self.last_surface_roi = None
        self.display_surface_roi = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.on_surface_backend_changed()

        if cleared_name:
            messagebox.showinfo(
                self.tr("model_cleared"),
                self.tr("model_cleared_msg").format(cleared_name)
            )

    def clear_line_model(self):
        if self.running:
            self.stop_processing(silent=True)

        backend = self.line_backend_var.get()
        cleared_name = None

        if backend == "YOLO":
            if self.line_model is not None:
                cleared_name = self.line_yolo_name or self.tr("no_model")
                self.line_model = None
                self.line_yolo_name = None
        elif backend == "HAILO":
            if self.line_hailo_model is not None:
                cleared_name = self.line_hailo_name or self.tr("no_model")
                try:
                    self.line_hailo_model.close()
                except Exception:
                    pass
                self.line_hailo_model = None
                self.line_hailo_name = None
        elif backend == "UNET":
            if self.line_unet_model is not None:
                cleared_name = self.line_unet_name or self.tr("no_model")
                self.line_unet_model = None
                self.line_unet_name = None
        elif backend == "DEEPLABV3":
            if self.line_deeplab_model is not None:
                cleared_name = self.line_deeplab_name or self.tr("no_model")
                self.line_deeplab_model = None
                self.line_deeplab_name = None
                self.line_deeplab_arch = None
        else:
            if self.line_segformer_model is not None:
                cleared_name = self.line_segformer_name or self.tr("no_model")
                self.line_segformer_model = None
                self.line_segformer_processor = None
                self.line_segformer_name = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.on_line_backend_changed()

        if cleared_name:
            messagebox.showinfo(
                self.tr("model_cleared"),
                self.tr("model_cleared_msg").format(cleared_name)
            )

    def _predict_segmentation(self, model, device, img_size, frame_infer_bgr, image_processor=None):
        h, w = frame_infer_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_infer_bgr, cv2.COLOR_BGR2RGB)

        if image_processor is not None:
            enc = image_processor(images=rgb, return_tensors="pt")
            x = enc["pixel_values"].to(device, non_blocking=True)

            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(pixel_values=x)
                logits = outputs.logits
                logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
                probs = F.softmax(logits, dim=1)
                conf, pred = torch.max(probs, dim=1)

            pred = pred.squeeze(0).to("cpu").numpy().astype(np.uint8)
            conf = conf.squeeze(0).to("cpu").numpy().astype(np.float32)
            return pred, conf

        inp = cv2.resize(
            rgb,
            (img_size, img_size),
            interpolation=cv2.INTER_LINEAR
        )

        x = torch.from_numpy(inp).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        x = x.to(device, non_blocking=True)

        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = self._forward_segmentation_logits(model, x)
            probs = F.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)

        pred = pred.squeeze(0).to("cpu").numpy().astype(np.uint8)
        conf = conf.squeeze(0).to("cpu").numpy().astype(np.float32)

        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
        conf = cv2.resize(conf, (w, h), interpolation=cv2.INTER_LINEAR)
        return pred, conf

    def _run_line_segmentation_on_frame(
        self,
        model,
        device,
        img_size,
        num_classes,
        class_names,
        min_area,
        frame_infer_bgr,
        frame_draw_bgr,
        frame_idx,
        surface_roi_mask=None,
        analysis_roi_mask=None,
        image_processor=None,
    ):
        if model is None:
            return frame_draw_bgr

        overlay_line = frame_draw_bgr.copy()
        pred, conf = self._predict_segmentation(model, device, img_size, frame_infer_bgr, image_processor=image_processor)

        valid = (pred > 0) & (conf >= self.conf_value)
        if analysis_roi_mask is not None:
            valid = valid & analysis_roi_mask.astype(bool)

        roi_dilated = None
        if surface_roi_mask is not None:
            roi = surface_roi_mask.astype(np.uint8)
            k = max(1, int(self.surface_roi_dilate))
            roi_dilated = cv2.dilate(roi, np.ones((k, k), np.uint8), iterations=1)
            valid = valid & roi_dilated.astype(bool)

        for cls_id in range(1, num_classes):
            class_name = class_names.get(cls_id, str(cls_id))
            color = self.get_line_color_by_name(class_name)
            cls_mask = valid & (pred == cls_id)
            if not cls_mask.any():
                continue
            for c in range(3):
                overlay_line[:, :, c][cls_mask] = color[c]

        frame_draw_bgr = cv2.addWeighted(overlay_line, 0.4, frame_draw_bgr, 0.6, 0)

        for cls_id in range(1, num_classes):
            class_name = class_names.get(cls_id, str(cls_id))
            color = self.get_line_color_by_name(class_name)

            cls_mask = ((pred == cls_id) & (conf >= self.conf_value))
            if analysis_roi_mask is not None:
                cls_mask = cls_mask & analysis_roi_mask.astype(bool)
            if roi_dilated is not None:
                cls_mask = cls_mask & roi_dilated.astype(bool)
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

            if best_cid is None or best_area < min_area:
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

                x, y, w_box, h_box = cv2.boundingRect(biggest)
                bbox = (x, y, x + w_box, y + h_box)
                self.register_confirmed_candidate(frame_idx, "line", class_name, region_conf, bbox)
                self.observe_line_candidate(class_name)

            cx, cy = centroids[best_cid]
            translated_name = self.translate_class_name(class_name)
            label = f"{translated_name}: {region_conf:.2%}"
            cv2.putText(
                frame_draw_bgr,
                label,
                (int(cx), int(cy)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        return frame_draw_bgr

    def _run_surface_segmentation_on_frame(
        self,
        model,
        device,
        img_size,
        num_classes,
        class_names,
        min_area,
        frame_infer_bgr,
        frame_draw_bgr,
        frame_idx,
        analysis_roi_mask=None,
        image_processor=None,
    ):
        if model is None:
            self.last_surface_roi = None
            return frame_draw_bgr

        overlay_surface = frame_draw_bgr.copy()
        pred, conf = self._predict_segmentation(model, device, img_size, frame_infer_bgr, image_processor=image_processor)

        valid = (pred > 0) & (conf >= self.conf_value)
        if analysis_roi_mask is not None:
            valid = valid & analysis_roi_mask.astype(bool)

        roi = valid.astype(np.uint8)
        if roi.any():
            self.last_surface_roi = roi

        for cls_id in range(1, num_classes):
            class_name = class_names.get(cls_id, str(cls_id))
            color = self.get_surface_color_by_name(class_name, cls_id)
            cls_mask_overlay = valid & (pred == cls_id)
            if not cls_mask_overlay.any():
                continue
            for c in range(3):
                overlay_surface[:, :, c][cls_mask_overlay] = color[c]
        frame_draw_bgr = cv2.addWeighted(overlay_surface, 0.4, frame_draw_bgr, 0.6, 0)

        for cls_id in range(1, num_classes):
            cls_mask = ((pred == cls_id) & (conf >= self.conf_value))
            if analysis_roi_mask is not None:
                cls_mask = cls_mask & analysis_roi_mask.astype(bool)
            cls_mask = cls_mask.astype(np.uint8)

            if cls_mask.max() == 0:
                continue

            num, labels, stats, centroids = cv2.connectedComponentsWithStats(cls_mask, connectivity=8)

            for cid in range(1, num):
                area = int(stats[cid, cv2.CC_STAT_AREA])
                if area < min_area:
                    continue

                region = (labels == cid)
                region_conf = float(conf[region].mean())
                if region_conf < self.conf_value:
                    continue

                class_name = class_names.get(cls_id, str(cls_id))
                color = self.get_surface_color_by_name(class_name, cls_id)
                self.observe_surface_candidate(class_name, area)

                reg_mask = (region.astype(np.uint8) * 255)
                contours, _ = cv2.findContours(reg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    cv2.drawContours(frame_draw_bgr, contours, -1, color, 2)
                    biggest = max(contours, key=cv2.contourArea)
                    x, y, w_box, h_box = cv2.boundingRect(biggest)
                    bbox = (x, y, x + w_box, y + h_box)
                    self.register_confirmed_candidate(frame_idx, "surface", class_name, region_conf, bbox)

                cx, cy = centroids[cid]
                translated_name = self.translate_class_name(class_name)
                label = f"{translated_name}: {region_conf:.2%}"
                cv2.putText(
                    frame_draw_bgr,
                    label,
                    (int(cx), int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        return frame_draw_bgr

    def run_unet_line_on_frame(
        self,
        frame_infer_bgr,
        frame_draw_bgr,
        frame_idx,
        surface_roi_mask=None,
        analysis_roi_mask=None
    ):
        return self._run_line_segmentation_on_frame(
            model=self.line_unet_model,
            device=self.line_unet_device,
            img_size=self.line_unet_img_size,
            num_classes=self.line_unet_num_classes,
            class_names=self.line_unet_names,
            min_area=self.line_unet_min_area,
            frame_infer_bgr=frame_infer_bgr,
            frame_draw_bgr=frame_draw_bgr,
            frame_idx=frame_idx,
            surface_roi_mask=surface_roi_mask,
            analysis_roi_mask=analysis_roi_mask,
        )

    def run_deeplab_line_on_frame(
        self,
        frame_infer_bgr,
        frame_draw_bgr,
        frame_idx,
        surface_roi_mask=None,
        analysis_roi_mask=None
    ):
        return self._run_line_segmentation_on_frame(
            model=self.line_deeplab_model,
            device=self.line_deeplab_device,
            img_size=self.line_deeplab_img_size,
            num_classes=self.line_deeplab_num_classes,
            class_names=self.line_deeplab_names,
            min_area=self.line_deeplab_min_area,
            frame_infer_bgr=frame_infer_bgr,
            frame_draw_bgr=frame_draw_bgr,
            frame_idx=frame_idx,
            surface_roi_mask=surface_roi_mask,
            analysis_roi_mask=analysis_roi_mask,
        )


    def run_segformer_line_on_frame(
        self,
        frame_infer_bgr,
        frame_draw_bgr,
        frame_idx,
        surface_roi_mask=None,
        analysis_roi_mask=None
    ):
        return self._run_line_segmentation_on_frame(
            model=self.line_segformer_model,
            device=self.line_segformer_device,
            img_size=0,
            num_classes=self.line_segformer_num_classes,
            class_names=self.line_segformer_names,
            min_area=self.line_segformer_min_area,
            frame_infer_bgr=frame_infer_bgr,
            frame_draw_bgr=frame_draw_bgr,
            frame_idx=frame_idx,
            surface_roi_mask=surface_roi_mask,
            analysis_roi_mask=analysis_roi_mask,
            image_processor=self.line_segformer_processor,
        )

    def run_unet_surface_on_frame(self, frame_infer_bgr, frame_draw_bgr, frame_idx, analysis_roi_mask=None):
        return self._run_surface_segmentation_on_frame(
            model=self.unet_model,
            device=self.unet_device,
            img_size=self.unet_img_size,
            num_classes=self.unet_num_classes,
            class_names=self.unet_names,
            min_area=self.unet_min_area,
            frame_infer_bgr=frame_infer_bgr,
            frame_draw_bgr=frame_draw_bgr,
            frame_idx=frame_idx,
            analysis_roi_mask=analysis_roi_mask,
        )

    def run_deeplab_surface_on_frame(self, frame_infer_bgr, frame_draw_bgr, frame_idx, analysis_roi_mask=None):
        return self._run_surface_segmentation_on_frame(
            model=self.surface_deeplab_model,
            device=self.deeplab_device,
            img_size=self.deeplab_img_size,
            num_classes=self.deeplab_num_classes,
            class_names=self.deeplab_names,
            min_area=self.deeplab_min_area,
            frame_infer_bgr=frame_infer_bgr,
            frame_draw_bgr=frame_draw_bgr,
            frame_idx=frame_idx,
            analysis_roi_mask=analysis_roi_mask,
        )


    def run_segformer_surface_on_frame(self, frame_infer_bgr, frame_draw_bgr, frame_idx, analysis_roi_mask=None):
        return self._run_surface_segmentation_on_frame(
            model=self.surface_segformer_model,
            device=self.segformer_device,
            img_size=0,
            num_classes=self.segformer_num_classes,
            class_names=self.segformer_names,
            min_area=self.segformer_min_area,
            frame_infer_bgr=frame_infer_bgr,
            frame_draw_bgr=frame_draw_bgr,
            frame_idx=frame_idx,
            analysis_roi_mask=analysis_roi_mask,
            image_processor=self.surface_segformer_processor,
        )

    def _run_hailo_surface_on_frame(self, model, frame_infer_bgr, frame_draw_bgr, frame_idx, analysis_roi_mask=None):
        if model is None:
            self.last_surface_roi = None
            return frame_draw_bgr

        detections = model.infer(frame_infer_bgr, conf_thres=self.conf_value)
        if not detections:
            return frame_draw_bgr

        overlay_surface = frame_draw_bgr.copy()
        surface_roi = np.zeros(frame_draw_bgr.shape[:2], dtype=np.uint8)

        for det in detections:
            cls_id = int(det.get("class_id", 0))
            class_name = str(det.get("class_name", cls_id))
            conf = float(det.get("score", 0.0))
            color = self.get_surface_color_by_name(class_name, cls_id)
            mask = det.get("mask")
            if mask is None:
                continue
            mask = mask.astype(bool)
            if analysis_roi_mask is not None:
                mask = mask & analysis_roi_mask.astype(bool)
            if not mask.any():
                continue

            surface_roi[mask] = 1
            self.observe_surface_candidate(class_name, int(mask.sum()))

            for c in range(3):
                overlay_surface[:, :, c][mask] = color[c]

            reg_mask = (mask.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(reg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            biggest = max(contours, key=cv2.contourArea)
            cv2.drawContours(frame_draw_bgr, [biggest], -1, color, 2)

            M = cv2.moments(biggest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                translated_name = self.translate_class_name(class_name)
                label = f"{translated_name}: {conf:.2%}"
                cv2.putText(
                    frame_draw_bgr,
                    label,
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

            x, y, w_box, h_box = cv2.boundingRect(biggest)
            bbox = (x, y, x + w_box, y + h_box)
            self.register_confirmed_candidate(frame_idx, "surface", class_name, conf, bbox)

        frame_draw_bgr = cv2.addWeighted(overlay_surface, 0.4, frame_draw_bgr, 0.6, 0)
        if surface_roi.any():
            self.last_surface_roi = surface_roi
        return frame_draw_bgr

    def _run_hailo_line_on_frame(
        self,
        model,
        frame_infer_bgr,
        frame_draw_bgr,
        frame_idx,
        surface_roi_mask=None,
        analysis_roi_mask=None,
    ):
        if model is None:
            return frame_draw_bgr

        detections = model.infer(frame_infer_bgr, conf_thres=self.conf_value)
        if not detections:
            return frame_draw_bgr

        overlay_line = frame_draw_bgr.copy()

        if analysis_roi_mask is not None and surface_roi_mask is not None:
            line_roi = (analysis_roi_mask.astype(bool) & surface_roi_mask.astype(bool))
        elif analysis_roi_mask is not None:
            line_roi = analysis_roi_mask.astype(bool)
        elif surface_roi_mask is not None:
            line_roi = surface_roi_mask.astype(bool)
        else:
            line_roi = None

        for det in detections:
            cls_id = int(det.get("class_id", 0))
            class_name = str(det.get("class_name", cls_id))
            conf = float(det.get("score", 0.0))
            color = self.get_line_color_by_name(class_name)
            mask = det.get("mask")
            if mask is None:
                continue
            mask = mask.astype(bool)
            if line_roi is not None:
                mask = mask & line_roi
            if not mask.any():
                continue

            for c in range(3):
                overlay_line[:, :, c][mask] = color[c]

            reg_mask = (mask.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(reg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            biggest = max(contours, key=cv2.contourArea)
            cv2.drawContours(frame_draw_bgr, [biggest], -1, color, 2)

            M = cv2.moments(biggest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                translated_name = self.translate_class_name(class_name)
                label = f"{translated_name}: {conf:.2%}"
                cv2.putText(
                    frame_draw_bgr,
                    label,
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

            x, y, w_box, h_box = cv2.boundingRect(biggest)
            bbox = (x, y, x + w_box, y + h_box)
            self.register_confirmed_candidate(frame_idx, "line", class_name, conf, bbox)
            self.observe_line_candidate(class_name)

        frame_draw_bgr = cv2.addWeighted(overlay_line, 0.4, frame_draw_bgr, 0.6, 0)
        return frame_draw_bgr

    def run_detection_on_frame(self, frame, frame_idx):
        self.begin_frame_event_scan()

        frame_vis = frame.copy()
        height, width = frame.shape[:2]

        analysis_roi, roi_y = self.get_analysis_roi_mask(height, width)
        analysis_roi255 = (analysis_roi * 255).astype(np.uint8) if analysis_roi is not None else None
        frame_for_models = frame

        surface_should_update = self.should_update_surface_model(frame_idx)

        surface_backend = self.surface_backend_var.get()
        if surface_should_update and surface_backend == "YOLO":
            if self.surface_model is not None:
                overlay_surface = frame_vis.copy()
                surface_roi = np.zeros((height, width), dtype=np.uint8)

                results_surface = self.surface_model(
                    frame_for_models,
                    conf=self.conf_value,
                    device=self.yolo_device,
                    verbose=False
                )

                for r in results_surface:
                    if hasattr(r, "masks") and r.masks is not None and r.boxes is not None:
                        masks = r.masks.data.cpu().numpy()
                        n = min(len(masks), len(r.boxes))

                        for idx in range(n):
                            mask = masks[idx]
                            cls_id = int(r.boxes.cls[idx].item())
                            conf = float(r.boxes.conf[idx].item())
                            class_name = self.surface_model.names.get(cls_id, str(cls_id))
                            color = self.get_class_color(cls_id)

                            mask_resized = cv2.resize((mask * 255).astype("uint8"), (width, height))

                            if analysis_roi255 is not None:
                                mask_resized = cv2.bitwise_and(mask_resized, analysis_roi255)
                                if mask_resized.max() == 0:
                                    continue

                            surface_roi[mask_resized > 0] = 1
                            self.observe_surface_candidate(class_name, int((mask_resized > 0).sum()))

                            for c in range(3):
                                overlay_surface[:, :, c][mask_resized > 0] = color[c]

                            contours, _ = cv2.findContours(
                                mask_resized,
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE
                            )
                            cv2.drawContours(frame_vis, contours, -1, color, 2)

                            if contours:
                                biggest = max(contours, key=cv2.contourArea)
                                M = cv2.moments(biggest)
                                if M["m00"] != 0:
                                    cx = int(M["m10"] / M["m00"])
                                    cy = int(M["m01"] / M["m00"])
                                    translated_name = self.translate_class_name(class_name)
                                    label = f"{translated_name}: {conf:.2%}"
                                    cv2.putText(
                                        frame_vis,
                                        label,
                                        (cx, cy),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6,
                                        color,
                                        2
                                    )

                                x, y, w_box, h_box = cv2.boundingRect(biggest)
                                bbox = (x, y, x + w_box, y + h_box)
                                self.register_confirmed_candidate(frame_idx, "surface", class_name, conf, bbox)

                    else:
                        if r.boxes is None:
                            continue

                        for box in r.boxes:
                            cls_id = int(box.cls[0].item())
                            conf = float(box.conf[0].item())
                            class_name = self.surface_model.names.get(cls_id, str(cls_id))
                            color = self.get_class_color(cls_id)

                            x1, y1, x2, y2 = map(int, box.xyxy[0])

                            x1c, y1c = max(0, x1), max(0, y1)
                            x2c, y2c = min(width, x2), min(height, y2)
                            if x2c <= x1c or y2c <= y1c:
                                continue

                            if analysis_roi is not None:
                                if not analysis_roi[y1c:y2c, x1c:x2c].any():
                                    continue
                                y1c = max(y1c, roi_y)

                            if y2c <= y1c:
                                continue

                            surface_roi[y1c:y2c, x1c:x2c] = 1
                            self.observe_surface_candidate(class_name, (x2c - x1c) * (y2c - y1c))
                            cv2.rectangle(frame_vis, (x1c, y1c), (x2c, y2c), color, 2)

                            translated_name = self.translate_class_name(class_name)
                            label = f"{translated_name}: {conf:.2%}"
                            cv2.putText(
                                frame_vis,
                                label,
                                (x1c, max(15, y1c - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                color,
                                2
                            )

                            bbox = (x1c, y1c, x2c, y2c)
                            self.register_confirmed_candidate(frame_idx, "surface", class_name, conf, bbox)

                frame_vis = cv2.addWeighted(overlay_surface, 0.4, frame_vis, 0.6, 0)
                if surface_roi.any():
                    self.last_surface_roi = surface_roi

        elif surface_should_update and surface_backend == "HAILO":
            frame_vis = self._run_hailo_surface_on_frame(
                self.surface_hailo_model,
                frame_for_models,
                frame_vis,
                frame_idx,
                analysis_roi_mask=analysis_roi
            )
        elif surface_should_update and surface_backend == "UNET":
            frame_vis = self.run_unet_surface_on_frame(
                frame_for_models,
                frame_vis,
                frame_idx,
                analysis_roi_mask=analysis_roi
            )
        elif surface_should_update and surface_backend == "DEEPLABV3":
            frame_vis = self.run_deeplab_surface_on_frame(
                frame_for_models,
                frame_vis,
                frame_idx,
                analysis_roi_mask=analysis_roi
            )
        elif surface_should_update:
            frame_vis = self.run_segformer_surface_on_frame(
                frame_for_models,
                frame_vis,
                frame_idx,
                analysis_roi_mask=analysis_roi
            )

        line_backend = self.line_backend_var.get()
        if line_backend == "YOLO":
            if self.line_model is not None:
                overlay_line = frame_vis.copy()
                results_line = self.line_model(
                    frame_for_models,
                    conf=self.conf_value,
                    device=self.yolo_device,
                    verbose=False
                )

                surface_roi = self.last_surface_roi

                if analysis_roi is not None and surface_roi is not None:
                    line_roi = (analysis_roi.astype(bool) & surface_roi.astype(bool)).astype(np.uint8)
                elif analysis_roi is not None:
                    line_roi = analysis_roi
                else:
                    line_roi = surface_roi

                roi255 = (line_roi * 255).astype(np.uint8) if line_roi is not None else None

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

                            mask_resized = cv2.resize((mask * 255).astype("uint8"), (width, height))

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

                        for cls_id, combined_mask in class_masks.items():
                            class_name = self.line_model.names.get(cls_id, str(cls_id))
                            color = self.get_line_color_by_name(class_name)

                            if roi255 is not None:
                                combined_mask = cv2.bitwise_and(combined_mask, roi255)
                                if combined_mask.max() == 0:
                                    continue

                            for c in range(3):
                                overlay_line[:, :, c][combined_mask > 0] = color[c]

                            contours, _ = cv2.findContours(
                                combined_mask,
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE
                            )
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
                                cv2.putText(
                                    frame_vis,
                                    label,
                                    (cx, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    color,
                                    2
                                )

                            x, y, w_box, h_box = cv2.boundingRect(biggest)
                            bbox = (x, y, x + w_box, y + h_box)
                            conf_best = class_best_conf.get(cls_id, 0.0)
                            self.register_confirmed_candidate(frame_idx, "line", class_name, conf_best, bbox)
                            self.observe_line_candidate(class_name)

                    else:
                        if r.boxes is None:
                            continue

                        for box in r.boxes:
                            cls_id = int(box.cls[0].item())
                            conf = float(box.conf[0].item())
                            class_name = self.line_model.names.get(cls_id, str(cls_id))
                            color = self.get_line_color_by_name(class_name)

                            x1, y1, x2, y2 = map(int, box.xyxy[0])

                            x1c, y1c = max(0, x1), max(0, y1)
                            x2c, y2c = min(width, x2), min(height, y2)
                            if x2c <= x1c or y2c <= y1c:
                                continue

                            if line_roi is not None:
                                if not line_roi[y1c:y2c, x1c:x2c].any():
                                    continue
                                if analysis_roi is not None:
                                    y1c = max(y1c, roi_y)

                            if y2c <= y1c:
                                continue

                            cv2.rectangle(frame_vis, (x1c, y1c), (x2c, y2c), color, 2)

                            translated_name = self.translate_class_name(class_name)
                            label = f"{translated_name}: {conf:.2%}"
                            cv2.putText(
                                frame_vis,
                                label,
                                (x1c, max(15, y1c - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                color,
                                2
                            )

                            bbox = (x1c, y1c, x2c, y2c)
                            self.register_confirmed_candidate(frame_idx, "line", class_name, conf, bbox)
                            self.observe_line_candidate(class_name)

                frame_vis = cv2.addWeighted(overlay_line, 0.4, frame_vis, 0.6, 0)

        elif line_backend == "HAILO":
            surface_roi = self.last_surface_roi if self.surface_ready() else None
            frame_vis = self._run_hailo_line_on_frame(
                self.line_hailo_model,
                frame_for_models,
                frame_vis,
                frame_idx,
                surface_roi_mask=surface_roi,
                analysis_roi_mask=analysis_roi
            )
        elif line_backend == "UNET":
            surface_roi = self.last_surface_roi if self.surface_ready() else None
            frame_vis = self.run_unet_line_on_frame(
                frame_for_models,
                frame_vis,
                frame_idx,
                surface_roi_mask=surface_roi,
                analysis_roi_mask=analysis_roi
            )
        elif line_backend == "DEEPLABV3":
            surface_roi = self.last_surface_roi if self.surface_ready() else None
            frame_vis = self.run_deeplab_line_on_frame(
                frame_for_models,
                frame_vis,
                frame_idx,
                surface_roi_mask=surface_roi,
                analysis_roi_mask=analysis_roi
            )
        else:
            surface_roi = self.last_surface_roi if self.surface_ready() else None
            frame_vis = self.run_segformer_line_on_frame(
                frame_for_models,
                frame_vis,
                frame_idx,
                surface_roi_mask=surface_roi,
                analysis_roi_mask=analysis_roi
            )

        previous_surface_name = getattr(self, "current_surface_name", None)
        self.finalize_frame_event_scan()

        if getattr(self, "current_surface_name", None):
            self.cached_surface_name = self.current_surface_name
        elif getattr(self, "_frame_best_surface", None):
            self.cached_surface_name = self._frame_best_surface

        current_surface_name = getattr(self, "current_surface_name", None)
        if getattr(self, "last_surface_roi", None) is not None:
            if getattr(self, "display_surface_roi", None) is None:
                self.display_surface_roi = self.last_surface_roi.copy()
            elif current_surface_name is not None and current_surface_name != previous_surface_name:
                self.display_surface_roi = self.last_surface_roi.copy()

        frame_vis = self.draw_cached_surface_overlay(frame_vis)
        frame_vis = self.draw_current_surface_status(frame_vis)
        frame_vis = self.draw_roi_boundary(frame_vis, roi_y)
        return frame_vis
    pass