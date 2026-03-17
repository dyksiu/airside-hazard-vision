import os
import cv2
import random
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from tkinter import filedialog, messagebox

from models.unet import UNet


class DetectionMixin:

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

    def load_line_unet_weights(self, weights_path: str):
        model = UNet(
            in_channels=3,
            num_classes=self.line_unet_num_classes,
            base=self.line_unet_base
        ).to(self.line_unet_device)

        state = torch.load(weights_path, map_location=self.line_unet_device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

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
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

        model.load_state_dict(state)
        model.eval()

        self.unet_model = model
        self.surface_unet_name = os.path.basename(weights_path)

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

        weights_path = filedialog.askopenfilename(filetypes=[("U-Net weights", "*.pth")])
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

        weights_path = filedialog.askopenfilename(filetypes=[("U-Net weights", "*.pth")])
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
        else:
            if self.unet_model is not None:
                cleared_name = self.surface_unet_name or self.tr("no_model")
                self.unet_model = None
                self.surface_unet_name = None

        self.last_surface_roi = None

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
        else:
            if self.line_unet_model is not None:
                cleared_name = self.line_unet_name or self.tr("no_model")
                self.line_unet_model = None
                self.line_unet_name = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.on_line_backend_changed()

        if cleared_name:
            messagebox.showinfo(
                self.tr("model_cleared"),
                self.tr("model_cleared_msg").format(cleared_name)
            )

    def run_unet_line_on_frame(
        self,
        frame_infer_bgr,
        frame_draw_bgr,
        frame_idx,
        surface_roi_mask=None,
        analysis_roi_mask=None
    ):
        if self.line_unet_model is None:
            return frame_draw_bgr

        h, w = frame_draw_bgr.shape[:2]
        overlay_line = frame_draw_bgr.copy()

        rgb = cv2.cvtColor(frame_infer_bgr, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(
            rgb,
            (self.line_unet_img_size, self.line_unet_img_size),
            interpolation=cv2.INTER_LINEAR
        )

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
        if analysis_roi_mask is not None:
            valid = valid & analysis_roi_mask.astype(bool)

        roi_dilated = None
        if surface_roi_mask is not None:
            roi = surface_roi_mask.astype(np.uint8)
            k = max(1, int(self.surface_roi_dilate))
            roi_dilated = cv2.dilate(roi, np.ones((k, k), np.uint8), iterations=1)
            valid = valid & roi_dilated.astype(bool)

        for cls_id in range(1, self.line_unet_num_classes):
            class_name = self.line_unet_names.get(cls_id, str(cls_id))
            color = self.get_line_color_by_name(class_name)
            cls_mask = valid & (pred == cls_id)
            if not cls_mask.any():
                continue
            for c in range(3):
                overlay_line[:, :, c][cls_mask] = color[c]

        frame_draw_bgr = cv2.addWeighted(overlay_line, 0.4, frame_draw_bgr, 0.6, 0)

        for cls_id in range(1, self.line_unet_num_classes):
            class_name = self.line_unet_names.get(cls_id, str(cls_id))
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

                x, y, w_box, h_box = cv2.boundingRect(biggest)
                bbox = (x, y, x + w_box, y + h_box)
                self.register_confirmed_candidate(frame_idx, "line", class_name, region_conf, bbox)

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
	
    def run_unet_surface_on_frame(self, frame_infer_bgr, frame_draw_bgr, frame_idx, analysis_roi_mask=None):
        if self.unet_model is None:
            self.last_surface_roi = None
            return frame_draw_bgr

        h, w = frame_draw_bgr.shape[:2]
        overlay_surface = frame_draw_bgr.copy()

        rgb = cv2.cvtColor(frame_infer_bgr, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(
            rgb,
            (self.unet_img_size, self.unet_img_size),
            interpolation=cv2.INTER_LINEAR
        )

        x = torch.from_numpy(inp).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        x = x.to(self.unet_device, non_blocking=True)

        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(self.unet_device.type == "cuda")):
            logits = self.unet_model(x)
            probs = F.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)

        pred = pred.squeeze(0).to("cpu").numpy().astype(np.uint8)
        conf = conf.squeeze(0).to("cpu").numpy().astype(np.float32)

        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
        conf = cv2.resize(conf, (w, h), interpolation=cv2.INTER_LINEAR)

        valid = (pred > 0) & (conf >= self.conf_value)
        if analysis_roi_mask is not None:
            valid = valid & analysis_roi_mask.astype(bool)

        roi = valid.astype(np.uint8)
        self.last_surface_roi = roi if roi.any() else None

        color_mask = self.unet_colors_bgr[pred]
        overlay_surface[valid] = color_mask[valid]
        frame_draw_bgr = cv2.addWeighted(overlay_surface, 0.4, frame_draw_bgr, 0.6, 0)

        for cls_id in range(1, self.unet_num_classes):
            cls_mask = ((pred == cls_id) & (conf >= self.conf_value))
            if analysis_roi_mask is not None:
                cls_mask = cls_mask & analysis_roi_mask.astype(bool)
            cls_mask = cls_mask.astype(np.uint8)

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
                class_name = self.unet_names.get(cls_id, str(cls_id))

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

    def run_detection_on_frame(self, frame, frame_idx):
        frame_vis = frame.copy()
        height, width = frame.shape[:2]

        analysis_roi, roi_y = self.get_analysis_roi_mask(height, width)
        analysis_roi255 = (analysis_roi * 255).astype(np.uint8) if analysis_roi is not None else None
        frame_for_models = frame

        self.last_surface_roi = None

        if self.surface_backend_var.get() == "YOLO":
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
                self.last_surface_roi = surface_roi if surface_roi.any() else None

        else:
            frame_vis = self.run_unet_surface_on_frame(
                frame_for_models,
                frame_vis,
                frame_idx,
                analysis_roi_mask=analysis_roi
            )

        if self.line_backend_var.get() == "YOLO":
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

                frame_vis = cv2.addWeighted(overlay_line, 0.4, frame_vis, 0.6, 0)

        else:
            surface_roi = self.last_surface_roi if self.surface_ready() else None
            frame_vis = self.run_unet_line_on_frame(
                frame_for_models,
                frame_vis,
                frame_idx,
                surface_roi_mask=surface_roi,
                analysis_roi_mask=analysis_roi
            )

        frame_vis = self.draw_roi_boundary(frame_vis, roi_y)
        return frame_vis
    pass