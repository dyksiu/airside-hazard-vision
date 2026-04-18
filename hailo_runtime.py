import os
import threading
from pathlib import Path

import cv2
import numpy as np


class HailoYOLOSegModel:
    """
    Minimal wrapper for YOLOv8 segmentation HEF models running on Hailo.

    Inference is executed on the Hailo device, while postprocessing (NMS,
    mask decoding, ROI/mask resizing) is executed on the host CPU.
    """

    def __init__(self, hef_path, labels=None, default_labels=None, iou_threshold=0.7):
        self.hef_path = os.fspath(hef_path)
        self.iou_threshold = float(iou_threshold)
        self._closed = False

        self._hp = None
        self._format_order = None
        self.target = None
        self.hef = None
        self.infer_model = None
        self.config_ctx = None
        self.configured_model = None
        self.last_infer_job = None

        self.input_height = 0
        self.input_width = 0
        self.input_channels = 3
        self.nms_postprocess_enabled = False
        self.arch = None
        self.reg_max = None
        self.mask_channels = 32
        self.num_classes = 0

        self._load_runtime()
        self._configure_model()
        self._configure_labels(labels=labels, default_labels=default_labels)

    def _load_runtime(self):
        try:
            import hailo_platform as hp
            from hailo_platform.pyhailort.pyhailort import FormatOrder
        except ImportError as exc:
            raise ImportError(
                "Brakuje pakietu hailo_platform. Na Raspberry Pi z Hailo AI HAT "
                "uruchom aplikację w środowisku Hailo / z zainstalowanym PyHailoRT."
            ) from exc
        self._hp = hp
        self._format_order = FormatOrder

    def _configure_model(self):
        hp = self._hp
        params = hp.VDevice.create_params()
        params.scheduling_algorithm = hp.HailoSchedulingAlgorithm.ROUND_ROBIN
        params.group_id = "SHARED"

        self.target = hp.VDevice(params)
        self.hef = hp.HEF(self.hef_path)
        self.infer_model = self.target.create_infer_model(self.hef_path)
        self.infer_model.set_batch_size(1)

        first_output = self.infer_model.outputs[0]
        self.nms_postprocess_enabled = (
            first_output.format.order == self._format_order.HAILO_NMS_WITH_BYTE_MASK
        )
        if self.nms_postprocess_enabled:
            raise NotImplementedError(
                "Ten backend aplikacji obsługuje obecnie surowe modele YOLOv8-seg HEF "
                "(postprocessing na CPU), a nie HEF z wbudowanym Hailo NMS+mask."
            )

        for output in self.infer_model.outputs:
            self.infer_model.output(output.name).set_format_type(hp.FormatType.FLOAT32)

        self.config_ctx = self.infer_model.configure()
        self.configured_model = self.config_ctx.__enter__()
        self.configured_model.set_scheduler_priority(0)

        input_info = self.hef.get_input_vstream_infos()[0]
        ishape = tuple(int(x) for x in input_info.shape)
        if len(ishape) == 3:
            self.input_height, self.input_width, self.input_channels = ishape
        elif len(ishape) == 2:
            self.input_height, self.input_width = ishape
            self.input_channels = 3
        else:
            raise ValueError(f"Nieobsługiwany kształt wejścia HEF: {ishape}")

        self._infer_architecture_from_outputs()

    def _infer_architecture_from_outputs(self):
        infos = self.hef.get_output_vstream_infos()
        outputs = []
        for info in infos:
            shape = tuple(int(x) for x in info.shape)
            if len(shape) != 3:
                continue
            h, w, c = shape
            outputs.append({"name": info.name, "h": h, "w": w, "c": c})

        spatial = sorted({o["h"] for o in outputs if o["h"] == o["w"]})
        if not spatial:
            raise ValueError("Nie udało się rozpoznać wyjść HEF.")

        group_sizes = [s for s in spatial if s != max(spatial)]
        proto_size = max(spatial)
        if len(group_sizes) != 3:
            raise ValueError(
                "Backend HAILO w tej aplikacji oczekuje HEF typu YOLOv8-seg "
                f"z 3 poziomami skali. Wykryte skale: {spatial}"
            )

        reg_counts = []
        cls_counts = []
        for size in group_sizes:
            group = [o for o in outputs if o["h"] == size and o["w"] == size]
            if len(group) < 3:
                raise ValueError(
                    "Nie udało się jednoznacznie rozpoznać wyjść regresji/klas/masek "
                    f"dla skali {size}x{size}."
                )
            group = sorted(group, key=lambda x: x["c"])
            # Typical YOLOv8-seg: [classes, 32 coeff, 64 regression] or [32 coeff, classes, 64 regression]
            coeff = None
            for item in group:
                if item["c"] == 32:
                    coeff = item
                    break
            if coeff is None:
                coeff = min(group, key=lambda x: abs(x["c"] - 32))
            reg = max(group, key=lambda x: x["c"])
            cls = [item for item in group if item["name"] not in {coeff["name"], reg["name"]}]
            if not cls:
                raise ValueError("Nie udało się ustalić liczby klas modelu HEF.")
            cls = cls[0]
            reg_counts.append(reg["c"])
            cls_counts.append(cls["c"])

        self.arch = "yolov8_seg"
        self.reg_max = int(max(reg_counts) // 4 - 1)
        self.num_classes = int(max(cls_counts))
        self.mask_channels = 32

    def _configure_labels(self, labels=None, default_labels=None):
        labels_list = []
        if isinstance(labels, (list, tuple)):
            labels_list = [str(x).strip() for x in labels if str(x).strip()]
        elif isinstance(labels, str) and Path(labels).exists():
            labels_list = self._read_labels_file(labels)
        elif isinstance(default_labels, (list, tuple)):
            labels_list = [str(x).strip() for x in default_labels if str(x).strip()]

        if len(labels_list) == self.num_classes + 1 and labels_list[0].lower() in {"background", "tlo"}:
            labels_list = labels_list[1:]

        if len(labels_list) != self.num_classes:
            labels_list = [f"class_{idx}" for idx in range(self.num_classes)]

        self.labels = labels_list
        self.names = {idx: name for idx, name in enumerate(self.labels)}
        self.name = os.path.basename(self.hef_path)

    @staticmethod
    def _read_labels_file(path):
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def _build_preprocessed_rgb(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        ih, iw = rgb.shape[:2]
        scale = min(self.input_width / iw, self.input_height / ih)
        new_w = max(1, int(round(iw * scale)))
        new_h = max(1, int(round(ih * scale)))
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        pad_w = (self.input_width - new_w) // 2
        pad_h = (self.input_height - new_h) // 2
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        meta = {
            "scale": scale,
            "pad_w": pad_w,
            "pad_h": pad_h,
            "orig_h": ih,
            "orig_w": iw,
        }
        return padded, meta

    def _create_binding(self, frame_rgb):
        output_buffers = {
            output.name: np.empty(tuple(int(x) for x in output.shape), dtype=np.float32)
            for output in self.infer_model.outputs
        }
        binding = self.configured_model.create_bindings(output_buffers=output_buffers)
        binding.input().set_buffer(np.array(frame_rgb))
        return binding

    def _run_raw_inference(self, frame_rgb):
        holder = {}
        event = threading.Event()
        binding = self._create_binding(frame_rgb)
        output_names = [output.name for output in self.infer_model.outputs]

        def callback(*cb_args, **cb_kwargs):
            try:
                completion_info = cb_kwargs.get("completion_info")
                bindings_list = cb_kwargs.get("bindings_list")

                if completion_info is None and len(cb_args) >= 1:
                    completion_info = cb_args[0]
                if bindings_list is None and len(cb_args) >= 2:
                    bindings_list = cb_args[1]

                if completion_info is not None and getattr(completion_info, "exception", None):
                    holder["error"] = completion_info.exception
                    return

                if not bindings_list:
                    bindings_list = [binding]

                b = bindings_list[0]
                result = {}
                for name in output_names:
                    arr = b.output(name).get_buffer()
                    arr = np.asarray(arr)
                    if arr.ndim == 3:
                        arr = np.expand_dims(arr, axis=0)
                    result[name] = arr
                holder["result"] = result
            except Exception as exc:
                holder["error"] = exc
            finally:
                event.set()

        self.configured_model.wait_for_async_ready(timeout_ms=10000)
        try:
            job = self.configured_model.run_async([binding], callback=callback)
        except TypeError:
            job = self.configured_model.run_async([binding], callback)
        self.last_infer_job = job
        job.wait(10000)
        event.wait(10.0)

        if "error" in holder:
            raise RuntimeError(holder["error"])
        if "result" not in holder:
            raise RuntimeError("Brak wyniku inferencji z Hailo.")
        return holder["result"]

    @staticmethod
    def _softmax(x, axis=-1):
        x = x - np.max(x, axis=axis, keepdims=True)
        exp = np.exp(x)
        return exp / np.sum(exp, axis=axis, keepdims=True)

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _xywh_to_xyxy(boxes):
        out = boxes.copy()
        out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        return out

    def _decode_yolov8_boxes(self, raw_boxes_per_scale):
        decoded = []
        reg_range = np.arange(self.reg_max + 1, dtype=np.float32)
        image_dims = (self.input_h, self.input_w)
        for raw, stride in raw_boxes_per_scale:
            _, h, w, _ = raw.shape
            grid_x = np.arange(w, dtype=np.float32) + 0.5
            grid_y = np.arange(h, dtype=np.float32) + 0.5
            grid_x, grid_y = np.meshgrid(grid_x, grid_y)
            ct_row = grid_y.flatten() * stride
            ct_col = grid_x.flatten() * stride
            center = np.stack((ct_col, ct_row, ct_col, ct_row), axis=1)

            dist = raw.reshape((1, h * w, 4, self.reg_max + 1)).astype(np.float32)
            dist = self._softmax(dist, axis=-1)
            dist = np.sum(dist * reg_range.reshape((1, 1, 1, -1)), axis=-1)
            dist = dist * stride
            dist = np.concatenate([dist[:, :, :2] * (-1), dist[:, :, 2:]], axis=-1)
            decoded_xyxy = np.expand_dims(center, axis=0) + dist

            xmin = decoded_xyxy[:, :, 0]
            ymin = decoded_xyxy[:, :, 1]
            xmax = decoded_xyxy[:, :, 2]
            ymax = decoded_xyxy[:, :, 3]
            xywh = np.transpose([
                (xmin + xmax) / 2.0,
                (ymin + ymax) / 2.0,
                xmax - xmin,
                ymax - ymin,
            ], [1, 2, 0])
            decoded.append(xywh)
        return np.concatenate(decoded, axis=1)

    @staticmethod
    def _iou(box, boxes):
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        area1 = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
        area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        union = area1 + area2 - inter + 1e-6
        return inter / union

    def _nms(self, boxes, scores, classes, iou_thresh):
        keep = []
        order = np.argsort(scores)[::-1]
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            same_class = classes[rest] == classes[i]
            suppress = np.zeros(rest.shape[0], dtype=bool)
            if np.any(same_class):
                ious = self._iou(boxes[i], boxes[rest[same_class]])
                suppress[same_class] = ious > iou_thresh
            order = rest[~suppress]
        return np.array(keep, dtype=int)

    @staticmethod
    def _crop_mask_to_boxes(masks, boxes):
        out = np.zeros_like(masks)
        h, w = masks.shape[1:]
        boxes = np.round(boxes).astype(int)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)
        for i in range(masks.shape[0]):
            x1, y1, x2, y2 = boxes[i]
            out[i, y1:y2, x1:x2] = masks[i, y1:y2, x1:x2]
        return out

    def _process_masks(self, protos, masks_in, boxes_xyxy):
        if masks_in.size == 0:
            return np.zeros((0, self.input_height, self.input_width), dtype=np.float32)
        mh, mw, c = protos.shape
        protos_flat = protos.reshape(-1, c).T
        masks = masks_in @ protos_flat
        masks = self._sigmoid(masks).reshape(-1, mh, mw).astype(np.float32)
        resized = np.empty((masks.shape[0], self.input_height, self.input_width), dtype=np.float32)
        for i in range(masks.shape[0]):
            resized[i] = cv2.resize(masks[i], (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        return self._crop_mask_to_boxes(resized, boxes_xyxy)

    def _decode_original_boxes(self, boxes_xyxy, meta):
        boxes = boxes_xyxy.copy().astype(np.float32)
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - meta["pad_w"]) / max(meta["scale"], 1e-6)
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - meta["pad_h"]) / max(meta["scale"], 1e-6)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, meta["orig_w"] - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, meta["orig_h"] - 1)
        return boxes

    def _decode_original_mask(self, mask, meta):
        cropped = mask[
            meta["pad_h"]: self.input_height - meta["pad_h"],
            meta["pad_w"]: self.input_width - meta["pad_w"],
        ]
        if cropped.shape[:2] != (meta["orig_h"], meta["orig_w"]):
            cropped = cv2.resize(
                cropped.astype(np.uint8),
                (meta["orig_w"], meta["orig_h"]),
                interpolation=cv2.INTER_NEAREST,
            )
        return cropped.astype(bool)

    @property
    def input_h(self):
        return self.input_height

    @property
    def input_w(self):
        return self.input_width

    def infer(self, frame_bgr, conf_thres=0.25, iou_thres=None):
        if self._closed:
            raise RuntimeError("Model Hailo został już zamknięty.")
        iou_thres = self.iou_threshold if iou_thres is None else float(iou_thres)
        frame_rgb, meta = self._build_preprocessed_rgb(frame_bgr)
        raw = self._run_raw_inference(frame_rgb)
        detections = self._postprocess_yolov8_seg(raw, meta, conf_thres=float(conf_thres), iou_thres=iou_thres)
        return detections

    def _postprocess_yolov8_seg(self, raw_outputs, meta, conf_thres=0.25, iou_thres=0.7):
        groups = {}
        proto = None
        for arr in raw_outputs.values():
            shape = tuple(int(x) for x in arr.shape)
            if len(shape) != 4:
                continue
            _, h, w, c = shape
            if h == w and h in {20, 40, 80}:
                groups.setdefault(h, []).append(arr.astype(np.float32, copy=False))
            elif h == w and h == 160:
                proto = arr.astype(np.float32, copy=False)

        if proto is None or len(groups) != 3:
            raise ValueError("Nie udało się dopasować wyjść HEF do YOLOv8-seg.")

        score_outputs = []
        coeff_outputs = []
        raw_boxes = []
        for size in sorted(groups.keys()):
            arrs = groups[size]
            arrs_sorted = sorted(arrs, key=lambda a: a.shape[-1])
            coeff = None
            for item in arrs_sorted:
                if item.shape[-1] == self.mask_channels:
                    coeff = item
                    break
            if coeff is None:
                coeff = min(arrs_sorted, key=lambda a: abs(a.shape[-1] - self.mask_channels))
            reg = max(arrs_sorted, key=lambda a: a.shape[-1])
            cls_candidates = [item for item in arrs_sorted if id(item) not in {id(coeff), id(reg)}]
            if not cls_candidates:
                raise ValueError("Nie udało się rozpoznać tensora klas dla YOLOv8-seg.")
            cls = cls_candidates[0]
            raw_boxes.append((reg, self.input_height // size))
            score_outputs.append(cls.reshape((1, size * size, cls.shape[-1])))
            coeff_outputs.append(coeff.reshape((1, size * size, coeff.shape[-1])))

        scores = np.concatenate(score_outputs, axis=1)[0]
        coeffs = np.concatenate(coeff_outputs, axis=1)[0]
        decoded_xywh = self._decode_yolov8_boxes(raw_boxes)[0]
        boxes_xyxy = self._xywh_to_xyxy(decoded_xywh)

        class_ids = np.argmax(scores, axis=1).astype(int)
        class_scores = scores[np.arange(scores.shape[0]), class_ids]
        keep = class_scores >= conf_thres
        if not np.any(keep):
            return []

        boxes_xyxy = boxes_xyxy[keep].astype(np.float32)
        coeffs = coeffs[keep].astype(np.float32)
        class_ids = class_ids[keep]
        class_scores = class_scores[keep].astype(np.float32)

        nms_keep = self._nms(boxes_xyxy, class_scores, class_ids, iou_thres)
        boxes_xyxy = boxes_xyxy[nms_keep]
        coeffs = coeffs[nms_keep]
        class_ids = class_ids[nms_keep]
        class_scores = class_scores[nms_keep]

        masks_input = self._process_masks(proto[0], coeffs, boxes_xyxy) > 0.4
        boxes_orig = self._decode_original_boxes(boxes_xyxy, meta)

        detections = []
        for idx in range(len(class_ids)):
            mask_orig = self._decode_original_mask(masks_input[idx].astype(np.uint8), meta)
            if not mask_orig.any():
                continue
            x1, y1, x2, y2 = boxes_orig[idx]
            detections.append({
                "class_id": int(class_ids[idx]),
                "class_name": self.names.get(int(class_ids[idx]), f"class_{int(class_ids[idx])}"),
                "score": float(class_scores[idx]),
                "bbox": (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
                "mask": mask_orig,
            })
        return detections

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self.last_infer_job is not None:
                self.last_infer_job.wait(10000)
        except Exception:
            pass
        try:
            if self.config_ctx is not None:
                self.config_ctx.__exit__(None, None, None)
        except Exception:
            pass
        self.config_ctx = None
        self.configured_model = None
        self.infer_model = None
        self.hef = None
        self.target = None
